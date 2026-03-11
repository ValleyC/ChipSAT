"""
Multi-Proposal ML Initial Placement Training

Pipeline: GNN heatmap → K Gumbel proposals → CP-SAT legalize → ALNS refine

Modes:
    collect  — Generate pseudo-labels via multi-start legalize + ALNS
    train    — Stage 1: min-over-seeds CE  |  Stage 2: WA-HPWL + density + CE anchor
    reinforce — REINFORCE with post-legalization HPWL reward (no pseudo-labels)
    eval     — Fair best-of-K comparison: ML vs random vs force-directed vs reference

Usage:
    python train_placement.py collect --circuits ibm01 --n_seeds 5 --n_iterations 200
    python train_placement.py train --train_circuits ibm01 --n_epochs 100
    python train_placement.py eval --test_circuits ibm01 --checkpoint best.pt
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F

from benchmark_loader import load_bookshelf_circuit
from cpsat_solver import legalize, compute_net_hpwl, check_overlap, check_boundary
from lns_solver import LNSSolver
from placement_model import (
    PlacementGNN, build_node_features,
    decode_heatmap_hard, decode_heatmap_soft, decode_heatmap_argmax,
    decode_heatmap_jitter,
)
from differentiable_hpwl import (
    build_net_tensors, wa_hpwl, density_penalty, boundary_penalty,
    canonicalize_placement, build_grid_centers, assign_to_cells,
)


# ──────────────────────────────────────────────────────────────────────────
# Data loading utilities
# ──────────────────────────────────────────────────────────────────────────

def load_circuit(circuit_name, benchmark_base='benchmarks', seed=42):
    """Load a circuit with macros only."""
    circuit_dir = os.path.join(benchmark_base, 'iccad04', 'extracted', circuit_name)
    data = load_bookshelf_circuit(circuit_dir, circuit_name, macros_only=True, seed=seed)
    return data


def load_synthetic_circuit(npz_path):
    """Load a synthetic circuit from .npz file."""
    d = np.load(npz_path, allow_pickle=True)
    return {
        'node_features': d['node_features'],
        'edge_index': d['edge_index'],
        'edge_attr': d['edge_attr'],
        'positions': d['positions'],
        'nets': d['nets'].tolist(),
        'n_components': int(d['n_components']),
        'circuit_name': str(d['circuit_name']),
    }


def load_synthetic_pseudo_labels(labels_dir, circuit_name, max_seeds=10):
    """Load pseudo-labels for a synthetic circuit."""
    labels = []
    for seed in range(max_seeds):
        path = os.path.join(labels_dir, f'{circuit_name}_seed{seed}.npz')
        if os.path.exists(path):
            d = np.load(path)
            labels.append({
                'positions': d['positions'],
                'hpwl': float(d['hpwl']),
            })
    return labels


def prepare_circuit_tensors(data, device='cpu'):
    """Prepare tensors for a single circuit."""
    sizes_t = torch.from_numpy(data['node_features']).float().to(device)
    edge_index_t = torch.from_numpy(data['edge_index']).long().to(device)
    edge_attr_t = torch.from_numpy(data['edge_attr']).float().to(device)
    V = data['n_components']
    node_features = build_node_features(sizes_t, edge_index_t, V)
    net_tensors = build_net_tensors(data['nets'], V, device)
    return {
        'sizes_t': sizes_t,
        'edge_index_t': edge_index_t,
        'edge_attr_t': edge_attr_t,
        'node_features': node_features,
        'net_tensors': net_tensors,
        'V': V,
    }


CASCADING_WINDOWS = [0.05, 0.07, 0.10, 0.15, 0.25, 0.50]


def legalize_cascading(positions, sizes, time_limit=30.0, windows=None):
    """Try legalization with progressively wider windows.

    Returns (legal_positions, window_used) or (None, None) if all fail.
    """
    if windows is None:
        windows = CASCADING_WINDOWS
    for w in windows:
        legal = legalize(positions, sizes, time_limit=time_limit, window_fraction=w)
        if legal is not None:
            return legal, w
    return None, None


# ──────────────────────────────────────────────────────────────────────────
# Force-directed baseline
# ──────────────────────────────────────────────────────────────────────────

def force_directed_init(
    sizes_t, net_tensors, V,
    n_iters=200, lr=0.01, repulsion_weight=0.01, seed=0,
):
    """
    Force-directed placement: WA-HPWL attraction + 1/dist repulsion + boundary.

    Returns (V, 2) numpy positions.
    """
    rng = torch.Generator().manual_seed(seed)
    pos = torch.randn(V, 2, generator=rng) * 0.3
    pos = pos.to(sizes_t.device)
    pos.requires_grad_(True)
    opt = torch.optim.Adam([pos], lr=lr)

    for _ in range(n_iters):
        hpwl_val, _ = wa_hpwl(
            pos, net_tensors['net_node_indices'],
            net_tensors['net_pin_offsets'], net_tensors['net_mask'],
            gamma=10.0,
        )
        # Repulsion
        dists = torch.cdist(pos, pos) + 1e-4
        repulsion = (1.0 / dists).triu(diagonal=1).sum()
        # Boundary
        bnd = boundary_penalty(pos, sizes_t)
        loss = hpwl_val - repulsion_weight * repulsion + 10.0 * bnd
        opt.zero_grad()
        loss.backward()
        opt.step()

    return pos.detach().cpu().numpy()


# ──────────────────────────────────────────────────────────────────────────
# Collect mode: generate pseudo-labels
# ──────────────────────────────────────────────────────────────────────────

def collect_mode(args):
    """Generate pseudo-labels by running ALNS with multiple seeds."""
    os.makedirs(args.save_dir, exist_ok=True)
    circuits = [c.strip() for c in args.circuits.split(',')]

    for circuit_name in circuits:
        print(f"\n{'='*60}")
        print(f"Collecting pseudo-labels for {circuit_name}")
        print(f"{'='*60}")

        data = load_circuit(circuit_name, args.benchmark_base)
        positions = data['positions']
        sizes = data['node_features']
        nets = data['nets']
        V = data['n_components']

        # Check if legalization needed
        _, n_ov = check_overlap(positions, sizes)
        if n_ov > 0:
            print(f"  Reference has {n_ov} overlapping pairs, legalizing...")
            legal_pos = legalize(positions, sizes, time_limit=60.0, window_fraction=0.3)
            if legal_pos is None:
                print(f"  ERROR: legalization failed for {circuit_name}, skipping")
                continue
            positions = legal_pos
            print(f"  Legalized: HPWL={compute_net_hpwl(positions, sizes, nets):.4f}")

        for seed in range(args.n_seeds):
            print(f"\n  Seed {seed}:")
            solver = LNSSolver(
                positions=positions.copy(),
                sizes=sizes,
                nets=nets,
                edge_index=data['edge_index'],
                congestion_weight=0.0,
                subset_size=30,
                window_fraction=0.15,
                cpsat_time_limit=0.3,
                seed=seed * 1000 + 42,
            )
            result = solver.solve(
                n_iterations=args.n_iterations,
                log_every=50,
                verbose=True,
            )
            best_pos = result['best_positions']
            best_hpwl = result['best_hpwl']

            # Save raw positions (skip canonicalization — flipping positions
            # without flipping pin offsets corrupts HPWL; min-over-seeds CE
            # handles multimodality from different orientations)
            save_path = os.path.join(
                args.save_dir, f'{circuit_name}_seed{seed}.npz')
            np.savez(save_path,
                     positions=best_pos,
                     hpwl=best_hpwl,
                     sizes=sizes,
                     circuit_name=circuit_name)
            print(f"  Saved: {save_path} (HPWL={best_hpwl:.4f})")


# ──────────────────────────────────────────────────────────────────────────
# Train mode
# ──────────────────────────────────────────────────────────────────────────

def load_pseudo_labels(save_dir, circuit_name, n_seeds):
    """Load canonicalized pseudo-labels for a circuit."""
    labels = []
    for seed in range(n_seeds):
        path = os.path.join(save_dir, f'{circuit_name}_seed{seed}.npz')
        if os.path.exists(path):
            d = np.load(path)
            labels.append({
                'positions': d['positions'],
                'hpwl': float(d['hpwl']),
            })
    return labels


def train_mode(args):
    """Two-stage training: pseudo-supervision + objective fine-tuning.

    Supports three training phases via --training_phase:
      pretrain: synthetic circuits only
      finetune: real IBM circuits only (from pretrained checkpoint)
      mixed:    both, with --synthetic_ratio controlling the mix
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    phase = getattr(args, 'training_phase', None) or 'default'

    # Determine which circuits to load
    real_circuits = []
    if args.train_circuits and args.train_circuits.strip():
        real_circuits = [c.strip() for c in args.train_circuits.split(',') if c.strip()]
    val_circuits = [c.strip() for c in args.val_circuits.split(',')] if args.val_circuits else []

    G = args.grid_size
    grid_centers = build_grid_centers(G, device)

    # Load circuit data
    print("Loading circuits...")
    circuit_data = {}
    circuit_tensors = {}
    pseudo_labels = {}

    # Load real IBM circuits
    for cname in real_circuits + val_circuits:
        data = load_circuit(cname, args.benchmark_base)
        circuit_data[cname] = data
        circuit_tensors[cname] = prepare_circuit_tensors(data, device)
        labels = load_pseudo_labels(args.pseudo_label_dir, cname, args.n_seeds)
        if len(labels) == 0 and cname in real_circuits:
            print(f"  WARNING: no pseudo-labels for {cname}, skipping")
            real_circuits.remove(cname)
            continue
        pseudo_labels[cname] = labels
        print(f"  {cname}: V={data['n_components']}, {len(labels)} pseudo-labels")

    # Load synthetic circuits
    synthetic_circuits = []
    synthetic_dir = getattr(args, 'synthetic_dir', None)
    if synthetic_dir:
        import json
        summary_path = os.path.join(synthetic_dir, 'summary.json')
        if os.path.exists(summary_path):
            with open(summary_path, 'r') as f:
                summary = json.load(f)
            # Only load circuits that are kept and in the train split
            for entry in summary:
                if entry.get('split', 'train') not in ('train',):
                    continue
                if not entry.get('pseudo_labels', {}).get('kept', True):
                    continue
                cname = entry['circuit_name']
                npz_path = os.path.join(synthetic_dir, 'circuits', f'{cname}.npz')
                if not os.path.exists(npz_path):
                    continue
                data = load_synthetic_circuit(npz_path)
                circuit_data[cname] = data
                circuit_tensors[cname] = prepare_circuit_tensors(data, device)
                labels = load_synthetic_pseudo_labels(
                    os.path.join(synthetic_dir, 'pseudo_labels'), cname)
                if len(labels) == 0:
                    continue
                pseudo_labels[cname] = labels
                synthetic_circuits.append(cname)
                print(f"  {cname}: V={data['n_components']}, {len(labels)} pseudo-labels "
                      f"[synthetic, {entry.get('family', '?')}]")

    # Determine train set based on phase
    if phase == 'pretrain':
        train_circuits = synthetic_circuits
        print(f"\nPhase: PRETRAIN (synthetic only, {len(train_circuits)} circuits)")
    elif phase == 'finetune':
        train_circuits = real_circuits
        print(f"\nPhase: FINETUNE (real only, {len(train_circuits)} circuits)")
    elif phase == 'mixed':
        train_circuits = real_circuits + synthetic_circuits
        print(f"\nPhase: MIXED ({len(real_circuits)} real + {len(synthetic_circuits)} synthetic)")
    else:
        # Default: use real circuits (backwards compatible)
        train_circuits = real_circuits
        if synthetic_circuits:
            train_circuits = real_circuits + synthetic_circuits

    if len(train_circuits) == 0:
        print("ERROR: no training circuits with pseudo-labels")
        return

    # Model
    model = PlacementGNN(
        node_input_dim=14,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        grid_size=G,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params, G={G}")

    # Resume from checkpoint if specified
    if hasattr(args, 'resume_from') and args.resume_from:
        state = torch.load(args.resume_from, map_location=device, weights_only=True)
        model.load_state_dict(state)
        print(f"Resumed from {args.resume_from}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    total_epochs = args.stage1_epochs + args.stage2_epochs
    # Use cosine with warm restarts — keeps LR alive longer
    # T_0 = min(total_epochs, 200) so LR doesn't die too fast on short runs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=min(total_epochs, 200), T_mult=1)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_val_hpwl = float('inf')

    np.random.seed(args.seed)

    for epoch in range(total_epochs):
        model.train()
        is_stage2 = epoch >= args.stage1_epochs
        stage_name = "S2" if is_stage2 else "S1"

        epoch_loss = 0.0
        epoch_count = 0

        # Shuffle training circuits (with synthetic ratio for mixed phase)
        if phase == 'mixed' and synthetic_circuits and real_circuits:
            syn_ratio = getattr(args, 'synthetic_ratio', 0.3)
            n_syn = max(1, int(len(real_circuits) * syn_ratio / (1.0 - syn_ratio)))
            n_syn = min(n_syn, len(synthetic_circuits))
            syn_subset = list(np.random.choice(synthetic_circuits, size=n_syn, replace=False))
            train_order = list(real_circuits) + syn_subset
        else:
            train_order = list(train_circuits)
        np.random.shuffle(train_order)

        for cname in train_order:
            ct = circuit_tensors[cname]
            labels = pseudo_labels[cname]
            if len(labels) == 0:
                continue

            logits = model(ct['node_features'], ct['edge_index_t'], ct['edge_attr_t'])

            # Min-over-seeds CE
            ce_losses = []
            for lbl in labels:
                cell_ids = assign_to_cells(lbl['positions'], grid_centers)
                ce = F.cross_entropy(logits, cell_ids)
                ce_losses.append(ce)
            ce_min = torch.stack(ce_losses).min()

            if not is_stage2:
                # Stage 1: pure pseudo-supervision
                loss = ce_min
            else:
                # Stage 2: objective + CE anchor
                stage2_epoch = epoch - args.stage1_epochs
                stage2_frac = stage2_epoch / max(args.stage2_epochs - 1, 1)

                # CE anchor with decaying weight
                lambda_ce = args.lambda_ce_start + (args.lambda_ce_end - args.lambda_ce_start) * stage2_frac

                # Gumbel temperature annealing
                tau = args.tau_init * (args.tau_min / args.tau_init) ** stage2_frac

                # Gamma annealing
                gamma = args.gamma_init + (args.gamma_final - args.gamma_init) * stage2_frac

                # K proposals via Gumbel-softmax, softmin
                M = ct['net_tensors']['n_nets']
                obj_losses = []
                for k in range(args.K):
                    soft_pos = decode_heatmap_soft(logits, grid_centers, tau=tau)
                    hpwl_val, _ = wa_hpwl(
                        soft_pos,
                        ct['net_tensors']['net_node_indices'],
                        ct['net_tensors']['net_pin_offsets'],
                        ct['net_tensors']['net_mask'],
                        gamma=gamma,
                    )
                    den = density_penalty(logits, ct['sizes_t'], G)
                    bnd = boundary_penalty(soft_pos, ct['sizes_t'])
                    obj = hpwl_val / max(M, 1) + args.w_density * den + args.w_boundary * bnd
                    obj_losses.append(obj)

                # Softmin over K proposals
                obj_stack = torch.stack(obj_losses)
                softmin_tau = args.softmin_tau
                obj_loss = -torch.logsumexp(-obj_stack / softmin_tau, dim=0) * softmin_tau

                loss = obj_loss + lambda_ce * ce_min

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_count += 1

        scheduler.step()

        avg_loss = epoch_loss / max(epoch_count, 1)

        # Logging
        if epoch % args.log_every == 0 or epoch == total_epochs - 1:
            # Quick diagnostics
            model.eval()
            with torch.no_grad():
                diag_circuit = train_circuits[0]
                ct = circuit_tensors[diag_circuit]
                logits = model(ct['node_features'], ct['edge_index_t'], ct['edge_attr_t'])
                probs = F.softmax(logits, dim=1)
                entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()
                max_prob = probs.max(dim=1).values.mean()

            print(f"  [{epoch+1:4d}/{total_epochs}] [{stage_name}] "
                  f"loss={avg_loss:.4f}  entropy={entropy:.2f}  "
                  f"max_prob={max_prob:.3f}  lr={scheduler.get_last_lr()[0]:.6f}")

        # Validation (post-legalization HPWL with cascading windows)
        if val_circuits and (epoch % args.val_every == 0 or epoch == total_epochs - 1):
            model.eval()
            val_hpwls = []
            with torch.no_grad():
                for cname in val_circuits:
                    ct = circuit_tensors[cname]
                    logits = model(ct['node_features'], ct['edge_index_t'], ct['edge_attr_t'])
                    # Best of 2 jittered proposals → cascading legalize
                    best_h = float('inf')
                    for k in range(2):
                        pos = decode_heatmap_jitter(logits, grid_centers, G, seed=k * 137)
                        pos_np = pos.cpu().numpy()
                        legal, w = legalize_cascading(
                            pos_np, circuit_data[cname]['node_features'], time_limit=15.0)
                        if legal is not None:
                            h = compute_net_hpwl(legal, circuit_data[cname]['node_features'],
                                                  circuit_data[cname]['nets'])
                            best_h = min(best_h, h)
                    val_hpwls.append(best_h)
            finite_vals = [h for h in val_hpwls if h < float('inf')]
            mean_val = np.mean(finite_vals) if finite_vals else float('inf')
            print(f"    Val HPWL (post-legalize): {mean_val:.4f} "
                  f"({', '.join(f'{h:.1f}' if h < float('inf') else 'INF' for h in val_hpwls)})")

            if mean_val < best_val_hpwl:
                best_val_hpwl = mean_val
                save_path = os.path.join(args.checkpoint_dir, 'best_model.pt')
                torch.save(model.state_dict(), save_path)
                print(f"    Saved best model (val HPWL={mean_val:.4f})")

        # Full eval (predict → cascading legalize → HPWL)
        if epoch % args.full_eval_every == 0 and epoch > 0:
            model.eval()
            with torch.no_grad():
                cname = train_circuits[0]
                ct = circuit_tensors[cname]
                logits = model(ct['node_features'], ct['edge_index_t'], ct['edge_attr_t'])
                # Best of K jittered samples with cascading legalization
                best_post_hpwl = float('inf')
                best_window = None
                for k in range(args.K):
                    pos = decode_heatmap_jitter(logits, grid_centers, G, seed=k * 137)
                    pos_np = pos.cpu().numpy()
                    legal, w = legalize_cascading(
                        pos_np, circuit_data[cname]['node_features'], time_limit=30.0)
                    if legal is not None:
                        h = compute_net_hpwl(legal, circuit_data[cname]['node_features'],
                                             circuit_data[cname]['nets'])
                        if h < best_post_hpwl:
                            best_post_hpwl = h
                            best_window = w
                if best_post_hpwl < float('inf'):
                    print(f"    Full eval {cname}: best-of-{args.K} post-legalize HPWL = "
                          f"{best_post_hpwl:.4f} (window={best_window})")
                else:
                    print(f"    Full eval {cname}: all {args.K} proposals failed legalization")

    # Final save
    save_path = os.path.join(args.checkpoint_dir, 'final_model.pt')
    torch.save(model.state_dict(), save_path)
    print(f"\nTraining complete. Final model saved to {save_path}")


# ──────────────────────────────────────────────────────────────────────────
# Reinforce mode: RL with post-legalization reward (no pseudo-labels needed)
# ──────────────────────────────────────────────────────────────────────────

def reinforce_mode(args):
    """REINFORCE training with post-legalization HPWL as reward.

    One-shot heatmap prediction (bandit, not MDP):
      logits → sample cell assignments → decode to positions → legalize → reward
    Self-critical baseline: argmax decode → legalize → baseline reward.
    No pseudo-labels needed — works on any circuit (real or synthetic).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    G = args.grid_size
    grid_centers = build_grid_centers(G, device)

    # Load circuits (real + synthetic, no pseudo-labels needed)
    real_circuits = []
    if args.train_circuits and args.train_circuits.strip():
        real_circuits = [c.strip() for c in args.train_circuits.split(',') if c.strip()]

    circuit_data = {}
    circuit_tensors = {}
    train_circuit_names = []

    # Load real IBM circuits
    for cname in real_circuits:
        data = load_circuit(cname, args.benchmark_base)
        circuit_data[cname] = data
        circuit_tensors[cname] = prepare_circuit_tensors(data, device)
        train_circuit_names.append(cname)
        print(f"  {cname}: V={data['n_components']} [real]")

    # Load synthetic circuits (no pseudo-labels needed!)
    synthetic_dir = getattr(args, 'synthetic_dir', None)
    if synthetic_dir:
        import json
        circuits_dir = os.path.join(synthetic_dir, 'circuits')
        if os.path.isdir(circuits_dir):
            import glob
            for npz_path in sorted(glob.glob(os.path.join(circuits_dir, '*.npz'))):
                data = load_synthetic_circuit(npz_path)
                cname = data['circuit_name']
                circuit_data[cname] = data
                circuit_tensors[cname] = prepare_circuit_tensors(data, device)
                train_circuit_names.append(cname)
            print(f"  Loaded {len(train_circuit_names) - len(real_circuits)} synthetic circuits")

    if not train_circuit_names:
        print("ERROR: no training circuits")
        return

    print(f"\nTotal training circuits: {len(train_circuit_names)}")

    # Model
    model = PlacementGNN(
        node_input_dim=14,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        grid_size=G,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params, G={G}")

    # Load checkpoint (typically Stage 1 model)
    if args.resume_from:
        state = torch.load(args.resume_from, map_location=device, weights_only=True)
        model.load_state_dict(state)
        print(f"Resumed from {args.resume_from}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=min(args.n_epochs, 200), T_mult=1)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_mean_reward = -float('inf')

    K = args.K
    entropy_weight = args.entropy_weight
    legal_time = args.legal_time_limit
    infeasible_penalty = args.infeasible_penalty

    print(f"K={K}, entropy_weight={entropy_weight}, "
          f"legal_time={legal_time}s, infeasible_penalty={infeasible_penalty}")

    for epoch in range(args.n_epochs):
        model.train()
        epoch_rewards = []
        epoch_baselines = []
        epoch_loss = 0.0
        epoch_entropy = 0.0
        epoch_feasible = 0
        epoch_total = 0

        train_order = list(train_circuit_names)
        np.random.shuffle(train_order)

        for cname in train_order:
            ct = circuit_tensors[cname]
            data = circuit_data[cname]
            sizes = data['node_features']
            nets = data['nets']
            V = ct['V']

            logits = model(ct['node_features'], ct['edge_index_t'], ct['edge_attr_t'])
            probs = F.softmax(logits, dim=1)  # (V, G²)

            # ── Self-critical baseline: argmax decode → legalize ──
            with torch.no_grad():
                baseline_pos = decode_heatmap_jitter(logits, grid_centers, G, seed=epoch)
                baseline_np = baseline_pos.cpu().numpy()
                baseline_legal, _ = legalize_cascading(
                    baseline_np, sizes, time_limit=legal_time)
                if baseline_legal is not None:
                    baseline_reward = -compute_net_hpwl(baseline_legal, sizes, nets)
                else:
                    baseline_reward = -infeasible_penalty

            # ── Sample K proposals ──
            dist = torch.distributions.Categorical(probs)
            sample_rewards = []
            sample_log_probs = []

            for k in range(K):
                cell_ids = dist.sample()  # (V,)
                log_prob = dist.log_prob(cell_ids).sum()  # scalar

                # Decode with jitter
                cell_size = 2.0 / G
                pos = grid_centers[cell_ids].clone()
                offset = (torch.rand(V, 2, device=device) - 0.5) * cell_size * 0.67
                pos = (pos + offset).clamp(-1.0, 1.0)
                pos_np = pos.detach().cpu().numpy()

                # Legalize (cascading windows, short time limit)
                legal, w = legalize_cascading(pos_np, sizes, time_limit=legal_time)
                if legal is not None:
                    reward = -compute_net_hpwl(legal, sizes, nets)
                    epoch_feasible += 1
                else:
                    reward = -infeasible_penalty
                epoch_total += 1

                sample_rewards.append(reward)
                sample_log_probs.append(log_prob)

            # ── REINFORCE loss ──
            rewards_t = torch.tensor(sample_rewards, device=device, dtype=torch.float32)
            log_probs_t = torch.stack(sample_log_probs)
            advantages = rewards_t - baseline_reward  # self-critical

            policy_loss = -(log_probs_t * advantages.detach()).mean()

            # Entropy bonus (prevent heatmap collapse)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()

            loss = policy_loss - entropy_weight * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_rewards.extend(sample_rewards)
            epoch_baselines.append(baseline_reward)
            epoch_loss += loss.item()
            epoch_entropy += entropy.item()

        scheduler.step()

        # ── Logging ──
        n_circuits = len(train_order)
        mean_reward = np.mean(epoch_rewards) if epoch_rewards else 0.0
        mean_baseline = np.mean(epoch_baselines) if epoch_baselines else 0.0
        feasible_rate = epoch_feasible / max(epoch_total, 1)
        avg_loss = epoch_loss / max(n_circuits, 1)
        avg_entropy = epoch_entropy / max(n_circuits, 1)

        if epoch % args.log_every == 0 or epoch == args.n_epochs - 1:
            print(f"  [{epoch+1:4d}/{args.n_epochs}] "
                  f"reward={mean_reward:.2f}  baseline={mean_baseline:.2f}  "
                  f"feasible={feasible_rate:.2%}  "
                  f"entropy={avg_entropy:.2f}  loss={avg_loss:.4f}  "
                  f"lr={scheduler.get_last_lr()[0]:.6f}")

        # ── Save best ──
        if mean_reward > best_mean_reward:
            best_mean_reward = mean_reward
            save_path = os.path.join(args.checkpoint_dir, 'best_model.pt')
            torch.save(model.state_dict(), save_path)

        # ── Full eval (periodic) ──
        if epoch % args.full_eval_every == 0 and epoch > 0 and real_circuits:
            model.eval()
            with torch.no_grad():
                cname = real_circuits[0]
                ct = circuit_tensors[cname]
                logits = model(ct['node_features'], ct['edge_index_t'], ct['edge_attr_t'])
                best_post_hpwl = float('inf')
                best_w = None
                for k in range(K):
                    pos = decode_heatmap_jitter(logits, grid_centers, G, seed=k * 137)
                    pos_np = pos.cpu().numpy()
                    legal, w = legalize_cascading(
                        pos_np, circuit_data[cname]['node_features'], time_limit=30.0)
                    if legal is not None:
                        h = compute_net_hpwl(legal, circuit_data[cname]['node_features'],
                                             circuit_data[cname]['nets'])
                        if h < best_post_hpwl:
                            best_post_hpwl = h
                            best_w = w
                if best_post_hpwl < float('inf'):
                    print(f"    Eval {cname}: post-legal={best_post_hpwl:.2f} (w={best_w})")
                else:
                    print(f"    Eval {cname}: all proposals INFEASIBLE")

    # Final save
    save_path = os.path.join(args.checkpoint_dir, 'final_model.pt')
    torch.save(model.state_dict(), save_path)
    print(f"\nREINFORCE complete. Best reward={best_mean_reward:.2f}")
    print(f"Saved to {save_path}")


# ──────────────────────────────────────────────────────────────────────────
# Eval mode: fair best-of-K comparison
# ──────────────────────────────────────────────────────────────────────────

def eval_mode(args):
    """Four-way comparison: ML vs random vs force-directed vs reference."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_circuits = [c.strip() for c in args.test_circuits.split(',')]
    G = args.grid_size
    K = args.K
    grid_centers = build_grid_centers(G, device)

    # Load model
    model = PlacementGNN(
        node_input_dim=14,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        grid_size=G,
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device,
                                     weights_only=True))
    model.eval()
    print(f"Loaded model from {args.checkpoint}")

    results = {}

    for cname in test_circuits:
        print(f"\n{'='*60}")
        print(f"Evaluating {cname}")
        print(f"{'='*60}")

        data = load_circuit(cname, args.benchmark_base)
        ct = prepare_circuit_tensors(data, device)
        sizes = data['node_features']
        nets = data['nets']
        V = data['n_components']
        ref_pos = data['positions']

        ref_hpwl = compute_net_hpwl(ref_pos, sizes, nets)
        print(f"  Reference HPWL: {ref_hpwl:.4f}")

        circuit_results = {}

        # --- 1. ML-init (best of K) ---
        print(f"\n  [ML-init] best of {K}:")
        with torch.no_grad():
            logits = model(ct['node_features'], ct['edge_index_t'], ct['edge_attr_t'])

        ml_best_hpwl_pre = float('inf')
        ml_best_hpwl_post = float('inf')
        ml_best_pos = None
        ml_best_window = None
        ml_legal_count = 0

        for k in range(K):
            with torch.no_grad():
                pos = decode_heatmap_jitter(logits, grid_centers, G, seed=k * 137)
            pos_np = pos.cpu().numpy()
            pre_hpwl = compute_net_hpwl(pos_np, sizes, nets)

            legal, w = legalize_cascading(pos_np, sizes, time_limit=30.0)
            if legal is not None:
                ml_legal_count += 1
                post_hpwl = compute_net_hpwl(legal, sizes, nets)
                if post_hpwl < ml_best_hpwl_post:
                    ml_best_hpwl_pre = pre_hpwl
                    ml_best_hpwl_post = post_hpwl
                    ml_best_pos = legal
                    ml_best_window = w
                print(f"    k={k}: pre={pre_hpwl:.4f} → post={post_hpwl:.4f} (w={w})")
            else:
                print(f"    k={k}: pre={pre_hpwl:.4f} → INFEASIBLE")

        circuit_results['ml'] = {
            'best_hpwl_post': ml_best_hpwl_post,
            'best_window': ml_best_window,
            'legal_count': ml_legal_count,
        }

        # --- 2. Random-init (best of K) ---
        print(f"\n  [Random-init] best of {K}:")
        rand_best_hpwl_post = float('inf')
        rand_best_pos = None
        rand_legal_count = 0

        for k in range(K):
            rng = np.random.default_rng(seed=k * 100 + 7)
            pos_np = rng.uniform(-0.95, 0.95, size=(V, 2)).astype(np.float32)
            pre_hpwl = compute_net_hpwl(pos_np, sizes, nets)

            legal, w = legalize_cascading(pos_np, sizes, time_limit=30.0)
            if legal is not None:
                rand_legal_count += 1
                post_hpwl = compute_net_hpwl(legal, sizes, nets)
                if post_hpwl < rand_best_hpwl_post:
                    rand_best_hpwl_post = post_hpwl
                    rand_best_pos = legal
                print(f"    k={k}: pre={pre_hpwl:.4f} → post={post_hpwl:.4f} (w={w})")
            else:
                print(f"    k={k}: pre={pre_hpwl:.4f} → INFEASIBLE")

        circuit_results['random'] = {
            'best_hpwl_post': rand_best_hpwl_post,
            'legal_count': rand_legal_count,
        }

        # --- 3. Force-directed (best of K) ---
        print(f"\n  [Force-directed] best of {K}:")
        fd_best_hpwl_post = float('inf')
        fd_best_pos = None
        fd_legal_count = 0

        for k in range(K):
            pos_np = force_directed_init(
                ct['sizes_t'], ct['net_tensors'], V,
                n_iters=200, lr=0.01, seed=k * 100 + 13,
            )
            pre_hpwl = compute_net_hpwl(pos_np, sizes, nets)

            legal, w = legalize_cascading(pos_np, sizes, time_limit=30.0)
            if legal is not None:
                fd_legal_count += 1
                post_hpwl = compute_net_hpwl(legal, sizes, nets)
                if post_hpwl < fd_best_hpwl_post:
                    fd_best_hpwl_post = post_hpwl
                    fd_best_pos = legal
                print(f"    k={k}: pre={pre_hpwl:.4f} → post={post_hpwl:.4f} (w={w})")
            else:
                print(f"    k={k}: pre={pre_hpwl:.4f} → INFEASIBLE")

        circuit_results['force_directed'] = {
            'best_hpwl_post': fd_best_hpwl_post,
            'legal_count': fd_legal_count,
        }

        # --- 4. Reference-init ---
        print(f"\n  [Reference-init]:")
        _, n_ov = check_overlap(ref_pos, sizes)
        if n_ov > 0:
            legal_ref, w = legalize_cascading(ref_pos, sizes, time_limit=60.0)
            if legal_ref is not None:
                ref_post_hpwl = compute_net_hpwl(legal_ref, sizes, nets)
                ref_best_pos = legal_ref
                print(f"    post-legalize HPWL: {ref_post_hpwl:.4f} (w={w})")
            else:
                ref_post_hpwl = float('inf')
                ref_best_pos = None
                print(f"    post-legalize: INFEASIBLE")
        else:
            ref_post_hpwl = ref_hpwl
            ref_best_pos = ref_pos
            print(f"    post-legalize HPWL: {ref_post_hpwl:.4f} (no overlap)")

        circuit_results['reference'] = {'best_hpwl_post': ref_post_hpwl}

        # --- ALNS refinement on best of each ---
        if args.n_alns_iterations > 0:
            print(f"\n  ALNS refinement ({args.n_alns_iterations} iterations):")
            for method, best_pos in [
                ('ml', ml_best_pos),
                ('random', rand_best_pos),
                ('force_directed', fd_best_pos),
                ('reference', ref_best_pos),
            ]:
                if best_pos is None:
                    print(f"    {method:15s}: SKIPPED (no legal init)")
                    circuit_results[method]['alns_hpwl'] = float('inf')
                    continue

                solver = LNSSolver(
                    positions=best_pos.copy(),
                    sizes=sizes,
                    nets=nets,
                    edge_index=data['edge_index'],
                    congestion_weight=0.0,
                    subset_size=30,
                    window_fraction=0.15,
                    cpsat_time_limit=0.3,
                    seed=42,
                )
                result = solver.solve(
                    n_iterations=args.n_alns_iterations,
                    log_every=100,
                    verbose=False,
                )
                alns_hpwl = result['best_hpwl']
                circuit_results[method]['alns_hpwl'] = alns_hpwl
                print(f"    {method:15s}: {circuit_results[method]['best_hpwl_post']:.4f} "
                      f"→ {alns_hpwl:.4f}")

        # --- Summary ---
        print(f"\n  Summary for {cname}:")
        print(f"    {'Method':18s} {'Post-Legal':>12s} {'Post-ALNS':>12s} {'Legal':>6s}")
        print(f"    {'-'*52}")
        for method in ['ml', 'random', 'force_directed', 'reference']:
            r = circuit_results[method]
            post = f"{r['best_hpwl_post']:.4f}" if r['best_hpwl_post'] < float('inf') else "N/A"
            alns = f"{r.get('alns_hpwl', float('inf')):.4f}" if r.get('alns_hpwl', float('inf')) < float('inf') else "N/A"
            legal = f"{r.get('legal_count', 'N/A')}"
            print(f"    {method:18s} {post:>12s} {alns:>12s} {legal:>6s}")

        results[cname] = circuit_results

    # --- Overall summary ---
    if len(test_circuits) > 1:
        print(f"\n{'='*60}")
        print("Overall Summary")
        print(f"{'='*60}")
        for method in ['ml', 'random', 'force_directed', 'reference']:
            hpwls = [results[c][method].get('alns_hpwl', results[c][method]['best_hpwl_post'])
                     for c in test_circuits if results[c][method]['best_hpwl_post'] < float('inf')]
            if hpwls:
                print(f"  {method:18s}: mean={np.mean(hpwls):.4f}")


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='ML Initial Placement')
    subparsers = parser.add_subparsers(dest='mode', required=True)

    # --- Collect ---
    p_collect = subparsers.add_parser('collect', help='Generate pseudo-labels')
    p_collect.add_argument('--circuits', type=str, required=True)
    p_collect.add_argument('--n_seeds', type=int, default=5)
    p_collect.add_argument('--n_iterations', type=int, default=200)
    p_collect.add_argument('--save_dir', type=str, default='pseudo_labels')
    p_collect.add_argument('--benchmark_base', type=str, default='benchmarks')

    # --- Train ---
    p_train = subparsers.add_parser('train', help='Train placement model')
    p_train.add_argument('--train_circuits', type=str, default='',
                         help='Comma-separated IBM circuit names (empty for pretrain-only)')
    p_train.add_argument('--val_circuits', type=str, default=None)
    p_train.add_argument('--pseudo_label_dir', type=str, default='pseudo_labels')
    p_train.add_argument('--n_seeds', type=int, default=5)
    p_train.add_argument('--stage1_epochs', type=int, default=100)
    p_train.add_argument('--stage2_epochs', type=int, default=200)
    p_train.add_argument('--lr', type=float, default=3e-4)
    p_train.add_argument('--hidden_dim', type=int, default=64)
    p_train.add_argument('--n_layers', type=int, default=5)
    p_train.add_argument('--grid_size', type=int, default=16)
    p_train.add_argument('--K', type=int, default=4)
    p_train.add_argument('--tau_init', type=float, default=1.0)
    p_train.add_argument('--tau_min', type=float, default=0.1)
    p_train.add_argument('--gamma_init', type=float, default=10.0)
    p_train.add_argument('--gamma_final', type=float, default=50.0)
    p_train.add_argument('--w_density', type=float, default=10.0)
    p_train.add_argument('--w_boundary', type=float, default=5.0)
    p_train.add_argument('--lambda_ce_start', type=float, default=0.5)
    p_train.add_argument('--lambda_ce_end', type=float, default=0.1)
    p_train.add_argument('--softmin_tau', type=float, default=0.1)
    p_train.add_argument('--checkpoint_dir', type=str, default='checkpoints_placement')
    p_train.add_argument('--log_every', type=int, default=10)
    p_train.add_argument('--val_every', type=int, default=10)
    p_train.add_argument('--full_eval_every', type=int, default=50)
    p_train.add_argument('--seed', type=int, default=42)
    p_train.add_argument('--benchmark_base', type=str, default='benchmarks')
    p_train.add_argument('--resume_from', type=str, default=None,
                         help='Checkpoint to resume from (e.g., Stage 1 model for Stage 2)')
    p_train.add_argument('--synthetic_dir', type=str, default=None,
                         help='Directory with synthetic circuits and pseudo-labels')
    p_train.add_argument('--training_phase', type=str, default=None,
                         choices=['pretrain', 'finetune', 'mixed'],
                         help='Training phase: pretrain (synthetic only), finetune (real only), mixed')
    p_train.add_argument('--synthetic_ratio', type=float, default=0.3,
                         help='Fraction of synthetic circuits per epoch in mixed phase')

    # --- Reinforce ---
    p_rl = subparsers.add_parser('reinforce', help='REINFORCE with post-legalization reward')
    p_rl.add_argument('--train_circuits', type=str, default='',
                      help='Comma-separated IBM circuit names')
    p_rl.add_argument('--synthetic_dir', type=str, default=None,
                      help='Directory with synthetic circuits (no pseudo-labels needed)')
    p_rl.add_argument('--resume_from', type=str, default=None,
                      help='Checkpoint to start from (e.g., Stage 1 model)')
    p_rl.add_argument('--n_epochs', type=int, default=200)
    p_rl.add_argument('--lr', type=float, default=1e-4)
    p_rl.add_argument('--K', type=int, default=4)
    p_rl.add_argument('--hidden_dim', type=int, default=64)
    p_rl.add_argument('--n_layers', type=int, default=5)
    p_rl.add_argument('--grid_size', type=int, default=16)
    p_rl.add_argument('--entropy_weight', type=float, default=0.01)
    p_rl.add_argument('--legal_time_limit', type=float, default=0.5,
                      help='Time limit per legalization call (seconds)')
    p_rl.add_argument('--infeasible_penalty', type=float, default=5000.0,
                      help='Penalty for infeasible placements')
    p_rl.add_argument('--checkpoint_dir', type=str, default='checkpoints_rl')
    p_rl.add_argument('--log_every', type=int, default=5)
    p_rl.add_argument('--full_eval_every', type=int, default=20)
    p_rl.add_argument('--benchmark_base', type=str, default='benchmarks')

    # --- Eval ---
    p_eval = subparsers.add_parser('eval', help='Evaluate placement model')
    p_eval.add_argument('--test_circuits', type=str, required=True)
    p_eval.add_argument('--checkpoint', type=str, required=True)
    p_eval.add_argument('--K', type=int, default=4)
    p_eval.add_argument('--hidden_dim', type=int, default=64)
    p_eval.add_argument('--n_layers', type=int, default=5)
    p_eval.add_argument('--grid_size', type=int, default=16)
    p_eval.add_argument('--n_alns_iterations', type=int, default=0)
    p_eval.add_argument('--benchmark_base', type=str, default='benchmarks')

    args = parser.parse_args()

    if args.mode == 'collect':
        collect_mode(args)
    elif args.mode == 'train':
        train_mode(args)
    elif args.mode == 'reinforce':
        reinforce_mode(args)
    elif args.mode == 'eval':
        eval_mode(args)


if __name__ == '__main__':
    main()
