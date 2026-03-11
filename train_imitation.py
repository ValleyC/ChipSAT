"""
Heuristic Imitation Warm-Start for Learned Destroy Operator

Pipeline:
  1. collect: Run pure ALNS on circuits, at sampled states evaluate all 4
     heuristic subsets, label best-of-4 as target. Save dataset to disk.
  2. imitate: Train subset_head with weighted BCE on best-of-4 subset masks.
  3. eval: Validate warm-start quality (entropy, overlap, learned-only vs untrained).
  4. finetune: RL fine-tune with KL regularization to imitation policy.

Usage:
    # Step 1: Collect imitation data (multi-circuit)
    python train_imitation.py collect \
        --circuits ibm01,ibm02,ibm03,ibm04 \
        --n_iterations 200 --sample_every 2 \
        --save_path imitation_data/train.pt

    # Step 2: Train with imitation
    python train_imitation.py imitate \
        --data_path imitation_data/train.pt \
        --n_epochs 100 --lr 3e-4 \
        --save_dir checkpoints_imitation

    # Step 3: Evaluate warm-start quality
    python train_imitation.py eval \
        --test_circuits ibm01 \
        --checkpoint checkpoints_imitation/best_model.pt \
        --n_iterations 100

    # Step 4: RL fine-tune (optional)
    python train_imitation.py finetune \
        --train_circuits ibm01,ibm02,ibm03,ibm04 \
        --checkpoint checkpoints_imitation/best_model.pt \
        --n_epochs 20
"""

import argparse
import numpy as np
import os
import time
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from benchmark_loader import load_bookshelf_circuit
from cpsat_solver import (
    legalize, solve_subset, compute_net_hpwl,
    compute_incremental_hpwl, check_overlap,
)
from lns_solver import LNSSolver, ALNSWeights, compute_rudy_np
from net_spatial_gnn import NetSpatialGNN


# ---------------------------------------------------------------------------
# Shared helpers (from train_rl_lns.py)
# ---------------------------------------------------------------------------

def load_and_legalize(circuit_name, benchmark_base='benchmarks', seed=42):
    """Load circuit, legalize reference placement, return data dict."""
    circuit_dir = os.path.join(benchmark_base, "iccad04", "extracted", circuit_name)
    data = load_bookshelf_circuit(
        circuit_dir, circuit_name, macros_only=True, seed=seed)

    positions = data['positions']
    sizes = data['node_features']

    _, n_ov = check_overlap(positions, sizes)
    if n_ov > 0:
        legal_pos = legalize(positions, sizes, time_limit=60.0, window_fraction=0.3)
        if legal_pos is not None:
            positions = legal_pos
        else:
            print(f"  WARNING: {circuit_name} legalization failed, using reference")

    return {
        'circuit_name': circuit_name,
        'positions': positions,
        'sizes': sizes,
        'nets': data['nets'],
        'edge_index': data['edge_index'],
        'edge_attr': data['edge_attr'],
        'n_components': data['n_components'],
    }


def make_pure_solver(circuit_data, args):
    """Create an LNSSolver WITHOUT model (pure heuristic ALNS)."""
    return LNSSolver(
        positions=circuit_data['positions'].copy(),
        sizes=circuit_data['sizes'],
        nets=circuit_data['nets'],
        edge_index=circuit_data['edge_index'],
        congestion_weight=args.congestion_weight,
        subset_size=args.subset_size,
        window_fraction=args.window_fraction,
        cpsat_time_limit=args.cpsat_time_limit,
        plateau_threshold=args.plateau_threshold,
        adapt_threshold=args.adapt_threshold,
        seed=args.seed,
    )


def make_model_solver(circuit_data, model, args):
    """Create an LNSSolver WITH model."""
    return LNSSolver(
        positions=circuit_data['positions'].copy(),
        sizes=circuit_data['sizes'],
        nets=circuit_data['nets'],
        edge_index=circuit_data['edge_index'],
        edge_attr=circuit_data['edge_attr'],
        congestion_weight=args.congestion_weight,
        subset_size=args.subset_size,
        window_fraction=args.window_fraction,
        cpsat_time_limit=args.cpsat_time_limit,
        plateau_threshold=args.plateau_threshold,
        adapt_threshold=args.adapt_threshold,
        seed=args.seed,
        model=model,
    )


# ---------------------------------------------------------------------------
# Step 1: Collect imitation dataset
# ---------------------------------------------------------------------------

def evaluate_heuristic_subset(solver, strategy, k):
    """Evaluate one heuristic subset: get indices and delta_cost."""
    subset = solver.get_neighborhood(strategy, k)
    new_positions = solve_subset(
        solver.current_pos, solver.sizes, solver.nets, subset,
        time_limit=solver.cpsat_time_limit,
        window_fraction=solver.window_fraction,
    )
    if new_positions is not None:
        new_hpwl, _ = compute_incremental_hpwl(
            new_positions, solver.nets, subset,
            solver.net_hpwls, solver.macro_nets)
        new_cost = solver._compute_cost(new_positions, new_hpwl)
        delta_cost = solver.current_cost - new_cost  # positive = improvement
        return subset, delta_cost, True
    return subset, -1.0, False


def build_features_np(solver):
    """Build 10D node features as numpy (no GPU needed).

    Mirrors LNSSolver._build_gnn_features() exactly, including RUDY
    when congestion_weight > 0.
    """
    from lns_solver import compute_per_macro_rudy
    N = solver.N
    feats = np.zeros((N, 10), dtype=np.float32)
    feats[:, 0:2] = solver.current_pos
    feats[:, 2:4] = solver.sizes
    max_hpwl = max(solver.macro_hpwl.max(), 1e-8)
    feats[:, 4] = solver.macro_hpwl / max_hpwl
    if solver.congestion_weight > 0:
        macro_rudy = compute_per_macro_rudy(
            solver.current_pos, solver.sizes, solver.nets, solver.macro_nets)
        p95 = max(np.percentile(macro_rudy, 95), 1e-8)
        feats[:, 5] = np.clip(macro_rudy / p95, 0, 5.0) / 5.0
    feats[:, 6] = solver.last_subset_mask
    feats[:, 7] = solver.last_delta
    feats[:, 8] = min(
        solver.stagnation_count / max(solver.adapt_threshold, 1), 1.0)
    feats[:, 9] = solver.window_fraction / max(solver.max_window, 1e-8)
    return feats


def collect_state_sample(solver, k, strategies):
    """
    At the current solver state, evaluate all 4 heuristic subsets.
    Return the best-of-4 subset mask and quality weight.

    RNG state is snapshot/restored so labeling doesn't perturb the
    ongoing ALNS rollout trajectory.
    """
    N = solver.N
    best_delta = -float('inf')
    best_subset = None
    all_results = []

    # Snapshot RNG state — random/connected strategies consume solver.rng
    rng_state = solver.rng.bit_generator.state

    for strategy in strategies:
        subset, delta, feasible = evaluate_heuristic_subset(solver, strategy, k)
        all_results.append((strategy, subset, delta, feasible))
        if feasible and delta > best_delta:
            best_delta = delta
            best_subset = subset

    # Restore RNG state so subsequent solver.step() is unperturbed
    solver.rng.bit_generator.state = rng_state

    if best_subset is None:
        # All infeasible — skip this state
        return None

    # Build binary mask for best subset
    mask = np.zeros(N, dtype=np.float32)
    mask[best_subset] = 1.0

    # Weight: quality of improvement (higher delta = more informative)
    # Clamp at 0 — non-improving best-of-4 gets low weight
    weight = max(best_delta, 0.0)

    # Build GNN features (10D) from solver state — pure numpy, no GPU
    node_features = build_features_np(solver)
    positions = solver.current_pos.copy()
    sizes = solver.sizes.copy()

    return {
        'node_features': node_features,     # (N, 10)
        'positions': positions,             # (N, 2)
        'sizes': sizes,                     # (N, 2)
        'target_mask': mask,                # (N,)
        'weight': weight,                   # scalar
        'best_delta': best_delta,           # scalar
        'best_strategy': [r[0] for r in all_results if r[1] is best_subset][0],
    }


def collect_mode(args):
    """Run ALNS on circuits, sample states, evaluate best-of-4, save dataset."""
    np.random.seed(args.seed)
    strategies = ['random', 'worst_hpwl', 'congestion', 'connected']

    circuits = args.circuits.split(',')
    print(f"Collecting imitation data from {len(circuits)} circuits")
    print(f"  {args.n_iterations} ALNS iterations, sample every {args.sample_every}")
    print(f"  k={args.subset_size}, strategies={strategies}")

    all_samples = []
    # Also save per-circuit graph structure for training
    graph_data = {}

    for cname in circuits:
        print(f"\n  {cname}...", end=' ', flush=True)
        cdata = load_and_legalize(cname, args.benchmark_base, args.seed)
        N = cdata['n_components']
        print(f"({N} macros)")

        # Save graph structure (edge_index, edge_attr) per circuit
        graph_data[cname] = {
            'edge_index': cdata['edge_index'],
            'edge_attr': cdata['edge_attr'],
            'n_components': N,
        }

        solver = make_pure_solver(cdata, args)

        # Skip first few iterations (warmup) before sampling
        warmup = min(20, args.n_iterations // 5)
        circuit_samples = 0
        circuit_skipped = 0

        for it in range(args.n_iterations):
            # Run one ALNS step to advance state
            solver.step()

            # Sample after warmup, at intervals
            if it >= warmup and (it - warmup) % args.sample_every == 0:
                sample = collect_state_sample(
                    solver, args.subset_size, strategies)
                if sample is not None:
                    sample['circuit_name'] = cname
                    sample['iteration'] = it
                    all_samples.append(sample)
                    circuit_samples += 1
                else:
                    circuit_skipped += 1

        print(f"    {circuit_samples} samples collected "
              f"({circuit_skipped} skipped, all infeasible)")
        print(f"    Final HPWL: {solver.best_hpwl:.2f}")

    print(f"\nTotal samples: {len(all_samples)}")

    # Compute weight statistics for normalization
    weights = np.array([s['weight'] for s in all_samples])
    nonzero = weights[weights > 0]
    if len(nonzero) > 0:
        p95 = np.percentile(nonzero, 95)
        print(f"Weight stats: mean={weights.mean():.4f} "
              f"p95={p95:.4f} max={weights.max():.4f} "
              f"zero_frac={np.mean(weights == 0):.2%}")
        # Normalize: clip to p95, then scale to [0, 1]
        for s in all_samples:
            s['weight'] = min(s['weight'] / max(p95, 1e-8), 1.0)
    else:
        print("WARNING: all weights are zero (no improving subsets found)")

    # Strategy distribution
    from collections import Counter
    strat_counts = Counter(s['best_strategy'] for s in all_samples)
    print(f"Best-of-4 strategy distribution: {dict(strat_counts)}")

    # Save
    os.makedirs(os.path.dirname(args.save_path) or '.', exist_ok=True)
    torch.save({
        'samples': all_samples,
        'graph_data': graph_data,
        'args': vars(args),
    }, args.save_path)
    print(f"Saved to {args.save_path}")


# ---------------------------------------------------------------------------
# Step 2: Imitation training (weighted BCE)
# ---------------------------------------------------------------------------

def imitate_mode(args):
    """Train subset_head with weighted BCE on best-of-4 subset masks."""
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load data
    data = torch.load(args.data_path, map_location='cpu', weights_only=False)
    samples = data['samples']
    graph_data = data['graph_data']
    n_samples = len(samples)
    print(f"Loaded {n_samples} samples from {args.data_path}")
    print(f"Circuits: {list(graph_data.keys())}")

    # Group samples by circuit (needed for GNN forward: each circuit has
    # different edge_index/edge_attr)
    circuit_samples = {}
    for s in samples:
        cname = s['circuit_name']
        if cname not in circuit_samples:
            circuit_samples[cname] = []
        circuit_samples[cname].append(s)
    for cname, ss in circuit_samples.items():
        print(f"  {cname}: {len(ss)} samples")

    # Model
    model = NetSpatialGNN(
        node_input_dim=10, mode='dual', hidden_dim=64, n_layers=5)
    if args.checkpoint:
        model.load_state_dict(
            torch.load(args.checkpoint, map_location='cpu', weights_only=True))
        print(f"Loaded checkpoint from {args.checkpoint}")
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params")

    # Freeze heads irrelevant to subset selection
    for head in [model.displacement_head, model.heatmap_head, model.value_mlp]:
        for p in head.parameters():
            p.requires_grad = False
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    print(f"Trainable params: {n_trainable:,} "
          f"(frozen: displacement_head, heatmap_head, value_mlp)")
    optimizer = torch.optim.Adam(trainable, lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.n_epochs)

    os.makedirs(args.save_dir, exist_ok=True)
    best_loss = float('inf')

    # Pre-cache graph structure tensors per circuit
    graph_tensors = {}
    for cname, gd in graph_data.items():
        graph_tensors[cname] = {
            'edge_index_t': torch.from_numpy(
                gd['edge_index']).long().to(device),
            'edge_attr_t': torch.from_numpy(
                gd['edge_attr']).float().to(device),
        }

    print(f"\n{'='*60}")
    print(f"Imitation training: {args.n_epochs} epochs, "
          f"batch_size={args.imitation_batch}, lr={args.lr}")
    print(f"{'='*60}")

    for epoch in range(args.n_epochs):
        model.train()
        epoch_t0 = time.time()

        # Shuffle samples within each circuit, then interleave
        all_indices = []
        for cname, ss in circuit_samples.items():
            perm = np.random.permutation(len(ss))
            for i in perm:
                all_indices.append((cname, i))
        np.random.shuffle(all_indices)

        epoch_loss = 0.0
        epoch_bce = 0.0
        epoch_count = 0
        n_batches = 0

        # Mini-batch: accumulate gradients over imitation_batch samples,
        # then step. Samples within a batch may come from different circuits.
        optimizer.zero_grad()
        batch_loss = torch.tensor(0.0, device=device)
        batch_count = 0

        for idx, (cname, sidx) in enumerate(all_indices):
            s = circuit_samples[cname][sidx]
            gt = graph_tensors[cname]

            # Build inputs
            node_feats = torch.from_numpy(s['node_features']).to(device)
            positions = torch.from_numpy(s['positions']).float().to(device)
            sizes = torch.from_numpy(s['sizes']).float().to(device)
            target = torch.from_numpy(s['target_mask']).to(device)
            weight = s['weight']

            # Forward
            outputs = model(
                node_feats, positions, sizes,
                gt['edge_index_t'], gt['edge_attr_t'],
            )
            logits = outputs['subset_logits']  # (N,)

            # Weighted BCE loss with pos_weight for class imbalance
            # k/N ≈ 4% positive → weight positives by (N-k)/k
            n_pos = target.sum().clamp(min=1)
            n_neg = (target.numel() - n_pos).clamp(min=1)
            pw = (n_neg / n_pos).unsqueeze(0)  # scalar tensor for BCE
            sample_weight = max(weight, 0.1)
            bce = F.binary_cross_entropy_with_logits(
                logits, target, pos_weight=pw, reduction='mean')
            batch_loss = batch_loss + bce * sample_weight
            batch_count += 1

            epoch_bce += bce.item()
            epoch_count += 1

            # Step every imitation_batch samples
            if batch_count >= args.imitation_batch:
                batch_loss = batch_loss / batch_count
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

                epoch_loss += batch_loss.item()
                n_batches += 1
                batch_loss = torch.tensor(0.0, device=device)
                batch_count = 0

        # Flush remaining
        if batch_count > 0:
            batch_loss = batch_loss / batch_count
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += batch_loss.item()
            n_batches += 1

        scheduler.step()
        epoch_dt = time.time() - epoch_t0

        avg_loss = epoch_loss / max(n_batches, 1)
        avg_bce = epoch_bce / max(epoch_count, 1)

        # Policy sharpness diagnostic
        model.eval()
        with torch.no_grad():
            # Pick a random sample for sharpness check
            diag_cname, diag_sidx = all_indices[0]
            diag_s = circuit_samples[diag_cname][diag_sidx]
            diag_gt = graph_tensors[diag_cname]
            diag_feats = torch.from_numpy(
                diag_s['node_features']).to(device)
            diag_pos = torch.from_numpy(
                diag_s['positions']).float().to(device)
            diag_sizes = torch.from_numpy(
                diag_s['sizes']).float().to(device)
            diag_out = model(
                diag_feats, diag_pos, diag_sizes,
                diag_gt['edge_index_t'], diag_gt['edge_attr_t'],
            )
            diag_logits = diag_out['subset_logits']
            logit_std = diag_logits.std().item()
            p = torch.softmax(diag_logits, 0)
            entropy = -(p * torch.log(p + 1e-8)).sum().item()
            max_entropy = np.log(diag_logits.size(0))

        if (epoch + 1) % max(args.n_epochs // 20, 1) == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}: loss={avg_loss:.4f} "
                  f"bce={avg_bce:.4f} "
                  f"logit_std={logit_std:.3f} "
                  f"entropy={entropy:.2f}/{max_entropy:.2f} "
                  f"({epoch_dt:.1f}s)")

        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt = os.path.join(args.save_dir, 'best_model.pt')
            torch.save(model.state_dict(), ckpt)

    # Save final
    final = os.path.join(args.save_dir, 'final_model.pt')
    torch.save(model.state_dict(), final)
    print(f"\nSaved final model to {final}")
    print(f"Best loss: {best_loss:.4f}")


# ---------------------------------------------------------------------------
# Step 3: Evaluate warm-start quality
# ---------------------------------------------------------------------------

def eval_mode(args):
    """
    Evaluate warm-start model vs untrained baseline.
    Checks: entropy, heuristic overlap, learned-only HPWL, ALNS+learned.
    """
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load trained model
    trained = NetSpatialGNN(
        node_input_dim=10, mode='dual', hidden_dim=64, n_layers=5)
    if args.checkpoint:
        trained.load_state_dict(
            torch.load(args.checkpoint, map_location=device, weights_only=True))
        print(f"Loaded trained model from {args.checkpoint}")
    trained.to(device)
    trained.eval()

    # Untrained baseline
    torch.manual_seed(999)
    untrained = NetSpatialGNN(
        node_input_dim=10, mode='dual', hidden_dim=64, n_layers=5)
    untrained.to(device)
    untrained.eval()

    test_circuits = args.test_circuits.split(',')

    for cname in test_circuits:
        print(f"\n{'='*60}")
        print(f"Evaluating {cname}")
        print(f"{'='*60}")

        cdata = load_and_legalize(cname, args.benchmark_base, args.seed)
        N = cdata['n_components']

        # Build a solver to get features and heuristic subsets
        solver = make_pure_solver(cdata, args)
        # Run a few ALNS steps to get to a non-trivial state
        for _ in range(20):
            solver.step()

        # --- Diagnostic 1: Policy sharpness ---
        node_features = torch.from_numpy(
            build_features_np(solver)).to(device)
        positions_t = torch.from_numpy(
            solver.current_pos).float().to(device)
        sizes_t = torch.from_numpy(solver.sizes).float().to(device)
        edge_index_t = torch.from_numpy(
            cdata['edge_index']).long().to(device)
        edge_attr_t = torch.from_numpy(
            cdata['edge_attr']).float().to(device)

        with torch.no_grad():
            out_t = trained(node_features, positions_t, sizes_t,
                            edge_index_t, edge_attr_t)
            out_u = untrained(node_features, positions_t, sizes_t,
                              edge_index_t, edge_attr_t)

        logits_t = out_t['subset_logits']
        logits_u = out_u['subset_logits']

        p_t = torch.softmax(logits_t, 0)
        p_u = torch.softmax(logits_u, 0)
        ent_t = -(p_t * torch.log(p_t + 1e-8)).sum().item()
        ent_u = -(p_u * torch.log(p_u + 1e-8)).sum().item()
        max_ent = np.log(N)

        print(f"\n  Policy sharpness ({N} macros, max_ent={max_ent:.2f}):")
        print(f"    Trained:   std={logits_t.std():.4f} "
              f"entropy={ent_t:.2f} "
              f"top10_gap={torch.topk(logits_t,10).values.mean() - logits_t.median():.4f}")
        print(f"    Untrained: std={logits_u.std():.4f} "
              f"entropy={ent_u:.2f} "
              f"top10_gap={torch.topk(logits_u,10).values.mean() - logits_u.median():.4f}")

        # --- Diagnostic 2: Heuristic overlap ---
        k = args.subset_size
        top_k_trained = set(
            torch.topk(logits_t, k).indices.cpu().numpy().tolist())

        strategies = ['connected', 'congestion', 'worst_hpwl', 'random']
        print(f"\n  Subset overlap (top-{k}):")
        for strat in strategies:
            heur_sub = set(solver.get_neighborhood(strat, k).tolist())
            overlap = len(top_k_trained & heur_sub)
            print(f"    trained vs {strat:12s}: {overlap}/{k}")

        top_k_untrained = set(
            torch.topk(logits_u, k).indices.cpu().numpy().tolist())
        print(f"    trained vs untrained:  "
              f"{len(top_k_trained & top_k_untrained)}/{k}")

        # --- Diagnostic 3: Three-way HPWL eval ---
        print(f"\n  Three-way eval ({args.n_iterations} iterations):")

        # Pure ALNS
        solver_pure = make_pure_solver(cdata, args)
        t0 = time.time()
        res_pure = solver_pure.solve(
            n_iterations=args.n_iterations, log_every=999, verbose=False)
        dt_pure = time.time() - t0
        print(f"    Pure ALNS:      HPWL={res_pure['best_hpwl']:.2f} "
              f"accepted={solver_pure.n_accepted}/{args.n_iterations} "
              f"({dt_pure:.1f}s)")

        # Trained learned-only
        solver_trained = make_model_solver(cdata, trained, args)
        solver_trained.strategies = ['learned']
        solver_trained.strategy_attempts = {'learned': 0}
        solver_trained.strategy_successes = {'learned': 0}
        solver_trained.alns = ALNSWeights(
            n_strategies=1, segment_size=25, rho=0.1)
        t0 = time.time()
        res_trained = solver_trained.solve(
            n_iterations=args.n_iterations, log_every=999, verbose=False)
        dt_trained = time.time() - t0
        print(f"    Trained learn:  HPWL={res_trained['best_hpwl']:.2f} "
              f"accepted={solver_trained.n_accepted}/{args.n_iterations} "
              f"infeas={solver_trained.n_infeasible} "
              f"({dt_trained:.1f}s)")

        # Untrained learned-only
        solver_untrained = make_model_solver(cdata, untrained, args)
        solver_untrained.strategies = ['learned']
        solver_untrained.strategy_attempts = {'learned': 0}
        solver_untrained.strategy_successes = {'learned': 0}
        solver_untrained.alns = ALNSWeights(
            n_strategies=1, segment_size=25, rho=0.1)
        t0 = time.time()
        res_untrained = solver_untrained.solve(
            n_iterations=args.n_iterations, log_every=999, verbose=False)
        dt_untrained = time.time() - t0
        print(f"    Untrained learn: HPWL={res_untrained['best_hpwl']:.2f} "
              f"accepted={solver_untrained.n_accepted}/{args.n_iterations} "
              f"infeas={solver_untrained.n_infeasible} "
              f"({dt_untrained:.1f}s)")

        # ALNS + trained
        solver_mixed = make_model_solver(cdata, trained, args)
        t0 = time.time()
        res_mixed = solver_mixed.solve(
            n_iterations=args.n_iterations, log_every=999, verbose=False)
        dt_mixed = time.time() - t0
        w = solver_mixed.alns.get_weights_dict(solver_mixed.strategies)
        print(f"    ALNS+trained:   HPWL={res_mixed['best_hpwl']:.2f} "
              f"accepted={solver_mixed.n_accepted}/{args.n_iterations} "
              f"({dt_mixed:.1f}s)")
        print(f"    Strategy attempts: {dict(solver_mixed.strategy_attempts)}")
        print(f"    ALNS weights: { {k: f'{v:.3f}' for k,v in w.items()} }")

        # --- Success criteria ---
        print(f"\n  Success criteria:")
        ent_ratio = ent_t / max_ent
        print(f"    [{'PASS' if ent_ratio < 0.95 else 'FAIL'}] "
              f"Entropy < 0.95*max: {ent_t:.2f} / {max_ent:.2f} "
              f"= {ent_ratio:.3f}")
        print(f"    [{'PASS' if res_trained['best_hpwl'] < res_untrained['best_hpwl'] else 'FAIL'}] "
              f"Trained < untrained: "
              f"{res_trained['best_hpwl']:.2f} vs {res_untrained['best_hpwl']:.2f}")
        mixed_vs_pure = res_mixed['best_hpwl'] / res_pure['best_hpwl']
        print(f"    [{'PASS' if mixed_vs_pure <= 1.0 else 'FAIL'}] "
              f"ALNS+learned <= pure ALNS: "
              f"{res_mixed['best_hpwl']:.2f} vs {res_pure['best_hpwl']:.2f} "
              f"(ratio={mixed_vs_pure:.3f})"
              f"{' [WARN: >5% regression]' if mixed_vs_pure > 1.05 else ''}")


# ---------------------------------------------------------------------------
# Step 4: RL fine-tune with KL regularization
# ---------------------------------------------------------------------------

def finetune_mode(args):
    """RL fine-tune imitation policy with KL regularization."""
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load imitation policy as reference (frozen)
    ref_model = NetSpatialGNN(
        node_input_dim=10, mode='dual', hidden_dim=64, n_layers=5)
    ref_model.load_state_dict(
        torch.load(args.checkpoint, map_location=device, weights_only=True))
    ref_model.to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False
    print(f"Loaded reference (frozen) from {args.checkpoint}")

    # Trainable model (start from same checkpoint)
    model = NetSpatialGNN(
        node_input_dim=10, mode='dual', hidden_dim=64, n_layers=5)
    model.load_state_dict(
        torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable model: {n_params:,} params")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.n_epochs)

    # Load circuits
    train_circuits = args.train_circuits.split(',')
    print(f"\nLoading {len(train_circuits)} training circuits...")
    train_data = []
    for c in train_circuits:
        print(f"  {c}...", end=' ', flush=True)
        d = load_and_legalize(c, args.benchmark_base, args.seed)
        print(f"({d['n_components']} macros)")
        train_data.append(d)

    os.makedirs(args.save_dir, exist_ok=True)
    best_reward = -float('inf')

    print(f"\n{'='*60}")
    print(f"RL fine-tune: {args.n_epochs} epochs, "
          f"batch={args.batch_size}, kl_weight={args.kl_weight}")
    print(f"{'='*60}")

    for epoch in range(args.n_epochs):
        model.train()
        epoch_t0 = time.time()

        # Temperature: fixed low for fine-tuning
        temperature = args.temperature

        # Fresh solvers
        np.random.shuffle(train_data)
        solvers = []
        for d in train_data:
            solvers.append(make_model_solver(d, model, args))

        n_steps = args.steps_per_circuit * len(train_data)
        n_batches = max(n_steps // args.batch_size, 1)

        epoch_metrics = {
            'policy_loss': [], 'value_loss': [], 'kl_loss': [],
            'mean_reward': [], 'feasible_rate': [], 'entropy': [],
        }
        solver_idx = [0]

        for batch_idx in range(n_batches):
            log_probs = []
            rewards = []
            values = []
            entropies = []
            kl_divs = []
            n_feasible = 0

            for _ in range(args.batch_size):
                solver = solvers[solver_idx[0] % len(solvers)]
                solver_idx[0] += 1

                # Warmup steps
                for _ in range(args.heuristic_warmup):
                    solver.step()

                # GNN forward
                node_features = solver._build_gnn_features()
                positions_t = torch.from_numpy(
                    solver.current_pos).float().to(device)
                ct = solver._cached_tensors

                outputs = model(
                    node_features, positions_t, ct['sizes_t'],
                    ct['edge_index_t'], ct['edge_attr_t'],
                )
                logits = outputs['subset_logits']
                value = outputs['value']

                # Reference logits (frozen)
                with torch.no_grad():
                    ref_out = ref_model(
                        node_features, positions_t, ct['sizes_t'],
                        ct['edge_index_t'], ct['edge_attr_t'],
                    )
                    ref_logits = ref_out['subset_logits']

                # KL divergence: KL(current || reference)
                # batchmean normalizes by input size, so KL is
                # comparable across circuits with different macro counts
                kl = F.kl_div(
                    F.log_softmax(logits, dim=0),
                    F.softmax(ref_logits, dim=0),
                    reduction='batchmean',
                )
                kl_divs.append(kl)

                # Sample subset
                k = min(solver.subset_size, solver.N)
                indices, log_prob = NetSpatialGNN.select_subset(
                    logits, k=k, temperature=temperature, explore=True)
                subset_np = indices.detach().cpu().numpy()

                # CP-SAT solve
                pre_positions = solver.current_pos.copy()
                old_cost = solver.current_cost
                new_positions = solve_subset(
                    solver.current_pos, solver.sizes, solver.nets, subset_np,
                    time_limit=solver.cpsat_time_limit,
                    window_fraction=solver.window_fraction,
                )

                if new_positions is not None:
                    new_hpwl, _ = compute_incremental_hpwl(
                        new_positions, solver.nets, subset_np,
                        solver.net_hpwls, solver.macro_nets)
                    new_cost = solver._compute_cost(new_positions, new_hpwl)
                    reward = old_cost - new_cost
                    n_feasible += 1
                else:
                    reward = -1.0

                solver._apply_candidate_result(
                    new_positions, subset_np, 'learned', pre_positions)

                log_probs.append(log_prob)
                rewards.append(torch.tensor([reward], device=device))
                values.append(value)

                p = torch.softmax(logits, 0)
                ent = -(p * torch.log(p + 1e-8)).sum()
                entropies.append(ent)

            # Stack and compute losses
            log_probs = torch.cat(log_probs)
            rewards = torch.cat(rewards)
            values = torch.cat(values)
            entropies = torch.stack(entropies)
            kl_divs = torch.stack(kl_divs)

            advantages = rewards - values.detach()
            if advantages.std() > 1e-8:
                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-8)

            policy_loss = -(log_probs * advantages).mean()
            value_loss = F.mse_loss(values, rewards)
            kl_loss = kl_divs.mean()
            entropy_bonus = entropies.mean()

            loss = (policy_loss
                    + 0.5 * value_loss
                    + args.kl_weight * kl_loss
                    - 0.01 * entropy_bonus)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_metrics['policy_loss'].append(policy_loss.item())
            epoch_metrics['value_loss'].append(value_loss.item())
            epoch_metrics['kl_loss'].append(kl_loss.item())
            epoch_metrics['mean_reward'].append(rewards.mean().item())
            epoch_metrics['feasible_rate'].append(
                n_feasible / args.batch_size)
            epoch_metrics['entropy'].append(entropy_bonus.item())

        scheduler.step()
        epoch_dt = time.time() - epoch_t0

        avg = {k: np.mean(v) for k, v in epoch_metrics.items()}
        print(f"  Epoch {epoch+1:3d}: "
              f"ploss={avg['policy_loss']:.4f} "
              f"vloss={avg['value_loss']:.4f} "
              f"kl={avg['kl_loss']:.4f} "
              f"ent={avg['entropy']:.2f} "
              f"reward={avg['mean_reward']:.4f} "
              f"feas={avg['feasible_rate']:.2f} "
              f"({epoch_dt:.1f}s)")

        if avg['mean_reward'] > best_reward:
            best_reward = avg['mean_reward']
            ckpt = os.path.join(args.save_dir, 'best_finetuned.pt')
            torch.save(model.state_dict(), ckpt)

    final = os.path.join(args.save_dir, 'final_finetuned.pt')
    torch.save(model.state_dict(), final)
    print(f"\nSaved final fine-tuned model to {final}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Heuristic Imitation + RL Fine-tune for ALNS')
    subparsers = parser.add_subparsers(dest='mode', help='Mode')

    # Common args
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--benchmark_base', type=str, default='benchmarks')
    common.add_argument('--seed', type=int, default=42)
    common.add_argument('--subset_size', type=int, default=10)
    common.add_argument('--window_fraction', type=float, default=0.15)
    common.add_argument('--cpsat_time_limit', type=float, default=0.2)
    common.add_argument('--congestion_weight', type=float, default=0.0)
    common.add_argument('--plateau_threshold', type=int, default=20)
    common.add_argument('--adapt_threshold', type=int, default=30)

    # Collect mode
    collect_parser = subparsers.add_parser('collect', parents=[common])
    collect_parser.add_argument('--circuits', type=str, required=True,
                                help='Comma-separated circuit names')
    collect_parser.add_argument('--n_iterations', type=int, default=200)
    collect_parser.add_argument('--sample_every', type=int, default=2)
    collect_parser.add_argument('--save_path', type=str,
                                default='imitation_data/train.pt')

    # Imitate mode
    imitate_parser = subparsers.add_parser('imitate', parents=[common])
    imitate_parser.add_argument('--data_path', type=str, required=True)
    imitate_parser.add_argument('--checkpoint', type=str, default=None,
                                help='Resume from checkpoint')
    imitate_parser.add_argument('--n_epochs', type=int, default=100)
    imitate_parser.add_argument('--imitation_batch', type=int, default=32)
    imitate_parser.add_argument('--lr', type=float, default=3e-4)
    imitate_parser.add_argument('--save_dir', type=str,
                                default='checkpoints_imitation')

    # Eval mode
    eval_parser = subparsers.add_parser('eval', parents=[common])
    eval_parser.add_argument('--test_circuits', type=str, required=True)
    eval_parser.add_argument('--checkpoint', type=str, default=None)
    eval_parser.add_argument('--n_iterations', type=int, default=100)

    # Finetune mode
    ft_parser = subparsers.add_parser('finetune', parents=[common])
    ft_parser.add_argument('--train_circuits', type=str, required=True)
    ft_parser.add_argument('--checkpoint', type=str, required=True,
                           help='Imitation checkpoint to fine-tune from')
    ft_parser.add_argument('--n_epochs', type=int, default=20)
    ft_parser.add_argument('--steps_per_circuit', type=int, default=50)
    ft_parser.add_argument('--batch_size', type=int, default=8)
    ft_parser.add_argument('--lr', type=float, default=1e-5)
    ft_parser.add_argument('--temperature', type=float, default=0.1)
    ft_parser.add_argument('--kl_weight', type=float, default=0.1,
                           help='KL penalty to imitation reference')
    ft_parser.add_argument('--heuristic_warmup', type=int, default=3)
    ft_parser.add_argument('--save_dir', type=str,
                           default='checkpoints_finetuned')
    ft_parser.add_argument('--alns_segment_size', type=int, default=25)
    ft_parser.add_argument('--alns_rho', type=float, default=0.1)

    args = parser.parse_args()

    if args.mode == 'collect':
        collect_mode(args)
    elif args.mode == 'imitate':
        imitate_mode(args)
    elif args.mode == 'eval':
        eval_mode(args)
    elif args.mode == 'finetune':
        finetune_mode(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
