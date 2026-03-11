"""
RL-Trained Destroy Operator for ALNS Macro Placement

One-step actor-critic training: GNN selects subsets, CP-SAT repairs.
Batched gradient updates with advantage normalization.

Usage:
    # Train on ibm01 (quick test)
    python train_rl_lns.py train --train_circuits ibm01 --n_epochs 5

    # Train on multiple circuits
    python train_rl_lns.py train \
        --train_circuits ibm01,ibm02,ibm03,ibm04,ibm06,ibm07,ibm08,ibm09,ibm10,ibm11,ibm12 \
        --val_circuits ibm13,ibm14,ibm15 \
        --n_epochs 50

    # Evaluate (three-way comparison)
    python train_rl_lns.py eval \
        --test_circuits ibm01 \
        --checkpoint checkpoints_rl/best_model.pt \
        --n_iterations 200
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
from lns_solver import LNSSolver, compute_rudy_np
from net_spatial_gnn import NetSpatialGNN


def load_and_legalize(circuit_name, benchmark_base='benchmarks', seed=42):
    """Load circuit, legalize reference placement, return data dict."""
    circuit_dir = os.path.join(benchmark_base, "iccad04", "extracted", circuit_name)
    data = load_bookshelf_circuit(
        circuit_dir, circuit_name, macros_only=True, seed=seed)

    positions = data['positions']
    sizes = data['node_features']
    nets = data['nets']

    # Legalize if needed
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
        'nets': nets,
        'edge_index': data['edge_index'],
        'edge_attr': data['edge_attr'],
        'n_components': data['n_components'],
    }


def make_solver(circuit_data, model, args):
    """Create an LNSSolver from circuit data."""
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


def collect_transition(model, solver, temperature, device):
    """
    One RL transition: GNN -> subset -> CP-SAT -> reward.
    No gradient step — returns tensors for batched update.
    Uses solver's adaptive subset_size (not a fixed external value).
    """
    node_features = solver._build_gnn_features()
    positions_t = torch.from_numpy(
        solver.current_pos).float().to(device)
    ct = solver._cached_tensors

    # Forward pass WITH gradients
    outputs = model(
        node_features, positions_t, ct['sizes_t'],
        ct['edge_index_t'], ct['edge_attr_t'],
    )
    subset_logits = outputs['subset_logits']
    value = outputs['value']  # (1,)

    # Sample subset via Gumbel-top-k — use solver's adaptive subset_size
    k = min(solver.subset_size, solver.N)
    indices, log_prob = NetSpatialGNN.select_subset(
        subset_logits, k=k, temperature=temperature, explore=True)
    subset_np = indices.detach().cpu().numpy()

    # Save pre-solve state
    pre_positions = solver.current_pos.copy()
    old_cost = solver.current_cost

    # CP-SAT solve
    new_positions = solve_subset(
        solver.current_pos, solver.sizes, solver.nets, subset_np,
        time_limit=solver.cpsat_time_limit,
        window_fraction=solver.window_fraction,
    )

    # Compute reward = old_cost - new_cost (positive = improvement)
    if new_positions is not None:
        new_hpwl, _ = compute_incremental_hpwl(
            new_positions, solver.nets, subset_np,
            solver.net_hpwls, solver.macro_nets)
        new_cost = solver._compute_cost(new_positions, new_hpwl)
        reward = old_cost - new_cost  # positive = improvement
        feasible = True
    else:
        reward = -1.0  # infeasible penalty
        feasible = False

    # Apply state transition via centralized function
    solver._apply_candidate_result(
        new_positions, subset_np, 'learned', pre_positions)

    # Entropy of subset logits
    p = torch.softmax(subset_logits, 0)
    entropy = -(p * torch.log(p + 1e-8)).sum()

    return (
        log_prob,
        torch.tensor([reward], device=device),
        value,
        entropy,
        feasible,
    )


def train_batch(model, optimizer, solvers, temperature, args, device,
                solver_idx_state):
    """
    Collect B transitions round-robin across solvers,
    normalize advantages, one optimizer step.

    Between each RL transition, run heuristic_warmup ALNS steps
    to keep solvers in good states (local, feasible neighborhoods).
    """
    log_probs = []
    rewards = []
    values = []
    entropies = []
    n_feasible = 0

    for _ in range(args.batch_size):
        solver = solvers[solver_idx_state[0] % len(solvers)]
        solver_idx_state[0] += 1

        # Warmup: run a few ALNS steps (any strategy, including learned
        # which uses torch.no_grad) to advance solver state
        for _ in range(args.heuristic_warmup):
            solver.step()

        lp, r, v, ent, feas = collect_transition(
            model, solver, temperature, device)
        log_probs.append(lp)
        rewards.append(r)
        values.append(v)
        entropies.append(ent)
        if feas:
            n_feasible += 1

    # Stack
    log_probs = torch.cat(log_probs)       # (B,)
    rewards = torch.cat(rewards)            # (B,)
    values = torch.cat(values)              # (B,)
    entropies = torch.stack(entropies)      # (B,)

    # Normalize advantages over batch
    advantages = rewards - values.detach()
    if advantages.std() > 1e-8:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Loss
    policy_loss = -(log_probs * advantages).mean()
    value_loss = F.mse_loss(values, rewards)
    entropy_bonus = entropies.mean()
    loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_bonus

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return {
        'policy_loss': policy_loss.item(),
        'value_loss': value_loss.item(),
        'entropy': entropy_bonus.item(),
        'mean_reward': rewards.mean().item(),
        'feasible_rate': n_feasible / args.batch_size,
        'mean_value': values.mean().item(),
    }


def evaluate_solver(solver, n_iterations, log_every=50):
    """Run solver for n_iterations, return best HPWL and stats."""
    t0 = time.time()
    result = solver.solve(
        n_iterations=n_iterations, log_every=log_every, verbose=False)
    dt = time.time() - t0

    return {
        'best_hpwl': result['best_hpwl'],
        'best_cost': result['best_cost'],
        'n_accepted': solver.n_accepted,
        'n_improved': solver.n_improved,
        'n_infeasible': solver.n_infeasible,
        'wall_time': dt,
        'history': result['history'],
        'alns_weights': solver.alns.get_weights_dict(solver.strategies),
        'strategy_successes': dict(solver.strategy_successes),
        'strategy_attempts': dict(solver.strategy_attempts),
    }


def evaluate(model, circuit_data_list, args, device):
    """
    Three-way evaluation: pure ALNS / learned-only / ALNS+learned.
    Returns per-circuit results.
    """
    results = {}

    for cdata in circuit_data_list:
        cname = cdata['circuit_name']
        print(f"\n  Evaluating {cname} ({cdata['n_components']} macros)...")
        circuit_results = {}

        # 1. Pure ALNS (no model)
        solver_pure = LNSSolver(
            positions=cdata['positions'].copy(),
            sizes=cdata['sizes'],
            nets=cdata['nets'],
            edge_index=cdata['edge_index'],
            congestion_weight=args.congestion_weight,
            subset_size=args.subset_size,
            window_fraction=args.window_fraction,
            cpsat_time_limit=args.cpsat_time_limit,
            plateau_threshold=args.plateau_threshold,
            adapt_threshold=args.adapt_threshold,
            seed=args.seed,
        )
        circuit_results['pure_alns'] = evaluate_solver(
            solver_pure, args.n_iterations)

        # 2. Learned-only (force 'learned' strategy every iteration)
        if model is not None:
            model.eval()
            solver_learned = make_solver(cdata, model, args)
            # Override strategy selection to always use 'learned'
            solver_learned.strategies = ['learned']
            solver_learned.strategy_attempts = {'learned': 0}
            solver_learned.strategy_successes = {'learned': 0}
            solver_learned.alns = type(solver_learned.alns)(
                n_strategies=1, segment_size=args.alns_segment_size,
                rho=args.alns_rho)
            circuit_results['learned_only'] = evaluate_solver(
                solver_learned, args.n_iterations)

        # 3. ALNS + learned (5 strategies)
        if model is not None:
            solver_mixed = make_solver(cdata, model, args)
            circuit_results['alns_learned'] = evaluate_solver(
                solver_mixed, args.n_iterations)

        # Print summary
        pure = circuit_results['pure_alns']
        print(f"    Pure ALNS:     HPWL={pure['best_hpwl']:.2f} "
              f"accepted={pure['n_accepted']}/{args.n_iterations} "
              f"({pure['wall_time']:.1f}s)")
        if 'learned_only' in circuit_results:
            lo = circuit_results['learned_only']
            print(f"    Learned-only:  HPWL={lo['best_hpwl']:.2f} "
                  f"accepted={lo['n_accepted']}/{args.n_iterations} "
                  f"({lo['wall_time']:.1f}s)")
        if 'alns_learned' in circuit_results:
            al = circuit_results['alns_learned']
            print(f"    ALNS+learned:  HPWL={al['best_hpwl']:.2f} "
                  f"accepted={al['n_accepted']}/{args.n_iterations} "
                  f"({al['wall_time']:.1f}s)")
            if 'learned' in al.get('alns_weights', {}):
                print(f"    Learned ALNS weight: "
                      f"{al['alns_weights']['learned']:.4f}")

        results[cname] = circuit_results

    return results


def train_mode(args):
    """Online RL training with batched actor-critic."""
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Model
    model = NetSpatialGNN(
        node_input_dim=10, mode='dual', hidden_dim=64, n_layers=5)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: NetSpatialGNN (dual, {n_params:,} params)")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.n_epochs)

    # Load circuits
    train_circuits = args.train_circuits.split(',')
    val_circuits = args.val_circuits.split(',') if args.val_circuits else []

    print(f"\nLoading {len(train_circuits)} training circuits...")
    train_data = []
    for c in train_circuits:
        print(f"  {c}...", end=' ', flush=True)
        d = load_and_legalize(c, args.benchmark_base, args.seed)
        print(f"({d['n_components']} macros)")
        train_data.append(d)

    val_data = []
    if val_circuits:
        print(f"Loading {len(val_circuits)} validation circuits...")
        for c in val_circuits:
            print(f"  {c}...", end=' ', flush=True)
            d = load_and_legalize(c, args.benchmark_base, args.seed)
            print(f"({d['n_components']} macros)")
            val_data.append(d)

    # Checkpointing
    os.makedirs(args.save_dir, exist_ok=True)
    best_val_hpwl = float('inf')

    print(f"\n{'='*60}")
    print(f"Training: {args.n_epochs} epochs, "
          f"{args.steps_per_circuit} steps/circuit, "
          f"batch_size={args.batch_size}")
    print(f"  temperature: {args.temp_start} -> {args.temp_end}")
    print(f"  heuristic_warmup: {args.heuristic_warmup} steps/transition")
    print(f"{'='*60}")

    for epoch in range(args.n_epochs):
        model.train()
        epoch_t0 = time.time()

        # Temperature schedule: linear anneal from temp_start to temp_end
        epoch_frac = epoch / max(args.n_epochs - 1, 1)
        temperature = args.temp_start + (args.temp_end - args.temp_start) * epoch_frac

        # Fresh solvers each epoch (from legalized reference)
        np.random.shuffle(train_data)
        solvers = []
        for d in train_data:
            s = make_solver(d, model, args)
            solvers.append(s)

        n_batches = (args.steps_per_circuit * len(train_data)) // args.batch_size
        n_batches = max(n_batches, 1)

        epoch_metrics = {
            'policy_loss': [], 'value_loss': [], 'entropy': [],
            'mean_reward': [], 'feasible_rate': [],
        }
        solver_idx_state = [0]

        for batch_idx in range(n_batches):
            metrics = train_batch(
                model, optimizer, solvers, temperature, args, device,
                solver_idx_state)

            for k in epoch_metrics:
                epoch_metrics[k].append(metrics[k])

            if (batch_idx + 1) % max(n_batches // 5, 1) == 0:
                print(f"  [{epoch+1}/{args.n_epochs}] batch {batch_idx+1}/{n_batches}: "
                      f"ploss={metrics['policy_loss']:.4f} "
                      f"vloss={metrics['value_loss']:.4f} "
                      f"ent={metrics['entropy']:.2f} "
                      f"reward={metrics['mean_reward']:.4f} "
                      f"feas={metrics['feasible_rate']:.2f}")

        scheduler.step()
        epoch_dt = time.time() - epoch_t0

        # Epoch summary
        avg = {k: np.mean(v) for k, v in epoch_metrics.items()}
        print(f"\n  Epoch {epoch+1}: "
              f"ploss={avg['policy_loss']:.4f} "
              f"vloss={avg['value_loss']:.4f} "
              f"ent={avg['entropy']:.2f} "
              f"reward={avg['mean_reward']:.4f} "
              f"feas={avg['feasible_rate']:.2f} "
              f"temp={temperature:.3f} "
              f"({epoch_dt:.1f}s)")

        # Validation
        if val_data and (epoch + 1) % args.val_every == 0:
            print(f"\n  --- Validation (epoch {epoch+1}) ---")
            model.eval()
            val_results = evaluate(model, val_data, args, device)

            # Average HPWL across val circuits
            val_hpwls = []
            for cname, cres in val_results.items():
                if 'alns_learned' in cres:
                    val_hpwls.append(cres['alns_learned']['best_hpwl'])

            if val_hpwls:
                mean_val = np.mean(val_hpwls)
                print(f"  Val mean HPWL (ALNS+learned): {mean_val:.2f}")
                if mean_val < best_val_hpwl:
                    best_val_hpwl = mean_val
                    ckpt_path = os.path.join(args.save_dir, 'best_model.pt')
                    torch.save(model.state_dict(), ckpt_path)
                    print(f"  Saved best model to {ckpt_path}")

        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            ckpt_path = os.path.join(
                args.save_dir, f'model_epoch{epoch+1}.pt')
            torch.save(model.state_dict(), ckpt_path)

    # Save final model
    final_path = os.path.join(args.save_dir, 'final_model.pt')
    torch.save(model.state_dict(), final_path)
    print(f"\nSaved final model to {final_path}")


def eval_mode(args):
    """Three-way evaluation: pure ALNS / learned-only / ALNS+learned."""
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = None
    if args.checkpoint:
        model = NetSpatialGNN(node_input_dim=10, mode='dual', hidden_dim=64, n_layers=5)
        model.load_state_dict(
            torch.load(args.checkpoint, map_location=device, weights_only=True))
        model.to(device)
        model.eval()
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Loaded model from {args.checkpoint} ({n_params:,} params)")

    # Load circuits
    test_circuits = args.test_circuits.split(',')
    print(f"\nLoading {len(test_circuits)} test circuits...")
    test_data = []
    for c in test_circuits:
        print(f"  {c}...", end=' ', flush=True)
        d = load_and_legalize(c, args.benchmark_base, args.seed)
        print(f"({d['n_components']} macros)")
        test_data.append(d)

    print(f"\n{'='*60}")
    print(f"Evaluation: {args.n_iterations} iterations per circuit")
    print(f"{'='*60}")

    results = evaluate(model, test_data, args, device)

    # Final summary table
    print(f"\n{'='*60}")
    print(f"{'Circuit':10s} {'Pure ALNS':>12s} {'Learned':>12s} {'ALNS+Learn':>12s}")
    print(f"{'-'*60}")
    for cname, cres in results.items():
        pure = f"{cres['pure_alns']['best_hpwl']:.2f}"
        learned = (f"{cres['learned_only']['best_hpwl']:.2f}"
                   if 'learned_only' in cres else 'N/A')
        mixed = (f"{cres['alns_learned']['best_hpwl']:.2f}"
                 if 'alns_learned' in cres else 'N/A')
        print(f"{cname:10s} {pure:>12s} {learned:>12s} {mixed:>12s}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description='RL-Trained Destroy Operator for ALNS Macro Placement')
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
    common.add_argument('--alns_segment_size', type=int, default=25)
    common.add_argument('--alns_rho', type=float, default=0.1)

    # Train mode
    train_parser = subparsers.add_parser('train', parents=[common])
    train_parser.add_argument('--train_circuits', type=str, required=True,
                              help='Comma-separated circuit names')
    train_parser.add_argument('--val_circuits', type=str, default=None,
                              help='Comma-separated validation circuits')
    train_parser.add_argument('--n_epochs', type=int, default=50)
    train_parser.add_argument('--steps_per_circuit', type=int, default=100)
    train_parser.add_argument('--batch_size', type=int, default=16)
    train_parser.add_argument('--lr', type=float, default=1e-4)
    train_parser.add_argument('--temp_start', type=float, default=0.05,
                              help='Initial temperature (low = near-greedy)')
    train_parser.add_argument('--temp_end', type=float, default=0.5,
                              help='Final temperature (anneal up for exploration)')
    train_parser.add_argument('--heuristic_warmup', type=int, default=3,
                              help='Heuristic ALNS steps before each RL transition')
    train_parser.add_argument('--val_every', type=int, default=5)
    train_parser.add_argument('--save_dir', type=str, default='checkpoints_rl')
    train_parser.add_argument('--n_iterations', type=int, default=200,
                              help='Iterations for validation eval')

    # Eval mode
    eval_parser = subparsers.add_parser('eval', parents=[common])
    eval_parser.add_argument('--test_circuits', type=str, required=True,
                             help='Comma-separated circuit names')
    eval_parser.add_argument('--checkpoint', type=str, default=None,
                             help='Path to trained model checkpoint')
    eval_parser.add_argument('--n_iterations', type=int, default=500)

    args = parser.parse_args()

    if args.mode == 'train':
        train_mode(args)
    elif args.mode == 'eval':
        eval_mode(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
