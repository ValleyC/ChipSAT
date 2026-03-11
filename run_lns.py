"""
LNS + CP-SAT Chip Placement Runner

Usage:
    # Quick test (ibm01 macros, 50 iterations)
    python run_lns.py --circuit ibm01 --n_iterations 50

    # Full run
    python run_lns.py --circuit ibm01 --n_iterations 500 --subset_size 30

    # Larger search window
    python run_lns.py --circuit ibm01 --n_iterations 500 --window_fraction 0.2
"""

import argparse
import numpy as np
import os
import time

from benchmark_loader import load_bookshelf_circuit
from cpsat_solver import legalize, compute_net_hpwl, check_overlap, check_boundary
from lns_solver import LNSSolver, compute_rudy_np


def main():
    parser = argparse.ArgumentParser(
        description='LNS + CP-SAT Chip Placement Solver')

    # Data
    parser.add_argument('--circuit', type=str, default='ibm01',
                        help='ICCAD04 circuit name')
    parser.add_argument('--benchmark_base', type=str, default='benchmarks',
                        help='Benchmark base directory')
    parser.add_argument('--macros_only', action='store_true', default=True,
                        help='Only place macros')
    parser.add_argument('--max_nodes', type=int, default=None,
                        help='Max nodes (BFS subsample)')

    # LNS parameters
    parser.add_argument('--n_iterations', type=int, default=500,
                        help='Number of LNS iterations')
    parser.add_argument('--subset_size', type=int, default=30,
                        help='Initial subset size per iteration')
    parser.add_argument('--window_fraction', type=float, default=0.15,
                        help='Initial search window fraction')
    parser.add_argument('--cpsat_time_limit', type=float, default=0.3,
                        help='CP-SAT time limit per subproblem (seconds)')
    parser.add_argument('--congestion_weight', type=float, default=0.1,
                        help='Congestion weight in cost')
    parser.add_argument('--plateau_threshold', type=int, default=20,
                        help='Iterations before SA activates')
    parser.add_argument('--adapt_threshold', type=int, default=30,
                        help='Iterations before window/subset grows')

    # ALNS
    parser.add_argument('--alns_segment_size', type=int, default=25,
                        help='ALNS segment length for weight updates')
    parser.add_argument('--alns_rho', type=float, default=0.1,
                        help='ALNS reaction factor (weight blending rate)')
    parser.add_argument('--n_parallel', type=int, default=1,
                        help='Parallel candidates per iteration (1=sequential ALNS)')

    # Legalization
    parser.add_argument('--legalize_time_limit', type=float, default=60.0,
                        help='CP-SAT time limit for initial legalization')
    parser.add_argument('--legalize_window', type=float, default=0.3,
                        help='Window fraction for legalization')
    parser.add_argument('--skip_legalize', action='store_true',
                        help='Skip initial legalization (if already legal)')

    # ML model
    parser.add_argument('--model_checkpoint', type=str, default=None,
                        help='Path to trained NetSpatialGNN checkpoint')

    # Output
    parser.add_argument('--save_dir', type=str, default='checkpoints_lns',
                        help='Save directory')
    parser.add_argument('--log_every', type=int, default=10,
                        help='Log frequency')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()
    np.random.seed(args.seed)

    # Print config
    print(f"\n{'='*70}")
    print(f"LNS + CP-SAT Chip Placement Solver")
    print(f"{'='*70}")
    print(f"Circuit: {args.circuit}")
    print(f"  macros_only={args.macros_only}, max_nodes={args.max_nodes}")
    print(f"\nLNS:")
    print(f"  n_iterations={args.n_iterations}")
    print(f"  subset_size={args.subset_size}, window_fraction={args.window_fraction}")
    print(f"  cpsat_time_limit={args.cpsat_time_limit}s")
    print(f"  plateau_threshold={args.plateau_threshold}, adapt_threshold={args.adapt_threshold}")
    print(f"  alns_segment_size={args.alns_segment_size}, alns_rho={args.alns_rho}")
    print(f"  n_parallel={args.n_parallel}")
    print(f"\nLegalization:")
    print(f"  time_limit={args.legalize_time_limit}s, window={args.legalize_window}")
    print(f"{'='*70}\n")

    # Load benchmark
    circuit_dir = os.path.join(
        args.benchmark_base, "iccad04", "extracted", args.circuit)
    data = load_bookshelf_circuit(
        circuit_dir, args.circuit,
        macros_only=args.macros_only,
        max_nodes=args.max_nodes,
        seed=args.seed,
    )

    positions = data['positions']       # (N, 2) center coords in [-1, 1]
    sizes = data['node_features']       # (N, 2) normalized sizes
    nets = data['nets']                 # list of nets
    edge_index = data['edge_index']     # (2, E) for adjacency
    N = data['n_components']

    print(f"Loaded: {N} macros, {len(nets)} nets, "
          f"{edge_index.shape[1]} edges (star decomposition)")

    # Reference HPWL and RUDY (computed before legalization may overwrite positions)
    ref_hpwl = compute_net_hpwl(positions, sizes, nets)
    ref_overlap, ref_ov_pairs = check_overlap(positions, sizes)
    ref_boundary = check_boundary(positions, sizes)
    ref_rudy = compute_rudy_np(positions, sizes, nets)
    print(f"\nReference placement:")
    print(f"  HPWL (net-level): {ref_hpwl:.4f}")
    print(f"  Overlap: {ref_overlap:.6f} ({ref_ov_pairs} pairs)")
    print(f"  Boundary: {ref_boundary:.6f}")
    print(f"  RUDY: max={ref_rudy['rudy_max']:.4f} "
          f"p95={ref_rudy['rudy_p95']:.4f} overflow={ref_rudy['overflow_sum']:.4f}")

    # Initial legalization
    if not args.skip_legalize and ref_ov_pairs > 0:
        print(f"\n--- Initial Legalization (CP-SAT, {args.legalize_time_limit}s) ---")
        t0 = time.time()
        legal_pos = legalize(
            positions, sizes,
            time_limit=args.legalize_time_limit,
            window_fraction=args.legalize_window,
        )
        dt = time.time() - t0

        if legal_pos is not None:
            legal_hpwl = compute_net_hpwl(legal_pos, sizes, nets)
            legal_overlap, legal_ov_pairs = check_overlap(legal_pos, sizes)
            legal_boundary = check_boundary(legal_pos, sizes)
            print(f"  Legalized in {dt:.1f}s:")
            print(f"    HPWL: {ref_hpwl:.4f} -> {legal_hpwl:.4f}")
            print(f"    Overlap: {ref_overlap:.6f} -> {legal_overlap:.6f} "
                  f"({legal_ov_pairs} pairs)")
            print(f"    Boundary: {ref_boundary:.6f} -> {legal_boundary:.6f}")
            positions = legal_pos
        else:
            print(f"  Legalization FAILED after {dt:.1f}s — using reference as-is")
            print(f"  (LNS will maintain whatever feasibility CP-SAT achieves)")
    elif ref_ov_pairs == 0:
        print(f"\nReference placement is already legal — skipping legalization")

    # Load ML model if checkpoint provided
    model = None
    edge_attr = None
    if args.model_checkpoint:
        import torch
        from net_spatial_gnn import NetSpatialGNN
        model = NetSpatialGNN(node_input_dim=10, mode='dual')
        model.load_state_dict(torch.load(args.model_checkpoint, weights_only=True))
        model.eval()
        edge_attr = data['edge_attr']
        print(f"\nLoaded ML model from {args.model_checkpoint}")
        print(f"  Strategies: random, worst_hpwl, congestion, connected, learned")

    # LNS optimization
    print(f"\n--- LNS Optimization ({args.n_iterations} iterations) ---")
    solver = LNSSolver(
        positions=positions,
        sizes=sizes,
        nets=nets,
        edge_index=edge_index,
        congestion_weight=args.congestion_weight,
        subset_size=args.subset_size,
        window_fraction=args.window_fraction,
        cpsat_time_limit=args.cpsat_time_limit,
        plateau_threshold=args.plateau_threshold,
        adapt_threshold=args.adapt_threshold,
        seed=args.seed,
        alns_segment_size=args.alns_segment_size,
        alns_rho=args.alns_rho,
        n_parallel_candidates=args.n_parallel,
        model=model,
        edge_attr=edge_attr,
    )

    result = solver.solve(
        n_iterations=args.n_iterations,
        log_every=args.log_every,
    )

    # RUDY stats on best placement
    rudy_stats = result.get('rudy_stats') or compute_rudy_np(
        result['best_positions'], sizes, nets)

    # Save best placement
    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_path = os.path.join(args.save_dir, f'{args.circuit}_lns_best.npz')
    np.savez(
        ckpt_path,
        positions=result['best_positions'],
        sizes=sizes,
        hpwl=result['best_hpwl'],
        ref_hpwl=ref_hpwl,
        rudy_max=rudy_stats['rudy_max'],
        rudy_p95=rudy_stats['rudy_p95'],
        rudy_p99=rudy_stats['rudy_p99'],
        overflow_sum=rudy_stats['overflow_sum'],
        circuit=args.circuit,
    )
    print(f"\nSaved best placement to {ckpt_path}")

    # Final summary
    print(f"\n{'='*70}")
    print(f"Final Summary")
    print(f"{'='*70}")
    print(f"  Reference HPWL:  {ref_hpwl:.4f}")
    print(f"  Best LNS HPWL:   {result['best_hpwl']:.4f}")
    ratio = result['best_hpwl'] / max(ref_hpwl, 1e-8)
    print(f"  Ratio:           {ratio:.3f}x reference")
    print(f"  RUDY (reference): max={ref_rudy['rudy_max']:.4f} "
          f"p95={ref_rudy['rudy_p95']:.4f} overflow={ref_rudy['overflow_sum']:.4f}")
    print(f"  RUDY (best LNS):  max={rudy_stats['rudy_max']:.4f} "
          f"p95={rudy_stats['rudy_p95']:.4f} overflow={rudy_stats['overflow_sum']:.4f}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
