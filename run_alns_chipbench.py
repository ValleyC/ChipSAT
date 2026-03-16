"""
Make-or-break experiment: ALNS + CP-SAT refinement of ChiPBench reference placement.

Tests whether iterative CP-SAT local repair improves end-to-end PPA
when starting from the ChiPBench reference macro placement.

Usage:
    # Quick test (50 iterations, no PPA eval)
    python run_alns_chipbench.py --data_dir /path/to/bp_fe --n_iterations 50

    # Full run with ChiPBench PPA evaluation
    python run_alns_chipbench.py --data_dir /path/to/bp_fe --n_iterations 500 \
        --eval --docker_id 8191c3740d44

    # List available circuits
    python run_alns_chipbench.py --list --chipbench_dir /path/to/ChiPBench/dataset/data
"""

import os
import sys
import time
import json
import argparse
import subprocess
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from def_loader import (
    load_chipbench_circuit,
    write_placement_def,
    compute_macro_hpwl,
    denormalize_positions,
    list_chipbench_circuits,
)
from cpsat_solver import (
    compute_net_hpwl, check_overlap, check_boundary,
)
from lns_solver import LNSSolver, compute_rudy_np
from routing_constraints import compute_routing_constraints


def run_chipbench_eval(
    def_path: str,
    config_mk: str,
    evaluate_name: str,
    docker_id: str,
    wsl_distro: str = "Ubuntu-22.04",
) -> dict:
    """Run ChiPBench evaluation via WSL2 + Docker."""
    if docker_id is None:
        print("ERROR: docker_id required for ChiPBench evaluation")
        return {}

    cmd = (
        f'source /ChiPBench/env.sh && cd /ChiPBench && '
        f'python3 benchmarking/benchmarking.py '
        f'--config_setting {config_mk} '
        f'--def_path {def_path} '
        f'--evaluate_name {evaluate_name} '
        f'--mode macro'
    )

    full_cmd = f'wsl -d {wsl_distro} -- bash -c "docker exec {docker_id} bash -c \'{cmd}\'"'
    print(f"  Running ChiPBench evaluation...")
    print(f"  Command: {full_cmd}")

    result = subprocess.run(full_cmd, shell=True, capture_output=True, text=True, timeout=3600)
    if result.returncode != 0:
        print(f"  stdout: {result.stdout[-500:]}")
        print(f"  stderr: {result.stderr[-500:]}")
        print(f"  Evaluation failed (exit code {result.returncode})")

    # Read metrics
    metrics_cmd = (
        f'wsl -d {wsl_distro} -- bash -c "'
        f'docker exec {docker_id} cat /ChiPBench/benchmarking_result/{evaluate_name}/metrics.json'
        f'"'
    )
    metrics_result = subprocess.run(metrics_cmd, shell=True, capture_output=True, text=True)
    if metrics_result.returncode == 0:
        try:
            return json.loads(metrics_result.stdout)
        except json.JSONDecodeError:
            pass
    return {}


def copy_def_to_docker(local_def: str, docker_path: str, docker_id: str,
                       wsl_distro: str = "Ubuntu-22.04"):
    """Copy a local DEF file into the Docker container."""
    # First copy to WSL, then docker cp
    wsl_tmp = "/tmp/chipsat_placed.def"
    # Convert Windows path to WSL path
    local_def_abs = os.path.abspath(local_def)
    drive = local_def_abs[0].lower()
    wsl_path = f"/mnt/{drive}/{local_def_abs[2:].replace(os.sep, '/')}"

    cmd = (
        f'wsl -d {wsl_distro} -- bash -c '
        f'"cp \'{wsl_path}\' {wsl_tmp} && docker cp {wsl_tmp} {docker_id}:{docker_path}"'
    )
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  Copy failed: {result.stderr}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description='ALNS + CP-SAT refinement of ChiPBench reference placement')

    # Data
    parser.add_argument('--data_dir', type=str,
                        help='Path to ChiPBench circuit data directory')
    parser.add_argument('--output_def', type=str, default=None,
                        help='Output DEF path (default: {circuit}_alns.def)')

    # ALNS parameters
    parser.add_argument('--n_iterations', type=int, default=200,
                        help='Number of ALNS iterations')
    parser.add_argument('--subset_size', type=int, default=5,
                        help='Initial subset size (macros per iteration)')
    parser.add_argument('--window_fraction', type=float, default=0.05,
                        help='Initial search window fraction')
    parser.add_argument('--cpsat_time_limit', type=float, default=1.0,
                        help='CP-SAT time limit per subproblem (seconds)')
    parser.add_argument('--congestion_weight', type=float, default=0.1,
                        help='RUDY congestion weight in cost')

    # Routing channel constraints
    parser.add_argument('--min_tracks', type=int, default=10,
                        help='Minimum routing tracks between macros')
    parser.add_argument('--boundary_tracks', type=int, default=20,
                        help='Minimum routing tracks from die boundary')
    parser.add_argument('--track_pitch', type=float, default=0.28,
                        help='Metal track pitch in microns (0.28 for Nangate45)')
    parser.add_argument('--no_routing_constraints', action='store_true',
                        help='Disable routing channel constraints')

    # Evaluation
    parser.add_argument('--eval', action='store_true',
                        help='Run ChiPBench PPA evaluation')
    parser.add_argument('--docker_id', type=str, default='8191c3740d44',
                        help='Docker container ID')
    parser.add_argument('--wsl_distro', type=str, default='Ubuntu-22.04')
    parser.add_argument('--eval_name', type=str, default='chipsat_alns',
                        help='Evaluation name for ChiPBench')

    # Listing
    parser.add_argument('--list', action='store_true',
                        help='List available circuits')
    parser.add_argument('--chipbench_dir', type=str,
                        help='ChiPBench dataset/data directory')

    # Other
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log_every', type=int, default=10)

    args = parser.parse_args()
    np.random.seed(args.seed)

    if args.list:
        if not args.chipbench_dir:
            print("--chipbench_dir required with --list")
            return
        circuits = list_chipbench_circuits(args.chipbench_dir)
        print(f"Available circuits ({len(circuits)}):")
        for c in circuits:
            print(f"  {c}")
        return

    if not args.data_dir:
        parser.print_help()
        return

    # ===== Load circuit =====
    print(f"{'='*60}")
    print(f"ALNS + CP-SAT Refinement of ChiPBench Reference Placement")
    print(f"{'='*60}")

    data = load_chipbench_circuit(args.data_dir, use_reference=True)
    circuit_name = data['circuit_name']

    positions = data['positions']       # (N, 2) centers in [-1, 1]
    sizes = data['node_features']       # (N, 2) normalized sizes
    nets = data['nets']
    edge_index = data['edge_index']
    N = data['n_components']

    print(f"Circuit: {circuit_name}")
    print(f"Macros: {N}")
    for i, (name, typ) in enumerate(zip(data['_macro_names'], data['_macro_types'])):
        w, h = data['_sizes_def'][i]
        print(f"  [{i}] {name} ({typ}) size={w/data['_def_units']:.1f}x{h/data['_def_units']:.1f}um")
    print(f"Nets (macro-only): {len(nets)}")
    print(f"Die area: {data['chip_size']} microns")

    # ===== Reference metrics =====
    ref_hpwl = compute_net_hpwl(positions, sizes, nets)
    ref_overlap, ref_ov_pairs = check_overlap(positions, sizes)
    ref_boundary = check_boundary(positions, sizes)
    ref_rudy = compute_rudy_np(positions, sizes, nets)
    ref_macro_hpwl = compute_macro_hpwl(positions, nets)

    print(f"\nReference placement:")
    print(f"  HPWL (net-level): {ref_hpwl:.4f}")
    print(f"  Macro HPWL:       {ref_macro_hpwl:.4f}")
    print(f"  Overlap: {ref_overlap:.6f} ({ref_ov_pairs} pairs)")
    print(f"  Boundary: {ref_boundary:.6f}")
    print(f"  RUDY: max={ref_rudy['rudy_max']:.4f} "
          f"p95={ref_rudy['rudy_p95']:.4f} overflow={ref_rudy['overflow_sum']:.4f}")

    # Skip legalization if reference is already legal
    if ref_ov_pairs > 0:
        print(f"\n  WARNING: Reference has {ref_ov_pairs} overlapping pairs!")
        print(f"  Running legalization first...")
        from cpsat_solver import legalize
        legal_pos = legalize(positions, sizes, time_limit=60.0, window_fraction=0.3)
        if legal_pos is not None:
            positions = legal_pos
            print(f"  Legalized. HPWL: {ref_hpwl:.4f} -> {compute_net_hpwl(positions, sizes, nets):.4f}")
        else:
            print(f"  Legalization failed — proceeding with reference as-is")
    else:
        print(f"\n  Reference is overlap-free — starting ALNS refinement directly")

    # ===== Adjust subset_size for small circuits =====
    if args.subset_size > N:
        args.subset_size = max(2, N // 2)
        print(f"  Adjusted subset_size to {args.subset_size} (circuit has {N} macros)")

    # ===== Compute routing channel constraints =====
    if not args.no_routing_constraints:
        routing_constraints = compute_routing_constraints(
            data, sizes, nets,
            min_tracks=args.min_tracks,
            boundary_tracks=args.boundary_tracks,
            track_pitch_um=args.track_pitch,
        )
    else:
        routing_constraints = None
        print(f"  Routing constraints: DISABLED")

    # ===== ALNS optimization =====
    print(f"\n--- ALNS Refinement ({args.n_iterations} iterations) ---")
    print(f"  subset_size={args.subset_size}, window_fraction={args.window_fraction}")
    print(f"  cpsat_time_limit={args.cpsat_time_limit}s, congestion_weight={args.congestion_weight}")

    solver = LNSSolver(
        positions=positions,
        sizes=sizes,
        nets=nets,
        edge_index=edge_index,
        congestion_weight=args.congestion_weight,
        subset_size=args.subset_size,
        window_fraction=args.window_fraction,
        cpsat_time_limit=args.cpsat_time_limit,
        plateau_threshold=20,
        adapt_threshold=30,
        min_subset=2,
        min_window=0.03,
        max_window=0.15,
        seed=args.seed,
        routing_constraints=routing_constraints,
    )

    t0 = time.time()
    result = solver.solve(
        n_iterations=args.n_iterations,
        log_every=args.log_every,
    )
    alns_time = time.time() - t0

    best_pos = result['best_positions']
    best_hpwl = result['best_hpwl']
    best_macro_hpwl = compute_macro_hpwl(best_pos, nets)
    best_rudy = compute_rudy_np(best_pos, sizes, nets)

    # ===== Summary =====
    print(f"\n{'='*60}")
    print(f"ALNS Refinement Results")
    print(f"{'='*60}")
    print(f"  HPWL:    {ref_hpwl:.4f} -> {best_hpwl:.4f} ({best_hpwl/ref_hpwl:.3f}x)")
    print(f"  Macro HPWL: {ref_macro_hpwl:.4f} -> {best_macro_hpwl:.4f}")
    print(f"  RUDY:    max={ref_rudy['rudy_max']:.4f}->{best_rudy['rudy_max']:.4f} "
          f"p95={ref_rudy['rudy_p95']:.4f}->{best_rudy['rudy_p95']:.4f} "
          f"overflow={ref_rudy['overflow_sum']:.4f}->{best_rudy['overflow_sum']:.4f}")
    print(f"  Overlap: {check_overlap(best_pos, sizes)[1]} pairs")
    print(f"  Time:    {alns_time:.1f}s ({args.n_iterations} iterations)")
    print(f"  Accepted: {result.get('n_accepted', '?')}/{args.n_iterations}")

    # ===== Write output DEF =====
    output_def = args.output_def or f"{circuit_name}_alns.def"
    print(f"\nWriting output DEF to {output_def}...")
    write_placement_def(data, best_pos, output_def)

    # Show placed positions
    bl = denormalize_positions(best_pos, data['_norm_bbox'], data['_sizes_def'])
    print(f"Placed macro positions (DEF units):")
    for i, name in enumerate(data['_macro_names']):
        print(f"  {name}: ({bl[i,0]}, {bl[i,1]})")

    # ===== Save summary =====
    summary = {
        'circuit': circuit_name,
        'n_macros': N,
        'n_nets': len(nets),
        'ref_hpwl': float(ref_hpwl),
        'best_hpwl': float(best_hpwl),
        'hpwl_ratio': float(best_hpwl / ref_hpwl) if ref_hpwl > 0 else None,
        'ref_macro_hpwl': float(ref_macro_hpwl),
        'best_macro_hpwl': float(best_macro_hpwl),
        'ref_rudy_max': float(ref_rudy['rudy_max']),
        'best_rudy_max': float(best_rudy['rudy_max']),
        'ref_rudy_overflow': float(ref_rudy['overflow_sum']),
        'best_rudy_overflow': float(best_rudy['overflow_sum']),
        'n_iterations': args.n_iterations,
        'alns_time': float(alns_time),
        'n_accepted': result.get('n_accepted'),
        'subset_size': args.subset_size,
        'window_fraction': args.window_fraction,
        'cpsat_time_limit': args.cpsat_time_limit,
        'congestion_weight': args.congestion_weight,
        'routing_constraints': not args.no_routing_constraints,
        'min_tracks': args.min_tracks if not args.no_routing_constraints else None,
        'boundary_tracks': args.boundary_tracks if not args.no_routing_constraints else None,
        'track_pitch': args.track_pitch if not args.no_routing_constraints else None,
        'output_def': output_def,
    }
    summary_path = output_def.replace('.def', '_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_path}")

    # ===== Optional: ChiPBench PPA evaluation =====
    if args.eval:
        print(f"\n--- ChiPBench PPA Evaluation ---")

        # Copy DEF to Docker
        docker_def_path = f"/ChiPBench/{os.path.basename(output_def)}"
        print(f"  Copying {output_def} to Docker:{docker_def_path}")
        if not copy_def_to_docker(output_def, docker_def_path, args.docker_id, args.wsl_distro):
            print("  Failed to copy DEF to Docker")
            return

        # Find config.mk
        config_mk = f"flow/designs/nangate45/{data.get('_design_name', circuit_name + '_top')}/config.mk"

        metrics = run_chipbench_eval(
            def_path=docker_def_path,
            config_mk=config_mk,
            evaluate_name=args.eval_name,
            docker_id=args.docker_id,
            wsl_distro=args.wsl_distro,
        )
        if metrics:
            print(f"\nPPA Metrics:")
            print(json.dumps(metrics, indent=2))
            # Append to summary
            summary['ppa_metrics'] = metrics
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Done.")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
