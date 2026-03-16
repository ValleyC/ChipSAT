"""
End-to-end ChiPBench macro placement + evaluation.

Loads a ChiPBench circuit natively from DEF/LEF, runs CP-SAT macro placement,
writes output DEF, and optionally triggers ChiPBench PPA evaluation.

Usage:
    # Place macros and write DEF (local, no Docker needed)
    python run_chipbench.py --data_dir /path/to/ChiPBench/dataset/data/bp_fe --output_def placed.def

    # Full evaluation via ChiPBench Docker
    python run_chipbench.py --data_dir /path/to/bp_fe --eval --docker_id CONTAINER_ID

    # List available circuits
    python run_chipbench.py --list --chipbench_dir /path/to/ChiPBench/dataset/data
"""

import os
import sys
import time
import json
import argparse
import subprocess
import numpy as np

# Add parent dir if needed
sys.path.insert(0, os.path.dirname(__file__))

from def_loader import (
    load_chipbench_circuit,
    write_placement_def,
    compute_macro_hpwl,
    denormalize_positions,
    list_chipbench_circuits,
)
from cpsat_solver import legalize, SCALE


def place_macros_cpsat(
    data: dict,
    time_limit: float = 120.0,
    window_fraction: float = 1.0,
    num_workers: int = 8,
) -> np.ndarray:
    """
    Run CP-SAT legalization on macro placement.

    For initial placement from scratch (macros at origin), uses full canvas
    with displacement minimization disabled. For refining existing placement,
    uses window constraints.

    Args:
        data: dict from load_chipbench_circuit
        time_limit: CP-SAT time limit in seconds
        window_fraction: movement window as fraction of canvas
        num_workers: parallel workers for CP-SAT

    Returns:
        (V, 2) placed positions in normalized [-1, 1] space
    """
    positions = data['positions'].copy()
    sizes = data['node_features'].copy()
    V = data['n_components']

    # Check if macros are at origin (unplaced)
    all_at_origin = np.all(np.abs(positions + 1.0) < 0.01)  # normalized (0,0) -> -1,-1
    if all_at_origin:
        print("  Macros at origin — generating initial spread placement...")
        # Place macros in a grid pattern to give CP-SAT a feasible start
        cols = int(np.ceil(np.sqrt(V)))
        for i in range(V):
            r, c = divmod(i, cols)
            positions[i, 0] = -0.8 + 1.6 * (c + 0.5) / cols
            positions[i, 1] = -0.8 + 1.6 * (r + 0.5) / max(1, (V + cols - 1) // cols)

    print(f"  Running CP-SAT legalization (V={V}, tl={time_limit}s, wf={window_fraction})...")
    t0 = time.time()
    result = legalize(
        positions, sizes,
        time_limit=time_limit,
        window_fraction=window_fraction,
        num_workers=num_workers,
        minimize_displacement=True,
    )
    elapsed = time.time() - t0

    if result is None:
        print(f"  CP-SAT legalization FAILED after {elapsed:.1f}s")
        print("  Retrying with full canvas...")
        # Retry with full canvas
        result = legalize(
            positions, sizes,
            time_limit=time_limit * 2,
            window_fraction=1.0,
            num_workers=num_workers,
            minimize_displacement=False,
        )
        elapsed = time.time() - t0
        if result is None:
            print(f"  CP-SAT still failed. Returning input positions.")
            return positions

    print(f"  CP-SAT done in {elapsed:.1f}s")
    return result


def place_macros_hpwl(
    data: dict,
    time_limit: float = 300.0,
    num_workers: int = 8,
) -> np.ndarray:
    """
    Run CP-SAT with HPWL minimization on macro placement.

    Uses solve_subset from cpsat_solver.py to minimize HPWL
    with NoOverlap2D constraints.
    """
    from cpsat_solver import solve_subset

    positions = data['positions'].copy()
    sizes = data['node_features'].copy()
    nets = data['nets']
    V = data['n_components']

    # Check if macros are at origin
    all_at_origin = np.all(np.abs(positions + 1.0) < 0.01)
    if all_at_origin:
        print("  Macros at origin — generating initial spread placement...")
        cols = int(np.ceil(np.sqrt(V)))
        for i in range(V):
            r, c = divmod(i, cols)
            positions[i, 0] = -0.8 + 1.6 * (c + 0.5) / cols
            positions[i, 1] = -0.8 + 1.6 * (r + 0.5) / max(1, (V + cols - 1) // cols)

    # First legalize
    print("  Step 1: Legalize initial placement...")
    legal = legalize(
        positions, sizes,
        time_limit=60.0,
        window_fraction=1.0,
        num_workers=num_workers,
        minimize_displacement=True,
    )
    if legal is not None:
        positions = legal
        print(f"    Legalized HPWL: {compute_macro_hpwl(positions, nets):.4f}")
    else:
        print("    Legalization failed, continuing with grid placement")

    # HPWL optimization: all macros movable
    subset = list(range(V))
    print(f"  Step 2: HPWL optimization (V={V}, tl={time_limit}s)...")
    t0 = time.time()
    result = solve_subset(
        positions, sizes, nets,
        subset_indices=subset,
        time_limit=time_limit,
        window_fraction=1.0,
        num_workers=num_workers,
    )
    elapsed = time.time() - t0

    if result is not None:
        print(f"    HPWL after optimization: {compute_macro_hpwl(result, nets):.4f}")
        print(f"    Solve time: {elapsed:.1f}s")
        return result
    else:
        print(f"    HPWL optimization failed after {elapsed:.1f}s, returning legalized")
        return positions


def run_chipbench_eval(
    def_path: str,
    config_mk: str,
    evaluate_name: str = "chipsat",
    docker_id: str = None,
    wsl_distro: str = "Ubuntu-22.04",
) -> dict:
    """
    Run ChiPBench evaluation via Docker.

    Args:
        def_path: path to placed DEF file (inside Docker filesystem)
        config_mk: path to config.mk (inside Docker filesystem)
        evaluate_name: name for this evaluation run
        docker_id: Docker container ID
        wsl_distro: WSL2 distribution name

    Returns:
        dict of PPA metrics
    """
    if docker_id is None:
        print("ERROR: docker_id required for ChiPBench evaluation")
        return {}

    cmd = (
        f'cd /ChiPBench && python benchmarking/benchmarking.py '
        f'--config_setting {config_mk} '
        f'--def_path {def_path} '
        f'--evaluate_name {evaluate_name} '
        f'--mode macro'
    )

    # Run via WSL2 → Docker
    full_cmd = f'wsl -d {wsl_distro} -- bash -c "docker exec {docker_id} bash -c \'{cmd}\'"'
    print(f"  Running ChiPBench evaluation...")
    print(f"  Command: {full_cmd}")

    result = subprocess.run(full_cmd, shell=True, capture_output=True, text=True, timeout=1800)
    print(result.stdout)
    if result.returncode != 0:
        print(f"  Evaluation failed: {result.stderr}")

    # Try to read metrics
    metrics_cmd = (
        f'wsl -d {wsl_distro} -- bash -c '
        f'"docker exec {docker_id} cat /ChiPBench/benchmarking_result/{evaluate_name}/metrics.json"'
    )
    metrics_result = subprocess.run(metrics_cmd, shell=True, capture_output=True, text=True)
    if metrics_result.returncode == 0:
        try:
            return json.loads(metrics_result.stdout)
        except json.JSONDecodeError:
            pass

    return {}


def main():
    parser = argparse.ArgumentParser(description='ChiPBench macro placement + evaluation')
    parser.add_argument('--data_dir', type=str, help='Path to ChiPBench circuit data directory')
    parser.add_argument('--output_def', type=str, default='chipsat_placed.def',
                       help='Output DEF path')
    parser.add_argument('--mode', choices=['legalize', 'hpwl'], default='hpwl',
                       help='Placement mode: legalize (feasibility) or hpwl (optimize wirelength)')
    parser.add_argument('--time_limit', type=float, default=300.0,
                       help='CP-SAT time limit in seconds')
    parser.add_argument('--num_workers', type=int, default=8,
                       help='CP-SAT parallel workers')

    # Evaluation
    parser.add_argument('--eval', action='store_true', help='Run ChiPBench PPA evaluation')
    parser.add_argument('--docker_id', type=str, help='Docker container ID')
    parser.add_argument('--wsl_distro', type=str, default='Ubuntu-22.04')
    parser.add_argument('--config_mk', type=str,
                       help='Path to config.mk inside Docker')
    parser.add_argument('--eval_name', type=str, default='chipsat',
                       help='Evaluation name for ChiPBench')

    # Listing
    parser.add_argument('--list', action='store_true', help='List available circuits')
    parser.add_argument('--chipbench_dir', type=str, help='ChiPBench dataset/data directory')

    args = parser.parse_args()

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

    # Load circuit
    print(f"=" * 60)
    print(f"ChiPBench Macro Placement")
    print(f"=" * 60)

    data = load_chipbench_circuit(args.data_dir, use_reference=True)
    print(f"Circuit: {data['circuit_name']}")
    print(f"Macros: {data['n_components']}")
    for i, (name, typ) in enumerate(zip(data['_macro_names'], data['_macro_types'])):
        w, h = data['_sizes_def'][i]
        print(f"  [{i}] {name} ({typ}) size={w/data['_def_units']:.1f}x{h/data['_def_units']:.1f}um")
    print(f"Nets (macro-only): {len(data['nets'])}")
    print(f"Die area: {data['chip_size']} microns")

    ref_hpwl = compute_macro_hpwl(data['positions'], data['nets'])
    print(f"Reference macro HPWL (normalized): {ref_hpwl:.4f}")

    # Run placement
    print(f"\n--- Placement (mode={args.mode}) ---")
    if args.mode == 'legalize':
        result_pos = place_macros_cpsat(
            data,
            time_limit=args.time_limit,
            num_workers=args.num_workers,
        )
    else:
        result_pos = place_macros_hpwl(
            data,
            time_limit=args.time_limit,
            num_workers=args.num_workers,
        )

    our_hpwl = compute_macro_hpwl(result_pos, data['nets'])
    print(f"\nResult HPWL (normalized): {our_hpwl:.4f}")
    print(f"Ratio to reference: {our_hpwl / ref_hpwl:.3f}x" if ref_hpwl > 0 else "")

    # Write output DEF
    print(f"\nWriting output DEF to {args.output_def}...")
    write_placement_def(data, result_pos, args.output_def)
    print(f"Done.")

    # Show placed positions in microns
    bl = denormalize_positions(result_pos, data['_norm_bbox'], data['_sizes_def'])
    print(f"\nPlaced macro positions (DEF units):")
    for i, name in enumerate(data['_macro_names']):
        print(f"  {name}: ({bl[i,0]}, {bl[i,1]})")

    # Optional: ChiPBench evaluation
    if args.eval:
        print(f"\n--- ChiPBench PPA Evaluation ---")
        metrics = run_chipbench_eval(
            def_path=args.output_def,
            config_mk=args.config_mk or '',
            evaluate_name=args.eval_name,
            docker_id=args.docker_id,
            wsl_distro=args.wsl_distro,
        )
        if metrics:
            print(f"\nPPA Metrics:")
            print(json.dumps(metrics, indent=2))

    # Save summary
    summary = {
        'circuit': data['circuit_name'],
        'n_macros': data['n_components'],
        'n_nets': len(data['nets']),
        'ref_hpwl': float(ref_hpwl),
        'our_hpwl': float(our_hpwl),
        'ratio': float(our_hpwl / ref_hpwl) if ref_hpwl > 0 else None,
        'mode': args.mode,
        'time_limit': args.time_limit,
        'output_def': args.output_def,
    }
    summary_path = args.output_def.replace('.def', '_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == '__main__':
    main()
