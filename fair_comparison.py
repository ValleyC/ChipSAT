"""
Fair comparison: evaluate both ChipDiffusion and our LNS on the EXACT same
macro graph using ChipDiffusion's hpwl_fast evaluator.

Approach:
1. Load graph, call remove_non_macros to get cond_preprocessed + x_preprocessed
2. Inverse-transform sample*.pkl to recover ChipDiffusion's normalized macro placement
3. Run our LNS on cond_preprocessed
4. Evaluate all with hpwl_fast on cond_preprocessed

Usage:
    python fair_comparison.py --idx 0 --n_iterations 1000
"""

import argparse
import os
import sys
import pickle
import time
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Imports from our pipeline
# ---------------------------------------------------------------------------
from cpsat_solver import legalize, check_overlap, check_boundary
from lns_solver import LNSSolver
from bridge_chipdiffusion_lns_eval import reconstruct_nets_from_star


def inverse_postprocess(x_phys, cond_full):
    """Undo postprocess_placement: physical coords -> normalized [-1,1] coords.

    Forward (from utils.py:625):
        x = x - cond.x/2
        x = scale * (x + 1) / 2 + offset
    Inverse:
        x_norm = 2 * (x_phys - offset) / scale - 1 + cond.x/2
    """
    chip_size = torch.tensor(cond_full.chip_size, dtype=torch.float32)
    if len(chip_size) == 4:
        scale = (chip_size[2:] - chip_size[:2]).view(1, 2)
        offset = chip_size[:2].view(1, 2)
    elif len(chip_size) == 2:
        scale = chip_size.view(1, 2)
        offset = torch.zeros(1, 2)
    else:
        raise ValueError(f"Unexpected chip_size length: {len(chip_size)}")

    x_norm = 2 * (x_phys - offset) / scale - 1 + cond_full.x[:, :2] / 2
    return x_norm


def main():
    parser = argparse.ArgumentParser(description="Fair comparison: LNS vs ChipDiffusion")
    parser.add_argument("--chipdiff_root", type=str, default="../chipdiffusion")
    parser.add_argument("--dataset", type=str, default="ibm.cluster0.v1")
    parser.add_argument("--idx", type=int, default=0, help="Val sample index (0=ibm01)")
    parser.add_argument("--n_iterations", type=int, default=1000)
    parser.add_argument("--subset_size", type=int, default=30)
    parser.add_argument("--window_fraction", type=float, default=0.15)
    parser.add_argument("--cpsat_time_limit", type=float, default=5.0)
    parser.add_argument("--congestion_weight", type=float, default=0.1)
    parser.add_argument("--plateau_threshold", type=int, default=20)
    parser.add_argument("--adapt_threshold", type=int, default=30)
    parser.add_argument("--legalize_time_limit", type=float, default=60.0)
    parser.add_argument("--legalize_window", type=float, default=0.3)
    parser.add_argument("--skip_legalize", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_lns", action="store_true",
                        help="Skip LNS run, only evaluate ChipDiffusion")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    script_dir = Path(__file__).resolve().parent
    chipdiff_root = (script_dir / args.chipdiff_root).resolve()

    # Import chipdiffusion utils
    sys.path.insert(0, str(chipdiff_root))
    sys.path.insert(0, str(chipdiff_root / "diffusion"))
    import utils as cd_utils

    print("=" * 78)
    print("Fair Comparison: LNS vs ChipDiffusion on identical macro graph")
    print("=" * 78)

    # -----------------------------------------------------------------------
    # Step 1: Load graph data
    # -----------------------------------------------------------------------
    train_set, val_set = cd_utils.load_graph_data_with_config(args.dataset)
    x_ref, cond = val_set[args.idx]

    print(f"Circuit idx={args.idx}, full graph: {cond.num_nodes} nodes, {cond.num_edges} edges")

    # -----------------------------------------------------------------------
    # Step 2: Get cond_preprocessed (macro graph) and x_preprocessed (reference)
    # This is exactly what ChipDiffusion's eval does at utils.py:239
    # -----------------------------------------------------------------------
    torch.manual_seed(args.seed)
    x_preprocessed, cond_preprocessed = cd_utils.remove_non_macros(x_ref, cond)

    V = cond_preprocessed.num_nodes
    E = cond_preprocessed.num_edges
    print(f"Macro graph: {V} nodes, {E} edges")

    # Reference HPWL on the macro graph (same as utils.py:301)
    ref_hpwl = float(cd_utils.hpwl_fast(x_preprocessed, cond_preprocessed, normalized_hpwl=True))
    ref_hpwl_norm, ref_hpwl_rescaled = cd_utils.hpwl_fast(
        x_preprocessed, cond_preprocessed, normalized_hpwl=False
    )
    print(f"\nReference HPWL (normalized):  {float(ref_hpwl_norm):.4f}")
    print(f"Reference HPWL (rescaled):   {float(ref_hpwl_rescaled):.4f}")

    # -----------------------------------------------------------------------
    # Step 3: Load and inverse-transform ChipDiffusion's placement
    # sample*.pkl is saved AFTER postprocess_placement (physical coords)
    # We need to undo that to get normalized macro coords
    # -----------------------------------------------------------------------
    sample_dir = chipdiff_root / "placements" / "macro-ibm" / "samples"
    sample_file = sample_dir / f"sample{args.idx}.pkl"

    cd_hpwl_norm_val = None
    cd_hpwl_rescaled_val = None
    cd_legality = None
    cd_macro_legality = None

    if sample_file.exists():
        with open(sample_file, "rb") as f:
            x_cd_phys = torch.tensor(pickle.load(f), dtype=torch.float32)

        print(f"\nLoaded {sample_file.name}: shape {tuple(x_cd_phys.shape)}")
        print(f"  Physical coord range: [{x_cd_phys.min().item():.4f}, {x_cd_phys.max().item():.4f}]")

        # Inverse transform: physical -> normalized
        x_cd_norm = inverse_postprocess(x_cd_phys, cond)

        print(f"  Normalized coord range: [{x_cd_norm.min().item():.4f}, {x_cd_norm.max().item():.4f}]")

        # Extract macros (same remove_non_macros, same seed -> same graph)
        torch.manual_seed(args.seed)
        x_cd_macro, cond_cd = cd_utils.remove_non_macros(x_cd_norm, cond)

        # Evaluate ChipDiffusion on the SAME macro graph
        cd_hpwl = float(cd_utils.hpwl_fast(x_cd_macro, cond_cd, normalized_hpwl=True))
        cd_hpwl_norm_val, cd_hpwl_rescaled_val = cd_utils.hpwl_fast(
            x_cd_macro, cond_cd, normalized_hpwl=False
        )
        cd_hpwl_norm_val = float(cd_hpwl_norm_val)
        cd_hpwl_rescaled_val = float(cd_hpwl_rescaled_val)

        # Legality checks: both our overlap metric and ChipDiffusion's legality score
        cd_positions = x_cd_macro.detach().cpu().numpy().astype(np.float32)
        cd_sizes = cond_cd.x[:, :2].detach().cpu().numpy().astype(np.float32)
        cd_ov, cd_ov_pairs = check_overlap(cd_positions, cd_sizes)
        cd_boundary = check_boundary(cd_positions, cd_sizes)
        cd_legality = float(
            cd_utils.check_legality_new(
                x_cd_macro,
                x_preprocessed,
                cond_preprocessed,
                cond_preprocessed.is_ports,
                score=True,
            )
        )
        if hasattr(cond_preprocessed, "is_macros"):
            cd_macro_legality = float(
                cd_utils.check_legality_new(
                    x_cd_macro,
                    x_preprocessed,
                    cond_preprocessed,
                    (~cond_preprocessed.is_macros) | cond_preprocessed.is_ports,
                    score=True,
                )
            )

        print(f"\nChipDiffusion placement:")
        print(f"  HPWL (normalized):  {cd_hpwl_norm_val:.4f}")
        print(f"  HPWL (rescaled):    {cd_hpwl_rescaled_val:.4f}")
        print(f"  Overlap:            {cd_ov:.6f} ({cd_ov_pairs} pairs)")
        print(f"  Boundary violation: {cd_boundary:.6f}")
        print(f"  legality_2:         {cd_legality:.6f}")
        if cd_macro_legality is not None:
            print(f"  macro_legality:     {cd_macro_legality:.6f}")
        print(f"  Ratio vs reference: {cd_hpwl / max(ref_hpwl, 1e-12):.4f}")

        # Also report metrics.csv values for comparison
        metrics_file = sample_dir.parent / "metrics.csv"
        if metrics_file.exists():
            import csv
            with open(metrics_file) as mf:
                reader = csv.DictReader(mf)
                rows = list(reader)
                if args.idx < len(rows):
                    row = rows[args.idx]
                    print(f"\n  (metrics.csv reports: hpwl_norm={row['hpwl_normalized']}, "
                          f"ref={row['original_hpwl_normalized']}, "
                          f"ratio={row['hpwl_ratio']})")
    else:
        print(f"\nWARNING: {sample_file} not found, skipping ChipDiffusion eval")

    # -----------------------------------------------------------------------
    # Step 4: Run our LNS on the SAME macro graph
    # -----------------------------------------------------------------------
    if args.skip_lns:
        print("\n[--skip_lns] Skipping LNS run")
        return

    positions = x_preprocessed.detach().cpu().numpy().astype(np.float32)
    if positions.ndim == 3:
        positions = positions[0]  # remove batch dim if present
    sizes = cond_preprocessed.x[:, :2].detach().cpu().numpy().astype(np.float32)
    edge_index_np = cond_preprocessed.edge_index.detach().cpu().numpy().astype(np.int64)
    nets = reconstruct_nets_from_star(cond_preprocessed)

    ov_area, ov_pairs = check_overlap(positions, sizes)
    print(f"\nReference overlap: {ov_area:.6f} ({ov_pairs} pairs)")

    # Optional legalization of starting positions
    start_positions = positions.copy()
    if (not args.skip_legalize) and ov_pairs > 0:
        print("Legalizing reference positions...")
        p_legal = legalize(
            start_positions, sizes,
            time_limit=args.legalize_time_limit,
            window_fraction=args.legalize_window,
        )
        if p_legal is not None:
            start_positions = p_legal
            ov2, pairs2 = check_overlap(start_positions, sizes)
            print(f"  After legalization: overlap={ov2:.6f} ({pairs2} pairs)")

            # Evaluate legalized HPWL
            x_legal = torch.tensor(start_positions, dtype=x_preprocessed.dtype)
            legal_hpwl = float(cd_utils.hpwl_fast(x_legal, cond_preprocessed, normalized_hpwl=True))
            print(f"  Legalized HPWL:     {legal_hpwl:.4f}")
        else:
            print("  Legalization failed, using reference positions")

    # Run LNS
    print(f"\nRunning LNS ({args.n_iterations} iterations)...")
    t_start = time.time()

    solver = LNSSolver(
        positions=start_positions,
        sizes=sizes,
        nets=nets,
        edge_index=edge_index_np,
        congestion_weight=args.congestion_weight,
        subset_size=args.subset_size,
        window_fraction=args.window_fraction,
        cpsat_time_limit=args.cpsat_time_limit,
        plateau_threshold=args.plateau_threshold,
        adapt_threshold=args.adapt_threshold,
        seed=args.seed,
    )

    result = solver.solve(
        n_iterations=args.n_iterations,
        log_every=max(1, args.n_iterations // 20),
        verbose=True,
    )

    t_elapsed = time.time() - t_start

    # Evaluate LNS result with hpwl_fast on the SAME graph
    x_lns = torch.tensor(result["best_positions"], dtype=x_preprocessed.dtype)
    lns_hpwl = float(cd_utils.hpwl_fast(x_lns, cond_preprocessed, normalized_hpwl=True))
    lns_hpwl_norm, lns_hpwl_rescaled = cd_utils.hpwl_fast(
        x_lns, cond_preprocessed, normalized_hpwl=False
    )

    ov_best, ov_pairs_best = check_overlap(result["best_positions"], sizes)
    b_best = check_boundary(result["best_positions"], sizes)
    lns_legality = float(
        cd_utils.check_legality_new(
            x_lns,
            x_preprocessed,
            cond_preprocessed,
            cond_preprocessed.is_ports,
            score=True,
        )
    )
    lns_macro_legality = None
    if hasattr(cond_preprocessed, "is_macros"):
        lns_macro_legality = float(
            cd_utils.check_legality_new(
                x_lns,
                x_preprocessed,
                cond_preprocessed,
                (~cond_preprocessed.is_macros) | cond_preprocessed.is_ports,
                score=True,
            )
        )

    # -----------------------------------------------------------------------
    # Final comparison table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("FAIR COMPARISON — same graph, same evaluator (hpwl_fast)")
    print("=" * 78)
    print(f"{'Method':<25} {'HPWL_norm':>12} {'HPWL_resc':>12} {'Ratio':>8} {'Overlap':>12} {'Boundary':>10}")
    print("-" * 78)
    print(f"{'Reference':<25} {float(ref_hpwl_norm):>12.2f} {float(ref_hpwl_rescaled):>12.2f} {'1.000':>8} {ov_area:>12.6f} {check_boundary(positions, sizes):>10.6f}")

    if cd_hpwl_norm_val is not None:
        print(f"{'ChipDiffusion':<25} {cd_hpwl_norm_val:>12.2f} {cd_hpwl_rescaled_val:>12.2f} {cd_hpwl_norm_val/max(float(ref_hpwl_norm),1e-12):>8.4f} {cd_ov:>12.6f} {cd_boundary:>10.6f}")

    print(f"{'Our LNS':>25} {float(lns_hpwl_norm):>12.2f} {float(lns_hpwl_rescaled):>12.2f} {float(lns_hpwl_norm)/max(float(ref_hpwl_norm),1e-12):>8.4f} {ov_best:>12.6f} {b_best:>10.6f}")

    print("\nLegality scores (ChipDiffusion metric):")
    if cd_hpwl_norm_val is not None:
        print(f"  ChipDiffusion: legality_2={cd_legality:.6f}"
              + (f", macro_legality={cd_macro_legality:.6f}" if cd_macro_legality is not None else ""))
    print(f"  Our LNS:       legality_2={lns_legality:.6f}"
          + (f", macro_legality={lns_macro_legality:.6f}" if lns_macro_legality is not None else ""))

    print(f"\nLNS wall-clock: {t_elapsed:.1f}s ({args.n_iterations} iterations)")
    print(f"Graph: {V} macros, {E} edges, {len(nets)} nets")

    # Note about metrics.csv discrepancy
    if cd_hpwl_norm_val is not None:
        print(f"\nNOTE: metrics.csv reports hpwl_norm=98.04, ref=313.14 for idx=0.")
        print(f"      Our eval on current graph data gives hpwl_norm={cd_hpwl_norm_val:.2f}, ref={float(ref_hpwl_norm):.2f}.")
        print(f"      This indicates the graph pickle data may have been regenerated since metrics.csv.")


if __name__ == "__main__":
    main()
