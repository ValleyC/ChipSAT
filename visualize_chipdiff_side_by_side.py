"""Render a side-by-side figure for ChipDiffusion vs our LNS on the same macro graph.

This script is intended for fair apples-to-apples visualization on the
ChipDiffusion macro graph. It loads:
  1. The current local ChipDiffusion macro graph via remove_non_macros()
  2. The saved ChipDiffusion sample{idx}.pkl placement
  3. An LNS NPZ produced on the same macro graph, ideally from:
       bridge_chipdiffusion_lns_eval.py --save_npz ...

It then evaluates all placements with ChipDiffusion's hpwl_fast and legality
metric and saves a multi-panel figure on a shared canvas.
"""

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from cpsat_solver import check_boundary, check_overlap
from fair_comparison import inverse_postprocess
from visualize_bridge_result import plot_placement


def _import_chipdiff_utils(chipdiff_root: Path):
    """Import chipdiffusion.diffusion.utils from a local checkout."""
    diffusion_dir = chipdiff_root / "diffusion"
    if not diffusion_dir.exists():
        raise FileNotFoundError(f"chipdiffusion diffusion dir not found: {diffusion_dir}")

    sys.path.insert(0, str(chipdiff_root))
    sys.path.insert(0, str(diffusion_dir))
    import utils as cd_utils  # noqa: E402

    return cd_utils


def _compute_legality(cd_utils, x_eval, x_ref, cond_macro):
    """Compute ChipDiffusion legality metrics for a macro placement."""
    legality = float(
        cd_utils.check_legality_new(
            x_eval,
            x_ref,
            cond_macro,
            cond_macro.is_ports,
            score=True,
        )
    )
    macro_legality = None
    if hasattr(cond_macro, "is_macros"):
        macro_legality = float(
            cd_utils.check_legality_new(
                x_eval,
                x_ref,
                cond_macro,
                (~cond_macro.is_macros) | cond_macro.is_ports,
                score=True,
            )
        )
    return legality, macro_legality


def _canvas_from_positions(positions_list, sizes):
    """Build a shared plot canvas across multiple placements."""
    x_mins = []
    x_maxs = []
    y_mins = []
    y_maxs = []
    for pos in positions_list:
        x_mins.append((pos[:, 0] - sizes[:, 0] / 2).min())
        x_maxs.append((pos[:, 0] + sizes[:, 0] / 2).max())
        y_mins.append((pos[:, 1] - sizes[:, 1] / 2).min())
        y_maxs.append((pos[:, 1] + sizes[:, 1] / 2).max())

    pad = max(max(x_maxs) - min(x_mins), max(y_maxs) - min(y_mins)) * 0.05
    return (
        min(x_mins) - pad,
        max(x_maxs) + pad,
        min(y_mins) - pad,
        max(y_maxs) + pad,
    )


def _load_lns_positions(npz_path: Path):
    """Load positions from an LNS NPZ file."""
    data = np.load(npz_path)
    if "best_positions" in data:
        return data["best_positions"]
    if "positions" in data:
        return data["positions"]
    raise KeyError(f"No 'best_positions' or 'positions' key in {npz_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize ChipDiffusion and LNS side by side on the same macro graph"
    )
    parser.add_argument("--chipdiff_root", type=str, default="../chipdiffusion")
    parser.add_argument("--dataset", type=str, default="ibm.cluster0.v1")
    parser.add_argument("--idx", type=int, default=0, help="Validation sample index")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--lns_npz",
        type=str,
        default="",
        help="Optional LNS NPZ from bridge_chipdiffusion_lns_eval.py --save_npz",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional output PNG path",
    )
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    script_dir = Path(__file__).resolve().parent
    chipdiff_root = (script_dir / args.chipdiff_root).resolve()
    cd_utils = _import_chipdiff_utils(chipdiff_root)

    train_set, val_set = cd_utils.load_graph_data_with_config(args.dataset)
    x_ref_full, cond_full = val_set[args.idx]

    torch.manual_seed(args.seed)
    x_ref_macro, cond_macro = cd_utils.remove_non_macros(x_ref_full, cond_full)
    ref_positions = x_ref_macro[:, :2].detach().cpu().numpy().astype(np.float32)
    sizes = cond_macro.x[:, :2].detach().cpu().numpy().astype(np.float32)

    ref_hpwl_norm, ref_hpwl_rescaled = cd_utils.hpwl_fast(
        x_ref_macro, cond_macro, normalized_hpwl=False
    )
    ref_legality, ref_macro_legality = _compute_legality(
        cd_utils, x_ref_macro, x_ref_macro, cond_macro
    )
    ref_overlap, ref_pairs = check_overlap(ref_positions, sizes)
    ref_boundary = check_boundary(ref_positions, sizes)

    sample_dir = chipdiff_root / "placements" / "macro-ibm" / "samples"
    sample_file = sample_dir / f"sample{args.idx}.pkl"
    if not sample_file.exists():
        raise FileNotFoundError(f"ChipDiffusion sample not found: {sample_file}")

    with open(sample_file, "rb") as f:
        x_cd_phys = torch.tensor(pickle.load(f), dtype=torch.float32)
    x_cd_norm = inverse_postprocess(x_cd_phys, cond_full)

    torch.manual_seed(args.seed)
    x_cd_macro, _ = cd_utils.remove_non_macros(x_cd_norm, cond_full)
    cd_positions = x_cd_macro[:, :2].detach().cpu().numpy().astype(np.float32)
    cd_hpwl_norm, cd_hpwl_rescaled = cd_utils.hpwl_fast(
        x_cd_macro, cond_macro, normalized_hpwl=False
    )
    cd_legality, cd_macro_legality = _compute_legality(
        cd_utils, x_cd_macro, x_ref_macro, cond_macro
    )
    cd_overlap, cd_pairs = check_overlap(cd_positions, sizes)
    cd_boundary = check_boundary(cd_positions, sizes)

    lns_positions = None
    lns_hpwl_norm = None
    lns_legality = None
    lns_macro_legality = None
    lns_overlap = None
    lns_pairs = None
    lns_boundary = None

    if args.lns_npz:
        npz_path = (script_dir / args.lns_npz).resolve()
        if not npz_path.exists():
            raise FileNotFoundError(f"LNS NPZ not found: {npz_path}")
        lns_positions = _load_lns_positions(npz_path).astype(np.float32)
        x_lns = torch.tensor(lns_positions, dtype=x_ref_macro.dtype)
        lns_hpwl_norm, _ = cd_utils.hpwl_fast(x_lns, cond_macro, normalized_hpwl=False)
        lns_legality, lns_macro_legality = _compute_legality(
            cd_utils, x_lns, x_ref_macro, cond_macro
        )
        lns_overlap, lns_pairs = check_overlap(lns_positions, sizes)
        lns_boundary = check_boundary(lns_positions, sizes)

    positions_list = [ref_positions, cd_positions]
    if lns_positions is not None:
        positions_list.append(lns_positions)
    canvas = _canvas_from_positions(positions_list, sizes)

    n_cols = 3 if lns_positions is not None else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(7 * n_cols, 7))
    if n_cols == 1:
        axes = [axes]

    plot_placement(
        axes[0],
        ref_positions,
        sizes,
        title=(
            f"Reference\n"
            f"HPWL={float(ref_hpwl_norm):.2f}  legality_2={ref_legality:.3f}\n"
            f"overlap={ref_overlap:.4f} ({ref_pairs})  boundary={ref_boundary:.4f}"
        ),
        canvas=canvas,
        cmap="YlOrRd",
    )
    plot_placement(
        axes[1],
        cd_positions,
        sizes,
        title=(
            f"ChipDiffusion\n"
            f"HPWL={float(cd_hpwl_norm):.2f}  legality_2={cd_legality:.3f}\n"
            f"overlap={cd_overlap:.4f} ({cd_pairs})  boundary={cd_boundary:.4f}"
        ),
        canvas=canvas,
        cmap="YlGnBu",
    )

    if lns_positions is not None:
        plot_placement(
            axes[2],
            lns_positions,
            sizes,
            title=(
                f"Our LNS\n"
                f"HPWL={float(lns_hpwl_norm):.2f}  legality_2={lns_legality:.3f}\n"
                f"overlap={lns_overlap:.4f} ({lns_pairs})  boundary={lns_boundary:.4f}"
            ),
            canvas=canvas,
            cmap="YlGn",
        )

    fig.suptitle(
        f"ibm{args.idx + 1:02d} on current local ChipDiffusion macro graph",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()

    out_path = (
        Path(args.output)
        if args.output
        else script_dir / "viz_output" / f"chipdiff_side_by_side_idx{args.idx}.png"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
