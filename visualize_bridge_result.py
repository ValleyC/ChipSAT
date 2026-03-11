"""Visualize reference vs LNS-optimized placement from bridge evaluation results."""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection


def plot_placement(ax, positions, sizes, title="", canvas=None, cmap='YlOrRd',
                   alpha=0.7, show_overlap=True):
    """Plot macro placement as colored rectangles."""
    V = positions.shape[0]

    # Auto canvas from data
    if canvas is None:
        half_w = sizes[:, 0] / 2
        half_h = sizes[:, 1] / 2
        x_min = (positions[:, 0] - half_w).min()
        x_max = (positions[:, 0] + half_w).max()
        y_min = (positions[:, 1] - half_h).min()
        y_max = (positions[:, 1] + half_h).max()
        pad = max(x_max - x_min, y_max - y_min) * 0.05
        canvas = (x_min - pad, x_max + pad, y_min - pad, y_max + pad)

    cx_min, cx_max, cy_min, cy_max = canvas

    # Canvas boundary
    canvas_rect = patches.Rectangle(
        (cx_min, cy_min), cx_max - cx_min, cy_max - cy_min,
        linewidth=1.5, edgecolor='black', facecolor='#f5f5f5', zorder=0
    )
    ax.add_patch(canvas_rect)

    # Rectangles
    rects = []
    areas = []
    for i in range(V):
        cx, cy = positions[i]
        w, h = sizes[i]
        rect = patches.Rectangle(
            (cx - w / 2, cy - h / 2), w, h,
            linewidth=0.5, edgecolor='#333333'
        )
        rects.append(rect)
        areas.append(w * h)

    pc = PatchCollection(rects, alpha=alpha, edgecolor='#333333', linewidth=0.5)
    pc.set_array(np.array(areas))
    pc.set_cmap(cmap)
    ax.add_collection(pc)

    # Highlight overlapping pairs
    if show_overlap:
        overlap_pairs = []
        for i in range(V):
            for j in range(i + 1, V):
                dx = abs(positions[i, 0] - positions[j, 0])
                dy = abs(positions[i, 1] - positions[j, 1])
                min_dx = (sizes[i, 0] + sizes[j, 0]) / 2
                min_dy = (sizes[i, 1] + sizes[j, 1]) / 2
                if dx < min_dx and dy < min_dy:
                    overlap_pairs.append((i, j))
        if overlap_pairs:
            for i, j in overlap_pairs[:200]:  # limit for performance
                ax.plot(
                    [positions[i, 0], positions[j, 0]],
                    [positions[i, 1], positions[j, 1]],
                    'r-', linewidth=0.3, alpha=0.4, zorder=2
                )
            title += f"\n({len(overlap_pairs)} overlapping pairs)"

    ax.set_xlim(cx_min, cx_max)
    ax.set_ylim(cy_min, cy_max)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.15)


def main():
    parser = argparse.ArgumentParser(description="Visualize bridge evaluation results")
    parser.add_argument("--npz", type=str, default="results_bridge/ibm01_500.npz")
    parser.add_argument("--chipdiff_root", type=str, default="../chipdiffusion")
    parser.add_argument("--dataset", type=str, default="ibm.cluster0.v1")
    parser.add_argument("--idx", type=int, default=0)
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent

    # Load LNS result
    npz_path = script_dir / args.npz
    if not npz_path.exists():
        print(f"NPZ not found: {npz_path}")
        sys.exit(1)

    data = np.load(npz_path)
    best_positions = data['best_positions']
    sizes = data['sizes']
    hpwl_norm = float(data['hpwl_normalized'])
    orig_hpwl = float(data['original_hpwl_normalized'])
    ratio = float(data['hpwl_ratio'])
    V = int(data['model_vertices'])

    # Load reference positions from chipdiffusion
    chipdiff_root = (script_dir / args.chipdiff_root).resolve()
    ref_positions = None
    try:
        diffusion_dir = chipdiff_root / "diffusion"
        sys.path.insert(0, str(chipdiff_root))
        sys.path.insert(0, str(diffusion_dir))
        import utils as cd_utils
        train_set, val_set = cd_utils.load_graph_data_with_config(args.dataset)
        dataset = val_set
        x_ref, cond = dataset[args.idx]
        x_macro, cond_macro = cd_utils.remove_non_macros(x_ref, cond)
        ref_positions = x_macro[:, :2].detach().cpu().numpy().astype(np.float32)
    except Exception as e:
        print(f"Warning: could not load reference from chipdiffusion: {e}")
        print("Will show LNS result only.")

    # Shared canvas
    all_pos = np.concatenate([best_positions] + ([ref_positions] if ref_positions is not None else []))
    half_w = np.tile(sizes[:, 0:1] / 2, (len(all_pos) // V + 1, 1))[:len(all_pos)]
    half_h = np.tile(sizes[:, 1:2] / 2, (len(all_pos) // V + 1, 1))[:len(all_pos)]

    # Use simple approach: compute canvas from both placements
    all_positions_list = [best_positions]
    if ref_positions is not None:
        all_positions_list.append(ref_positions)

    x_mins, x_maxs, y_mins, y_maxs = [], [], [], []
    for pos in all_positions_list:
        x_mins.append((pos[:, 0] - sizes[:, 0] / 2).min())
        x_maxs.append((pos[:, 0] + sizes[:, 0] / 2).max())
        y_mins.append((pos[:, 1] - sizes[:, 1] / 2).min())
        y_maxs.append((pos[:, 1] + sizes[:, 1] / 2).max())

    pad = max(max(x_maxs) - min(x_mins), max(y_maxs) - min(y_mins)) * 0.05
    canvas = (min(x_mins) - pad, max(x_maxs) + pad, min(y_mins) - pad, max(y_maxs) + pad)

    # Plot
    n_cols = 3 if ref_positions is not None else 1
    fig, axes = plt.subplots(1, n_cols, figsize=(7 * n_cols, 7))
    if n_cols == 1:
        axes = [axes]

    circuit_name = f"ibm01 (idx={args.idx})"
    fig.suptitle(f"{circuit_name} — {V} macros | LNS/GT HPWL ratio: {ratio:.3f}", fontsize=13, fontweight='bold')

    if ref_positions is not None:
        plot_placement(axes[0], ref_positions, sizes,
                       title=f"Ground Truth (HPWL={orig_hpwl:.1f})",
                       canvas=canvas, cmap='YlOrRd')
        plot_placement(axes[1], best_positions, sizes,
                       title=f"Our LNS (500 iter, HPWL={hpwl_norm:.1f})",
                       canvas=canvas, cmap='YlGn')

        # Displacement arrows
        ax_d = axes[2]
        plot_placement(ax_d, best_positions, sizes,
                       title="Displacement (GT → LNS)",
                       canvas=canvas, cmap='YlGn', alpha=0.3, show_overlap=False)
        # Draw arrows from ground truth to LNS
        for i in range(V):
            dx = best_positions[i, 0] - ref_positions[i, 0]
            dy = best_positions[i, 1] - ref_positions[i, 1]
            dist = np.sqrt(dx**2 + dy**2)
            if dist > 0.001:  # only show meaningful moves
                ax_d.annotate(
                    '', xy=best_positions[i], xytext=ref_positions[i],
                    arrowprops=dict(arrowstyle='->', color='red', lw=0.8, alpha=0.6),
                    zorder=5
                )
        # GT positions as small dots
        ax_d.scatter(ref_positions[:, 0], ref_positions[:, 1],
                     s=3, c='blue', alpha=0.5, zorder=6, label='Ground Truth')
        ax_d.legend(fontsize=8, loc='upper right')
    else:
        plot_placement(axes[0], best_positions, sizes,
                       title=f"Our LNS (500 iter, HPWL={hpwl_norm:.1f})",
                       canvas=canvas, cmap='YlGn')

    plt.tight_layout()

    out_path = args.output or str(script_dir / "viz_output" / f"bridge_idx{args.idx}_500.png")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
