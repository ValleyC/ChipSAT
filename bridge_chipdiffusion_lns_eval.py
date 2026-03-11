import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from cpsat_solver import legalize, check_overlap, check_boundary
from lns_solver import LNSSolver, compute_rudy_np


def _import_chipdiff_utils(chipdiff_root: Path):
    """Import chipdiffusion.diffusion.utils from a local checkout."""
    diffusion_dir = chipdiff_root / "diffusion"
    if not diffusion_dir.exists():
        raise FileNotFoundError(f"chipdiffusion diffusion dir not found: {diffusion_dir}")

    # Ensure chipdiffusion root and diffusion dir are importable.
    # utils.py imports modules like `common`, `guidance`, `policies` by name.
    sys.path.insert(0, str(chipdiff_root))
    sys.path.insert(0, str(diffusion_dir))

    import utils as cd_utils  # noqa: E402

    return cd_utils


def reconstruct_nets_from_star(cond, round_decimals: int = 8) -> List[List[Tuple[int, float, float]]]:
    """
    Reconstruct net-level pin groups from star-decomposed directed edges.

    ChipDiffusion's macro graph stores directed star edges (plus reverse copy).
    When edge_pin_id is available (preserved through remove_non_macros), group
    by integer source pin ID for robustness. Otherwise fall back to float-rounded
    (src_node, src_dx, src_dy).

    Returns:
        nets: list of nets, each net = [(node_idx, pin_dx, pin_dy), ...]
    """
    edge_index = cond.edge_index.detach().cpu().numpy()
    edge_attr = cond.edge_attr.detach().cpu().numpy()

    E = edge_attr.shape[0]
    if E == 0:
        return []

    # Convention in this codebase: second half is reverse edges.
    half = E // 2

    has_pin_id = hasattr(cond, 'edge_pin_id') and cond.edge_pin_id is not None
    if has_pin_id:
        pin_ids = cond.edge_pin_id.detach().cpu().numpy()  # (E, 2)

    grouped: Dict = {}

    for e in range(half):
        src = int(edge_index[0, e])
        dst = int(edge_index[1, e])
        sdx, sdy, ddx, ddy = edge_attr[e].tolist()

        if has_pin_id:
            key = int(pin_ids[e, 0])  # source pin ID — integer, no rounding needed
        else:
            key = (src, round(float(sdx), round_decimals), round(float(sdy), round_decimals))

        if key not in grouped:
            grouped[key] = {
                "src": (src, float(sdx), float(sdy)),
                "sinks": {},
            }

        sink_key = (dst, float(ddx), float(ddy))
        grouped[key]["sinks"][sink_key] = True

    nets: List[List[Tuple[int, float, float]]] = []
    for v in grouped.values():
        src_pin = v["src"]
        sink_pins = list(v["sinks"].keys())
        if len(sink_pins) == 0:
            continue
        net = [src_pin] + sink_pins
        nets.append(net)

    return nets


def _pick_split_set(train_set, val_set, split: str, idx: int):
    dataset = train_set if split == "train" else val_set
    if idx < 0 or idx >= len(dataset):
        raise IndexError(f"idx={idx} out of range for split={split}, size={len(dataset)}")
    return dataset[idx]


def main():
    parser = argparse.ArgumentParser(
        description="Bridge: run ChipSAT LNS on ChipDiffusion macro graph and evaluate with hpwl_fast"
    )

    # ChipDiffusion data source
    parser.add_argument("--chipdiff_root", type=str, default="../chipdiffusion",
                        help="Path to chipdiffusion repo root")
    parser.add_argument("--dataset", type=str, default="ibm.cluster0.v1",
                        help="ChipDiffusion dataset name (from cluster.py out_name)")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--idx", type=int, default=0,
                        help="Sample index within selected split")

    # LNS params
    parser.add_argument("--n_iterations", type=int, default=200)
    parser.add_argument("--subset_size", type=int, default=30)
    parser.add_argument("--window_fraction", type=float, default=0.15)
    parser.add_argument("--cpsat_time_limit", type=float, default=5.0)
    parser.add_argument("--congestion_weight", type=float, default=0.1)
    parser.add_argument("--plateau_threshold", type=int, default=20)
    parser.add_argument("--adapt_threshold", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)

    # optional initial legalization
    parser.add_argument("--skip_legalize", action="store_true")
    parser.add_argument("--legalize_time_limit", type=float, default=60.0)
    parser.add_argument("--legalize_window", type=float, default=0.3)

    # output
    parser.add_argument("--save_npz", type=str, default="",
                        help="Optional npz output path")

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    script_dir = Path(__file__).resolve().parent
    chipdiff_root = (script_dir / args.chipdiff_root).resolve()
    cd_utils = _import_chipdiff_utils(chipdiff_root)

    print("=" * 78)
    print("Bridge Eval: ChipSAT LNS on ChipDiffusion Macro Graph")
    print("=" * 78)
    print(f"chipdiff_root: {chipdiff_root}")
    print(f"dataset/split/idx: {args.dataset}/{args.split}/{args.idx}")

    train_set, val_set = cd_utils.load_graph_data_with_config(args.dataset)
    x_ref, cond = _pick_split_set(train_set, val_set, args.split, args.idx)

    # Match chipdiffusion macro-IBM evaluation path: remove non-macros first.
    x_macro, cond_macro = cd_utils.remove_non_macros(x_ref, cond)

    # Ensure shape (V,2)
    if x_macro.dim() != 2 or x_macro.shape[1] < 2:
        raise ValueError(f"Unexpected x_macro shape: {tuple(x_macro.shape)}")

    positions = x_macro[:, :2].detach().cpu().numpy().astype(np.float32)
    sizes = cond_macro.x[:, :2].detach().cpu().numpy().astype(np.float32)
    edge_index_np = cond_macro.edge_index.detach().cpu().numpy().astype(np.int64)
    nets = reconstruct_nets_from_star(cond_macro)

    V = positions.shape[0]
    E = int(cond_macro.num_edges)

    print(f"model_vertices/model_edges: {V}/{E}")
    print(f"reconstructed_nets: {len(nets)}")

    # Reference metrics on the exact same graph/metric as chipdiffusion.
    orig_hpwl_norm = cd_utils.hpwl_fast(x_macro, cond_macro, normalized_hpwl=True)
    orig_hpwl_pair = cd_utils.hpwl_fast(x_macro, cond_macro, normalized_hpwl=False)
    if isinstance(orig_hpwl_pair, tuple):
        orig_hpwl_norm2, orig_hpwl_rescaled = orig_hpwl_pair
    else:
        orig_hpwl_norm2 = float(orig_hpwl_pair)
        orig_hpwl_rescaled = float("nan")

    ov_area, ov_pairs = check_overlap(positions, sizes)
    boundary = check_boundary(positions, sizes)

    print(f"reference hpwl_normalized: {float(orig_hpwl_norm):.6f}")
    print(f"reference hpwl_rescaled: {float(orig_hpwl_rescaled):.6f}")
    print(f"reference overlap: {ov_area:.6f} ({ov_pairs} pairs), boundary: {boundary:.6f}")

    # Optional initial legalization
    start_positions = positions.copy()
    if (not args.skip_legalize) and ov_pairs > 0:
        p_legal = legalize(
            start_positions,
            sizes,
            time_limit=args.legalize_time_limit,
            window_fraction=args.legalize_window,
        )
        if p_legal is not None:
            start_positions = p_legal
            ov2, pairs2 = check_overlap(start_positions, sizes)
            b2 = check_boundary(start_positions, sizes)
            print(f"after legalize overlap: {ov2:.6f} ({pairs2} pairs), boundary: {b2:.6f}")
        else:
            print("legalize failed (no feasible solution in time); continuing with reference")

    # Run LNS
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

    result = solver.solve(n_iterations=args.n_iterations, log_every=max(1, args.n_iterations // 20), verbose=True)

    x_best = torch.tensor(result["best_positions"], dtype=x_macro.dtype, device=x_macro.device)

    hpwl_norm, hpwl_rescaled = cd_utils.hpwl_fast(x_best, cond_macro, normalized_hpwl=False)
    macro_hpwl_norm, macro_hpwl_rescaled = cd_utils.macro_hpwl(x_best, cond_macro, normalized_hpwl=False)

    ratio = float(hpwl_norm) / max(float(orig_hpwl_norm), 1e-12)
    ov_best, ov_pairs_best = check_overlap(result["best_positions"], sizes)
    b_best = check_boundary(result["best_positions"], sizes)

    # RUDY congestion stats
    rudy_ref = compute_rudy_np(positions, sizes, nets)
    rudy_best = result.get('rudy_stats') or compute_rudy_np(
        result["best_positions"], sizes, nets)

    print("\n" + "=" * 78)
    print("Comparable Metrics (ChipDiffusion evaluator on ChipDiffusion graph)")
    print("=" * 78)
    print(f"hpwl_normalized:        {float(hpwl_norm):.6f}")
    print(f"hpwl_rescaled:          {float(hpwl_rescaled):.6f}")
    print(f"macro_hpwl_normalized:  {float(macro_hpwl_norm):.6f}")
    print(f"macro_hpwl_rescaled:    {float(macro_hpwl_rescaled):.6f}")
    print(f"original_hpwl_normalized:{float(orig_hpwl_norm):.6f}")
    print(f"hpwl_ratio:             {ratio:.6f}")
    print(f"overlap/boundary:       {ov_best:.6f} ({ov_pairs_best} pairs) / {b_best:.6f}")
    print(f"model_vertices/edges:   {V}/{E}")
    print(f"RUDY (reference):       max={rudy_ref['rudy_max']:.4f} "
          f"p95={rudy_ref['rudy_p95']:.4f} p99={rudy_ref['rudy_p99']:.4f} "
          f"overflow={rudy_ref['overflow_sum']:.4f}")
    print(f"RUDY (LNS best):        max={rudy_best['rudy_max']:.4f} "
          f"p95={rudy_best['rudy_p95']:.4f} p99={rudy_best['rudy_p99']:.4f} "
          f"overflow={rudy_best['overflow_sum']:.4f}")

    if args.save_npz:
        out_path = Path(args.save_npz)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            out_path,
            best_positions=result["best_positions"],
            sizes=sizes,
            hpwl_normalized=float(hpwl_norm),
            hpwl_rescaled=float(hpwl_rescaled),
            macro_hpwl_normalized=float(macro_hpwl_norm),
            macro_hpwl_rescaled=float(macro_hpwl_rescaled),
            original_hpwl_normalized=float(orig_hpwl_norm),
            hpwl_ratio=ratio,
            model_vertices=V,
            model_edges=E,
            rudy_max=rudy_best['rudy_max'],
            rudy_p95=rudy_best['rudy_p95'],
            rudy_p99=rudy_best['rudy_p99'],
            overflow_sum=rudy_best['overflow_sum'],
        )
        print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
