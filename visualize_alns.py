"""
Visualize ALNS + CP-SAT chip placement process.

Modes:
    static      — 3-panel: legalized placement | pure step | GNN trust step
    video       — animated GIF showing ALNS iterations (pure vs trust_only)
    trust       — side-by-side search windows: pure | degree heuristic | GNN
    convergence — HPWL convergence curves across conditions

Usage:
    python visualize_alns.py --circuit ibm01 --mode static
    python visualize_alns.py --circuit ibm01 --mode video --n_iters 100
    python visualize_alns.py --circuit ibm01 --mode trust
    python visualize_alns.py --circuit ibm01 --mode convergence --n_iters 200
"""

import argparse
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.cm as cm
import torch

sys.path.insert(0, os.path.dirname(__file__))
from benchmark_loader import load_bookshelf_circuit
from cpsat_solver import (
    legalize, solve_subset_guided, compute_net_hpwl, check_overlap,
)
from lns_solver import LNSSolver
from train_local_reviser import (
    extract_local_instance, _gnn_inference, build_model,
    BEST_SS, BEST_WF, BEST_TL,
)

# ────────────────────────────── helpers ───────────────────────────────────────

BG_DARK  = '#1a1a2e'
BG_PANEL = '#16213e'
C_SUBSET = '#e94560'   # red   — selected subset macros
C_ANCHOR = '#0f3460'   # navy  — non-subset macros
C_NET    = '#a8dadc'   # teal  — net lines
C_ARROW  = '#ffd700'   # gold  — movement arrows
C_PURE   = '#a8dadc'   # teal  — pure HPWL curve
C_ML     = '#e94560'   # red   — ML HPWL curve


def _load_circuit(args):
    circuit_dir = os.path.join(
        args.benchmark_base, 'iccad04', 'extracted', args.circuit)
    data = load_bookshelf_circuit(
        circuit_dir, args.circuit, macros_only=True, seed=args.seed)
    positions = data['positions']
    sizes     = data['node_features']
    nets      = data['nets']
    edge_index = data['edge_index']
    _, ov = check_overlap(positions, sizes)
    if ov > 0:
        print(f"Legalizing ({ov} overlapping pairs)...")
        positions = legalize(positions, sizes, time_limit=300.0,
                             window_fraction=0.3, num_workers=4)
        if positions is None:
            raise RuntimeError("Legalization failed")
    return positions, sizes, nets, edge_index, data


def _load_model(args, device):
    model = build_model(args, device)
    ckpt = os.path.join(args.save_dir, 'local_reviser_best.pt')
    if os.path.exists(ckpt):
        model.load_state_dict(
            torch.load(ckpt, map_location=device, weights_only=False))
        model.eval()
        print(f"Loaded model from {ckpt}")
        return model
    print(f"No checkpoint at {ckpt}")
    return None


def _macro_degree(positions, nets):
    deg = np.zeros(len(positions), dtype=np.float32)
    for net in nets:
        for nidx, _, _ in net:
            if nidx < len(positions):
                deg[nidx] += 1
    return deg


def _degree_trust(subset, positions, nets, window_fraction):
    deg = _macro_degree(positions, nets)
    max_deg = max(float(deg.max()), 1.0)
    per_macro = np.full(len(positions), window_fraction, dtype=np.float64)
    for gidx in subset:
        trust = float(np.clip(1.0 - deg[gidx] / max_deg, 0.1, 1.0))
        per_macro[gidx] = trust * window_fraction
    return per_macro


def _gnn_trust(model, inst, device, subset, positions, window_fraction):
    _, trust_local = _gnn_inference(model, inst, device)
    per_macro = np.full(len(positions), window_fraction, dtype=np.float64)
    for k, gidx in enumerate(subset):
        per_macro[gidx] = float(trust_local[k]) * window_fraction
    return per_macro


# ──────────────────────── drawing primitives ──────────────────────────────────

def draw_macros(ax, positions, sizes, subset_set, alpha=0.85, lw=0.4):
    for i, (pos, sz) in enumerate(zip(positions, sizes)):
        x = pos[0] - sz[0] / 2
        y = pos[1] - sz[1] / 2
        fc = C_SUBSET if i in subset_set else C_ANCHOR
        rect = mpatches.Rectangle(
            (x, y), sz[0], sz[1],
            linewidth=lw, edgecolor='white', facecolor=fc, alpha=alpha, zorder=2)
        ax.add_patch(rect)


def draw_nets(ax, positions, nets, top_k=40, alpha=0.12):
    spans = []
    for net in nets:
        idxs = [n for n, _, _ in net if n < len(positions)]
        if len(idxs) < 2:
            continue
        pts = positions[idxs]
        span = float(pts[:, 0].ptp() + pts[:, 1].ptp())
        spans.append((span, idxs))
    spans.sort(key=lambda x: -x[0])
    for _, idxs in spans[:top_k]:
        cx = positions[idxs, 0].mean()
        cy = positions[idxs, 1].mean()
        for idx in idxs:
            ax.plot([positions[idx, 0], cx], [positions[idx, 1], cy],
                    color=C_NET, alpha=alpha, lw=0.5, zorder=1)


def draw_windows(ax, positions, subset, per_macro, window_fraction, cmap_name='RdYlGn'):
    norm = Normalize(vmin=0.05, vmax=1.0)
    cmap = cm.get_cmap(cmap_name)
    for gidx in subset:
        wf = per_macro[gidx] if per_macro is not None else window_fraction
        trust_frac = wf / window_fraction
        color = cmap(norm(trust_frac))
        cx, cy = float(positions[gidx, 0]), float(positions[gidx, 1])
        rect = mpatches.Rectangle(
            (cx - wf, cy - wf), 2 * wf, 2 * wf,
            linewidth=1.5, edgecolor=color, facecolor=color,
            alpha=0.25, zorder=3, linestyle='--')
        ax.add_patch(rect)


def draw_arrows(ax, pre_pos, post_pos, subset):
    for gidx in subset:
        dx = post_pos[gidx, 0] - pre_pos[gidx, 0]
        dy = post_pos[gidx, 1] - pre_pos[gidx, 1]
        if abs(dx) + abs(dy) > 1e-6:
            ax.annotate('',
                xy=(post_pos[gidx, 0], post_pos[gidx, 1]),
                xytext=(pre_pos[gidx, 0], pre_pos[gidx, 1]),
                arrowprops=dict(arrowstyle='->', color=C_ARROW, lw=1.8), zorder=5)


def _setup_ax(ax, positions, sizes, title='', margin=0.04):
    all_x, all_y = positions[:, 0], positions[:, 1]
    mx = all_x.ptp() * margin
    my = all_y.ptp() * margin
    ax.set_xlim(all_x.min() - mx, all_x.max() + mx)
    ax.set_ylim(all_y.min() - my, all_y.max() + my)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_facecolor(BG_PANEL)
    if title:
        ax.set_title(title, color='white', fontsize=11, pad=6)


# ────────────────────────── collect traces ────────────────────────────────────

def collect_trace(positions_init, sizes, nets, edge_index, model, device,
                  n_iters=100, condition='pure', seed=42):
    """
    Run ALNS for n_iters. Returns:
        hpwl_curve  — HPWL after every iteration (length n_iters)
        key_frames  — list of dicts for improving steps, each with:
                      {iter, pre_pos, post_pos, subset, per_macro, hpwl}
    """
    wf = BEST_WF / 2.0 if condition == 'uniform_shrink' else BEST_WF
    solver = LNSSolver(
        positions=positions_init.copy(), sizes=sizes, nets=nets,
        edge_index=edge_index, congestion_weight=0.0,
        subset_size=BEST_SS, window_fraction=wf,
        cpsat_time_limit=BEST_TL, plateau_threshold=20,
        adapt_threshold=30, seed=seed,
    )
    deg = _macro_degree(positions_init, nets)
    max_deg = max(float(deg.max()), 1.0)

    hpwl_curve = []
    key_frames  = []

    for it in range(n_iters):
        strategy = solver.select_strategy()
        subset   = solver.get_neighborhood(strategy, solver.subset_size)
        solver.strategy_attempts[strategy] += 1
        pre_pos  = solver.current_pos.copy()

        per_macro = None
        if condition == 'trust_only' and model is not None:
            inst = extract_local_instance(
                pre_pos, pre_pos, subset, sizes, nets, BEST_WF)
            per_macro = _gnn_trust(model, inst, device, subset, pre_pos, wf)
        elif condition == 'degree_trust':
            per_macro = _degree_trust(subset, pre_pos, nets, wf)

        res = solve_subset_guided(
            solver.current_pos, sizes, nets, subset,
            time_limit=BEST_TL, window_fraction=wf,
            per_macro_windows=per_macro,
        )
        result = solver._apply_candidate_result(
            res['new_positions'], subset, strategy, pre_pos)

        hpwl_curve.append(float(solver.best_hpwl))
        if result['improved']:
            key_frames.append({
                'iter':      it,
                'pre_pos':   pre_pos,
                'post_pos':  solver.current_pos.copy(),
                'subset':    subset,
                'per_macro': per_macro,
                'hpwl':      float(solver.best_hpwl),
            })

    return hpwl_curve, key_frames


# ══════════════════════════ MODE: static ══════════════════════════════════════

def mode_static(args):
    """3 panels: legalized | one pure step | one GNN trust step."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    positions, sizes, nets, edge_index, _ = _load_circuit(args)
    model = _load_model(args, device)
    ref_hpwl = compute_net_hpwl(positions, sizes, nets)

    # Warm up 10 iterations to get past trivially-easy positions
    solver = LNSSolver(
        positions=positions.copy(), sizes=sizes, nets=nets,
        edge_index=edge_index, congestion_weight=0.0,
        subset_size=BEST_SS, window_fraction=BEST_WF,
        cpsat_time_limit=BEST_TL, plateau_threshold=20,
        adapt_threshold=30, seed=args.seed,
    )
    np.random.seed(args.seed)
    for _ in range(10):
        s = solver.select_strategy()
        sub = solver.get_neighborhood(s, solver.subset_size)
        solver.strategy_attempts[s] += 1
        pp = solver.current_pos.copy()
        r = solve_subset_guided(solver.current_pos, sizes, nets, sub,
                                time_limit=BEST_TL, window_fraction=BEST_WF)
        solver._apply_candidate_result(r['new_positions'], sub, s, pp)

    # Grab one step
    strategy = solver.select_strategy()
    subset   = solver.get_neighborhood(strategy, solver.subset_size)
    cur_pos  = solver.current_pos.copy()
    subset_set = set(int(i) for i in subset)

    # GNN trust predictions
    per_macro_gnn = None
    if model is not None:
        inst = extract_local_instance(cur_pos, cur_pos, subset, sizes, nets, BEST_WF)
        per_macro_gnn = _gnn_trust(model, inst, device, subset, cur_pos, BEST_WF)

    # ── figure ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor(BG_DARK)

    panels = [
        ('Legalized Placement\n(starting point)', None, True),
        (f'Pure ALNS step\n(uniform window = {BEST_WF})', None, False),
        (f'GNN Trust-Only step\n(adaptive per-macro window)', per_macro_gnn, False),
    ]

    for ax, (title, per_macro, is_legal) in zip(axes, panels):
        draw_macros(ax, cur_pos, sizes, set() if is_legal else subset_set)
        draw_nets(ax, cur_pos, nets, top_k=50)
        if not is_legal:
            draw_windows(ax, cur_pos, subset,
                         per_macro if per_macro is not None else None,
                         BEST_WF)
        _setup_ax(ax, cur_pos, sizes, title)

    # Colorbar for trust radius
    norm = Normalize(vmin=0.05, vmax=1.0)
    sm = ScalarMappable(cmap=cm.get_cmap('RdYlGn'), norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[2], orientation='vertical',
                         fraction=0.046, pad=0.04)
    cbar.set_label('Trust radius fraction', color='white', fontsize=9)
    cbar.ax.tick_params(colors='white')

    legend_elems = [
        mpatches.Patch(facecolor=C_ANCHOR, label='Non-subset macro'),
        mpatches.Patch(facecolor=C_SUBSET, label=f'Subset ({BEST_SS} macros)'),
        mpatches.Patch(facecolor='green',  alpha=0.3, linestyle='--',
                       edgecolor='green',  label='Large trust window'),
        mpatches.Patch(facecolor='red',    alpha=0.3, linestyle='--',
                       edgecolor='red',    label='Small trust window'),
    ]
    fig.legend(handles=legend_elems, loc='lower center', ncol=4,
               facecolor=BG_DARK, labelcolor='white', fontsize=9,
               bbox_to_anchor=(0.5, -0.04))

    plt.suptitle(
        f'{args.circuit}  |  {len(cur_pos)} macros  |  '
        f'ref HPWL={ref_hpwl:.1f}  |  current HPWL={solver.best_hpwl:.1f}',
        color='white', fontsize=12, y=1.01)
    plt.tight_layout()

    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, f'{args.circuit}_static.png')
    plt.savefig(out, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f"Saved → {out}")
    plt.close()


# ══════════════════════════ MODE: video ═══════════════════════════════════════

def mode_video(args):
    """Animated GIF: left = chip evolving, right = HPWL curves (pure vs GNN)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    positions, sizes, nets, edge_index, _ = _load_circuit(args)
    model  = _load_model(args, device)
    ref_hpwl = compute_net_hpwl(positions, sizes, nets)

    print(f"Collecting {args.n_iters} iters — trust_only ...")
    hpwl_ml, frames_ml = collect_trace(
        positions, sizes, nets, edge_index, model, device,
        n_iters=args.n_iters, condition='trust_only', seed=args.seed)

    print(f"Collecting {args.n_iters} iters — pure ...")
    hpwl_pure, _ = collect_trace(
        positions, sizes, nets, edge_index, None, device,
        n_iters=args.n_iters, condition='pure', seed=args.seed)

    print(f"  pure={hpwl_pure[-1]:.4f}  ml={hpwl_ml[-1]:.4f}  "
          f"({len(frames_ml)} improving frames)")

    if not frames_ml:
        print("No improving frames — nothing to animate.")
        return

    # ── figure layout ──
    fig = plt.figure(figsize=(14, 7), facecolor=BG_DARK)
    ax_chip = fig.add_axes([0.02, 0.12, 0.54, 0.80])
    ax_hpwl = fig.add_axes([0.62, 0.55, 0.35, 0.38])
    ax_info = fig.add_axes([0.62, 0.12, 0.35, 0.36])

    for ax in (ax_chip, ax_hpwl, ax_info):
        ax.set_facecolor(BG_PANEL)

    def _style_ax(ax):
        ax.tick_params(colors='white', labelsize=8)
        for s in ('bottom', 'left'):
            ax.spines[s].set_color('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Thin out frames to ≤60 for manageable GIF
    n_frames = min(len(frames_ml), 60)
    stride   = max(1, len(frames_ml) // n_frames)
    keyframe_idxs = list(range(0, len(frames_ml), stride))

    def animate(fi):
        f = frames_ml[keyframe_idxs[fi]]
        ax_chip.cla(); ax_hpwl.cla(); ax_info.cla()
        for ax in (ax_chip, ax_hpwl, ax_info):
            ax.set_facecolor(BG_PANEL)

        ss = set(int(i) for i in f['subset'])
        draw_macros(ax_chip, f['post_pos'], sizes, ss)
        draw_nets(ax_chip, f['post_pos'], nets, top_k=40)
        if f['per_macro'] is not None:
            draw_windows(ax_chip, f['pre_pos'], f['subset'],
                         f['per_macro'], BEST_WF)
        draw_arrows(ax_chip, f['pre_pos'], f['post_pos'], f['subset'])
        _setup_ax(ax_chip, f['post_pos'], sizes,
                  f"Iter {f['iter']+1}  |  HPWL {f['hpwl']:.2f}  "
                  f"({f['hpwl']/ref_hpwl:.3f}× ref)")

        # HPWL curve
        it = f['iter'] + 1
        ax_hpwl.plot(range(1, it + 1), hpwl_pure[:it],
                     color=C_PURE, lw=1.5, label='Pure')
        ax_hpwl.plot(range(1, it + 1), hpwl_ml[:it],
                     color=C_ML,   lw=1.5, label='GNN trust')
        ax_hpwl.axhline(ref_hpwl, color='white', lw=1, ls=':', alpha=0.5)
        ax_hpwl.set_xlim(0, args.n_iters)
        lo = min(min(hpwl_ml), min(hpwl_pure)) * 0.93
        ax_hpwl.set_ylim(lo, ref_hpwl * 1.08)
        ax_hpwl.set_title('HPWL', color='white', fontsize=9)
        ax_hpwl.set_xlabel('Iteration', color='white', fontsize=8)
        ax_hpwl.legend(fontsize=8, facecolor=BG_DARK, labelcolor='white',
                       framealpha=0.8)
        _style_ax(ax_hpwl)

        # Info box
        ax_info.axis('off')
        lines = [
            f"Circuit:  {args.circuit}",
            f"Macros:   {len(f['post_pos'])}",
            f"Subset:   {BEST_SS}  (window {BEST_WF})",
            f"Ref HPWL: {ref_hpwl:.2f}",
            "",
            f"Pure final:  {hpwl_pure[-1]:.2f}  "
            f"({hpwl_pure[-1]/ref_hpwl:.3f}×)",
            f"GNN final:   {hpwl_ml[-1]:.2f}  "
            f"({hpwl_ml[-1]/ref_hpwl:.3f}×)",
        ]
        ax_info.text(0.05, 0.95, '\n'.join(lines),
                     transform=ax_info.transAxes,
                     color='white', fontsize=9, va='top', fontfamily='monospace')
        ax_info.set_title('Stats', color='white', fontsize=9)

        return []

    ani = animation.FuncAnimation(
        fig, animate, frames=len(keyframe_idxs), interval=200, blit=False)

    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, f'{args.circuit}_alns.gif')
    print(f"Saving {len(keyframe_idxs)}-frame GIF → {out} ...")
    ani.save(out, writer='pillow', fps=args.fps, dpi=90)
    print(f"Saved → {out}")
    plt.close()


# ══════════════════════════ MODE: trust ═══════════════════════════════════════

def mode_trust(args):
    """3 panels: pure uniform | degree heuristic | GNN — for one ALNS step."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    positions, sizes, nets, edge_index, _ = _load_circuit(args)
    model = _load_model(args, device)

    # Warm-up 20 iters
    solver = LNSSolver(
        positions=positions.copy(), sizes=sizes, nets=nets,
        edge_index=edge_index, congestion_weight=0.0,
        subset_size=BEST_SS, window_fraction=BEST_WF,
        cpsat_time_limit=BEST_TL, plateau_threshold=20,
        adapt_threshold=30, seed=args.seed,
    )
    np.random.seed(args.seed)
    for _ in range(20):
        s  = solver.select_strategy()
        sub = solver.get_neighborhood(s, solver.subset_size)
        solver.strategy_attempts[s] += 1
        pp = solver.current_pos.copy()
        r  = solve_subset_guided(solver.current_pos, sizes, nets, sub,
                                 time_limit=BEST_TL, window_fraction=BEST_WF)
        solver._apply_candidate_result(r['new_positions'], sub, s, pp)

    strategy = solver.select_strategy()
    subset   = solver.get_neighborhood(strategy, solver.subset_size)
    cur_pos  = solver.current_pos.copy()

    # Per-condition trust windows
    per_deg = _degree_trust(subset, cur_pos, nets, BEST_WF)
    per_gnn = None
    if model is not None:
        inst    = extract_local_instance(cur_pos, cur_pos, subset, sizes, nets, BEST_WF)
        per_gnn = _gnn_trust(model, inst, device, subset, cur_pos, BEST_WF)

    # Degree info for annotation
    deg = _macro_degree(cur_pos, nets)
    max_deg = max(float(deg.max()), 1.0)

    norm = Normalize(vmin=0.05, vmax=1.0)
    cmap = cm.get_cmap('RdYlGn')

    fig, axes = plt.subplots(1, 3, figsize=(19, 7))
    fig.patch.set_facecolor(BG_DARK)

    configs = [
        ('Pure\n(uniform window)', None),
        ('Degree heuristic\n(trust = 1 − degree_norm)', per_deg),
        ('GNN Trust-Only\n(learned per-macro)', per_gnn),
    ]

    subset_set = set(int(i) for i in subset)
    for ax, (title, per_macro) in zip(axes, configs):
        draw_macros(ax, cur_pos, sizes, subset_set)
        draw_nets(ax, cur_pos, nets, top_k=40)
        # Always draw windows; None → uniform
        pmw = per_macro if per_macro is not None else np.full(
            len(cur_pos), BEST_WF, dtype=np.float64)
        draw_windows(ax, cur_pos, subset, pmw, BEST_WF)

        # Annotate each subset macro with trust fraction + degree
        for gidx in subset:
            wf = float(pmw[gidx])
            trust = wf / BEST_WF
            d = int(deg[gidx])
            color = cmap(norm(trust))
            ax.text(cur_pos[gidx, 0], cur_pos[gidx, 1],
                    f'{trust:.2f}\n(d={d})',
                    fontsize=4.5, ha='center', va='center',
                    color='white', zorder=6, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.1', facecolor=color,
                              alpha=0.6, edgecolor='none'))

        _setup_ax(ax, cur_pos, sizes, title)

    # Colorbar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation='horizontal',
                         fraction=0.025, pad=0.06, aspect=50)
    cbar.set_label('Trust radius  (green=large window, red=tight window)',
                   color='white', fontsize=10)
    cbar.ax.tick_params(colors='white')

    plt.suptitle(
        f'Search window comparison — {args.circuit}  |  '
        f'subset={BEST_SS}  |  base window_fraction={BEST_WF}',
        color='white', fontsize=13)

    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, f'{args.circuit}_trust.png')
    plt.savefig(out, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f"Saved → {out}")
    plt.close()


# ══════════════════════════ MODE: convergence ═════════════════════════════════

def mode_convergence(args):
    """HPWL convergence curves for pure / uniform_shrink / degree_trust / trust_only."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    positions, sizes, nets, edge_index, _ = _load_circuit(args)
    model     = _load_model(args, device)
    ref_hpwl  = compute_net_hpwl(positions, sizes, nets)

    runs = [
        ('pure',          None,  C_PURE,    '-',  'Pure (wf=0.10)'),
        ('uniform_shrink',None,  '#457b9d', '--', 'Uniform shrink (wf=0.05)'),
        ('degree_trust',  None,  '#f4a261', '-.', 'Degree heuristic'),
        ('trust_only',    model, C_ML,      '-',  'GNN trust-only'),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(BG_DARK)
    for ax in (ax1, ax2):
        ax.set_facecolor(BG_PANEL)
        ax.tick_params(colors='white')
        for s in ('bottom', 'left'):
            ax.spines[s].set_color('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    for cond, mdl, color, ls, label in runs:
        print(f"  {cond} ...")
        hpwl_curve, _ = collect_trace(
            positions, sizes, nets, edge_index, mdl, device,
            n_iters=args.n_iters, condition=cond, seed=args.seed)
        iters = list(range(1, len(hpwl_curve) + 1))
        ax1.plot(iters, hpwl_curve,
                 color=color, lw=1.8, ls=ls, label=label)
        ax2.plot(iters, [h / ref_hpwl for h in hpwl_curve],
                 color=color, lw=1.8, ls=ls, label=label)

    for ax, ylabel, ref_val in [
        (ax1, 'HPWL', ref_hpwl),
        (ax2, 'HPWL / ref', 1.0),
    ]:
        ax.axhline(ref_val, color='white', lw=1, ls=':', alpha=0.5,
                   label='Reference')
        ax.set_xlabel('Iteration', color='white', fontsize=10)
        ax.set_ylabel(ylabel, color='white', fontsize=10)
        ax.legend(fontsize=9, facecolor=BG_DARK, labelcolor='white',
                  framealpha=0.9)

    ax1.set_title('Absolute HPWL', color='white', fontsize=11)
    ax2.set_title('Ratio vs. reference', color='white', fontsize=11)
    plt.suptitle(f'Convergence comparison — {args.circuit}',
                 color='white', fontsize=13)
    plt.tight_layout()

    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, f'{args.circuit}_convergence.png')
    plt.savefig(out, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f"Saved → {out}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='static',
                        choices=['static', 'video', 'trust', 'convergence'])
    parser.add_argument('--circuit',        default='ibm01')
    parser.add_argument('--benchmark_base', default='benchmarks')
    parser.add_argument('--save_dir',       default='local_reviser_ckpt')
    parser.add_argument('--output_dir',     default='viz_output')
    parser.add_argument('--n_iters',  type=int, default=100)
    parser.add_argument('--seed',     type=int, default=42)
    parser.add_argument('--fps',      type=int, default=5)
    # model arch (must match trained checkpoint)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--n_layers',   type=int, default=5)
    args = parser.parse_args()

    dispatch = {
        'static':      mode_static,
        'video':       mode_video,
        'trust':       mode_trust,
        'convergence': mode_convergence,
    }
    dispatch[args.mode](args)


if __name__ == '__main__':
    main()
