"""
Local Reviser Training — Learning solver-aware neighborhoods for exact local repair.

Paper story: ALNS selects WHICH macros to optimize (destroy operator);
the local reviser predicts WHERE they should go (displacement hints) and
HOW TIGHTLY to search (trust radii). CP-SAT does exact repair within
hint-centered trust-region domains.

Modes:
    collect  — Run ALNS with best sweep config, extract local subproblem
               instances from accepted improving moves.
    train    — Supervised regression: displacement MSE + trust radius MSE.
    eval     — 5-condition ablation (equal wall-clock):
               pure / random_hints / hint_only / trust_only / hint_plus_trust

Usage:
    # Collect training data (2000 iters per circuit × 3 circuits)
    python train_local_reviser.py --mode collect --n_iterations 2000

    # Train supervised local reviser
    python train_local_reviser.py --mode train --epochs 100

    # Run 5-condition ablation on ibm01
    python train_local_reviser.py --mode eval --circuit ibm01 --wall_clock 60
"""

import argparse
import json
import os
import sys
import time
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))

from benchmark_loader import load_bookshelf_circuit
from cpsat_solver import (
    legalize, solve_subset, solve_subset_guided,
    compute_net_hpwl, check_overlap,
)
from lns_solver import LNSSolver
from net_spatial_gnn import NetSpatialGNN


# Best config from Phase 1 trimmed sweep
BEST_SS = 10
BEST_WF = 0.1
BEST_TL = 0.3

TRAIN_CIRCUITS = ['ibm01', 'ibm03', 'ibm07']


# ─────────────────────── Local subproblem extraction ──────────────────────────

def extract_local_instance(
    pre_pos: np.ndarray,
    post_pos: np.ndarray,
    subset_indices: np.ndarray,
    sizes: np.ndarray,
    nets: list,
    window_fraction: float,
    bbox_margin: float = 0.2,
) -> dict:
    """
    Extract a normalized local subproblem from one accepted ALNS move.

    Local coordinate system: bounding box of subset pre-positions + margin,
    normalized so longest axis = 1.0 (GLOP-style coordinate_transformation).

    Args:
        pre_pos:         (N, 2) positions before CP-SAT solve
        post_pos:        (N, 2) positions after CP-SAT solve (accepted)
        subset_indices:  (K,) global indices of moved macros
        sizes:           (N, 2) macro sizes
        nets:            list of nets (from load_bookshelf_circuit)
        window_fraction: ALNS window fraction (for trust radius normalization)
        bbox_margin:     fraction of bbox extent to add as padding (default 20%)

    Returns dict with:
        node_features:       (L, 6) [x, y, w, h, is_movable, degree_norm]
        edge_index:          (2, E) local edge pairs
        edge_attr:           (E, 4) pin offsets in local coords
        movable_mask:        (L,) bool — True for subset (movable) nodes
        displacement_target: (K, 2) target displacement normalized to local scale
        trust_radius_target: (K,) in [0.05, 1.0], proportional to actual move size
        n_movable:           K = len(subset_indices)
        local_scale:         scalar for back-converting predictions to canvas coords
        local_center:        (2,) canvas coords of local bbox center
    """
    N = pre_pos.shape[0]
    subset_set = set(int(i) for i in subset_indices)
    K = len(subset_indices)

    # Bounding box of subset pre-positions (using macro extents)
    sub_pre = pre_pos[list(subset_indices)]
    sub_sz  = sizes[list(subset_indices)]
    x_min = float((sub_pre[:, 0] - sub_sz[:, 0] / 2).min())
    x_max = float((sub_pre[:, 0] + sub_sz[:, 0] / 2).max())
    y_min = float((sub_pre[:, 1] - sub_sz[:, 1] / 2).min())
    y_max = float((sub_pre[:, 1] + sub_sz[:, 1] / 2).max())

    mx = (x_max - x_min) * bbox_margin
    my = (y_max - y_min) * bbox_margin
    x_min -= mx;  x_max += mx
    y_min -= my;  y_max += my

    # Anchor macros that overlap with the bbox (context nodes)
    anchor_indices = []
    for i in range(N):
        if i in subset_set:
            continue
        cx, cy = float(pre_pos[i, 0]), float(pre_pos[i, 1])
        hw, hh = float(sizes[i, 0]) / 2, float(sizes[i, 1]) / 2
        if cx + hw > x_min and cx - hw < x_max and cy + hh > y_min and cy - hh < y_max:
            anchor_indices.append(i)

    # Local index ordering: movable nodes first (indices 0..K-1), then anchors
    local_indices = [int(i) for i in subset_indices] + anchor_indices
    L = len(local_indices)
    local_idx_map = {gidx: lidx for lidx, gidx in enumerate(local_indices)}

    movable_mask = np.zeros(L, dtype=bool)
    movable_mask[:K] = True

    # Coordinate normalization: shift to bbox center, scale longest axis to 1.0
    cx_local = (x_min + x_max) / 2.0
    cy_local = (y_min + y_max) / 2.0
    local_scale = max(x_max - x_min, y_max - y_min, 1e-8)

    local_pre = np.zeros((L, 2), dtype=np.float32)
    local_sz  = np.zeros((L, 2), dtype=np.float32)
    for lidx, gidx in enumerate(local_indices):
        local_pre[lidx, 0] = (pre_pos[gidx, 0] - cx_local) / local_scale
        local_pre[lidx, 1] = (pre_pos[gidx, 1] - cy_local) / local_scale
        local_sz[lidx, 0]  = sizes[gidx, 0] / local_scale
        local_sz[lidx, 1]  = sizes[gidx, 1] / local_scale

    # Displacement target (movable nodes only)
    local_post = np.zeros((K, 2), dtype=np.float32)
    for k in range(K):
        gidx = int(subset_indices[k])
        local_post[k, 0] = (post_pos[gidx, 0] - cx_local) / local_scale
        local_post[k, 1] = (post_pos[gidx, 1] - cy_local) / local_scale
    delta_local = local_post - local_pre[:K]  # (K, 2)

    # Trust radius target: smallest trust fraction that contains the actual move + 20% slack.
    # Trust radius = 1.0 means full window_fraction allowed; 0.05 is minimum.
    # window_fraction in canvas space → local_window = window_fraction / local_scale
    local_window = window_fraction / local_scale  # one-sided, in local coords
    trust_targets = np.zeros(K, dtype=np.float32)
    for k in range(K):
        max_delta = float(np.abs(delta_local[k]).max())
        trust_targets[k] = float(np.clip(
            1.2 * max_delta / max(local_window, 1e-8), 0.05, 1.0))

    # Nets touching at least one subset macro, restricted to local nodes
    local_edge_index = []
    local_edge_attr  = []
    degrees = np.zeros(L, dtype=np.float32)

    for net in nets:
        if not any(nidx in subset_set for nidx, _, _ in net):
            continue
        local_pins = [
            (local_idx_map[nidx], dx / local_scale, dy / local_scale)
            for nidx, dx, dy in net if nidx in local_idx_map
        ]
        if len(local_pins) < 2:
            continue
        for lidx, _, _ in local_pins:
            degrees[lidx] += 1
        # Star decomposition: first pin as source
        src_lidx, src_dx, src_dy = local_pins[0]
        for dst_lidx, dst_dx, dst_dy in local_pins[1:]:
            local_edge_index.append([src_lidx, dst_lidx])
            local_edge_index.append([dst_lidx, src_lidx])
            local_edge_attr.append([src_dx, src_dy, dst_dx, dst_dy])
            local_edge_attr.append([dst_dx, dst_dy, src_dx, src_dy])

    if local_edge_index:
        edge_index = np.array(local_edge_index, dtype=np.int64).T   # (2, E)
        edge_attr  = np.array(local_edge_attr,  dtype=np.float32)   # (E, 4)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_attr  = np.zeros((0, 4), dtype=np.float32)

    # Node features: [x, y, w, h, is_movable, degree_norm]
    max_deg = max(float(degrees.max()), 1.0)
    node_features = np.zeros((L, 6), dtype=np.float32)
    node_features[:, 0:2] = local_pre
    node_features[:, 2:4] = local_sz
    node_features[:, 4]   = movable_mask.astype(np.float32)
    node_features[:, 5]   = degrees / max_deg

    return {
        'node_features':       node_features,    # (L, 6)
        'edge_index':          edge_index,        # (2, E)
        'edge_attr':           edge_attr,         # (E, 4)
        'movable_mask':        movable_mask,      # (L,) bool
        'displacement_target': delta_local,       # (K, 2)
        'trust_radius_target': trust_targets,     # (K,)
        'n_movable':           K,
        'local_scale':         float(local_scale),
        'local_center':        np.array([cx_local, cy_local], dtype=np.float32),
    }


def _gnn_inference(model, inst, device):
    """Run GNN on a local instance. Returns (disp_local, trust_local) numpy arrays."""
    nf = torch.from_numpy(inst['node_features']).float().to(device)
    ei = torch.from_numpy(inst['edge_index']).long().to(device)
    ea = torch.from_numpy(inst['edge_attr']).float().to(device)
    if ei.shape[1] == 0:
        K = inst['n_movable']
        return np.zeros((K, 2), dtype=np.float32), np.full(K, 0.5, dtype=np.float32)
    positions_t = nf[:, :2]
    sizes_t     = nf[:, 2:4]
    with torch.no_grad():
        out = model(nf, positions_t, sizes_t, ei, ea)
    disp_local  = out['displacement_pred'].cpu().numpy()   # (L, 2) — use first K rows
    trust_local = out['trust_radius_pred'].cpu().numpy()   # (L,) — use first K rows
    return disp_local, trust_local


# ─────────────────────────────── Collect mode ─────────────────────────────────

def _collect_one_circuit(job):
    """Module-level worker: collect instances from one (circuit, seed) pair."""
    (circuit_name, seed, benchmark_base, subset_size,
     window_fraction, cpsat_time_limit, n_iterations) = job

    circuit_dir = os.path.join(benchmark_base, 'iccad04', 'extracted', circuit_name)
    data = load_bookshelf_circuit(
        circuit_dir, circuit_name, macros_only=True, seed=seed)

    positions  = data['positions']
    sizes      = data['node_features']
    nets       = data['nets']
    edge_index = data['edge_index']
    N          = data['n_components']

    _, ov_pairs = check_overlap(positions, sizes)
    if ov_pairs > 0:
        print(f"  [{circuit_name} s{seed}] Legalizing ({ov_pairs} overlapping pairs)...",
              flush=True)
        legal_pos = legalize(
            positions, sizes, time_limit=60.0,
            window_fraction=0.3, num_workers=4)
        if legal_pos is None:
            return circuit_name, seed, [], f"legalization failed"
        positions = legal_pos

    initial_hpwl = compute_net_hpwl(positions, sizes, nets)
    print(f"  [{circuit_name} s{seed}] N={N} initial_hpwl={initial_hpwl:.4f} — starting ALNS",
          flush=True)

    solver = LNSSolver(
        positions=positions, sizes=sizes, nets=nets, edge_index=edge_index,
        congestion_weight=0.0, subset_size=subset_size,
        window_fraction=window_fraction, cpsat_time_limit=cpsat_time_limit,
        plateau_threshold=20, adapt_threshold=30, seed=seed,
    )

    instances = []
    t0 = time.time()
    for it in range(n_iterations):
        strategy = solver.select_strategy()
        subset   = solver.get_neighborhood(strategy, solver.subset_size)
        solver.strategy_attempts[strategy] += 1
        pre_pos  = solver.current_pos.copy()

        new_pos = solve_subset(
            solver.current_pos, solver.sizes, solver.nets, subset,
            time_limit=solver.cpsat_time_limit,
            window_fraction=solver.window_fraction,
        )
        result = solver._apply_candidate_result(new_pos, subset, strategy, pre_pos)

        if result['accepted'] and result['delta_cost'] < -1e-8 and new_pos is not None:
            inst = extract_local_instance(
                pre_pos, solver.current_pos,
                subset, sizes, nets, solver.window_fraction,
            )
            inst['weight']  = float(-result['delta_cost']) / max(initial_hpwl, 1e-8)
            inst['circuit'] = circuit_name
            inst['seed']    = seed
            instances.append(inst)

        if (it + 1) % 200 == 0:
            print(f"  [{circuit_name} s{seed}] iter {it+1}/{n_iterations} "
                  f"hpwl={solver.best_hpwl:.4f} collected={len(instances)}",
                  flush=True)

    elapsed = time.time() - t0
    msg = (f"N={N} iters={n_iterations} in {elapsed:.1f}s "
           f"— {len(instances)} instances | best HPWL: {solver.best_hpwl:.4f}")
    return circuit_name, seed, instances, msg


def _run_subprocess_job(job_info):
    """Thread worker: launch one collect subprocess, wait for it, load results."""
    circuit_name, seed, tmp_dir, cmd = job_info
    # stdout inherited (live progress to terminal); stderr captured for error reporting
    proc = subprocess.run(cmd, stderr=subprocess.PIPE, text=True)
    inst_path = os.path.join(tmp_dir, 'instances.pt')
    if proc.returncode != 0:
        return circuit_name, seed, [], f"FAILED rc={proc.returncode}: {proc.stderr[-300:]}"
    if os.path.exists(inst_path):
        instances = torch.load(inst_path, weights_only=False)
        return circuit_name, seed, instances, f"{len(instances)} instances"
    return circuit_name, seed, [], "no output file"


def collect(args):
    os.makedirs(args.data_dir, exist_ok=True)

    seeds = list(range(args.seed, args.seed + args.n_seeds))
    job_pairs = [(c, s) for c in args.circuits for s in seeds]

    print(f"Collect: {len(args.circuits)} circuits × {args.n_seeds} seeds "
          f"= {len(job_pairs)} jobs  (n_workers={args.n_workers})")

    all_instances = []

    if args.n_workers == 1:
        for circuit_name, seed in job_pairs:
            job = (circuit_name, seed, args.benchmark_base, args.subset_size,
                   args.window_fraction, args.cpsat_time_limit, args.n_iterations)
            circuit_name, seed, instances, msg = _collect_one_circuit(job)
            print(f"  [{circuit_name} seed={seed}] {msg}")
            all_instances.extend(instances)
    else:
        # Each job runs as a fresh subprocess — avoids fork/spawn deadlocks with
        # OR-Tools + PyTorch. ThreadPoolExecutor manages concurrency (threads only
        # wait on subprocess I/O, no GIL issue).
        script = os.path.abspath(__file__)
        bench  = os.path.abspath(args.benchmark_base)
        job_infos = []
        for circuit_name, seed in job_pairs:
            tmp_dir = os.path.join(args.data_dir, f'_tmp_{circuit_name}_s{seed}')
            os.makedirs(tmp_dir, exist_ok=True)
            cmd = [
                sys.executable, script,
                '--mode', 'collect',
                '--benchmark_base', bench,
                '--circuits', circuit_name,
                '--n_iterations', str(args.n_iterations),
                '--seed', str(seed),
                '--n_seeds', '1',
                '--n_workers', '1',   # single-job subprocess, no recursion
                '--data_dir', tmp_dir,
                '--subset_size',      str(args.subset_size),
                '--window_fraction',  str(args.window_fraction),
                '--cpsat_time_limit', str(args.cpsat_time_limit),
            ]
            job_infos.append((circuit_name, seed, tmp_dir, cmd))

        n_concurrent = min(args.n_workers, len(job_infos))
        with ThreadPoolExecutor(max_workers=n_concurrent) as tex:
            futures = {tex.submit(_run_subprocess_job, ji): ji for ji in job_infos}
            for future in as_completed(futures):
                circuit_name, seed, instances, msg = future.result()
                print(f"  [{circuit_name} seed={seed}] {msg}", flush=True)
                all_instances.extend(instances)

        for _, _, tmp_dir, _ in job_infos:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    out_path = os.path.join(args.data_dir, 'instances.pt')
    torch.save(all_instances, out_path)
    print(f"\nSaved {len(all_instances)} instances → {out_path}")


# ─────────────────────────────── Train mode ───────────────────────────────────

def build_model(args, device):
    model = NetSpatialGNN(
        node_input_dim=6,
        topo_edge_dim=4,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        mode='topology_only',
    ).to(device)
    return model


def train(args):
    data_path = os.path.join(args.data_dir, 'instances.pt')
    if not os.path.exists(data_path):
        print(f"ERROR: no data at {data_path} — run --mode collect first")
        return
    instances = torch.load(data_path)
    print(f"Loaded {len(instances)} instances")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = build_model(args, device)

    # Freeze heads not used by the local reviser
    for name, param in model.named_parameters():
        if any(h in name for h in ('subset_head', 'value_mlp', 'heatmap_head')):
            param.requires_grad_(False)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_trainable:,}  (device: {device})")

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
    )

    os.makedirs(args.save_dir, exist_ok=True)
    best_loss = float('inf')
    ckpt = os.path.join(args.save_dir, 'local_reviser_best.pt')

    for epoch in range(args.epochs):
        perm = np.random.permutation(len(instances))
        total_disp = total_trust = total_w = 0.0
        model.train()

        for idx in perm:
            inst = instances[int(idx)]
            nf       = torch.from_numpy(inst['node_features']).float().to(device)
            ei       = torch.from_numpy(inst['edge_index']).long().to(device)
            ea       = torch.from_numpy(inst['edge_attr']).float().to(device)
            mask     = torch.from_numpy(inst['movable_mask']).to(device)
            disp_tgt = torch.from_numpy(inst['displacement_target']).float().to(device)
            trst_tgt = torch.from_numpy(inst['trust_radius_target']).float().to(device)
            w        = float(inst.get('weight', 1.0))

            if ei.shape[1] == 0:
                continue  # no local edges → skip (GNN has nothing to propagate)

            positions_t = nf[:, :2]
            sizes_t     = nf[:, 2:4]

            out = model(nf, positions_t, sizes_t, ei, ea)
            disp_pred  = out['displacement_pred'][mask]    # (K, 2)
            trust_pred = out['trust_radius_pred'][mask]    # (K,)

            disp_loss  = nn.functional.mse_loss(disp_pred, disp_tgt)
            trust_loss = nn.functional.mse_loss(trust_pred, trst_tgt)
            loss = w * (disp_loss + 0.5 * trust_loss)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_disp  += float(disp_loss) * w
            total_trust += float(trust_loss) * w
            total_w     += w

        avg_disp  = total_disp  / max(total_w, 1e-8)
        avg_trust = total_trust / max(total_w, 1e-8)
        total_loss = avg_disp + 0.5 * avg_trust

        print(f"Epoch {epoch+1:4d}/{args.epochs}  "
              f"disp={avg_disp:.6f}  trust={avg_trust:.6f}  total={total_loss:.6f}")

        if total_loss < best_loss:
            best_loss = total_loss
            torch.save(model.state_dict(), ckpt)

    print(f"\nBest loss: {best_loss:.6f}  →  {ckpt}")


# ─────────────────────────────── Eval mode ────────────────────────────────────

def run_condition(
    condition: str,
    positions_init: np.ndarray,
    sizes: np.ndarray,
    nets: list,
    edge_index: np.ndarray,
    wall_clock: float,
    model,
    device,
    window_fraction: float = BEST_WF,
    cpsat_time_limit: float = BEST_TL,
    subset_size: int = BEST_SS,
    seed: int = 42,
) -> dict:
    """Run one wall-clock-bounded condition and return metrics."""
    solver = LNSSolver(
        positions=positions_init.copy(),
        sizes=sizes,
        nets=nets,
        edge_index=edge_index,
        congestion_weight=0.0,
        subset_size=subset_size,
        window_fraction=window_fraction,
        cpsat_time_limit=cpsat_time_limit,
        plateau_threshold=20,
        adapt_threshold=30,
        seed=seed,
    )

    n_iters = n_improved = n_infeasible = 0
    total_branches = total_conflicts = total_solver_wall = 0.0
    rng = np.random.default_rng(seed + 1)
    t0 = time.time()

    while time.time() - t0 < wall_clock:
        strategy = solver.select_strategy()
        subset   = solver.get_neighborhood(strategy, solver.subset_size)
        solver.strategy_attempts[strategy] += 1
        pre_pos  = solver.current_pos.copy()

        # All conditions go through solve_subset_guided so we get uniform telemetry.
        # pure: no hints, no per-macro windows → identical to solve_subset().
        hint_pos       = None
        per_macro_wins = None

        if condition != 'pure':
            inst = extract_local_instance(
                pre_pos, pre_pos,  # post=pre: targets unused at inference time
                subset, sizes, nets, window_fraction,
            )

            if condition in ('hint_only', 'hint_plus_trust') and model is not None:
                disp_local, trust_local = _gnn_inference(model, inst, device)
                local_scale = inst['local_scale']
                hint_pos = pre_pos.copy()
                for k, gidx in enumerate(subset):
                    hint_pos[gidx, 0] = pre_pos[gidx, 0] + disp_local[k, 0] * local_scale
                    hint_pos[gidx, 1] = pre_pos[gidx, 1] + disp_local[k, 1] * local_scale
                if condition == 'hint_plus_trust':
                    per_macro_wins = np.full(len(pre_pos), window_fraction, dtype=np.float64)
                    for k, gidx in enumerate(subset):
                        per_macro_wins[gidx] = float(trust_local[k]) * window_fraction

            elif condition == 'trust_only' and model is not None:
                _, trust_local = _gnn_inference(model, inst, device)
                per_macro_wins = np.full(len(pre_pos), window_fraction, dtype=np.float64)
                for k, gidx in enumerate(subset):
                    per_macro_wins[gidx] = float(trust_local[k]) * window_fraction

            elif condition == 'random_hints':
                hint_pos = pre_pos.copy()
                noise = rng.uniform(-0.5, 0.5, size=(len(subset), 2)) * window_fraction
                for k, gidx in enumerate(subset):
                    hint_pos[gidx, 0] = pre_pos[gidx, 0] + noise[k, 0]
                    hint_pos[gidx, 1] = pre_pos[gidx, 1] + noise[k, 1]

        res_guided = solve_subset_guided(
            solver.current_pos, sizes, nets, subset,
            time_limit=cpsat_time_limit,
            window_fraction=window_fraction,
            hint_positions=hint_pos,
            per_macro_windows=per_macro_wins,
        )
        new_pos = res_guided['new_positions']
        total_branches    += res_guided.get('branches', 0)
        total_conflicts   += res_guided.get('conflicts', 0)
        total_solver_wall += res_guided.get('solver_wall_time', 0.0)

        result = solver._apply_candidate_result(new_pos, subset, strategy, pre_pos)
        n_iters += 1
        if result['improved']:
            n_improved += 1
        if not result['feasible']:
            n_infeasible += 1

    elapsed = time.time() - t0
    n_feasible = max(n_iters - n_infeasible, 1)
    return {
        'condition':          condition,
        'best_hpwl':          solver.best_hpwl,
        'n_iters':            n_iters,
        'n_improved':         n_improved,
        'n_infeasible':       n_infeasible,
        'elapsed_s':          elapsed,
        'branches_per_call':  total_branches    / n_feasible,
        'conflicts_per_call': total_conflicts   / n_feasible,
        'solver_ms_per_call': total_solver_wall / n_feasible * 1000,
        'impr_per_s':         n_improved / max(elapsed, 1e-8),
    }


def eval_mode(args):
    circuit_dir = os.path.join(
        args.benchmark_base, 'iccad04', 'extracted', args.circuit)
    data = load_bookshelf_circuit(
        circuit_dir, args.circuit, macros_only=True, seed=args.seed)

    positions  = data['positions']
    sizes      = data['node_features']
    nets       = data['nets']
    edge_index = data['edge_index']

    _, ov_pairs = check_overlap(positions, sizes)
    if ov_pairs > 0:
        print(f"Legalizing ({ov_pairs} overlapping pairs)...")
        positions = legalize(
            positions, sizes, time_limit=60.0,
            window_fraction=0.3, num_workers=4)
        if positions is None:
            print("ERROR: legalization failed")
            return

    ref_hpwl = compute_net_hpwl(positions, sizes, nets)
    print(f"Circuit: {args.circuit}, N={data['n_components']}, "
          f"ref HPWL={ref_hpwl:.4f}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = None
    ckpt   = os.path.join(args.save_dir, 'local_reviser_best.pt')
    if os.path.exists(ckpt):
        model = build_model(args, device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()
        print(f"Loaded model from {ckpt}")
    else:
        print(f"No checkpoint at {ckpt} — ML conditions will behave like pure/random")

    conditions = ['pure', 'random_hints', 'hint_only', 'trust_only', 'hint_plus_trust']
    results = []

    for cond in conditions:
        print(f"\nRunning: {cond} ({args.wall_clock}s wall-clock)...", flush=True)
        np.random.seed(args.seed)
        r = run_condition(
            cond, positions, sizes, nets, edge_index,
            wall_clock=args.wall_clock,
            model=model, device=device,
            window_fraction=BEST_WF,
            cpsat_time_limit=BEST_TL,
            subset_size=BEST_SS,
            seed=args.seed,
        )
        r['ratio_vs_ref'] = r['best_hpwl'] / max(ref_hpwl, 1e-8)
        results.append(r)
        print(f"  HPWL={r['best_hpwl']:.4f} ({r['ratio_vs_ref']:.3f}x ref)  "
              f"iters={r['n_iters']} improved={r['n_improved']} "
              f"infeas={r['n_infeasible']}  "
              f"branches/call={r['branches_per_call']:.1f}  "
              f"solver_ms/call={r['solver_ms_per_call']:.1f}")

    print(f"\n{'='*90}")
    print(f"5-CONDITION ABLATION: {args.circuit} ({args.wall_clock}s per condition)")
    print(f"{'='*90}")
    hdr = (f"{'Condition':<20} {'HPWL':>8} {'Ratio':>7} {'Iters':>7} {'Impr':>6} "
           f"{'Infeas':>7} {'Branch/c':>9} {'Conf/c':>7} {'ms/c':>7} {'Impr/s':>7}")
    print(hdr)
    print('-' * len(hdr))
    for r in results:
        print(f"{r['condition']:<20} {r['best_hpwl']:8.4f} {r['ratio_vs_ref']:7.3f}x "
              f"{r['n_iters']:7d} {r['n_improved']:6d} {r['n_infeasible']:7d} "
              f"{r['branches_per_call']:9.1f} {r['conflicts_per_call']:7.1f} "
              f"{r['solver_ms_per_call']:7.1f} {r['impr_per_s']:7.3f}")

    os.makedirs(args.save_dir, exist_ok=True)
    out_path = os.path.join(args.save_dir, f'ablation_{args.circuit}.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved ablation results → {out_path}")


# ──────────────────────────────────── Main ────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Local Reviser Training')
    parser.add_argument('--mode', choices=['collect', 'train', 'eval'], required=True)

    # Data
    parser.add_argument('--benchmark_base', default='benchmarks')
    parser.add_argument('--circuits', nargs='+', default=TRAIN_CIRCUITS,
                        help='Circuits for collect mode')
    parser.add_argument('--circuit', default='ibm01',
                        help='Single circuit for eval mode')
    parser.add_argument('--data_dir',  default='local_reviser_data')
    parser.add_argument('--save_dir',  default='local_reviser_ckpt')
    parser.add_argument('--seed', type=int, default=42)

    # ALNS params (best from Phase 1 sweep)
    parser.add_argument('--subset_size',      type=int,   default=BEST_SS)
    parser.add_argument('--window_fraction',  type=float, default=BEST_WF)
    parser.add_argument('--cpsat_time_limit', type=float, default=BEST_TL)
    parser.add_argument('--n_iterations',     type=int,   default=2000,
                        help='ALNS iterations per (circuit, seed) in collect mode')
    parser.add_argument('--n_seeds',          type=int,   default=1,
                        help='Number of independent seeds per circuit (more diverse data)')
    parser.add_argument('--n_workers',        type=int,   default=1,
                        help='Parallel collect workers; each uses 4 CP-SAT threads')

    # Model
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--n_layers',   type=int, default=5)

    # Train
    parser.add_argument('--epochs', type=int,   default=100)
    parser.add_argument('--lr',     type=float, default=1e-3)

    # Eval
    parser.add_argument('--wall_clock', type=float, default=60.0,
                        help='Wall-clock seconds per condition in eval mode')

    args = parser.parse_args()

    if args.mode == 'collect':
        collect(args)
    elif args.mode == 'train':
        train(args)
    elif args.mode == 'eval':
        eval_mode(args)


if __name__ == '__main__':
    main()
