"""
CP-SAT Solver for Chip Placement

Three modes:
  1. legalize(): Feasibility-only — NoOverlap2D + boundary, no objective.
     Used once at initialization to make the reference placement legal.
  2. solve_subset(): HPWL optimization — NoOverlap2D + boundary + minimize net-level HPWL.
     Used per LNS iteration on a subset of movable macros.
  3. solve_subset_guided(): Like solve_subset but with per-macro GNN hints,
     variable windows, and CP-SAT internal logging for ML-guided LNS.

All coordinates are in normalized [-1, 1] canvas space, scaled to integers
[0, SCALE] for CP-SAT.
"""

import time as _time
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from ortools.sat.python import cp_model

SCALE = 10000  # [-1, 1] -> [0, SCALE], precision ~0.0002


def _to_int(val: float) -> int:
    """Convert normalized [-1, 1] coordinate to integer [0, SCALE]."""
    return int(round((val + 1.0) / 2.0 * SCALE))


def _to_float(val: int) -> float:
    """Convert integer [0, SCALE] back to normalized [-1, 1]."""
    return val / SCALE * 2.0 - 1.0


def _size_to_int(size_val: float) -> int:
    """Convert normalized size to integer, rounding UP to prevent float-space overlap."""
    import math
    return max(1, math.ceil(size_val / 2.0 * SCALE))


def legalize(
    positions: np.ndarray,
    sizes: np.ndarray,
    time_limit: float = 60.0,
    window_fraction: float = 0.3,
    num_workers: int = 4,
    minimize_displacement: bool = True,
) -> Optional[np.ndarray]:
    """
    CP-SAT legalization: place all macros without overlap or boundary violation.

    Args:
        positions: (N, 2) center coordinates in [-1, 1]
        sizes: (N, 2) component sizes in [-1, 1] canvas scale
        time_limit: max solve time in seconds
        window_fraction: each macro can move ±window_fraction of canvas
        minimize_displacement: if True, minimize total Manhattan displacement
            from input positions. If False, pure feasibility (no objective).

    Returns:
        new_positions: (N, 2) legalized center coordinates, or None if infeasible
    """
    N = positions.shape[0]
    model = cp_model.CpModel()

    # Integer sizes — ceil to prevent float-space overlap after conversion
    w_int = [_size_to_int(sizes[i, 0]) for i in range(N)]
    h_int = [_size_to_int(sizes[i, 1]) for i in range(N)]

    window = int(round(window_fraction * SCALE))

    x_vars = []
    y_vars = []
    x_intervals = []
    y_intervals = []
    hint_xs = []
    hint_ys = []

    for i in range(N):
        cx_int = _to_int(positions[i, 0])
        cy_int = _to_int(positions[i, 1])

        # Bottom-left from center
        bl_x = cx_int - w_int[i] // 2
        bl_y = cy_int - h_int[i] // 2

        # Window constraints + boundary
        x_lo = max(0, bl_x - window)
        x_hi = min(SCALE - w_int[i], bl_x + window)
        y_lo = max(0, bl_y - window)
        y_hi = min(SCALE - h_int[i], bl_y + window)

        if x_lo > x_hi:
            x_lo, x_hi = 0, max(0, SCALE - w_int[i])
        if y_lo > y_hi:
            y_lo, y_hi = 0, max(0, SCALE - h_int[i])

        x = model.new_int_var(x_lo, x_hi, f'x_{i}')
        y = model.new_int_var(y_lo, y_hi, f'y_{i}')
        x_vars.append(x)
        y_vars.append(y)

        # Save hints for displacement objective
        hint_x = max(x_lo, min(x_hi, bl_x))
        hint_y = max(y_lo, min(y_hi, bl_y))
        hint_xs.append(hint_x)
        hint_ys.append(hint_y)

        # Interval variables for NoOverlap2D
        x_end = model.new_int_var(x_lo + w_int[i], x_hi + w_int[i], f'xe_{i}')
        y_end = model.new_int_var(y_lo + h_int[i], y_hi + h_int[i], f'ye_{i}')
        model.add(x_end == x + w_int[i])
        model.add(y_end == y + h_int[i])

        xi = model.new_interval_var(x, w_int[i], x_end, f'xi_{i}')
        yi = model.new_interval_var(y, h_int[i], y_end, f'yi_{i}')
        x_intervals.append(xi)
        y_intervals.append(yi)

        # Warm-start hint
        model.add_hint(x, hint_x)
        model.add_hint(y, hint_y)

    # Non-overlap constraint
    model.add_no_overlap_2d(x_intervals, y_intervals)

    # Displacement minimization objective
    if minimize_displacement:
        disp_terms = []
        for i in range(N):
            # |x_i - hint_x_i| via linear relaxation (exact under minimization)
            abs_dx = model.new_int_var(0, 2 * SCALE, f'adx_{i}')
            model.add(abs_dx >= x_vars[i] - hint_xs[i])
            model.add(abs_dx >= hint_xs[i] - x_vars[i])
            abs_dy = model.new_int_var(0, 2 * SCALE, f'ady_{i}')
            model.add(abs_dy >= y_vars[i] - hint_ys[i])
            model.add(abs_dy >= hint_ys[i] - y_vars[i])
            disp_terms.append(abs_dx)
            disp_terms.append(abs_dy)
        model.minimize(sum(disp_terms))

    # Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = num_workers

    status = solver.solve(model)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        new_positions = np.zeros_like(positions)
        for i in range(N):
            bl_x = solver.value(x_vars[i])
            bl_y = solver.value(y_vars[i])
            # Convert bottom-left back to center in [-1, 1]
            new_positions[i, 0] = _to_float(bl_x + w_int[i] // 2)
            new_positions[i, 1] = _to_float(bl_y + h_int[i] // 2)
        return new_positions
    else:
        return None


def solve_subset(
    positions: np.ndarray,
    sizes: np.ndarray,
    nets: List[List[Tuple[int, float, float]]],
    subset_indices: np.ndarray,
    time_limit: float = 5.0,
    window_fraction: float = 0.15,
    num_workers: int = 4,
) -> Optional[np.ndarray]:
    """
    Solve LNS subproblem: optimize net-level HPWL for subset macros.

    Frozen macros are included in NoOverlap2D as fixed intervals and in
    HPWL as constants. Only subset macros have CP-SAT variables.

    Args:
        positions: (N, 2) current center positions in [-1, 1]
        sizes: (N, 2) component sizes in [-1, 1] canvas scale
        nets: list of nets, each = [(node_idx, pin_dx, pin_dy), ...]
        subset_indices: indices of movable macros
        time_limit: max solve time in seconds
        window_fraction: each macro can move ±window_fraction of canvas

    Returns:
        new_positions: (N, 2) with updated subset positions, or None if infeasible
    """
    N = positions.shape[0]
    subset_set = set(int(i) for i in subset_indices)
    model = cp_model.CpModel()

    # Integer sizes — ceil to prevent float-space overlap
    w_int = [_size_to_int(sizes[i, 0]) for i in range(N)]
    h_int = [_size_to_int(sizes[i, 1]) for i in range(N)]

    window = int(round(window_fraction * SCALE))

    # Spatial filter: only include frozen macros that could overlap with
    # some subset macro after movement. Reduces NoOverlap2D constraint size.
    # Note: window in int space = window_fraction * SCALE, which maps to
    # 2 * window_fraction in float [-1,1] space. Add safety margin.
    float_window = 2.0 * window_fraction + 0.01  # extra margin for int rounding
    sub_pos = positions[subset_indices]  # (K, 2)
    sub_sizes = sizes[subset_indices]    # (K, 2)
    frozen_mask = np.ones(N, dtype=bool)
    frozen_mask[subset_indices] = False
    frozen_idx = np.where(frozen_mask)[0]
    if len(frozen_idx) > 0:
        # Vectorized: for each frozen macro, check if ANY subset macro could overlap
        f_pos = positions[frozen_idx]    # (F, 2)
        f_sizes = sizes[frozen_idx]      # (F, 2)
        # dx[f, k] = |frozen_f_x - subset_k_x|, etc.
        dx = np.abs(f_pos[:, 0:1] - sub_pos[:, 0].reshape(1, -1))  # (F, K)
        dy = np.abs(f_pos[:, 1:2] - sub_pos[:, 1].reshape(1, -1))  # (F, K)
        min_dx = (f_sizes[:, 0:1] + sub_sizes[:, 0].reshape(1, -1)) / 2.0  # (F, K)
        min_dy = (f_sizes[:, 1:2] + sub_sizes[:, 1].reshape(1, -1)) / 2.0  # (F, K)
        could_overlap = (np.maximum(0.0, dx - float_window) < min_dx) & \
                        (np.maximum(0.0, dy - float_window) < min_dy)  # (F, K)
        relevant_frozen_mask = could_overlap.any(axis=1)  # (F,)
        relevant_frozen = set(frozen_idx[relevant_frozen_mask].tolist())
    else:
        relevant_frozen = set()
    relevant_set = subset_set | relevant_frozen

    x_vars = {}  # only for subset
    y_vars = {}
    x_intervals = []
    y_intervals = []

    for i in range(N):
        if i not in relevant_set:
            continue

        cx_int = _to_int(positions[i, 0])
        cy_int = _to_int(positions[i, 1])
        bl_x = cx_int - w_int[i] // 2
        bl_y = cy_int - h_int[i] // 2

        if i in subset_set:
            # Movable: create variables with window constraint
            x_lo = max(0, bl_x - window)
            x_hi = min(SCALE - w_int[i], bl_x + window)
            y_lo = max(0, bl_y - window)
            y_hi = min(SCALE - h_int[i], bl_y + window)

            if x_lo > x_hi:
                x_lo, x_hi = 0, max(0, SCALE - w_int[i])
            if y_lo > y_hi:
                y_lo, y_hi = 0, max(0, SCALE - h_int[i])

            x = model.new_int_var(x_lo, x_hi, f'x_{i}')
            y = model.new_int_var(y_lo, y_hi, f'y_{i}')
            x_vars[i] = x
            y_vars[i] = y

            x_end = model.new_int_var(x_lo + w_int[i], x_hi + w_int[i], f'xe_{i}')
            y_end = model.new_int_var(y_lo + h_int[i], y_hi + h_int[i], f'ye_{i}')
            model.add(x_end == x + w_int[i])
            model.add(y_end == y + h_int[i])

            xi = model.new_interval_var(x, w_int[i], x_end, f'xi_{i}')
            yi = model.new_interval_var(y, h_int[i], y_end, f'yi_{i}')

            # Warm-start hint
            model.add_hint(x, max(x_lo, min(x_hi, bl_x)))
            model.add_hint(y, max(y_lo, min(y_hi, bl_y)))
        else:
            # Frozen: fixed interval
            bl_x_clamped = max(0, min(SCALE - w_int[i], bl_x))
            bl_y_clamped = max(0, min(SCALE - h_int[i], bl_y))

            xi = model.new_fixed_size_interval_var(bl_x_clamped, w_int[i], f'xi_{i}')
            yi = model.new_fixed_size_interval_var(bl_y_clamped, h_int[i], f'yi_{i}')

        x_intervals.append(xi)
        y_intervals.append(yi)

    # Non-overlap constraint (only relevant macros)
    model.add_no_overlap_2d(x_intervals, y_intervals)

    # Net-level HPWL objective: min sum over nets of (max_x - min_x + max_y - min_y)
    # Only include nets that touch at least one subset macro
    objective_terms = []

    for net in nets:
        # Check if any pin in this net belongs to subset
        touches_subset = any(node_idx in subset_set for node_idx, _, _ in net)
        if not touches_subset:
            continue

        pin_xs = []
        pin_ys = []

        for (node_idx, dx, dy) in net:
            dx_int = int(round(dx / 2.0 * SCALE))
            dy_int = int(round(dy / 2.0 * SCALE))

            if node_idx in subset_set:
                # Variable: bottom-left + half_w + pin_offset
                # Bounds must allow negative when pin offset is negative
                px_lo = min(0, dx_int)
                px_hi = SCALE + abs(dx_int)
                px = model.new_int_var(px_lo, px_hi, f'px_{node_idx}_{len(pin_xs)}')
                model.add(px == x_vars[node_idx] + w_int[node_idx] // 2 + dx_int)
                py_lo = min(0, dy_int)
                py_hi = SCALE + abs(dy_int)
                py = model.new_int_var(py_lo, py_hi, f'py_{node_idx}_{len(pin_ys)}')
                model.add(py == y_vars[node_idx] + h_int[node_idx] // 2 + dy_int)
                pin_xs.append(px)
                pin_ys.append(py)
            else:
                # Frozen: constant pin position
                cx_int = _to_int(positions[node_idx, 0])
                cy_int = _to_int(positions[node_idx, 1])
                pin_x_val = cx_int + dx_int
                pin_y_val = cy_int + dy_int
                pin_xs.append(pin_x_val)
                pin_ys.append(pin_y_val)

        if len(pin_xs) < 2:
            continue

        # Net bounding box HPWL = (max_x - min_x) + (max_y - min_y)
        # Bounds must allow negative pin positions (negative offsets near canvas edge)
        net_min_x = model.new_int_var(-SCALE, SCALE * 2, f'nminx_{len(objective_terms)}')
        net_max_x = model.new_int_var(-SCALE, SCALE * 2, f'nmaxx_{len(objective_terms)}')
        net_min_y = model.new_int_var(-SCALE, SCALE * 2, f'nminy_{len(objective_terms)}')
        net_max_y = model.new_int_var(-SCALE, SCALE * 2, f'nmaxy_{len(objective_terms)}')

        model.add_min_equality(net_min_x, pin_xs)
        model.add_max_equality(net_max_x, pin_xs)
        model.add_min_equality(net_min_y, pin_ys)
        model.add_max_equality(net_max_y, pin_ys)

        span_x = model.new_int_var(0, SCALE * 3, f'sx_{len(objective_terms)}')
        span_y = model.new_int_var(0, SCALE * 3, f'sy_{len(objective_terms)}')
        model.add(span_x == net_max_x - net_min_x)
        model.add(span_y == net_max_y - net_min_y)

        objective_terms.append(span_x)
        objective_terms.append(span_y)

    if objective_terms:
        model.minimize(sum(objective_terms))

    # Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = num_workers

    status = solver.solve(model)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        new_positions = positions.copy()
        for i in subset_indices:
            i = int(i)
            bl_x = solver.value(x_vars[i])
            bl_y = solver.value(y_vars[i])
            new_positions[i, 0] = _to_float(bl_x + w_int[i] // 2)
            new_positions[i, 1] = _to_float(bl_y + h_int[i] // 2)
        return new_positions
    else:
        return None


def solve_subset_guided(
    positions: np.ndarray,
    sizes: np.ndarray,
    nets: List[List[Tuple[int, float, float]]],
    subset_indices: np.ndarray,
    time_limit: float = 5.0,
    window_fraction: float = 0.15,
    hint_positions: Optional[np.ndarray] = None,
    per_macro_windows: Optional[np.ndarray] = None,
    num_workers: int = 4,
) -> Dict[str, Any]:
    """
    Guided LNS subproblem: like solve_subset but with per-macro hints,
    variable windows, and CP-SAT internal logging.

    Args:
        positions: (N, 2) current center positions in [-1, 1]
        sizes: (N, 2) component sizes in [-1, 1] canvas scale
        nets: list of nets, each = [(node_idx, pin_dx, pin_dy), ...]
        subset_indices: indices of movable macros
        time_limit: max solve time in seconds
        window_fraction: fallback uniform window (used when per_macro_windows is None)
        hint_positions: (N, 2) GNN-suggested target centers; None → use current positions
        per_macro_windows: (N,) per-macro window fractions; None → use uniform window_fraction

    Returns:
        dict with keys:
            new_positions: (N, 2) updated positions, or None if infeasible
            status: str ('OPTIMAL', 'FEASIBLE', 'INFEASIBLE', 'MODEL_INVALID', 'UNKNOWN')
            objective_value: float (best HPWL objective found, nan if infeasible)
            branches: int
            conflicts: int
            wall_time: float (total wall-clock including Python overhead)
            solver_wall_time: float (CP-SAT solver.wall_time)
            user_time: float (CP-SAT solver.user_time)
    """
    t0 = _time.perf_counter()

    N = positions.shape[0]
    subset_set = set(int(i) for i in subset_indices)
    model = cp_model.CpModel()

    w_int = [_size_to_int(sizes[i, 0]) for i in range(N)]
    h_int = [_size_to_int(sizes[i, 1]) for i in range(N)]

    x_vars: Dict[int, Any] = {}
    y_vars: Dict[int, Any] = {}
    x_intervals = []
    y_intervals = []

    for i in range(N):
        cx_int = _to_int(positions[i, 0])
        cy_int = _to_int(positions[i, 1])
        bl_x = cx_int - w_int[i] // 2
        bl_y = cy_int - h_int[i] // 2

        if i in subset_set:
            # Per-macro or uniform window
            if per_macro_windows is not None:
                window_i = int(round(float(per_macro_windows[i]) * SCALE))
            else:
                window_i = int(round(window_fraction * SCALE))

            # Domain center: hint position when provided, else current position.
            # This is the critical difference from solve_subset: when hint_positions
            # is given, the trust region is centered around the hint (not current pos),
            # so CP-SAT searches in [hint - window, hint + window] instead of
            # [current - window, current + window].
            if hint_positions is not None:
                hint_cx = _to_int(float(hint_positions[i, 0]))
                hint_cy = _to_int(float(hint_positions[i, 1]))
                center_bl_x = hint_cx - w_int[i] // 2
                center_bl_y = hint_cy - h_int[i] // 2
            else:
                center_bl_x = bl_x
                center_bl_y = bl_y

            x_lo = max(0, center_bl_x - window_i)
            x_hi = min(SCALE - w_int[i], center_bl_x + window_i)
            y_lo = max(0, center_bl_y - window_i)
            y_hi = min(SCALE - h_int[i], center_bl_y + window_i)

            if x_lo > x_hi:
                x_lo, x_hi = 0, max(0, SCALE - w_int[i])
            if y_lo > y_hi:
                y_lo, y_hi = 0, max(0, SCALE - h_int[i])

            x = model.new_int_var(x_lo, x_hi, f'x_{i}')
            y = model.new_int_var(y_lo, y_hi, f'y_{i}')
            x_vars[i] = x
            y_vars[i] = y

            x_end = model.new_int_var(x_lo + w_int[i], x_hi + w_int[i], f'xe_{i}')
            y_end = model.new_int_var(y_lo + h_int[i], y_hi + h_int[i], f'ye_{i}')
            model.add(x_end == x + w_int[i])
            model.add(y_end == y + h_int[i])

            xi = model.new_interval_var(x, w_int[i], x_end, f'xi_{i}')
            yi = model.new_interval_var(y, h_int[i], y_end, f'yi_{i}')

            # Warm-start at domain center (hint or current)
            model.add_hint(x, max(x_lo, min(x_hi, center_bl_x)))
            model.add_hint(y, max(y_lo, min(y_hi, center_bl_y)))
        else:
            bl_x_clamped = max(0, min(SCALE - w_int[i], bl_x))
            bl_y_clamped = max(0, min(SCALE - h_int[i], bl_y))
            xi = model.new_fixed_size_interval_var(bl_x_clamped, w_int[i], f'xi_{i}')
            yi = model.new_fixed_size_interval_var(bl_y_clamped, h_int[i], f'yi_{i}')

        x_intervals.append(xi)
        y_intervals.append(yi)

    model.add_no_overlap_2d(x_intervals, y_intervals)

    # Net-level HPWL objective (same logic as solve_subset)
    objective_terms = []
    for net in nets:
        touches_subset = any(node_idx in subset_set for node_idx, _, _ in net)
        if not touches_subset:
            continue

        pin_xs = []
        pin_ys = []
        for (node_idx, dx, dy) in net:
            dx_int = int(round(dx / 2.0 * SCALE))
            dy_int = int(round(dy / 2.0 * SCALE))
            if node_idx in subset_set:
                px_lo = min(0, dx_int)
                px_hi = SCALE + abs(dx_int)
                px = model.new_int_var(px_lo, px_hi, f'px_{node_idx}_{len(pin_xs)}')
                model.add(px == x_vars[node_idx] + w_int[node_idx] // 2 + dx_int)
                py_lo = min(0, dy_int)
                py_hi = SCALE + abs(dy_int)
                py = model.new_int_var(py_lo, py_hi, f'py_{node_idx}_{len(pin_ys)}')
                model.add(py == y_vars[node_idx] + h_int[node_idx] // 2 + dy_int)
                pin_xs.append(px)
                pin_ys.append(py)
            else:
                cx_int = _to_int(positions[node_idx, 0])
                cy_int = _to_int(positions[node_idx, 1])
                pin_xs.append(cx_int + dx_int)
                pin_ys.append(cy_int + dy_int)

        if len(pin_xs) < 2:
            continue

        net_min_x = model.new_int_var(-SCALE, SCALE * 2, f'nminx_{len(objective_terms)}')
        net_max_x = model.new_int_var(-SCALE, SCALE * 2, f'nmaxx_{len(objective_terms)}')
        net_min_y = model.new_int_var(-SCALE, SCALE * 2, f'nminy_{len(objective_terms)}')
        net_max_y = model.new_int_var(-SCALE, SCALE * 2, f'nmaxy_{len(objective_terms)}')
        model.add_min_equality(net_min_x, pin_xs)
        model.add_max_equality(net_max_x, pin_xs)
        model.add_min_equality(net_min_y, pin_ys)
        model.add_max_equality(net_max_y, pin_ys)

        span_x = model.new_int_var(0, SCALE * 3, f'sx_{len(objective_terms)}')
        span_y = model.new_int_var(0, SCALE * 3, f'sy_{len(objective_terms)}')
        model.add(span_x == net_max_x - net_min_x)
        model.add(span_y == net_max_y - net_min_y)
        objective_terms.append(span_x)
        objective_terms.append(span_y)

    if objective_terms:
        model.minimize(sum(objective_terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = num_workers

    status = solver.solve(model)
    wall_time = _time.perf_counter() - t0

    status_names = {
        cp_model.OPTIMAL: 'OPTIMAL',
        cp_model.FEASIBLE: 'FEASIBLE',
        cp_model.INFEASIBLE: 'INFEASIBLE',
        cp_model.MODEL_INVALID: 'MODEL_INVALID',
        cp_model.UNKNOWN: 'UNKNOWN',
    }

    result: Dict[str, Any] = {
        'status': status_names.get(status, 'UNKNOWN'),
        'objective_value': float('nan'),
        'branches': int(solver.num_branches),
        'conflicts': int(solver.num_conflicts),
        'wall_time': wall_time,
        'solver_wall_time': float(solver.wall_time),
        'user_time': float(solver.user_time),
        'new_positions': None,
    }

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        result['objective_value'] = float(solver.objective_value)
        new_positions = positions.copy()
        for i in subset_indices:
            i = int(i)
            bl_x = solver.value(x_vars[i])
            bl_y = solver.value(y_vars[i])
            new_positions[i, 0] = _to_float(bl_x + w_int[i] // 2)
            new_positions[i, 1] = _to_float(bl_y + h_int[i] // 2)
        result['new_positions'] = new_positions

    return result


def compute_net_hpwl(
    positions: np.ndarray,
    sizes: np.ndarray,
    nets: List[List[Tuple[int, float, float]]],
) -> float:
    """
    Compute true net-level HPWL: sum of bounding-box half-perimeter per net.

    Args:
        positions: (N, 2) center positions
        sizes: (N, 2) component sizes (unused, pin offsets are relative to center)
        nets: list of nets

    Returns:
        total_hpwl: scalar
    """
    total = 0.0
    for net in nets:
        xs = []
        ys = []
        for (node_idx, dx, dy) in net:
            xs.append(positions[node_idx, 0] + dx)
            ys.append(positions[node_idx, 1] + dy)
        if len(xs) >= 2:
            total += (max(xs) - min(xs)) + (max(ys) - min(ys))
    return total


def compute_net_hpwl_cached(
    positions: np.ndarray,
    nets: List[List[Tuple[int, float, float]]],
) -> np.ndarray:
    """Compute per-net HPWL array (for cache initialization).

    Returns:
        net_hpwls: (M,) array of per-net HPWL values
    """
    M = len(nets)
    net_hpwls = np.zeros(M, dtype=np.float64)
    for idx, net in enumerate(nets):
        if len(net) < 2:
            continue
        xs = [positions[n, 0] + dx for n, dx, dy in net]
        ys = [positions[n, 1] + dy for n, dx, dy in net]
        net_hpwls[idx] = (max(xs) - min(xs)) + (max(ys) - min(ys))
    return net_hpwls


def compute_incremental_hpwl(
    positions: np.ndarray,
    nets: List[List[Tuple[int, float, float]]],
    subset_indices: np.ndarray,
    net_hpwls: np.ndarray,
    macro_nets: List[List[int]],
) -> Tuple[float, np.ndarray]:
    """Only recompute nets touching moved subset. Returns (total, updated_cache)."""
    updated = net_hpwls.copy()
    dirty_nets = set()
    for i in subset_indices:
        for net_idx in macro_nets[int(i)]:
            dirty_nets.add(net_idx)
    for net_idx in dirty_nets:
        net = nets[net_idx]
        if len(net) < 2:
            updated[net_idx] = 0.0
            continue
        xs = [positions[n, 0] + dx for n, dx, dy in net]
        ys = [positions[n, 1] + dy for n, dx, dy in net]
        updated[net_idx] = (max(xs) - min(xs)) + (max(ys) - min(ys))
    return float(updated.sum()), updated


def check_overlap(
    positions: np.ndarray,
    sizes: np.ndarray,
) -> Tuple[float, int]:
    """
    Check pairwise overlap between all macros.

    Returns:
        total_overlap_area: sum of pairwise overlap areas
        n_overlapping_pairs: number of overlapping pairs
    """
    N = positions.shape[0]
    total_overlap = 0.0
    n_pairs = 0
    for i in range(N):
        for j in range(i + 1, N):
            dx = abs(positions[i, 0] - positions[j, 0])
            dy = abs(positions[i, 1] - positions[j, 1])
            ow = (sizes[i, 0] + sizes[j, 0]) / 2.0 - dx
            oh = (sizes[i, 1] + sizes[j, 1]) / 2.0 - dy
            if ow > 1e-6 and oh > 1e-6:
                total_overlap += ow * oh
                n_pairs += 1
    return total_overlap, n_pairs


def check_boundary(
    positions: np.ndarray,
    sizes: np.ndarray,
    canvas_min: float = -1.0,
    canvas_max: float = 1.0,
) -> float:
    """Check total boundary violation."""
    half_w = sizes[:, 0] / 2.0
    half_h = sizes[:, 1] / 2.0
    viol = 0.0
    viol += np.maximum(0, canvas_min - (positions[:, 0] - half_w)).sum()
    viol += np.maximum(0, (positions[:, 0] + half_w) - canvas_max).sum()
    viol += np.maximum(0, canvas_min - (positions[:, 1] - half_h)).sum()
    viol += np.maximum(0, (positions[:, 1] + half_h) - canvas_max).sum()
    return float(viol)
