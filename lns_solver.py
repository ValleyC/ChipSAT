"""
Large Neighborhood Search (LNS) Solver for Chip Placement

Orchestrates iterative CP-SAT subproblem solving:
  1. Select a neighborhood (subset of macros to re-place)
  2. Solve subset with CP-SAT (NoOverlap2D + minimize net-level HPWL)
  3. Accept/reject based on full cost recomputation
  4. Adaptive window/subset sizing

Acceptance policy:
  - Phase 1: improvement-only (greedy descent)
  - Phase 2: simulated annealing (activated after plateau)
"""

import numpy as np
import time
from typing import List, Tuple, Optional, Dict
from collections import deque

from cpsat_solver import (
    legalize, solve_subset, compute_net_hpwl, check_overlap, check_boundary,
    compute_net_hpwl_cached, compute_incremental_hpwl,
)


def compute_density_np(
    positions: np.ndarray,
    sizes: np.ndarray,
    grid_size: int = 8,
    canvas_min: float = -1.0,
    canvas_max: float = 1.0,
) -> float:
    """Cell-area density (deprecated, use compute_rudy_np for congestion)."""
    G = grid_size
    cell_size = (canvas_max - canvas_min) / G
    cell_area = cell_size * cell_size
    total_macro_area = (sizes[:, 0] * sizes[:, 1]).sum()
    canvas_area = (canvas_max - canvas_min) ** 2
    target_density = total_macro_area / canvas_area
    edges = np.linspace(canvas_min, canvas_max, G + 1)
    half_w = sizes[:, 0] / 2
    half_h = sizes[:, 1] / 2
    comp_xmin = positions[:, 0] - half_w
    comp_xmax = positions[:, 0] + half_w
    comp_ymin = positions[:, 1] - half_h
    comp_ymax = positions[:, 1] + half_h
    density = np.zeros((G, G))
    for gx in range(G):
        for gy in range(G):
            cx_lo, cx_hi = edges[gx], edges[gx + 1]
            cy_lo, cy_hi = edges[gy], edges[gy + 1]
            ox = np.maximum(0, np.minimum(comp_xmax, cx_hi) - np.maximum(comp_xmin, cx_lo))
            oy = np.maximum(0, np.minimum(comp_ymax, cy_hi) - np.maximum(comp_ymin, cy_lo))
            density[gx, gy] = (ox * oy).sum() / cell_area
    excess = np.maximum(0, density - target_density)
    return float((excess ** 2).sum())


def compute_rudy_np(
    positions: np.ndarray,
    sizes: np.ndarray,
    nets: List[List[Tuple[int, float, float]]],
    grid_size: int = 32,
    canvas_min: float = -1.0,
    canvas_max: float = 1.0,
) -> Dict:
    """
    Net-based RUDY (Rectangular Uniform wire DensitY) congestion.

    For each net, distributes wire density weight = (w_eff + h_eff) / (w_eff * h_eff)
    uniformly over the net bounding box, accumulated on a grid.
    Overflow = max(0, rudy_map - capacity) where capacity = mean(rudy_map).

    Returns dict with: cost, rudy_map, rudy_max, rudy_p95, rudy_p99, overflow_sum
    """
    G = grid_size
    tile_w = (canvas_max - canvas_min) / G
    tile_h = tile_w
    edges = np.linspace(canvas_min, canvas_max, G + 1)

    # Fixed minimum bbox extent — independent of grid_size for invariance.
    # A net's wire must span at least this fraction of the canvas.
    min_extent = (canvas_max - canvas_min) / 16.0

    # Precompute net bboxes and weights — vectorized over pins per net
    net_bboxes = []  # (min_x, max_x, min_y, max_y)
    net_weights = []

    for net in nets:
        if len(net) < 2:
            continue
        pin_xs = np.array([positions[n, 0] + dx for n, dx, dy in net])
        pin_ys = np.array([positions[n, 1] + dy for n, dx, dy in net])
        x_lo, x_hi = float(pin_xs.min()), float(pin_xs.max())
        y_lo, y_hi = float(pin_ys.min()), float(pin_ys.max())
        bbox_w = x_hi - x_lo
        bbox_h = y_hi - y_lo
        # Degenerate clamping: expand bbox to fixed min_extent (grid-size invariant)
        w_eff = max(bbox_w, min_extent)
        h_eff = max(bbox_h, min_extent)
        # Expand bbox symmetrically if degenerate
        if bbox_w < min_extent:
            cx = (x_lo + x_hi) / 2
            x_lo = cx - w_eff / 2
            x_hi = cx + w_eff / 2
        if bbox_h < min_extent:
            cy = (y_lo + y_hi) / 2
            y_lo = cy - h_eff / 2
            y_hi = cy + h_eff / 2
        weight = (w_eff + h_eff) / (w_eff * h_eff)
        net_bboxes.append((x_lo, x_hi, y_lo, y_hi))
        net_weights.append(weight)

    rudy_map = np.zeros((G, G), dtype=np.float64)

    if len(net_bboxes) == 0:
        return {
            'cost': 0.0,
            'rudy_map': rudy_map,
            'rudy_max': 0.0,
            'rudy_p95': 0.0,
            'rudy_p99': 0.0,
            'overflow_sum': 0.0,
        }

    # Vectorized: (M, 4) bboxes, (M,) weights
    bboxes = np.array(net_bboxes, dtype=np.float64)  # (M, 4)
    weights = np.array(net_weights, dtype=np.float64)  # (M,)

    # Accumulate RUDY on grid — vectorized over nets per tile
    # Normalize by tile area so values are density (wire demand per unit area),
    # making the map grid-size invariant.
    tile_area = tile_w * tile_h
    for gx in range(G):
        for gy in range(G):
            tx_lo, tx_hi = edges[gx], edges[gx + 1]
            ty_lo, ty_hi = edges[gy], edges[gy + 1]
            # Overlap of each net bbox with this tile
            ox = np.maximum(0, np.minimum(bboxes[:, 1], tx_hi) - np.maximum(bboxes[:, 0], tx_lo))
            oy = np.maximum(0, np.minimum(bboxes[:, 3], ty_hi) - np.maximum(bboxes[:, 2], ty_lo))
            overlap_area = ox * oy  # (M,)
            rudy_map[gx, gy] = (weights * overlap_area).sum() / tile_area

    # Overflow-style cost.
    # capacity = mean(rudy_map): a heuristic proxy for uniform routing capacity.
    # This is NOT calibrated to actual routing resources; it represents the
    # average wire density that would result from perfectly uniform distribution.
    # Suitable as an optimization proxy; real evaluation requires global routing.
    capacity = rudy_map.mean()
    overflow = np.maximum(0, rudy_map - capacity)

    # Multiply overflow by tile_area to get a spatial integral (grid-size invariant).
    # Without this, sum(overflow) scales with number of tiles.
    overflow_integral = float((overflow * tile_area).sum())

    flat = rudy_map.flatten()
    return {
        'cost': overflow_integral,
        'rudy_map': rudy_map,
        'rudy_max': float(flat.max()),
        'rudy_p95': float(np.percentile(flat, 95)),
        'rudy_p99': float(np.percentile(flat, 99)),
        'overflow_sum': overflow_integral,
    }


def compute_per_macro_rudy(
    positions: np.ndarray,
    sizes: np.ndarray,
    nets: List[List[Tuple[int, float, float]]],
    macro_nets: List[List[int]],
    grid_size: int = 32,
    canvas_min: float = -1.0,
    canvas_max: float = 1.0,
) -> np.ndarray:
    """
    Per-macro RUDY score: mean RUDY density over tiles touched by the macro's nets.

    Averaged over incident net count to avoid bias toward high-degree macros.
    This measures "how congested is this macro's routing neighborhood" rather
    than "how much total wire passes through this macro."

    Returns:
        (N,) array of per-macro RUDY scores
    """
    N = positions.shape[0]
    G = grid_size
    tile_w = (canvas_max - canvas_min) / G

    # First compute full RUDY map
    rudy_info = compute_rudy_np(positions, sizes, nets, grid_size, canvas_min, canvas_max)
    rudy_map = rudy_info['rudy_map']

    # For each macro, average RUDY over tiles touched by its nets
    scores = np.zeros(N, dtype=np.float64)

    for i in range(N):
        valid_nets = 0
        macro_rudy = 0.0
        for net_idx in macro_nets[i]:
            net = nets[net_idx]
            if len(net) < 2:
                continue
            pin_xs = [positions[n, 0] + dx for n, dx, dy in net]
            pin_ys = [positions[n, 1] + dy for n, dx, dy in net]
            x_lo, x_hi = min(pin_xs), max(pin_xs)
            y_lo, y_hi = min(pin_ys), max(pin_ys)
            # Find overlapping tiles
            gx_lo = max(0, int((x_lo - canvas_min) / tile_w))
            gx_hi = min(G - 1, int((x_hi - canvas_min) / tile_w))
            gy_lo = max(0, int((y_lo - canvas_min) / tile_w))
            gy_hi = min(G - 1, int((y_hi - canvas_min) / tile_w))
            region = rudy_map[gx_lo:gx_hi + 1, gy_lo:gy_hi + 1]
            n_tiles = max(region.size, 1)
            macro_rudy += region.sum() / n_tiles  # mean density per net
            valid_nets += 1
        # Average over incident nets to reduce degree bias
        if valid_nets > 0:
            scores[i] = macro_rudy / valid_nets

    return scores


class ALNSWeights:
    """Adaptive operator selection via segmented roulette wheel (Ropke & Pisinger 2006).

    Scores per outcome:
      new_best=33, improved=9, accepted=3, rejected=0
    Weights updated at segment boundaries:
      w = (1 - rho) * w + rho * (segment_score / segment_uses)
    Floor weight 0.01, renormalized after update.
    """

    SCORES = {'new_best': 33, 'improved': 9, 'accepted': 3, 'rejected': 0}

    def __init__(self, n_strategies: int, segment_size: int = 25, rho: float = 0.1):
        self.n = n_strategies
        self.segment_size = segment_size
        self.rho = rho
        self.weights = np.ones(n_strategies) / n_strategies
        # Segment accumulators
        self.seg_scores = np.zeros(n_strategies)
        self.seg_uses = np.zeros(n_strategies, dtype=int)
        self.seg_iter = 0

    def select(self, rng: np.random.Generator) -> int:
        """Roulette wheel selection from current weights."""
        p = self.weights / self.weights.sum()
        return int(rng.choice(self.n, p=p))

    def sample_multiple(self, rng: np.random.Generator, m: int) -> np.ndarray:
        """Sample m operators without replacement from current weights."""
        m = min(m, self.n)
        p = self.weights / self.weights.sum()
        return rng.choice(self.n, size=m, replace=False, p=p)

    def record_outcome(self, strategy_idx: int, outcome: str):
        """Accumulate score for this strategy. Triggers update at segment end."""
        self.seg_scores[strategy_idx] += self.SCORES.get(outcome, 0)
        self.seg_uses[strategy_idx] += 1
        self.seg_iter += 1
        if self.seg_iter >= self.segment_size:
            self._update_weights()

    def _update_weights(self):
        """Blend segment performance into weights."""
        for i in range(self.n):
            if self.seg_uses[i] > 0:
                avg_score = self.seg_scores[i] / self.seg_uses[i]
                self.weights[i] = (1 - self.rho) * self.weights[i] + self.rho * avg_score
        # Floor and renormalize
        self.weights = np.maximum(self.weights, 0.01)
        self.weights /= self.weights.sum()
        # Reset segment
        self.seg_scores[:] = 0
        self.seg_uses[:] = 0
        self.seg_iter = 0

    def get_weights_dict(self, strategy_names: list) -> dict:
        """Return weights as {name: weight} for logging."""
        return {name: float(self.weights[i]) for i, name in enumerate(strategy_names)}


# Top-level worker function for parallel mode (Windows spawn requires this)
_pool_static_data = {}  # set by pool initializer


def _pool_initializer(sizes, nets):
    """Store static data in worker process."""
    _pool_static_data['sizes'] = sizes
    _pool_static_data['nets'] = nets


def _pool_solve_subset(args):
    """Worker function: solve_subset with static sizes/nets from initializer."""
    positions, subset, time_limit, window_fraction, num_workers = args
    sizes = _pool_static_data['sizes']
    nets = _pool_static_data['nets']
    return solve_subset(
        positions, sizes, nets, subset,
        time_limit=time_limit,
        window_fraction=window_fraction,
        num_workers=num_workers,
    )


class LNSSolver:
    """Large Neighborhood Search for chip macro placement."""

    def __init__(
        self,
        positions: np.ndarray,
        sizes: np.ndarray,
        nets: List[List[Tuple[int, float, float]]],
        edge_index: np.ndarray,
        congestion_weight: float = 0.1,
        subset_size: int = 30,
        window_fraction: float = 0.15,
        cpsat_time_limit: float = 5.0,
        plateau_threshold: int = 20,
        adapt_threshold: int = 30,
        min_subset: int = 10,
        min_window: float = 0.05,
        max_window: float = 0.4,
        sa_t_init: float = 0.5,
        sa_cooling: float = 0.995,
        seed: int = 42,
        alns_segment_size: int = 25,
        alns_rho: float = 0.1,
        n_parallel_candidates: int = 1,
        model=None,
        edge_attr=None,
    ):
        # Hard guard: model requires edge_attr
        if model is not None and edge_attr is None:
            raise ValueError("edge_attr required when model is provided")

        self.N = positions.shape[0]
        self.sizes = sizes.copy()
        self.nets = nets
        self.edge_index = edge_index
        self.edge_attr_np = edge_attr  # (E, 4) pin offsets for GNN
        self.congestion_weight = congestion_weight
        self.rng = np.random.default_rng(seed)

        # ML model for learned strategy
        self.model = model
        self._cached_tensors = None
        if model is not None:
            self._cache_static_tensors()

        # Per-strategy tracking — add 'learned' if model present
        self.strategies = ['random', 'worst_hpwl', 'congestion', 'connected']
        if model is not None:
            self.strategies.append('learned')
        self.strategy_attempts = {s: 0 for s in self.strategies}
        self.strategy_successes = {s: 0 for s in self.strategies}

        # Precompute net adjacency (macro_nets: which nets each macro belongs to)
        self._precompute_macro_nets()

        # Initialize per-net HPWL cache
        self.net_hpwls = compute_net_hpwl_cached(positions, nets)

        # Current and best solutions
        self.current_pos = positions.copy()
        self.best_pos = positions.copy()
        self.current_hpwl = float(self.net_hpwls.sum())
        self.best_hpwl = self.current_hpwl
        self.current_cost = self._compute_cost(positions, self.current_hpwl)
        self.best_cost = self.current_cost

        # Compute per-macro HPWL from net cache
        self.macro_hpwl = np.zeros(self.N)
        self._update_macro_hpwl()  # full recompute from net_hpwls

        # Adaptive parameters
        self.subset_size = subset_size
        self.window_fraction = window_fraction
        self.cpsat_time_limit = cpsat_time_limit
        self.min_subset = min_subset
        self.min_window = min_window
        self.max_window = max_window
        self.plateau_threshold = plateau_threshold
        self.adapt_threshold = adapt_threshold

        # Simulated annealing
        self.sa_t_init = sa_t_init
        self.sa_cooling = sa_cooling
        self.sa_temperature = sa_t_init
        self.sa_active = False

        # Tracking
        self.stagnation_count = 0
        self.iteration = 0
        self.n_accepted = 0
        self.n_improved = 0
        self.n_infeasible = 0

        # Search state for GNN features
        self.last_subset_mask = np.zeros(self.N, dtype=np.float32)
        self.last_delta = np.zeros(self.N, dtype=np.float32)

        # ALNS operator selection
        self.alns = ALNSWeights(
            n_strategies=len(self.strategies),
            segment_size=alns_segment_size,
            rho=alns_rho,
        )

        # Parallel candidate evaluation
        self.n_parallel_candidates = n_parallel_candidates
        self._pool = None  # lazy ProcessPoolExecutor

        # Build adjacency from edge_index for connected strategy
        self.adj = [[] for _ in range(self.N)]
        if edge_index is not None and edge_index.shape[1] > 0:
            for e in range(edge_index.shape[1]):
                src, dst = int(edge_index[0, e]), int(edge_index[1, e])
                if src < self.N and dst < self.N:
                    self.adj[src].append(dst)

    def _compute_cost(self, positions: np.ndarray, hpwl: float) -> float:
        """Compute full cost: HPWL + congestion_weight * RUDY_overflow."""
        if self.congestion_weight > 0:
            rudy = compute_rudy_np(positions, self.sizes, self.nets)
            return hpwl + self.congestion_weight * rudy['cost']
        return hpwl

    def _precompute_macro_nets(self):
        """Build macro_nets adjacency: which nets each macro belongs to."""
        self.macro_nets = [[] for _ in range(self.N)]
        for net_idx, net in enumerate(self.nets):
            if len(net) < 2:
                continue
            for (node_idx, _, _) in net:
                if node_idx < self.N:
                    self.macro_nets[node_idx].append(net_idx)

    def _update_macro_hpwl(self, subset_indices=None):
        """Update per-macro HPWL from net_hpwls cache.

        If subset_indices given, only recompute macros whose nets were dirty.
        Otherwise full recompute (used at initialization).
        """
        if subset_indices is None:
            # Full recompute
            self.macro_hpwl[:] = 0.0
            for net_idx, net in enumerate(self.nets):
                if len(net) < 2:
                    continue
                hpwl = self.net_hpwls[net_idx]
                for (node_idx, _, _) in net:
                    if node_idx < self.N:
                        self.macro_hpwl[node_idx] += hpwl
            return

        # Incremental: find affected macros via dirty nets
        dirty_nets = set()
        for i in subset_indices:
            for net_idx in self.macro_nets[int(i)]:
                dirty_nets.add(net_idx)
        affected_macros = set()
        for net_idx in dirty_nets:
            for (node_idx, _, _) in self.nets[net_idx]:
                if node_idx < self.N:
                    affected_macros.add(node_idx)
        for m in affected_macros:
            self.macro_hpwl[m] = sum(self.net_hpwls[ni] for ni in self.macro_nets[m])

    def _cache_static_tensors(self):
        """Cache immutable tensors for GNN inference (avoid repeated conversion)."""
        import torch
        device = next(self.model.parameters()).device if self.model else 'cpu'
        self._cached_tensors = {
            'edge_index_t': torch.from_numpy(self.edge_index).long().to(device),
            'edge_attr_t': torch.from_numpy(self.edge_attr_np).float().to(device),
            'sizes_t': torch.from_numpy(self.sizes).float().to(device),
        }
        self._gnn_device = device

    def _build_gnn_features(self):
        """Build 10D node features for GNN. Only positions/state rebuilt per call."""
        import torch
        N = self.N
        feats = np.zeros((N, 10), dtype=np.float32)
        feats[:, 0:2] = self.current_pos
        feats[:, 2:4] = self.sizes
        max_hpwl = max(self.macro_hpwl.max(), 1e-8)
        feats[:, 4] = self.macro_hpwl / max_hpwl
        # RUDY (skip if congestion_weight == 0 for speed)
        if self.congestion_weight > 0:
            macro_rudy = compute_per_macro_rudy(
                self.current_pos, self.sizes, self.nets, self.macro_nets)
            p95 = max(np.percentile(macro_rudy, 95), 1e-8)
            feats[:, 5] = np.clip(macro_rudy / p95, 0, 5.0) / 5.0
        feats[:, 6] = self.last_subset_mask
        feats[:, 7] = self.last_delta
        feats[:, 8] = min(self.stagnation_count / max(self.adapt_threshold, 1), 1.0)
        feats[:, 9] = self.window_fraction / max(self.max_window, 1e-8)
        return torch.from_numpy(feats).to(self._gnn_device)

    def _apply_candidate_result(
        self,
        new_positions: Optional[np.ndarray],
        subset: np.ndarray,
        strategy: str,
        pre_positions: np.ndarray,
    ) -> Dict:
        """
        Apply a CP-SAT candidate result: acceptance, state update, adaptation.

        Called by both heuristic step() and RL training. Single source of truth
        for state transitions.

        Returns dict with: accepted, improved, delta_cost, feasible
        """
        accepted = False
        improved = False
        delta_cost = 0.0
        feasible = new_positions is not None

        if not feasible:
            self.n_infeasible += 1
            if strategy in self.strategies:
                s_idx = self.strategies.index(strategy)
                self.alns.record_outcome(s_idx, 'rejected')
        else:
            # Incremental HPWL
            new_hpwl, new_net_hpwls = compute_incremental_hpwl(
                new_positions, self.nets, subset, self.net_hpwls, self.macro_nets)
            new_cost = self._compute_cost(new_positions, new_hpwl)
            delta_cost = new_cost - self.current_cost

            if self.accept(delta_cost):
                accepted = True
                self.n_accepted += 1
                self.current_pos = new_positions
                self.current_hpwl = new_hpwl
                self.current_cost = new_cost
                self.net_hpwls = new_net_hpwls
                self._update_macro_hpwl(subset)

                if new_cost < self.best_cost - 1e-8:
                    improved = True
                    self.n_improved += 1
                    self.best_pos = new_positions.copy()
                    self.best_hpwl = new_hpwl
                    self.best_cost = new_cost
                    self.stagnation_count = 0
                    self.window_fraction = max(
                        self.window_fraction * 0.8, self.min_window)
                    self.subset_size = max(
                        self.subset_size - 5, self.min_subset)
                    if self.sa_active:
                        self.sa_active = False
                        self.sa_temperature = self.sa_t_init
                    self.strategy_successes[strategy] = \
                        self.strategy_successes.get(strategy, 0) + 1

            # ALNS credit
            if strategy in self.strategies:
                s_idx = self.strategies.index(strategy)
                if improved:
                    self.alns.record_outcome(s_idx, 'new_best')
                elif accepted and delta_cost < -1e-8:
                    self.alns.record_outcome(s_idx, 'improved')
                elif accepted:
                    self.alns.record_outcome(s_idx, 'accepted')
                else:
                    self.alns.record_outcome(s_idx, 'rejected')

        # Stagnation / SA / adaptation
        if not improved:
            self.stagnation_count += 1
            if not self.sa_active and self.stagnation_count >= self.plateau_threshold:
                self.sa_active = True
                self.sa_temperature = self.sa_t_init
            if self.stagnation_count >= self.adapt_threshold:
                self.window_fraction = min(
                    self.window_fraction * 1.5, self.max_window)
                self.subset_size = min(
                    self.subset_size + 10, self.N // 2)
        if self.sa_active:
            self.sa_temperature *= self.sa_cooling

        # Update search state for GNN features
        self.last_subset_mask[:] = 0.0
        self.last_subset_mask[subset] = 1.0
        if accepted and new_positions is not None:
            delta = np.linalg.norm(new_positions - pre_positions, axis=1)
            self.last_delta = delta / max(delta.max(), 1e-8)
        else:
            self.last_delta[:] = 0.0

        return {
            'accepted': accepted,
            'improved': improved,
            'delta_cost': delta_cost,
            'feasible': feasible,
        }

    def select_strategy(self) -> str:
        """ALNS roulette wheel strategy selection."""
        idx = self.alns.select(self.rng)
        return self.strategies[idx]

    def get_neighborhood(self, strategy: str, k: int) -> np.ndarray:
        """Select k macros according to the given strategy."""
        k = min(k, self.N)

        if strategy == 'random':
            return self.rng.choice(self.N, size=k, replace=False)

        elif strategy == 'worst_hpwl':
            # Top-k macros by per-macro HPWL contribution
            indices = np.argsort(-self.macro_hpwl)[:k]
            return indices

        elif strategy == 'congestion':
            # Macros in highest-RUDY regions (net-based wire density)
            macro_rudy = compute_per_macro_rudy(
                self.current_pos, self.sizes, self.nets, self.macro_nets,
            )
            # Add noise for diversity
            max_val = macro_rudy.max()
            if max_val > 0:
                macro_rudy += self.rng.uniform(0, max_val * 0.1, size=self.N)
            indices = np.argsort(-macro_rudy)[:k]
            return indices

        elif strategy == 'connected':
            # BFS from random seed along netlist edges
            seed = self.rng.integers(0, self.N)
            visited = set([seed])
            frontier = [seed]
            while len(visited) < k and frontier:
                self.rng.shuffle(frontier)
                next_frontier = []
                for node in frontier:
                    for neighbor in self.adj[node]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            next_frontier.append(neighbor)
                            if len(visited) >= k:
                                break
                    if len(visited) >= k:
                        break
                frontier = next_frontier
            # If BFS didn't reach k, pad with random
            if len(visited) < k:
                remaining = list(set(range(self.N)) - visited)
                self.rng.shuffle(remaining)
                for r in remaining[:k - len(visited)]:
                    visited.add(r)
            return np.array(list(visited), dtype=int)

        elif strategy == 'learned' and self.model is not None:
            import torch
            node_features = self._build_gnn_features()
            positions_t = torch.from_numpy(
                self.current_pos).float().to(self._gnn_device)
            ct = self._cached_tensors
            with torch.no_grad():
                outputs = self.model(
                    node_features, positions_t, ct['sizes_t'],
                    ct['edge_index_t'], ct['edge_attr_t'],
                )
            subset_logits = outputs['subset_logits'].cpu().numpy()
            indices = np.argsort(-subset_logits)[:k]
            return indices

        else:
            return self.rng.choice(self.N, size=k, replace=False)

    def accept(self, delta_cost: float) -> bool:
        """Accept or reject a move based on cost change."""
        if delta_cost < -1e-8:
            return True

        if self.sa_active:
            if self.sa_temperature > 1e-10:
                prob = np.exp(-delta_cost / self.sa_temperature)
                return self.rng.random() < prob
            return False

        return False

    def step(self) -> Dict:
        """Run one LNS iteration."""
        self.iteration += 1
        t0 = time.time()

        # Select strategy and neighborhood
        strategy = self.select_strategy()
        subset = self.get_neighborhood(strategy, self.subset_size)
        self.strategy_attempts[strategy] += 1
        pre_positions = self.current_pos.copy()

        # Solve CP-SAT subproblem
        new_positions = solve_subset(
            self.current_pos, self.sizes, self.nets, subset,
            time_limit=self.cpsat_time_limit,
            window_fraction=self.window_fraction,
        )

        dt = time.time() - t0

        # Centralized state transition
        result = self._apply_candidate_result(
            new_positions, subset, strategy, pre_positions)

        return {
            'iteration': self.iteration,
            'strategy': strategy,
            **result,
            'current_hpwl': self.current_hpwl,
            'best_hpwl': self.best_hpwl,
            'current_cost': self.current_cost,
            'best_cost': self.best_cost,
            'subset_size': self.subset_size,
            'window_fraction': self.window_fraction,
            'sa_active': self.sa_active,
            'sa_temperature': self.sa_temperature if self.sa_active else 0.0,
            'stagnation': self.stagnation_count,
            'time': dt,
        }

    def _get_pool(self):
        """Lazy-create ProcessPoolExecutor with static data initializer."""
        if self._pool is None:
            from concurrent.futures import ProcessPoolExecutor
            # Each parallel candidate gets 1 CP-SAT worker to avoid oversubscription
            self._pool = ProcessPoolExecutor(
                max_workers=self.n_parallel_candidates,
                initializer=_pool_initializer,
                initargs=(self.sizes, self.nets),
            )
        return self._pool

    def step_parallel(self) -> Dict:
        """Run one LNS iteration with M parallel candidate evaluations.

        Samples M operators from ALNS weights (without replacement),
        generates M subsets, solves all in parallel, accepts the best
        via _apply_candidate_result(). Losers get 'rejected' ALNS credit.
        """
        self.iteration += 1
        t0 = time.time()

        M = self.n_parallel_candidates
        pool = self._get_pool()

        # Sample M operators from ALNS weights
        op_indices = self.alns.sample_multiple(self.rng, M)
        strategies = [self.strategies[idx] for idx in op_indices]
        subsets = [self.get_neighborhood(s, self.subset_size) for s in strategies]

        for s in strategies:
            self.strategy_attempts[s] += 1

        # Submit M solve_subset calls (each with num_workers=1)
        futures = []
        for subset in subsets:
            args = (self.current_pos, subset, self.cpsat_time_limit,
                    self.window_fraction, 1)
            futures.append(pool.submit(_pool_solve_subset, args))

        # Collect results
        results = [f.result() for f in futures]
        dt = time.time() - t0

        # Evaluate each result with incremental HPWL to find best candidate
        candidates = []
        for k, (new_pos, subset) in enumerate(zip(results, subsets)):
            if new_pos is None:
                candidates.append(None)
                continue
            new_hpwl, _ = compute_incremental_hpwl(
                new_pos, self.nets, subset, self.net_hpwls, self.macro_nets)
            new_cost = self._compute_cost(new_pos, new_hpwl)
            delta_cost = new_cost - self.current_cost
            candidates.append((new_pos, delta_cost, subset))

        # Pick best feasible candidate
        best_k = None
        best_delta = float('inf')
        for k, cand in enumerate(candidates):
            if cand is not None and cand[1] < best_delta:
                best_k = k
                best_delta = cand[1]

        winning_strategy = strategies[best_k] if best_k is not None else strategies[0]
        pre_positions = self.current_pos.copy()

        # Apply best candidate through centralized state transition
        if best_k is not None:
            best_pos, _, best_subset = candidates[best_k]
            result = self._apply_candidate_result(
                best_pos, best_subset, winning_strategy, pre_positions)
        else:
            # All infeasible — apply None result
            result = self._apply_candidate_result(
                None, subsets[0], winning_strategy, pre_positions)

        # ALNS credit for losers (not handled by _apply_candidate_result)
        for k in range(M):
            if k != best_k:
                s_idx = int(op_indices[k])
                if candidates[k] is None:
                    self.n_infeasible += 1
                self.alns.record_outcome(s_idx, 'rejected')

        return {
            'iteration': self.iteration,
            'strategy': winning_strategy,
            **result,
            'current_hpwl': self.current_hpwl,
            'best_hpwl': self.best_hpwl,
            'current_cost': self.current_cost,
            'best_cost': self.best_cost,
            'subset_size': self.subset_size,
            'window_fraction': self.window_fraction,
            'sa_active': self.sa_active,
            'sa_temperature': self.sa_temperature if self.sa_active else 0.0,
            'stagnation': self.stagnation_count,
            'time': dt,
            'n_candidates': M,
        }

    def solve(
        self,
        n_iterations: int = 500,
        log_every: int = 10,
        verbose: bool = True,
    ) -> Dict:
        """
        Run full LNS optimization.

        Returns:
            dict with best_positions, best_hpwl, best_cost, history
        """
        history = []

        if verbose:
            print(f"\nLNS: {self.N} macros, {len(self.nets)} nets")
            print(f"  Initial HPWL: {self.current_hpwl:.4f}")
            overlap, n_ov = check_overlap(self.current_pos, self.sizes)
            boundary = check_boundary(self.current_pos, self.sizes)
            print(f"  Initial overlap: {overlap:.6f} ({n_ov} pairs)")
            print(f"  Initial boundary: {boundary:.6f}")
            print(f"  subset_size={self.subset_size}, window={self.window_fraction:.2f}")
            print()

        use_parallel = self.n_parallel_candidates > 1

        for i in range(n_iterations):
            metrics = self.step_parallel() if use_parallel else self.step()
            history.append(metrics)

            if verbose and (i % log_every == 0 or i == n_iterations - 1 or metrics['improved']):
                mode = "SA" if metrics['sa_active'] else "GD"
                status = "IMPROVED" if metrics['improved'] else (
                    "accepted" if metrics['accepted'] else "rejected")
                print(
                    f"  [{i+1:4d}/{n_iterations}] [{mode}] "
                    f"best={metrics['best_hpwl']:.4f} "
                    f"cur={metrics['current_hpwl']:.4f} "
                    f"delta={metrics['delta_cost']:+.4f} "
                    f"strat={metrics['strategy']:12s} "
                    f"{status:8s} "
                    f"w={metrics['window_fraction']:.2f} "
                    f"k={metrics['subset_size']:3d} "
                    f"stag={metrics['stagnation']:3d} "
                    f"({metrics['time']:.1f}s)"
                )

        # Compute RUDY stats on best placement
        rudy_stats = compute_rudy_np(self.best_pos, self.sizes, self.nets)

        # Final summary
        if verbose:
            print(f"\nLNS complete after {n_iterations} iterations:")
            print(f"  Best HPWL: {self.best_hpwl:.4f}")
            overlap, n_ov = check_overlap(self.best_pos, self.sizes)
            boundary = check_boundary(self.best_pos, self.sizes)
            print(f"  Overlap: {overlap:.6f} ({n_ov} pairs)")
            print(f"  Boundary: {boundary:.6f}")
            print(f"  RUDY: max={rudy_stats['rudy_max']:.4f} "
                  f"p95={rudy_stats['rudy_p95']:.4f} "
                  f"p99={rudy_stats['rudy_p99']:.4f} "
                  f"overflow={rudy_stats['overflow_sum']:.4f}")
            print(f"  Accepted: {self.n_accepted}/{n_iterations} "
                  f"({self.n_accepted/max(n_iterations,1)*100:.1f}%)")
            print(f"  Improved: {self.n_improved}")
            print(f"  Infeasible: {self.n_infeasible}")
            print(f"  Strategy success rates:")
            for s in self.strategies:
                attempts = self.strategy_attempts[s]
                successes = self.strategy_successes[s]
                rate = successes / max(attempts, 1) * 100
                print(f"    {s:15s}: {successes}/{attempts} ({rate:.1f}%)")
            print(f"  ALNS weights:")
            alns_w = self.alns.get_weights_dict(self.strategies)
            for s, w in alns_w.items():
                print(f"    {s:15s}: {w:.4f}")
            if use_parallel:
                print(f"  Parallel candidates per iteration: {self.n_parallel_candidates}")

        # Clean up pool if created
        if self._pool is not None:
            self._pool.shutdown(wait=False)
            self._pool = None

        return {
            'best_positions': self.best_pos,
            'best_hpwl': self.best_hpwl,
            'best_cost': self.best_cost,
            'rudy_stats': rudy_stats,
            'history': history,
        }
