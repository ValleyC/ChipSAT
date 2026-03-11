"""
Greedy Sequential Placer with Constraint Masking (Vectorized)

GLOP-inspired sequential decoder:
  - Components placed one-by-one, largest first
  - Hard constraints enforced via masking (boundary + overlap)
  - Decode-time occupancy conditioning via gated OccMLP
  - Curriculum legality: repair early, terminate+penalty later
  - Per-graph validity tracking (not per-sample)

All n_samples are processed in parallel within each graph,
giving ~n_samples× speedup vs the sequential version.
"""

import torch
import torch.distributions as dist
from typing import Tuple, Optional


class GreedyPlacer:
    """Sequential placement with constraint masking and occupancy conditioning."""

    def __init__(
        self,
        grid_size: int = 32,
        canvas_min: float = -1.0,
        canvas_max: float = 1.0,
    ):
        self.grid_size = grid_size
        self.canvas_min = canvas_min
        self.canvas_max = canvas_max
        self.cell_size = (canvas_max - canvas_min) / grid_size

        # Precompute grid cell centers: (G², 2)
        g = grid_size
        cx = torch.linspace(
            canvas_min + self.cell_size / 2,
            canvas_max - self.cell_size / 2,
            g,
        )
        cy = torch.linspace(
            canvas_min + self.cell_size / 2,
            canvas_max - self.cell_size / 2,
            g,
        )
        # grid_y[i,j] = cy[i], grid_x[i,j] = cx[j]
        # Flattened: cell(i,j) = index i*G + j → center (cx[j], cy[i])
        grid_y, grid_x = torch.meshgrid(cy, cx, indexing='ij')
        self.grid_centers = torch.stack(
            [grid_x.flatten(), grid_y.flatten()], dim=-1
        )  # (G², 2)

    def to(self, device):
        """Move grid centers to device."""
        self.grid_centers = self.grid_centers.to(device)
        return self

    def place_components(
        self,
        model,
        logits: torch.Tensor,
        component_sizes: torch.Tensor,
        node_graph_idx: torch.Tensor,
        n_graphs: int,
        n_samples: int = 1,
        greedy: bool = False,
        repair_mode: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Vectorized sequential greedy placement with constraint masking.

        Batches all n_samples within each graph in parallel:
        - Single batched OccMLP call per component step (not per sample)
        - Vectorized mask computation across samples
        - Batched Categorical sampling

        Args:
            model: ChipHeatmapModel (for compute_occ_bias)
            logits: (N, G²) raw heatmap logits from GNN
            component_sizes: (N, 2) width, height per component
            node_graph_idx: (N,) graph assignment
            n_graphs: number of graphs in batch
            n_samples: number of independent placement samples
            greedy: if True, use argmax instead of sampling
            repair_mode: if True, repair impossible placements;
                         if False, terminate + flag invalid

        Returns:
            positions: (N, n_samples, 2) placed center positions
            log_probs: (n_graphs, n_samples) cumulative log-prob per graph
            entropies: (n_graphs, n_samples) cumulative entropy per graph
            valid_flags: (n_graphs, n_samples) bool — True if no termination
            repaired_flags: (n_graphs, n_samples) bool — True if any step needed repair
        """
        device = logits.device
        N = logits.shape[0]
        G = self.grid_size
        G2 = G * G
        grid_centers = self.grid_centers.to(device)

        positions = torch.zeros(N, n_samples, 2, device=device)
        log_probs = torch.zeros(n_graphs, n_samples, device=device)
        entropies = torch.zeros(n_graphs, n_samples, device=device)
        valid_flags = torch.ones(n_graphs, n_samples, dtype=torch.bool, device=device)
        repaired_flags = torch.zeros(n_graphs, n_samples, dtype=torch.bool, device=device)

        # Process each graph (component order is sequential, but samples are parallel)
        for g in range(n_graphs):
            node_mask = (node_graph_idx == g)
            g_indices = torch.where(node_mask)[0]
            g_logits = logits[g_indices]        # (Vg, G²)
            g_sizes = component_sizes[g_indices]  # (Vg, 2)
            Vg = g_indices.shape[0]

            # Sort by area (largest first) — deterministic ordering
            areas = g_sizes[:, 0] * g_sizes[:, 1]
            sort_order = torch.argsort(areas, descending=True)

            # --- Batched state across all S samples ---
            S = n_samples
            occupancy_grids = torch.zeros(S, G, G, device=device)
            placed_pos = torch.zeros(S, Vg, 2, device=device)
            placed_sizes = torch.zeros(Vg, 2, device=device)
            n_placed = 0
            sample_log_probs = torch.zeros(S, device=device)
            sample_entropies = torch.zeros(S, device=device)
            sample_valid = torch.ones(S, dtype=torch.bool, device=device)

            for local_idx in sort_order:
                comp_logits = g_logits[local_idx]   # (G²,)
                comp_size = g_sizes[local_idx]       # (2,)
                global_idx = g_indices[local_idx]

                # If all samples invalidated, just place at origin
                if not sample_valid.any():
                    positions[global_idx, :] = 0.0
                    placed_sizes[n_placed] = comp_size
                    n_placed += 1
                    continue

                # 1. Batched OccMLP: (S, G²) → (S, G²)
                occ_bias = model.compute_occ_bias(
                    occupancy_grids.reshape(S, -1).clone()
                )  # (S, G²)

                # 2. Conditioned logits: broadcast (G²,) → (S, G²)
                conditioned = comp_logits.unsqueeze(0) + occ_bias  # (S, G²)

                # 3. Boundary mask (identical for all samples): (G²,)
                boundary_mask = self._compute_boundary_mask(
                    comp_size, grid_centers, device)

                # 4. Overlap mask (per-sample): (S, G²)
                overlap_mask = self._compute_overlap_mask_batched(
                    comp_size, placed_pos, placed_sizes,
                    n_placed, grid_centers, S, device)

                # 5. Combined valid mask: (S, G²)
                valid_mask = boundary_mask.unsqueeze(0) & overlap_mask
                has_valid = valid_mask.any(dim=1)  # (S,)

                # Initialize step outputs
                cell_indices = torch.zeros(S, dtype=torch.long, device=device)
                step_lp = torch.zeros(S, device=device)
                step_ent = torch.zeros(S, device=device)

                # --- Samples with valid cells: masked sampling ---
                can_sample = has_valid & sample_valid
                if can_sample.any():
                    s_logits = conditioned[can_sample].clone()  # (V, G²)
                    s_mask = valid_mask[can_sample]              # (V, G²)
                    s_logits[~s_mask] = float('-inf')

                    cat = dist.Categorical(logits=s_logits)
                    if greedy:
                        s_cells = s_logits.argmax(dim=-1)
                    else:
                        s_cells = cat.sample()

                    cell_indices[can_sample] = s_cells
                    step_lp[can_sample] = cat.log_prob(s_cells)
                    step_ent[can_sample] = cat.entropy()

                # --- Samples needing repair or invalidation ---
                needs_fix = ~has_valid & sample_valid
                if needs_fix.any():
                    if repair_mode:
                        repair_cells = self._repair_batched(
                            comp_size, placed_pos[needs_fix],
                            placed_sizes, n_placed,
                            boundary_mask, grid_centers, device)
                        cell_indices[needs_fix] = repair_cells
                        repaired_flags[g, needs_fix] = True
                        # No log_prob for repair (stays 0)
                    else:
                        sample_valid[needs_fix] = False
                        valid_flags[g, needs_fix] = False

                # 6. Get positions from grid centers
                pos_batch = grid_centers[cell_indices]  # (S, 2)

                # Invalid samples → origin
                invalid = ~sample_valid
                if invalid.any():
                    pos_batch[invalid] = 0.0

                positions[global_idx] = pos_batch

                # 7. Accumulate log_probs and entropies
                sample_log_probs = sample_log_probs + step_lp
                sample_entropies = sample_entropies + step_ent

                # 8. Update placed tracking and occupancy
                placed_pos[:, n_placed] = pos_batch
                placed_sizes[n_placed] = comp_size
                n_placed += 1

                self._update_occupancy_batched(
                    occupancy_grids, pos_batch, comp_size,
                    grid_centers, G)

            log_probs[g] = sample_log_probs
            entropies[g] = sample_entropies

        return positions, log_probs, entropies, valid_flags, repaired_flags

    def _compute_boundary_mask(
        self,
        comp_size: torch.Tensor,
        grid_centers: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Which grid cells keep component within canvas bounds.

        Args:
            comp_size: (2,) component width, height

        Returns:
            mask: (G²,) bool — True if cell is valid
        """
        w, h = comp_size[0], comp_size[1]
        cx = grid_centers[:, 0]
        cy = grid_centers[:, 1]

        valid_x = (cx - w / 2 >= self.canvas_min) & (cx + w / 2 <= self.canvas_max)
        valid_y = (cy - h / 2 >= self.canvas_min) & (cy + h / 2 <= self.canvas_max)
        return valid_x & valid_y

    def _compute_overlap_mask_batched(
        self,
        comp_size: torch.Tensor,
        placed_pos: torch.Tensor,
        placed_sizes: torch.Tensor,
        n_placed: int,
        grid_centers: torch.Tensor,
        n_samples: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Fully batched overlap mask across samples and placed components.

        Single kernel launch: broadcasts (S, P, G²) to check all placed
        components simultaneously, then reduces with any(dim=1).

        Args:
            comp_size: (2,) current component width, height
            placed_pos: (S, Vg, 2) placed positions (only [:, :n_placed] valid)
            placed_sizes: (Vg, 2) placed sizes (only [:n_placed] valid)
            n_placed: number of already-placed components
            grid_centers: (G², 2)
            n_samples: S
            device: torch.device

        Returns:
            mask: (S, G²) bool — True if no overlap with any placed component
        """
        G2 = grid_centers.shape[0]
        if n_placed == 0:
            return torch.ones(n_samples, G2, dtype=torch.bool, device=device)

        w, h = comp_size[0], comp_size[1]

        # Placed positions/sizes: (S, P, 1) and (1, P, 1)
        pp_x = placed_pos[:, :n_placed, 0:1]           # (S, P, 1)
        pp_y = placed_pos[:, :n_placed, 1:2]           # (S, P, 1)
        pw = placed_sizes[:n_placed, 0].view(1, -1, 1)  # (1, P, 1)
        ph = placed_sizes[:n_placed, 1].view(1, -1, 1)  # (1, P, 1)

        # Grid centers: (1, 1, G²)
        cx = grid_centers[:, 0].view(1, 1, -1)
        cy = grid_centers[:, 1].view(1, 1, -1)

        # Broadcast overlap check: (S, P, G²)
        has_ov_x = torch.abs(cx - pp_x) < (w + pw) / 2
        has_ov_y = torch.abs(cy - pp_y) < (h + ph) / 2

        # Any placed component overlaps → (S, G²)
        any_overlap = (has_ov_x & has_ov_y).any(dim=1)
        return ~any_overlap

    def _repair_batched(
        self,
        comp_size: torch.Tensor,
        placed_pos_subset: torch.Tensor,
        placed_sizes: torch.Tensor,
        n_placed: int,
        boundary_mask: torch.Tensor,
        grid_centers: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Batched repair for samples with no valid cells.

        Among boundary-valid cells, pick the one with minimum total overlap
        area. If no boundary-valid cells, pick global minimum.

        Args:
            comp_size: (2,) component width, height
            placed_pos_subset: (R, Vg, 2) placed positions for repair samples
            placed_sizes: (Vg, 2) — only [:n_placed] valid
            n_placed: int
            boundary_mask: (G²,) bool
            grid_centers: (G², 2)
            device: torch.device

        Returns:
            cell_indices: (R,) long — repair cell indices
        """
        R = placed_pos_subset.shape[0]
        G2 = grid_centers.shape[0]
        w, h = comp_size[0], comp_size[1]

        if n_placed == 0:
            # No overlap possible, pick first boundary-valid cell
            if boundary_mask.any():
                return boundary_mask.long().argmax().expand(R)
            return torch.zeros(R, dtype=torch.long, device=device)

        # Placed positions: (R, P, 1), sizes: (1, P, 1)
        pp_x = placed_pos_subset[:, :n_placed, 0:1]
        pp_y = placed_pos_subset[:, :n_placed, 1:2]
        pw = placed_sizes[:n_placed, 0].view(1, -1, 1)
        ph = placed_sizes[:n_placed, 1].view(1, -1, 1)

        # Grid centers: (1, 1, G²)
        cx = grid_centers[:, 0].view(1, 1, -1)
        cy = grid_centers[:, 1].view(1, 1, -1)

        # Per-placed overlap area: (R, P, G²)
        ov_x = torch.clamp((w + pw) / 2 - torch.abs(cx - pp_x), min=0.0)
        ov_y = torch.clamp((h + ph) / 2 - torch.abs(cy - pp_y), min=0.0)

        # Sum over placed components: (R, G²)
        total_overlap = (ov_x * ov_y).sum(dim=1)

        if boundary_mask.any():
            total_overlap[:, ~boundary_mask] = float('inf')

        return total_overlap.argmin(dim=1)

    def _update_occupancy_batched(
        self,
        occupancy_grids: torch.Tensor,
        positions: torch.Tensor,
        comp_size: torch.Tensor,
        grid_centers: torch.Tensor,
        G: int,
    ):
        """Update occupancy grids for all samples simultaneously.

        Area-weighted: each cell gets the fraction of its area covered
        by the placed component.

        Args:
            occupancy_grids: (S, G, G) — modified in-place
            positions: (S, 2) placed center positions
            comp_size: (2,) component width, height
            grid_centers: (G², 2)
            G: grid size
        """
        w, h = comp_size[0], comp_size[1]
        cell_size = self.cell_size

        # Component bboxes: (S, 1, 1)
        comp_xmin = (positions[:, 0] - w / 2).view(-1, 1, 1)
        comp_xmax = (positions[:, 0] + w / 2).view(-1, 1, 1)
        comp_ymin = (positions[:, 1] - h / 2).view(-1, 1, 1)
        comp_ymax = (positions[:, 1] + h / 2).view(-1, 1, 1)

        # Cell bboxes: (1, G, G)
        cell_cx = grid_centers[:, 0].view(1, G, G)
        cell_cy = grid_centers[:, 1].view(1, G, G)
        cell_half = cell_size / 2
        cell_xmin = cell_cx - cell_half
        cell_xmax = cell_cx + cell_half
        cell_ymin = cell_cy - cell_half
        cell_ymax = cell_cy + cell_half

        # Overlap area: (S, G, G)
        ov_x = torch.clamp(
            torch.minimum(comp_xmax, cell_xmax) -
            torch.maximum(comp_xmin, cell_xmin),
            min=0.0)
        ov_y = torch.clamp(
            torch.minimum(comp_ymax, cell_ymax) -
            torch.maximum(comp_ymin, cell_ymin),
            min=0.0)

        occupancy_grids += (ov_x * ov_y) / (cell_size * cell_size)


def positions_to_cell_indices(
    positions: torch.Tensor,
    grid_size: int,
    canvas_min: float = -1.0,
    canvas_max: float = 1.0,
) -> torch.Tensor:
    """Convert continuous positions to nearest grid cell indices.

    Used for imitation pretraining: reference positions → cell labels.

    Args:
        positions: (N, 2) component center positions
        grid_size: G for G×G grid

    Returns:
        cell_indices: (N,) long — flattened grid cell index per component
    """
    cell_size = (canvas_max - canvas_min) / grid_size

    # Map to grid coordinates [0, G)
    gx = ((positions[:, 0] - canvas_min) / cell_size).clamp(0, grid_size - 1e-6).long()
    gy = ((positions[:, 1] - canvas_min) / cell_size).clamp(0, grid_size - 1e-6).long()

    # Flattened index: row * G + col
    return gy * grid_size + gx
