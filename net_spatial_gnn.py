"""
NetSpatialGNN — Dual-Stream Topology+Spatial GNN for Chip Placement

Processes two complementary graph structures simultaneously:
1. Topology stream: netlist connectivity (fixed star-decomposed edges with pin offsets)
2. Spatial stream: k-NN proximity graph on current macro positions (size-aware edges)

Cross-stream fusion each layer lets the model reason about both connectivity and geometry.

Output heads:
    displacement_pred: (V, 2) predicted position delta for CP-SAT warm-start hints
    heatmap_logits:    (V, G²) per-macro grid distribution (Stage 2 REINFORCE)
    value:             (n_graphs,) per-graph scalar (Stage 2 baseline)

Ablation via mode={'topology_only', 'spatial_only', 'dual'}.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from gnn_layers import (
    LinearMessagePassingLayer, ReluMLP, ValueMLP, scatter_sum,
)


def build_spatial_graph(
    positions: torch.Tensor,
    sizes: torch.Tensor,
    k: int = 8,
) -> tuple:
    """
    Build k-NN spatial graph from macro positions with size-aware edge features.

    Args:
        positions: (V, 2) macro center coordinates
        sizes: (V, 2) macro widths and heights
        k: number of nearest neighbors per macro

    Returns:
        edge_index: (2, V*k) directed edges (src → dst)
        edge_attr: (V*k, 6) [dx, dy, dist, gap_x, gap_y, size_ratio]
    """
    V = positions.size(0)
    k = min(k, V - 1)

    # Pairwise displacement and distance
    diffs = positions.unsqueeze(1) - positions.unsqueeze(0)  # (V, V, 2)
    dists = torch.norm(diffs, dim=-1)  # (V, V)
    dists.fill_diagonal_(float('inf'))

    # Top-k nearest neighbors per node
    _, knn_idx = torch.topk(dists, k=k, largest=False)  # (V, k)

    # Build edge_index
    src = torch.arange(V, device=positions.device).unsqueeze(1).expand(-1, k).reshape(-1)
    dst = knn_idx.reshape(-1)
    edge_index = torch.stack([src, dst])  # (2, V*k)

    # Edge features
    dx = diffs[src, dst, 0]
    dy = diffs[src, dst, 1]
    dist = dists[src, dst]

    # Size-aware gap: positive = space between, zero = touching/overlapping
    half_w_src = sizes[src, 0] / 2
    half_h_src = sizes[src, 1] / 2
    half_w_dst = sizes[dst, 0] / 2
    half_h_dst = sizes[dst, 1] / 2
    gap_x = torch.clamp(dx.abs() - half_w_src - half_w_dst, min=0.0)
    gap_y = torch.clamp(dy.abs() - half_h_src - half_h_dst, min=0.0)

    # Relative size ratio (area)
    area_src = sizes[src, 0] * sizes[src, 1]
    area_dst = sizes[dst, 0] * sizes[dst, 1]
    size_ratio = area_dst / (area_src + 1e-8)

    edge_attr = torch.stack([dx, dy, dist, gap_x, gap_y, size_ratio], dim=-1)

    return edge_index, edge_attr


class NetSpatialGNN(nn.Module):
    """
    Dual-stream GNN for chip placement: topology + spatial.

    Args:
        node_input_dim: Input node features dimension (default 6:
            pos_x, pos_y, size_w, size_h, hpwl_norm, congestion_norm)
        topo_edge_dim: Topology edge features (4: pin offsets)
        spatial_edge_dim: Spatial edge features (6: dx, dy, dist, gap_x, gap_y, size_ratio)
        hidden_dim: Hidden dimension for both streams
        n_layers: Number of message passing layers
        grid_size: G for G×G heatmap output
        k_spatial: Number of nearest neighbors for spatial graph
        mode: 'topology_only' | 'spatial_only' | 'dual'
    """

    def __init__(
        self,
        node_input_dim: int = 6,
        topo_edge_dim: int = 4,
        spatial_edge_dim: int = 6,
        hidden_dim: int = 64,
        n_layers: int = 5,
        grid_size: int = 32,
        k_spatial: int = 8,
        mode: str = 'dual',
        mean_aggr: bool = False,
    ):
        super().__init__()
        assert mode in ('topology_only', 'spatial_only', 'dual')
        self.mode = mode
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.grid_size = grid_size
        self.k_spatial = k_spatial

        # Shared node encoder (both streams start from same node embedding)
        self.node_encoder = ReluMLP([node_input_dim, hidden_dim, hidden_dim])

        # Topology stream
        if mode in ('topology_only', 'dual'):
            self.topo_edge_encoder = ReluMLP([topo_edge_dim, hidden_dim, hidden_dim])
            self.topo_layers = nn.ModuleList([
                LinearMessagePassingLayer(
                    node_dim=hidden_dim, edge_dim=hidden_dim,
                    message_dim=hidden_dim, mean_aggr=mean_aggr,
                )
                for _ in range(n_layers)
            ])

        # Spatial stream
        if mode in ('spatial_only', 'dual'):
            self.spatial_edge_encoder = ReluMLP([spatial_edge_dim, hidden_dim, hidden_dim])
            self.spatial_layers = nn.ModuleList([
                LinearMessagePassingLayer(
                    node_dim=hidden_dim, edge_dim=hidden_dim,
                    message_dim=hidden_dim, mean_aggr=mean_aggr,
                )
                for _ in range(n_layers)
            ])

        # Fusion layers (dual mode only)
        if mode == 'dual':
            self.fusion_projs = nn.ModuleList([
                nn.Linear(hidden_dim * 2, hidden_dim)
                for _ in range(n_layers)
            ])
            self.fusion_norms = nn.ModuleList([
                nn.LayerNorm(hidden_dim)
                for _ in range(n_layers)
            ])
            for proj in self.fusion_projs:
                nn.init.kaiming_normal_(proj.weight, nonlinearity='relu')
                nn.init.zeros_(proj.bias)

        # Displacement prediction head (Stage 1: supervised warm-start)
        self.displacement_head = nn.Sequential(
            ReluMLP([hidden_dim, hidden_dim]),
            nn.Linear(hidden_dim, 2),
        )
        # Initialize near zero so initial predictions are small displacements
        nn.init.xavier_normal_(self.displacement_head[-1].weight, gain=0.01)
        nn.init.zeros_(self.displacement_head[-1].bias)

        # Heatmap head (Stage 2: REINFORCE, not trained in Stage 1)
        g2 = grid_size * grid_size
        self.heatmap_head = nn.Sequential(
            ReluMLP([hidden_dim, hidden_dim]),
            nn.Linear(hidden_dim, g2),
        )
        nn.init.xavier_normal_(self.heatmap_head[-1].weight, gain=0.1)
        nn.init.zeros_(self.heatmap_head[-1].bias)

        # Subset selection head (per-macro logit for destroy operator)
        self.subset_head = nn.Sequential(
            ReluMLP([hidden_dim, hidden_dim]),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.xavier_normal_(self.subset_head[-1].weight, gain=0.1)
        nn.init.zeros_(self.subset_head[-1].bias)

        # Trust radius head (per-macro scalar in (0, 1) — scales window_fraction)
        # Output: trust_pred[i] * window_fraction = per-macro search window
        # Near-zero init so initial trust radii start near 0.5 (mid-range)
        self.trust_radius_head = nn.Sequential(
            ReluMLP([hidden_dim, hidden_dim // 2]),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        nn.init.xavier_normal_(self.trust_radius_head[-2].weight, gain=0.1)
        nn.init.zeros_(self.trust_radius_head[-2].bias)

        # Value head (REINFORCE baseline)
        self.value_mlp = ValueMLP([hidden_dim, 120, 64, 1])

    def forward(
        self,
        node_features: torch.Tensor,
        positions: torch.Tensor,
        sizes: torch.Tensor,
        topo_edge_index: torch.Tensor,
        topo_edge_attr: torch.Tensor,
        node_graph_idx: Optional[torch.Tensor] = None,
        n_graphs: int = 1,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            node_features: (V, node_input_dim) macro features
            positions: (V, 2) current macro centers (for spatial graph construction)
            sizes: (V, 2) macro sizes (for spatial edge features)
            topo_edge_index: (2, E_topo) netlist edges
            topo_edge_attr: (E_topo, topo_edge_dim) pin offsets
            node_graph_idx: (V,) graph assignment for batching
            n_graphs: number of graphs in batch

        Returns:
            dict with displacement_pred, heatmap_logits, value, node_embeddings
        """
        V = node_features.size(0)

        # Encode nodes (shared)
        h = self.node_encoder(node_features)  # (V, hidden_dim)

        # Encode topology edges (if needed)
        if self.mode in ('topology_only', 'dual'):
            topo_edges_enc = self.topo_edge_encoder(topo_edge_attr)

        # Build and encode spatial graph (if needed)
        if self.mode in ('spatial_only', 'dual'):
            spatial_edge_index, spatial_edge_attr = build_spatial_graph(
                positions, sizes, k=self.k_spatial,
            )
            spatial_edges_enc = self.spatial_edge_encoder(spatial_edge_attr)

        # Message passing layers with cross-stream fusion
        for l in range(self.n_layers):
            if self.mode == 'topology_only':
                h = self.topo_layers[l](h, topo_edge_index, topo_edges_enc)

            elif self.mode == 'spatial_only':
                h = self.spatial_layers[l](h, spatial_edge_index, spatial_edges_enc)

            elif self.mode == 'dual':
                h_topo = self.topo_layers[l](h, topo_edge_index, topo_edges_enc)
                h_spatial = self.spatial_layers[l](h, spatial_edge_index, spatial_edges_enc)

                # Fusion: project concatenation + dual residual
                h_cat = torch.cat([h_topo, h_spatial], dim=-1)  # (V, 2*hidden)
                h_fused = self.fusion_projs[l](h_cat)            # (V, hidden)
                h = self.fusion_norms[l](h_fused + h_topo + h_spatial)

        # Output heads
        displacement_pred = self.displacement_head(h)  # (V, 2)

        heatmap_logits = self.heatmap_head(h)  # (V, G²)

        # Value: aggregate node embeddings per graph
        if node_graph_idx is None:
            node_graph_idx = torch.zeros(V, dtype=torch.long, device=h.device)
        graph_embed = scatter_sum(h, node_graph_idx, dim=0, dim_size=n_graphs)
        nodes_per_graph = scatter_sum(
            torch.ones(V, 1, device=h.device),
            node_graph_idx, dim=0, dim_size=n_graphs,
        )
        graph_embed = graph_embed / torch.sqrt(torch.clamp(nodes_per_graph, min=1.0))
        value = self.value_mlp(graph_embed).squeeze(-1)  # (n_graphs,)

        subset_logits = self.subset_head(h).squeeze(-1)  # (V,)

        trust_radius_pred = self.trust_radius_head(h).squeeze(-1)  # (V,) in (0, 1)

        return {
            'displacement_pred': displacement_pred,
            'heatmap_logits': heatmap_logits,
            'value': value,
            'node_embeddings': h,
            'subset_logits': subset_logits,
            'trust_radius_pred': trust_radius_pred,
        }


    @staticmethod
    def select_subset(
        logits: torch.Tensor,
        k: int,
        temperature: float = 1.0,
        explore: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gumbel-top-k subset selection with Plackett-Luce log-probability.

        Args:
            logits: (V,) per-macro logits from subset_head
            k: number of macros to select
            temperature: Gumbel noise temperature (lower = more greedy)
            explore: if True, add Gumbel noise for exploration

        Returns:
            indices: (k,) selected macro indices
            log_prob: scalar, log P(ordered sequence) under Plackett-Luce
        """
        k = min(k, logits.size(0))

        if explore:
            u = torch.rand_like(logits).clamp(1e-8, 1 - 1e-8)
            gumbel = -torch.log(-torch.log(u))
            perturbed = logits / temperature + gumbel
        else:
            perturbed = logits

        _, indices = torch.topk(perturbed, k)

        # Plackett-Luce log-probability for the ORDERED sampled sequence
        # log P = sum_{i=0}^{k-1} [ logits[indices[i]] - logsumexp(logits[remaining]) ]
        # Avoid in-place ops: track removed indices as a set, rebuild mask each step
        log_prob = torch.zeros(1, device=logits.device)
        removed = set()
        V = logits.size(0)
        for i in range(k):
            idx = indices[i].item()
            remaining_idx = [j for j in range(V) if j not in removed]
            remaining_logits = logits[remaining_idx]
            lse = torch.logsumexp(remaining_logits, dim=0)
            log_prob = log_prob + logits[idx] - lse
            removed.add(idx)

        return indices, log_prob


def _toy_test():
    """Quick sanity check: forward/backward pass, all 3 modes, correct shapes."""
    print("NetSpatialGNN toy test")
    print("=" * 50)

    V, E = 8, 16
    device = 'cpu'

    node_features = torch.randn(V, 6, device=device)
    positions = torch.randn(V, 2, device=device)
    sizes = torch.rand(V, 2, device=device) * 0.3 + 0.05
    edge_index = torch.randint(0, V, (2, E), device=device)
    edge_attr = torch.randn(E, 4, device=device)

    G = 32

    for mode in ['topology_only', 'spatial_only', 'dual']:
        model = NetSpatialGNN(mode=mode, grid_size=G, k_spatial=4)
        model.to(device)

        out = model(node_features, positions, sizes, edge_index, edge_attr)

        assert out['displacement_pred'].shape == (V, 2), \
            f"{mode}: displacement_pred shape {out['displacement_pred'].shape}"
        assert out['heatmap_logits'].shape == (V, G * G), \
            f"{mode}: heatmap_logits shape {out['heatmap_logits'].shape}"
        assert out['value'].shape == (1,), \
            f"{mode}: value shape {out['value'].shape}"
        assert out['node_embeddings'].shape == (V, 64), \
            f"{mode}: node_embeddings shape {out['node_embeddings'].shape}"
        assert out['subset_logits'].shape == (V,), \
            f"{mode}: subset_logits shape {out['subset_logits'].shape}"
        assert out['trust_radius_pred'].shape == (V,), \
            f"{mode}: trust_radius_pred shape {out['trust_radius_pred'].shape}"
        assert out['trust_radius_pred'].min() >= 0 and out['trust_radius_pred'].max() <= 1, \
            f"{mode}: trust_radius_pred must be in (0, 1)"

        # Backward pass (include all heads)
        loss = (out['displacement_pred'].sum() + out['heatmap_logits'].sum()
                + out['value'].sum() + out['subset_logits'].sum()
                + out['trust_radius_pred'].sum())
        loss.backward()

        n_params = sum(p.numel() for p in model.parameters())
        print(f"  {mode:16s}: OK  ({n_params:,} params)")

    # Test select_subset (Gumbel-top-k + Plackett-Luce)
    model_ss = NetSpatialGNN(mode='dual', grid_size=G, k_spatial=4)
    out_ss = model_ss(node_features, positions, sizes, edge_index, edge_attr)
    logits = out_ss['subset_logits']
    k_sel = 3
    indices, log_prob = NetSpatialGNN.select_subset(logits, k=k_sel, explore=True)
    assert indices.shape == (k_sel,), f"select_subset indices shape {indices.shape}"
    assert log_prob.shape == (1,), f"select_subset log_prob shape {log_prob.shape}"
    assert log_prob.item() < 0, "log_prob should be negative"
    # Verify gradient flows through log_prob
    log_prob.backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in model_ss.subset_head.parameters())
    assert has_grad, "select_subset gradient must flow to subset_head"
    print(f"  {'select_subset':16s}: OK  (k={k_sel}, log_prob={log_prob.item():.4f})")

    # Test with V=3 (edge case: k_spatial > V-1)
    V_small = 3
    nf_small = torch.randn(V_small, 6)
    pos_small = torch.randn(V_small, 2)
    sz_small = torch.rand(V_small, 2) * 0.3 + 0.05
    ei_small = torch.tensor([[0, 1, 2], [1, 2, 0]])
    ea_small = torch.randn(3, 4)
    model_small = NetSpatialGNN(mode='dual', k_spatial=8)
    out_small = model_small(nf_small, pos_small, sz_small, ei_small, ea_small)
    assert out_small['displacement_pred'].shape == (V_small, 2)
    print(f"  {'small graph':16s}: OK  (V={V_small}, k_spatial clamped)")

    print("\nAll tests passed!")


if __name__ == '__main__':
    _toy_test()
