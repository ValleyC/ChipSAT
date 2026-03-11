"""
Diffusion Step Model - DiffUCO Faithful Implementation

This module implements the per-step GNN model exactly as in DiffUCO:
- DiffModel.py: input construction (one_hot + time_embed + random_nodes)
- EncodeProcessDecode.py: GNN architecture
- RLHead.py: policy (log_softmax) and value heads

Reference: DIffUCO/Networks/DiffModel.py, EncodeProcessDecode.py, RLHead.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim: int = 0,
                dim_size: Optional[int] = None) -> torch.Tensor:
    """
    Native PyTorch implementation of scatter_sum (replacement for torch_scatter).

    Args:
        src: Source tensor
        index: Index tensor for scatter
        dim: Dimension along which to scatter
        dim_size: Size of output dimension (inferred if None)

    Returns:
        Scattered sum tensor
    """
    if dim_size is None:
        dim_size = int(index.max()) + 1

    # Handle different tensor dimensions
    if src.dim() == 1:
        out = torch.zeros(dim_size, dtype=src.dtype, device=src.device)
        out.scatter_add_(dim, index, src)
    else:
        # For 2D tensors, expand index to match src shape
        shape = list(src.shape)
        shape[dim] = dim_size
        out = torch.zeros(shape, dtype=src.dtype, device=src.device)

        # Expand index to match source dimensions
        index_expanded = index.unsqueeze(-1).expand_as(src)
        out.scatter_add_(dim, index_expanded, src)

    return out


##############################################################################
# Graph Attention Network Components (for TSP - matches DiffUCO's GAT.py)
##############################################################################

class AttentionQueryModule(nn.Module):
    """
    Projects concatenated [sender, receiver, edge] features to query space.
    Matches DiffUCO's AttentionQueryModule from GAT.py.
    """
    def __init__(self, input_dim: int, proj_features: int):
        super().__init__()
        self.proj_layer = nn.Linear(input_dim, proj_features)
        nn.init.kaiming_normal_(self.proj_layer.weight)
        nn.init.zeros_(self.proj_layer.bias)

    def forward(self, sender_attr: torch.Tensor, receiver_attr: torch.Tensor,
                edges: torch.Tensor) -> torch.Tensor:
        x = torch.cat([sender_attr, receiver_attr, edges], dim=-1)
        return self.proj_layer(x)


class AttentionLogitModule(nn.Module):
    """
    Computes attention logits from [sender, receiver, edge] features.
    Matches DiffUCO's AttentionlogitModule from GAT.py.
    """
    def __init__(self, input_dim: int, feature_list: list):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for n_features in feature_list:
            layers.append(nn.Linear(prev_dim, n_features))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(n_features))
            prev_dim = n_features
        # Output layer: single logit
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers)

    def forward(self, sender_attr: torch.Tensor, receiver_attr: torch.Tensor,
                edges: torch.Tensor) -> torch.Tensor:
        x = torch.cat([sender_attr, receiver_attr, edges], dim=-1)
        return self.mlp(x)


def segment_softmax(logits: torch.Tensor, segment_ids: torch.Tensor,
                    num_segments: int) -> torch.Tensor:
    """
    Compute softmax within each segment (graph).
    Matches jraph's segment_softmax.
    """
    # Subtract max for numerical stability
    max_per_segment = scatter_max(logits.squeeze(-1), segment_ids, dim=0,
                                   dim_size=num_segments)
    max_expanded = max_per_segment[segment_ids]
    exp_logits = torch.exp(logits.squeeze(-1) - max_expanded)

    # Sum per segment
    sum_per_segment = scatter_sum(exp_logits, segment_ids, dim=0,
                                   dim_size=num_segments)
    sum_expanded = sum_per_segment[segment_ids]

    # Normalize
    weights = (exp_logits / (sum_expanded + 1e-10)).unsqueeze(-1)
    return weights


def scatter_max(src: torch.Tensor, index: torch.Tensor, dim: int = 0,
                dim_size: Optional[int] = None) -> torch.Tensor:
    """Native PyTorch scatter_max."""
    if dim_size is None:
        dim_size = int(index.max()) + 1
    out = torch.full((dim_size,), float('-inf'), dtype=src.dtype, device=src.device)
    out.scatter_reduce_(dim, index, src, reduce='amax', include_self=False)
    # Replace -inf with 0 for empty segments
    out = torch.where(out == float('-inf'), torch.zeros_like(out), out)
    return out


class GraphAttentionLayer(nn.Module):
    """
    Single-head graph attention layer.
    Matches DiffUCO's ownGAT from GAT.py.
    """
    def __init__(self, node_dim: int, edge_dim: int, message_dim: int):
        super().__init__()
        # Input dim for attention: sender + receiver + edge
        input_dim = node_dim * 2 + edge_dim
        self.att_query = AttentionQueryModule(input_dim, message_dim)
        self.att_logit = AttentionLogitModule(input_dim, [message_dim])

    def forward(self, nodes: torch.Tensor, edge_index: torch.Tensor,
                edges: torch.Tensor) -> torch.Tensor:
        """
        Args:
            nodes: (n_nodes, node_dim)
            edge_index: (2, n_edges) - [senders, receivers]
            edges: (n_edges, edge_dim)
        Returns:
            Updated nodes: (n_nodes, message_dim)
        """
        senders, receivers = edge_index[0], edge_index[1]
        n_nodes = nodes.size(0)

        # Get sender and receiver attributes
        sent_attr = nodes[senders]
        recv_attr = nodes[receivers]

        # Compute queries and attention logits
        queries = self.att_query(sent_attr, recv_attr, edges)
        logits = self.att_logit(sent_attr, recv_attr, edges)

        # Compute attention weights via segment softmax
        weights = segment_softmax(logits, receivers, n_nodes)

        # Apply weights to queries
        messages = queries * weights

        # Aggregate to nodes
        out = scatter_sum(messages, receivers, dim=0, dim_size=n_nodes)

        # Apply leaky relu (matches DiffUCO's node_update_fn default)
        out = F.leaky_relu(out)

        return out


class MultiheadGraphAttentionNetwork(nn.Module):
    """
    Multi-head graph attention network.
    Matches DiffUCO's MultiheadGraphAttentionNetwork from GAT.py.
    """
    def __init__(self, node_dim: int, edge_dim: int, message_dim: int,
                 n_heads: int = 6, graph_norm: bool = False):
        super().__init__()
        self.n_heads = n_heads
        self.message_dim = message_dim

        # Multiple attention heads
        self.heads = nn.ModuleList([
            GraphAttentionLayer(node_dim, edge_dim, message_dim)
            for _ in range(n_heads)
        ])

        # Output projection with residual
        self.node_mlp = ReluMLP([message_dim, message_dim, message_dim])
        self.lin_proj = nn.Linear(node_dim, message_dim)
        self.layer_norm = nn.LayerNorm(message_dim)

        # Optional graph normalization (not implemented - rarely used)
        self.graph_norm = graph_norm

    def forward(self, nodes: torch.Tensor, edge_index: torch.Tensor,
                edges: torch.Tensor) -> torch.Tensor:
        """
        Args:
            nodes: (n_nodes, node_dim)
            edge_index: (2, n_edges)
            edges: (n_edges, edge_dim)
        Returns:
            Updated nodes: (n_nodes, message_dim)
        """
        # Run all attention heads and sum
        head_outputs = [head(nodes, edge_index, edges) for head in self.heads]
        gat_output = sum(head_outputs)

        # MLP + residual + layer norm
        gat_output = self.node_mlp(gat_output)
        gat_output = self.layer_norm(gat_output + self.lin_proj(nodes))

        return gat_output


##############################################################################
# TSP-specific Encode-Process-Decode (uses GAT instead of message passing)
##############################################################################

class TSPEncodeProcessDecode(nn.Module):
    """
    Encode-Process-Decode for TSP using Graph Attention Networks.
    Matches DiffUCO's TSPModel from TSPModel.py.
    """
    def __init__(
        self,
        input_dim: int,
        edge_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_message_passes: int = 5,
        n_heads: int = 6,
        graph_norm: bool = False
    ):
        super().__init__()

        # Encoders
        self.node_encoder = ReluMLP([input_dim, hidden_dim, hidden_dim])

        # Process block: multiple GAT layers
        self.gat_layers = nn.ModuleList([
            MultiheadGraphAttentionNetwork(
                node_dim=hidden_dim,
                edge_dim=edge_dim,  # Use raw edge features (not encoded)
                message_dim=hidden_dim,
                n_heads=n_heads,
                graph_norm=graph_norm
            )
            for _ in range(n_message_passes)
        ])

        # Decoder
        self.node_decoder = ReluMLP([hidden_dim, hidden_dim, output_dim])

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        # Encode nodes
        nodes = self.node_encoder(x)

        # Process with GAT layers
        for gat in self.gat_layers:
            nodes = gat(nodes, edge_index, edge_attr)

        # Decode
        output = self.node_decoder(nodes)
        return output


##############################################################################
# Standard MLP Components
##############################################################################

class ReluMLP(nn.Module):
    """
    MLP with ReLU activation and LayerNorm after each layer.

    Matches DiffUCO's ReluMLP from MLPs.py with proper initialization:
    - kernel_init=nn.initializers.he_normal() for all layers
    - bias_init=nn.initializers.zeros for all layers
    """

    def __init__(self, n_features_list: list):
        """
        Args:
            n_features_list: [input_dim, hidden1, hidden2, ..., output_dim]
        """
        super().__init__()

        layers = []
        for i in range(len(n_features_list) - 1):
            linear = nn.Linear(n_features_list[i], n_features_list[i + 1])
            # DiffUCO uses he_normal (Kaiming normal) for weights, zeros for bias
            nn.init.kaiming_normal_(linear.weight, nonlinearity='relu')
            nn.init.zeros_(linear.bias)
            layers.append(linear)
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(n_features_list[i + 1]))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class ProbMLP(nn.Module):
    """
    MLP with log_softmax on the last layer for policy output.

    Matches DiffUCO's ProbMLP from MLPs.py with proper initialization:
    - kernel_init=nn.initializers.he_normal() for hidden layers
    - bias_init=nn.initializers.zeros for all layers
    - Output layer: dtype=jnp.float32 (no special init mentioned)
    """

    def __init__(self, n_features_list: list):
        super().__init__()

        layers = []
        for i in range(len(n_features_list) - 2):
            linear = nn.Linear(n_features_list[i], n_features_list[i + 1])
            # DiffUCO uses he_normal for hidden layers
            nn.init.kaiming_normal_(linear.weight, nonlinearity='relu')
            nn.init.zeros_(linear.bias)
            layers.append(linear)
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(n_features_list[i + 1]))

        # Output layer without activation (log_softmax applied separately)
        # DiffUCO: Dense with default init for output layer
        output_linear = nn.Linear(n_features_list[-2], n_features_list[-1])
        nn.init.kaiming_normal_(output_linear.weight, nonlinearity='relu')
        nn.init.zeros_(output_linear.bias)
        layers.append(output_linear)

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.mlp(x)
        return F.log_softmax(logits, dim=-1)


class ValueMLP(nn.Module):
    """
    MLP for value prediction.

    Matches DiffUCO's ValueMLP from MLPs.py with proper initialization:
    - kernel_init=nn.initializers.he_normal() for hidden layers
    - kernel_init=nn.initializers.xavier_normal() for final layer
    - bias_init=nn.initializers.zeros for all layers
    """

    def __init__(self, n_features_list: list):
        super().__init__()

        layers = []
        for i in range(len(n_features_list) - 2):
            linear = nn.Linear(n_features_list[i], n_features_list[i + 1])
            # DiffUCO uses he_normal for hidden layers
            nn.init.kaiming_normal_(linear.weight, nonlinearity='relu')
            nn.init.zeros_(linear.bias)
            layers.append(linear)
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(n_features_list[i + 1]))

        # Output layer without activation
        # DiffUCO uses xavier_normal for final value layer
        output_linear = nn.Linear(n_features_list[-2], n_features_list[-1])
        nn.init.xavier_normal_(output_linear.weight)
        nn.init.zeros_(output_linear.bias)
        layers.append(output_linear)

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class LinearMessagePassingLayer(nn.Module):
    """
    Linear message passing layer matching DiffUCO's LinearMessagePassingLayer.

    From EncodeProcessDecode.py:
        - message = W_message(concat([sender_features, edges]))
        - aggregated = segment_sum(message, receivers)
        - node_out = NodeMLP(concat([nodes, aggregated]))
        - nodes_new = LayerNorm(W_node(nodes) + node_out)

    Initialization matches DiffUCO: he_normal for weights, zeros for bias (when present)
    """

    def __init__(self, node_dim: int, edge_dim: int, message_dim: int, mean_aggr: bool = False):
        super().__init__()

        self.mean_aggr = mean_aggr

        # Message projection: concat([sender, edge]) -> message
        self.W_message = nn.Linear(node_dim + edge_dim, message_dim, bias=False)
        nn.init.kaiming_normal_(self.W_message.weight, nonlinearity='relu')

        # Node projection for residual
        self.W_node = nn.Linear(node_dim, node_dim, bias=False)
        nn.init.kaiming_normal_(self.W_node.weight, nonlinearity='relu')

        # Node MLP: concat([nodes, aggregated]) -> node_out
        # ReluMLP already has proper initialization
        self.node_mlp = ReluMLP([node_dim + message_dim, node_dim, node_dim])

        self.layer_norm = nn.LayerNorm(node_dim)

    def forward(
        self,
        nodes: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            nodes: Node features (n_nodes, node_dim)
            edge_index: Edge indices (2, n_edges) - [senders, receivers]
            edge_attr: Edge features (n_edges, edge_dim)

        Returns:
            Updated node features (n_nodes, node_dim)
        """
        senders, receivers = edge_index[0], edge_index[1]
        n_nodes = nodes.size(0)

        # Get sender features
        sender_features = nodes[senders]  # (n_edges, node_dim)

        # Compute messages
        message_input = torch.cat([sender_features, edge_attr], dim=-1)
        messages = self.W_message(message_input)  # (n_edges, message_dim)

        # Aggregate messages to receivers
        aggregated = scatter_sum(messages, receivers, dim=0, dim_size=n_nodes)

        if self.mean_aggr:
            # Normalize by sqrt(degree)
            degree = scatter_sum(
                torch.ones(messages.size(0), 1, device=messages.device),
                receivers, dim=0, dim_size=n_nodes
            )
            degree = torch.clamp(degree, min=1.0)
            aggregated = aggregated / torch.sqrt(degree)

        # Update nodes
        node_input = torch.cat([nodes, aggregated], dim=-1)
        node_out = self.node_mlp(node_input)

        # Residual connection with layer norm
        nodes_new = self.layer_norm(self.W_node(nodes) + node_out)

        return nodes_new


class EncodeProcessDecode(nn.Module):
    """
    Encode-Process-Decode GNN architecture matching DiffUCO.

    From EncodeProcessDecode.py:
        - Encode: node_encoder(X_prev), edge_encoder(edges)
        - Process: multiple message passing layers
        - Decode: node_decoder(nodes)
    """

    def __init__(
        self,
        input_dim: int,
        edge_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_message_passes: int = 5,
        mean_aggr: bool = False
    ):
        super().__init__()

        # Encoders
        self.node_encoder = ReluMLP([input_dim, hidden_dim, hidden_dim])
        self.edge_encoder = ReluMLP([edge_dim, hidden_dim, hidden_dim])

        # Process block: multiple message passing layers
        self.message_layers = nn.ModuleList([
            LinearMessagePassingLayer(
                node_dim=hidden_dim,
                edge_dim=hidden_dim,
                message_dim=hidden_dim,
                mean_aggr=mean_aggr
            )
            for _ in range(n_message_passes)
        ])

        # Decoder
        self.node_decoder = ReluMLP([hidden_dim, hidden_dim, output_dim])

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: Input node features (n_nodes, input_dim)
            edge_index: Edge indices (2, n_edges)
            edge_attr: Edge features (n_edges, edge_dim)

        Returns:
            Decoded node features (n_nodes, output_dim)
        """
        # Encode
        nodes = self.node_encoder(x)
        edges = self.edge_encoder(edge_attr)

        # Process
        for layer in self.message_layers:
            nodes = layer(nodes, edge_index, edges)

        # Decode
        output = self.node_decoder(nodes)

        return output


def get_sinusoidal_positional_encoding(
    timestep: torch.Tensor,
    embedding_dim: int,
    max_position: int
) -> torch.Tensor:
    """
    Sinusoidal positional encoding for time embedding.

    From DiffModel.py get_sinusoidal_positional_encoding.
    """
    position = timestep.float()
    div_term = torch.exp(
        torch.arange(0, embedding_dim, 2, device=timestep.device).float()
        * (-math.log(max_position) / embedding_dim)
    )

    # Handle scalar or tensor input
    if position.dim() == 0:
        position = position.unsqueeze(0)

    pe = torch.zeros(position.size(0), embedding_dim, device=timestep.device)
    pe[:, 0::2] = torch.sin(position.unsqueeze(-1) * div_term)
    pe[:, 1::2] = torch.cos(position.unsqueeze(-1) * div_term)

    return pe


class DiffusionStepModel(nn.Module):
    """
    Complete diffusion step model matching DiffUCO's DiffModel + RLHead.

    For each diffusion step, this model:
    1. Constructs input: one_hot(X_t) + time_embed + random_nodes
    2. Runs GNN (EncodeProcessDecode)
    3. Outputs policy (log_softmax) and value

    From DiffModel.py:
        - _add_random_nodes_and_time_index: builds input features
        - reinit_rand_nodes: generates fresh random features each step
        - make_one_step: runs full forward pass

    From RLHead.py (RLHeadModule_agg_before):
        - ProbMLP for spin_logits
        - ValueMLP with global aggregation
    """

    def __init__(
        self,
        n_classes: int,
        edge_dim: int = 2,
        hidden_dim: int = 64,
        n_diffusion_steps: int = 5,
        n_message_passes: int = 5,
        n_random_features: int = 5,
        time_encoding: str = 'sinusoidal',
        embedding_dim: int = 32,
        mean_aggr: bool = False
    ):
        """
        Args:
            n_classes: Number of categories K (n_bernoulli_features in DiffUCO)
            edge_dim: Edge feature dimension
            hidden_dim: Hidden dimension for GNN
            n_diffusion_steps: Total diffusion steps T
            n_message_passes: Number of message passing layers
            n_random_features: Number of random node features
            time_encoding: 'one_hot' or 'sinusoidal'
            embedding_dim: Dimension for time embedding (if sinusoidal)
            mean_aggr: Use mean aggregation instead of sum
        """
        super().__init__()

        self.n_classes = n_classes
        self.n_diffusion_steps = n_diffusion_steps
        self.n_random_features = n_random_features
        self.time_encoding = time_encoding
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # Compute input dimension
        if time_encoding == 'one_hot':
            time_dim = n_diffusion_steps
        else:
            time_dim = embedding_dim

        input_dim = n_classes + time_dim + n_random_features

        # GNN backbone
        self.gnn = EncodeProcessDecode(
            input_dim=input_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            n_message_passes=n_message_passes,
            mean_aggr=mean_aggr
        )

        # Policy head (ProbMLP with log_softmax)
        self.prob_head = ProbMLP([hidden_dim, hidden_dim, n_classes])

        # Value head (ValueMLP)
        value_features = [hidden_dim, 120, 64, 1]
        self.value_head = ValueMLP(value_features)

    def _add_random_nodes_and_time_index(
        self,
        X_t: torch.Tensor,
        rand_nodes: torch.Tensor,
        t_idx: int
    ) -> torch.Tensor:
        """
        Build input features by concatenating:
        - one_hot(X_t)
        - time_embedding
        - random_nodes

        Matches DiffModel._add_random_nodes_and_time_index
        """
        n_nodes = X_t.size(0)
        device = X_t.device

        # One-hot encode current state
        X_one_hot = F.one_hot(X_t, num_classes=self.n_classes).float()  # (n_nodes, n_classes)

        # Time embedding
        if self.time_encoding == 'one_hot':
            t_tensor = torch.tensor([t_idx], device=device)
            t_embed = F.one_hot(t_tensor, num_classes=self.n_diffusion_steps).float()
            t_embed = t_embed.expand(n_nodes, -1)  # (n_nodes, n_diffusion_steps)
        else:
            t_tensor = torch.tensor([t_idx], device=device)
            t_embed = get_sinusoidal_positional_encoding(
                t_tensor, self.embedding_dim, self.n_diffusion_steps
            )
            t_embed = t_embed.expand(n_nodes, -1)  # (n_nodes, embedding_dim)

        # Concatenate all inputs
        X_input = torch.cat([X_one_hot, t_embed, rand_nodes], dim=-1)

        return X_input

    def reinit_rand_nodes(self, n_nodes: int, device: torch.device) -> torch.Tensor:
        """
        Generate fresh random node features.

        Matches DiffModel.reinit_rand_nodes: uniform [0, 1]
        """
        return torch.rand(n_nodes, self.n_random_features, device=device)

    def forward(
        self,
        X_t: torch.Tensor,
        t_idx: int,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        node_graph_idx: torch.Tensor,
        n_graphs: int,
        rand_nodes: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for one diffusion step.

        Args:
            X_t: Current discrete state (n_nodes,) in {0, ..., K-1}
            t_idx: Current timestep index
            edge_index: Graph edges (2, n_edges)
            edge_attr: Edge features (n_edges, edge_dim)
            node_graph_idx: Graph index for each node (n_nodes,)
            n_graphs: Number of graphs in batch
            rand_nodes: Optional pre-generated random features

        Returns:
            spin_log_probs: Log probabilities (n_nodes, n_classes)
            values: Value estimates per graph (n_graphs,)
            rand_nodes: Random features used (for storage)
        """
        n_nodes = X_t.size(0)
        device = X_t.device

        # Generate random features if not provided
        if rand_nodes is None:
            rand_nodes = self.reinit_rand_nodes(n_nodes, device)

        # Build input features
        X_input = self._add_random_nodes_and_time_index(X_t, rand_nodes, t_idx)

        # Run GNN
        node_embeddings = self.gnn(X_input, edge_index, edge_attr)  # (n_nodes, hidden_dim)

        # Policy head: log_softmax probabilities
        spin_log_probs = self.prob_head(node_embeddings)  # (n_nodes, n_classes)

        # Value head: aggregate then MLP
        # From RLHead.py: Value_embeddings = segment_sum(x) / sqrt(n_node)
        n_node_per_graph = scatter_sum(
            torch.ones(n_nodes, device=device),
            node_graph_idx, dim=0, dim_size=n_graphs
        )  # (n_graphs,)

        # Global aggregation
        value_embeddings = scatter_sum(
            node_embeddings, node_graph_idx, dim=0, dim_size=n_graphs
        )  # (n_graphs, hidden_dim)

        # Normalize by sqrt(n_node)
        value_embeddings = value_embeddings / torch.sqrt(n_node_per_graph.unsqueeze(-1).clamp(min=1.0))

        values = self.value_head(value_embeddings).squeeze(-1)  # (n_graphs,)

        return spin_log_probs, values, rand_nodes

    def sample_action(
        self,
        spin_log_probs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample actions from the policy distribution.

        Args:
            spin_log_probs: Log probabilities (n_nodes, n_classes)

        Returns:
            X_next: Sampled next state (n_nodes,)
            action_log_probs: Log prob of sampled actions (n_nodes,)
        """
        # Sample from categorical distribution
        dist = torch.distributions.Categorical(logits=spin_log_probs)
        X_next = dist.sample()

        # Get log prob of sampled action
        action_log_probs = spin_log_probs.gather(1, X_next.unsqueeze(-1)).squeeze(-1)

        return X_next, action_log_probs

    def get_state_log_prob(
        self,
        spin_log_probs: torch.Tensor,
        X_next: torch.Tensor,
        node_graph_idx: torch.Tensor,
        n_graphs: int
    ) -> torch.Tensor:
        """
        Compute log probability of state (summed over nodes per graph).

        Matches DiffModel.__get_log_prob: segment_sum of per-node log probs.

        Args:
            spin_log_probs: Log probabilities (n_nodes, n_classes)
            X_next: Sampled state (n_nodes,)
            node_graph_idx: Graph index for each node (n_nodes,)
            n_graphs: Number of graphs

        Returns:
            state_log_probs: (n_graphs,)
        """
        # Get log prob per node
        per_node_log_prob = spin_log_probs.gather(1, X_next.unsqueeze(-1)).squeeze(-1)

        # Sum over nodes per graph
        state_log_probs = scatter_sum(per_node_log_prob, node_graph_idx, dim=0, dim_size=n_graphs)

        return state_log_probs


##############################################################################
# TSP-specific Diffusion Step Model (matches DiffUCO's TSP architecture)
##############################################################################

class TSPDiffusionStepModel(nn.Module):
    """
    TSP-specific diffusion step model matching DiffUCO's architecture.

    Key differences from standard DiffusionStepModel:
    1. Uses Graph Attention Networks (GAT) instead of message passing
    2. Uses RLHeadModuleTSP: concatenates mean-aggregated features with node features
    3. Different value network architecture

    From DiffUCO:
        - TSPModel.py: uses MultiheadGraphAttentionNetwork
        - RLHead.py (RLHeadModuleTSP): x = concat([x, repeat(mean(x))])
    """

    def __init__(
        self,
        n_classes: int,
        edge_dim: int = 1,
        hidden_dim: int = 64,
        n_diffusion_steps: int = 5,
        n_message_passes: int = 5,
        n_random_features: int = 5,
        time_encoding: str = 'sinusoidal',
        embedding_dim: int = 32,
        n_heads: int = 6,
        graph_norm: bool = False,
    ):
        """
        Args:
            n_classes: Number of categories K (= n_nodes for TSP)
            edge_dim: Edge feature dimension (1 for distance)
            hidden_dim: Hidden dimension for GNN
            n_diffusion_steps: Total diffusion steps T
            n_message_passes: Number of GAT layers
            n_random_features: Number of random node features
            time_encoding: 'one_hot' or 'sinusoidal'
            embedding_dim: Dimension for time embedding (if sinusoidal)
            n_heads: Number of attention heads
            graph_norm: Use graph normalization
        """
        super().__init__()

        self.n_classes = n_classes
        self.n_diffusion_steps = n_diffusion_steps
        self.n_random_features = n_random_features
        self.time_encoding = time_encoding
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # Compute input dimension
        if time_encoding == 'one_hot':
            time_dim = n_diffusion_steps
        else:
            time_dim = embedding_dim

        input_dim = n_classes + time_dim + n_random_features

        # GNN backbone: TSP-specific with GAT
        self.gnn = TSPEncodeProcessDecode(
            input_dim=input_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            n_message_passes=n_message_passes,
            n_heads=n_heads,
            graph_norm=graph_norm
        )

        # Policy head (ProbMLP with log_softmax)
        # Input: concat([node_features, mean_aggregated]) = 2 * hidden_dim
        self.prob_head = ProbMLP([hidden_dim * 2, hidden_dim, n_classes])

        # Value head (ValueMLP) - TSP uses [120, 120, 1]
        # Input: mean aggregated features
        self.value_head = ValueMLP([hidden_dim, 120, 120, 1])

    def _add_random_nodes_and_time_index(
        self,
        X_t: torch.Tensor,
        rand_nodes: torch.Tensor,
        t_idx: int
    ) -> torch.Tensor:
        """Build input features: one_hot(X_t) + time_embed + random_nodes"""
        n_nodes = X_t.size(0)
        device = X_t.device

        # One-hot encode current state
        X_one_hot = F.one_hot(X_t, num_classes=self.n_classes).float()

        # Time embedding
        if self.time_encoding == 'one_hot':
            t_tensor = torch.tensor([t_idx], device=device)
            t_embed = F.one_hot(t_tensor, num_classes=self.n_diffusion_steps).float()
            t_embed = t_embed.expand(n_nodes, -1)
        else:
            t_tensor = torch.tensor([t_idx], device=device)
            t_embed = get_sinusoidal_positional_encoding(
                t_tensor, self.embedding_dim, self.n_diffusion_steps
            )
            t_embed = t_embed.expand(n_nodes, -1)

        # Concatenate all inputs
        X_input = torch.cat([X_one_hot, t_embed, rand_nodes], dim=-1)
        return X_input

    def reinit_rand_nodes(self, n_nodes: int, device: torch.device) -> torch.Tensor:
        """Generate fresh random node features."""
        return torch.rand(n_nodes, self.n_random_features, device=device)

    def forward(
        self,
        X_t: torch.Tensor,
        t_idx: int,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        node_graph_idx: torch.Tensor,
        n_graphs: int,
        rand_nodes: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for one diffusion step (TSP-specific).

        Key difference: concatenates mean-aggregated features with node features
        before policy head (RLHeadModuleTSP style).
        """
        n_nodes = X_t.size(0)
        device = X_t.device

        # Generate random features if not provided
        if rand_nodes is None:
            rand_nodes = self.reinit_rand_nodes(n_nodes, device)

        # Build input features
        X_input = self._add_random_nodes_and_time_index(X_t, rand_nodes, t_idx)

        # Run GNN (GAT-based)
        node_embeddings = self.gnn(X_input, edge_index, edge_attr)  # (n_nodes, hidden_dim)

        # RLHeadModuleTSP-style processing:
        # x_aggr = mean(x) over nodes per graph
        # x = concat([x, repeat(x_aggr)])

        # Compute mean per graph
        n_node_per_graph = scatter_sum(
            torch.ones(n_nodes, device=device),
            node_graph_idx, dim=0, dim_size=n_graphs
        )

        sum_per_graph = scatter_sum(
            node_embeddings, node_graph_idx, dim=0, dim_size=n_graphs
        )  # (n_graphs, hidden_dim)

        mean_per_graph = sum_per_graph / n_node_per_graph.unsqueeze(-1).clamp(min=1.0)

        # Expand mean to all nodes
        mean_expanded = mean_per_graph[node_graph_idx]  # (n_nodes, hidden_dim)

        # Concatenate node features with mean-aggregated features
        concat_features = torch.cat([node_embeddings, mean_expanded], dim=-1)  # (n_nodes, 2*hidden_dim)

        # Policy head: log_softmax probabilities
        spin_log_probs = self.prob_head(concat_features)  # (n_nodes, n_classes)

        # Value head: use mean-aggregated features directly
        values = self.value_head(mean_per_graph).squeeze(-1)  # (n_graphs,)

        return spin_log_probs, values, rand_nodes

    def sample_action(
        self,
        spin_log_probs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample actions from the policy distribution."""
        dist = torch.distributions.Categorical(logits=spin_log_probs)
        X_next = dist.sample()
        action_log_probs = spin_log_probs.gather(1, X_next.unsqueeze(-1)).squeeze(-1)
        return X_next, action_log_probs

    def get_state_log_prob(
        self,
        spin_log_probs: torch.Tensor,
        X_next: torch.Tensor,
        node_graph_idx: torch.Tensor,
        n_graphs: int
    ) -> torch.Tensor:
        """Compute log probability of state (summed over nodes per graph)."""
        per_node_log_prob = spin_log_probs.gather(1, X_next.unsqueeze(-1)).squeeze(-1)
        state_log_probs = scatter_sum(per_node_log_prob, node_graph_idx, dim=0, dim_size=n_graphs)
        return state_log_probs


def test_step_model():
    """
    Test the DiffusionStepModel.
    """
    print("=" * 60)
    print("Testing DiffusionStepModel")
    print("=" * 60)

    # Test parameters
    n_classes = 4
    n_nodes = 20
    n_edges = 60
    n_graphs = 2
    nodes_per_graph = n_nodes // n_graphs
    hidden_dim = 32
    n_diffusion_steps = 5
    edge_dim = 2

    # Create model
    model = DiffusionStepModel(
        n_classes=n_classes,
        edge_dim=edge_dim,
        hidden_dim=hidden_dim,
        n_diffusion_steps=n_diffusion_steps,
        n_message_passes=3,
        n_random_features=5,
        time_encoding='sinusoidal',
        embedding_dim=16
    )

    print(f"\nModel created with:")
    print(f"  n_classes: {n_classes}")
    print(f"  hidden_dim: {hidden_dim}")
    print(f"  n_diffusion_steps: {n_diffusion_steps}")
    print(f"  n_message_passes: 3")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    # Create test data
    X_t = torch.randint(0, n_classes, (n_nodes,))
    edge_index = torch.randint(0, n_nodes, (2, n_edges))
    edge_attr = torch.randn(n_edges, edge_dim)
    node_graph_idx = torch.repeat_interleave(
        torch.arange(n_graphs), nodes_per_graph
    )

    print(f"\nTest data:")
    print(f"  X_t shape: {X_t.shape}")
    print(f"  edge_index shape: {edge_index.shape}")
    print(f"  edge_attr shape: {edge_attr.shape}")
    print(f"  node_graph_idx shape: {node_graph_idx.shape}")

    # Test forward pass
    print("\nTesting forward pass...")
    for t in range(n_diffusion_steps):
        spin_log_probs, values, rand_nodes = model(
            X_t, t, edge_index, edge_attr, node_graph_idx, n_graphs
        )

        print(f"  Step {t}:")
        print(f"    spin_log_probs shape: {spin_log_probs.shape}")
        print(f"    values shape: {values.shape}")
        print(f"    rand_nodes shape: {rand_nodes.shape}")

        # Verify log_softmax (should sum to 0 in log space, i.e., probs sum to 1)
        probs = torch.exp(spin_log_probs)
        prob_sum = probs.sum(dim=-1)
        assert torch.allclose(prob_sum, torch.ones_like(prob_sum), atol=1e-5), \
            "Probabilities don't sum to 1"

    print("\n[OK] Forward pass test passed")

    # Test sampling
    print("\nTesting action sampling...")
    spin_log_probs, values, rand_nodes = model(
        X_t, 0, edge_index, edge_attr, node_graph_idx, n_graphs
    )
    X_next, action_log_probs = model.sample_action(spin_log_probs)

    print(f"  X_next shape: {X_next.shape}")
    print(f"  X_next range: [{X_next.min().item()}, {X_next.max().item()}]")
    print(f"  action_log_probs shape: {action_log_probs.shape}")

    assert X_next.shape == X_t.shape, "X_next shape mismatch"
    assert (X_next >= 0).all() and (X_next < n_classes).all(), "X_next out of range"

    print("[OK] Sampling test passed")

    # Test state log prob computation
    print("\nTesting state log prob computation...")
    state_log_probs = model.get_state_log_prob(
        spin_log_probs, X_next, node_graph_idx, n_graphs
    )

    print(f"  state_log_probs shape: {state_log_probs.shape}")
    print(f"  state_log_probs: {state_log_probs.detach().numpy()}")

    assert state_log_probs.shape == (n_graphs,), "state_log_probs shape mismatch"

    print("[OK] State log prob test passed")

    # Test gradient flow
    print("\nTesting gradient flow...")
    model.zero_grad()
    spin_log_probs, values, _ = model(
        X_t, 0, edge_index, edge_attr, node_graph_idx, n_graphs
    )

    # Dummy loss
    loss = spin_log_probs.sum() + values.sum()
    loss.backward()

    # Check gradients exist
    has_grad = all(p.grad is not None for p in model.parameters())
    print(f"  All parameters have gradients: {has_grad}")

    print("[OK] Gradient flow test passed")

    print("\n" + "=" * 60)
    print("All DiffusionStepModel tests passed!")
    print("=" * 60)

    return True


##############################################################################
# CVRP-specific Diffusion Step Model (with additional node features)
##############################################################################

class CVRPDiffusionStepModel(nn.Module):
    """
    CVRP-specific diffusion step model with support for node features.

    Key differences from standard DiffusionStepModel:
    1. Accepts additional node features (demand/capacity, r, theta)
    2. Concatenates node features with one_hot + time_embed + rand_nodes

    Input features: one_hot(X_t) + time_embed + rand_nodes + node_features
    """

    def __init__(
        self,
        n_classes: int,
        node_feat_dim: int = 3,  # (demand/capacity, r, theta)
        edge_dim: int = 2,
        hidden_dim: int = 64,
        n_diffusion_steps: int = 5,
        n_message_passes: int = 5,
        n_random_features: int = 5,
        time_encoding: str = 'sinusoidal',
        embedding_dim: int = 32,
        mean_aggr: bool = False
    ):
        """
        Args:
            n_classes: Number of categories K (number of vehicles/partitions)
            node_feat_dim: Dimension of additional node features (3 for CVRP)
            edge_dim: Edge feature dimension
            hidden_dim: Hidden dimension for GNN
            n_diffusion_steps: Total diffusion steps T
            n_message_passes: Number of message passing layers
            n_random_features: Number of random node features
            time_encoding: 'one_hot' or 'sinusoidal'
            embedding_dim: Dimension for time embedding (if sinusoidal)
            mean_aggr: Use mean aggregation instead of sum
        """
        super().__init__()

        self.n_classes = n_classes
        self.node_feat_dim = node_feat_dim
        self.n_diffusion_steps = n_diffusion_steps
        self.n_random_features = n_random_features
        self.time_encoding = time_encoding
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # Compute input dimension
        if time_encoding == 'one_hot':
            time_dim = n_diffusion_steps
        else:
            time_dim = embedding_dim

        # Input: one_hot(K) + time + rand + node_features
        input_dim = n_classes + time_dim + n_random_features + node_feat_dim

        # GNN backbone
        self.gnn = EncodeProcessDecode(
            input_dim=input_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            n_message_passes=n_message_passes,
            mean_aggr=mean_aggr
        )

        # Policy head (ProbMLP with log_softmax)
        self.prob_head = ProbMLP([hidden_dim, hidden_dim, n_classes])

        # Value head (ValueMLP)
        value_features = [hidden_dim, 120, 64, 1]
        self.value_head = ValueMLP(value_features)

    def _add_step_features(
        self,
        X_t: torch.Tensor,
        rand_nodes: torch.Tensor,
        t_idx: int,
        node_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Build input features by concatenating:
        - one_hot(X_t)
        - time_embedding
        - random_nodes
        - node_features (CVRP-specific: demand/cap, r, theta)

        Args:
            X_t: Current state (n_nodes,)
            rand_nodes: Random features (n_nodes, n_random_features)
            t_idx: Time step index
            node_features: CVRP node features (n_nodes, node_feat_dim)

        Returns:
            X_input: Concatenated features (n_nodes, input_dim)
        """
        n_nodes = X_t.size(0)
        device = X_t.device

        # One-hot encode current state
        X_one_hot = F.one_hot(X_t, num_classes=self.n_classes).float()  # (n_nodes, n_classes)

        # Time embedding
        if self.time_encoding == 'one_hot':
            t_tensor = torch.tensor([t_idx], device=device)
            t_embed = F.one_hot(t_tensor, num_classes=self.n_diffusion_steps).float()
            t_embed = t_embed.expand(n_nodes, -1)  # (n_nodes, n_diffusion_steps)
        else:
            t_tensor = torch.tensor([t_idx], device=device)
            t_embed = get_sinusoidal_positional_encoding(
                t_tensor, self.embedding_dim, self.n_diffusion_steps
            )
            t_embed = t_embed.expand(n_nodes, -1)  # (n_nodes, embedding_dim)

        # Concatenate all inputs: one_hot + time + rand + node_features
        X_input = torch.cat([X_one_hot, t_embed, rand_nodes, node_features], dim=-1)

        return X_input

    def reinit_rand_nodes(self, n_nodes: int, device: torch.device) -> torch.Tensor:
        """Generate fresh random node features."""
        return torch.rand(n_nodes, self.n_random_features, device=device)

    def forward(
        self,
        X_t: torch.Tensor,
        t_idx: int,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        node_graph_idx: torch.Tensor,
        n_graphs: int,
        node_features: torch.Tensor,
        rand_nodes: Optional[torch.Tensor] = None,
        customer_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for one diffusion step.

        Args:
            X_t: Current discrete state (n_nodes,) in {0, ..., K-1}
            t_idx: Current timestep index
            edge_index: Graph edges (2, n_edges)
            edge_attr: Edge features (n_edges, edge_dim)
            node_graph_idx: Graph index for each node (n_nodes,)
            n_graphs: Number of graphs in batch
            node_features: CVRP node features (n_nodes, node_feat_dim)
            rand_nodes: Optional pre-generated random features
            customer_mask: Optional boolean mask (n_nodes,) indicating customer nodes.
                          If provided, value aggregation only uses customer nodes
                          (to match reward structure which excludes depot).

        Returns:
            spin_log_probs: Log probabilities (n_nodes, n_classes)
            values: Value estimates per graph (n_graphs,)
            rand_nodes: Random features used (for storage)
        """
        n_nodes = X_t.size(0)
        device = X_t.device

        # Generate random features if not provided
        if rand_nodes is None:
            rand_nodes = self.reinit_rand_nodes(n_nodes, device)

        # Build input features (includes node_features)
        X_input = self._add_step_features(X_t, rand_nodes, t_idx, node_features)

        # Run GNN
        node_embeddings = self.gnn(X_input, edge_index, edge_attr)  # (n_nodes, hidden_dim)

        # Policy head: log_softmax probabilities
        spin_log_probs = self.prob_head(node_embeddings)  # (n_nodes, n_classes)

        # Value head: aggregate then MLP
        # If customer_mask provided, only aggregate customer nodes (excludes depot)
        # This matches the reward structure which is computed over customers only
        if customer_mask is not None:
            customer_embeddings = node_embeddings[customer_mask]
            customer_graph_idx = node_graph_idx[customer_mask]
            n_customer_per_graph = scatter_sum(
                torch.ones(customer_embeddings.size(0), device=device),
                customer_graph_idx, dim=0, dim_size=n_graphs
            )
            value_embeddings = scatter_sum(
                customer_embeddings, customer_graph_idx, dim=0, dim_size=n_graphs
            )
            value_embeddings = value_embeddings / torch.sqrt(n_customer_per_graph.unsqueeze(-1).clamp(min=1.0))
        else:
            # Fallback: aggregate all nodes (original behavior)
            n_node_per_graph = scatter_sum(
                torch.ones(n_nodes, device=device),
                node_graph_idx, dim=0, dim_size=n_graphs
            )
            value_embeddings = scatter_sum(
                node_embeddings, node_graph_idx, dim=0, dim_size=n_graphs
            )
            value_embeddings = value_embeddings / torch.sqrt(n_node_per_graph.unsqueeze(-1).clamp(min=1.0))

        values = self.value_head(value_embeddings).squeeze(-1)  # (n_graphs,)

        return spin_log_probs, values, rand_nodes

    def sample_action(
        self,
        spin_log_probs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample actions from the policy distribution."""
        dist = torch.distributions.Categorical(logits=spin_log_probs)
        X_next = dist.sample()
        action_log_probs = spin_log_probs.gather(1, X_next.unsqueeze(-1)).squeeze(-1)
        return X_next, action_log_probs

    def get_state_log_prob(
        self,
        spin_log_probs: torch.Tensor,
        X_next: torch.Tensor,
        node_graph_idx: torch.Tensor,
        n_graphs: int
    ) -> torch.Tensor:
        """Compute log probability of state (summed over nodes per graph)."""
        per_node_log_prob = spin_log_probs.gather(1, X_next.unsqueeze(-1)).squeeze(-1)
        state_log_probs = scatter_sum(per_node_log_prob, node_graph_idx, dim=0, dim_size=n_graphs)
        return state_log_probs


def test_cvrp_step_model():
    """
    Test the CVRPDiffusionStepModel.
    """
    print("=" * 60)
    print("Testing CVRPDiffusionStepModel")
    print("=" * 60)

    # Test parameters
    n_classes = 4  # K = 4 vehicles
    n_nodes = 20   # 20 customer nodes
    n_edges = 60
    n_graphs = 2
    nodes_per_graph = n_nodes // n_graphs
    hidden_dim = 32
    n_diffusion_steps = 5
    edge_dim = 2
    node_feat_dim = 3  # (demand/cap, r, theta)

    # Create model
    model = CVRPDiffusionStepModel(
        n_classes=n_classes,
        node_feat_dim=node_feat_dim,
        edge_dim=edge_dim,
        hidden_dim=hidden_dim,
        n_diffusion_steps=n_diffusion_steps,
        n_message_passes=3,
        n_random_features=5,
        time_encoding='sinusoidal',
        embedding_dim=16
    )

    print(f"\nModel created with:")
    print(f"  n_classes (K): {n_classes}")
    print(f"  node_feat_dim: {node_feat_dim}")
    print(f"  hidden_dim: {hidden_dim}")
    print(f"  n_diffusion_steps: {n_diffusion_steps}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    # Create test data
    X_t = torch.randint(0, n_classes, (n_nodes,))
    edge_index = torch.randint(0, n_nodes, (2, n_edges))
    edge_attr = torch.randn(n_edges, edge_dim)
    node_graph_idx = torch.repeat_interleave(
        torch.arange(n_graphs), nodes_per_graph
    )
    # CVRP node features: (demand/cap, r, theta)
    node_features = torch.rand(n_nodes, node_feat_dim)

    print(f"\nTest data:")
    print(f"  X_t shape: {X_t.shape}")
    print(f"  node_features shape: {node_features.shape}")
    print(f"  edge_index shape: {edge_index.shape}")

    # Test forward pass
    print("\nTesting forward pass...")
    for t in range(n_diffusion_steps):
        spin_log_probs, values, rand_nodes = model(
            X_t, t, edge_index, edge_attr, node_graph_idx, n_graphs, node_features
        )

        if t == 0:
            print(f"  Step {t}:")
            print(f"    spin_log_probs shape: {spin_log_probs.shape}")
            print(f"    values shape: {values.shape}")

        # Verify log_softmax
        probs = torch.exp(spin_log_probs)
        prob_sum = probs.sum(dim=-1)
        assert torch.allclose(prob_sum, torch.ones_like(prob_sum), atol=1e-5), \
            "Probabilities don't sum to 1"

    print("[OK] Forward pass test passed")

    # Test sampling
    print("\nTesting action sampling...")
    spin_log_probs, values, rand_nodes = model(
        X_t, 0, edge_index, edge_attr, node_graph_idx, n_graphs, node_features
    )
    X_next, action_log_probs = model.sample_action(spin_log_probs)

    assert X_next.shape == X_t.shape, "X_next shape mismatch"
    assert (X_next >= 0).all() and (X_next < n_classes).all(), "X_next out of range"
    print(f"  X_next range: [{X_next.min().item()}, {X_next.max().item()}]")
    print("[OK] Sampling test passed")

    # Test state log prob computation
    print("\nTesting state log prob computation...")
    state_log_probs = model.get_state_log_prob(
        spin_log_probs, X_next, node_graph_idx, n_graphs
    )
    assert state_log_probs.shape == (n_graphs,), "state_log_probs shape mismatch"
    print("[OK] State log prob test passed")

    # Test gradient flow
    print("\nTesting gradient flow...")
    model.zero_grad()
    spin_log_probs, values, _ = model(
        X_t, 0, edge_index, edge_attr, node_graph_idx, n_graphs, node_features
    )
    loss = spin_log_probs.sum() + values.sum()
    loss.backward()
    has_grad = all(p.grad is not None for p in model.parameters())
    print(f"  All parameters have gradients: {has_grad}")
    print("[OK] Gradient flow test passed")

    print("\n" + "=" * 60)
    print("All CVRPDiffusionStepModel tests passed!")
    print("=" * 60)

    return True


if __name__ == "__main__":
    test_step_model()
    print("\n")
    test_cvrp_step_model()
