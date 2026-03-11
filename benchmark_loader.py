"""
Benchmark Loader - Parses BookShelf format and converts to ChipSAT training format.

Supports ICCAD04 (IBM) and ISPD2005 benchmarks in BookShelf format.

Usage:
    # Load a single circuit
    data = load_bookshelf_circuit("path/to/ibm01", circuit_name="ibm01")

    # Load and convert to training batch
    batch = load_benchmark_batch(
        ["path/to/ibm01", "path/to/ibm02"],
        circuit_names=["ibm01", "ibm02"],
        max_nodes=200,  # subsample for tractability
    )
"""

import os
import re
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional


def parse_bookshelf_nodes(nodes_path: str) -> Dict:
    """
    Parse BookShelf .nodes file.

    Returns:
        dict with:
            'names': list of object names (ordered)
            'sizes': dict name -> (width, height)
            'is_terminal': dict name -> bool
    """
    names = []
    sizes = {}
    is_terminal = {}

    with open(nodes_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('UCLA') or line.startswith('Num'):
                continue

            parts = line.split()
            if len(parts) >= 3:
                name = parts[0]
                try:
                    w = float(parts[1])
                    h = float(parts[2])
                except ValueError:
                    continue

                names.append(name)
                sizes[name] = (w, h)
                is_terminal[name] = len(parts) >= 4 and 'terminal' in parts[3].lower()

    return {'names': names, 'sizes': sizes, 'is_terminal': is_terminal}


def parse_bookshelf_nets(nets_path: str, name_to_idx: Dict[str, int]) -> Tuple[List, List, List]:
    """
    Parse BookShelf .nets file.

    Args:
        name_to_idx: mapping from object name to index

    Returns:
        edges: list of (src_idx, dst_idx) pairs (star decomposition)
        edge_attrs: list of (src_dx, src_dy, dst_dx, dst_dy) pin offsets
        nets: list of nets, each net = [(node_idx, pin_dx, pin_dy), ...]
    """
    edges = []
    edge_attrs = []
    nets = []

    # Regex patterns
    header_pattern = re.compile(r'^\s*NetDegree\s*:\s*(\d+)\s+(\S+)')
    # Pin line: name direction [: offset_x offset_y]
    pin_pattern = re.compile(r'^\s*(\S+)\s+([BIO])\s*(?::\s*(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?))?')

    with open(nets_path, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        header_match = header_pattern.match(lines[i])
        i += 1
        if not header_match:
            continue

        degree = int(header_match.group(1))
        if degree < 2:
            i += degree
            continue

        # Collect pins for this net
        pins = []
        for _ in range(degree):
            if i >= len(lines):
                break
            pin_match = pin_pattern.match(lines[i])
            i += 1
            if pin_match:
                obj_name = pin_match.group(1)
                if obj_name not in name_to_idx:
                    continue
                dx = float(pin_match.group(3)) if pin_match.group(3) else 0.0
                dy = float(pin_match.group(4)) if pin_match.group(4) else 0.0
                pins.append((name_to_idx[obj_name], dx, dy))

        if len(pins) < 2:
            continue

        # Preserve full net for net-level HPWL computation
        nets.append(pins)

        # Star decomposition: first pin is source, rest are sinks
        # Create bidirectional edges for each source-sink pair
        src_idx, src_dx, src_dy = pins[0]
        for sink_idx, sink_dx, sink_dy in pins[1:]:
            # Forward edge
            edges.append((src_idx, sink_idx))
            edge_attrs.append((src_dx, src_dy, sink_dx, sink_dy))
            # Reverse edge
            edges.append((sink_idx, src_idx))
            edge_attrs.append((sink_dx, sink_dy, src_dx, src_dy))

    return edges, edge_attrs, nets


def parse_bookshelf_pl(pl_path: str, name_to_idx: Dict[str, int]) -> np.ndarray:
    """
    Parse BookShelf .pl file.

    Returns:
        positions: (V, 2) array of (x, y) positions (bottom-left corner)
    """
    V = len(name_to_idx)
    positions = np.zeros((V, 2), dtype=np.float32)

    pl_pattern = re.compile(r'^\s*(\S+)\s+(-?\d+(?:\.\d*)?)\s+(-?\d+(?:\.\d*)?)\s*:')

    with open(pl_path, 'r') as f:
        for line in f:
            match = pl_pattern.match(line)
            if match:
                name = match.group(1)
                if name in name_to_idx:
                    idx = name_to_idx[name]
                    positions[idx, 0] = float(match.group(2))
                    positions[idx, 1] = float(match.group(3))

    return positions


def load_bookshelf_circuit(
    circuit_dir: str,
    circuit_name: str,
    macros_only: bool = False,
    max_nodes: Optional[int] = None,
    seed: int = 42,
) -> Dict:
    """
    Load a BookShelf circuit and convert to ChipSAT format.

    Args:
        circuit_dir: path to directory containing .nodes, .nets, .pl files
        circuit_name: base name of circuit (e.g. "ibm01")
        macros_only: if True, keep only macro nodes (height > row height, excluding I/O pads)
        max_nodes: if set, subsample to this many nodes (preserving connectivity)
        seed: random seed for subsampling

    Returns:
        dict with:
            'node_features': (V, 2) component sizes (width, height)
            'edge_index': (2, E) int64 edge list
            'edge_attr': (E, 4) pin offsets
            'positions': (V, 2) reference placement (center coords, normalized to [-1, 1])
            'n_components': int
            'circuit_name': str
            'chip_size': (4,) [x_min, y_min, x_max, y_max] in original units
    """
    nodes_path = os.path.join(circuit_dir, f"{circuit_name}.nodes")
    nets_path = os.path.join(circuit_dir, f"{circuit_name}.nets")
    pl_path = os.path.join(circuit_dir, f"{circuit_name}.pl")

    # Parse files
    node_data = parse_bookshelf_nodes(nodes_path)
    names = node_data['names']
    sizes_dict = node_data['sizes']
    is_terminal = node_data['is_terminal']

    # Build initial index mapping
    name_to_idx = {name: i for i, name in enumerate(names)}

    # Parse placement positions (bottom-left corner, original units)
    raw_positions = parse_bookshelf_pl(pl_path, name_to_idx)

    # Build sizes array
    all_sizes = np.array([sizes_dict[n] for n in names], dtype=np.float32)

    # Parse nets first (needed for BFS subsampling)
    full_name_to_idx = {name: i for i, name in enumerate(names)}
    raw_edges, raw_edge_attrs, raw_nets = parse_bookshelf_nets(nets_path, full_name_to_idx)

    # Build adjacency list for BFS
    adj = [[] for _ in range(len(names))]
    for (src, dst), _ in zip(raw_edges, raw_edge_attrs):
        adj[src].append(dst)

    # Filtering: which nodes to keep
    if macros_only:
        # Macros = multi-row cells (height > standard cell row height), excluding I/O terminals.
        # This matches chipdiffusion's DEF/LEF CLASS BLOCK identification exactly.
        from collections import Counter
        non_term_heights = [sizes_dict[n][1] for n in names if not is_terminal[n]]
        row_height = Counter(non_term_heights).most_common(1)[0][0] if non_term_heights else 0
        keep_mask = np.array([
            (not is_terminal[n]) and (sizes_dict[n][1] > row_height)
            for n in names
        ])
        if keep_mask.sum() < 2:
            raise ValueError(
                f"Only {keep_mask.sum()} macros found in {circuit_name}. "
                f"This circuit may not have macros (e.g. ibm05)."
            )
    else:
        keep_mask = np.ones(len(names), dtype=bool)

    if max_nodes is not None and keep_mask.sum() > max_nodes:
        # BFS-based subsampling: preserves graph connectivity
        rng = np.random.default_rng(seed)
        eligible = np.where(keep_mask)[0]

        # Start BFS from a random eligible node
        start = rng.choice(eligible)
        visited = set()
        visited.add(int(start))
        frontier = [int(start)]

        while len(visited) < max_nodes and frontier:
            # Shuffle frontier for randomness
            rng.shuffle(frontier)
            next_frontier = []
            for node in frontier:
                for neighbor in adj[node]:
                    if neighbor not in visited and keep_mask[neighbor]:
                        visited.add(neighbor)
                        next_frontier.append(neighbor)
                        if len(visited) >= max_nodes:
                            break
                if len(visited) >= max_nodes:
                    break
            frontier = next_frontier

        new_keep = np.zeros(len(names), dtype=bool)
        for idx in visited:
            new_keep[idx] = True
        keep_mask = new_keep

    # Apply filter: remap indices
    old_to_new = {}
    new_names = []
    for old_idx, name in enumerate(names):
        if keep_mask[old_idx]:
            old_to_new[old_idx] = len(new_names)
            new_names.append(name)

    V = len(new_names)

    # Filtered sizes and positions
    keep_indices = np.array([i for i in range(len(names)) if keep_mask[i]])
    sizes = all_sizes[keep_indices]
    positions_bl = raw_positions[keep_indices]  # bottom-left

    # Convert positions to center coordinates
    positions_center = positions_bl + sizes / 2

    # Compute bounding box of the KEPT nodes (not full chip)
    # This ensures subsampled placements fill the [-1, 1] canvas
    kept_bl = positions_bl
    kept_tr = positions_bl + sizes
    bbox_x_min = kept_bl[:, 0].min()
    bbox_y_min = kept_bl[:, 1].min()
    bbox_x_max = kept_tr[:, 0].max()
    bbox_y_max = kept_tr[:, 1].max()

    # Add small padding (5%) to avoid components touching the boundary
    bbox_w = bbox_x_max - bbox_x_min
    bbox_h = bbox_y_max - bbox_y_min
    pad_x = bbox_w * 0.05
    pad_y = bbox_h * 0.05
    bbox_x_min -= pad_x
    bbox_y_min -= pad_y
    bbox_x_max += pad_x
    bbox_y_max += pad_y
    bbox_w = bbox_x_max - bbox_x_min
    bbox_h = bbox_y_max - bbox_y_min

    # Also store full chip size for reference
    all_bl = raw_positions
    all_tr = raw_positions + all_sizes
    chip_size = np.array([
        all_bl[:, 0].min(), all_bl[:, 1].min(),
        all_tr[:, 0].max(), all_tr[:, 1].max(),
    ], dtype=np.float32)

    # Normalize positions to [-1, 1] using the kept-node bounding box
    positions_norm = np.zeros_like(positions_center)
    positions_norm[:, 0] = 2.0 * (positions_center[:, 0] - bbox_x_min) / bbox_w - 1.0
    positions_norm[:, 1] = 2.0 * (positions_center[:, 1] - bbox_y_min) / bbox_h - 1.0

    # Normalize sizes to bounding-box-relative
    sizes_norm = np.zeros_like(sizes)
    sizes_norm[:, 0] = sizes[:, 0] / bbox_w * 2.0  # scale to [-1,1] canvas width
    sizes_norm[:, 1] = sizes[:, 1] / bbox_h * 2.0

    # Filter edges to only include kept nodes
    edges = []
    edge_attrs = []
    for (src, dst), (sdx, sdy, ddx, ddy) in zip(raw_edges, raw_edge_attrs):
        if src in old_to_new and dst in old_to_new:
            new_src = old_to_new[src]
            new_dst = old_to_new[dst]
            if new_src != new_dst:  # no self-loops
                edges.append((new_src, new_dst))
                # Normalize pin offsets to bounding-box-relative scale
                edge_attrs.append((
                    sdx / bbox_w * 2.0,
                    sdy / bbox_h * 2.0,
                    ddx / bbox_w * 2.0,
                    ddy / bbox_h * 2.0,
                ))

    if len(edges) == 0:
        # Fallback: create a simple chain if no edges survived filtering
        for i in range(V - 1):
            edges.append((i, i + 1))
            edges.append((i + 1, i))
            edge_attrs.append((0.0, 0.0, 0.0, 0.0))
            edge_attrs.append((0.0, 0.0, 0.0, 0.0))

    edge_index = np.array(edges, dtype=np.int64).T  # (2, E)
    edge_attr = np.array(edge_attrs, dtype=np.float32)  # (E, 4)

    # Filter and remap nets to kept nodes with normalized pin offsets
    filtered_nets = []
    for net in raw_nets:
        remapped = []
        for (node_idx, dx, dy) in net:
            if node_idx in old_to_new:
                remapped.append((
                    old_to_new[node_idx],
                    dx / bbox_w * 2.0,
                    dy / bbox_h * 2.0,
                ))
        if len(remapped) >= 2:
            filtered_nets.append(remapped)

    return {
        'node_features': sizes_norm,          # (V, 2) normalized component sizes
        'edge_index': edge_index,             # (2, E) edge list
        'edge_attr': edge_attr,               # (E, 4) normalized pin offsets
        'positions': positions_norm,          # (V, 2) reference placement in [-1, 1]
        'nets': filtered_nets,                # list of nets, each = [(node_idx, dx, dy), ...]
        'n_components': V,
        'circuit_name': circuit_name,
        'chip_size': chip_size,               # (4,) original units
    }


def load_benchmark_batch(
    circuit_dirs: List[str],
    circuit_names: List[str],
    max_nodes: Optional[int] = None,
    macros_only: bool = False,
    seed: int = 42,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load multiple circuits and batch them for training.

    Same return format as generate_chip_batch():
        node_features: (total_nodes, 2) component sizes
        edge_index: (2, total_edges) with global node indices
        edge_attr: (total_edges, 4) terminal offsets
        node_graph_idx: (total_nodes,) graph assignment
        initial_positions: (total_nodes, 2) randomized starting positions
        legal_positions: (total_nodes, 2) reference placements
    """
    if device is None:
        device = torch.device('cpu')

    rng = np.random.default_rng(seed)

    all_sizes = []
    all_edges_src = []
    all_edges_dst = []
    all_edge_attr = []
    all_legal = []
    all_graph_idx = []

    node_offset = 0

    for g, (cdir, cname) in enumerate(zip(circuit_dirs, circuit_names)):
        instance = load_bookshelf_circuit(
            cdir, cname,
            macros_only=macros_only,
            max_nodes=max_nodes,
            seed=seed + g,
        )

        V = instance['n_components']
        all_sizes.append(instance['node_features'])
        all_legal.append(instance['positions'])
        all_graph_idx.append(np.full(V, g, dtype=np.int64))

        # Offset edge indices for batching
        ei = instance['edge_index']
        all_edges_src.append(ei[0] + node_offset)
        all_edges_dst.append(ei[1] + node_offset)
        all_edge_attr.append(instance['edge_attr'])

        node_offset += V

    node_features = torch.tensor(
        np.concatenate(all_sizes, axis=0), dtype=torch.float32, device=device
    )
    edge_index = torch.tensor(
        np.array([np.concatenate(all_edges_src), np.concatenate(all_edges_dst)]),
        dtype=torch.long, device=device
    )
    edge_attr = torch.tensor(
        np.concatenate(all_edge_attr, axis=0), dtype=torch.float32, device=device
    )
    node_graph_idx = torch.tensor(
        np.concatenate(all_graph_idx), dtype=torch.long, device=device
    )
    legal_positions = torch.tensor(
        np.concatenate(all_legal, axis=0), dtype=torch.float32, device=device
    )

    # Randomized starting positions (uniform in [-1, 1])
    total_nodes = node_features.shape[0]
    initial_positions = torch.tensor(
        rng.uniform(-1.0, 1.0, size=(total_nodes, 2)).astype(np.float32),
        device=device,
    )

    return node_features, edge_index, edge_attr, node_graph_idx, initial_positions, legal_positions


# ─── Convenience functions for ICCAD04 ──────────────────────────────────────

ICCAD04_CIRCUITS = [f"ibm{i:02d}" for i in range(1, 19)]

# Approximate sizes (total nodes / terminals) for planning
ICCAD04_INFO = {
    "ibm01": {"nodes": 12752, "terminals": 246, "nets": 14111},
    "ibm02": {"nodes": 19601, "terminals": 259, "nets": 19584},
    "ibm03": {"nodes": 23136, "terminals": 283, "nets": 27401},
    "ibm04": {"nodes": 27507, "terminals": 287, "nets": 31970},
    "ibm05": {"nodes": 29347, "terminals": 1201, "nets": 28446},
    "ibm06": {"nodes": 32498, "terminals": 166, "nets": 34826},
    "ibm07": {"nodes": 45926, "terminals": 287, "nets": 48117},
    "ibm08": {"nodes": 51309, "terminals": 286, "nets": 50513},
    "ibm09": {"nodes": 53395, "terminals": 285, "nets": 60902},
    "ibm10": {"nodes": 69429, "terminals": 744, "nets": 75196},
    "ibm11": {"nodes": 70558, "terminals": 406, "nets": 81454},
    "ibm12": {"nodes": 71076, "terminals": 637, "nets": 77240},
    "ibm13": {"nodes": 84199, "terminals": 490, "nets": 99666},
    "ibm14": {"nodes": 147605, "terminals": 517, "nets": 152772},
    "ibm15": {"nodes": 161570, "terminals": 383, "nets": 186608},
    "ibm16": {"nodes": 183484, "terminals": 504, "nets": 190048},
    "ibm17": {"nodes": 185495, "terminals": 743, "nets": 189581},
    "ibm18": {"nodes": 210613, "terminals": 272, "nets": 201920},
}


def load_iccad04_circuit(
    benchmark_base: str,
    circuit_name: str = "ibm01",
    max_nodes: Optional[int] = None,
    macros_only: bool = False,
    seed: int = 42,
) -> Dict:
    """
    Load an ICCAD04 (IBM) circuit.

    Args:
        benchmark_base: path to benchmarks directory (e.g. "chipdiffusion/benchmarks")
        circuit_name: "ibm01" through "ibm18"
        max_nodes: subsample to this many nodes (None = all)
        macros_only: keep only terminal nodes
        seed: random seed for subsampling
    """
    circuit_dir = os.path.join(benchmark_base, "iccad04", "extracted", circuit_name)
    return load_bookshelf_circuit(circuit_dir, circuit_name, macros_only, max_nodes, seed)


def load_iccad04_batch(
    benchmark_base: str,
    circuit_names: List[str] = None,
    max_nodes: Optional[int] = 200,
    macros_only: bool = False,
    seed: int = 42,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load multiple IBM circuits as a training batch.

    Args:
        benchmark_base: path to benchmarks directory
        circuit_names: list of circuit names (default: ["ibm01"])
        max_nodes: subsample each circuit to this many nodes
        seed: random seed
        device: torch device
    """
    if circuit_names is None:
        circuit_names = ["ibm01"]

    circuit_dirs = [
        os.path.join(benchmark_base, "iccad04", "extracted", cn)
        for cn in circuit_names
    ]

    return load_benchmark_batch(
        circuit_dirs, circuit_names,
        max_nodes=max_nodes,
        macros_only=macros_only,
        seed=seed,
        device=device,
    )


# ─── Test ───────────────────────────────────────────────────────────────────

def test_benchmark_loader():
    """Test loading ibm01 benchmark."""
    import sys

    # Try to find benchmarks directory
    base_candidates = [
        os.path.join(os.path.dirname(__file__), 'benchmarks'),
        os.path.join(os.path.dirname(__file__), '..', 'chipdiffusion', 'benchmarks'),
    ]

    benchmark_base = None
    for candidate in base_candidates:
        test_path = os.path.join(candidate, 'iccad04', 'extracted', 'ibm01', 'ibm01.nodes')
        if os.path.exists(test_path):
            benchmark_base = candidate
            break

    if benchmark_base is None:
        print("ICCAD04 benchmarks not found. Run download_benchmarks.py first.")
        print(f"Searched in: {base_candidates}")
        return

    print("=" * 60)
    print("Testing Benchmark Loader")
    print("=" * 60)

    # Test single circuit load (full)
    print("\n--- ibm01 (full) ---")
    data = load_iccad04_circuit(benchmark_base, "ibm01")
    print(f"  Components: {data['n_components']}")
    print(f"  Edges: {data['edge_index'].shape[1]}")
    print(f"  Size range: [{data['node_features'].min():.6f}, {data['node_features'].max():.6f}]")
    print(f"  Position range: [{data['positions'].min():.3f}, {data['positions'].max():.3f}]")
    print(f"  Chip size (orig): {data['chip_size']}")

    # Test subsampled load
    for max_n in [50, 100, 200]:
        print(f"\n--- ibm01 (max_nodes={max_n}) ---")
        data = load_iccad04_circuit(benchmark_base, "ibm01", max_nodes=max_n)
        print(f"  Components: {data['n_components']}")
        print(f"  Edges: {data['edge_index'].shape[1]}")
        print(f"  Size range: [{data['node_features'].min():.6f}, {data['node_features'].max():.6f}]")
        print(f"  Position range: [{data['positions'].min():.3f}, {data['positions'].max():.3f}]")

    # Test batch loading
    print("\n--- Batch: ibm01 x 4 (max_nodes=100) ---")
    nf, ei, ea, ngi, pos, legal = load_iccad04_batch(
        benchmark_base, ["ibm01"] * 4, max_nodes=100, seed=42
    )
    print(f"  node_features: {nf.shape}")
    print(f"  edge_index: {ei.shape}")
    print(f"  edge_attr: {ea.shape}")
    print(f"  node_graph_idx: {ngi.shape}, unique: {ngi.unique().tolist()}")
    print(f"  positions: {pos.shape}")
    print(f"  legal_positions: {legal.shape}")

    # Verify no cross-graph edges
    src_graphs = ngi[ei[0]]
    dst_graphs = ngi[ei[1]]
    cross = (src_graphs != dst_graphs).sum().item()
    print(f"  Cross-graph edges: {cross} {'(OK)' if cross == 0 else '(FAIL)'}")

    # Test batch with different circuits
    print("\n--- Batch: ibm01 + ibm02 (max_nodes=100) ---")
    nf2, ei2, ea2, ngi2, pos2, legal2 = load_iccad04_batch(
        benchmark_base, ["ibm01", "ibm02"], max_nodes=100, seed=42
    )
    print(f"  node_features: {nf2.shape}")
    print(f"  edge_index: {ei2.shape}")
    print(f"  node_graph_idx unique: {ngi2.unique().tolist()}")

    print("\n" + "=" * 60)
    print("Benchmark loader tests complete")
    print("=" * 60)


if __name__ == "__main__":
    test_benchmark_loader()
