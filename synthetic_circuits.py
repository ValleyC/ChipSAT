"""
Synthetic Circuit Generator for ML Placement Training

Generates diverse synthetic circuits with 5 topology families to improve
generalization of the heatmap GNN placement model.

Key design: topology is decoupled from positions.
  1. Assign macros to abstract modules (no spatial info)
  2. Generate nets from module membership + noisy affinity
  3. Generate feasible positions consistent with modules

Families:
  clustered    — local modules, 10-30% long-range
  hierarchical — 2-3 level module tree
  hub_bus      — high-fanout hub macros + local connections
  mixed        — local modules + 30-50% cross-module nets
  chain_tree   — DAG backbone + shortcut nets

Usage:
    python synthetic_circuits.py --n_circuits 100 --save_dir synthetic_data/
    python synthetic_circuits.py --n_circuits 10 --save_dir synthetic_data/ --skip_pseudo_labels
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from typing import Dict, List, Tuple, Optional


# ──────────────────────────────────────────────────────────────────────────
# Degree distribution (shared across families, matches ICCAD04 statistics)
# ──────────────────────────────────────────────────────────────────────────

DEGREE_PROBS = [0.55, 0.25, 0.12, 0.05, 0.03]  # for pin counts 2,3,4,5,6+
DEGREE_VALUES = [2, 3, 4, 5, 6]

FAMILIES = ['clustered', 'hierarchical', 'hub_bus', 'mixed', 'chain_tree']

# Per-family parameter ranges
FAMILY_PARAMS = {
    'clustered': {
        'density_range': (0.20, 0.50),
        'size_modes': ['uniform', 'bimodal', 'heavy_tail'],
        'size_mode_weights': [0.4, 0.3, 0.3],
        'pin_offset_scale': (0.6, 0.9),
        'intra_prob': (0.7, 0.9),
        'long_range_target': (0.10, 0.30),
    },
    'hierarchical': {
        'density_range': (0.15, 0.45),
        'size_modes': ['bimodal', 'heavy_tail', 'uniform'],
        'size_mode_weights': [0.5, 0.3, 0.2],
        'pin_offset_scale': (0.5, 0.8),
        'intra_prob': (0.75, 0.95),
        'long_range_target': (0.05, 0.20),
    },
    'hub_bus': {
        'density_range': (0.25, 0.55),
        'size_modes': ['bimodal', 'bimodal', 'heavy_tail'],
        'size_mode_weights': [0.6, 0.2, 0.2],
        'pin_offset_scale': (0.7, 1.0),
        'intra_prob': (0.5, 0.7),
        'long_range_target': (0.25, 0.50),
    },
    'mixed': {
        'density_range': (0.20, 0.55),
        'size_modes': ['uniform', 'bimodal', 'heavy_tail'],
        'size_mode_weights': [0.33, 0.34, 0.33],
        'pin_offset_scale': (0.5, 0.9),
        'intra_prob': (0.50, 0.70),
        'long_range_target': (0.30, 0.50),
    },
    'chain_tree': {
        'density_range': (0.15, 0.40),
        'size_modes': ['uniform', 'uniform', 'heavy_tail'],
        'size_mode_weights': [0.5, 0.3, 0.2],
        'pin_offset_scale': (0.6, 0.8),
        'intra_prob': (0.6, 0.8),
        'long_range_target': (0.10, 0.25),
    },
}


# ──────────────────────────────────────────────────────────────────────────
# Macro size generation
# ──────────────────────────────────────────────────────────────────────────

def generate_sizes(rng, V, size_mode, density_target):
    """Generate macro sizes (V, 2) in normalized canvas scale [-1, 1].

    Returns sizes such that total area ≈ density_target * canvas_area (4.0).
    """
    if size_mode == 'uniform':
        widths = rng.uniform(0.02, 0.15, size=V)
        heights = rng.uniform(0.02, 0.15, size=V)
    elif size_mode == 'bimodal':
        n_large = max(1, int(V * rng.uniform(0.15, 0.25)))
        n_small = V - n_large
        widths = np.concatenate([
            rng.uniform(0.10, 0.30, size=n_large),
            rng.uniform(0.02, 0.06, size=n_small),
        ])
        heights = np.concatenate([
            rng.uniform(0.10, 0.30, size=n_large),
            rng.uniform(0.02, 0.06, size=n_small),
        ])
        # Shuffle so large aren't all at the front
        perm = rng.permutation(V)
        widths = widths[perm]
        heights = heights[perm]
    elif size_mode == 'heavy_tail':
        log_widths = rng.normal(-2.5, 0.8, size=V)
        log_heights = rng.normal(-2.5, 0.8, size=V)
        widths = np.clip(np.exp(log_widths), 0.02, 0.30)
        heights = np.clip(np.exp(log_heights), 0.02, 0.30)
    else:
        raise ValueError(f"Unknown size_mode: {size_mode}")

    sizes = np.stack([widths, heights], axis=1).astype(np.float32)

    # Scale to match density target
    total_area = (sizes[:, 0] * sizes[:, 1]).sum()
    canvas_area = 4.0  # [-1, 1] x [-1, 1]
    target_area = density_target * canvas_area
    if total_area > 1e-6:
        scale = np.sqrt(target_area / total_area)
        sizes *= scale
        # Clip to reasonable range
        sizes = np.clip(sizes, 0.01, 0.50)

    return sizes


# ──────────────────────────────────────────────────────────────────────────
# Module assignment (abstract, no spatial info)
# ──────────────────────────────────────────────────────────────────────────

def assign_modules_balanced(rng, V, n_modules):
    """Assign macros to modules with roughly balanced partition."""
    assignments = np.zeros(V, dtype=np.int32)
    base = V // n_modules
    remainder = V % n_modules
    idx = 0
    perm = rng.permutation(V)
    for m in range(n_modules):
        count = base + (1 if m < remainder else 0)
        assignments[perm[idx:idx + count]] = m
        idx += count
    return assignments


def assign_modules_hierarchical(rng, V, n_top=None, sub_per_top=None):
    """Assign macros to a 2-level module hierarchy.

    Returns:
        assignments: (V,) sub-module IDs
        top_assignments: (V,) top-module IDs
        hierarchy: dict mapping top_id -> list of sub_ids
    """
    if n_top is None:
        n_top = max(2, int(rng.uniform(2, min(6, V // 5 + 1))))
    if sub_per_top is None:
        sub_per_top = max(1, int(rng.uniform(2, 4)))

    total_sub = n_top * sub_per_top
    # Assign macros to sub-modules (balanced)
    sub_assignments = assign_modules_balanced(rng, V, total_sub)
    # Map sub-module -> top-module
    sub_to_top = np.array([s // sub_per_top for s in range(total_sub)])
    top_assignments = sub_to_top[sub_assignments]

    hierarchy = {}
    for t in range(n_top):
        hierarchy[t] = list(range(t * sub_per_top, (t + 1) * sub_per_top))

    return sub_assignments, top_assignments, hierarchy


# ──────────────────────────────────────────────────────────────────────────
# Net generation (from module membership, not spatial proximity)
# ──────────────────────────────────────────────────────────────────────────

def sample_net_degree(rng):
    """Sample a net degree from the ICCAD04-like distribution."""
    r = rng.random()
    cumul = 0.0
    for p, d in zip(DEGREE_PROBS, DEGREE_VALUES):
        cumul += p
        if r < cumul:
            return d
    return DEGREE_VALUES[-1]


def generate_nets_clustered(rng, V, module_assignments, n_modules, M, intra_prob):
    """Generate nets with module-based locality."""
    module_members = [np.where(module_assignments == m)[0] for m in range(n_modules)]
    # Remove empty modules
    module_members = [mm for mm in module_members if len(mm) > 0]
    n_modules_eff = len(module_members)
    if n_modules_eff == 0:
        return []

    nets = []
    for _ in range(M):
        degree = sample_net_degree(rng)
        degree = min(degree, V)  # can't have more pins than macros

        # Pick home module
        home = rng.integers(0, n_modules_eff)
        pins = set()

        # First pin from home module
        if len(module_members[home]) > 0:
            pins.add(int(rng.choice(module_members[home])))

        # Remaining pins
        attempts = 0
        while len(pins) < degree and attempts < degree * 5:
            attempts += 1
            if rng.random() < intra_prob and len(module_members[home]) > 1:
                # Intra-module
                pin = int(rng.choice(module_members[home]))
            else:
                # Cross-module
                other = rng.integers(0, n_modules_eff)
                if len(module_members[other]) > 0:
                    pin = int(rng.choice(module_members[other]))
                else:
                    continue
            pins.add(pin)

        if len(pins) >= 2:
            nets.append(list(pins))

    return nets


def generate_nets_mixed(rng, V, module_assignments, n_modules, M, intra_prob, long_range_frac):
    """Generate nets with explicit local + long-range split.

    Unlike `clustered`, this explicitly constructs cross-module nets that span
    2-4 different modules (global signals), controlled by long_range_frac.
    """
    module_members = [np.where(module_assignments == m)[0] for m in range(n_modules)]
    module_members = [mm for mm in module_members if len(mm) > 0]
    n_modules_eff = len(module_members)
    if n_modules_eff == 0:
        return []

    n_long_range = int(M * long_range_frac)
    n_local = M - n_long_range
    nets = []

    # Local nets (intra-module)
    for _ in range(n_local):
        degree = sample_net_degree(rng)
        degree = min(degree, V)
        home = rng.integers(0, n_modules_eff)
        pins = set()
        if len(module_members[home]) > 0:
            pins.add(int(rng.choice(module_members[home])))
        attempts = 0
        while len(pins) < degree and attempts < degree * 5:
            attempts += 1
            if rng.random() < intra_prob and len(module_members[home]) > 1:
                pin = int(rng.choice(module_members[home]))
            else:
                other = rng.integers(0, n_modules_eff)
                if len(module_members[other]) > 0:
                    pin = int(rng.choice(module_members[other]))
                else:
                    continue
            pins.add(pin)
        if len(pins) >= 2:
            nets.append(list(pins))

    # Long-range nets (explicitly span 2-4 modules)
    for _ in range(n_long_range):
        degree = sample_net_degree(rng)
        degree = max(degree, 2)
        degree = min(degree, V)
        # Pick 2-4 distinct modules to span
        n_span = min(rng.integers(2, min(5, n_modules_eff + 1)), degree)
        span_modules = rng.choice(n_modules_eff, size=n_span, replace=False).tolist()
        pins = set()
        # At least one pin from each spanned module
        for m in span_modules:
            if len(module_members[m]) > 0:
                pins.add(int(rng.choice(module_members[m])))
        # Fill remaining pins from any of the spanned modules
        attempts = 0
        while len(pins) < degree and attempts < degree * 5:
            attempts += 1
            m = rng.choice(span_modules)
            if len(module_members[m]) > 0:
                pins.add(int(rng.choice(module_members[m])))
        if len(pins) >= 2:
            nets.append(list(pins))

    return nets


def generate_nets_hierarchical(rng, V, sub_assignments, top_assignments, hierarchy, M, params):
    """Generate nets with hierarchical module preference.

    Same sub-module: P=0.80, same top-module: P=0.15, cross-top: P=0.05
    """
    n_sub = sub_assignments.max() + 1
    sub_members = [np.where(sub_assignments == s)[0] for s in range(n_sub)]
    sub_members = {s: mm for s, mm in enumerate(sub_members) if len(mm) > 0}

    n_top = top_assignments.max() + 1
    top_members = [np.where(top_assignments == t)[0] for t in range(n_top)]
    top_members = {t: mm for t, mm in enumerate(top_members) if len(mm) > 0}

    nets = []
    for _ in range(M):
        degree = sample_net_degree(rng)
        degree = min(degree, V)

        # Pick home sub-module
        valid_subs = list(sub_members.keys())
        if not valid_subs:
            continue
        home_sub = rng.choice(valid_subs)
        home_top = home_sub // max(1, (n_sub // n_top))
        if home_top >= n_top:
            home_top = n_top - 1

        pins = set()
        if home_sub in sub_members and len(sub_members[home_sub]) > 0:
            pins.add(int(rng.choice(sub_members[home_sub])))

        attempts = 0
        while len(pins) < degree and attempts < degree * 5:
            attempts += 1
            r = rng.random()
            if r < 0.80 and home_sub in sub_members:
                # Same sub-module
                pin = int(rng.choice(sub_members[home_sub]))
            elif r < 0.95 and home_top in top_members:
                # Same top-module, different sub
                pin = int(rng.choice(top_members[home_top]))
            else:
                # Cross-top-module
                other_top = rng.choice(list(top_members.keys()))
                pin = int(rng.choice(top_members[other_top]))
            pins.add(pin)

        if len(pins) >= 2:
            nets.append(list(pins))

    return nets


def generate_nets_hub_bus(rng, V, M):
    """Generate hub/bus topology: few high-fanout hubs + local connections."""
    n_hubs = max(2, min(8, V // 15))
    hub_indices = rng.choice(V, size=n_hubs, replace=False).tolist()
    non_hubs = [i for i in range(V) if i not in hub_indices]

    nets = []

    # Hub nets: each hub connects to a fraction of all macros
    for hub in hub_indices:
        n_connections = max(2, int(V * rng.uniform(0.15, 0.40)))
        # Split into multiple nets (6-15 pins each)
        connected = rng.choice(V, size=min(n_connections, V), replace=False).tolist()
        i = 0
        while i < len(connected):
            fan = min(int(rng.uniform(4, 12)), len(connected) - i)
            net_pins = {hub}
            for j in range(fan):
                net_pins.add(connected[i + j])
            if len(net_pins) >= 2:
                nets.append(list(net_pins))
            i += fan

    # Local nets between non-hubs (fill up to M)
    remaining = M - len(nets)
    if remaining > 0 and len(non_hubs) >= 2:
        for _ in range(remaining):
            degree = min(sample_net_degree(rng), len(non_hubs))
            degree = max(2, min(degree, 4))  # keep local nets small
            pins = set(rng.choice(non_hubs, size=degree, replace=False).tolist())
            if len(pins) >= 2:
                nets.append(list(pins))

    return nets


def generate_nets_chain_tree(rng, V, M):
    """Generate DAG/tree backbone + shortcut nets."""
    # Build random tree: each macro (except root) has one parent
    order = rng.permutation(V)
    parent = np.full(V, -1, dtype=np.int32)
    children_count = np.zeros(V, dtype=np.int32)
    max_children = 3

    for i in range(1, V):
        # Pick parent from earlier in order, respecting max children
        candidates = [order[j] for j in range(i) if children_count[order[j]] < max_children]
        if not candidates:
            candidates = [order[j] for j in range(i)]
        p = rng.choice(candidates)
        parent[order[i]] = p
        children_count[p] += 1

    nets = []
    # Tree edges as 2-pin nets
    for node in range(V):
        if parent[node] >= 0:
            nets.append([int(parent[node]), int(node)])

    # Shortcut nets (10-20% random)
    n_shortcuts = max(1, int(len(nets) * rng.uniform(0.10, 0.20)))
    for _ in range(n_shortcuts):
        degree = sample_net_degree(rng)
        degree = min(degree, V)
        pins = set(rng.choice(V, size=degree, replace=False).tolist())
        if len(pins) >= 2:
            nets.append(list(pins))

    # Fill remaining to reach M
    remaining = M - len(nets)
    for _ in range(max(0, remaining)):
        degree = sample_net_degree(rng)
        degree = min(degree, V)
        # Prefer nearby in tree order
        start = rng.integers(0, V)
        window = min(V, max(5, V // 4))
        candidates = [(start + j) % V for j in range(window)]
        degree = min(degree, len(candidates))
        pins = set(rng.choice(candidates, size=degree, replace=False).tolist())
        if len(pins) >= 2:
            nets.append(list(pins))

    return nets


# ──────────────────────────────────────────────────────────────────────────
# Pin offset generation
# ──────────────────────────────────────────────────────────────────────────

def add_pin_offsets(rng, nets_indices, sizes, pin_offset_scale):
    """Convert index-only nets to (node_idx, pin_dx, pin_dy) format.

    Pin offsets are sampled uniformly within the macro body.
    """
    nets_with_pins = []
    for net in nets_indices:
        pins = []
        for node_idx in net:
            w, h = sizes[node_idx]
            dx = rng.uniform(-w / 2, w / 2) * pin_offset_scale
            dy = rng.uniform(-h / 2, h / 2) * pin_offset_scale
            pins.append((int(node_idx), float(dx), float(dy)))
        nets_with_pins.append(pins)
    return nets_with_pins


# ──────────────────────────────────────────────────────────────────────────
# Star decomposition (matching benchmark_loader.py)
# ──────────────────────────────────────────────────────────────────────────

def star_decompose(nets):
    """Convert nets to bidirectional star-decomposed edges.

    Returns:
        edge_index: (2, E) int64
        edge_attr:  (E, 4) float32 — [src_dx, src_dy, dst_dx, dst_dy]
    """
    edges_src, edges_dst = [], []
    edge_attrs = []

    for net in nets:
        if len(net) < 2:
            continue
        src_idx, src_dx, src_dy = net[0]
        for sink_idx, sink_dx, sink_dy in net[1:]:
            if src_idx == sink_idx:
                continue
            # Forward edge
            edges_src.append(src_idx)
            edges_dst.append(sink_idx)
            edge_attrs.append([src_dx, src_dy, sink_dx, sink_dy])
            # Reverse edge
            edges_src.append(sink_idx)
            edges_dst.append(src_idx)
            edge_attrs.append([sink_dx, sink_dy, src_dx, src_dy])

    if len(edges_src) == 0:
        # Fallback chain
        V = max(max(p[0] for p in net) for net in nets if len(net) > 0) + 1
        for i in range(V - 1):
            edges_src.extend([i, i + 1])
            edges_dst.extend([i + 1, i])
            edge_attrs.extend([[0, 0, 0, 0], [0, 0, 0, 0]])

    edge_index = np.array([edges_src, edges_dst], dtype=np.int64)
    edge_attr = np.array(edge_attrs, dtype=np.float32)
    return edge_index, edge_attr


# ──────────────────────────────────────────────────────────────────────────
# Reference placement (module-aware, non-overlapping)
# ──────────────────────────────────────────────────────────────────────────

def generate_reference_placement(rng, V, sizes, module_assignments, n_modules):
    """Generate non-overlapping placement with module-aware spatial grouping.

    Assigns each module a rough region, then places macros within their region.
    """
    positions = np.zeros((V, 2), dtype=np.float32)
    placed = np.zeros(V, dtype=bool)

    # Assign module regions: divide canvas into a grid
    grid_dim = max(2, int(np.ceil(np.sqrt(n_modules))))
    cell_w = 2.0 / grid_dim
    cell_h = 2.0 / grid_dim

    module_regions = {}
    for m in range(n_modules):
        row = m // grid_dim
        col = m % grid_dim
        cx = -1.0 + (col + 0.5) * cell_w
        cy = -1.0 + (row + 0.5) * cell_h
        # Add noise to region center
        cx += rng.uniform(-cell_w * 0.2, cell_w * 0.2)
        cy += rng.uniform(-cell_h * 0.2, cell_h * 0.2)
        module_regions[m] = (cx, cy, cell_w * 0.8, cell_h * 0.8)

    # Place macros largest-first
    areas = sizes[:, 0] * sizes[:, 1]
    order = np.argsort(-areas)

    for idx in order:
        w, h = sizes[idx]
        m = module_assignments[idx]
        rcx, rcy, rw, rh = module_regions[m]

        success = False
        # Try within module region first
        for attempt in range(80):
            x = rng.uniform(rcx - rw / 2, rcx + rw / 2)
            y = rng.uniform(rcy - rh / 2, rcy + rh / 2)
            # Clamp to canvas
            x = np.clip(x, -1.0 + w / 2 + 0.01, 1.0 - w / 2 - 0.01)
            y = np.clip(y, -1.0 + h / 2 + 0.01, 1.0 - h / 2 - 0.01)

            if not _overlaps_any(x, y, w, h, positions, sizes, placed):
                positions[idx] = [x, y]
                placed[idx] = True
                success = True
                break

        if not success:
            # Expand to full canvas
            for attempt in range(120):
                x = rng.uniform(-1.0 + w / 2 + 0.01, 1.0 - w / 2 - 0.01)
                y = rng.uniform(-1.0 + h / 2 + 0.01, 1.0 - h / 2 - 0.01)
                if not _overlaps_any(x, y, w, h, positions, sizes, placed):
                    positions[idx] = [x, y]
                    placed[idx] = True
                    success = True
                    break

        if not success:
            # Last resort: shrink and place
            scale = 0.9
            for shrink_attempt in range(5):
                w_s, h_s = w * scale, h * scale
                for attempt in range(50):
                    x = rng.uniform(-1.0 + w_s / 2 + 0.01, 1.0 - w_s / 2 - 0.01)
                    y = rng.uniform(-1.0 + h_s / 2 + 0.01, 1.0 - h_s / 2 - 0.01)
                    if not _overlaps_any(x, y, w_s, h_s, positions, sizes, placed):
                        positions[idx] = [x, y]
                        sizes[idx] = [w_s, h_s]
                        placed[idx] = True
                        success = True
                        break
                if success:
                    break
                scale *= 0.9

        if not success:
            # Force place at random position
            positions[idx] = [rng.uniform(-0.8, 0.8), rng.uniform(-0.8, 0.8)]
            placed[idx] = True

    return positions, sizes


def _overlaps_any(x, y, w, h, positions, sizes, placed):
    """Check if a macro at (x, y) with size (w, h) overlaps any placed macro."""
    for i in range(len(positions)):
        if not placed[i]:
            continue
        ox, oy = positions[i]
        ow, oh = sizes[i]
        if (abs(x - ox) < (w + ow) / 2 - 1e-6 and
                abs(y - oy) < (h + oh) / 2 - 1e-6):
            return True
    return False


# ──────────────────────────────────────────────────────────────────────────
# Main generator
# ──────────────────────────────────────────────────────────────────────────

def generate_circuit(seed, V=None, family=None, v_range=(30, 300)):
    """Generate a single synthetic circuit.

    Args:
        seed: random seed
        V: number of macros (None = sample from v_range)
        family: generator family (None = random)
        v_range: (min, max) macro count range

    Returns:
        data: dict matching load_bookshelf_circuit() output
        metadata: dict with generation parameters
    """
    rng = np.random.default_rng(seed)

    # Choose family
    if family is None:
        family = rng.choice(FAMILIES)
    params = FAMILY_PARAMS[family]

    # Choose V
    if V is None:
        V = int(rng.integers(v_range[0], v_range[1] + 1))

    # Choose size mode (biased by family)
    size_mode = rng.choice(params['size_modes'], p=params['size_mode_weights'])

    # Choose density
    density = rng.uniform(*params['density_range'])

    # Choose pin offset scale
    pin_offset_scale = rng.uniform(*params['pin_offset_scale'])

    # Choose intra-module probability
    intra_prob = rng.uniform(*params['intra_prob'])

    # ── Stage 1: Generate macro sizes ──
    sizes = generate_sizes(rng, V, size_mode, density)

    # ── Stage 2: Assign modules (abstract, no spatial info) ──
    n_modules = max(2, int(rng.uniform(3, max(4, V // 15 + 1))))

    if family == 'hierarchical':
        sub_assignments, top_assignments, hierarchy = assign_modules_hierarchical(rng, V)
        n_modules_eff = sub_assignments.max() + 1
        module_assignments = sub_assignments
    elif family == 'hub_bus':
        # For hub_bus, module assignment is less important; use simple partition
        module_assignments = assign_modules_balanced(rng, V, n_modules)
        n_modules_eff = n_modules
    elif family == 'chain_tree':
        module_assignments = assign_modules_balanced(rng, V, n_modules)
        n_modules_eff = n_modules
    else:
        module_assignments = assign_modules_balanced(rng, V, n_modules)
        n_modules_eff = n_modules

    # ── Stage 3: Generate nets (from module membership) ──
    M = int(V * rng.uniform(1.5, 3.0))

    if family == 'clustered':
        nets_indices = generate_nets_clustered(
            rng, V, module_assignments, n_modules_eff, M, intra_prob)
    elif family == 'hierarchical':
        nets_indices = generate_nets_hierarchical(
            rng, V, sub_assignments, top_assignments, hierarchy, M, params)
    elif family == 'hub_bus':
        nets_indices = generate_nets_hub_bus(rng, V, M)
    elif family == 'mixed':
        long_range_frac = rng.uniform(*params['long_range_target'])
        nets_indices = generate_nets_mixed(
            rng, V, module_assignments, n_modules_eff, M, intra_prob, long_range_frac)
    elif family == 'chain_tree':
        nets_indices = generate_nets_chain_tree(rng, V, M)
    else:
        raise ValueError(f"Unknown family: {family}")

    # Ensure every macro appears in at least one net
    connected = set()
    for net in nets_indices:
        for idx in net:
            connected.add(idx)
    unconnected = [i for i in range(V) if i not in connected]
    for uc in unconnected:
        # Add a 2-pin net connecting to a random connected macro
        partner = rng.integers(0, V)
        while partner == uc:
            partner = rng.integers(0, V)
        nets_indices.append([uc, int(partner)])

    # ── Stage 4: Generate reference placement BEFORE pin offsets ──
    # Placement may shrink macros that don't fit, so sizes can change.
    # Pin offsets must be sampled from the final sizes to stay consistent.
    positions, sizes = generate_reference_placement(
        rng, V, sizes.copy(), module_assignments, n_modules_eff)

    # ── Stage 5: Add pin offsets (from final sizes) + star decomposition ──
    nets = add_pin_offsets(rng, nets_indices, sizes, pin_offset_scale)
    edge_index, edge_attr = star_decompose(nets)

    # Compute actual statistics
    actual_density = (sizes[:, 0] * sizes[:, 1]).sum() / 4.0
    mean_degree = np.mean([len(net) for net in nets]) if nets else 0.0

    # Count long-range nets (cross-module)
    long_range_count = 0
    for net in nets_indices:
        modules_in_net = set(module_assignments[idx] for idx in net)
        if len(modules_in_net) > 1:
            long_range_count += 1
    long_range_frac = long_range_count / max(1, len(nets_indices))

    # Module balance: std(module_sizes) / mean(module_sizes)
    module_sizes_arr = np.array([np.sum(module_assignments == m) for m in range(n_modules_eff)])
    module_sizes_arr = module_sizes_arr[module_sizes_arr > 0]
    if len(module_sizes_arr) > 1:
        module_balance = 1.0 - np.std(module_sizes_arr) / max(1e-6, np.mean(module_sizes_arr))
    else:
        module_balance = 1.0

    circuit_name = f"synth_{seed:04d}"

    data = {
        'node_features': sizes.astype(np.float32),
        'edge_index': edge_index,
        'edge_attr': edge_attr,
        'positions': positions.astype(np.float32),
        'nets': nets,
        'n_components': int(V),
        'circuit_name': circuit_name,
    }

    metadata = {
        'circuit_name': circuit_name,
        'family': family,
        'size_mode': size_mode,
        'V': int(V),
        'M': len(nets),
        'E': int(edge_index.shape[1]),
        'density': float(actual_density),
        'mean_net_degree': float(mean_degree),
        'long_range_frac': float(long_range_frac),
        'pin_offset_scale': float(pin_offset_scale),
        'module_count': int(n_modules_eff),
        'module_balance': float(module_balance),
        'seed': int(seed),
    }

    return data, metadata


# ──────────────────────────────────────────────────────────────────────────
# Pseudo-label generation
# ──────────────────────────────────────────────────────────────────────────

def generate_pseudo_label(data, n_seeds=3, n_iterations=None, verbose=True):
    """Generate pseudo-labels via ALNS from random starts.

    Args:
        data: circuit dict
        n_seeds: number of random starts
        n_iterations: ALNS iterations (None = adaptive based on V)
        verbose: print progress

    Returns:
        labels: list of dicts with 'positions', 'hpwl', 'post_legal_hpwl'
    """
    from cpsat_solver import legalize, compute_net_hpwl
    from lns_solver import LNSSolver

    V = data['n_components']
    sizes = data['node_features']
    nets = data['nets']
    edge_index = data['edge_index']

    if n_iterations is None:
        if V <= 100:
            n_iterations = 100
        elif V <= 200:
            n_iterations = 150
        else:
            n_iterations = 200

    labels = []
    for s in range(n_seeds):
        rng = np.random.default_rng(s * 1000 + 7)
        init_pos = rng.uniform(-0.8, 0.8, size=(V, 2)).astype(np.float32)

        # Legalize
        legal_pos = legalize(init_pos, sizes, time_limit=30.0, window_fraction=0.5)
        if legal_pos is None:
            if verbose:
                print(f"  Seed {s}: legalization failed, skipping")
            continue

        post_legal_hpwl = compute_net_hpwl(legal_pos, sizes, nets)

        # ALNS
        solver = LNSSolver(
            positions=legal_pos,
            sizes=sizes,
            nets=nets,
            edge_index=edge_index,
            subset_size=min(30, max(10, V // 5)),
            window_fraction=0.15,
            cpsat_time_limit=0.3,
            seed=s,
            congestion_weight=0.0,
        )
        result = solver.solve(n_iterations=n_iterations, log_every=50, verbose=verbose)

        labels.append({
            'positions': result['best_positions'],
            'hpwl': float(result['best_hpwl']),
            'post_legal_hpwl': float(post_legal_hpwl),
            'legalization_distortion': float((post_legal_hpwl - result['best_hpwl']) / max(1e-6, result['best_hpwl'])),
        })

        if verbose:
            print(f"  Seed {s}: post-legal={post_legal_hpwl:.1f}, post-ALNS={result['best_hpwl']:.1f}")

    return labels


# ──────────────────────────────────────────────────────────────────────────
# Dataset generation
# ──────────────────────────────────────────────────────────────────────────

def generate_dataset(
    n_circuits,
    save_dir,
    v_range=(30, 300),
    n_seeds=3,
    skip_pseudo_labels=False,
    holdout_family=None,
    seed_offset=0,
    verbose=True,
):
    """Generate a full synthetic dataset.

    Args:
        n_circuits: number of circuits to generate
        save_dir: output directory
        v_range: (min, max) macro count
        n_seeds: pseudo-label seeds per circuit
        skip_pseudo_labels: if True, only generate circuits (no ALNS)
        holdout_family: family to reserve for val_unseen split
        seed_offset: offset for random seeds
        verbose: print progress
    """
    circuits_dir = os.path.join(save_dir, 'circuits')
    labels_dir = os.path.join(save_dir, 'pseudo_labels')
    os.makedirs(circuits_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    summary = []
    total_start = time.time()

    for i in range(n_circuits):
        seed = seed_offset + i
        t0 = time.time()

        if verbose:
            print(f"\n{'='*50}")
            print(f"Circuit {i+1}/{n_circuits} (seed={seed})")
            print(f"{'='*50}")

        data, metadata = generate_circuit(seed, v_range=v_range)

        # Save circuit
        circuit_path = os.path.join(circuits_dir, f"{data['circuit_name']}.npz")
        np.savez_compressed(
            circuit_path,
            node_features=data['node_features'],
            edge_index=data['edge_index'],
            edge_attr=data['edge_attr'],
            positions=data['positions'],
            nets=np.array(data['nets'], dtype=object),
            n_components=data['n_components'],
            circuit_name=data['circuit_name'],
        )

        if verbose:
            print(f"  Family={metadata['family']}, V={metadata['V']}, "
                  f"M={metadata['M']}, density={metadata['density']:.3f}")

        # Assign split
        if holdout_family and metadata['family'] == holdout_family:
            metadata['split'] = 'val_unseen'
        else:
            r = np.random.default_rng(seed).random()
            if r < 0.80:
                metadata['split'] = 'train'
            elif r < 0.90:
                metadata['split'] = 'val_seen'
            else:
                metadata['split'] = 'val_unseen'

        # Generate pseudo-labels
        if not skip_pseudo_labels:
            labels = generate_pseudo_label(data, n_seeds=n_seeds, verbose=verbose)

            if labels:
                best_label = min(labels, key=lambda l: l['hpwl'])
                worst_label = max(labels, key=lambda l: l['hpwl'])

                for s, label in enumerate(labels):
                    label_path = os.path.join(labels_dir, f"{data['circuit_name']}_seed{s}.npz")
                    np.savez_compressed(
                        label_path,
                        positions=label['positions'],
                        hpwl=label['hpwl'],
                        sizes=data['node_features'],
                    )

                metadata['pseudo_labels'] = {
                    'best_hpwl': float(best_label['hpwl']),
                    'worst_hpwl': float(worst_label['hpwl']),
                    'legalization_distortion': float(best_label['legalization_distortion']),
                    'n_seeds_succeeded': len(labels),
                    'kept': True,
                }
            else:
                metadata['pseudo_labels'] = {
                    'best_hpwl': None,
                    'worst_hpwl': None,
                    'legalization_distortion': None,
                    'n_seeds_succeeded': 0,
                    'kept': False,
                }

        elapsed = time.time() - t0
        metadata['generation_time_s'] = float(elapsed)
        summary.append(metadata)

        if verbose:
            print(f"  Time: {elapsed:.1f}s")

    # Quality filtering: drop circuits with degenerate HPWL
    if not skip_pseudo_labels:
        _apply_quality_filter(summary, verbose)

    # Save summary
    summary_path = os.path.join(save_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    total_elapsed = time.time() - total_start
    n_kept = sum(1 for s in summary if s.get('pseudo_labels', {}).get('kept', True))

    if verbose:
        print(f"\n{'='*50}")
        print(f"Dataset generation complete")
        print(f"  Total circuits: {n_circuits}")
        print(f"  Kept after filtering: {n_kept}")
        print(f"  Total time: {total_elapsed:.0f}s")
        print(f"  Saved to: {save_dir}")
        print(f"{'='*50}")

    return summary


def _apply_quality_filter(summary, verbose=True):
    """Filter out circuits with degenerate pseudo-label quality.

    Drop circuits whose best HPWL > 2x median for their V range.
    """
    # Group by V range
    ranges = [(30, 100), (100, 200), (200, 500)]

    for v_lo, v_hi in ranges:
        in_range = [s for s in summary
                    if v_lo <= s['V'] < v_hi
                    and s.get('pseudo_labels', {}).get('best_hpwl') is not None]
        if len(in_range) < 3:
            continue

        hpwls = [s['pseudo_labels']['best_hpwl'] for s in in_range]
        median_hpwl = np.median(hpwls)
        threshold = median_hpwl * 2.0

        for s in in_range:
            if s['pseudo_labels']['best_hpwl'] > threshold:
                s['pseudo_labels']['kept'] = False
                if verbose:
                    print(f"  FILTERED: {s['circuit_name']} "
                          f"(hpwl={s['pseudo_labels']['best_hpwl']:.1f} > "
                          f"threshold={threshold:.1f})")


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Synthetic Circuit Generator')
    parser.add_argument('--n_circuits', type=int, default=10,
                        help='Number of circuits to generate')
    parser.add_argument('--save_dir', type=str, default='synthetic_data',
                        help='Output directory')
    parser.add_argument('--v_min', type=int, default=30,
                        help='Minimum macro count')
    parser.add_argument('--v_max', type=int, default=300,
                        help='Maximum macro count')
    parser.add_argument('--n_seeds', type=int, default=3,
                        help='Number of ALNS seeds per circuit')
    parser.add_argument('--skip_pseudo_labels', action='store_true',
                        help='Only generate circuits, skip ALNS pseudo-labels')
    parser.add_argument('--holdout_family', type=str, default=None,
                        choices=FAMILIES,
                        help='Family to hold out for val_unseen split')
    parser.add_argument('--seed_offset', type=int, default=0,
                        help='Offset for random seeds')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress verbose output')

    args = parser.parse_args()

    generate_dataset(
        n_circuits=args.n_circuits,
        save_dir=args.save_dir,
        v_range=(args.v_min, args.v_max),
        n_seeds=args.n_seeds,
        skip_pseudo_labels=args.skip_pseudo_labels,
        holdout_family=args.holdout_family,
        seed_offset=args.seed_offset,
        verbose=not args.quiet,
    )


if __name__ == '__main__':
    main()
