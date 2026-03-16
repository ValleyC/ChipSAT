"""
Routing-channel-aware constraints for CP-SAT macro placement.

Computes per-macro padding (inflated intervals) so that NoOverlap2D
automatically enforces minimum routing channel widths between macros.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional

SCALE = 10000  # must match cpsat_solver.py


def compute_routing_constraints(
    data: dict,
    sizes: np.ndarray,
    nets: list,
    min_tracks: int = 10,
    boundary_tracks: int = 20,
    track_pitch_um: float = 0.28,
) -> dict:
    """
    Compute routing channel constraints from physical parameters.

    Args:
        data: circuit data from load_chipbench_circuit()
        sizes: (N, 2) normalized macro sizes
        nets: net connectivity list
        min_tracks: minimum routing tracks between any two macros
        boundary_tracks: minimum routing tracks from die boundary
        track_pitch_um: metal track pitch in microns (0.28 for Nangate45 metal3/4)

    Returns:
        dict with:
            'pad_int': (N, 4) per-macro padding [left, right, bottom, top] in SCALE units
            'boundary_margin_int': int, boundary inset in SCALE units
    """
    N = sizes.shape[0]
    norm_bbox = data['_norm_bbox']  # (x_min, y_min, x_max, y_max) in DEF units
    def_units = data.get('_def_units', 2000)  # DEF units per micron

    # Die span in DEF units
    die_span_x = norm_bbox[2] - norm_bbox[0]
    die_span_y = norm_bbox[3] - norm_bbox[1]

    # Physical gap in DEF units
    gap_def = min_tracks * track_pitch_um * def_units
    boundary_gap_def = boundary_tracks * track_pitch_um * def_units

    # Convert to SCALE units: physical_gap_def / die_span_def * SCALE
    # The full die span maps to SCALE in CP-SAT integer space
    pad_x = int(round(gap_def / die_span_x * SCALE))
    pad_y = int(round(gap_def / die_span_y * SCALE))
    boundary_margin_x = int(round(boundary_gap_def / die_span_x * SCALE))
    boundary_margin_y = int(round(boundary_gap_def / die_span_y * SCALE))

    # Use the larger of x/y padding for uniform padding (v1: uniform on all edges)
    # Each macro gets half the gap on each side: if both macros have pad,
    # total gap = pad_left_i + pad_right_j >= gap
    # So per-side padding = gap / 2
    half_pad_x = max(1, pad_x // 2)
    half_pad_y = max(1, pad_y // 2)

    # Uniform padding for all macros (v1)
    pad_int = np.zeros((N, 4), dtype=int)
    pad_int[:, 0] = half_pad_x  # left
    pad_int[:, 1] = half_pad_x  # right
    pad_int[:, 2] = half_pad_y  # bottom
    pad_int[:, 3] = half_pad_y  # top

    # Boundary margin (use max of x/y for simplicity)
    boundary_margin_int = max(boundary_margin_x, boundary_margin_y)

    # Print summary
    gap_um = min_tracks * track_pitch_um
    print(f"  Routing constraints:")
    print(f"    Min channel: {min_tracks} tracks x {track_pitch_um}um = {gap_um:.1f}um")
    print(f"    Gap in DEF units: {gap_def:.0f}")
    print(f"    Half-pad in SCALE: x={half_pad_x}, y={half_pad_y}")
    print(f"    Boundary margin in SCALE: {boundary_margin_int}")
    print(f"    (SCALE={SCALE}, die_span={die_span_x:.0f}x{die_span_y:.0f} DEF units)")

    return {
        'pad_int': pad_int,
        'boundary_margin_int': boundary_margin_int,
    }
