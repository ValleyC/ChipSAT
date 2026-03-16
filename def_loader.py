"""
DEF/LEF Loader for ChiPBench circuits.

Parses DEF/LEF format directly and returns data structures compatible
with benchmark_loader.py, so we can reuse the existing CP-SAT placement
infrastructure without format conversion.

Also provides write_placement_def() to write macro positions back to DEF.

Usage:
    data = load_chipbench_circuit("/path/to/ChiPBench/dataset/data/bp_fe")
    # data has same keys as load_bookshelf_circuit() output
    # ... run placement ...
    write_placement_def(data, new_positions, "output.def")
"""

import os
import re
import numpy as np
from typing import Dict, List, Tuple, Optional


# ─── LEF Parsing ─────────────────────────────────────────────────────────────

def parse_lef_macros(lef_paths: List[str]) -> Dict[str, dict]:
    """
    Parse LEF files to extract macro definitions.

    Returns:
        dict: macro_type -> {
            'size': (width, height) in microns,
            'is_macro': bool (CLASS BLOCK),
            'pins': {pin_name: (cx, cy)} pin center offsets from macro origin
        }
    """
    macros = {}

    for lef_path in lef_paths:
        with open(lef_path, 'r') as f:
            text = f.read()

        # Split by MACRO keyword
        parts = text.split('MACRO ')
        for part in parts[1:]:
            lines = part.split('\n')
            name = lines[0].strip().rstrip(';').strip()

            # Parse class
            is_macro = False
            size = (0.0, 0.0)
            pins = {}

            for line in lines:
                line_s = line.strip()
                if line_s.startswith('CLASS BLOCK'):
                    is_macro = True
                size_match = re.match(r'SIZE\s+([\d.]+)\s+BY\s+([\d.]+)', line_s)
                if size_match:
                    size = (float(size_match.group(1)), float(size_match.group(2)))

            # Parse pins
            pin_sections = part.split('PIN ')
            for pin_sec in pin_sections[1:]:
                pin_name = pin_sec.split('\n')[0].strip()
                # Skip power pins
                if pin_name in ('VPWR', 'VPB', 'VNB', 'VGND', 'VSSD', 'VSSA', 'VDD', 'VSS'):
                    continue
                # Find first RECT
                rect_match = re.search(
                    r'RECT\s+([\d.e\-]+)\s+([\d.e\-]+)\s+([\d.e\-]+)\s+([\d.e\-]+)',
                    pin_sec
                )
                if rect_match:
                    x1 = float(rect_match.group(1))
                    y1 = float(rect_match.group(2))
                    x2 = float(rect_match.group(3))
                    y2 = float(rect_match.group(4))
                    pins[pin_name] = ((x1 + x2) / 2, (y1 + y2) / 2)

            if name:
                macros[name] = {
                    'size': size,
                    'is_macro': is_macro,
                    'pins': pins,
                }

    return macros


# ─── DEF Parsing ─────────────────────────────────────────────────────────────

def _parse_def_header(text: str) -> dict:
    """Parse DEF header for DESIGN name, UNITS, DIEAREA, and core area (from ROW definitions)."""
    design_match = re.search(r'DESIGN\s+(\S+)\s*;', text)
    design_name = design_match.group(1) if design_match else None

    units_match = re.search(r'UNITS DISTANCE MICRONS\s+(\d+)', text)
    units = int(units_match.group(1)) if units_match else 1000

    die_match = re.search(
        r'DIEAREA\s+\(\s*(\d+)\s+(\d+)\s*\)\s+\(\s*(\d+)\s+(\d+)\s*\)',
        text
    )
    if die_match:
        die_area = tuple(int(die_match.group(i)) for i in range(1, 5))
    else:
        die_area = (0, 0, 0, 0)

    # Parse core area from ROW definitions
    # ROW name site x y orient DO count BY 1 STEP step 0 ;
    row_pattern = re.compile(
        r'ROW\s+\S+\s+\S+\s+(\d+)\s+(\d+)\s+\S+\s+DO\s+(\d+)\s+BY\s+\d+\s+STEP\s+(\d+)'
    )
    rows = row_pattern.findall(text)
    if rows:
        row_xs = []
        row_ys = []
        row_x_maxs = []
        for x_str, y_str, count_str, step_str in rows:
            x, y, count, step = int(x_str), int(y_str), int(count_str), int(step_str)
            row_xs.append(x)
            row_ys.append(y)
            row_x_maxs.append(x + count * step)
        core_area = (min(row_xs), min(row_ys), max(row_x_maxs), max(row_ys))
    else:
        core_area = die_area

    return {'units': units, 'die_area': die_area, 'core_area': core_area, 'design_name': design_name}


def _parse_def_components(text: str, lef_macros: Dict) -> Tuple[List[dict], List[dict]]:
    """
    Parse COMPONENTS section.

    Returns:
        macros: list of {name, type, x, y, orient, size_microns}
        stdcells: list of {name, type, x, y, orient}
    """
    # Extract COMPONENTS section
    comp_match = re.search(r'COMPONENTS\s+\d+\s*;(.*?)END COMPONENTS', text, re.DOTALL)
    if not comp_match:
        return [], []

    comp_text = comp_match.group(1)

    # Pattern: - inst_name type + PLACED|FIXED ( x y ) orient ;
    pattern = re.compile(
        r'-\s+(\S+)\s+(\S+)\s+\+\s+(PLACED|FIXED)\s+\(\s*(-?\d+)\s+(-?\d+)\s*\)\s+(\S+)\s*;'
    )

    macros = []
    stdcells = []

    for match in pattern.finditer(comp_text):
        inst_name = match.group(1)
        inst_type = match.group(2)
        status = match.group(3)
        x = int(match.group(4))
        y = int(match.group(5))
        orient = match.group(6)

        if inst_type in lef_macros and lef_macros[inst_type]['is_macro']:
            macros.append({
                'name': inst_name,
                'type': inst_type,
                'x': x,
                'y': y,
                'orient': orient,
                'status': status,
                'size_microns': lef_macros[inst_type]['size'],
            })
        else:
            stdcells.append({
                'name': inst_name,
                'type': inst_type,
                'x': x,
                'y': y,
                'orient': orient,
            })

    return macros, stdcells


def _parse_def_pins(text: str) -> Dict[str, Tuple[int, int]]:
    """
    Parse PINS section for I/O pad positions.

    Returns:
        dict: pin_name -> (x, y) in DEF units
    """
    pins_match = re.search(r'PINS\s+\d+\s*;(.*?)END PINS', text, re.DOTALL)
    if not pins_match:
        return {}

    pins_text = pins_match.group(1)
    pins = {}

    # Each pin block: - pin_name + NET ... + PLACED ( x y ) orient ;
    # Multi-line, so process block by block
    blocks = pins_text.split('\n    - ')
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        # Pin name is first word
        name_match = re.match(r'(\S+)', block)
        if not name_match:
            continue
        pin_name = name_match.group(1)

        # Find PLACED position
        pos_match = re.search(r'PLACED\s+\(\s*(-?\d+)\s+(-?\d+)\s*\)', block)
        if pos_match:
            pins[pin_name] = (int(pos_match.group(1)), int(pos_match.group(2)))

    return pins


def _parse_def_nets(text: str) -> List[List[Tuple[str, str]]]:
    """
    Parse NETS section.

    Returns:
        list of nets, each net = [(component_name, pin_name), ...]
    """
    nets_match = re.search(r'NETS\s+\d+\s*;(.*?)END NETS', text, re.DOTALL)
    if not nets_match:
        return []

    nets_text = nets_match.group(1)
    nets = []

    # Split by net delimiter: each net starts with "- net_name"
    # Handle multi-line nets by joining continuation lines
    net_blocks = re.split(r'\n\s*-\s+', nets_text)

    for block in net_blocks:
        block = block.strip()
        if not block:
            continue

        # Extract all pin references: ( component_name pin_name )
        pin_refs = re.findall(r'\(\s*(\S+)\s+(\S+)\s*\)', block)
        if len(pin_refs) >= 2:
            nets.append(pin_refs)

    return nets


# ─── Main Loader ─────────────────────────────────────────────────────────────

def load_chipbench_circuit(
    data_dir: str,
    circuit_name: Optional[str] = None,
    use_reference: bool = True,
) -> Dict:
    """
    Load a ChiPBench circuit from DEF/LEF format.

    Args:
        data_dir: path to circuit data directory
                  (e.g. "/ChiPBench/dataset/data/bp_fe")
        circuit_name: optional name override (default: dirname)
        use_reference: if True, load positions from macro_placed.def;
                      if False, use pre_place.def (macros at origin)

    Returns:
        dict compatible with benchmark_loader.py output:
        - node_features: (V, 2) normalized macro sizes
        - edge_index: (2, E) edge list
        - edge_attr: (E, 4) pin offsets (zeros for center approx)
        - positions: (V, 2) reference placement in [-1, 1]
        - nets: list of nets, each = [(macro_idx, dx, dy), ...]
        - n_components: int
        - circuit_name: str
        - chip_size: (4,) die area in microns

        Plus metadata for writing back:
        - _macro_names: list of instance names
        - _macro_types: list of LEF types
        - _macro_orientations: list of orientations
        - _pre_place_def: path to pre_place.def
        - _def_units: MICRONS multiplier
        - _die_area_def: (4,) in DEF units
        - _norm_bbox: (x_min, y_min, x_max, y_max) for denormalization
        - _io_pins: dict pin_name -> (x, y) in DEF units
        - _lef_macros: parsed LEF macro data
    """
    if circuit_name is None:
        circuit_name = os.path.basename(data_dir)

    # Find LEF files
    lef_dir = os.path.join(data_dir, 'lef')
    lef_paths = []
    if os.path.isdir(lef_dir):
        for f in os.listdir(lef_dir):
            if f.endswith('.lef'):
                lef_paths.append(os.path.join(lef_dir, f))

    # Parse LEF
    lef_macros = parse_lef_macros(lef_paths)

    # Choose DEF file
    def_dir = os.path.join(data_dir, 'def')
    if use_reference:
        def_path = os.path.join(def_dir, 'macro_placed.def')
        if not os.path.exists(def_path):
            def_path = os.path.join(def_dir, 'pre_place.def')
    else:
        def_path = os.path.join(def_dir, 'pre_place.def')

    pre_place_path = os.path.join(def_dir, 'pre_place.def')

    with open(def_path, 'r') as f:
        def_text = f.read()

    # Parse DEF
    header = _parse_def_header(def_text)
    def_units = header['units']
    die_area = header['die_area']
    core_area = header['core_area']
    design_name = header['design_name']

    macros, stdcells = _parse_def_components(def_text, lef_macros)
    io_pins = _parse_def_pins(def_text)
    raw_nets = _parse_def_nets(def_text)

    if len(macros) == 0:
        raise ValueError(f"No macros found in {def_path}")

    V = len(macros)

    # Build macro name → index mapping
    macro_name_to_idx = {m['name']: i for i, m in enumerate(macros)}

    # Build component name → info for standard cells (for net resolution)
    stdcell_positions = {}
    for sc in stdcells:
        stdcell_positions[sc['name']] = (sc['x'], sc['y'])

    # Macro sizes in DEF units
    sizes_def = np.zeros((V, 2), dtype=np.float64)
    for i, m in enumerate(macros):
        w_microns, h_microns = m['size_microns']
        sizes_def[i, 0] = w_microns * def_units
        sizes_def[i, 1] = h_microns * def_units

    # Macro positions (bottom-left) in DEF units
    positions_bl_def = np.zeros((V, 2), dtype=np.float64)
    for i, m in enumerate(macros):
        positions_bl_def[i, 0] = m['x']
        positions_bl_def[i, 1] = m['y']

    # Center positions
    positions_center_def = positions_bl_def + sizes_def / 2

    # Build net hypergraph (macro-only HPWL)
    # For each net, collect pins on macros. Skip nets with <2 macro pins.
    nets_macro = []
    for net in raw_nets:
        macro_pins = []
        for comp_name, pin_name in net:
            if comp_name in macro_name_to_idx:
                idx = macro_name_to_idx[comp_name]
                macro_type = macros[idx]['type']
                # Get pin offset from LEF (in microns, convert to DEF units)
                lef_pin_data = lef_macros.get(macro_type, {}).get('pins', {})
                if pin_name in lef_pin_data:
                    px, py = lef_pin_data[pin_name]
                    dx = px * def_units
                    dy = py * def_units
                else:
                    # Center approximation
                    dx = sizes_def[idx, 0] / 2
                    dy = sizes_def[idx, 1] / 2
                # Pin offset from macro center
                pin_dx = dx - sizes_def[idx, 0] / 2
                pin_dy = dy - sizes_def[idx, 1] / 2
                macro_pins.append((idx, pin_dx, pin_dy))

        if len(macro_pins) >= 2:
            nets_macro.append(macro_pins)

    # Normalization: map to [-1, 1] canvas
    # Use CORE AREA (from ROW bounds) as the placement region, not die area.
    # ChiPBench pre_macro.py checks macros against core area bounds.
    core_x_min, core_y_min, core_x_max, core_y_max = core_area
    bbox_w = core_x_max - core_x_min
    bbox_h = core_y_max - core_y_min

    # Store normalization bbox for denormalization (core area)
    norm_bbox = (float(core_x_min), float(core_y_min),
                 float(core_x_max), float(core_y_max))

    # Normalized positions (center, [-1, 1]) — using core area bounds
    positions_norm = np.zeros((V, 2), dtype=np.float32)
    positions_norm[:, 0] = 2.0 * (positions_center_def[:, 0] - core_x_min) / bbox_w - 1.0
    positions_norm[:, 1] = 2.0 * (positions_center_def[:, 1] - core_y_min) / bbox_h - 1.0

    # Normalized sizes
    sizes_norm = np.zeros((V, 2), dtype=np.float32)
    sizes_norm[:, 0] = sizes_def[:, 0] / bbox_w * 2.0
    sizes_norm[:, 1] = sizes_def[:, 1] / bbox_h * 2.0

    # Normalize net pin offsets
    nets_norm = []
    for net in nets_macro:
        net_norm = []
        for (idx, dx, dy) in net:
            net_norm.append((idx, dx / bbox_w * 2.0, dy / bbox_h * 2.0))
        nets_norm.append(net_norm)

    # Build edge_index from nets (star decomposition, bidirectional)
    edges = []
    edge_attrs = []
    for net in nets_norm:
        if len(net) < 2:
            continue
        src_idx, src_dx, src_dy = net[0]
        for sink_idx, sink_dx, sink_dy in net[1:]:
            if src_idx != sink_idx:
                edges.append((src_idx, sink_idx))
                edge_attrs.append((src_dx, src_dy, sink_dx, sink_dy))
                edges.append((sink_idx, src_idx))
                edge_attrs.append((sink_dx, sink_dy, src_dx, src_dy))

    if len(edges) == 0:
        # Fallback chain
        for i in range(V - 1):
            edges.append((i, i + 1))
            edges.append((i + 1, i))
            edge_attrs.append((0.0, 0.0, 0.0, 0.0))
            edge_attrs.append((0.0, 0.0, 0.0, 0.0))

    edge_index = np.array(edges, dtype=np.int64).T  # (2, E)
    edge_attr = np.array(edge_attrs, dtype=np.float32)  # (E, 4)

    # Chip size in microns (die area)
    chip_size = np.array([
        die_area[0] / def_units,
        die_area[1] / def_units,
        die_area[2] / def_units,
        die_area[3] / def_units,
    ], dtype=np.float32)

    return {
        # Compatible with benchmark_loader.py
        'node_features': sizes_norm,
        'edge_index': edge_index,
        'edge_attr': edge_attr,
        'positions': positions_norm,
        'nets': nets_norm,
        'n_components': V,
        'circuit_name': circuit_name,
        'chip_size': chip_size,

        # Metadata for writing back
        '_macro_names': [m['name'] for m in macros],
        '_macro_types': [m['type'] for m in macros],
        '_macro_orientations': [m['orient'] for m in macros],
        '_pre_place_def': pre_place_path,
        '_def_units': def_units,
        '_die_area_def': die_area,
        '_core_area_def': core_area,
        '_design_name': design_name,
        '_norm_bbox': norm_bbox,
        '_io_pins': io_pins,
        '_lef_macros': lef_macros,
        '_sizes_def': sizes_def,
    }


# ─── DEF Writer ──────────────────────────────────────────────────────────────

def denormalize_positions(
    positions_norm: np.ndarray,
    norm_bbox: Tuple[float, float, float, float],
    sizes_def: np.ndarray,
) -> np.ndarray:
    """
    Convert normalized center positions [-1, 1] back to DEF bottom-left coordinates.

    Args:
        positions_norm: (V, 2) center positions in [-1, 1]
        norm_bbox: (x_min, y_min, x_max, y_max) in DEF units
        sizes_def: (V, 2) sizes in DEF units

    Returns:
        (V, 2) bottom-left positions in DEF integer units, clamped to die area
    """
    x_min, y_min, x_max, y_max = norm_bbox
    bbox_w = x_max - x_min
    bbox_h = y_max - y_min

    # Normalized [-1,1] → DEF center coordinates
    centers_def = np.zeros_like(positions_norm, dtype=np.float64)
    centers_def[:, 0] = (positions_norm[:, 0] + 1.0) / 2.0 * bbox_w + x_min
    centers_def[:, 1] = (positions_norm[:, 1] + 1.0) / 2.0 * bbox_h + y_min

    # Center → bottom-left
    bl_def = centers_def - sizes_def / 2

    # Round to manufacturing grid (10 DEF units = 0.005 microns for Nangate45)
    GRID = 10
    bl_int = (np.round(bl_def / GRID) * GRID).astype(np.int64)

    # Clamp to placement area: bottom-left >= area_min, bottom-left + size <= area_max
    bl_int[:, 0] = np.clip(bl_int[:, 0], int(x_min),
                            int(x_max) - sizes_def[:, 0].astype(np.int64))
    bl_int[:, 1] = np.clip(bl_int[:, 1], int(y_min),
                            int(y_max) - sizes_def[:, 1].astype(np.int64))

    # Re-snap after clamp
    bl_int = (bl_int // GRID) * GRID

    return bl_int


def write_placement_def(
    data: Dict,
    positions_norm: np.ndarray,
    output_path: str,
) -> str:
    """
    Write a macro-placed DEF for ChiPBench evaluation.

    Reads pre_place.def as template, replaces macro positions with our
    placement, marks them as FIXED, writes to output_path.

    Args:
        data: dict from load_chipbench_circuit
        positions_norm: (V, 2) normalized center positions in [-1, 1]
        output_path: where to write the output DEF

    Returns:
        output_path
    """
    # Denormalize
    bl_positions = denormalize_positions(
        positions_norm,
        data['_norm_bbox'],
        data['_sizes_def'],
    )

    # Build macro name → (x, y) mapping
    macro_placements = {}
    for i, name in enumerate(data['_macro_names']):
        macro_placements[name] = (int(bl_positions[i, 0]), int(bl_positions[i, 1]))

    # Read template DEF
    with open(data['_pre_place_def'], 'r') as f:
        lines = f.readlines()

    # Process line by line, replacing macro placements
    output_lines = []
    # DEF component lines may span multiple lines; handle single-line case
    # Pattern: - inst_name type + PLACED|FIXED ( x y ) orient ;
    comp_pattern = re.compile(
        r'^(\s*-\s+)(\S+)(\s+\S+\s+\+\s+)(?:PLACED|FIXED)(\s+\(\s*)-?\d+\s+-?\d+(\s*\)\s+\S+\s*;)'
    )

    for line in lines:
        match = comp_pattern.match(line)
        if match:
            inst_name = match.group(2)
            if inst_name in macro_placements:
                x, y = macro_placements[inst_name]
                new_line = (
                    f"{match.group(1)}{inst_name}{match.group(3)}"
                    f"FIXED{match.group(4)}{x} {y}{match.group(5)}\n"
                )
                output_lines.append(new_line)
                continue
        output_lines.append(line)

    with open(output_path, 'w') as f:
        f.writelines(output_lines)

    return output_path


# ─── Utilities ───────────────────────────────────────────────────────────────

def compute_macro_hpwl(positions: np.ndarray, nets: list) -> float:
    """
    Compute net-level HPWL for macro placement.

    Args:
        positions: (V, 2) center positions in any coordinate system
        nets: list of nets, each = [(macro_idx, dx, dy), ...]

    Returns:
        total HPWL
    """
    total = 0.0
    for net in nets:
        if len(net) < 2:
            continue
        xs = []
        ys = []
        for idx, dx, dy in net:
            xs.append(positions[idx, 0] + dx)
            ys.append(positions[idx, 1] + dy)
        total += max(xs) - min(xs) + max(ys) - min(ys)
    return total


def list_chipbench_circuits(chipbench_data_dir: str) -> List[str]:
    """List available circuit names in a ChiPBench dataset directory."""
    circuits = []
    for name in sorted(os.listdir(chipbench_data_dir)):
        def_dir = os.path.join(chipbench_data_dir, name, 'def')
        if os.path.isdir(def_dir):
            circuits.append(name)
    return circuits


# ─── Test ────────────────────────────────────────────────────────────────────

def test_loader(data_dir: str):
    """Test loading a ChiPBench circuit."""
    print(f"Loading circuit from {data_dir}...")
    data = load_chipbench_circuit(data_dir)

    print(f"  Circuit: {data['circuit_name']}")
    print(f"  Macros: {data['n_components']}")
    print(f"  Macro names: {data['_macro_names']}")
    print(f"  Macro types: {set(data['_macro_types'])}")
    print(f"  Die area (microns): {data['chip_size']}")
    print(f"  Die area (DEF): {data['_die_area_def']}")
    print(f"  DEF units: {data['_def_units']}")
    print(f"  Sizes (norm): min={data['node_features'].min(axis=0)}, "
          f"max={data['node_features'].max(axis=0)}")
    print(f"  Positions (norm): min={data['positions'].min(axis=0)}, "
          f"max={data['positions'].max(axis=0)}")
    print(f"  Nets (macro-only): {len(data['nets'])}")
    print(f"  Edges: {data['edge_index'].shape[1]}")
    print(f"  I/O pins: {len(data['_io_pins'])}")

    # HPWL
    hpwl = compute_macro_hpwl(data['positions'], data['nets'])
    print(f"  Reference macro HPWL (normalized): {hpwl:.4f}")

    # Test denormalization round-trip
    bl = denormalize_positions(
        data['positions'], data['_norm_bbox'], data['_sizes_def']
    )
    print(f"  Denormalized BL range: x=[{bl[:,0].min()}, {bl[:,0].max()}], "
          f"y=[{bl[:,1].min()}, {bl[:,1].max()}]")

    return data


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        test_loader(sys.argv[1])
    else:
        print("Usage: python def_loader.py <chipbench_circuit_dir>")
        print("Example: python def_loader.py /ChiPBench/dataset/data/bp_fe")
