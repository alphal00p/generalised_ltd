from __future__ import annotations
from typing import Any, Dict, Optional, Sequence
import json, random
import pydot
import mpmath as mp
from . import graph_io as GIO
from . import structure as ST
from . import validator as VAL
from .pretty import render_pretty
from .orientation_bundle import build_pure_cff_bundle, build_pure_ltd_bundle, build_hybrid_bundle_raw


def load_dot_graph(path: str):
    graphs = pydot.graph_from_dot_file(path)
    if not graphs:
        raise RuntimeError(f'No DOT graph found in {path}')
    return graphs[0]


def validate_graph(dot) -> dict:
    parsed = GIO.parse_dot_graph(dot)
    return VAL.validate_parsed_graph(parsed)


def _build_bundle(parsed, family: str):
    if family == 'ltd':
        return build_pure_ltd_bundle(tuple(e.signature for e in parsed.internal_edges), len(parsed.ext_names)), 'bundle'
    if family == 'cff':
        return build_pure_cff_bundle(parsed), 'bundle'
    return build_hybrid_bundle_raw(parsed), 'bundle'


def build_structure(dot, family: str) -> dict:
    parsed = GIO.parse_dot_graph(dot)
    validation = VAL.validate_parsed_graph(parsed)
    bundle, backend = _build_bundle(parsed, family)
    data = ST.minimal_structure_from_bundle(bundle, parsed, backend, family, validation)
    data['graph'].update(GIO.graph_info(parsed))
    return data


def pretty_structure(data: dict, dot=None, use_color: bool = True, orientation_id: Optional[str] = None, show_details: bool = False, details_for: Optional[str] = None) -> str:
    parsed = GIO.parse_dot_graph(dot) if dot is not None else None
    return render_pretty(data, parsed=parsed, use_color=use_color, orientation_id=orientation_id, show_details=show_details, details_for=details_for)


def _random_default_inputs(dot, seed: int = 1337):
    parsed = GIO.parse_dot_graph(dot)
    rng = random.Random(seed)
    ext4 = []
    for _ in parsed.ext_names:
        e0 = rng.uniform(-0.7, 0.7)
        px = rng.uniform(-0.3, 0.3)
        py = rng.uniform(-0.3, 0.3)
        pz = rng.uniform(-0.3, 0.3)
        ext4.append((e0, px, py, pz))
    loop3 = []
    for _ in parsed.loop_names:
        loop3.append((rng.uniform(-0.25, 0.25), rng.uniform(-0.25, 0.25), rng.uniform(-0.25, 0.25)))
    masses = {}
    for edge in parsed.internal_edges:
        if edge.mass_key is not None and edge.mass_key not in masses:
            masses[edge.mass_key] = round(rng.uniform(0.4, 1.3), 6)
    return ext4, loop3, masses


def evaluate_structure(data: dict, dot, ext4, loop3, numerator_expr: str, dps: int = 80, mass_map: Optional[Dict[str, Any]] = None):
    mp.mp.dps = dps
    num_fn = ST.numerator_from_expr(numerator_expr)
    return ST.evaluate_minimal_bundle(data, dot, ext4, loop3, num_fn, mass_map=mass_map)


def _extrapolate_last_split_value(seq):
    # Conservative default for the command-line diagnostic: use the smallest
    # mass split as the numerical proxy for the limiting non-raised LTD value.
    # This keeps the reported three-way values tied to the mass-shift route when
    # repeated channels are present.
    return mp.mpf(seq[-1]['value']) if seq else mp.mpf('0')

def compare_three_modes(dot, ext4, loop3, numerator_expr: str, dps: int = 80, epsilons=(1e-1, 5e-2, 2.5e-2, 1.25e-2), mass_map: Optional[Dict[str, Any]] = None, cff_data: Optional[dict] = None, hybrid_data: Optional[dict] = None):
    mp.mp.dps = dps
    parsed_merged = GIO.parse_dot_graph(dot)

    merged_cff = cff_data if cff_data is not None else build_structure(dot, 'cff')
    merged_hybrid = hybrid_data if hybrid_data is not None else build_structure(dot, 'hybrid')
    cff_val = evaluate_structure(merged_cff, dot, ext4, loop3, numerator_expr, dps, mass_map)
    hybrid_val = evaluate_structure(merged_hybrid, dot, ext4, loop3, numerator_expr, dps, mass_map)

    split_dot, _ = GIO.build_split_mass_dot(dot)
    split_ltd = build_structure(split_dot, 'ltd')
    seq = []
    for eps in epsilons:
        split_masses = GIO.build_split_mass_assignments(parsed_merged, mass_map or {}, float(eps))
        val = evaluate_structure(split_ltd, split_dot, ext4, loop3, numerator_expr, dps, split_masses)
        seq.append({
            'epsilon': float(eps),
            'value': str(val),
            'abs_to_cff': str(abs(val - cff_val)),
            'abs_to_hybrid': str(abs(val - hybrid_val)),
        })

    split_proxy = _extrapolate_last_split_value(seq)

    return {
        'cff': str(cff_val),
        'hybrid': str(hybrid_val),
        'abs_cff_minus_hybrid': str(abs(cff_val - hybrid_val)),
        'split_ltd_proxy': str(split_proxy),
        'abs_cff_minus_split_proxy': str(abs(cff_val - split_proxy)),
        'abs_hybrid_minus_split_proxy': str(abs(hybrid_val - split_proxy)),
        'split_ltd': seq,
    }

def run_test(dot, ext4=None, loop3=None, numerator_expr: str = '1', dps: int = 80, epsilons=(1e-1, 5e-2, 2.5e-2, 1.25e-2), mass_map: Optional[Dict[str, Any]] = None, seed: int = 1337, cff_data: Optional[dict] = None, hybrid_data: Optional[dict] = None):
    rnd_ext4, rnd_loop3, rnd_masses = _random_default_inputs(dot, seed)
    ext4 = rnd_ext4 if ext4 is None else ext4
    loop3 = rnd_loop3 if loop3 is None else loop3
    masses = dict(rnd_masses)
    if mass_map:
        masses.update({str(k): float(v) for k, v in mass_map.items()})
    report = compare_three_modes(dot, ext4, loop3, numerator_expr, dps, epsilons, masses, cff_data=cff_data, hybrid_data=hybrid_data)
    report['external'] = [list(x) for x in ext4]
    report['loop3'] = [list(x) for x in loop3]
    report['masses'] = masses
    report['numerator'] = numerator_expr
    return report
