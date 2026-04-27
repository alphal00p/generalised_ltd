from __future__ import annotations
from typing import Any, Dict, Optional, Sequence
import json, random
import pydot
import mpmath as mp
from . import graph_io as GIO
from . import structure as ST
from . import validator as VAL
from .pretty import render_pretty
from .orientation_bundle import (
    build_pure_cff_bundle,
    build_bounded_degree_cff_bundle,
    build_pure_ltd_bundle,
    build_hybrid_bundle_raw,
    energy_divergence_report,
    normalize_energy_degree_bounds,
    assert_energy_uv_convergent,
)


def load_dot_graph(path: str):
    graphs = pydot.graph_from_dot_file(path)
    if not graphs:
        raise RuntimeError(f'No DOT graph found in {path}')
    return graphs[0]


def validate_graph(dot) -> dict:
    parsed = GIO.parse_dot_graph(dot)
    return VAL.validate_parsed_graph(parsed)


def _build_bundle(parsed, family: str, energy_degree_bounds=None):
    if family == 'ltd':
        if energy_degree_bounds is not None:
            assert_energy_uv_convergent(tuple(e.signature for e in parsed.internal_edges), energy_degree_bounds)
        return build_pure_ltd_bundle(tuple(e.signature for e in parsed.internal_edges), len(parsed.ext_names)), 'bundle'
    if family == 'cff':
        if energy_degree_bounds is not None:
            return build_bounded_degree_cff_bundle(parsed, energy_degree_bounds), 'bounded_degree_bundle'
        return build_pure_cff_bundle(parsed), 'bundle'
    if energy_degree_bounds is not None:
        bounds = normalize_energy_degree_bounds(energy_degree_bounds, len(parsed.internal_edges))
        assert_energy_uv_convergent(tuple(e.signature for e in parsed.internal_edges), bounds)
        if not GIO.repeated_groups(parsed):
            return build_pure_ltd_bundle(tuple(e.signature for e in parsed.internal_edges), len(parsed.ext_names)), 'bounded_degree_ltd_collapse'
        return build_hybrid_bundle_raw(parsed, energy_degree_bounds=bounds), 'bounded_degree_hybrid_bundle'
    return build_hybrid_bundle_raw(parsed), 'bundle'


def build_structure(dot, family: str, energy_degree_bounds=None) -> dict:
    parsed = GIO.parse_dot_graph(dot)
    validation = VAL.validate_parsed_graph(parsed)
    bundle, backend = _build_bundle(parsed, family, energy_degree_bounds)
    data = ST.minimal_structure_from_bundle(bundle, parsed, backend, family, validation)
    data['graph'].update(GIO.graph_info(parsed))
    if energy_degree_bounds is not None:
        bounds = normalize_energy_degree_bounds(energy_degree_bounds, len(parsed.internal_edges))
        data['graph']['energy_degree_bounds'] = list(bounds)
        data['graph']['energy_divergence'] = energy_divergence_report(tuple(e.signature for e in parsed.internal_edges), bounds)
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

def compare_cff_ltd(dot, ext4, loop3, numerator_expr: str, dps: int = 80, mass_map: Optional[Dict[str, Any]] = None, energy_degree_bounds=None, cff_data: Optional[dict] = None, ltd_data: Optional[dict] = None):
    mp.mp.dps = dps
    cff = cff_data if cff_data is not None else build_structure(dot, 'cff', energy_degree_bounds=energy_degree_bounds)
    ltd = ltd_data if ltd_data is not None else build_structure(dot, 'ltd')
    cff_val = evaluate_structure(cff, dot, ext4, loop3, numerator_expr, dps, mass_map)
    ltd_val = evaluate_structure(ltd, dot, ext4, loop3, numerator_expr, dps, mass_map)
    return {
        'cff': str(cff_val),
        'ltd': str(ltd_val),
        'abs_cff_minus_ltd': str(abs(cff_val - ltd_val)),
        'energy_degree_bounds': cff.get('graph', {}).get('energy_degree_bounds'),
        'energy_divergence': cff.get('graph', {}).get('energy_divergence'),
        'numerator': numerator_expr,
    }

def compare_three_modes(dot, ext4, loop3, numerator_expr: str, dps: int = 80, epsilons=('0.1', '0.05', '0.025', '0.0125'), mass_map: Optional[Dict[str, Any]] = None, cff_data: Optional[dict] = None, hybrid_data: Optional[dict] = None, cff_energy_degree_bounds=None, energy_degree_bounds=None):
    mp.mp.dps = dps
    parsed_merged = GIO.parse_dot_graph(dot)
    common_bounds = energy_degree_bounds
    cff_bounds = common_bounds if common_bounds is not None else cff_energy_degree_bounds

    merged_cff = cff_data if cff_data is not None else build_structure(dot, 'cff', energy_degree_bounds=cff_bounds)
    merged_hybrid = hybrid_data if hybrid_data is not None else build_structure(dot, 'hybrid', energy_degree_bounds=common_bounds)
    cff_val = evaluate_structure(merged_cff, dot, ext4, loop3, numerator_expr, dps, mass_map)
    hybrid_val = evaluate_structure(merged_hybrid, dot, ext4, loop3, numerator_expr, dps, mass_map)

    split_dot, _ = GIO.build_split_mass_dot(dot)
    split_ltd = build_structure(split_dot, 'ltd', energy_degree_bounds=common_bounds)
    seq = []
    for eps in epsilons:
        split_masses = GIO.build_split_mass_assignments(parsed_merged, mass_map or {}, eps)
        val = evaluate_structure(split_ltd, split_dot, ext4, loop3, numerator_expr, dps, split_masses)
        seq.append({
            'epsilon': float(eps),
            'value': str(val),
            'abs_to_cff': str(abs(val - cff_val)),
            'abs_to_hybrid': str(abs(val - hybrid_val)),
        })

    split_proxy = _extrapolate_last_split_value(seq)
    exact_equalities = {
        'cff_hybrid': bool(cff_val == hybrid_val),
        'cff_split_proxy': bool(cff_val == split_proxy),
        'hybrid_split_proxy': bool(hybrid_val == split_proxy),
    }
    has_repeated = bool(GIO.repeated_groups(parsed_merged))

    return {
        'cff': str(cff_val),
        'hybrid': str(hybrid_val),
        'abs_cff_minus_hybrid': str(abs(cff_val - hybrid_val)),
        'split_ltd_proxy': str(split_proxy),
        'abs_cff_minus_split_proxy': str(abs(cff_val - split_proxy)),
        'abs_hybrid_minus_split_proxy': str(abs(hybrid_val - split_proxy)),
        'exact_equalities': exact_equalities,
        'pairwise_distinct_required': has_repeated,
        'pairwise_distinct': (not any(exact_equalities.values())) if has_repeated else None,
        'energy_degree_bounds': merged_cff.get('graph', {}).get('energy_degree_bounds'),
        'split_ltd': seq,
    }

def run_cff_ltd_test(dot, ext4=None, loop3=None, numerator_expr: str = '1', dps: int = 80, mass_map: Optional[Dict[str, Any]] = None, seed: int = 1337, energy_degree_bounds=None, cff_data: Optional[dict] = None, ltd_data: Optional[dict] = None):
    rnd_ext4, rnd_loop3, rnd_masses = _random_default_inputs(dot, seed)
    ext4 = rnd_ext4 if ext4 is None else ext4
    loop3 = rnd_loop3 if loop3 is None else loop3
    masses = dict(rnd_masses)
    if mass_map:
        masses.update({str(k): v for k, v in mass_map.items()})
    report = compare_cff_ltd(dot, ext4, loop3, numerator_expr, dps, masses, energy_degree_bounds=energy_degree_bounds, cff_data=cff_data, ltd_data=ltd_data)
    report['external'] = [list(x) for x in ext4]
    report['loop3'] = [list(x) for x in loop3]
    report['masses'] = masses
    return report

def run_test(dot, ext4=None, loop3=None, numerator_expr: str = '1', dps: int = 80, epsilons=('0.1', '0.05', '0.025', '0.0125'), mass_map: Optional[Dict[str, Any]] = None, seed: int = 1337, cff_data: Optional[dict] = None, hybrid_data: Optional[dict] = None, cff_energy_degree_bounds=None, energy_degree_bounds=None):
    rnd_ext4, rnd_loop3, rnd_masses = _random_default_inputs(dot, seed)
    ext4 = rnd_ext4 if ext4 is None else ext4
    loop3 = rnd_loop3 if loop3 is None else loop3
    masses = dict(rnd_masses)
    if mass_map:
        masses.update({str(k): v for k, v in mass_map.items()})
    report = compare_three_modes(dot, ext4, loop3, numerator_expr, dps, epsilons, masses, cff_data=cff_data, hybrid_data=hybrid_data, cff_energy_degree_bounds=cff_energy_degree_bounds, energy_degree_bounds=energy_degree_bounds)
    report['external'] = [list(x) for x in ext4]
    report['loop3'] = [list(x) for x in loop3]
    report['masses'] = masses
    report['numerator'] = numerator_expr
    return report
