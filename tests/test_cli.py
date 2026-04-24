import json, pathlib, math
import pytest

from hybrid3d_core.api import load_dot_graph, validate_graph, build_structure, evaluate_structure, compare_three_modes, run_test
from hybrid3d_core import graph_io as GIO
from hybrid3d_core.structure import numerator_from_expr

ROOT = pathlib.Path(__file__).resolve().parents[1]
BOX_MASSES = {"m1":0.8,"m2":1.1,"m3":0.9,"m4":1.2}
ITER_MASSES = {"mA":0.8,"mB":0.9,"mC":1.1,"mD":0.75,"mE":1.0,"mF":0.6}
MERC_MASSES = {"mA":0.95,"mB":1.05,"mC":0.85,"mD":0.9,"mE":1.0,"mF":1.1}
ALL_MASSES = {**BOX_MASSES, **ITER_MASSES, **MERC_MASSES}

def dot(name):
    return load_dot_graph(str(ROOT/'examples'/name))

def test_validate_all_examples_except_noisy():
    for path in (ROOT/'examples').glob('*.dot'):
        if path.name == 'noisy_example.dot':
            continue
        assert validate_graph(dot(path.name))['ok'], path.name

def test_noisy_rejected():
    assert validate_graph(dot('noisy_example.dot'))['ok'] is False

def test_cff_box_surface_shifts_have_unit_external_coefficients_and_internal_ose_head():
    data = build_structure(dot('box_pow3.dot'), 'cff')
    assert data['family'] == 'cff'
    for surf in data['surfaces']:
        assert all(abs(int(c)) <= 1 for _, c in surf['e'].get('x', []))
    # Internal energies are stored in i-terms and rendered by the pretty layer as OSE.
    assert any(s['e'].get('i') for s in data['surfaces'])

def test_cff_box_contains_nontrivial_contraction_branch():
    data = build_structure(dot('box_pow3.dot'), 'cff')
    assert any(
        len(node['children']) > 1
        for orient in data['orientations']
        for node in orient['tree']['nodes']
    )

def test_hybrid_repeated_box_uses_coupled_cone_kernel():
    data = build_structure(dot('box_pow3.dot'), 'hybrid')
    assert any(o['orient_label'].endswith('|coupled') for o in data['orientations'])
    assert all(o['meta']['source'] == 'hybrid_coupled_cff_cone_kernel' for o in data['orientations'])
    assert all(o['meta']['coupled_reason'] == 'general_repeated_channel_cff_cone' for o in data['orientations'])

def test_multiloop_hybrid_uses_same_coupled_cone_kernel():
    data = build_structure(dot('sunrise_pow4.dot'), 'hybrid')
    assert any(o['orient_label'].endswith('|coupled') for o in data['orientations'])
    assert all(o['meta']['source'] == 'hybrid_coupled_cff_cone_kernel' for o in data['orientations'])

def test_hybrid_surfaces_have_unit_energy_coefficients():
    for path in (ROOT/'examples').glob('*.dot'):
        if path.name == 'noisy_example.dot':
            continue
        data = build_structure(dot(path.name), 'hybrid')
        for surface in data['surfaces']:
            assert all(abs(int(coeff)) == 1 for _, coeff in surface['e'].get('i', [])), (path.name, surface)
            assert all(abs(int(coeff)) == 1 for _, coeff in surface['e'].get('x', [])), (path.name, surface)

def test_json_evaluator_consumes_surface_tree_and_substitution_map():
    d = dot('box_pow3.dot')
    data = build_structure(d, 'hybrid')
    ext = [(0.3,0.1,-0.2,0.05),(-0.15,0.2,0.05,-0.1),(0.25,-0.1,0.15,0.07)]
    loop3 = [(0.2,-0.3,0.1)]
    num = 'dot(edges[0], ext[0]) + dot(edges[3], ext[0])'
    v0 = evaluate_structure(data, d, ext, loop3, num, 50, BOX_MASSES)
    mut = json.loads(json.dumps(data))
    orient = mut['orientations'][0]
    sid = orient['tree']['nodes'][orient['tree']['roots'][0]]['surfaces'][0]
    mut['surfaces'][sid]['e'] = {'i': [], 'x': [], 'c': '1'}
    v1 = evaluate_structure(mut, d, ext, loop3, num, 50, BOX_MASSES)
    assert abs(v0 - v1) > 1e-12
    mut2 = json.loads(json.dumps(data))
    for orient2 in mut2['orientations']:
        orient2['edge_q0'][3] = {'i': [], 'x': [], 'c':'1'}
    v2 = evaluate_structure(mut2, d, ext, loop3, 'dot(edges[3], ext[0])', 50, BOX_MASSES)
    v3 = evaluate_structure(data, d, ext, loop3, 'dot(edges[3], ext[0])', 50, BOX_MASSES)
    assert abs(v2 - v3) > 1e-12

def test_family_structures_differ_for_repeated_box():
    d = dot('box_pow3.dot')
    ltd = build_structure(d, 'ltd')
    cff = build_structure(d, 'cff')
    hyb = build_structure(d, 'hybrid')
    assert len(ltd['orientations']) != len(cff['orientations'])
    assert [o['orient_label'] for o in hyb['orientations']] != [o['orient_label'] for o in cff['orientations']]
    assert any('|' in o['orient_label'] for o in hyb['orientations'])
    assert all('|' not in o['orient_label'] for o in cff['orientations'])

def test_three_way_box_with_edge_numerator_reports_convergence():
    d = dot('box_pow3.dot')
    rep = run_test(d, numerator_expr='dot(edges[0], ext[0]) + dot(edges[3], ext[0])', dps=40, mass_map=BOX_MASSES, seed=1337)
    assert float(rep['abs_cff_minus_hybrid']) < 1e-35
    diffs = [float(x['abs_to_hybrid']) for x in rep['split_ltd']]
    assert diffs[-1] <= diffs[0]
    assert float(rep['abs_hybrid_minus_split_proxy']) < 1e-5

def test_iterated_sandwiched_bubble_with_numerators():
    d = dot('proper_iterated_sandwiched_bubble.dot')
    rep = run_test(d, numerator_expr='dot(edges[1], ext[0]) + dot(edges[4], ext[0])', dps=35, mass_map=ITER_MASSES, seed=1337)
    assert float(rep['abs_cff_minus_hybrid']) < 1e-30
    assert 'split_ltd' in rep

def test_mercedes_build_and_test_runs():
    d = dot('mercedes_multi_repeats.dot')
    data = build_structure(d, 'hybrid')
    assert any('|' in o['orient_label'] for o in data['orientations'])
    rep = run_test(d, numerator_expr='dot(edges[0], edges[3])', dps=30, mass_map=MERC_MASSES, seed=7)
    assert float(rep['abs_cff_minus_hybrid']) < 1e-25

@pytest.mark.parametrize('name,numerator', [
    ('box_pow3.dot', 'dot(edges[0], ext[0]) + dot(edges[-1], ext[0])'),
    ('sunrise_pow4.dot', 'dot(edges[0], ext[0]) + dot(edges[-1], ext[0])'),
    ('kite_double_nested_repeats.dot', 'dot(edges[0], ext[0]) + dot(edges[-1], ext[0])'),
    ('kite_nested_repeats.dot', 'dot(edges[0], ext[0]) + dot(edges[-1], ext[0])'),
    ('kite_sandwich_repeats.dot', 'dot(edges[0], ext[0]) + dot(edges[-1], ext[0])'),
    ('proper_iterated_sandwiched_bubble.dot', 'dot(edges[1], ext[0]) + dot(edges[4], ext[0])'),
    ('mercedes_multi_repeats.dot', 'dot(edges[0], edges[3])'),
])
def test_three_way_report_runs_for_every_valid_example(name, numerator):
    rep = run_test(dot(name), numerator_expr=numerator, dps=45, mass_map=ALL_MASSES, seed=1337, epsilons=(0.1, 0.05, 0.025, 0.0125))
    assert float(rep['abs_cff_minus_hybrid']) < 1e-35
    assert len(rep['split_ltd']) == 4
    assert all(math.isfinite(float(item['value'])) for item in rep['split_ltd'])
    assert 'split_ltd_proxy' in rep

@pytest.mark.parametrize('name', [
    'box_pow3.dot',
    'sunrise_pow4.dot',
    'kite_double_nested_repeats.dot',
    'kite_nested_repeats.dot',
    'kite_sandwich_repeats.dot',
    'proper_iterated_sandwiched_bubble.dot',
    'mercedes_multi_repeats.dot',
])
def test_cff_hybrid_match_all_edge_dot_numerators_for_every_valid_example(name):
    d = dot(name)
    parsed = GIO.parse_dot_graph(d)
    ext4, loop3, default_masses = __import__('hybrid3d_core.api').api._random_default_inputs(d, 1337)
    masses = {**default_masses, **ALL_MASSES}
    cff_data = build_structure(d, 'cff')
    hybrid_data = build_structure(d, 'hybrid')
    numerators = ['1']
    if parsed.ext_names:
        numerators.extend(f'dot(edges[{i}], ext[0])' for i in range(len(parsed.internal_edges)))
    if len(parsed.internal_edges) >= 4:
        numerators.append('dot(edges[0], edges[3])')
    else:
        numerators.append('dot(edges[0], edges[-1])')
    for numerator in numerators:
        cff = evaluate_structure(cff_data, d, ext4, loop3, numerator, 80, masses)
        hybrid = evaluate_structure(hybrid_data, d, ext4, loop3, numerator, 80, masses)
        assert abs(cff - hybrid) < 1e-65, (name, numerator, cff, hybrid)

@pytest.mark.parametrize('name', [
    'box_pow3.dot',
    'sunrise_pow4.dot',
    'kite_double_nested_repeats.dot',
    'kite_nested_repeats.dot',
    'kite_sandwich_repeats.dot',
    'proper_iterated_sandwiched_bubble.dot',
    'mercedes_multi_repeats.dot',
])
def test_split_mass_pure_cff_matches_split_mass_pure_ltd_for_every_valid_example(name):
    d = dot(name)
    parsed = GIO.parse_dot_graph(d)
    ext4, loop3, default_masses = __import__('hybrid3d_core.api').api._random_default_inputs(d, 1337)
    masses = {**default_masses, **ALL_MASSES}
    split_dot, _ = GIO.build_split_mass_dot(d)
    split_masses = GIO.build_split_mass_assignments(parsed, masses, '0.001')
    cff_data = build_structure(split_dot, 'cff')
    ltd_data = build_structure(split_dot, 'ltd')
    numerators = ['1']
    if parsed.ext_names:
        numerators.extend(f'dot(edges[{i}], ext[0])' for i in range(len(parsed.internal_edges)))
    if len(parsed.internal_edges) >= 4:
        numerators.append('dot(edges[0], edges[3])')
    else:
        numerators.append('dot(edges[0], edges[-1])')
    for numerator in numerators:
        cff = evaluate_structure(cff_data, split_dot, ext4, loop3, numerator, 80, split_masses)
        ltd = evaluate_structure(ltd_data, split_dot, ext4, loop3, numerator, 80, split_masses)
        assert abs(cff - ltd) < 1e-65, (name, numerator, cff, ltd)


def test_split_mass_pure_cff_ltd_numerator_agreement_improves_with_precision():
    d = dot('box_pow3.dot')
    parsed = GIO.parse_dot_graph(d)
    ext4, loop3, default_masses = __import__('hybrid3d_core.api').api._random_default_inputs(d, 1337)
    masses = {**default_masses, **ALL_MASSES}
    split_dot, _ = GIO.build_split_mass_dot(d)
    split_masses = GIO.build_split_mass_assignments(parsed, masses, '0.001')
    cff_data = build_structure(split_dot, 'cff')
    ltd_data = build_structure(split_dot, 'ltd')
    numerator = 'dot(edges[0], ext[0]) + dot(edges[-1], ext[0])'
    diffs = []
    for dps in (40, 80):
        cff = evaluate_structure(cff_data, split_dot, ext4, loop3, numerator, dps, split_masses)
        ltd = evaluate_structure(ltd_data, split_dot, ext4, loop3, numerator, dps, split_masses)
        diffs.append(abs(cff - ltd))
    assert diffs[1] < diffs[0] * 1e-25
