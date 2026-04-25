import itertools, json, pathlib, math, os, subprocess, sys
from fractions import Fraction
import pytest
import mpmath as mp
import pydot

from src.api import load_dot_graph, validate_graph, build_structure, evaluate_structure, compare_three_modes, run_test, run_cff_ltd_test
from src import graph_io as GIO
from src import graph_signatures as SIG2G
from src.orientation_bundle import (
    ExpressionBundle,
    LinearEnergyExpr,
    OrientationTerm,
    SurfaceCacheBuilder,
    assert_energy_uv_convergent,
    compute_internal_E_values,
    energy_divergence_report,
)
from src.structure import minimal_structure_from_bundle, evaluate_minimal_bundle
from src.structure import numerator_from_expr

ROOT = pathlib.Path(__file__).resolve().parents[1]
BOX_MASSES = {"m1":0.8,"m2":1.1,"m3":0.9,"m4":1.2}
ITER_MASSES = {"mA":0.8,"mB":0.9,"mC":1.1,"mD":0.75,"mE":1.0,"mF":0.6}
MERC_MASSES = {"mA":0.95,"mB":1.05,"mC":0.85,"mD":0.9,"mE":1.0,"mF":1.1}
ALL_MASSES = {**BOX_MASSES, **ITER_MASSES, **MERC_MASSES}
ULTIMATE_MASSES = {
    "mR4": "1.11",
    "mR3": "0.93",
    "mB1": "0.71",
    "mB2": "1.23",
    "mK1": "0.82",
    "mK2": "1.06",
    "mK3": "0.88",
    "mRet": "1.17",
}
FOUR_LOOP_STRESS_MASSES = {
    "mR1": "1.10",
    "mB1": "0.70",
    "mB2": "1.20",
    "mR2": "0.90",
    "mK1": "0.80",
    "mK2": "1.05",
    "mRet": "1.15",
}
ULTIMATE_BASIS_MATRICES = {
    "five_loop_ultimate_basis1.dot": (
        (1, 0, 0, 0, 0),
        (0, 1, 0, 0, 0),
        (0, 0, 1, 0, 0),
        (0, 0, 0, 1, 0),
        (0, 0, 0, 0, 1),
    ),
    "five_loop_ultimate_basis2.dot": (
        (1, 1, 0, 0, 0),
        (0, 1, 0, 0, 0),
        (0, 0, 1, 0, 0),
        (0, 0, 0, 1, 0),
        (0, 0, 0, 0, 1),
    ),
    "five_loop_ultimate_basis3.dot": (
        (1, 0, 0, 0, 0),
        (0, 1, 0, 0, 0),
        (0, 0, 1, 1, 1),
        (0, 0, 0, 1, 0),
        (0, 0, 0, 0, 1),
    ),
    "five_loop_ultimate_basis4.dot": (
        (1, 1, 0, 0, 0),
        (0, 1, 0, 0, 0),
        (0, 0, 1, 1, 1),
        (0, 0, 0, 1, 0),
        (0, 0, 0, 0, 1),
    ),
    "five_loop_ultimate_basis5.dot": (
        (1, 1, 0, 0, 0),
        (0, 1, 0, 0, 0),
        (-1, -1, 1, 1, 1),
        (0, 0, 0, 1, 0),
        (0, 0, 0, 0, 1),
    ),
}
ULTIMATE_EXT4 = (
    ("0.37", "0.11", "-0.07", "0.05"),
    ("-0.21", "0.04", "0.13", "-0.09"),
    ("0.18", "-0.08", "0.02", "0.14"),
    ("-0.09", "0.06", "-0.11", "0.03"),
)
ULTIMATE_CANONICAL_LOOP3 = (
    ("0.07", "-0.04", "0.09"),
    ("-0.05", "0.08", "0.02"),
    ("0.03", "0.06", "-0.07"),
    ("0.11", "-0.02", "0.04"),
    ("-0.06", "-0.03", "0.10"),
)

def dot(name):
    return load_dot_graph(str(ROOT/'examples'/'graphs'/name))

def _graph_from_dot_text(text):
    graphs = pydot.graph_from_dot_data(text)
    assert graphs
    return graphs[0]

def _internal_signatures_and_masses(dot_graph):
    parsed = GIO.parse_dot_graph(dot_graph)
    return (
        [edge.signature for edge in parsed.internal_edges],
        [edge.mass_key or '0' for edge in parsed.internal_edges],
        list(parsed.loop_names),
        list(parsed.ext_names),
    )

def _solve_loop_basis(matrix, canonical_loop3):
    n = len(matrix)
    per_dim = []
    base_matrix = [[Fraction(x) for x in row] for row in matrix]
    for dim in range(3):
        A = [row[:] for row in base_matrix]
        rhs = [Fraction(str(canonical_loop3[i][dim])) for i in range(n)]
        for col in range(n):
            pivot = next(row for row in range(col, n) if A[row][col])
            if pivot != col:
                A[col], A[pivot] = A[pivot], A[col]
                rhs[col], rhs[pivot] = rhs[pivot], rhs[col]
            pv = A[col][col]
            A[col] = [x / pv for x in A[col]]
            rhs[col] /= pv
            for row in range(n):
                if row == col:
                    continue
                factor = A[row][col]
                if factor:
                    A[row] = [a - factor * b for a, b in zip(A[row], A[col])]
                    rhs[row] -= factor * rhs[col]
        per_dim.append(rhs)
    return tuple(tuple(str(per_dim[dim][i]) for dim in range(3)) for i in range(n))

def _all_edge_external_numerator(n_edges, n_external=4):
    return ' + '.join(f'dot(edges[{i}], ext[{i % n_external}])' for i in range(n_edges))

def _edge_energy_monomial(bounds):
    return ' * '.join(f'edges[{i}][0]**{power}' for i, power in enumerate(bounds) if power) or '1'

def _edge_map_key(orient):
    return tuple(json.dumps(expr, sort_keys=True) for expr in orient['edge_q0'])

def _denominator_surface_kinds(data):
    out = set()
    surface_kinds = {int(surface['id']): surface['k'] for surface in data['surfaces']}
    for orient in data['orientations']:
        for variant in orient.get('variants', []):
            for node in variant['tree']['nodes']:
                out.update(surface_kinds[int(surface_id)] for surface_id in node.get('surfaces', []))
    return out

def assert_unique_edge_numerator_maps(data):
    seen = set()
    for orient in data['orientations']:
        key = _edge_map_key(orient)
        assert key not in seen, (orient['id'], orient.get('orient_label'))
        seen.add(key)
        for variant in orient.get('variants', []):
            assert variant['edge_q0'] == orient['edge_q0']

def test_validate_all_examples_except_noisy():
    for path in (ROOT/'examples'/'graphs').glob('*.dot'):
        if path.name == 'noisy_example.dot':
            continue
        assert validate_graph(dot(path.name))['ok'], path.name

def test_noisy_rejected():
    assert validate_graph(dot('noisy_example.dot'))['ok'] is False

def test_box_pow3_showcase_script_is_executable():
    script = ROOT / 'examples' / 'scripts' / 'box_pow3_cli_showcase.sh'
    assert script.exists()
    assert os.access(script, os.X_OK)

def test_five_loop_runtime_script_is_executable():
    script = ROOT / 'examples' / 'scripts' / 'five_loop_symbolica_runtime_compare.sh'
    assert script.exists()
    assert os.access(script, os.X_OK)

def test_one_loop_15_external_runtime_script_is_executable():
    script = ROOT / 'examples' / 'scripts' / 'one_loop_15_external_runtime_compare.sh'
    assert script.exists()
    assert os.access(script, os.X_OK)

def test_five_loop_no_repeats_graph_has_four_externals_and_no_repeats():
    d = dot('five_loop_no_repeats.dot')
    validation = validate_graph(d)
    assert validation['ok']
    assert validation['n_loops_from_labels'] == 5
    assert validation['n_external_symbols'] == 4
    assert validation['repeated_groups'] == []

def test_one_loop_15_external_graph_shape():
    d = dot('one_loop_15_external.dot')
    validation = validate_graph(d)
    assert validation['ok']
    assert validation['n_internal_edges'] == 15
    assert validation['n_loops_from_labels'] == 1
    assert validation['n_external_symbols'] == 15
    assert validation['n_external_edges'] == 15
    assert validation['repeated_groups'] == []

def test_graph_from_signatures_cli_stdout_round_trips_prop_expression():
    expr = 'prop(k1+p1,mA)*prop(k1+p1-q1,mB)*prop(k1-p2+q2,mC)*prop(k1,mD)'
    expected_signatures, expected_loop_names, expected_ext_names, expected_masses = SIG2G.extract_signatures_and_masses_from_symbolica_expression(
        expr,
        loop_prefix='k',
        external_prefixes=('p', 'q'),
    )
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / 'hybrid3d.py'),
            'graph_from_signatures',
            '--signatures',
            expr,
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    generated = _graph_from_dot_text(proc.stdout)
    actual_signatures, actual_masses, actual_loop_names, actual_ext_names = _internal_signatures_and_masses(generated)

    assert actual_signatures == expected_signatures
    assert actual_masses == expected_masses
    assert actual_loop_names == expected_loop_names
    assert actual_ext_names == expected_ext_names
    assert validate_graph(generated)['ok']

def test_graph_from_signatures_cli_writes_vakint_dot_file(tmp_path):
    expr = 'prop(k1+p1,0)*prop(k1+p1-q1,0)*prop(k1-p2+q2,0)*prop(k1-p2,0)*prop(k1,m1)'
    out = tmp_path / 'from_signatures.dot'
    subprocess.run(
        [
            sys.executable,
            str(ROOT / 'hybrid3d.py'),
            'graph_from_signatures',
            '--signatures',
            expr,
            '--format',
            'vakint',
            '--dot-output',
            str(out),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    generated = load_dot_graph(str(out))
    expected_signatures, expected_loop_names, expected_ext_names, expected_masses = SIG2G.extract_signatures_and_masses_from_symbolica_expression(
        expr,
        loop_prefix='k',
        external_prefixes=('p', 'q'),
    )
    actual_signatures, actual_masses, actual_loop_names, actual_ext_names = _internal_signatures_and_masses(generated)

    assert actual_signatures == expected_signatures
    assert actual_masses == expected_masses
    assert actual_loop_names == expected_loop_names
    assert actual_ext_names == expected_ext_names
    assert validate_graph(generated)['ok']

def test_symbolica_compile_and_evaluate_cli_matches_builtin_double(tmp_path):
    pytest.importorskip('symbolica')
    json_path = tmp_path / 'box_cff.json'
    so_path = tmp_path / 'box_cff_eval.so'
    numerator = 'dot(edges[0], ext[0]) + edges[1][0]'
    masses = json.dumps(BOX_MASSES)
    subprocess.run(
        [
            sys.executable,
            str(ROOT / 'hybrid3d.py'),
            'build',
            '--family',
            'cff',
            '--dot',
            str(ROOT / 'examples' / 'graphs' / 'box.dot'),
            '--json-out',
            str(json_path),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    compiled = subprocess.run(
        [
            sys.executable,
            str(ROOT / 'hybrid3d.py'),
            'compile',
            '--orientation-json',
            str(json_path),
            '--dot',
            str(ROOT / 'examples' / 'graphs' / 'box.dot'),
            '--numerator-expr',
            numerator,
            '--output',
            str(so_path),
            '--inline-asm',
            'none',
            '--optimization-level',
            '0',
            '--n-cores',
            '1',
            '--no-native',
            '--display-expression',
            '--display-line-length',
            '80',
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    assert 'Symbolica evaluator input' in compiled.stdout
    assert 'Function map' in compiled.stdout
    assert 'Top-level expression' in compiled.stdout
    assert 'Compiled evaluator metadata' in compiled.stdout
    compiled_json = json.loads(json_path.read_text())
    assert compiled_json['evaluator']['library'] == so_path.name
    assert set(compiled_json['evaluators']) == {'symbolica_compiled', 'symbolica_eager', 'symbolica_eager_symjit'}
    eager_state = tmp_path / compiled_json['evaluators']['symbolica_eager']['state']
    assert eager_state.exists()
    assert compiled_json['evaluators']['symbolica_eager_symjit']['state'] == compiled_json['evaluators']['symbolica_eager']['state']
    assert so_path.exists()

    internal = subprocess.run(
        [
            sys.executable,
            str(ROOT / 'hybrid3d.py'),
            'evaluate',
            '--orientation-json',
            str(json_path),
            '--dot',
            str(ROOT / 'examples' / 'graphs' / 'box.dot'),
            '--numerator-expr',
            numerator,
            '--masses',
            masses,
            '--seed',
            '1337',
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    symbolica = subprocess.run(
        [
            sys.executable,
            str(ROOT / 'hybrid3d.py'),
            'evaluate',
            '--orientation-json',
            str(json_path),
            '--dot',
            str(ROOT / 'examples' / 'graphs' / 'box.dot'),
            '--use-symbolica',
            '--masses',
            masses,
            '--seed',
            '1337',
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    assert abs(float(internal.stdout.strip()) - float(symbolica.stdout.strip())) < 1e-10

    eager = subprocess.run(
        [
            sys.executable,
            str(ROOT / 'hybrid3d.py'),
            'evaluate',
            '--orientation-json',
            str(json_path),
            '--dot',
            str(ROOT / 'examples' / 'graphs' / 'box.dot'),
            '--evaluator-backend',
            'symbolica_eager',
            '--masses',
            masses,
            '--seed',
            '1337',
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    eager_symjit = subprocess.run(
        [
            sys.executable,
            str(ROOT / 'hybrid3d.py'),
            'evaluate',
            '--orientation-json',
            str(json_path),
            '--dot',
            str(ROOT / 'examples' / 'graphs' / 'box.dot'),
            '--evaluator-backend',
            'symbolica_eager_symjit',
            '--masses',
            masses,
            '--seed',
            '1337',
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    assert abs(float(internal.stdout.strip()) - float(eager.stdout.strip())) < 1e-10
    assert abs(float(internal.stdout.strip()) - float(eager_symjit.stdout.strip())) < 1e-10

    profiled = subprocess.run(
        [
            sys.executable,
            str(ROOT / 'hybrid3d.py'),
            'evaluate',
            '--orientation-json',
            str(json_path),
            '--dot',
            str(ROOT / 'examples' / 'graphs' / 'box.dot'),
            '--use-symbolica',
            '--profiling',
            '8',
            '--masses',
            masses,
            '--seed',
            '1337',
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    profile = json.loads(profiled.stdout)
    assert profile['batch_size'] == 8
    assert profile['profile_calls'] == 10
    assert profile['total_evaluations'] == 80
    assert profile['seconds_per_sample'] > 0
    assert profile['seconds_per_call'] > 0
    assert profile['per_sample'].endswith(('us', 'ms', 's'))
    assert profile['per_call'].endswith(('us', 'ms', 's'))

    profile_table = subprocess.run(
        [
            sys.executable,
            str(ROOT / 'hybrid3d.py'),
            'evaluate',
            '--orientation-json',
            str(json_path),
            '--profile-label',
            'cff-a',
            '--profile-json',
            f'cff-b={json_path}',
            '--dot',
            str(ROOT / 'examples' / 'graphs' / 'box.dot'),
            '--use-symbolica',
            '--profiling',
            '2',
            '--masses',
            masses,
            '--seed',
            '1337',
            '--no-color',
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    assert 'cff-a' in profile_table.stdout
    assert 'cff-b' in profile_table.stdout
    assert 'symbolica_compiled' in profile_table.stdout
    assert 'symbolica_eager' in profile_table.stdout
    assert 'symbolica_eager_symjit' in profile_table.stdout
    assert 'edges' in profile_table.stdout
    assert 'ext' in profile_table.stdout
    assert 'orientations' in profile_table.stdout
    assert 'relative' in profile_table.stdout
    assert '100.0%' in profile_table.stdout

    builtin_profiled = subprocess.run(
        [
            sys.executable,
            str(ROOT / 'hybrid3d.py'),
            'evaluate',
            '--orientation-json',
            str(json_path),
            '--dot',
            str(ROOT / 'examples' / 'graphs' / 'box.dot'),
            '--numerator-expr',
            numerator,
            '--profiling',
            '4',
            '--masses',
            masses,
            '--seed',
            '1337',
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    builtin_profile = json.loads(builtin_profiled.stdout)
    assert builtin_profile['batch_size'] == 4
    assert builtin_profile['profile_calls'] == 10
    assert builtin_profile['total_evaluations'] == 40
    assert builtin_profile['seconds_per_sample'] > 0
    assert builtin_profile['seconds_per_call'] > 0
    assert abs(float(internal.stdout.strip()) - float(builtin_profile['value'])) < 1e-10

    complex_json = tmp_path / 'box_cff_complex.json'
    complex_so = tmp_path / 'box_cff_complex_eval.so'
    subprocess.run(
        [
            sys.executable,
            str(ROOT / 'hybrid3d.py'),
            'compile',
            '--orientation-json',
            str(json_path),
            '--dot',
            str(ROOT / 'examples' / 'graphs' / 'box.dot'),
            '--numerator-expr',
            numerator,
            '--output',
            str(complex_so),
            '--json-out',
            str(complex_json),
            '--value-type',
            'complex',
            '--inline-asm',
            'none',
            '--optimization-level',
            '0',
            '--n-cores',
            '1',
            '--no-native',
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    complex_symbolica = subprocess.run(
        [
            sys.executable,
            str(ROOT / 'hybrid3d.py'),
            'evaluate',
            '--orientation-json',
            str(complex_json),
            '--dot',
            str(ROOT / 'examples' / 'graphs' / 'box.dot'),
            '--use-symbolica',
            '--masses',
            masses,
            '--seed',
            '1337',
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    complex_value = complex(complex_symbolica.stdout.strip())
    assert abs(float(internal.stdout.strip()) - complex_value.real) < 1e-10
    assert abs(complex_value.imag) < 1e-14

def test_symbolica_hybrid_compile_matches_builtin_double(tmp_path):
    pytest.importorskip('symbolica')
    json_path = tmp_path / 'box_pow3_hybrid.json'
    so_path = tmp_path / 'box_pow3_hybrid_eval.so'
    numerator = 'dot(edges[0], ext[0]) + edges[3][0]'
    masses = json.dumps(BOX_MASSES)
    subprocess.run(
        [
            sys.executable,
            str(ROOT / 'hybrid3d.py'),
            'build',
            '--family',
            'hybrid',
            '--dot',
            str(ROOT / 'examples' / 'graphs' / 'box_pow3.dot'),
            '--json-out',
            str(json_path),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(ROOT / 'hybrid3d.py'),
            'compile',
            '--orientation-json',
            str(json_path),
            '--dot',
            str(ROOT / 'examples' / 'graphs' / 'box_pow3.dot'),
            '--numerator-expr',
            numerator,
            '--output',
            str(so_path),
            '--inline-asm',
            'none',
            '--optimization-level',
            '0',
            '--n-cores',
            '1',
            '--no-native',
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    internal = subprocess.run(
        [
            sys.executable,
            str(ROOT / 'hybrid3d.py'),
            'evaluate',
            '--orientation-json',
            str(json_path),
            '--dot',
            str(ROOT / 'examples' / 'graphs' / 'box_pow3.dot'),
            '--numerator-expr',
            numerator,
            '--masses',
            masses,
            '--seed',
            '1337',
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    symbolica = subprocess.run(
        [
            sys.executable,
            str(ROOT / 'hybrid3d.py'),
            'evaluate',
            '--orientation-json',
            str(json_path),
            '--dot',
            str(ROOT / 'examples' / 'graphs' / 'box_pow3.dot'),
            '--use-symbolica',
            '--masses',
            masses,
            '--seed',
            '1337',
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    assert abs(float(internal.stdout.strip()) - float(symbolica.stdout.strip())) < 5e-10

def test_symbolica_evaluate_errors_without_compiled_metadata(tmp_path):
    pytest.importorskip('symbolica')
    json_path = tmp_path / 'box_cff.json'
    subprocess.run(
        [
            sys.executable,
            str(ROOT / 'hybrid3d.py'),
            'build',
            '--family',
            'cff',
            '--dot',
            str(ROOT / 'examples' / 'graphs' / 'box.dot'),
            '--json-out',
            str(json_path),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / 'hybrid3d.py'),
            'evaluate',
            '--orientation-json',
            str(json_path),
            '--dot',
            str(ROOT / 'examples' / 'graphs' / 'box.dot'),
            '--use-symbolica',
            '--masses',
            json.dumps(BOX_MASSES),
        ],
        text=True,
        capture_output=True,
    )
    assert proc.returncode != 0
    assert 'no evaluator metadata' in proc.stderr.lower()

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

def test_normal_box_has_no_repeated_propagators():
    data = build_structure(dot('box.dot'), 'cff')
    assert data['graph']['repeated_groups'] == []
    assert len(data['graph']['loop_names']) == 1
    assert data['graph']['n_internal_edges'] == 4

def test_numerator_surface_cache_reuses_ids_and_marks_numerator_only():
    d = dot('box.dot')
    parsed = GIO.parse_dot_graph(d)
    signatures = tuple(edge.signature for edge in parsed.internal_edges)
    sb = SurfaceCacheBuilder()
    unit = sb.intern('e', LinearEnergyExpr(tuple(), tuple(), '1'), 'unit')
    shared = sb.intern('auto', LinearEnergyExpr.E(0), 'shared-den-num')
    assert sb.intern('auto', LinearEnergyExpr.E(0), 'same-shared') == shared
    h_num = sb.intern('auto', LinearEnergyExpr.E(0) - LinearEnergyExpr.E(1), 'num-h')
    two = sb.intern('e', LinearEnergyExpr(tuple(), tuple(), '2'), 'num-two')

    term = OrientationTerm(
        'synthetic',
        'cff',
        0,
        tuple(0 for _ in parsed.internal_edges),
        1,
        tuple(),
        (unit, shared),
        tuple(LinearEnergyExpr.zero() for _ in parsed.loop_names),
        tuple(LinearEnergyExpr.zero() for _ in parsed.internal_edges),
        {'source': 'synthetic_num_surface_test'},
        (shared, h_num, two),
    )
    bundle = ExpressionBundle('synthetic', tuple(), tuple(), signatures, sb.build(), (term,))
    data = minimal_structure_from_bundle(bundle, parsed, 'synthetic', 'cff', {'ok': True})
    surfaces = {surface['id']: surface for surface in data['surfaces']}

    assert surfaces[shared]['numerator_only'] is False
    assert surfaces[h_num]['numerator_only'] is True
    assert surfaces[two]['numerator_only'] is True
    assert data['orientations'][0]['num_surfaces'] == [shared, h_num, two]

    ext4, loop3, masses = __import__('src.api').api._random_default_inputs(d, 1337)
    E_vals = compute_internal_E_values(signatures, GIO.resolve_edge_masses(parsed, masses), loop3, ext4)
    value = evaluate_structure(data, d, ext4, loop3, '1', 80, masses)
    assert abs(value - 2 * (E_vals[0] - E_vals[1])) < mp.mpf('1e-70')

    pretty = __import__('src.api').api.pretty_structure(data, d, use_color=False)
    assert 'class' in pretty
    assert '(h)' in pretty
    assert '(e)' in pretty

def test_hybrid_repeated_box_uses_confluent_kernel():
    data = build_structure(dot('box_pow3.dot'), 'hybrid')
    assert not any(o['orient_label'].endswith('|coupled') for o in data['orientations'])
    assert all(o['meta']['source'] == 'hybrid_confluent_ltd_interpolation' for o in data['orientations'])
    assert all(o['meta']['numerator_sample_kind'] == 'physical_on_shell_affine_orbit' for o in data['orientations'])
    assert any(o['meta']['alpha'] != [0] for o in data['orientations'])
    assert len(data['orientations']) != len(build_structure(dot('box_pow3.dot'), 'cff')['orientations'])

def test_multiloop_hybrid_uses_confluent_kernel():
    data = build_structure(dot('sunrise_pow4.dot'), 'hybrid')
    assert not any(o['orient_label'].endswith('|coupled') for o in data['orientations'])
    assert all(o['meta']['source'] == 'hybrid_confluent_ltd_interpolation' for o in data['orientations'])
    assert any(max(o['meta']['basis_powers']) > 1 for o in data['orientations'])

def test_hybrid_collapses_structurally_to_ltd_without_repeated_masses():
    split_dot, _ = GIO.build_split_mass_dot(dot('box_pow3.dot'))
    ltd = build_structure(split_dot, 'ltd')
    hybrid = build_structure(split_dot, 'hybrid')
    ltd_cmp = json.loads(json.dumps(ltd))
    hybrid_cmp = json.loads(json.dumps(hybrid))
    ltd_cmp['family'] = hybrid_cmp['family'] = 'same'
    assert hybrid_cmp == ltd_cmp

def test_bounded_degree_hybrid_without_repeats_still_collapses_to_ltd():
    d = dot('box.dot')
    ltd = build_structure(d, 'ltd')
    hybrid = build_structure(d, 'hybrid', energy_degree_bounds={0: 2, 1: 2})
    ltd_cmp = json.loads(json.dumps(ltd))
    hybrid_cmp = json.loads(json.dumps(hybrid))
    for item in (ltd_cmp, hybrid_cmp):
        item['family'] = 'same'
        item['backend'] = 'same'
        item['graph'].pop('energy_degree_bounds', None)
        item['graph'].pop('energy_divergence', None)
    assert hybrid_cmp == ltd_cmp

def test_bounded_degree_hybrid_repeated_matches_split_mass_ltd_limit():
    d = dot('box_pow3.dot')
    parsed = GIO.parse_dot_graph(d)
    ext4, loop3, default_masses = __import__('src.api').api._random_default_inputs(d, 1337)
    masses = {**default_masses, **BOX_MASSES}
    hybrid = build_structure(d, 'hybrid', energy_degree_bounds={3: 3})
    assert hybrid['backend'] == 'bounded_degree_hybrid_bundle'
    assert any(
        var['meta'].get('numerator_sample_kind') == 'bounded_degree_finite_difference'
        for orient in hybrid['orientations']
        for var in orient['variants']
    )
    numerator = 'edges[3][0]**3'
    hval = evaluate_structure(hybrid, d, ext4, loop3, numerator, 80, masses)

    split_dot, _ = GIO.build_split_mass_dot(d)
    coarse = GIO.build_split_mass_assignments(parsed, masses, '0.01')
    fine = GIO.build_split_mass_assignments(parsed, masses, '0.001')
    ltd = build_structure(split_dot, 'ltd', energy_degree_bounds={3: 3})
    coarse_diff = abs(hval - evaluate_structure(ltd, split_dot, ext4, loop3, numerator, 80, coarse))
    fine_diff = abs(hval - evaluate_structure(ltd, split_dot, ext4, loop3, numerator, 80, fine))
    assert fine_diff < coarse_diff * mp.mpf('0.02')
    assert fine_diff < mp.mpf('1e-7')

def test_bounded_degree_hybrid_repeated_supports_quadratic_combinations():
    d = dot('proper_iterated_sandwiched_bubble.dot')
    parsed = GIO.parse_dot_graph(d)
    ext4, loop3, default_masses = __import__('src.api').api._random_default_inputs(d, 1337)
    masses = {**default_masses, **ITER_MASSES}
    bounds = {0: 2, 1: 2}
    numerator = 'edges[0][0]**2 * edges[1][0]**2'
    hybrid = build_structure(d, 'hybrid', energy_degree_bounds=bounds)
    hval = evaluate_structure(hybrid, d, ext4, loop3, numerator, 80, masses)

    split_dot, _ = GIO.build_split_mass_dot(d)
    coarse = GIO.build_split_mass_assignments(parsed, masses, '0.01')
    fine = GIO.build_split_mass_assignments(parsed, masses, '0.001')
    ltd = build_structure(split_dot, 'ltd', energy_degree_bounds=bounds)
    coarse_diff = abs(hval - evaluate_structure(ltd, split_dot, ext4, loop3, numerator, 80, coarse))
    fine_diff = abs(hval - evaluate_structure(ltd, split_dot, ext4, loop3, numerator, 80, fine))
    assert fine_diff < coarse_diff * mp.mpf('0.02')
    assert fine_diff < mp.mpf('1e-7')

def test_hybrid_surfaces_have_unit_energy_coefficients():
    for path in (ROOT/'examples'/'graphs').glob('*.dot'):
        if path.name == 'noisy_example.dot':
            continue
        data = build_structure(dot(path.name), 'hybrid')
        for surface in data['surfaces']:
            assert all(abs(int(coeff)) == 1 for _, coeff in surface['e'].get('i', [])), (path.name, surface)
            assert all(abs(int(coeff)) == 1 for _, coeff in surface['e'].get('x', [])), (path.name, surface)

@pytest.mark.parametrize('name', [
    'box_pow3.dot',
    'sunrise_pow4.dot',
    'kite_double_nested_repeats.dot',
    'kite_nested_repeats.dot',
    'kite_sandwich_repeats.dot',
    'proper_iterated_sandwiched_bubble.dot',
    'mercedes_multi_repeats.dot',
])
def test_hybrid_repeated_orientations_have_unique_physical_numerator_maps(name):
    data = build_structure(dot(name), 'hybrid')
    assert_unique_edge_numerator_maps(data)
    for orient in data['orientations']:
        for expr in orient['loop_q0'] + orient['edge_q0']:
            assert expr.get('c', '0') in {'0', '0.0'}, (name, orient['id'], expr)
        for variant in orient.get('variants', []):
            assert variant['loop_q0'] == orient['loop_q0']
            assert variant['edge_q0'] == orient['edge_q0']

def test_every_builder_emits_unique_edge_numerator_maps():
    cases = [
        ('box.dot', 'ltd', None),
        ('box.dot', 'cff', None),
        ('box.dot', 'hybrid', None),
        ('box.dot', 'cff', {0: 2, 1: 2, 2: 2}),
        ('box_pow3.dot', 'cff', None),
        ('box_pow3.dot', 'hybrid', None),
    ]
    for name, family, bounds in cases:
        data = build_structure(dot(name), family, energy_degree_bounds=bounds)
        assert_unique_edge_numerator_maps(data)

def test_json_evaluator_calls_numerator_once_per_orientation():
    d = dot('box.dot')
    data = build_structure(d, 'cff', energy_degree_bounds={0: 2, 1: 2, 2: 2})
    ext4, loop3, masses = __import__('src.api').api._random_default_inputs(d, 1337)
    calls = {}

    def numerator(loop_four, external_four, edge_four=None, source_edge_four=None):
        key = tuple(str(edge[0]) for edge in (edge_four or ()))
        calls[key] = calls.get(key, 0) + 1
        return mp.mpf(1)

    evaluate_minimal_bundle(data, d, ext4, loop3, numerator, mass_map=masses)
    assert len(calls) == len(data['orientations'])
    assert all(count == 1 for count in calls.values())

def test_json_evaluator_consumes_surface_tree_and_substitution_map():
    d = dot('box_pow3.dot')
    data = build_structure(d, 'hybrid')
    ext = [(0.3,0.1,-0.2,0.05),(-0.15,0.2,0.05,-0.1),(0.25,-0.1,0.15,0.07)]
    loop3 = [(0.2,-0.3,0.1)]
    num = 'dot(edges[0], ext[0]) + dot(edges[3], ext[0])'
    v0 = evaluate_structure(data, d, ext, loop3, num, 50, BOX_MASSES)
    mut = json.loads(json.dumps(data))
    orient = mut['orientations'][0]
    tree = orient.get('variants', [orient])[0]['tree']
    sid = tree['nodes'][tree['roots'][0]]['surfaces'][0]
    mut['surfaces'][sid]['e'] = {'i': [], 'x': [], 'c': '1'}
    v1 = evaluate_structure(mut, d, ext, loop3, num, 50, BOX_MASSES)
    assert abs(v0 - v1) > 1e-12
    mut2 = json.loads(json.dumps(data))
    for orient2 in mut2['orientations']:
        orient2['edge_q0'][3] = {'i': [], 'x': [], 'c':'1'}
        for variant in orient2.get('variants', []):
            variant['edge_q0'][3] = {'i': [], 'x': [], 'c':'1'}
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
    assert any(v.get('origin', '').startswith('B') for o in hyb['orientations'] for v in o.get('variants', []))
    assert all(o['orient_label'].replace('+', '').replace('-', '').replace('0', '').replace('x', '') == '' for o in cff['orientations'])

def test_three_way_box_with_edge_numerator_reports_convergence():
    d = dot('box_pow3.dot')
    rep = run_test(d, numerator_expr='dot(edges[0], ext[0]) + dot(edges[3], ext[0])', dps=40, mass_map=BOX_MASSES, seed=1337)
    assert float(rep['abs_cff_minus_hybrid']) < 1e-35
    assert rep['pairwise_distinct'] is True
    assert not any(rep['exact_equalities'].values())
    diffs = [float(x['abs_to_hybrid']) for x in rep['split_ltd']]
    assert diffs[-1] <= diffs[0]
    assert float(rep['abs_hybrid_minus_split_proxy']) < 1e-5

def test_iterated_sandwiched_bubble_with_numerators():
    d = dot('proper_iterated_sandwiched_bubble.dot')
    rep = run_test(d, numerator_expr='dot(edges[1], ext[0]) + dot(edges[4], ext[0])', dps=35, mass_map=ITER_MASSES, seed=1337)
    assert float(rep['abs_cff_minus_hybrid']) < 1e-30
    assert rep['pairwise_distinct'] is True
    assert 'split_ltd' in rep

def test_mercedes_build_and_test_runs():
    d = dot('mercedes_multi_repeats.dot')
    data = build_structure(d, 'hybrid')
    assert any(v.get('origin', '').startswith('B') for o in data['orientations'] for v in o.get('variants', []))
    rep = run_test(d, numerator_expr='edges[0][0] + edges[3][0]', dps=30, mass_map=MERC_MASSES, seed=7)
    assert float(rep['abs_cff_minus_hybrid']) < 1e-25
    assert rep['pairwise_distinct'] is True

def test_four_loop_stress_default_replaces_ultimate_slow_path():
    d = dot('four_loop_stress.dot')
    parsed = GIO.parse_dot_graph(d)
    assert len(parsed.internal_edges) == 10
    assert len(parsed.loop_names) == 4
    assert len(parsed.ext_names) == 4
    assert sorted(len(group.edge_ids) for group in GIO.repeated_groups(parsed)) == [2, 3]

    cff_data = build_structure(d, 'cff')
    hybrid_data = build_structure(d, 'hybrid')
    assert_unique_edge_numerator_maps(cff_data)
    assert_unique_edge_numerator_maps(hybrid_data)
    assert len(cff_data['orientations']) == 210
    assert len(hybrid_data['orientations']) < sum(len(o.get('variants', [])) for o in hybrid_data['orientations'])

    rep = run_test(
        d,
        numerator_expr='dot(edges[0], ext[0]) + dot(edges[5], ext[1])',
        dps=45,
        epsilons=('0.06', '0.03', '0.015'),
        mass_map=FOUR_LOOP_STRESS_MASSES,
        seed=5,
        cff_data=cff_data,
        hybrid_data=hybrid_data,
    )
    assert mp.mpf(rep['abs_cff_minus_hybrid']) < mp.mpf('1e-40')
    assert rep['pairwise_distinct'] is True
    assert mp.mpf(rep['split_ltd'][-1]['abs_to_hybrid']) < mp.mpf('1e-6')

@pytest.mark.skipif(not os.environ.get('HYBRID3D_RUN_SLOW'), reason='set HYBRID3D_RUN_SLOW=1 to run the five-loop ultimate basis test')
def test_ultimate_five_loop_bases_three_way_and_aligned_momenta():
    numerator = _all_edge_external_numerator(13)
    cff_values = []
    for name, matrix in ULTIMATE_BASIS_MATRICES.items():
        d = dot(name)
        parsed = GIO.parse_dot_graph(d)
        assert len(parsed.internal_edges) == 13
        assert len(parsed.loop_names) == 5
        assert len(parsed.ext_names) == 4
        assert sorted(len(group.edge_ids) for group in GIO.repeated_groups(parsed)) == [3, 4]

        loop3 = _solve_loop_basis(matrix, ULTIMATE_CANONICAL_LOOP3)
        cff_data = build_structure(d, 'cff')
        hybrid_data = build_structure(d, 'hybrid')
        assert len(cff_data['orientations']) == 930
        assert len(hybrid_data['orientations']) != len(cff_data['orientations'])
        assert any(o['meta']['source'] == 'hybrid_confluent_ltd_interpolation' for o in hybrid_data['orientations'])

        rep = run_test(
            d,
            ext4=ULTIMATE_EXT4,
            loop3=loop3,
            numerator_expr=numerator,
            dps=70,
            epsilons=('0.04', '0.02', '0.01'),
            mass_map=ULTIMATE_MASSES,
            cff_data=cff_data,
            hybrid_data=hybrid_data,
        )
        assert mp.mpf(rep['abs_cff_minus_hybrid']) < mp.mpf('1e-60')
        assert rep['pairwise_distinct'] is True
        assert not any(rep['exact_equalities'].values())

        split_diffs = [mp.mpf(item['abs_to_hybrid']) for item in rep['split_ltd']]
        assert split_diffs[2] < split_diffs[1] < split_diffs[0]
        assert split_diffs[-1] < mp.mpf('1e-7')
        assert split_diffs[-1] > mp.mpf('1e-12')
        cff_values.append(mp.mpf(rep['cff']))

    reference = cff_values[0]
    assert max(abs(value - reference) for value in cff_values) < mp.mpf('1e-65')

@pytest.mark.parametrize('name,numerator', [
    ('box_pow3.dot', 'dot(edges[0], ext[0]) + dot(edges[-1], ext[0])'),
    ('sunrise_pow4.dot', 'dot(edges[0], ext[0]) + dot(edges[-1], ext[0])'),
    ('kite_double_nested_repeats.dot', 'dot(edges[0], ext[0]) + dot(edges[-1], ext[0])'),
    ('kite_nested_repeats.dot', 'dot(edges[0], ext[0]) + dot(edges[-1], ext[0])'),
    ('kite_sandwich_repeats.dot', 'dot(edges[0], ext[0]) + dot(edges[-1], ext[0])'),
    ('proper_iterated_sandwiched_bubble.dot', 'dot(edges[1], ext[0]) + dot(edges[4], ext[0])'),
    ('mercedes_multi_repeats.dot', 'edges[0][0] + edges[3][0]'),
])
def test_three_way_report_runs_for_every_valid_example(name, numerator):
    rep = run_test(dot(name), numerator_expr=numerator, dps=45, mass_map=ALL_MASSES, seed=1337, epsilons=(0.1, 0.05, 0.025, 0.0125))
    assert float(rep['abs_cff_minus_hybrid']) < 1e-35
    assert rep['pairwise_distinct'] is True
    assert not any(rep['exact_equalities'].values())
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
    ext4, loop3, default_masses = __import__('src.api').api._random_default_inputs(d, 1337)
    masses = {**default_masses, **ALL_MASSES}
    cff_data = build_structure(d, 'cff')
    hybrid_data = build_structure(d, 'hybrid')
    numerators = ['1']
    if parsed.ext_names:
        numerators.extend(f'dot(edges[{i}], ext[0])' for i in range(len(parsed.internal_edges)))
    else:
        numerators.extend(f'edges[{i}][0]' for i in range(len(parsed.internal_edges)))
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
    ext4, loop3, default_masses = __import__('src.api').api._random_default_inputs(d, 1337)
    masses = {**default_masses, **ALL_MASSES}
    split_dot, _ = GIO.build_split_mass_dot(d)
    split_masses = GIO.build_split_mass_assignments(parsed, masses, '0.001')
    cff_data = build_structure(split_dot, 'cff')
    ltd_data = build_structure(split_dot, 'ltd')
    numerators = ['1']
    if parsed.ext_names:
        numerators.extend(f'dot(edges[{i}], ext[0])' for i in range(len(parsed.internal_edges)))
    else:
        numerators.extend(f'edges[{i}][0]' for i in range(len(parsed.internal_edges)))
    for numerator in numerators:
        cff = evaluate_structure(cff_data, split_dot, ext4, loop3, numerator, 80, split_masses)
        ltd = evaluate_structure(ltd_data, split_dot, ext4, loop3, numerator, 80, split_masses)
        assert abs(cff - ltd) < 1e-65, (name, numerator, cff, ltd)


def test_split_mass_pure_cff_ltd_numerator_agreement_improves_with_precision():
    d = dot('box_pow3.dot')
    parsed = GIO.parse_dot_graph(d)
    ext4, loop3, default_masses = __import__('src.api').api._random_default_inputs(d, 1337)
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

def test_bounded_degree_cff_matches_ltd_for_quadratic_edge_energy():
    d = dot('box_pow3.dot')
    parsed = GIO.parse_dot_graph(d)
    split_dot, _ = GIO.build_split_mass_dot(d)
    ext4, loop3, default_masses = __import__('src.api').api._random_default_inputs(split_dot, 1337)
    split_masses = GIO.build_split_mass_assignments(parsed, {**default_masses, **BOX_MASSES}, '0.001')
    cff_data = build_structure(split_dot, 'cff', energy_degree_bounds={0: 2})
    ltd_data = build_structure(split_dot, 'ltd')
    assert cff_data['backend'] == 'bounded_degree_bundle'
    assert cff_data['graph']['energy_divergence']['convergent'] is True
    assert {s['k'] for s in cff_data['surfaces']} <= {'e'}
    assert any(o['meta']['source'] == 'bounded_degree_e_surface_pinch_cff' for o in cff_data['orientations'])

    numerator = 'edges[0][0]**2'
    cff = evaluate_structure(cff_data, split_dot, ext4, loop3, numerator, 80, split_masses)
    ltd = evaluate_structure(ltd_data, split_dot, ext4, loop3, numerator, 80, split_masses)
    assert abs(cff - ltd) < mp.mpf('1e-65'), (cff, ltd)

@pytest.mark.parametrize('name,masses,bounds', [
    ('sunrise_pow4.dot', ALL_MASSES, {0: 2}),
    ('proper_iterated_sandwiched_bubble.dot', ITER_MASSES, {0: 2, 1: 2, 2: 2}),
    ('four_loop_stress.dot', FOUR_LOOP_STRESS_MASSES, {0: 2, 1: 2}),
])
def test_bounded_degree_cff_matches_ltd_for_multiloop_quadratic_combinations(name, masses, bounds):
    d = dot(name)
    parsed = GIO.parse_dot_graph(d)
    split_dot, _ = GIO.build_split_mass_dot(d)
    ext4, loop3, default_masses = __import__('src.api').api._random_default_inputs(split_dot, 1337)
    split_masses = GIO.build_split_mass_assignments(parsed, {**default_masses, **masses}, '0.001')
    cff_data = build_structure(split_dot, 'cff', energy_degree_bounds=bounds)
    ltd_data = build_structure(split_dot, 'ltd')
    assert cff_data['backend'] == 'bounded_degree_bundle'
    assert _denominator_surface_kinds(cff_data) <= {'e'}
    assert any(
        var['meta'].get('quadratic_remainder_contact_decomposition')
        for orient in cff_data['orientations']
        for var in orient['variants']
    )
    split_parsed = GIO.parse_dot_graph(split_dot)
    numerator = _edge_energy_monomial([bounds.get(i, 0) for i in range(len(split_parsed.internal_edges))])
    cff = evaluate_structure(cff_data, split_dot, ext4, loop3, numerator, 80, split_masses)
    ltd = evaluate_structure(ltd_data, split_dot, ext4, loop3, numerator, 80, split_masses)
    assert abs(cff - ltd) < mp.mpf('1e-65'), (name, bounds, cff, ltd)

@pytest.mark.parametrize('name,masses,bounds', [
    ('sunrise_pow4.dot', ALL_MASSES, {0: 2}),
    ('proper_iterated_sandwiched_bubble.dot', ITER_MASSES, {0: 2, 1: 2, 2: 2}),
    ('four_loop_stress.dot', FOUR_LOOP_STRESS_MASSES, {0: 2, 1: 2}),
])
def test_bounded_degree_cff_matches_ltd_for_multiloop_squared_dot_numerators(name, masses, bounds):
    d = dot(name)
    parsed = GIO.parse_dot_graph(d)
    split_dot, _ = GIO.build_split_mass_dot(d)
    ext4, loop3, default_masses = __import__('src.api').api._random_default_inputs(split_dot, 1337)
    split_masses = GIO.build_split_mass_assignments(parsed, {**default_masses, **masses}, '0.001')
    cff_data = build_structure(split_dot, 'cff', energy_degree_bounds=bounds)
    ltd_data = build_structure(split_dot, 'ltd')
    assert _denominator_surface_kinds(cff_data) <= {'e'}

    for edge_id in sorted(bounds):
        numerator = f'dot(edges[{edge_id}], ext[0])**2'
        cff = evaluate_structure(cff_data, split_dot, ext4, loop3, numerator, 80, split_masses)
        ltd = evaluate_structure(ltd_data, split_dot, ext4, loop3, numerator, 80, split_masses)
        assert abs(cff - ltd) < mp.mpf('1e-65'), (name, numerator, cff, ltd)

def test_normal_box_bounded_degree_cff_matches_ltd_for_all_convergent_edge_power_bounds():
    d = dot('box.dot')
    ext4, loop3, masses = __import__('src.api').api._random_default_inputs(d, 1337)
    ltd_data = build_structure(d, 'ltd')
    # Four one-loop propagators give denominator degree 8 in k^0.  The
    # one-dimensional contour is convergent for total numerator degree <= 6.
    bounds_to_test = [
        bounds for bounds in itertools.product(range(3), repeat=4)
        if sum(bounds) <= 6
    ]
    bounds_to_test.extend(
        tuple(3 if i == edge else 0 for i in range(4))
        for edge in range(4)
    )
    assert len(bounds_to_test) == 80

    for bounds in bounds_to_test:
        cff_data = build_structure(d, 'cff', energy_degree_bounds=list(bounds))
        assert cff_data['graph']['energy_divergence']['convergent'] is True
        numerator = _edge_energy_monomial(bounds)
        cff = evaluate_structure(cff_data, d, ext4, loop3, numerator, 80, masses)
        ltd = evaluate_structure(ltd_data, d, ext4, loop3, numerator, 80, masses)
        assert abs(cff - ltd) < mp.mpf('1e-65'), (bounds, numerator, cff, ltd)

def test_normal_box_bounded_degree_cff_structure_morphs_after_affine_degree():
    d = dot('box.dot')
    ordinary = build_structure(d, 'cff')
    linear = build_structure(d, 'cff', energy_degree_bounds=[1, 0, 0, 0])
    bilinear = build_structure(d, 'cff', energy_degree_bounds=[1, 1, 0, 0])
    quadratic = build_structure(d, 'cff', energy_degree_bounds=[2, 0, 0, 0])
    maximal = build_structure(d, 'cff', energy_degree_bounds=[2, 2, 1, 1])
    triple_quadratic = build_structure(d, 'cff', energy_degree_bounds=[2, 2, 2, 0])

    assert len(ordinary['orientations']) == 14
    assert len(linear['orientations']) == len(ordinary['orientations'])
    assert len(bilinear['orientations']) == len(ordinary['orientations'])
    assert len(quadratic['orientations']) > len(ordinary['orientations'])
    assert len(maximal['orientations']) > len(quadratic['orientations'])
    assert {s['k'] for s in quadratic['surfaces']} <= {'e'}
    assert {s['k'] for s in maximal['surfaces']} <= {'e'}
    assert {s['k'] for s in triple_quadratic['surfaces']} <= {'e'}
    assert all(o['meta'].get('source') != 'bounded_degree_ltd_contact_part' for o in quadratic['orientations'])
    assert all(o['meta'].get('source') != 'bounded_degree_ltd_contact_part' for o in triple_quadratic['orientations'])
    assert any(o['meta']['source'] == 'bounded_degree_e_surface_pinch_cff' for o in quadratic['orientations'])
    assert any(o['meta']['source'] == 'bounded_degree_e_surface_pinch_cff' for o in triple_quadratic['orientations'])
    assert maximal['graph']['energy_divergence']['loops'][0]['divergence_degree'] == -2
    assert triple_quadratic['graph']['energy_divergence']['loops'][0]['divergence_degree'] == -2

def test_bounded_degree_cff_repairs_known_quadratic_cff_ltd_mismatch():
    d = dot('box_pow3.dot')
    parsed = GIO.parse_dot_graph(d)
    split_dot, _ = GIO.build_split_mass_dot(d)
    ext4, loop3, default_masses = __import__('src.api').api._random_default_inputs(split_dot, 1337)
    split_masses = GIO.build_split_mass_assignments(parsed, {**default_masses, **BOX_MASSES}, '0.001')
    numerator = 'edges[0][0]**2'
    ordinary = evaluate_structure(build_structure(split_dot, 'cff'), split_dot, ext4, loop3, numerator, 80, split_masses)
    bounded = evaluate_structure(build_structure(split_dot, 'cff', energy_degree_bounds={0: 2}), split_dot, ext4, loop3, numerator, 80, split_masses)
    ltd = evaluate_structure(build_structure(split_dot, 'ltd'), split_dot, ext4, loop3, numerator, 80, split_masses)
    assert abs(ordinary - ltd) > mp.mpf('1e-3')
    assert abs(bounded - ltd) < mp.mpf('1e-65')

def test_bounded_degree_cff_on_repeated_graph_uses_finite_confluent_limit():
    d = dot('box_pow3.dot')
    ext4, loop3, default_masses = __import__('src.api').api._random_default_inputs(d, 1337)
    masses = {**default_masses, **BOX_MASSES}
    bounds = {3: 2}
    numerator = 'edges[3][0]**2'
    cff = build_structure(d, 'cff', energy_degree_bounds=bounds)
    hybrid = build_structure(d, 'hybrid', energy_degree_bounds=bounds)
    assert cff['orientations'][0]['meta']['source'] == 'bounded_degree_repeated_cff_confluent_limit'
    cff_val = evaluate_structure(cff, d, ext4, loop3, numerator, 80, masses)
    hybrid_val = evaluate_structure(hybrid, d, ext4, loop3, numerator, 80, masses)
    assert abs(cff_val - hybrid_val) < mp.mpf('1e-65')

def test_bounded_degree_cff_rejects_nonconvergent_energy_bounds():
    split_dot = dot('box.dot')
    with pytest.raises(ValueError, match='residue at infinity'):
        build_structure(split_dot, 'cff', energy_degree_bounds={0: 7})

def test_energy_uv_check_scans_noncoordinate_loop_directions():
    signatures = (
        ((1, 0), tuple()),
        ((0, 1), tuple()),
        ((1, -1), tuple()),
    )
    report = energy_divergence_report(signatures, [2, 2, 0])
    assert report['coordinate_convergent'] is True
    assert report['directional_convergent'] is False
    with pytest.raises(ValueError, match='direction active='):
        assert_energy_uv_convergent(signatures, [2, 2, 0])

def test_bounded_degree_cff_rejects_unimplemented_higher_contact_recursion():
    split_dot = dot('box.dot')
    with pytest.raises(NotImplementedError, match='known numerator-side polynomial factors'):
        build_structure(split_dot, 'cff', energy_degree_bounds={0: 4})
    with pytest.raises(NotImplementedError, match='known numerator-side polynomial factors'):
        build_structure(split_dot, 'cff', energy_degree_bounds={0: 3, 1: 1})

def test_cff_ltd_test_helper_uses_bounded_degree_cff():
    d = dot('box_pow3.dot')
    parsed = GIO.parse_dot_graph(d)
    split_dot, _ = GIO.build_split_mass_dot(d)
    ext4, loop3, default_masses = __import__('src.api').api._random_default_inputs(split_dot, 1337)
    split_masses = GIO.build_split_mass_assignments(parsed, {**default_masses, **BOX_MASSES}, '0.001')
    rep = run_cff_ltd_test(
        split_dot,
        ext4=ext4,
        loop3=loop3,
        numerator_expr='edges[0][0]**3',
        dps=80,
        mass_map=split_masses,
        energy_degree_bounds={0: 3},
    )
    assert mp.mpf(rep['abs_cff_minus_ltd']) < mp.mpf('1e-65')
    assert rep['energy_divergence']['convergent'] is True
