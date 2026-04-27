#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, pathlib, sys, time
import mpmath as mp
try:
    from prettytable import PrettyTable
except Exception:  # pragma: no cover - optional dependency fallback
    PrettyTable = None
try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init()
except Exception:  # pragma: no cover - optional dependency fallback
    class _NoColor:
        BLACK = RED = GREEN = YELLOW = BLUE = MAGENTA = CYAN = WHITE = RESET_ALL = BRIGHT = NORMAL = DIM = ''
    Fore = Style = _NoColor()
ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src import load_dot_graph, validate_graph, build_structure, evaluate_structure, compare_three_modes, pretty_structure, run_test, run_cff_ltd_test
from src.api import _random_default_inputs
from src import structure as ST
from src import graph_io as GIO
from src import graph_signatures as SIG2G
from src import symbolica_eval as SYMEVAL
from src.orientation_bundle import normalize_energy_degree_bounds


def parse_json_arg(s: str):
    p = pathlib.Path(s)
    if p.exists() and p.is_file():
        return json.loads(p.read_text())
    return json.loads(s)


def parse_mass_map(args):
    if getattr(args, 'masses_file', None):
        return json.loads(pathlib.Path(args.masses_file).read_text())
    if getattr(args, 'masses', None):
        return parse_json_arg(args.masses)
    return None


def maybe_parse_four_vectors(arg):
    if arg is None:
        return None
    return [tuple(x) for x in parse_json_arg(arg)]


def resolve_evaluate_inputs(dot, args):
    ext4 = maybe_parse_four_vectors(args.external)
    loop3 = maybe_parse_four_vectors(args.loop3)
    user_masses = parse_mass_map(args)
    if ext4 is not None and loop3 is not None and user_masses is not None:
        return ext4, loop3, user_masses
    rnd_ext4, rnd_loop3, rnd_masses = _random_default_inputs(dot, args.seed)
    masses = dict(rnd_masses)
    if user_masses:
        masses.update({str(k): v for k, v in user_masses.items()})
    return (
        rnd_ext4 if ext4 is None else ext4,
        rnd_loop3 if loop3 is None else loop3,
        masses,
    )


def parse_epsilons_arg(arg):
    return tuple(x.strip() for x in arg.split(',') if x.strip())

def parse_energy_degree_bounds(arg):
    if arg is None:
        return None
    try:
        parsed = parse_json_arg(arg)
        if isinstance(parsed, (dict, list)):
            return parsed
    except Exception:
        pass
    out = {}
    for item in arg.split(','):
        item = item.strip()
        if not item:
            continue
        if ':' not in item:
            raise SystemExit(f'Invalid energy degree bound {item!r}; use EDGE:DEGREE or JSON')
        key, value = item.split(':', 1)
        key = key.strip()
        out[key if key == '*' else int(key)] = int(value.strip())
    return out


def parse_uniform_scales_arg(arg):
    if arg is None:
        return None
    items = [item.strip() for item in str(arg).split(',') if item.strip()]
    if not items:
        raise SystemExit('--uniform-scales must contain at least one value')
    return tuple(items)


def validate_uniform_scale_arg(value):
    if value is None:
        return None
    text = str(value).strip()
    try:
        parsed = complex(text.replace('J', 'j')) if ('j' in text.lower()) else float(text)
    except Exception as exc:
        raise SystemExit(f'Invalid --uniform-scale value {value!r}') from exc
    if parsed == 0:
        raise SystemExit('--uniform-scale must be nonzero')
    return text


def _uniform_scale_has_imaginary_part(value) -> bool:
    if value is None:
        return False
    text = str(value).strip().replace('J', 'j')
    if 'j' not in text:
        return False
    return complex(text).imag != 0


def _reject_real_symbolica_complex_uniform_scale(metadata, uniform_scale):
    if metadata.get('value_type') == 'real' and _uniform_scale_has_imaginary_part(uniform_scale):
        raise SystemExit('Complex --uniform-scale requires a complex-valued Symbolica evaluator.')


def _json_requires_uniform_scale(data: dict) -> bool:
    if str(data.get('graph', {}).get('uniform_numerator_sampling_scale', 'none')) != 'none':
        return True
    for surface in data.get('surfaces', []):
        if int(surface.get('e', {}).get('m', 0) or 0):
            return True
    for orient in data.get('orientations', []):
        for variant in orient.get('variants') or [orient]:
            if int(variant.get('uniform_scale_power', 0) or 0):
                return True
            for expr in list(variant.get('loop_q0', [])) + list(variant.get('edge_q0', [])):
                if int(expr.get('m', 0) or 0):
                    return True
    return False

def resolve_test_energy_degree_bounds(args):
    common = parse_energy_degree_bounds(getattr(args, 'energy_degree_bounds', None))
    cff_only = parse_energy_degree_bounds(getattr(args, 'cff_energy_degree_bounds', None))
    if common is not None and cff_only is not None:
        raise SystemExit('error: use either --energy-degree-bounds or --cff-energy-degree-bounds, not both')
    return common, cff_only


def auto_numerator_expr_for_bounds(dot, energy_degree_bounds) -> str:
    parsed = GIO.parse_dot_graph(dot)
    bounds = normalize_energy_degree_bounds(energy_degree_bounds, len(parsed.internal_edges))
    if bounds is None:
        raise SystemExit('--numerator-expr auto requires --energy-degree-bounds or JSON graph.energy_degree_bounds')
    if any(bounds) and not parsed.ext_names:
        raise SystemExit('--numerator-expr auto requires at least one external momentum symbol')
    factors = []
    n_ext = len(parsed.ext_names)
    for edge_id, degree in enumerate(bounds):
        for power_index in range(int(degree)):
            ext_id = (edge_id + power_index) % n_ext
            factors.append(f'dot(edges[{edge_id}], ext[{ext_id}])')
    return ' * '.join(factors) if factors else '1'


def resolve_numerator_expr(dot, requested_expr: str | None, energy_degree_bounds=None, data: dict | None = None) -> str:
    if requested_expr != 'auto':
        return requested_expr or '1'
    bounds = energy_degree_bounds
    if bounds is None and data is not None:
        bounds = data.get('graph', {}).get('energy_degree_bounds')
    return auto_numerator_expr_for_bounds(dot, bounds)


def _profile_target_from_arg(item: str) -> tuple[str, pathlib.Path]:
    if '=' in item:
        label, path = item.split('=', 1)
        label = label.strip()
        path = path.strip()
        if not label or not path:
            raise SystemExit(f'Invalid --profile-json target {item!r}; use LABEL=PATH')
        return label, pathlib.Path(path)
    path = pathlib.Path(item)
    return path.stem, path


def _color(text: str, style: str, use_color: bool) -> str:
    return f'{style}{text}{Style.RESET_ALL}' if use_color else text


def _render_profile_table(rows, use_color: bool = True) -> str:
    if not rows:
        return ''
    if PrettyTable is None:
        header = ['label', 'backend', 'family', 'value type', 'edges', 'externals', 'orientations', 'maps', 'per sample', 'relative']
        lines = ['\t'.join(header)]
        lines.extend('\t'.join(str(row[k]) for k in ('label', 'backend', 'family', 'value_type', 'internal_edges', 'external_symbols', 'orientations', 'maps', 'per_sample', 'relative')) for row in rows)
        return '\n'.join(lines)
    table = PrettyTable()
    table.field_names = [
        _color('label', Fore.CYAN + Style.BRIGHT, use_color),
        _color('backend', Fore.CYAN + Style.BRIGHT, use_color),
        _color('family', Fore.CYAN + Style.BRIGHT, use_color),
        _color('value', Fore.CYAN + Style.BRIGHT, use_color),
        _color('edges', Fore.CYAN + Style.BRIGHT, use_color),
        _color('ext', Fore.CYAN + Style.BRIGHT, use_color),
        _color('orientations', Fore.CYAN + Style.BRIGHT, use_color),
        _color('maps', Fore.CYAN + Style.BRIGHT, use_color),
        _color('per sample', Fore.CYAN + Style.BRIGHT, use_color),
        _color('relative', Fore.CYAN + Style.BRIGHT, use_color),
    ]
    for idx, row in enumerate(rows):
        label_style = Fore.GREEN + Style.BRIGHT if idx == 0 else Fore.WHITE
        rel_style = Fore.GREEN if idx == 0 else (Fore.YELLOW if row['relative_ratio'] > 1 else Fore.BLUE)
        table.add_row([
            _color(row['label'], label_style, use_color),
            row['backend'],
            row['family'],
            row['value_type'],
            row['internal_edges'],
            row['external_symbols'],
            row['orientations'],
            row['maps'],
            row['per_sample'],
            _color(row['relative'], rel_style, use_color),
        ])
    return str(table)


def _profile_warnings(rows) -> list[str]:
    labels = sorted({
        str(row.get('label', '?'))
        for row in rows
        if row.get('family') == 'ltd' and row.get('has_repeated_propagators')
    })
    if not labels:
        return []
    label_text = ', '.join(labels)
    return [
        'Reminder: LTD profiling rows for repeated-propagator graphs '
        f'({label_text}) do not include numerator derivatives and are not the '
        'valid repeated-propagator LTD formula. Use the mass-shift LTD test for '
        'physics comparisons; these rows are only runtime/structural references.'
    ]


def _render_stability_table(rows, use_color: bool = True) -> str:
    if not rows:
        return ''
    if PrettyTable is None:
        header = ['label', 'family', 'value type', 'edges', 'externals', 'orientations', 'maps', 'input digits', 'result digits', 'value']
        keys = ('label', 'family', 'value_type', 'internal_edges', 'external_symbols', 'orientations', 'maps', 'input_precision_digits', 'precision_digits', 'value_short')
        return '\n'.join(['\t'.join(header)] + ['\t'.join(str(row[k]) for k in keys) for row in rows])
    table = PrettyTable()
    table.field_names = [
        _color('label', Fore.CYAN + Style.BRIGHT, use_color),
        _color('family', Fore.CYAN + Style.BRIGHT, use_color),
        _color('type', Fore.CYAN + Style.BRIGHT, use_color),
        _color('edges', Fore.CYAN + Style.BRIGHT, use_color),
        _color('ext', Fore.CYAN + Style.BRIGHT, use_color),
        _color('orientations', Fore.CYAN + Style.BRIGHT, use_color),
        _color('maps', Fore.CYAN + Style.BRIGHT, use_color),
        _color('input digits', Fore.CYAN + Style.BRIGHT, use_color),
        _color('result digits', Fore.CYAN + Style.BRIGHT, use_color),
        _color('value', Fore.CYAN + Style.BRIGHT, use_color),
    ]
    for row in rows:
        precision = int(row['precision_digits'])
        prec_style = Fore.GREEN if precision >= int(row['input_precision_digits']) else (Fore.YELLOW if precision >= 8 else Fore.RED)
        table.add_row([
            row['label'],
            row['family'],
            row['value_type'],
            row['internal_edges'],
            row['external_symbols'],
            row['orientations'],
            row['maps'],
            row['input_precision_digits'],
            _color(str(row['precision_digits']), prec_style, use_color),
            row['value_short'],
        ])
    return str(table)


def _stability_symbolica_targets(targets, dot, ext4, loop3, masses, input_precision: int, work_precision: int, use_color: bool, uniform_scale=None):
    rows = []
    for label, path in targets:
        if not path.exists():
            raise SystemExit(f'Stability target JSON does not exist: {path}')
        data = json.loads(path.read_text())
        if 'symbolica_eager' not in SYMEVAL.available_symbolica_evaluator_modes(data):
            raise SystemExit(f'Stability requires a saved Symbolica eager evaluator in {path}. Run compile first.')
        graph = data.get('graph', {})
        try:
            metadata = SYMEVAL.symbolica_evaluator_metadata(data, "symbolica_eager")
            _reject_real_symbolica_complex_uniform_scale(metadata, uniform_scale)
            result = SYMEVAL.evaluate_symbolica_stability(
                data,
                dot,
                path,
                ext4,
                loop3,
                masses,
                input_precision=input_precision,
                work_precision=work_precision,
                uniform_scale=uniform_scale,
            )
        except Exception as exc:
            raise SystemExit(str(exc)) from exc
        rows.append({
            'label': label,
            'family': data.get('family', '?'),
            'value_type': result.get('value_type', '?'),
            'internal_edges': graph.get('n_internal_edges', '?'),
            'external_symbols': len(graph.get('ext_names', [])) if isinstance(graph.get('ext_names'), list) else '?',
            'has_repeated_propagators': bool(graph.get('repeated_groups', [])),
            'orientations': len(data.get('orientations', [])),
            'maps': result.get('map_count', '?'),
            'input_precision_digits': result['input_precision_digits'],
            'work_precision_digits': result['work_precision_digits'],
            'precision_digits': result['precision_digits'],
            'component_precision_digits': result.get('component_precision_digits', {}),
            'value': result['value'],
            'value_short': SYMEVAL.compact_stability_value(result['value']),
        })
    print(_render_stability_table(rows, use_color=use_color))
    for warning in _profile_warnings(rows):
        print(_color(warning, Fore.YELLOW + Style.BRIGHT, use_color))


def _split_backends_arg(value: str | None):
    if not value:
        return None
    return [item.strip() for item in value.split(',') if item.strip()]


def _profile_symbolica_targets(targets, dot, ext4, loop3, masses, batch_size: int, use_color: bool, requested_backends=None, uniform_scale=None):
    if batch_size < 1:
        raise SystemExit('--profiling must be positive')
    rows = []
    base_seconds = None
    for label, path in targets:
        if not path.exists():
            raise SystemExit(f'Profile target JSON does not exist: {path}')
        data = json.loads(path.read_text())
        modes = requested_backends or SYMEVAL.available_symbolica_evaluator_modes(data)
        for backend in modes:
            mode = SYMEVAL.canonical_symbolica_backend(backend)
            if mode not in SYMEVAL.available_symbolica_evaluator_modes(data):
                continue
            metadata = SYMEVAL.symbolica_evaluator_metadata(data, mode)
            _reject_real_symbolica_complex_uniform_scale(metadata, uniform_scale)
            graph = data.get('graph', {})
            repeated_groups = graph.get('repeated_groups', [])
            val, profile = SYMEVAL.evaluate_symbolica(
                data,
                dot,
                path,
                ext4,
                loop3,
                masses,
                batch_size=batch_size,
                profile=True,
                backend=mode,
                uniform_scale=uniform_scale,
            )
            seconds = profile['seconds_per_sample']
            if base_seconds is None:
                base_seconds = seconds
            ratio = seconds / base_seconds if base_seconds else 1.0
            rows.append({
                'label': label,
                'backend': mode,
                'family': data.get('family', '?'),
                'value_type': metadata.get('value_type', '?'),
                'internal_edges': graph.get('n_internal_edges', '?'),
                'external_symbols': len(graph.get('ext_names', [])) if isinstance(graph.get('ext_names'), list) else '?',
                'has_repeated_propagators': bool(repeated_groups),
                'orientations': len(data.get('orientations', [])),
                'maps': metadata.get('map_count', '?'),
                'per_sample': SYMEVAL.format_duration(seconds),
                'relative': f'{100 * ratio:.1f}%',
                'relative_ratio': ratio,
                'value': str(val),
                'seconds_per_sample': seconds,
            })
    print(_render_profile_table(rows, use_color=use_color))
    for warning in _profile_warnings(rows):
        print(_color(warning, Fore.YELLOW + Style.BRIGHT, use_color))


def cmd_validate(args):
    dot = load_dot_graph(args.dot)
    res = validate_graph(dot)
    print(json.dumps(res, indent=2))
    if not res['ok']:
        raise SystemExit(1)


def cmd_graph_from_signatures(args):
    if args.signatures_file:
        expr_text = pathlib.Path(args.signatures_file).read_text()
    else:
        expr_text = args.signatures
    ext_prefixes = tuple(p.strip() for p in args.external_prefixes.split(',') if p.strip())
    signatures, loop_names, ext_names, masses = SIG2G.extract_signatures_and_masses_from_symbolica_expression(
        expr_text,
        loop_prefix=args.loop_prefix,
        external_prefixes=ext_prefixes,
        prop_pattern=args.prop_pattern,
    )
    incoming_prefix = ext_prefixes[0] if ext_prefixes else 'p'
    outgoing_prefix = ext_prefixes[1] if len(ext_prefixes) >= 2 else 'q'
    dot = SIG2G.reconstruct_dot(
        signatures,
        loop_names=list(loop_names),
        ext_names=list(ext_names),
        num_vertices=args.num_vertices,
        require_connected=not args.allow_disconnected,
        max_degree=args.max_degree,
        edge_masses=masses,
        emit_vakint_dot_format=args.format == 'vakint',
        incoming_external_prefix=incoming_prefix,
        outgoing_external_prefix=outgoing_prefix,
        graph_engine=args.graph_engine,
        minimize_external_legs=args.minimize_externals,
    )
    dot_text = dot.to_string()
    if args.dot_output and args.dot_output != '-':
        out_path = pathlib.Path(args.dot_output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(dot_text)
    else:
        print(dot_text, end='' if dot_text.endswith('\n') else '\n')


def cmd_build(args):
    dot = load_dot_graph(args.dot)
    energy_degree_bounds = parse_energy_degree_bounds(getattr(args, 'energy_degree_bounds', None))
    try:
        data = build_structure(
            dot,
            args.family,
            energy_degree_bounds=energy_degree_bounds,
            uniform_numerator_sampling_scale=args.uniform_numerator_sampling_scale,
        )
    except NotImplementedError as exc:
        print(f'error: {exc}', file=sys.stderr)
        raise SystemExit(2)
    if getattr(args, 'numerator_expr', None) is not None:
        data.setdefault('graph', {})['numerator_expr'] = resolve_numerator_expr(
            dot,
            args.numerator_expr,
            energy_degree_bounds=energy_degree_bounds,
            data=data,
        )
    txt = json.dumps(data, indent=2)
    if args.json_out:
        pathlib.Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.json_out).write_text(txt)
    if args.show_json:
        print(txt)
    if args.pretty:
        sel = args.show_details_for_orientation if getattr(args, 'show_details_for_orientation', None) is not None else args.pretty_orientation
        print(pretty_structure(data, dot=dot, use_color=not args.no_color, orientation_id=args.pretty_orientation, show_details=(args.show_details or getattr(args, 'show_details_for_orientation', None) is not None), details_for=getattr(args, 'show_details_for_orientation', None)))
    elif not args.json_out and not args.show_json:
        print(txt)


def cmd_evaluate(args):
    data = parse_json_arg(args.orientation_json)
    dot = load_dot_graph(args.dot)
    numerator_expr = resolve_numerator_expr(dot, args.numerator_expr, data=data)
    ext4, loop3, masses = resolve_evaluate_inputs(dot, args)
    uniform_scale = validate_uniform_scale_arg(getattr(args, 'uniform_scale', None))
    if uniform_scale is None and _json_requires_uniform_scale(data):
        raise SystemExit('This JSON uses uniform numerator sampling. Pass --uniform-scale VALUE.')
    evaluator_backend = args.evaluator_backend
    if args.use_symbolica and evaluator_backend == 'builtin':
        evaluator_backend = 'symbolica_compiled'
    if args.stability is not None:
        if not pathlib.Path(args.orientation_json).exists():
            raise SystemExit('--stability requires --orientation-json to be a JSON file path')
        input_precision = int(args.stability)
        work_precision = int(args.stability_work_precision)
        targets = [(args.profile_label, pathlib.Path(args.orientation_json))]
        targets.extend(_profile_target_from_arg(item) for item in (args.profile_json or []))
        if args.profile_json:
            _stability_symbolica_targets(targets, dot, ext4, loop3, masses, input_precision, work_precision, use_color=not args.no_color, uniform_scale=uniform_scale)
        else:
            json_path = pathlib.Path(args.orientation_json)
            if 'symbolica_eager' not in SYMEVAL.available_symbolica_evaluator_modes(data):
                raise SystemExit('Stability requires a saved Symbolica eager evaluator. Run the compile subcommand first.')
            metadata = SYMEVAL.symbolica_evaluator_metadata(data, "symbolica_eager")
            _reject_real_symbolica_complex_uniform_scale(metadata, uniform_scale)
            result = SYMEVAL.evaluate_symbolica_stability(
                data,
                dot,
                json_path,
                ext4,
                loop3,
                masses,
                input_precision=input_precision,
                work_precision=work_precision,
                uniform_scale=uniform_scale,
            )
            print(json.dumps(result, indent=2))
        return
    if args.profile_json:
        if not args.profiling:
            raise SystemExit('--profile-json requires --profiling')
        if evaluator_backend == 'builtin':
            raise SystemExit('--profile-json profiles Symbolica evaluators; add --use-symbolica or --evaluator-backend symbolica')
        if not pathlib.Path(args.orientation_json).exists():
            raise SystemExit('--profile-json requires --orientation-json to be a JSON file path')
        targets = [(args.profile_label, pathlib.Path(args.orientation_json))]
        targets.extend(_profile_target_from_arg(item) for item in args.profile_json)
        requested = _split_backends_arg(args.profile_evaluator_backends)
        if requested is None and evaluator_backend not in {'symbolica', 'symbolica_compiled'}:
            requested = [evaluator_backend]
        _profile_symbolica_targets(targets, dot, ext4, loop3, masses, int(args.profiling), use_color=not args.no_color, requested_backends=requested, uniform_scale=uniform_scale)
        return
    if evaluator_backend != 'builtin':
        json_path = pathlib.Path(args.orientation_json)
        if not json_path.exists():
            raise SystemExit('Symbolica evaluation requires --orientation-json to be a JSON file path')
        symbolica_backend = 'symbolica_compiled' if evaluator_backend == 'symbolica' else evaluator_backend
        metadata = SYMEVAL.symbolica_evaluator_metadata(data, symbolica_backend)
        _reject_real_symbolica_complex_uniform_scale(metadata, uniform_scale)
        compiled_num = metadata.get('numerator_expr')
        if args.numerator_expr is not None and compiled_num is not None and numerator_expr != compiled_num:
            raise SystemExit(f'Compiled Symbolica numerator is {compiled_num!r}, got --numerator-expr {numerator_expr!r}')
        val, profile = SYMEVAL.evaluate_symbolica(
            data,
            dot,
            json_path,
            ext4,
            loop3,
            masses,
            batch_size=args.profiling if args.profiling is not None else 1,
            profile=args.profiling is not None,
            backend=symbolica_backend,
            uniform_scale=uniform_scale,
        )
        if profile:
            print(json.dumps({
                'value': str(val),
                'batch_size': profile['batch_size'],
                'profile_calls': profile['profile_calls'],
                'total_evaluations': profile['total_evaluations'],
                'total_time': SYMEVAL.format_duration(profile['total_seconds']),
                'per_sample': SYMEVAL.format_duration(profile['seconds_per_sample']),
                'per_call': SYMEVAL.format_duration(profile['seconds_per_call']),
                'total_seconds': profile['total_seconds'],
                'seconds_per_sample': profile['seconds_per_sample'],
                'seconds_per_call': profile['seconds_per_call'],
            }, indent=2))
        else:
            print(val)
        return
    if args.profiling:
        batch_size = int(args.profiling)
        if batch_size < 1:
            raise SystemExit('--profiling must be positive')
        mp.mp.dps = args.dps
        num_fn = ST.numerator_from_expr(numerator_expr)
        start = time.perf_counter()
        val = None
        for _ in range(SYMEVAL.DEFAULT_PROFILE_CALLS):
            for _ in range(batch_size):
                val = ST.evaluate_minimal_bundle(data, dot, ext4, loop3, num_fn, mass_map=masses, uniform_scale=uniform_scale)
        elapsed = time.perf_counter() - start
        total_evaluations = batch_size * SYMEVAL.DEFAULT_PROFILE_CALLS
        print(json.dumps({
            'value': str(val),
            'batch_size': batch_size,
            'profile_calls': SYMEVAL.DEFAULT_PROFILE_CALLS,
            'total_evaluations': total_evaluations,
            'total_time': SYMEVAL.format_duration(elapsed),
            'per_sample': SYMEVAL.format_duration(elapsed / total_evaluations),
            'per_call': SYMEVAL.format_duration(elapsed / SYMEVAL.DEFAULT_PROFILE_CALLS),
            'total_seconds': elapsed,
            'seconds_per_sample': elapsed / total_evaluations,
            'seconds_per_call': elapsed / SYMEVAL.DEFAULT_PROFILE_CALLS,
        }, indent=2))
        return
    val = evaluate_structure(data, dot, ext4, loop3, numerator_expr, args.dps, masses, uniform_scale=uniform_scale)
    print(val)


def cmd_compile(args):
    json_path = pathlib.Path(args.orientation_json)
    if not json_path.exists():
        raise SystemExit('--orientation-json must be a file path for compile')
    data = json.loads(json_path.read_text())
    dot = load_dot_graph(args.dot)
    numerator_expr = resolve_numerator_expr(dot, args.numerator_expr, data=data)
    out_json = pathlib.Path(args.json_out) if args.json_out else json_path
    compiler_flags = None
    if args.compiler_flags:
        compiler_flags = [x for item in args.compiler_flags for x in item.split() if x]
    if args.display_expression:
        print(SYMEVAL.format_symbolica_evaluator_inputs(
            data,
            dot,
            numerator_expr=numerator_expr,
            value_type=args.value_type,
            n_cores=args.n_cores,
            iterations=args.iterations,
            cpe_iterations=args.cpe_iterations,
            max_line_length=args.display_line_length,
        ))
        print()
        print('Compiled evaluator metadata')
        print('---------------------------')
    metadata = SYMEVAL.compile_symbolica_evaluator(
        data,
        dot,
        numerator_expr=numerator_expr,
        json_path=out_json,
        output_path=pathlib.Path(args.output) if args.output else None,
        eager_output_path=pathlib.Path(args.eager_output) if args.eager_output else None,
        value_type=args.value_type,
        function_name=args.function_name,
        inline_asm=args.inline_asm,
        optimization_level=args.optimization_level,
        native=not args.no_native,
        n_cores=args.n_cores,
        iterations=args.iterations,
        cpe_iterations=args.cpe_iterations,
        compiler_path=args.compiler_path,
        compiler_flags=compiler_flags,
        keep_cpp=args.keep_cpp,
    )
    data['evaluators'] = metadata
    data['evaluator'] = metadata['symbolica_compiled']
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(data, indent=2))
    print(json.dumps(metadata, indent=2))


def cmd_compare(args):
    dot = load_dot_graph(args.dot)
    ext4 = maybe_parse_four_vectors(args.external)
    loop3 = maybe_parse_four_vectors(args.loop3)
    masses = parse_mass_map(args)
    energy_bounds, cff_energy_bounds = resolve_test_energy_degree_bounds(args)
    numerator_expr = resolve_numerator_expr(dot, args.numerator_expr, energy_degree_bounds=energy_bounds if energy_bounds is not None else cff_energy_bounds)
    uniform_scales = parse_uniform_scales_arg(getattr(args, 'uniform_scales', None))
    if ext4 is None or loop3 is None or masses is None:
        rep = run_test(dot, ext4=ext4, loop3=loop3, numerator_expr=numerator_expr, dps=args.dps, mass_map=masses, seed=args.seed, epsilons=parse_epsilons_arg(args.epsilons) if args.epsilons else ('0.1','0.05','0.025','0.0125'), cff_data=parse_json_arg(args.cff_json) if args.cff_json else None, hybrid_data=parse_json_arg(args.hybrid_json) if args.hybrid_json else None, cff_energy_degree_bounds=cff_energy_bounds, energy_degree_bounds=energy_bounds, uniform_numerator_sampling_scale=args.uniform_numerator_sampling_scale, uniform_scales=uniform_scales)
        txt = json.dumps(rep, indent=2)
    else:
        eps = parse_epsilons_arg(args.epsilons) if args.epsilons else ('0.1', '0.05', '0.025', '0.0125')
        data = compare_three_modes(dot, ext4, loop3, numerator_expr, args.dps, eps, masses, cff_data=parse_json_arg(args.cff_json) if args.cff_json else None, hybrid_data=parse_json_arg(args.hybrid_json) if args.hybrid_json else None, cff_energy_degree_bounds=cff_energy_bounds, energy_degree_bounds=energy_bounds, uniform_numerator_sampling_scale=args.uniform_numerator_sampling_scale, uniform_scales=uniform_scales)
        txt = json.dumps(data, indent=2)
    if args.json_out:
        pathlib.Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.json_out).write_text(txt)
    print(txt)


def cmd_test(args):
    dot = load_dot_graph(args.dot)
    ext4 = maybe_parse_four_vectors(args.external)
    loop3 = maybe_parse_four_vectors(args.loop3)
    masses = parse_mass_map(args)
    eps = parse_epsilons_arg(args.epsilons) if args.epsilons else ('0.1', '0.05', '0.025', '0.0125')
    energy_bounds, cff_energy_bounds = resolve_test_energy_degree_bounds(args)
    numerator_expr = resolve_numerator_expr(dot, args.numerator_expr, energy_degree_bounds=energy_bounds if energy_bounds is not None else cff_energy_bounds)
    uniform_scales = parse_uniform_scales_arg(getattr(args, 'uniform_scales', None))
    rep = run_test(dot, ext4=ext4, loop3=loop3, numerator_expr=numerator_expr, dps=args.dps, mass_map=masses, seed=args.seed, epsilons=eps, cff_data=parse_json_arg(args.cff_json) if args.cff_json else None, hybrid_data=parse_json_arg(args.hybrid_json) if args.hybrid_json else None, cff_energy_degree_bounds=cff_energy_bounds, energy_degree_bounds=energy_bounds, uniform_numerator_sampling_scale=args.uniform_numerator_sampling_scale, uniform_scales=uniform_scales)
    txt = json.dumps(rep, indent=2)
    if args.json_out:
        pathlib.Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.json_out).write_text(txt)
    print(txt)

def cmd_test_cff_ltd(args):
    dot = load_dot_graph(args.dot)
    ext4 = maybe_parse_four_vectors(args.external)
    loop3 = maybe_parse_four_vectors(args.loop3)
    masses = parse_mass_map(args)
    energy_bounds = parse_energy_degree_bounds(args.energy_degree_bounds)
    numerator_expr = resolve_numerator_expr(dot, args.numerator_expr, energy_degree_bounds=energy_bounds)
    rep = run_cff_ltd_test(
        dot,
        ext4=ext4,
        loop3=loop3,
        numerator_expr=numerator_expr,
        dps=args.dps,
        mass_map=masses,
        seed=args.seed,
        energy_degree_bounds=energy_bounds,
    )
    txt = json.dumps(rep, indent=2)
    if args.json_out:
        pathlib.Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.json_out).write_text(txt)
    print(txt)


def main():
    ap = argparse.ArgumentParser(description='DOT-only hybrid LTD/CFF toolkit')
    sub = ap.add_subparsers(dest='cmd', required=True)
    v = sub.add_parser('validate')
    v.add_argument('--dot', required=True)
    v.set_defaults(func=cmd_validate)

    gfs = sub.add_parser('graph_from_signatures', help='Build a DOT graph from a prop(q,m) signature expression')
    src_group = gfs.add_mutually_exclusive_group(required=True)
    src_group.add_argument('--signatures', help='Expression containing prop(momentum, mass) factors')
    src_group.add_argument('--signatures-file', help='File containing an expression with prop(momentum, mass) factors')
    gfs.add_argument('--dot-output', default='-', help="Output DOT path, or '-' for stdout")
    gfs.add_argument('--loop-prefix', default='k')
    gfs.add_argument('--external-prefixes', default='p,q', help='Comma-separated external momentum prefixes')
    gfs.add_argument('--prop-pattern', default='prop(q_,m_)')
    gfs.add_argument('--num-vertices', type=int)
    gfs.add_argument('--allow-disconnected', action='store_true')
    gfs.add_argument('--max-degree', type=int)
    gfs.add_argument('--format', choices=['hybrid3d', 'vakint'], default='hybrid3d')
    gfs.add_argument('--graph-engine')
    gfs.add_argument('--minimize-externals', action='store_true')
    gfs.set_defaults(func=cmd_graph_from_signatures)

    b = sub.add_parser('build')
    b.add_argument('--family', choices=['ltd', 'cff', 'hybrid'], required=True)
    b.add_argument('--dot', required=True)
    b.add_argument('--json-out')
    b.add_argument('--energy-degree-bounds', help='Per-edge EMR energy degree bounds, e.g. "0:2,3:1" or JSON')
    b.add_argument('--uniform-numerator-sampling-scale', choices=['none', 'beyond-quadratic', 'all'], default='none', help='Use a runtime uniform scale M for repeated-channel numerator interpolation')
    b.add_argument('--numerator-expr', help='Optional numerator metadata; use "auto" with --energy-degree-bounds to record the generated edge-external dot-product numerator')
    b.add_argument('--show-json', action='store_true')
    b.add_argument('--pretty', action='store_true')
    b.add_argument('--pretty-orientation')
    b.add_argument('--show-details-for-orientation')
    b.add_argument('--no-color', action='store_true')
    b.add_argument('--show-details', action='store_true')
    b.set_defaults(func=cmd_build)

    e = sub.add_parser('evaluate')
    e.add_argument('--orientation-json', required=True)
    e.add_argument('--dot', required=True)
    e.add_argument('--external')
    e.add_argument('--loop3')
    e.add_argument('--numerator-expr', help='Numerator expression; use "auto" with bounded-degree JSON to saturate graph.energy_degree_bounds with edge-external dot products')
    e.add_argument('--dps', type=int, default=80)
    e.add_argument('--masses')
    e.add_argument('--masses-file')
    e.add_argument('--seed', type=int, default=1337)
    e.add_argument('--uniform-scale', help='Runtime value of the uniform numerator sampling scale M for JSONs built with --uniform-numerator-sampling-scale')
    e.add_argument('--use-symbolica', action='store_true')
    e.add_argument('--evaluator-backend', choices=['builtin', 'symbolica', 'symbolica_compiled', 'symbolica_eager', 'symbolica_eager_symjit'], default='builtin', help='Evaluator backend. "symbolica" is an alias for compiled evaluation for single runs and all available Symbolica modes in multi-JSON profiling.')
    e.add_argument('--profiling', nargs='?', const=100, type=int, help='Report timing for 10 evaluator calls over batches of this size; defaults to 100 when no size is given')
    e.add_argument('--stability', nargs='?', const=16, type=int, help='Evaluate with Symbolica eager precision-tracking Decimal inputs and report resulting precision; defaults to 16 input digits')
    e.add_argument('--stability-work-precision', type=int, default=80, help='Decimal work precision for --stability constants and operations')
    e.add_argument('--profile-label', default='baseline', help='Label for --orientation-json in a multi-JSON profiling table')
    e.add_argument('--profile-json', action='append', help='Additional JSON to profile as LABEL=PATH; repeat to compare several compiled structures')
    e.add_argument('--profile-evaluator-backends', help='Comma-separated Symbolica backends for --profile-json; default is all modes available in each JSON')
    e.add_argument('--no-color', action='store_true', help='Disable color in multi-JSON profiling tables')
    e.set_defaults(func=cmd_evaluate)

    comp = sub.add_parser('compile')
    comp.add_argument('--orientation-json', required=True)
    comp.add_argument('--dot', required=True)
    comp.add_argument('--numerator-expr', default='1', help='Numerator expression; use "auto" with bounded-degree JSON to saturate graph.energy_degree_bounds with edge-external dot products')
    comp.add_argument('--output')
    comp.add_argument('--eager-output', help='Path for the serialized Symbolica eager evaluator state')
    comp.add_argument('--json-out')
    comp.add_argument('--value-type', choices=['real', 'complex'], default='real')
    comp.add_argument('--function-name')
    comp.add_argument('--inline-asm', default='default', choices=['default', 'x64', 'avx2', 'aarch64', 'none'])
    comp.add_argument('--optimization-level', type=int, default=3, choices=[0, 1, 2, 3])
    comp.add_argument('--no-native', action='store_true')
    comp.add_argument('--n-cores', type=int, default=4)
    comp.add_argument('--iterations', type=int, default=1)
    comp.add_argument('--cpe-iterations', type=int)
    comp.add_argument('--compiler-path')
    comp.add_argument('--compiler-flags', action='append')
    comp.add_argument('--keep-cpp', action='store_true')
    comp.add_argument('--display-expression', action='store_true', help='Print the Symbolica evaluator expression, parameters, constants, and function map before compiling')
    comp.add_argument('--display-line-length', type=int, default=100, help='Preferred line length for --display-expression output')
    comp.set_defaults(func=cmd_compile)

    c = sub.add_parser('compare')
    c.add_argument('--dot', required=True)
    c.add_argument('--external')
    c.add_argument('--loop3')
    c.add_argument('--numerator-expr', default='1', help='Numerator expression; use "auto" with --energy-degree-bounds to saturate the bounds with edge-external dot products')
    c.add_argument('--dps', type=int, default=80)
    c.add_argument('--epsilons', default='0.1,0.05,0.025,0.0125')
    c.add_argument('--json-out')
    c.add_argument('--cff-json')
    c.add_argument('--hybrid-json')
    c.add_argument('--energy-degree-bounds', help='Build CFF, hybrid, and split-mass LTD test structures with these EMR energy-degree bounds')
    c.add_argument('--uniform-numerator-sampling-scale', choices=['none', 'beyond-quadratic', 'all'], default='none', help='Also compare uniform-M repeated-channel interpolation builds')
    c.add_argument('--uniform-scales', help='Comma-separated M values used for uniform-M comparisons; default 1.0,2.75')
    c.add_argument('--cff-energy-degree-bounds', help='Deprecated: build only the CFF side with bounded-degree finite-pole completion')
    c.add_argument('--masses')
    c.add_argument('--masses-file')
    c.add_argument('--seed', type=int, default=1337)
    c.set_defaults(func=cmd_compare)

    t = sub.add_parser('test')
    t.add_argument('--dot', required=True)
    t.add_argument('--external')
    t.add_argument('--loop3')
    t.add_argument('--numerator-expr', default='1', help='Numerator expression; use "auto" with --energy-degree-bounds to saturate the bounds with edge-external dot products')
    t.add_argument('--dps', type=int, default=80)
    t.add_argument('--epsilons', default='0.1,0.05,0.025,0.0125')
    t.add_argument('--json-out')
    t.add_argument('--cff-json')
    t.add_argument('--hybrid-json')
    t.add_argument('--energy-degree-bounds', help='Build CFF, hybrid, and split-mass LTD test structures with these EMR energy-degree bounds')
    t.add_argument('--uniform-numerator-sampling-scale', choices=['none', 'beyond-quadratic', 'all'], default='none', help='Also compare uniform-M repeated-channel interpolation builds')
    t.add_argument('--uniform-scales', help='Comma-separated M values used for uniform-M comparisons; default 1.0,2.75')
    t.add_argument('--cff-energy-degree-bounds', help='Deprecated: build only the CFF side with bounded-degree finite-pole completion')
    t.add_argument('--masses')
    t.add_argument('--masses-file')
    t.add_argument('--seed', type=int, default=1337)
    t.set_defaults(func=cmd_test)

    cl = sub.add_parser('test-cff-ltd')
    cl.add_argument('--dot', required=True)
    cl.add_argument('--external')
    cl.add_argument('--loop3')
    cl.add_argument('--numerator-expr', default='1', help='Numerator expression; use "auto" with --energy-degree-bounds to saturate the bounds with edge-external dot products')
    cl.add_argument('--dps', type=int, default=80)
    cl.add_argument('--json-out')
    cl.add_argument('--energy-degree-bounds', required=True, help='Per-edge EMR energy degree bounds, e.g. "0:2" or JSON')
    cl.add_argument('--masses')
    cl.add_argument('--masses-file')
    cl.add_argument('--seed', type=int, default=1337)
    cl.set_defaults(func=cmd_test_cff_ltd)

    args = ap.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
