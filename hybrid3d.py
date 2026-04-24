#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from hybrid3d_core import load_dot_graph, validate_graph, build_structure, evaluate_structure, compare_three_modes, pretty_structure, run_test


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


def parse_epsilons_arg(arg):
    return tuple(x.strip() for x in arg.split(',') if x.strip())


def cmd_validate(args):
    dot = load_dot_graph(args.dot)
    res = validate_graph(dot)
    print(json.dumps(res, indent=2))
    if not res['ok']:
        raise SystemExit(1)


def cmd_build(args):
    dot = load_dot_graph(args.dot)
    data = build_structure(dot, args.family)
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
    ext4 = maybe_parse_four_vectors(args.external)
    loop3 = maybe_parse_four_vectors(args.loop3)
    masses = parse_mass_map(args)
    if ext4 is None or loop3 is None or masses is None:
        rep = run_test(dot, ext4=ext4, loop3=loop3, numerator_expr=args.numerator_expr, dps=args.dps, mass_map=masses, seed=args.seed)
        ext4 = [tuple(x) for x in rep['external']]
        loop3 = [tuple(x) for x in rep['loop3']]
        masses = rep['masses']
    val = evaluate_structure(data, dot, ext4, loop3, args.numerator_expr, args.dps, masses)
    print(val)


def cmd_compare(args):
    dot = load_dot_graph(args.dot)
    ext4 = maybe_parse_four_vectors(args.external)
    loop3 = maybe_parse_four_vectors(args.loop3)
    masses = parse_mass_map(args)
    if ext4 is None or loop3 is None or masses is None:
        rep = run_test(dot, ext4=ext4, loop3=loop3, numerator_expr=args.numerator_expr, dps=args.dps, mass_map=masses, seed=args.seed, epsilons=parse_epsilons_arg(args.epsilons) if args.epsilons else ('0.1','0.05','0.025','0.0125'), cff_data=parse_json_arg(args.cff_json) if args.cff_json else None, hybrid_data=parse_json_arg(args.hybrid_json) if args.hybrid_json else None)
        txt = json.dumps(rep, indent=2)
    else:
        eps = parse_epsilons_arg(args.epsilons) if args.epsilons else ('0.1', '0.05', '0.025', '0.0125')
        data = compare_three_modes(dot, ext4, loop3, args.numerator_expr, args.dps, eps, masses, cff_data=parse_json_arg(args.cff_json) if args.cff_json else None, hybrid_data=parse_json_arg(args.hybrid_json) if args.hybrid_json else None)
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
    rep = run_test(dot, ext4=ext4, loop3=loop3, numerator_expr=args.numerator_expr, dps=args.dps, mass_map=masses, seed=args.seed, epsilons=eps, cff_data=parse_json_arg(args.cff_json) if args.cff_json else None, hybrid_data=parse_json_arg(args.hybrid_json) if args.hybrid_json else None)
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

    b = sub.add_parser('build')
    b.add_argument('--family', choices=['ltd', 'cff', 'hybrid'], required=True)
    b.add_argument('--dot', required=True)
    b.add_argument('--json-out')
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
    e.add_argument('--numerator-expr', default='1')
    e.add_argument('--dps', type=int, default=80)
    e.add_argument('--masses')
    e.add_argument('--masses-file')
    e.add_argument('--seed', type=int, default=1337)
    e.set_defaults(func=cmd_evaluate)

    c = sub.add_parser('compare')
    c.add_argument('--dot', required=True)
    c.add_argument('--external')
    c.add_argument('--loop3')
    c.add_argument('--numerator-expr', default='1')
    c.add_argument('--dps', type=int, default=80)
    c.add_argument('--epsilons', default='0.1,0.05,0.025,0.0125')
    c.add_argument('--json-out')
    c.add_argument('--cff-json')
    c.add_argument('--hybrid-json')
    c.add_argument('--masses')
    c.add_argument('--masses-file')
    c.add_argument('--seed', type=int, default=1337)
    c.set_defaults(func=cmd_compare)

    t = sub.add_parser('test')
    t.add_argument('--dot', required=True)
    t.add_argument('--external')
    t.add_argument('--loop3')
    t.add_argument('--numerator-expr', default='1')
    t.add_argument('--dps', type=int, default=80)
    t.add_argument('--epsilons', default='0.1,0.05,0.025,0.0125')
    t.add_argument('--json-out')
    t.add_argument('--cff-json')
    t.add_argument('--hybrid-json')
    t.add_argument('--masses')
    t.add_argument('--masses-file')
    t.add_argument('--seed', type=int, default=1337)
    t.set_defaults(func=cmd_test)

    args = ap.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
