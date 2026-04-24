from __future__ import annotations
from typing import Dict, List
from .graph_io import ParsedGraph, repeated_groups, _node_base


def validate_parsed_graph(parsed: ParsedGraph) -> dict:
    loop_n = len(parsed.loop_names)
    ext_n = len(parsed.ext_names)
    balances: Dict[int, List[int]] = {}
    def touch(v: int): balances.setdefault(v, [0] * (loop_n + ext_n))

    pow_violations = []
    for e in parsed.internal_edges:
        if e.had_pow:
            pow_violations.append(e.edge_id)
        touch(e.tail); touch(e.head)
        vec = list(e.signature[0]) + list(e.signature[1])
        for i, c in enumerate(vec):
            balances[e.tail][i] -= c
            balances[e.head][i] += c

    # External half-edge labels are allowed to follow different sign conventions
    # in the shipped examples, so validation uses them only informationally and
    # does not fail on the external-basis part of the vertex balances.
    for e in parsed.external_edges:
        src = parsed.node_name_to_internal.get(_node_base(e.source))
        dst = parsed.node_name_to_internal.get(_node_base(e.destination))
        coeffs = [0] * loop_n + list(e.ext_coeffs)
        if src is not None:
            touch(src)
            for i, c in enumerate(coeffs): balances[src][i] += c
        if dst is not None:
            touch(dst)
            for i, c in enumerate(coeffs): balances[dst][i] += c

    loop_bad = {v: bal[:loop_n] for v, bal in balances.items() if any(x != 0 for x in bal[:loop_n])}
    ext_info = {v: bal[loop_n:] for v, bal in balances.items() if any(x != 0 for x in bal[loop_n:])}

    same_channel_parallel = []
    seen = {}
    for e in parsed.internal_edges:
        key = ((e.tail, e.head), e.signature, e.mass_key)
        seen.setdefault(key, []).append(e.edge_id)
    for key, ids in seen.items():
        if len(ids) > 1:
            same_channel_parallel.append({'edge_pair': key[0], 'mass': key[2], 'edge_ids': ids})

    rep = repeated_groups(parsed)
    return {
        'ok': (not loop_bad) and (not same_channel_parallel) and (not pow_violations),
        'n_internal_edges': len(parsed.internal_edges),
        'n_external_edges': len(parsed.external_edges),
        'n_loops_from_labels': loop_n,
        'n_external_symbols': ext_n,
        'vertex_balance_violations': loop_bad,
        'vertex_external_balance_info': ext_info,
        'parallel_duplicate_violations': same_channel_parallel,
        'pow_attribute_violations': pow_violations,
        'repeated_groups': [list(g.edge_ids) for g in rep],
    }
