from __future__ import annotations
import math
from typing import Any, Dict, List, Sequence, Tuple, Optional
import mpmath as mp
from .orientation_bundle import compute_internal_E_values, compute_external_half_edge_energies, edge_spatial_momentum
from .graph_io import parse_dot_graph, resolve_edge_masses


def dot4(a, b):
    return a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]


def numerator_from_expr(expr: str):
    code = compile(expr, '<numerator-expr>', 'eval')
    safe = {'dot': dot4, 'abs': abs, 'min': min, 'max': max, 'sum': sum, 'math': math}
    def fn(loop_four, external_four, edge_four=None, source_edge_four=None):
        return eval(code, {'__builtins__': {}}, dict(safe, loops=loop_four, ext=external_four, edges=edge_four or (), src_edges=source_edge_four or ()))
    return fn


def _linear_to_min(expr):
    return {'i': [[a, b] for a, b in expr.internal_terms], 'x': [[a, b] for a, b in expr.external_terms], 'c': expr.const}


def _min_eval(d, E_vals, OSE_vals):
    total = mp.mpf(d.get('c', '0'))
    for edge_id, coeff in d.get('i', []):
        total += mp.mpf(coeff) * E_vals[int(edge_id)]
    for ext_id, coeff in d.get('x', []):
        total += mp.mpf(coeff) * OSE_vals[int(ext_id)]
    return total


def _compress_chains(chains: List[Tuple[int, ...]]):
    nodes: List[Dict[str, Any]] = []
    root_map: Dict[int, int] = {}
    def add_chain(chain, idxmap):
        if not chain:
            return None
        head = int(chain[0])
        if head not in idxmap:
            idx = len(nodes)
            idxmap[head] = idx
            nodes.append({'surfaces': [head], 'children': []})
        idx = idxmap[head]
        if len(chain) > 1:
            childmap = {nodes[c]['surfaces'][0]: c for c in nodes[idx]['children']}
            child = add_chain(chain[1:], childmap)
            if child is not None and child not in nodes[idx]['children']:
                nodes[idx]['children'].append(child)
        return idx
    roots: List[int] = []
    for chain in chains:
        idx = add_chain(chain, root_map)
        if idx is not None and idx not in roots:
            roots.append(idx)
    return {'roots': roots, 'nodes': nodes}


def _expr_key(expr: Dict[str, Any]):
    return (tuple((int(a), int(b)) for a, b in expr.get('i', [])), tuple((int(a), int(b)) for a, b in expr.get('x', [])), str(expr.get('c', '0')))


def minimal_structure_from_bundle(bundle, parsed, backend: str, family: str, validation: dict):
    surfaces = [{'id': s.surface_id, 'k': s.kind, 'e': _linear_to_min(s.expr)} for s in bundle.surface_cache]
    by_orient = {}
    for t in bundle.terms:
        loop_q0 = [_linear_to_min(x) for x in t.loop_energy_exprs]
        edge_q0 = [_linear_to_min(x) for x in t.edge_energy_exprs]
        key = (
            str(t.orientation_id),
            tuple(int(x) for x in t.edge_orientations),
            int(t.prefactor_sign),
            tuple(int(x) for x in t.prefactor_half_edges),
            tuple(_expr_key(x) for x in loop_q0),
            tuple(_expr_key(x) for x in edge_q0),
        )
        entry = by_orient.setdefault(key, {
            'id': None,
            'orient_label': str(t.orientation_id),
            'edge_signs': list(t.edge_orientations),
            'pref': int(t.prefactor_sign),
            'half_edges': list(t.prefactor_half_edges),
            'loop_q0': loop_q0,
            'edge_q0': edge_q0,
            'chains': [],
            'meta': t.meta,
        })
        entry['chains'].append(tuple(t.surface_chain))
    orientations = []
    for idx, entry in enumerate(by_orient.values()):
        entry['tree'] = _compress_chains(entry.pop('chains'))
        entry['id'] = idx
        orientations.append(entry)
    return {
        'schema_version': 4,
        'family': family,
        'backend': backend,
        'graph': {
            'loop_names': list(parsed.loop_names),
            'ext_names': list(parsed.ext_names),
            'n_internal_edges': len(parsed.internal_edges),
            'n_external_edges': len(parsed.external_edges),
            'validation': validation,
        },
        'surfaces': surfaces,
        'orientations': orientations,
    }


def compute_edge_four_vectors(signatures, masses, loop_spatial, ext4, edge_q0_exprs, parsed, ose_override=None):
    E_vals = compute_internal_E_values(signatures, masses, loop_spatial, ext4)
    OSE_vals = {i: mp.mpf(ext4[i][0]) for i in range(len(ext4))}
    OSE_vals.update(compute_external_half_edge_energies(parsed, ext4))
    if ose_override:
        OSE_vals.update({int(k): mp.mpf(v) for k, v in ose_override.items()})
    out = []
    for expr, sig in zip(edge_q0_exprs, signatures):
        q0 = _min_eval(expr, E_vals, OSE_vals)
        spatial = edge_spatial_momentum(sig, loop_spatial, ext4)
        out.append((float(q0), float(spatial[0]), float(spatial[1]), float(spatial[2])))
    return tuple(out)


def _sum_tree(tree, node_idx, surfaces, E_vals, OSE_vals):
    node = tree['nodes'][node_idx]
    factor = mp.mpf(1)
    for sid in node['surfaces']:
        val = _min_eval(surfaces[int(sid)]['e'], E_vals, OSE_vals)
        if val == 0:
            val = mp.mpf('1.0e-80')
        factor /= val
    if not node['children']:
        return factor
    subtotal = mp.mpf(0)
    for c in node['children']:
        subtotal += _sum_tree(tree, c, surfaces, E_vals, OSE_vals)
    return factor * subtotal


def evaluate_minimal_bundle(data, dot, ext4, loop3, numerator_fn, mass_map=None, ose_override: Optional[Dict[int, Any]] = None):
    parsed = parse_dot_graph(dot)
    masses = resolve_edge_masses(parsed, mass_map)
    signatures = tuple(e.signature for e in parsed.internal_edges)
    E_vals = compute_internal_E_values(signatures, masses, loop3, ext4)
    OSE_vals = {i: mp.mpf(ext4[i][0]) for i in range(len(ext4))}
    OSE_vals.update(compute_external_half_edge_energies(parsed, ext4))
    if ose_override:
        OSE_vals.update({int(k): mp.mpf(v) for k, v in ose_override.items()})

    total = mp.mpf(0)
    for orient in data['orientations']:
        variants = orient.get('variants')
        if not variants:
            variants = [{
                'pref': orient['pref'],
                'half_edges': orient['half_edges'],
                'loop_q0': orient['loop_q0'],
                'edge_q0': orient['edge_q0'],
                'tree': orient['tree'],
            }]
        for var in variants:
            pref = mp.mpf(var['pref'])
            for e in var['half_edges']:
                pref /= (2 * E_vals[int(e)])
            loop_q0 = [_min_eval(x, E_vals, OSE_vals) for x in var['loop_q0']]
            loop_four = tuple((float(loop_q0[i]), *loop3[i]) for i in range(len(loop3)))
            edge_four = compute_edge_four_vectors(signatures, masses, loop3, ext4, var['edge_q0'], parsed, ose_override=ose_override)
            num = mp.mpf(numerator_fn(loop_four, ext4, edge_four, edge_four))
            treesum = mp.mpf(0)
            for r in var['tree']['roots']:
                treesum += _sum_tree(var['tree'], r, data['surfaces'], E_vals, OSE_vals)
            total += pref * num * treesum
    return total
