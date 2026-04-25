from __future__ import annotations
import math
from fractions import Fraction
from typing import Any, Dict, List, Sequence, Tuple, Optional
import mpmath as mp
from .orientation_bundle import compute_internal_E_values, compute_external_half_edge_energies, edge_spatial_momentum, mpf
from .graph_io import parse_dot_graph, resolve_edge_masses


def dot4(a, b):
    return a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]


def numerator_from_expr(expr: str):
    code = compile(expr, '<numerator-expr>', 'eval')
    safe = {'dot': dot4, 'abs': abs, 'min': min, 'max': max, 'sum': sum, 'len': len, 'math': math}
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

def _json_pref(value):
    try:
        frac = Fraction(str(value))
    except Exception:
        return str(value)
    return frac.numerator if frac.denominator == 1 else str(value)

def _expr_is_zero(expr: Dict[str, Any]) -> bool:
    return not expr.get('i') and not expr.get('x') and str(expr.get('c', '0')) in {'0', '0.0'}

def _edge_map_symbol(edge_id: int, expr: Dict[str, Any]) -> str:
    if _expr_is_zero(expr):
        return '0'
    if not expr.get('x') and str(expr.get('c', '0')) in {'0', '0.0'}:
        terms = [(int(a), int(b)) for a, b in expr.get('i', [])]
        if terms == [(int(edge_id), 1)]:
            return '+'
        if terms == [(int(edge_id), -1)]:
            return '-'
    return 'x'

def _orientation_label_from_map(edge_q0: Sequence[Dict[str, Any]]) -> str:
    return ''.join(_edge_map_symbol(edge_id, expr) for edge_id, expr in enumerate(edge_q0))

def _edge_signs_from_map(edge_q0: Sequence[Dict[str, Any]]) -> List[int]:
    out: List[int] = []
    for edge_id, expr in enumerate(edge_q0):
        sym = _edge_map_symbol(edge_id, expr)
        out.append(1 if sym == '+' else -1 if sym == '-' else 0)
    return out

def _variant_origin_label(label: str, meta: Dict[str, Any]) -> str:
    label_s = str(label)
    if label_s.startswith('cff|'):
        return 'cff'
    if label_s.startswith('cff-rem|'):
        return 'cff-rem'
    if label_s.startswith('qpinch['):
        return label_s.split('|ltd=', 1)[0]
    if label_s.startswith('pinch['):
        return label_s.split('|cff=', 1)[0]
    source = str(meta.get('source', ''))
    if source == 'acyclic_orientation':
        return 'cff'
    if source == 'canonical_ltd_residue':
        return 'ltd'
    if source == 'hybrid_confluent_ltd_interpolation':
        parts = label_s.split('|')
        return '|'.join(parts[1:]) if len(parts) > 1 else 'hybrid'
    return source or label_s


def minimal_structure_from_bundle(bundle, parsed, backend: str, family: str, validation: dict):
    denominator_surface_ids = set()
    numerator_surface_ids = set()
    for term in bundle.terms:
        denominator_surface_ids.update(int(sid) for sid in term.surface_chain)
        numerator_surface_ids.update(int(sid) for sid in term.numerator_surface_chain)
    surfaces = [
        {
            'id': s.surface_id,
            'k': s.kind,
            'e': _linear_to_min(s.expr),
            'numerator_only': int(s.surface_id) in numerator_surface_ids and int(s.surface_id) not in denominator_surface_ids,
        }
        for s in bundle.surface_cache
    ]
    by_orient = {}
    tree_cache: Dict[Tuple[Tuple[int, ...], ...], Tuple[int, Dict[str, Any]]] = {}
    def intern_tree(chains: List[Tuple[int, ...]]):
        key = tuple(sorted(tuple(int(x) for x in chain) for chain in chains))
        cached = tree_cache.get(key)
        if cached is not None:
            return cached
        item = (len(tree_cache), _compress_chains(list(key)))
        tree_cache[key] = item
        return item

    for t in bundle.terms:
        loop_q0 = [_linear_to_min(x) for x in t.loop_energy_exprs]
        edge_q0 = [_linear_to_min(x) for x in t.edge_energy_exprs]
        map_key = tuple(_expr_key(x) for x in edge_q0)
        key = ('numap', map_key)
        orient_label = _orientation_label_from_map(edge_q0)
        entry = by_orient.setdefault(key, {
            'id': None,
            'orient_label': orient_label,
            'edge_signs': _edge_signs_from_map(edge_q0),
            'energy_map_symbols': list(orient_label),
            'loop_q0': loop_q0,
            'edge_q0': edge_q0,
            'terms': {},
            'meta': dict(t.meta),
            'merged_orientation_labels': [],
        })
        if str(t.orientation_id) not in entry['merged_orientation_labels']:
            entry['merged_orientation_labels'].append(str(t.orientation_id))
        term_key = (
            _variant_origin_label(str(t.orientation_id), t.meta),
            tuple(int(x) for x in t.prefactor_half_edges),
            tuple(int(x) for x in t.numerator_surface_chain),
            tuple(int(x) for x in t.surface_chain),
        )
        item = entry['terms'].setdefault(term_key, {
            'pref': Fraction(0),
            'meta': dict(t.meta),
            'orientation_labels': [],
        })
        item['pref'] += Fraction(str(t.prefactor_sign))
        if str(t.orientation_id) not in item['orientation_labels']:
            item['orientation_labels'].append(str(t.orientation_id))
    orientations = []
    for idx, entry in enumerate(by_orient.values()):
        variant_chains: Dict[Tuple[str, Fraction, Tuple[int, ...], Tuple[int, ...]], Dict[str, Any]] = {}
        for (origin, half_edges, num_surfaces, chain), item in entry.pop('terms').items():
            pref = item['pref']
            if not pref:
                continue
            key = (origin, pref, half_edges, num_surfaces)
            slot = variant_chains.setdefault(key, {
                'chains': [],
                'meta': dict(item['meta']),
                'orientation_labels': [],
            })
            slot['chains'].append(chain)
            for lbl in item['orientation_labels']:
                if lbl not in slot['orientation_labels']:
                    slot['orientation_labels'].append(lbl)
        variants = []
        for (origin, pref, half_edges, num_surfaces), item in variant_chains.items():
            chains = item['chains']
            denom_id, tree = intern_tree(chains)
            variants.append({
                'denom_id': denom_id,
                'origin': origin,
                'pref': _json_pref(pref),
                'half_edges': list(half_edges),
                'num_surfaces': list(num_surfaces),
                'loop_q0': entry['loop_q0'],
                'edge_q0': entry['edge_q0'],
                'tree': tree,
                'meta': item['meta'],
                'orientation_labels': item['orientation_labels'],
            })
        if not variants:
            continue
        first = variants[0]
        out = {
            'id': idx,
            'orient_label': entry['orient_label'],
            'edge_signs': entry['edge_signs'],
            'pref': first['pref'],
            'half_edges': first['half_edges'],
            'num_surfaces': first['num_surfaces'],
            'loop_q0': entry['loop_q0'],
            'edge_q0': entry['edge_q0'],
            'tree': first['tree'],
            'meta': entry['meta'],
            'energy_map_symbols': entry['energy_map_symbols'],
            'variants': variants,
        }
        out['meta']['merged_orientation_count'] = len(entry['merged_orientation_labels'])
        out['meta']['variant_count'] = len(variants)
        out['meta']['merged_orientation_labels'] = list(entry['merged_orientation_labels'])
        orientations.append(out)

    emitted_denominator_surface_ids = set()
    emitted_numerator_surface_ids = set()
    for orient in orientations:
        variants = orient.get('variants') or [{
            'num_surfaces': orient.get('num_surfaces', []),
            'tree': orient['tree'],
        }]
        for var in variants:
            emitted_numerator_surface_ids.update(int(sid) for sid in var.get('num_surfaces', []))
            for node in var['tree'].get('nodes', []):
                emitted_denominator_surface_ids.update(int(sid) for sid in node.get('surfaces', []))
    for surface in surfaces:
        sid = int(surface['id'])
        surface['numerator_only'] = sid in emitted_numerator_surface_ids and sid not in emitted_denominator_surface_ids

    return {
        'schema_version': 6,
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
    OSE_vals = {i: mpf(ext4[i][0]) for i in range(len(ext4))}
    OSE_vals.update(compute_external_half_edge_energies(parsed, ext4))
    if ose_override:
        OSE_vals.update({int(k): mpf(v) for k, v in ose_override.items()})
    out = []
    for expr, sig in zip(edge_q0_exprs, signatures):
        q0 = _min_eval(expr, E_vals, OSE_vals)
        spatial = edge_spatial_momentum(sig, loop_spatial, ext4)
        out.append((q0, mpf(spatial[0]), mpf(spatial[1]), mpf(spatial[2])))
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
    OSE_vals = {i: mpf(ext4[i][0]) for i in range(len(ext4))}
    OSE_vals.update(compute_external_half_edge_energies(parsed, ext4))
    if ose_override:
        OSE_vals.update({int(k): mpf(v) for k, v in ose_override.items()})
    ext4_mp = tuple(tuple(mpf(x) for x in p) for p in ext4)

    total = mp.mpf(0)
    denominator_cache: Dict[Tuple[Any, Tuple[int, ...]], mp.mpf] = {}
    numerator_cache: Dict[Tuple[Any, Any], mp.mpf] = {}
    for orient in data['orientations']:
        variants = orient.get('variants')
        if not variants:
            variants = [{
                'pref': orient['pref'],
                'half_edges': orient['half_edges'],
                'num_surfaces': orient.get('num_surfaces', []),
                'loop_q0': orient['loop_q0'],
                'edge_q0': orient['edge_q0'],
                'tree': orient['tree'],
            }]
        for var in variants:
            pref = mp.mpf(var['pref'])
            half_edges = tuple(int(e) for e in var['half_edges'])
            num_surface_ids = tuple(int(sid) for sid in var.get('num_surfaces', []))
            denom_key = (var.get('denom_id', id(var['tree'])), half_edges)
            denom = denominator_cache.get(denom_key)
            if denom is None:
                denom = mp.mpf(1)
                for e in half_edges:
                    denom /= (2 * E_vals[int(e)])
                treesum = mp.mpf(0)
                for r in var['tree']['roots']:
                    treesum += _sum_tree(var['tree'], r, data['surfaces'], E_vals, OSE_vals)
                denom *= treesum
                denominator_cache[denom_key] = denom
            num_surface_factor = mp.mpf(1)
            for sid in num_surface_ids:
                num_surface_factor *= _min_eval(data['surfaces'][int(sid)]['e'], E_vals, OSE_vals)
            num_key = (
                tuple(_expr_key(x) for x in var['loop_q0']),
                tuple(_expr_key(x) for x in var['edge_q0']),
            )
            num = numerator_cache.get(num_key)
            if num is None:
                loop_q0 = [_min_eval(x, E_vals, OSE_vals) for x in var['loop_q0']]
                loop_four = tuple((loop_q0[i], *(mpf(x) for x in loop3[i])) for i in range(len(loop3)))
                edge_four = compute_edge_four_vectors(signatures, masses, loop3, ext4_mp, var['edge_q0'], parsed, ose_override=ose_override)
                num = mp.mpf(numerator_fn(loop_four, ext4_mp, edge_four, edge_four))
                numerator_cache[num_key] = num
            total += pref * num_surface_factor * num * denom
    return total
