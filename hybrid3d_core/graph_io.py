from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional, Any
import re
import pydot
import mpmath as mp
from . import graph_signatures as SIG2G

Signature = Tuple[Tuple[int, ...], Tuple[int, ...]]

@dataclass(frozen=True)
class ParsedGraphInternalEdge:
    edge_id: int
    tail: int
    head: int
    label: str
    mass_key: Optional[str]
    signature: Signature
    had_pow: bool = False

@dataclass(frozen=True)
class ParsedGraphExternalEdge:
    edge_id: int
    source: str
    destination: str
    label: str
    ext_coeffs: Tuple[int, ...]

@dataclass(frozen=True)
class ParsedGraph:
    internal_edges: Tuple[ParsedGraphInternalEdge, ...]
    external_edges: Tuple[ParsedGraphExternalEdge, ...]
    loop_names: Tuple[str, ...]
    ext_names: Tuple[str, ...]
    node_name_to_internal: Dict[str, int]

@dataclass(frozen=True)
class RepeatedGroup:
    key: Tuple[Signature, Optional[str]]
    edge_ids: Tuple[int, ...]
    rel_signs: Tuple[int, ...]

def _strip_quotes(s: Optional[str]) -> str:
    if s is None:
        return ''
    out = str(s).strip()
    if len(out) >= 2 and out[0] == out[-1] and out[0] in {'"', "'"}:
        out = out[1:-1]
    return out

def _node_base(name: str) -> str:
    s = _strip_quotes(name)
    return s.split(':', 1)[0]

def _parse_lmb_rep(text: str) -> str:
    text = _strip_quotes(text)
    if not text:
        return ''
    # Very small parser for gammaLoop-like lmb_rep strings used in noisy examples.
    text = text.replace('*', '')
    text = re.sub(r'K\((\d+),[^)]*\)', lambda m: f"k{int(m.group(1))+1}", text)
    text = re.sub(r'P\((\d+),[^)]*\)', lambda m: f"p{int(m.group(1))+1}", text)
    text = text.replace('+-', '-').replace('-1', '-')
    return text

def _classify_nodes(dot: pydot.Dot) -> tuple[set[str], set[str]]:
    internal: set[str] = set(); external: set[str] = set()
    for node in dot.get_nodes():
        name = _node_base(node.get_name())
        if not name or name in {'graph', 'node', 'edge'}:
            continue
        shape = _strip_quotes(node.get('shape')).lower()
        style = _strip_quotes(node.get('style')).lower()
        if shape == 'box' or style == 'invis' or name.startswith('ext'):
            external.add(name)
        else:
            internal.add(name)
    return internal, external

_MOM_RE = re.compile(r'(?<![A-Za-z0-9_])([kpq][A-Za-z_]*\d*|[kpq]\d+)(?![A-Za-z0-9_])')
def _infer_names_from_labels(labels: List[str]) -> tuple[Tuple[str, ...], Tuple[str, ...]]:
    loops=set(); exts=set()
    for label in labels:
        for tok in re.findall(r'[A-Za-z_][A-Za-z0-9_]*', label):
            if tok.startswith('k'):
                loops.add(tok)
            elif tok.startswith('p') or tok.startswith('q'):
                exts.add(tok)
    def key(s):
        m=re.match(r'([A-Za-z_]+)(\d+)$', s)
        return (m.group(1), int(m.group(2))) if m else (s, -1)
    return tuple(sorted(loops, key=key)), tuple(sorted(exts, key=key))

def parse_dot_graph(dot: pydot.Dot, loop_names=None, ext_names=None) -> ParsedGraph:
    labels=[]
    for edge in dot.get_edges():
        lab = _strip_quotes(edge.get('label')) or _parse_lmb_rep(_strip_quotes(edge.get('lmb_rep')))
        if lab: labels.append(lab)
    inferred_loop_names, inferred_ext_names = _infer_names_from_labels(labels)
    loop_names = tuple(loop_names or inferred_loop_names)
    ext_names = tuple(ext_names or inferred_ext_names)

    internal_nodes, external_nodes = _classify_nodes(dot)
    for edge in dot.get_edges():
        for endpoint in (_node_base(edge.get_source()), _node_base(edge.get_destination())):
            if endpoint and endpoint not in external_nodes and not endpoint.startswith('ext'):
                internal_nodes.add(endpoint)
    node_name_to_internal = {name: i for i, name in enumerate(sorted(internal_nodes))}
    def node_to_internal(name: str) -> Optional[int]:
        return node_name_to_internal.get(_node_base(name))

    internal_edges: List[ParsedGraphInternalEdge] = []
    external_edges: List[ParsedGraphExternalEdge] = []
    next_internal_id = 0
    next_external_id = 10_000_000
    for edge in dot.get_edges():
        source = _strip_quotes(edge.get_source()); destination = _strip_quotes(edge.get_destination())
        label = _strip_quotes(edge.get('label')) or _parse_lmb_rep(_strip_quotes(edge.get('lmb_rep')))
        if not label: continue
        tail = node_to_internal(source); head = node_to_internal(destination)
        mass_key = _strip_quotes(edge.get('mass')) or None
        had_pow = bool(_strip_quotes(edge.get('pow')))
        if tail is not None and head is not None:
            signature = SIG2G._parse_momentum_label(label, loop_names, ext_names)
            internal_edges.append(ParsedGraphInternalEdge(next_internal_id, tail, head, label, mass_key, signature, had_pow))
            next_internal_id += 1
        elif tail is not None or head is not None:
            _, ext_coeffs = SIG2G._parse_momentum_label(label, [], ext_names)
            external_edges.append(ParsedGraphExternalEdge(next_external_id, source, destination, label, ext_coeffs))
            next_external_id += 1
    return ParsedGraph(tuple(internal_edges), tuple(external_edges), loop_names, ext_names, node_name_to_internal)

def _normalize_signature(sig: Signature) -> Tuple[Signature, int]:
    loop, ext = sig
    neg = (tuple(-x for x in loop), tuple(-x for x in ext))
    if (loop, ext) <= neg:
        return (loop, ext), 1
    return neg, -1

def repeated_groups(parsed: ParsedGraph) -> Tuple[RepeatedGroup, ...]:
    groups: Dict[Tuple[Signature, Optional[str]], List[Tuple[int, int]]] = {}
    for edge in parsed.internal_edges:
        nsig, rel = _normalize_signature(edge.signature)
        groups.setdefault((nsig, edge.mass_key), []).append((edge.edge_id, int(rel)))
    out=[]
    for key, members in groups.items():
        if len(members)>1:
            members=sorted(members, key=lambda x: x[0])
            out.append(RepeatedGroup(key=key, edge_ids=tuple(e for e,_ in members), rel_signs=tuple(r for _,r in members)))
    out.sort(key=lambda g: g.edge_ids)
    return tuple(out)

def _as_mpf(x: Any) -> mp.mpf:
    if isinstance(x, mp.mpf):
        return x
    if isinstance(x, float):
        return mp.mpf(repr(x))
    return mp.mpf(str(x))

def resolve_edge_masses(parsed: ParsedGraph, mass_map: Optional[Dict[str, Any]]) -> Tuple[Any, ...]:
    mass_map = mass_map or {}; out=[]; missing=[]
    for edge in parsed.internal_edges:
        if edge.mass_key is None:
            out.append('0')
        elif edge.mass_key in mass_map:
            out.append(mass_map[edge.mass_key])
        else:
            missing.append(edge.mass_key); out.append('0')
    if missing:
        raise KeyError(f'Missing numeric mass assignments for keys: {sorted(set(missing))}')
    return tuple(out)

def build_split_mass_dot(dot: pydot.Dot) -> tuple[pydot.Dot, dict[str, list[str]]]:
    parsed=parse_dot_graph(dot); rep_groups=repeated_groups(parsed)
    edge_to_new_mass={}; mapping={}
    for grp in rep_groups:
        orig_mass=grp.key[1]; base=orig_mass if orig_mass is not None else f'massless_group_{grp.edge_ids[0]}'
        mapping[base]=[]
        for i,eid in enumerate(grp.edge_ids):
            new_key=f'{base}__split{i}'; edge_to_new_mass[eid]=new_key; mapping[base].append(new_key)
    new_dot=pydot.Dot(graph_name=dot.get_name() or 'G', graph_type='digraph')
    for k,v in dot.get_attributes().items(): new_dot.set(k,v)
    for node in dot.get_nodes():
        nn=pydot.Node(node.get_name())
        for k,v in node.get_attributes().items(): nn.set(k,v)
        new_dot.add_node(nn)
    internal_index=0
    for e in dot.get_edges():
        ne=pydot.Edge(e.get_source(), e.get_destination())
        for k,v in e.get_attributes().items():
            if k!='pow': ne.set(k,v)
        tail=parsed.node_name_to_internal.get(_node_base(e.get_source()))
        head=parsed.node_name_to_internal.get(_node_base(e.get_destination()))
        label=_strip_quotes(e.get('label')) or _parse_lmb_rep(_strip_quotes(e.get('lmb_rep')))
        if tail is not None and head is not None and label:
            if internal_index in edge_to_new_mass: ne.set('mass', edge_to_new_mass[internal_index])
            internal_index += 1
        new_dot.add_edge(ne)
    return new_dot, mapping

def build_split_mass_assignments(parsed: ParsedGraph, base_mass_map: Dict[str, Any], epsilon: Any) -> Dict[str, str]:
    masses={str(k): str(_as_mpf(v)) for k,v in (base_mass_map or {}).items()}
    eps=_as_mpf(epsilon)
    for grp in repeated_groups(parsed):
        base=grp.key[1] if grp.key[1] is not None else f'massless_group_{grp.edge_ids[0]}'
        base_val=_as_mpf(masses.get(base, '0')); n=len(grp.edge_ids); center=mp.mpf(n-1)/2
        for i,_ in enumerate(grp.edge_ids):
            masses[f'{base}__split{i}']=str(base_val+(mp.mpf(i)-center)*eps)
    return masses

def graph_info(parsed: ParsedGraph) -> dict:
    return {'loop_names': list(parsed.loop_names), 'ext_names': list(parsed.ext_names), 'n_internal_edges': len(parsed.internal_edges), 'n_external_edges': len(parsed.external_edges), 'repeated_groups': [list(g.edge_ids) for g in repeated_groups(parsed)]}
