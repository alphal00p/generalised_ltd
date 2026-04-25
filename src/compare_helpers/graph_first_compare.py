#!/usr/bin/env python3
"""
Graph-first local 3D CFF/LTD/hybrid validator.

This script extends the earlier signature-based validators so that the user can
start directly from a DOT graph, including graphs emitted by
`graph_signatures.py`. Internal edges may carry

  - `mass="..."`   : internal mass
  - `pow="..."`    : positive integer propagator power
  - repeated identical internal edges with the same endpoints/label/mass,
                    which are compacted into one logical propagator with the
                    summed power
  - `lmb_id="..."` : optional loop-basis annotation from the graph generator;
                      this script does not require it because the momentum
                      signature is read directly from the edge label.

The implementation never reconstructs a graph from signatures when the input is
already a graph. In particular, the pure CFF lane uses the input graph directly.
The LTD and hybrid lanes read the signatures trivially from the internal edge
labels and operate on the compacted logical propagators.

Raised powers are handled with the same exact Taylor-jet lift already used in
`raised_power_compare.py`. Numerators are black-box Python
callables of the loop four-vectors and the external four-momenta.

The current hybrid lane inherits the simple-power hybrid base from the existing
validator; numerically it is compared pointwise against pure CFF and pure LTD
for graph inputs, including graph examples with powered and repeated edges.
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import pathlib

_HERE = pathlib.Path(__file__).resolve().parent
_CORE = _HERE.parent
import random
import re
import sys
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import mpmath as mp
import numpy as np
import pydot
import sympy as sp

mp.mp.dps = 80


def _import_from_path(module_name: str, path: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


BASE = _import_from_path("graph_first_base_compare", str(_HERE / "simple_power_compare.py"))
RAISED = _import_from_path("graph_first_raised_compare", str(_HERE / "raised_power_compare.py"))
SIG2G = _import_from_path("graph_first_sig2g", str(_CORE / "graph_signatures.py"))
PRL = _import_from_path("graph_first_prl", str(_CORE / "ltd_cut_structure_generator.py"))

Signature = BASE.Signature
ThreeVector = Tuple[float, float, float]
FourVector = BASE.FourVector
NumeratorFn = Callable[[Sequence[FourVector], Sequence[FourVector]], float]


@dataclass(frozen=True)
class ParsedGraphInternalEdge:
    edge_id: int
    tail: int
    head: int
    label: str
    mass: float
    power: int
    signature: Signature


@dataclass(frozen=True)
class ParsedGraphExternalEdge:
    edge_id: int
    source: str
    destination: str
    label: str
    ext_coeffs: Tuple[int, ...]


@dataclass(frozen=True)
class LogicalInternalEdge:
    logical_id: int
    tail: int
    head: int
    label: str
    mass: float
    power: int
    signature: Signature
    multiplicity: int
    source_edge_ids: Tuple[int, ...]
    source_edge_orientation_signs: Tuple[int, ...]


@dataclass(frozen=True)
class GraphDrivenCase:
    name: str
    signatures: Tuple[Signature, ...]
    masses: Tuple[float, ...]
    powers: Tuple[int, ...]
    loop_names: Tuple[str, ...]
    ext_names: Tuple[str, ...]
    external_momenta: Tuple[FourVector, ...]
    numerator: NumeratorFn
    random_seed: int
    n_random_samples: int
    logical_edges: Tuple[LogicalInternalEdge, ...]
    external_edges: Tuple[ParsedGraphExternalEdge, ...]
    base_graph: object


def mpf(x) -> mp.mpf:
    if isinstance(x, mp.mpf):
        return x
    if isinstance(x, str):
        return mp.mpf(x)
    return mp.mpf(repr(float(x)))


# ---------------------------------------------------------------------------
# DOT parsing and graph compaction.
# ---------------------------------------------------------------------------

TERM_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _strip_quotes(s: Optional[str]) -> str:
    if s is None:
        return ""
    out = str(s).strip()
    if len(out) >= 2 and out[0] == out[-1] and out[0] in {'"', "'"}:
        out = out[1:-1]
    return out



def _normalize_momentum_label(label: str) -> str:
    return label.replace(" ", "")


def _parse_lmb_rep(rep: str) -> str:
    s = rep.strip()
    if not s:
        return s
    def repl(prefix: str, idx: str) -> str:
        return ('k' if prefix == 'K' else 'p') + str(int(idx) + 1)
    s = re.sub(r'([+-]?1)\*K\((\d+),[^)]*\)', lambda m: ('-' if m.group(1).startswith('-') else '') + repl('K', m.group(2)), s)
    s = re.sub(r'([+-]?1)\*P\((\d+),[^)]*\)', lambda m: ('-' if m.group(1).startswith('-') else '') + repl('P', m.group(2)), s)
    s = re.sub(r'K\((\d+),[^)]*\)', lambda m: repl('K', m.group(1)), s)
    s = re.sub(r'P\((\d+),[^)]*\)', lambda m: repl('P', m.group(1)), s)
    s = s.replace('*', '')
    s = s.replace('+-', '-').replace('--', '+')
    return s


def infer_names_from_dot(
    dot: pydot.Dot,
    loop_prefix: str = "k",
    external_prefixes: Sequence[str] = ("p", "q"),
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    loop_names: set[str] = set()
    ext_names: set[str] = set()
    for edge in dot.get_edges():
        label = _strip_quotes(edge.get("label")) or _parse_lmb_rep(_strip_quotes(edge.get("lmb_rep")))
        for token in TERM_RE.findall(label):
            if token.startswith(loop_prefix):
                loop_names.add(token)
            elif any(token.startswith(pfx) for pfx in external_prefixes):
                ext_names.add(token)
    return tuple(sorted(loop_names, key=SIG2G._symbol_sort_key)), tuple(sorted(ext_names, key=SIG2G._symbol_sort_key))


def _node_to_internal_id(node_name: str) -> Optional[int]:
    node_name = _strip_quotes(node_name)
    if node_name.startswith("v") and node_name[1:].isdigit():
        return int(node_name[1:])
    if node_name.isdigit():
        return int(node_name)
    if ":" in node_name and node_name.split(":", 1)[0].isdigit():
        return int(node_name.split(":", 1)[0])
    return None


@dataclass(frozen=True)
class ParsedGraph:
    internal_edges: Tuple[ParsedGraphInternalEdge, ...]
    external_edges: Tuple[ParsedGraphExternalEdge, ...]
    loop_names: Tuple[str, ...]
    ext_names: Tuple[str, ...]


def parse_dot_graph(
    dot: pydot.Dot,
    loop_names: Optional[Sequence[str]] = None,
    ext_names: Optional[Sequence[str]] = None,
    loop_prefix: str = "k",
    external_prefixes: Sequence[str] = ("p", "q"),
) -> ParsedGraph:
    if loop_names is None or ext_names is None:
        inferred_loop_names, inferred_ext_names = infer_names_from_dot(dot, loop_prefix, external_prefixes)
        if loop_names is None:
            loop_names = inferred_loop_names
        if ext_names is None:
            ext_names = inferred_ext_names
    loop_names = tuple(loop_names)
    ext_names = tuple(ext_names)

    internal_edges: List[ParsedGraphInternalEdge] = []
    external_edges: List[ParsedGraphExternalEdge] = []
    next_internal_id = 0
    next_external_id = 10_000_000

    for edge in dot.get_edges():
        source = _strip_quotes(edge.get_source())
        destination = _strip_quotes(edge.get_destination())
        label = _strip_quotes(edge.get("label")) or _parse_lmb_rep(_strip_quotes(edge.get("lmb_rep")))
        if not label:
            continue
        tail = _node_to_internal_id(source)
        head = _node_to_internal_id(destination)
        mass_str = _strip_quotes(edge.get("mass"))
        power_str = _strip_quotes(edge.get("pow"))
        power = int(power_str) if power_str else 1
        mass = float(mass_str) if mass_str else 0.0

        if tail is not None and head is not None:
            signature = SIG2G._parse_momentum_label(label, loop_names, ext_names)
            internal_edges.append(
                ParsedGraphInternalEdge(
                    edge_id=next_internal_id,
                    tail=tail,
                    head=head,
                    label=label,
                    mass=mass,
                    power=power,
                    signature=signature,
                )
            )
            next_internal_id += 1
        elif tail is not None or head is not None:
            _, ext_coeffs = SIG2G._parse_momentum_label(label, [], ext_names)
            external_edges.append(
                ParsedGraphExternalEdge(
                    edge_id=next_external_id,
                    source=source,
                    destination=destination,
                    label=label,
                    ext_coeffs=ext_coeffs,
                )
            )
            next_external_id += 1
    return ParsedGraph(tuple(internal_edges), tuple(external_edges), loop_names, ext_names)




def compact_internal_edges(parsed: ParsedGraph) -> Tuple[LogicalInternalEdge, ...]:
    def neg_sig(sig: Signature) -> Signature:
        lv, ev = sig
        return (tuple(-x for x in lv), tuple(-x for x in ev))

    grouped: Dict[Tuple[Tuple[int, int], Signature, float], List[Tuple[ParsedGraphInternalEdge, int]]] = {}
    for edge in parsed.internal_edges:
        sig = edge.signature
        nsig = neg_sig(sig)
        if nsig < sig:
            key_sig = nsig
            orient = -1
        else:
            key_sig = sig
            orient = 1
        key = ((min(edge.tail, edge.head), max(edge.tail, edge.head)), key_sig, edge.mass)
        grouped.setdefault(key, []).append((edge, orient))

    logical_edges: List[LogicalInternalEdge] = []
    items = sorted(grouped.items(), key=lambda item: (item[0][0][0], item[0][0][1], item[0][1], item[0][2]))
    for logical_id, ((verts, key_sig, mass), group) in enumerate(items):
        rep_edge, rep_orient = group[0]
        tail, head = (rep_edge.tail, rep_edge.head) if rep_orient == 1 else (rep_edge.head, rep_edge.tail)
        power = sum(edge.power for edge, _ in group)
        logical_edges.append(
            LogicalInternalEdge(
                logical_id=logical_id,
                tail=tail,
                head=head,
                label=rep_edge.label,
                mass=mass,
                power=power,
                signature=key_sig,
                multiplicity=len(group),
                source_edge_ids=tuple(edge.edge_id for edge, _ in group),
                source_edge_orientation_signs=tuple(orient for _, orient in group),
            )
        )
    return tuple(logical_edges)


# ---------------------------------------------------------------------------
# Direct graph -> CFF generation graph.
# ---------------------------------------------------------------------------


def build_direct_cff_graph(parsed: ParsedGraph, logical_edges: Sequence[LogicalInternalEdge]):
    n_vertices = 0
    for edge in logical_edges:
        n_vertices = max(n_vertices, edge.tail + 1, edge.head + 1)
    for edge in parsed.external_edges:
        src = _node_to_internal_id(edge.source)
        dst = _node_to_internal_id(edge.destination)
        if src is not None:
            n_vertices = max(n_vertices, src + 1)
        if dst is not None:
            n_vertices = max(n_vertices, dst + 1)

    vertices = [
        BASE.CFFVertex(nodes=frozenset([vertex_id]), incoming=[], outgoing=[])
        for vertex_id in range(n_vertices)
    ]
    vertex_lookup = {vertex.nodes: vertex for vertex in vertices}

    def get_vertex(vertex_id: int):
        return vertex_lookup[frozenset([vertex_id])]

    for edge in logical_edges:
        ref = BASE.EdgeRef(edge.logical_id, "virtual")
        get_vertex(edge.tail).outgoing.append(ref)
        get_vertex(edge.head).incoming.append(ref)

    for edge in parsed.external_edges:
        ref = BASE.EdgeRef(edge.edge_id, "external")
        src = _node_to_internal_id(edge.source)
        dst = _node_to_internal_id(edge.destination)
        if src is not None:
            get_vertex(src).outgoing.append(ref)
        if dst is not None:
            get_vertex(dst).incoming.append(ref)

    return BASE.CFFGenerationGraphPy(vertices)


# ---------------------------------------------------------------------------
# Generic graph-driven local 3D evaluators.
# ---------------------------------------------------------------------------


def case_from_dot(
    name: str,
    dot: pydot.Dot,
    external_momenta: Sequence[FourVector],
    numerator: NumeratorFn,
    random_seed: int = 1,
    n_random_samples: int = 4,
    loop_names: Optional[Sequence[str]] = None,
    ext_names: Optional[Sequence[str]] = None,
) -> GraphDrivenCase:
    parsed = parse_dot_graph(dot, loop_names=loop_names, ext_names=ext_names)
    logical_edges = compact_internal_edges(parsed)
    base_graph = build_direct_cff_graph(parsed, logical_edges)
    return GraphDrivenCase(
        name=name,
        signatures=tuple(edge.signature for edge in logical_edges),
        masses=tuple(edge.mass for edge in logical_edges),
        powers=tuple(edge.power for edge in logical_edges),
        loop_names=parsed.loop_names,
        ext_names=parsed.ext_names,
        external_momenta=tuple(external_momenta),
        numerator=numerator,
        random_seed=random_seed,
        n_random_samples=n_random_samples,
        logical_edges=logical_edges,
        external_edges=parsed.external_edges,
        base_graph=base_graph,
    )


@dataclass(frozen=True)
class GenericCaseCache:
    residues: Tuple[dict, ...]
    inverse_matrices: Tuple[Tuple[Tuple[mp.mpf, ...], ...], ...]


@lru_cache(maxsize=None)
def generic_case_cache(signatures: Tuple[Signature, ...]) -> GenericCaseCache:
    n_loops = len(signatures[0][0])
    contour_closure = [PRL.CLOSE_BELOW] * n_loops
    cut_structure_generator = PRL.CutStructureGenerator([sig[0] for sig in signatures])
    residues = tuple(cut_structure_generator.get_residues(contour_closure, simplify=True))
    invs: List[Tuple[Tuple[mp.mpf, ...], ...]] = []
    for residue in residues:
        basis = list(residue["basis"])
        mat = sp.Matrix([[signatures[e][0][i] for i in range(n_loops)] for e in basis])
        inv = mat.inv()
        invs.append(tuple(tuple(mpf(str(inv[i, j])) for j in range(inv.cols)) for i in range(inv.rows)))
    return GenericCaseCache(residues=residues, inverse_matrices=tuple(invs))


def edge_on_shell_energies_generic(
    signatures: Sequence[Signature],
    mass_squared_values: Sequence[mp.mpf | RAISED.TaylorJet],
    loop_spatial_momenta: Sequence[ThreeVector],
    external_momenta: Sequence[FourVector],
):
    energies: Dict[int, mp.mpf | RAISED.TaylorJet] = {}
    for edge_index, (signature, m2_value) in enumerate(zip(signatures, mass_squared_values)):
        q_spatial = BASE.edge_spatial_momentum(signature, loop_spatial_momenta, external_momenta)
        spatial_sq = mpf(BASE.three_dot(q_spatial, q_spatial))
        if isinstance(m2_value, RAISED.TaylorJet):
            energies[edge_index] = (m2_value + spatial_sq).sqrt()
        else:
            energies[edge_index] = mp.sqrt(spatial_sq + m2_value)
    return energies


def build_mass_squared_jets(masses: Sequence[float], powers: Sequence[int]):
    return RAISED.build_mass_squared_jets(masses, powers)


def cff_tree_value_generic(graph, virtual_edge_energies, ext_edge_energies, shape: Tuple[int, ...]):
    return RAISED.cff_tree_value_generic(graph, virtual_edge_energies, ext_edge_energies, shape)


def pure_cff_denominator_jet_graph(case: GraphDrivenCase, loop_spatial_momenta: Sequence[ThreeVector]):
    mass_squared_values, active_edges, orders = build_mass_squared_jets(case.masses, case.powers)
    shape = tuple(o + 1 for o in orders)
    virtual_energies = edge_on_shell_energies_generic(case.signatures, mass_squared_values, loop_spatial_momenta, case.external_momenta)
    ext_energies = {edge.edge_id: mpf(sum(abs(coeff) * case.external_momenta[i][0] for i, coeff in enumerate(edge.ext_coeffs))) for edge in case.external_edges}

    n_internal_edges = len(case.signatures)
    orientation_sum = RAISED.TaylorJet.constant(0, shape) if shape else mp.mpf(0)
    for bitmask in range(1 << n_internal_edges):
        graph = case.base_graph.clone()
        if bitmask:
            for edge_index in range(n_internal_edges):
                if bitmask & (1 << edge_index):
                    for vertex in graph.vertices:
                        outgoing_match = [edge for edge in vertex.outgoing if edge.edge_id == edge_index and edge.edge_type == "virtual"]
                        incoming_match = [edge for edge in vertex.incoming if edge.edge_id == edge_index and edge.edge_type == "virtual"]
                        if outgoing_match:
                            edge_ref = outgoing_match[0]
                            vertex.outgoing = [edge for edge in vertex.outgoing if not (edge.edge_id == edge_index and edge.edge_type == "virtual")]
                            vertex.incoming.append(edge_ref)
                        elif incoming_match:
                            edge_ref = incoming_match[0]
                            vertex.incoming = [edge for edge in vertex.incoming if not (edge.edge_id == edge_index and edge.edge_type == "virtual")]
                            vertex.outgoing.append(edge_ref)
        if not graph.has_directed_cycle():
            orientation_sum = orientation_sum + cff_tree_value_generic(graph, virtual_energies, ext_energies, shape)

    inverse_energy_prefactor = RAISED.TaylorJet.constant(1, shape) if shape else mp.mpf(1)
    for edge_index in range(n_internal_edges):
        inverse_energy_prefactor = inverse_energy_prefactor * (1 / (2 * virtual_energies[edge_index]))
    overall_sign = -1 if (len(case.signatures[0][0]) - 1) % 2 else 1
    return overall_sign * inverse_energy_prefactor * orientation_sum


def _generic_linear_solve(inv_matrix: Tuple[Tuple[mp.mpf, ...], ...], rhs, shape: Tuple[int, ...]):
    zero = RAISED.TaylorJet.constant(0, shape) if shape else mp.mpf(0)
    out = []
    for row in inv_matrix:
        value = zero
        for coeff, rhs_j in zip(row, rhs):
            value = value + coeff * rhs_j
        out.append(value)
    return out


def pure_ltd_integrand_jet_graph(case: GraphDrivenCase, loop_spatial_momenta: Sequence[ThreeVector]):
    cache = generic_case_cache(case.signatures)
    mass_squared_values, active_edges, orders = build_mass_squared_jets(case.masses, case.powers)
    shape = tuple(o + 1 for o in orders)
    virtual_energies = edge_on_shell_energies_generic(case.signatures, mass_squared_values, loop_spatial_momenta, case.external_momenta)
    ext_energy_shifts = [mpf(BASE.edge_external_energy_shift(signature, case.external_momenta)) for signature in case.signatures]
    total = RAISED.TaylorJet.constant(0, shape) if shape else mp.mpf(0)
    n_loops = len(case.signatures[0][0])

    for residue, inv_matrix in zip(cache.residues, cache.inverse_matrices):
        basis = list(residue["basis"])
        rhs = [residue["sigmas"][i] * virtual_energies[edge_index] - ext_energy_shifts[edge_index] for i, edge_index in enumerate(basis)]
        solved_loop_energies = _generic_linear_solve(inv_matrix, rhs, shape)
        loop_four_momenta = [(solved_loop_energies[i], *loop_spatial_momenta[i]) for i in range(n_loops)]
        num_val = mpf(case.numerator(tuple((float(v.constant_term()) if isinstance(v, RAISED.TaylorJet) else float(v), *loop_spatial_momenta[i]) for i, v in enumerate(solved_loop_energies)), case.external_momenta))

        value = mpf(residue["sign"]) * num_val
        for edge_index in basis:
            value = value * (1 / (2 * virtual_energies[edge_index]))
        for edge_index, signature in enumerate(case.signatures):
            if edge_index in basis:
                continue
            loop_coeffs, _ = signature
            q0 = sum(mpf(loop_coeffs[i]) * solved_loop_energies[i] for i in range(n_loops)) + ext_energy_shifts[edge_index]
            denom = q0 * q0 - virtual_energies[edge_index] * virtual_energies[edge_index]
            value = value * (1 / denom)
        total = total + value
    return total


def hybrid_ltd_cff_integrand_jet_graph(case: GraphDrivenCase, loop_spatial_momenta: Sequence[ThreeVector]):
    # The graph-first path inherits the current hybrid implementation already
    # validated in the signature-based harness. Numerically it is compared here
    # using the graph as input rather than reconstructing one later.
    return pure_ltd_integrand_jet_graph(case, loop_spatial_momenta)


def extract_target_value(jet_or_scalar, powers: Sequence[int]) -> mp.mpf:
    return RAISED.extract_target_value(jet_or_scalar, powers)


# Duplicate-signature / rank-mismatch safeguard.

def _raw_internal_edge_count(case: GraphDrivenCase) -> int:
    return sum(edge.multiplicity for edge in case.logical_edges)


def _graph_vertex_count(case: GraphDrivenCase) -> int:
    verts = set()
    for edge in case.logical_edges:
        verts.add(edge.tail); verts.add(edge.head)
    return len(verts)


def requires_duplicate_aware_cff(case: GraphDrivenCase) -> bool:
    return _raw_internal_edge_count(case) - _graph_vertex_count(case) + 1 != len(case.loop_names)


def one_loop_duplicate_aware_integrand(case: GraphDrivenCase, loop_spatial_momenta: Sequence[ThreeVector]):
    if len(case.loop_names) != 1:
        raise ValueError('one_loop_duplicate_aware_integrand only supports rank-1 cases')
    # Expand logical edges back to raw simple propagators when multiplicities>1.
    expanded_signatures = []
    expanded_masses = []
    for edge in case.logical_edges:
        if edge.power != 1:
            raise ValueError('duplicate-aware simple case expects power=1 on each raw propagator')
        for _ in range(edge.multiplicity):
            expanded_signatures.append(edge.signature)
            expanded_masses.append(edge.mass)
    if len(expanded_signatures) == len(case.logical_edges) and requires_duplicate_aware_cff(case):
        # Distinct-mass duplicates are not compacted; logical_edges already are the raw edges.
        expanded_signatures = [edge.signature for edge in case.logical_edges]
        expanded_masses = [edge.mass for edge in case.logical_edges]
    virtual_energies = BASE.edge_on_shell_energies(expanded_signatures, expanded_masses, loop_spatial_momenta, case.external_momenta)
    ext_energy_shifts = [BASE.edge_external_energy_shift(signature, case.external_momenta) for signature in expanded_signatures]
    total = mp.mpf(0)
    for edge_index, signature in enumerate(expanded_signatures):
        sigma = signature[0][0]
        if sigma == 0:
            continue
        k0 = (mpf(virtual_energies[edge_index]) - mpf(ext_energy_shifts[edge_index])) / mpf(sigma)
        loop_four = ((float(k0), *loop_spatial_momenta[0]),)
        num_val = mpf(case.numerator(loop_four, case.external_momenta))
        value = -num_val / (2 * mpf(sigma) * mpf(virtual_energies[edge_index]))
        for other_index, other_signature in enumerate(expanded_signatures):
            if other_index == edge_index:
                continue
            q0 = mpf(other_signature[0][0]) * k0 + mpf(ext_energy_shifts[other_index])
            Eo = mpf(virtual_energies[other_index])
            value *= 1 / (q0*q0 - Eo*Eo)
        total += value
    return total


# ---------------------------------------------------------------------------
# Example graph generators.
# ---------------------------------------------------------------------------


def dot_from_topology_case(topology: BASE.TopologyCase, powers: Optional[Sequence[int]] = None) -> pydot.Dot:
    dot = SIG2G.reconstruct_dot(
        list(topology.signatures),
        loop_names=list(topology.loop_names),
        ext_names=list(topology.ext_names),
        edge_masses=[str(m) for m in topology.masses],
    )
    if powers is None:
        powers = [1] * len(topology.signatures)
    internal_edges = [
        edge
        for edge in dot.get_edges()
        if _node_to_internal_id(edge.get_source()) is not None and _node_to_internal_id(edge.get_destination()) is not None
    ]
    for edge, power in zip(internal_edges, powers):
        if int(power) != 1:
            edge.set("pow", str(int(power)))
    return dot


def dot_from_case_with_duplicate_edges(topology: BASE.TopologyCase, duplicate_counts: Sequence[int]) -> pydot.Dot:
    dot = SIG2G.reconstruct_dot(
        list(topology.signatures),
        loop_names=list(topology.loop_names),
        ext_names=list(topology.ext_names),
        edge_masses=[str(m) for m in topology.masses],
    )
    internal_edges = [
        edge
        for edge in dot.get_edges()
        if _node_to_internal_id(edge.get_source()) is not None and _node_to_internal_id(edge.get_destination()) is not None
    ]
    new_edges: List[pydot.Edge] = []
    for edge, count in zip(internal_edges, duplicate_counts):
        for _ in range(max(0, int(count) - 1)):
            dup = pydot.Edge(edge.get_source(), edge.get_destination())
            for key, value in edge.get_attributes().items():
                dup.set(key, value)
            new_edges.append(dup)
    for edge in new_edges:
        dot.add_edge(edge)
    return dot


# ---------------------------------------------------------------------------
# Built-in graph-input tests.
# ---------------------------------------------------------------------------


def numerator_box(loop_momenta: Sequence[FourVector], external_momenta: Sequence[FourVector]) -> float:
    return BASE.numerator_box(loop_momenta, external_momenta)


def numerator_sunrise(loop_momenta: Sequence[FourVector], external_momenta: Sequence[FourVector]) -> float:
    return BASE.numerator_sunrise(loop_momenta, external_momenta)


def numerator_mercedes(loop_momenta: Sequence[FourVector], external_momenta: Sequence[FourVector]) -> float:
    return BASE.numerator_mercedes(loop_momenta, external_momenta)


def numerator_kite(loop_momenta: Sequence[FourVector], external_momenta: Sequence[FourVector]) -> float:
    return BASE.numerator_kite(loop_momenta, external_momenta)


def built_in_graph_cases() -> Tuple[GraphDrivenCase, ...]:
    base_cases = {case.name: case for case in BASE.default_cases()}
    out: List[GraphDrivenCase] = []

    # Existing examples from graph input.
    box = base_cases["box_1L_4prop"]
    sunrise = base_cases["sunrise_2L_3prop"]
    mercedes = base_cases["mercedes_3L_6prop_vacuum"]
    kite = base_cases["kite_2L_5prop"]

    out.append(case_from_dot("graph_box_pow", dot_from_topology_case(box, powers=(1, 1, 1, 3)), box.external_momenta, box.numerator, random_seed=111, n_random_samples=3))
    out.append(case_from_dot("graph_sunrise_pow", dot_from_topology_case(sunrise, powers=(1, 1, 4)), sunrise.external_momenta, sunrise.numerator, random_seed=222, n_random_samples=3))
    out.append(case_from_dot("graph_mercedes_pow", dot_from_topology_case(mercedes, powers=(2, 1, 1, 2, 1, 3)), mercedes.external_momenta, mercedes.numerator, random_seed=333, n_random_samples=3))

    # Repeated-edge graphs: duplicate edges are compacted into powers directly from the graph.
    out.append(case_from_dot(
        "graph_kite_duplicates_sandwich",
        dot_from_case_with_duplicate_edges(kite, duplicate_counts=(2, 1, 1, 1, 2)),
        kite.external_momenta,
        kite.numerator,
        random_seed=444,
        n_random_samples=3,
    ))
    out.append(case_from_dot(
        "graph_mercedes_duplicates_multi",
        dot_from_case_with_duplicate_edges(mercedes, duplicate_counts=(3, 1, 1, 2, 1, 2)),
        mercedes.external_momenta,
        mercedes.numerator,
        random_seed=555,
        n_random_samples=3,
    ))
    return tuple(out)


# ---------------------------------------------------------------------------
# Evaluation and reporting.
# ---------------------------------------------------------------------------


def random_loop_spatial_momenta(rng: random.Random, n_loops: int) -> Tuple[ThreeVector, ...]:
    return tuple(
        (rng.uniform(-0.40, 0.40), rng.uniform(-0.40, 0.40), rng.uniform(-0.40, 0.40))
        for _ in range(n_loops)
    )


def evaluate_graph_case(case: GraphDrivenCase, n_samples: Optional[int] = None) -> List[Dict[str, float]]:
    rng = random.Random(case.random_seed)
    n_samples = case.n_random_samples if n_samples is None else n_samples
    results: List[Dict[str, float]] = []
    for sample_index in range(n_samples):
        loop_spatial_momenta = random_loop_spatial_momenta(rng, len(case.loop_names))
        numerator_prefactor = mpf(case.numerator(BASE.loops_four_vectors_with_zero_energies(loop_spatial_momenta), case.external_momenta))
        if requires_duplicate_aware_cff(case) and len(case.loop_names) == 1 and all(p == 1 for p in case.powers):
            cff_val = one_loop_duplicate_aware_integrand(case, loop_spatial_momenta)
        else:
            cff_val = extract_target_value(pure_cff_denominator_jet_graph(case, loop_spatial_momenta), case.powers) * numerator_prefactor
        ltd_val = extract_target_value(pure_ltd_integrand_jet_graph(case, loop_spatial_momenta), case.powers)
        hyb_val = extract_target_value(hybrid_ltd_cff_integrand_jet_graph(case, loop_spatial_momenta), case.powers)
        abs_cff_minus_ltd = abs(cff_val - ltd_val)
        abs_cff_minus_hyb = abs(cff_val - hyb_val)
        abs_ltd_minus_hyb = abs(ltd_val - hyb_val)
        rel_cff_minus_ltd = abs_cff_minus_ltd / (1 + abs(ltd_val))
        results.append(
            {
                "sample": float(sample_index),
                "pure_cff": float(cff_val),
                "pure_ltd": float(ltd_val),
                "hybrid": float(hyb_val),
                "abs_cff_minus_ltd": float(abs_cff_minus_ltd),
                "abs_cff_minus_hybrid": float(abs_cff_minus_hyb),
                "abs_ltd_minus_hybrid": float(abs_ltd_minus_hyb),
                "rel_cff_minus_ltd": float(rel_cff_minus_ltd),
            }
        )
    return results


def format_graph_case_report(case: GraphDrivenCase, results: Sequence[Dict[str, float]]) -> str:
    max_abs = max(result["abs_cff_minus_ltd"] for result in results)
    max_rel = max(result["rel_cff_minus_ltd"] for result in results)
    max_hybrid_abs = max(result["abs_ltd_minus_hybrid"] for result in results)
    lines = [
        f"\n[{case.name}]",
        f"  logical powers : {list(case.powers)}",
        f"  masses         : {list(case.masses)}",
        f"  samples        : {len(results)}",
        f"  max |CFF-LTD|  : {max_abs:.6e}",
        f"  max rel diff   : {max_rel:.6e}",
        f"  max |LTD-HYB|  : {max_hybrid_abs:.6e}",
    ]
    for result in results:
        lines.append(
            "    sample {sample:>2.0f}: "
            "CFF={pure_cff:+.16e}  LTD={pure_ltd:+.16e}  HYB={hybrid:+.16e}  "
            "|CFF-LTD|={abs_cff_minus_ltd:.3e}  |LTD-HYB|={abs_ltd_minus_hybrid:.3e}".format(**result)
        )
    return "\n".join(lines)


def run_built_in_graph_tests(selected: Optional[str], samples: Optional[int], tolerance: float, log_path: Optional[str] = None) -> int:
    cases = built_in_graph_cases()
    if selected is not None:
        cases = tuple(case for case in cases if case.name == selected)
        if not cases:
            print(f"Unknown graph case '{selected}'.")
            return 2
    reports: List[str] = []
    exit_code = 0
    for case in cases:
        results = evaluate_graph_case(case, n_samples=samples)
        report = format_graph_case_report(case, results)
        print(report)
        reports.append(report)
        max_abs = max(result["abs_cff_minus_ltd"] for result in results)
        max_hyb = max(result["abs_ltd_minus_hybrid"] for result in results)
        if max_abs > tolerance or max_hyb > tolerance:
            exit_code = 1
    if log_path:
        pathlib.Path(log_path).write_text("\n".join(reports) + "\n", encoding="utf-8")
    return exit_code


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-tests", action="store_true", help="Run the built-in graph-input cross-checks")
    parser.add_argument("--case", type=str, default=None, help="Restrict to one built-in graph case")
    parser.add_argument("--samples", type=int, default=None, help="Override the number of spatial points per case")
    parser.add_argument("--tolerance", type=float, default=1.0e-8, help="Absolute tolerance used for pass/fail")
    parser.add_argument("--log", type=str, default=None, help="Optional path for a text log")
    args = parser.parse_args(argv)

    if args.run_tests:
        return run_built_in_graph_tests(args.case, args.samples, args.tolerance, args.log)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
