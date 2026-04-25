#!/usr/bin/env python3
"""
Pure-CFF / pure-LTD / hybrid-LTD+CFF pointwise checker.

What is implemented here
========================

1. A pure-CFF denominator generator in Python that mirrors the graph-based
   source/sink-contraction algorithm of gammaLoop's
   `cff/generation.rs` + `cff/cff_graph.rs` + `cff/tree.rs`:

   - reconstruct a directed multigraph from loop-momentum signatures,
   - enumerate all orientations of the internal edges,
   - keep only acyclic orientations,
   - recursively contract source/sink vertices with connected complement,
   - build the CFF tree and evaluate the sum over branches of products of
     inverse E-surfaces,
   - multiply by the universal product over (2 E_e)^(-1) factors.

   This follows the denominator-generation logic of
   `generate_cff_expression` / `generate_cff_from_orientations` in
   gammaLoop, rather than the FORM numerator engine of `cLTD.m`.

2. A pure-LTD evaluator based on the PRL cut-structure generator supplied by
   the user in `ltd_cut_structure_generator.py`.

3. A hybrid LTD+CFF wrapper. In the simple-power limit nu_e = 1 for all edges,
   the hybrid representation is exactly the pure LTD one. The test cases used
   here all lie in that limit, so the hybrid evaluator is an exact collapse of
   the general hybrid formula.

Numerators
==========

The graph-based CFF generator implemented here is the denominator engine.
For the consistency tests below, the numerators are chosen to be explicit
black-box Python callables that depend only on the spatial components of the
loop four-momenta and on the external four-momenta. This makes them compatible
with all three evaluators without reimplementing the full FORM-based cLTD
numerator instruction layer.

Usage
=====

Run the built-in test suite with:

    python3 simple_power_compare.py --run-tests

or inspect one topology with e.g.:

    python3 simple_power_compare.py --case mercedes --samples 8
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import random
import sys
import pathlib
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
_CORE = _HERE.parent

# ---------------------------------------------------------------------------
# Load the two user-provided helpers.
# ---------------------------------------------------------------------------

def _import_from_path(module_name: str, path: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


SIG2G = _import_from_path("sig2g_runtime", str(_CORE / "graph_signatures.py"))
PRL = _import_from_path("prl_runtime", str(_CORE / "ltd_cut_structure_generator.py"))


# ---------------------------------------------------------------------------
# Types and small utilities.
# ---------------------------------------------------------------------------

LoopVec = Tuple[int, ...]
ExtVec = Tuple[int, ...]
Signature = Tuple[LoopVec, ExtVec]
FourVector = Tuple[float, float, float, float]
ThreeVector = Tuple[float, float, float]
NumeratorFn = Callable[[Sequence[FourVector], Sequence[FourVector]], float]


@dataclass(frozen=True)
class EdgeRef:
    edge_id: int
    edge_type: str  # 'virtual' or 'external'


@dataclass
class CFFVertex:
    nodes: frozenset[int]
    incoming: List[EdgeRef]
    outgoing: List[EdgeRef]

    def vertex_type(self) -> str:
        is_sink = all(edge.edge_type != "virtual" for edge in self.outgoing)
        is_source = all(edge.edge_type != "virtual" for edge in self.incoming)
        if is_sink:
            return "sink"
        if is_source:
            return "source"
        return "none"

    def generates_esurface(self) -> bool:
        vertex_type = self.vertex_type()
        if vertex_type == "sink":
            return all(edge.edge_type == "external" for edge in self.outgoing)
        if vertex_type == "source":
            return all(edge.edge_type == "external" for edge in self.incoming)
        return False


class CFFGenerationGraphPy:
    """Python analogue of the gammaLoop CFFGenerationGraph for full-graph CFF."""

    def __init__(self, vertices: Sequence[CFFVertex]):
        self.vertices = [
            CFFVertex(v.nodes, list(v.incoming), list(v.outgoing)) for v in vertices
        ]
        self.vertices.sort(key=lambda v: tuple(sorted(v.nodes)))

    def clone(self) -> "CFFGenerationGraphPy":
        return CFFGenerationGraphPy(self.vertices)

    def get_vertex(self, nodes: frozenset[int]) -> CFFVertex:
        for vertex in self.vertices:
            if vertex.nodes == nodes:
                return vertex
        raise KeyError(f"Vertex {sorted(nodes)} not found")

    def _directed_virtual_adjacency(self) -> Dict[frozenset[int], List[frozenset[int]]]:
        adjacency: Dict[frozenset[int], List[frozenset[int]]] = {
            vertex.nodes: [] for vertex in self.vertices
        }
        for vertex in self.vertices:
            for edge in vertex.outgoing:
                if edge.edge_type != "virtual":
                    continue
                for other in self.vertices:
                    if other.nodes == vertex.nodes:
                        continue
                    if edge in other.incoming:
                        adjacency[vertex.nodes].append(other.nodes)
                        break
        return adjacency

    def has_directed_cycle(self) -> bool:
        adjacency = self._directed_virtual_adjacency()
        visited: set[frozenset[int]] = set()
        stack: set[frozenset[int]] = set()

        def dfs(node: frozenset[int]) -> bool:
            if node in stack:
                return True
            if node in visited:
                return False
            visited.add(node)
            stack.add(node)
            for nxt in adjacency[node]:
                if dfs(nxt):
                    return True
            stack.remove(node)
            return False

        return any(dfs(node) for node in list(adjacency))

    def _are_directed_adjacent(self, left: frozenset[int], right: frozenset[int]) -> bool:
        left_vertex = self.get_vertex(left)
        right_vertex = self.get_vertex(right)
        return any(
            edge.edge_type == "virtual" and edge in right_vertex.incoming
            for edge in left_vertex.outgoing
        )

    def are_adjacent(self, left: frozenset[int], right: frozenset[int]) -> bool:
        return self._are_directed_adjacent(left, right) or self._are_directed_adjacent(
            right, left
        )

    def get_undirected_neighbours(self, nodes: frozenset[int]) -> List[CFFVertex]:
        return [
            vertex
            for vertex in self.vertices
            if vertex.nodes != nodes and self.are_adjacent(nodes, vertex.nodes)
        ]

    def has_connected_complement(self, removed_nodes: frozenset[int]) -> bool:
        others = [vertex.nodes for vertex in self.vertices if vertex.nodes != removed_nodes]
        if not others:
            return True

        start = others[0]
        visited = {start}
        frontier = [start]
        while frontier:
            current = frontier.pop()
            for neighbour in self.get_undirected_neighbours(current):
                if neighbour.nodes == removed_nodes:
                    continue
                if neighbour.nodes not in visited:
                    visited.add(neighbour.nodes)
                    frontier.append(neighbour.nodes)
        return len(visited) == len(others)

    def get_source_sink_greedy(self) -> Optional[CFFVertex]:
        for vertex in self.vertices:
            if vertex.vertex_type() in {"source", "sink"} and self.has_connected_complement(
                vertex.nodes
            ):
                return vertex
        return None

    def contract_vertices(
        self, left_nodes: frozenset[int], right_nodes: frozenset[int]
    ) -> "CFFGenerationGraphPy":
        left = self.get_vertex(left_nodes)
        right = self.get_vertex(right_nodes)

        right_outgoing = set(right.outgoing)
        right_incoming = set(right.incoming)
        left_outgoing = set(left.outgoing)
        left_incoming = set(left.incoming)

        new_incoming = [
            edge for edge in left.incoming if edge not in right_outgoing
        ] + [edge for edge in right.incoming if edge not in left_outgoing]
        new_outgoing = [
            edge for edge in left.outgoing if edge not in right_incoming
        ] + [edge for edge in right.outgoing if edge not in left_incoming]

        new_vertex = CFFVertex(
            nodes=left.nodes | right.nodes,
            incoming=sorted(new_incoming, key=lambda e: e.edge_id),
            outgoing=sorted(new_outgoing, key=lambda e: e.edge_id),
        )

        new_vertices = [
            CFFVertex(v.nodes, list(v.incoming), list(v.outgoing))
            for v in self.vertices
            if v.nodes not in {left_nodes, right_nodes}
        ]
        new_vertices.append(new_vertex)
        return CFFGenerationGraphPy(new_vertices)


@dataclass(frozen=True)
class ParsedInternalEdge:
    signature_index: int
    tail: int
    head: int


@dataclass(frozen=True)
class ParsedExternalEdge:
    edge_id: int
    source: str
    destination: str
    ext_coeffs: ExtVec
    label: str


@dataclass(frozen=True)
class TopologyCase:
    name: str
    signatures: Tuple[Signature, ...]
    masses: Tuple[float, ...]
    loop_names: Tuple[str, ...]
    ext_names: Tuple[str, ...]
    external_momenta: Tuple[FourVector, ...]
    numerator: NumeratorFn
    random_seed: int
    n_random_samples: int


# ---------------------------------------------------------------------------
# Graph reconstruction and parsing.
# ---------------------------------------------------------------------------

def reconstruct_internal_external_edges(
    signatures: Sequence[Signature],
    loop_names: Sequence[str],
    ext_names: Sequence[str],
) -> Tuple[List[ParsedInternalEdge], List[ParsedExternalEdge]]:
    dot = SIG2G.reconstruct_dot(
        list(signatures), loop_names=list(loop_names), ext_names=list(ext_names)
    )

    # Match internal DOT edges back to the original signature indices using their labels.
    signature_to_indices: Dict[Signature, List[int]] = {}
    for signature_index, signature in enumerate(signatures):
        signature_to_indices.setdefault(signature, []).append(signature_index)
    for indices in signature_to_indices.values():
        indices.reverse()

    internal_edges: List[ParsedInternalEdge] = []
    external_edges: List[ParsedExternalEdge] = []

    next_external_edge_id = len(signatures)
    for edge in dot.get_edges():
        source = str(edge.get_source()).strip('"')
        destination = str(edge.get_destination()).strip('"')
        label = str(edge.get("label")).strip('"') if edge.get("label") is not None else ""

        source_is_internal = source.startswith("v")
        destination_is_internal = destination.startswith("v")

        if source_is_internal and destination_is_internal:
            parsed_signature = SIG2G._parse_momentum_label(label, loop_names, ext_names)
            try:
                signature_index = signature_to_indices[parsed_signature].pop()
            except (KeyError, IndexError) as exc:
                raise ValueError(
                    f"Could not match internal edge label '{label}' back to an input signature"
                ) from exc
            internal_edges.append(
                ParsedInternalEdge(
                    signature_index=signature_index,
                    tail=int(source[1:]),
                    head=int(destination[1:]),
                )
            )
        elif source_is_internal or destination_is_internal:
            _, ext_coeffs = SIG2G._parse_momentum_label(label, [], ext_names)
            external_edges.append(
                ParsedExternalEdge(
                    edge_id=next_external_edge_id,
                    source=source,
                    destination=destination,
                    ext_coeffs=ext_coeffs,
                    label=label,
                )
            )
            next_external_edge_id += 1

    return internal_edges, external_edges


def build_initial_cff_graph(
    signatures: Sequence[Signature],
    loop_names: Sequence[str],
    ext_names: Sequence[str],
) -> Tuple[CFFGenerationGraphPy, List[ParsedExternalEdge]]:
    internal_edges, external_edges = reconstruct_internal_external_edges(
        signatures, loop_names, ext_names
    )

    n_vertices = 0
    if internal_edges:
        n_vertices = max(
            max(edge.tail, edge.head) for edge in internal_edges
        ) + 1
    for edge in external_edges:
        if edge.source.startswith("v"):
            n_vertices = max(n_vertices, int(edge.source[1:]) + 1)
        if edge.destination.startswith("v"):
            n_vertices = max(n_vertices, int(edge.destination[1:]) + 1)

    vertices = [
        CFFVertex(nodes=frozenset([vertex_id]), incoming=[], outgoing=[])
        for vertex_id in range(n_vertices)
    ]
    vertex_lookup = {vertex.nodes: vertex for vertex in vertices}

    def get_vertex(vertex_id: int) -> CFFVertex:
        return vertex_lookup[frozenset([vertex_id])]

    for edge in internal_edges:
        ref = EdgeRef(edge.signature_index, "virtual")
        get_vertex(edge.tail).outgoing.append(ref)
        get_vertex(edge.head).incoming.append(ref)

    for edge in external_edges:
        ref = EdgeRef(edge.edge_id, "external")
        if edge.source.startswith("v"):
            get_vertex(int(edge.source[1:])).outgoing.append(ref)
        if edge.destination.startswith("v"):
            get_vertex(int(edge.destination[1:])).incoming.append(ref)

    return CFFGenerationGraphPy(vertices), external_edges


# ---------------------------------------------------------------------------
# Momentum and on-shell-energy helpers.
# ---------------------------------------------------------------------------

def three_add(a: Sequence[float], b: Sequence[float]) -> ThreeVector:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def three_scale(c: float, v: Sequence[float]) -> ThreeVector:
    return (c * v[0], c * v[1], c * v[2])


def three_dot(a: Sequence[float], b: Sequence[float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def edge_spatial_momentum(
    signature: Signature,
    loop_spatial_momenta: Sequence[ThreeVector],
    external_momenta: Sequence[FourVector],
) -> ThreeVector:
    loop_coeffs, ext_coeffs = signature
    momentum = (0.0, 0.0, 0.0)
    for loop_index, coefficient in enumerate(loop_coeffs):
        if coefficient != 0:
            momentum = three_add(momentum, three_scale(coefficient, loop_spatial_momenta[loop_index]))
    for ext_index, coefficient in enumerate(ext_coeffs):
        if coefficient != 0:
            spatial = external_momenta[ext_index][1:]
            momentum = three_add(momentum, three_scale(coefficient, spatial))
    return momentum


def edge_external_energy_shift(signature: Signature, external_momenta: Sequence[FourVector]) -> float:
    _, ext_coeffs = signature
    return sum(coefficient * external_momenta[i][0] for i, coefficient in enumerate(ext_coeffs))


def edge_on_shell_energies(
    signatures: Sequence[Signature],
    masses: Sequence[float],
    loop_spatial_momenta: Sequence[ThreeVector],
    external_momenta: Sequence[FourVector],
) -> Dict[int, float]:
    energies: Dict[int, float] = {}
    for edge_index, (signature, mass) in enumerate(zip(signatures, masses)):
        q_spatial = edge_spatial_momentum(signature, loop_spatial_momenta, external_momenta)
        energies[edge_index] = math.sqrt(three_dot(q_spatial, q_spatial) + mass * mass)
    return energies


def external_edge_energies(
    parsed_external_edges: Sequence[ParsedExternalEdge],
    external_momenta: Sequence[FourVector],
) -> Dict[int, float]:
    # The graph generator uses unsigned external-edge energy symbols OSE(i).
    # The sign of the external shift is carried by the source/sink logic, not by the edge label.
    return {
        edge.edge_id: sum(abs(coeff) * external_momenta[i][0] for i, coeff in enumerate(edge.ext_coeffs))
        for edge in parsed_external_edges
    }


def loops_four_vectors_with_zero_energies(
    loop_spatial_momenta: Sequence[ThreeVector],
) -> Tuple[FourVector, ...]:
    return tuple((0.0, kx, ky, kz) for (kx, ky, kz) in loop_spatial_momenta)


# ---------------------------------------------------------------------------
# Pure CFF denominator generation.
# ---------------------------------------------------------------------------

def cff_tree_value(
    graph: CFFGenerationGraphPy,
    virtual_edge_energies: Dict[int, float],
    ext_edge_energies: Dict[int, float],
) -> float:
    def recurse(current_graph: CFFGenerationGraphPy) -> float:
        if len(current_graph.vertices) < 2:
            return 1.0

        vertex = current_graph.get_source_sink_greedy()
        if vertex is None:
            raise RuntimeError("No valid source/sink found during CFF tree generation")

        vertex_type = vertex.vertex_type()
        if vertex_type not in {"source", "sink"}:
            raise RuntimeError("Selected vertex is neither a source nor a sink")

        external_shift = 0.0
        for edge in vertex.incoming:
            if edge.edge_type == "external":
                external_shift += (1.0 if vertex_type == "sink" else -1.0) * ext_edge_energies[edge.edge_id]
        for edge in vertex.outgoing:
            if edge.edge_type == "external":
                external_shift += (-1.0 if vertex_type == "sink" else 1.0) * ext_edge_energies[edge.edge_id]

        incident_virtual_edges = {
            edge.edge_id
            for edge in list(vertex.incoming) + list(vertex.outgoing)
            if edge.edge_type == "virtual"
        }
        surface_value = sum(virtual_edge_energies[edge_id] for edge_id in incident_virtual_edges) + external_shift

        if len(current_graph.vertices) == 2:
            return 1.0 / surface_value

        child_sum = 0.0
        for neighbour in current_graph.get_undirected_neighbours(vertex.nodes):
            child_graph = current_graph.contract_vertices(vertex.nodes, neighbour.nodes)
            if not child_graph.has_directed_cycle():
                child_sum += recurse(child_graph)
        return (1.0 / surface_value) * child_sum

    return recurse(graph)


def pure_cff_denominator(
    signatures: Sequence[Signature],
    masses: Sequence[float],
    loop_names: Sequence[str],
    ext_names: Sequence[str],
    loop_spatial_momenta: Sequence[ThreeVector],
    external_momenta: Sequence[FourVector],
) -> float:
    base_graph, parsed_external_edges = build_initial_cff_graph(signatures, loop_names, ext_names)
    virtual_energies = edge_on_shell_energies(signatures, masses, loop_spatial_momenta, external_momenta)
    ext_energies = external_edge_energies(parsed_external_edges, external_momenta)

    n_internal_edges = len(signatures)
    orientation_sum = 0.0
    for bitmask in range(1 << n_internal_edges):
        graph = base_graph.clone()
        if bitmask:
            for edge_index in range(n_internal_edges):
                if bitmask & (1 << edge_index):
                    for vertex in graph.vertices:
                        outgoing_match = [
                            edge
                            for edge in vertex.outgoing
                            if edge.edge_id == edge_index and edge.edge_type == "virtual"
                        ]
                        incoming_match = [
                            edge
                            for edge in vertex.incoming
                            if edge.edge_id == edge_index and edge.edge_type == "virtual"
                        ]
                        if outgoing_match:
                            edge_ref = outgoing_match[0]
                            vertex.outgoing = [
                                edge
                                for edge in vertex.outgoing
                                if not (edge.edge_id == edge_index and edge.edge_type == "virtual")
                            ]
                            vertex.incoming.append(edge_ref)
                        elif incoming_match:
                            edge_ref = incoming_match[0]
                            vertex.incoming = [
                                edge
                                for edge in vertex.incoming
                                if not (edge.edge_id == edge_index and edge.edge_type == "virtual")
                            ]
                            vertex.outgoing.append(edge_ref)

        if not graph.has_directed_cycle():
            orientation_sum += cff_tree_value(graph, virtual_energies, ext_energies)

    inverse_energy_prefactor = 1.0
    for edge_index in range(n_internal_edges):
        inverse_energy_prefactor *= 1.0 / (2.0 * virtual_energies[edge_index])

    n_loops = len(signatures[0][0])
    overall_sign = (-1.0) ** (n_loops - 1)
    return overall_sign * inverse_energy_prefactor * orientation_sum


# ---------------------------------------------------------------------------
# Pure LTD and hybrid (simple-power collapse) evaluators.
# ---------------------------------------------------------------------------

def pure_ltd_integrand(
    signatures: Sequence[Signature],
    masses: Sequence[float],
    loop_spatial_momenta: Sequence[ThreeVector],
    external_momenta: Sequence[FourVector],
    numerator_fn: Optional[NumeratorFn] = None,
) -> float:
    n_loops = len(signatures[0][0])
    contour_closure = [PRL.CLOSE_BELOW] * n_loops
    cut_structure_generator = PRL.CutStructureGenerator([signature[0] for signature in signatures])
    residues = cut_structure_generator.get_residues(contour_closure, simplify=True)

    virtual_energies = edge_on_shell_energies(signatures, masses, loop_spatial_momenta, external_momenta)
    ext_energy_shifts = [
        edge_external_energy_shift(signature, external_momenta) for signature in signatures
    ]

    total = 0.0
    for residue in residues:
        basis = list(residue["basis"])
        signature_matrix = np.array(
            [[signatures[edge_index][0][loop_index] for loop_index in range(n_loops)] for edge_index in basis],
            dtype=float,
        )
        rhs = np.array(
            [
                residue["sigmas"][i] * virtual_energies[edge_index] - ext_energy_shifts[edge_index]
                for i, edge_index in enumerate(basis)
            ],
            dtype=float,
        )
        solved_loop_energies = np.linalg.solve(signature_matrix, rhs)

        loop_four_momenta = [
            (float(solved_loop_energies[i]), *loop_spatial_momenta[i]) for i in range(n_loops)
        ]

        value = float(residue["sign"])
        for edge_index in basis:
            value *= 1.0 / (2.0 * virtual_energies[edge_index])

        for edge_index, signature in enumerate(signatures):
            if edge_index in basis:
                continue
            loop_coeffs, _ = signature
            q0 = sum(loop_coeffs[i] * solved_loop_energies[i] for i in range(n_loops)) + ext_energy_shifts[edge_index]
            value *= 1.0 / (q0 * q0 - virtual_energies[edge_index] * virtual_energies[edge_index])

        if numerator_fn is not None:
            value *= numerator_fn(loop_four_momenta, external_momenta)
        total += value

    return total


def hybrid_ltd_cff_integrand_simple_limit(
    signatures: Sequence[Signature],
    masses: Sequence[float],
    powers: Sequence[int],
    loop_spatial_momenta: Sequence[ThreeVector],
    external_momenta: Sequence[FourVector],
    numerator_fn: Optional[NumeratorFn] = None,
) -> float:
    if any(power != 1 for power in powers):
        raise NotImplementedError(
            "This script implements the simple-power collapse of the hybrid formula. "
            "Raised-propagator specialisations were checked separately in the dedicated dotted-box "
            "and dotted-sunrise scripts produced earlier in the conversation."
        )
    return pure_ltd_integrand(
        signatures,
        masses,
        loop_spatial_momenta,
        external_momenta,
        numerator_fn=numerator_fn,
    )


# ---------------------------------------------------------------------------
# Test numerators (chosen energy-independent on purpose).
# ---------------------------------------------------------------------------

def numerator_box(loop_momenta: Sequence[FourVector], external_momenta: Sequence[FourVector]) -> float:
    k = loop_momenta[0]
    p1 = external_momenta[0]
    return 1.0 + k[1] * p1[1] + k[2] * p1[2] + k[3] * p1[3]


def numerator_sunrise(loop_momenta: Sequence[FourVector], external_momenta: Sequence[FourVector]) -> float:
    k1, k2 = loop_momenta
    p = external_momenta[0]
    return 1.0 + k1[1] * p[1] - k1[2] * p[2] + k2[1] * p[1]


def numerator_kite(loop_momenta: Sequence[FourVector], external_momenta: Sequence[FourVector]) -> float:
    k1, k2 = loop_momenta
    p1, p2 = external_momenta
    return (
        1.0
        + k1[1] * p1[1]
        - k2[2] * p2[2]
        + k1[1] * k2[1]
        + k1[2] * k2[2]
        + k1[3] * k2[3]
    )


def numerator_mercedes(loop_momenta: Sequence[FourVector], external_momenta: Sequence[FourVector]) -> float:
    del external_momenta
    k1, k2, k3 = loop_momenta
    dot12 = k1[1] * k2[1] + k1[2] * k2[2] + k1[3] * k2[3]
    dot23 = k2[1] * k3[1] + k2[2] * k3[2] + k2[3] * k3[3]
    return 1.0 + dot12 - dot23


# ---------------------------------------------------------------------------
# Built-in non-trivial topology cases.
# ---------------------------------------------------------------------------

def default_cases() -> Tuple[TopologyCase, ...]:
    return (
        TopologyCase(
            name="box_1L_4prop",
            signatures=(
                ((1,), (0, 0, 0)),
                ((1,), (1, 0, 0)),
                ((1,), (1, 1, 0)),
                ((1,), (1, 1, 1)),
            ),
            masses=(0.80, 1.10, 0.90, 1.20),
            loop_names=("k1",),
            ext_names=("p1", "p2", "p3"),
            external_momenta=(
                (0.30, 0.10, -0.20, 0.05),
                (-0.15, 0.20, 0.05, -0.10),
                (0.25, -0.10, 0.15, 0.07),
            ),
            numerator=numerator_box,
            random_seed=101,
            n_random_samples=8,
        ),
        TopologyCase(
            name="sunrise_2L_3prop",
            signatures=(
                ((1, 0), (0,)),
                ((0, 1), (0,)),
                ((-1, -1), (1,)),
            ),
            masses=(0.80, 1.00, 1.10),
            loop_names=("k1", "k2"),
            ext_names=("p1",),
            external_momenta=((0.60, 0.10, -0.07, 0.03),),
            numerator=numerator_sunrise,
            random_seed=202,
            n_random_samples=8,
        ),
        TopologyCase(
            name="kite_2L_5prop",
            signatures=(
                ((1, 0), (0, 0)),
                ((1, 0), (1, 0)),
                ((1, -1), (0, 0)),
                ((0, 1), (0, 0)),
                ((0, 1), (0, -1)),
            ),
            masses=(0.70, 1.20, 0.90, 1.10, 0.95),
            loop_names=("k1", "k2"),
            ext_names=("p1", "p2"),
            external_momenta=(
                (0.45, 0.13, -0.05, 0.02),
                (-0.20, 0.02, 0.08, -0.03),
            ),
            numerator=numerator_kite,
            random_seed=303,
            n_random_samples=8,
        ),
        TopologyCase(
            name="mercedes_3L_6prop_vacuum",
            signatures=(
                ((1, 0, 0), ()),
                ((0, 1, 0), ()),
                ((0, 0, 1), ()),
                ((1, -1, 0), ()),
                ((-1, 0, 1), ()),
                ((0, 1, -1), ()),
            ),
            masses=(0.90, 1.00, 1.10, 0.95, 1.05, 0.85),
            loop_names=("k1", "k2", "k3"),
            ext_names=(),
            external_momenta=(),
            numerator=numerator_mercedes,
            random_seed=404,
            n_random_samples=8,
        ),
    )


# ---------------------------------------------------------------------------
# Drivers.
# ---------------------------------------------------------------------------

def random_loop_spatial_momenta(rng: random.Random, n_loops: int) -> Tuple[ThreeVector, ...]:
    return tuple(
        (rng.uniform(-0.40, 0.40), rng.uniform(-0.40, 0.40), rng.uniform(-0.40, 0.40))
        for _ in range(n_loops)
    )


def evaluate_case(case: TopologyCase, n_samples: Optional[int] = None) -> List[Dict[str, float]]:
    rng = random.Random(case.random_seed)
    powers = [1] * len(case.signatures)
    n_samples = case.n_random_samples if n_samples is None else n_samples

    results: List[Dict[str, float]] = []
    for sample_index in range(n_samples):
        loop_spatial_momenta = random_loop_spatial_momenta(rng, len(case.loop_names))
        loop_four_zero = loops_four_vectors_with_zero_energies(loop_spatial_momenta)
        numerator_prefactor = case.numerator(loop_four_zero, case.external_momenta)

        pure_cff = pure_cff_denominator(
            case.signatures,
            case.masses,
            case.loop_names,
            case.ext_names,
            loop_spatial_momenta,
            case.external_momenta,
        ) * numerator_prefactor

        pure_ltd = pure_ltd_integrand(
            case.signatures,
            case.masses,
            loop_spatial_momenta,
            case.external_momenta,
            numerator_fn=case.numerator,
        )

        hybrid = hybrid_ltd_cff_integrand_simple_limit(
            case.signatures,
            case.masses,
            powers,
            loop_spatial_momenta,
            case.external_momenta,
            numerator_fn=case.numerator,
        )

        results.append(
            {
                "sample": float(sample_index),
                "pure_cff": pure_cff,
                "pure_ltd": pure_ltd,
                "hybrid": hybrid,
                "abs_cff_minus_ltd": abs(pure_cff - pure_ltd),
                "abs_cff_minus_hybrid": abs(pure_cff - hybrid),
                "abs_ltd_minus_hybrid": abs(pure_ltd - hybrid),
                "rel_cff_minus_ltd": abs(pure_cff - pure_ltd) / (1.0 + abs(pure_ltd)),
            }
        )
    return results


def print_case_report(case: TopologyCase, results: Sequence[Dict[str, float]]) -> None:
    max_abs = max(result["abs_cff_minus_ltd"] for result in results)
    max_rel = max(result["rel_cff_minus_ltd"] for result in results)
    print(f"\n[{case.name}]")
    print(f"  samples      : {len(results)}")
    print(f"  max |CFF-LTD|: {max_abs:.6e}")
    print(f"  max rel diff : {max_rel:.6e}")
    for result in results:
        print(
            "    sample {sample:>2.0f}: "
            "CFF={pure_cff:+.16e}  LTD={pure_ltd:+.16e}  HYB={hybrid:+.16e}  "
            "|Δ|={abs_cff_minus_ltd:.3e}".format(**result)
        )


def run_tests(selected_case: Optional[str], n_samples: Optional[int], tolerance: float) -> int:
    cases = default_cases()
    if selected_case is not None:
        filtered = [case for case in cases if case.name == selected_case]
        if not filtered:
            print(f"Unknown case '{selected_case}'. Available cases:")
            for case in cases:
                print(f"  - {case.name}")
            return 2
        cases = tuple(filtered)

    exit_code = 0
    for case in cases:
        results = evaluate_case(case, n_samples=n_samples)
        print_case_report(case, results)
        max_abs = max(result["abs_cff_minus_ltd"] for result in results)
        max_hybrid_abs = max(result["abs_ltd_minus_hybrid"] for result in results)
        if max_abs > tolerance or max_hybrid_abs > tolerance:
            exit_code = 1
    return exit_code


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-tests", action="store_true", help="Run the built-in multi-topology cross-checks")
    parser.add_argument("--case", type=str, default=None, help="Restrict to one built-in case name")
    parser.add_argument("--samples", type=int, default=None, help="Override the number of random spatial points per case")
    parser.add_argument("--tolerance", type=float, default=1.0e-9, help="Absolute tolerance used for pass/fail")
    args = parser.parse_args(argv)

    if args.run_tests:
        return run_tests(args.case, args.samples, args.tolerance)

    parser.print_help()
    print("\nAvailable built-in cases:")
    for case in default_cases():
        print(f"  - {case.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
