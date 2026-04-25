#!/usr/bin/env python3
"""
Extended raised-propagator pointwise cross-checks on top of
`simple_power_compare.py`.

This version replaces the earlier finite-difference promotion by an exact
multivariate Taylor-jet lift in the mass-squared variables.  It keeps the same
simple-power local 3D evaluators and the same topology/numerator choices as the
base harness, then promotes them to arbitrary positive propagator powers via

    I_{ {nu_e} } = [ prod_e 1/(nu_e-1)! * d^(nu_e-1)/d(m_e^2)^(nu_e-1) ] I_{all nu=1},

which for the Taylor series means that the desired dotted integrand is exactly
just the mixed Taylor coefficient with multi-index (nu_e-1).

Scope of the numerators
=======================
The built-in numerators are the same ones already used by the base harness and
remain energy-independent (they depend only on the spatial loop components and
external four-momenta).  That is the regime this validator targets.

What is compared
================
- pure CFF, via a Python port of the graph-side CFF generation logic,
- pure LTD, via the PRL cut-structure generator,
- hybrid LTD+CFF, in the current codebase's simple-power collapse.

For the higher-dot tests, the hybrid lane is obtained by applying the same
Taylor-jet lift to the simple-power hybrid representation.  Since the simple
hybrid equals simple LTD in the base harness, LTD and hybrid are expected to
agree up to arithmetic noise; the genuinely non-trivial comparison is pure CFF
versus LTD/hybrid.
"""

from __future__ import annotations

import argparse
import importlib.util
import pathlib

_HERE = pathlib.Path(__file__).resolve().parent
_CORE = _HERE.parent
import random
import sys
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import mpmath as mp
import numpy as np
import sympy as sp

mp.mp.dps = 80


# ---------------------------------------------------------------------------
# Load the base simple-power comparison harness.
# ---------------------------------------------------------------------------

def _import_from_path(module_name: str, path: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


BASE = _import_from_path("python_cff_ltd_hybrid_compare_base_v2", str((_HERE / "simple_power_compare.py")))
PRL = BASE.PRL

Signature = BASE.Signature
ThreeVector = Tuple[float, float, float]
FourVector = BASE.FourVector
TopologyCase = BASE.TopologyCase


def mpf(x) -> mp.mpf:
    if isinstance(x, mp.mpf):
        return x
    if isinstance(x, str):
        return mp.mpf(x)
    return mp.mpf(repr(float(x)))


# ---------------------------------------------------------------------------
# Sparse multivariate Taylor jet with coefficients scaled by 1/(multi-index!).
# ---------------------------------------------------------------------------

def _iter_multiindices(shape: Tuple[int, ...]) -> Iterator[Tuple[int, ...]]:
    if not shape:
        yield ()
        return
    import itertools

    for idx in itertools.product(*[range(n) for n in shape]):
        yield idx


def _iter_multiindices_by_degree(shape: Tuple[int, ...]) -> Iterator[Tuple[int, ...]]:
    all_idx = list(_iter_multiindices(shape))
    all_idx.sort(key=lambda idx: (sum(idx), idx))
    yield from all_idx


def _le_multi(a: Tuple[int, ...], b: Tuple[int, ...]) -> bool:
    return all(x <= y for x, y in zip(a, b))


def _sub_multi(a: Tuple[int, ...], b: Tuple[int, ...]) -> Tuple[int, ...]:
    return tuple(x - y for x, y in zip(a, b))


class TaylorJet:
    __slots__ = ("shape", "coeffs")
    __array_priority__ = 1000

    def __init__(self, shape: Tuple[int, ...], coeffs: Optional[Dict[Tuple[int, ...], mp.mpf]] = None):
        self.shape = tuple(shape)
        self.coeffs: Dict[Tuple[int, ...], mp.mpf] = coeffs if coeffs is not None else {}

    @staticmethod
    def constant(value, shape: Tuple[int, ...]) -> "TaylorJet":
        coeffs: Dict[Tuple[int, ...], mp.mpf] = {}
        value = mpf(value)
        if value != 0:
            coeffs[(0,) * len(shape)] = value
        return TaylorJet(shape, coeffs)

    @staticmethod
    def variable(base_value, var_index: int, shape: Tuple[int, ...]) -> "TaylorJet":
        jet = TaylorJet.constant(base_value, shape)
        idx = [0] * len(shape)
        idx[var_index] = 1
        jet.coeffs[tuple(idx)] = mp.mpf(1)
        return jet

    def constant_term(self) -> mp.mpf:
        return self.coeffs.get((0,) * len(self.shape), mp.mpf(0))

    def coefficient(self, idx: Tuple[int, ...]) -> mp.mpf:
        return self.coeffs.get(tuple(idx), mp.mpf(0))

    def _coerce(self, other) -> "TaylorJet":
        if isinstance(other, TaylorJet):
            if other.shape != self.shape:
                raise ValueError(f"jet shape mismatch: {self.shape} vs {other.shape}")
            return other
        return TaylorJet.constant(other, self.shape)

    def __add__(self, other):
        other = self._coerce(other)
        keys = set(self.coeffs) | set(other.coeffs)
        out = {k: self.coefficient(k) + other.coefficient(k) for k in keys}
        out = {k: v for k, v in out.items() if v != 0}
        return TaylorJet(self.shape, out)

    __radd__ = __add__

    def __sub__(self, other):
        other = self._coerce(other)
        keys = set(self.coeffs) | set(other.coeffs)
        out = {k: self.coefficient(k) - other.coefficient(k) for k in keys}
        out = {k: v for k, v in out.items() if v != 0}
        return TaylorJet(self.shape, out)

    def __rsub__(self, other):
        return self._coerce(other).__sub__(self)

    def __neg__(self):
        return TaylorJet(self.shape, {k: -v for k, v in self.coeffs.items()})

    def __mul__(self, other):
        other = self._coerce(other)
        out: Dict[Tuple[int, ...], mp.mpf] = {}
        for a, va in self.coeffs.items():
            for b, vb in other.coeffs.items():
                idx = tuple(x + y for x, y in zip(a, b))
                if any(idx[i] >= self.shape[i] for i in range(len(self.shape))):
                    continue
                out[idx] = out.get(idx, mp.mpf(0)) + va * vb
        out = {k: v for k, v in out.items() if v != 0}
        return TaylorJet(self.shape, out)

    __rmul__ = __mul__

    def reciprocal(self):
        a0 = self.constant_term()
        if a0 == 0:
            raise ZeroDivisionError("TaylorJet reciprocal with zero constant term")
        zero = (0,) * len(self.shape)
        out: Dict[Tuple[int, ...], mp.mpf] = {zero: 1 / a0}
        for alpha in _iter_multiindices_by_degree(self.shape):
            if alpha == zero:
                continue
            total = mp.mpf(0)
            for beta, fbeta in self.coeffs.items():
                if beta == zero or not _le_multi(beta, alpha):
                    continue
                gamma = _sub_multi(alpha, beta)
                total += fbeta * out.get(gamma, mp.mpf(0))
            if total != 0:
                out[alpha] = -total / a0
        return TaylorJet(self.shape, out)

    def __truediv__(self, other):
        return self * self._coerce(other).reciprocal()

    def __rtruediv__(self, other):
        return self._coerce(other) * self.reciprocal()

    def sqrt(self):
        a0 = self.constant_term()
        if a0 <= 0:
            raise ValueError(f"TaylorJet sqrt requires positive constant term, got {a0}")
        s0 = mp.sqrt(a0)
        zero = (0,) * len(self.shape)
        out: Dict[Tuple[int, ...], mp.mpf] = {zero: s0}
        for alpha in _iter_multiindices_by_degree(self.shape):
            if alpha == zero:
                continue
            total = mp.mpf(0)
            for beta, sbeta in list(out.items()):
                if beta == zero or beta == alpha or not _le_multi(beta, alpha):
                    continue
                gamma = _sub_multi(alpha, beta)
                total += sbeta * out.get(gamma, mp.mpf(0))
            coeff = (self.coefficient(alpha) - total) / (2 * s0)
            if coeff != 0:
                out[alpha] = coeff
        return TaylorJet(self.shape, out)

    def __repr__(self):
        return f"TaylorJet(shape={self.shape}, c0={self.constant_term()})"


# ---------------------------------------------------------------------------
# Mass^2 jet helpers.
# ---------------------------------------------------------------------------

def build_mass_squared_jets(masses: Sequence[float], powers: Sequence[int]):
    active_edges = tuple(i for i, nu in enumerate(powers) if nu > 1)
    orders = tuple(powers[i] - 1 for i in active_edges)
    shape = tuple(o + 1 for o in orders)
    if not active_edges:
        return [mpf(m * m) for m in masses], active_edges, orders
    active_pos = {edge: pos for pos, edge in enumerate(active_edges)}
    out = []
    for i, mass in enumerate(masses):
        base = mpf(mass * mass)
        if i in active_pos:
            out.append(TaylorJet.variable(base, active_pos[i], shape))
        else:
            out.append(TaylorJet.constant(base, shape))
    return out, active_edges, orders


def edge_on_shell_energies_generic(
    signatures: Sequence[Signature],
    mass_squared_values: Sequence[mp.mpf | TaylorJet],
    loop_spatial_momenta: Sequence[ThreeVector],
    external_momenta: Sequence[FourVector],
):
    energies: Dict[int, mp.mpf | TaylorJet] = {}
    for edge_index, (signature, m2_value) in enumerate(zip(signatures, mass_squared_values)):
        q_spatial = BASE.edge_spatial_momentum(signature, loop_spatial_momenta, external_momenta)
        spatial_sq = mpf(BASE.three_dot(q_spatial, q_spatial))
        if isinstance(m2_value, TaylorJet):
            energies[edge_index] = (m2_value + spatial_sq).sqrt()
        else:
            energies[edge_index] = mp.sqrt(spatial_sq + m2_value)
    return energies


# ---------------------------------------------------------------------------
# Precomputed caches.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CaseCache:
    base_graph: object
    parsed_external_edges: Tuple[object, ...]
    residues: Tuple[dict, ...]
    inverse_matrices: Tuple[Tuple[Tuple[mp.mpf, ...], ...], ...]


@lru_cache(maxsize=None)
def get_case_cache(case_name: str) -> CaseCache:
    case = next(c for c in BASE.default_cases() if c.name == case_name)
    base_graph, parsed_external_edges = BASE.build_initial_cff_graph(case.signatures, case.loop_names, case.ext_names)
    n_loops = len(case.signatures[0][0])
    contour_closure = [PRL.CLOSE_BELOW] * n_loops
    cut_structure_generator = PRL.CutStructureGenerator([sig[0] for sig in case.signatures])
    residues = tuple(cut_structure_generator.get_residues(contour_closure, simplify=True))
    invs: List[Tuple[Tuple[mp.mpf, ...], ...]] = []
    for residue in residues:
        basis = list(residue["basis"])
        mat = sp.Matrix([[case.signatures[e][0][i] for i in range(n_loops)] for e in basis])
        inv = mat.inv()
        invs.append(tuple(tuple(mpf(str(inv[i, j])) for j in range(inv.cols)) for i in range(inv.rows)))
    return CaseCache(
        base_graph=base_graph,
        parsed_external_edges=tuple(parsed_external_edges),
        residues=residues,
        inverse_matrices=tuple(invs),
    )


# ---------------------------------------------------------------------------
# Jet-aware local CFF/LTD/hybrid integrands.
# ---------------------------------------------------------------------------

def cff_tree_value_generic(graph, virtual_edge_energies, ext_edge_energies, shape: Tuple[int, ...]):
    zero = TaylorJet.constant(0, shape) if shape else mp.mpf(0)
    one = TaylorJet.constant(1, shape) if shape else mp.mpf(1)

    def recurse(current_graph):
        if len(current_graph.vertices) < 2:
            return one
        vertex = current_graph.get_source_sink_greedy()
        if vertex is None:
            raise RuntimeError("No valid source/sink found during CFF tree generation")
        vertex_type = vertex.vertex_type()

        external_shift = mp.mpf(0)
        for edge in vertex.incoming:
            if edge.edge_type == "external":
                external_shift += (1 if vertex_type == "sink" else -1) * ext_edge_energies[edge.edge_id]
        for edge in vertex.outgoing:
            if edge.edge_type == "external":
                external_shift += (-1 if vertex_type == "sink" else 1) * ext_edge_energies[edge.edge_id]

        incident_virtual_edges = {
            edge.edge_id
            for edge in list(vertex.incoming) + list(vertex.outgoing)
            if edge.edge_type == "virtual"
        }
        surface_value = zero + external_shift
        for edge_id in incident_virtual_edges:
            surface_value = surface_value + virtual_edge_energies[edge_id]

        if len(current_graph.vertices) == 2:
            return one / surface_value

        child_sum = zero
        for neighbour in current_graph.get_undirected_neighbours(vertex.nodes):
            child_graph = current_graph.contract_vertices(vertex.nodes, neighbour.nodes)
            if not child_graph.has_directed_cycle():
                child_sum = child_sum + recurse(child_graph)
        return (one / surface_value) * child_sum

    return recurse(graph)


def pure_cff_denominator_jet(case: TopologyCase, loop_spatial_momenta: Sequence[ThreeVector], powers: Sequence[int]):
    cache = get_case_cache(case.name)
    mass_squared_values, active_edges, orders = build_mass_squared_jets(case.masses, powers)
    shape = tuple(o + 1 for o in orders)
    virtual_energies = edge_on_shell_energies_generic(case.signatures, mass_squared_values, loop_spatial_momenta, case.external_momenta)
    ext_energies = {eid: mpf(val) for eid, val in BASE.external_edge_energies(cache.parsed_external_edges, case.external_momenta).items()}

    n_internal_edges = len(case.signatures)
    orientation_sum = TaylorJet.constant(0, shape) if shape else mp.mpf(0)
    for bitmask in range(1 << n_internal_edges):
        graph = cache.base_graph.clone()
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

    inverse_energy_prefactor = TaylorJet.constant(1, shape) if shape else mp.mpf(1)
    for edge_index in range(n_internal_edges):
        inverse_energy_prefactor = inverse_energy_prefactor * (1 / (2 * virtual_energies[edge_index]))
    overall_sign = -1 if (len(case.signatures[0][0]) - 1) % 2 else 1
    return overall_sign * inverse_energy_prefactor * orientation_sum


def _generic_linear_solve(inv_matrix: Tuple[Tuple[mp.mpf, ...], ...], rhs, shape: Tuple[int, ...]):
    zero = TaylorJet.constant(0, shape) if shape else mp.mpf(0)
    out = []
    for row in inv_matrix:
        value = zero
        for coeff, rhs_j in zip(row, rhs):
            value = value + coeff * rhs_j
        out.append(value)
    return out


def pure_ltd_integrand_jet(case: TopologyCase, loop_spatial_momenta: Sequence[ThreeVector], powers: Sequence[int]):
    cache = get_case_cache(case.name)
    mass_squared_values, active_edges, orders = build_mass_squared_jets(case.masses, powers)
    shape = tuple(o + 1 for o in orders)
    virtual_energies = edge_on_shell_energies_generic(case.signatures, mass_squared_values, loop_spatial_momenta, case.external_momenta)
    ext_energy_shifts = [mpf(BASE.edge_external_energy_shift(signature, case.external_momenta)) for signature in case.signatures]

    total = TaylorJet.constant(0, shape) if shape else mp.mpf(0)
    numerator_prefactor = mpf(case.numerator(BASE.loops_four_vectors_with_zero_energies(loop_spatial_momenta), case.external_momenta))
    n_loops = len(case.signatures[0][0])

    for residue, inv_matrix in zip(cache.residues, cache.inverse_matrices):
        basis = list(residue["basis"])
        rhs = [residue["sigmas"][i] * virtual_energies[edge_index] - ext_energy_shifts[edge_index] for i, edge_index in enumerate(basis)]
        solved_loop_energies = _generic_linear_solve(inv_matrix, rhs, shape)

        value = mpf(residue["sign"]) * numerator_prefactor
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


def hybrid_ltd_cff_integrand_jet(case: TopologyCase, loop_spatial_momenta: Sequence[ThreeVector], powers: Sequence[int]):
    return pure_ltd_integrand_jet(case, loop_spatial_momenta, powers)


# ---------------------------------------------------------------------------
# Raised-power test cases.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RaisedPowerCase:
    name: str
    topology: BASE.TopologyCase
    powers: Tuple[int, ...]
    n_random_samples: int
    description: str


def raised_cases() -> Tuple[RaisedPowerCase, ...]:
    base_cases = {case.name: case for case in BASE.default_cases()}

    def make(name: str, topology_name: str, powers: Sequence[int], samples: int, description: str) -> RaisedPowerCase:
        topology = base_cases[topology_name]
        if len(powers) != len(topology.signatures):
            raise ValueError(f"power list for {name} has wrong length")
        return RaisedPowerCase(
            name=name,
            topology=topology,
            powers=tuple(int(p) for p in powers),
            n_random_samples=int(samples),
            description=description,
        )

    return (
        # 1-loop box
        make("box__edge4_nu2", "box_1L_4prop", (1, 1, 1, 2), 3, "single dotted edge, nu_4=2"),
        make("box__edge4_nu3", "box_1L_4prop", (1, 1, 1, 3), 3, "single dotted edge, nu_4=3"),
        make("box__edge4_nu4", "box_1L_4prop", (1, 1, 1, 4), 3, "single dotted edge, nu_4=4"),
        make("box__edges1and4_nu2", "box_1L_4prop", (2, 1, 1, 2), 3, "two dotted edges, nu_1=nu_4=2"),
        make("box__edges1_3_4_mixed", "box_1L_4prop", (2, 1, 3, 2), 2, "three dotted edges, mixed powers (2,3,2)"),
        # 2-loop sunrise
        make("sunrise__edge3_nu2", "sunrise_2L_3prop", (1, 1, 2), 3, "single dotted edge, nu_3=2"),
        make("sunrise__edge3_nu3", "sunrise_2L_3prop", (1, 1, 3), 3, "single dotted edge, nu_3=3"),
        make("sunrise__edge3_nu4", "sunrise_2L_3prop", (1, 1, 4), 3, "single dotted edge, nu_3=4"),
        make("sunrise__edges1and3_nu2", "sunrise_2L_3prop", (2, 1, 2), 3, "two dotted edges, nu_1=nu_3=2"),
        make("sunrise__all_three_nu2", "sunrise_2L_3prop", (2, 2, 2), 3, "three dotted edges, all nu=2"),
        # 2-loop kite
        make("kite__edge5_nu2", "kite_2L_5prop", (1, 1, 1, 1, 2), 3, "single dotted edge, nu_5=2"),
        make("kite__edge5_nu3", "kite_2L_5prop", (1, 1, 1, 1, 3), 3, "single dotted edge, nu_5=3"),
        make("kite__edge5_nu4", "kite_2L_5prop", (1, 1, 1, 1, 4), 3, "single dotted edge, nu_5=4"),
        make("kite__edges1and5_nu2", "kite_2L_5prop", (2, 1, 1, 1, 2), 3, "two dotted edges, nu_1=nu_5=2"),
        make("kite__edges1_3_5_nu2", "kite_2L_5prop", (2, 1, 2, 1, 2), 3, "three dotted edges, nu_1=nu_3=nu_5=2"),
        # 3-loop Mercedes
        make("mercedes__edge6_nu2", "mercedes_3L_6prop_vacuum", (1, 1, 1, 1, 1, 2), 3, "single dotted edge, nu_6=2"),
        make("mercedes__edge6_nu3", "mercedes_3L_6prop_vacuum", (1, 1, 1, 1, 1, 3), 3, "single dotted edge, nu_6=3"),
        make("mercedes__edge6_nu4", "mercedes_3L_6prop_vacuum", (1, 1, 1, 1, 1, 4), 3, "single dotted edge, nu_6=4"),
        make("mercedes__edges1and4_nu2", "mercedes_3L_6prop_vacuum", (2, 1, 1, 2, 1, 1), 3, "two dotted edges, nu_1=nu_4=2"),
        make("mercedes__edges1_4_6_nu2", "mercedes_3L_6prop_vacuum", (2, 1, 1, 2, 1, 2), 3, "three dotted edges, nu_1=nu_4=nu_6=2"),
        make("mercedes__mixed_3_2_2", "mercedes_3L_6prop_vacuum", (3, 1, 1, 2, 1, 2), 3, "three dotted edges, mixed powers (3,2,2)"),
    )


# ---------------------------------------------------------------------------
# Evaluation and reporting.
# ---------------------------------------------------------------------------

def topology_loop_spatial_momenta(topology: BASE.TopologyCase, sample_index: int) -> Tuple[Tuple[float, float, float], ...]:
    rng = random.Random(topology.random_seed + 1009 * sample_index)
    return BASE.random_loop_spatial_momenta(rng, len(topology.loop_names))


def extract_target_value(jet_or_scalar, powers: Sequence[int]) -> mp.mpf:
    if isinstance(jet_or_scalar, TaylorJet):
        active_orders = tuple(nu - 1 for nu in powers if nu > 1)
        return jet_or_scalar.coefficient(active_orders)
    return mpf(jet_or_scalar)


def evaluate_raised_case(case: RaisedPowerCase, n_samples: Optional[int] = None) -> List[Dict[str, float]]:
    topology = case.topology
    n_samples = case.n_random_samples if n_samples is None else n_samples
    results: List[Dict[str, float]] = []
    for sample_index in range(n_samples):
        loop_spatial_momenta = topology_loop_spatial_momenta(topology, sample_index)
        numerator_prefactor = mpf(topology.numerator(BASE.loops_four_vectors_with_zero_energies(loop_spatial_momenta), topology.external_momenta))
        cff_val = extract_target_value(pure_cff_denominator_jet(topology, loop_spatial_momenta, case.powers), case.powers) * numerator_prefactor
        ltd_val = extract_target_value(pure_ltd_integrand_jet(topology, loop_spatial_momenta, case.powers), case.powers)
        hyb_val = extract_target_value(hybrid_ltd_cff_integrand_jet(topology, loop_spatial_momenta, case.powers), case.powers)
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


def format_case_report(case: RaisedPowerCase, results: Sequence[Dict[str, float]]) -> str:
    max_abs = max(result["abs_cff_minus_ltd"] for result in results)
    max_rel = max(result["rel_cff_minus_ltd"] for result in results)
    max_hybrid_abs = max(result["abs_ltd_minus_hybrid"] for result in results)
    lines = [
        f"\n[{case.name}]",
        f"  topology      : {case.topology.name}",
        f"  powers        : {list(case.powers)}",
        f"  description   : {case.description}",
        f"  samples       : {len(results)}",
        f"  max |CFF-LTD| : {max_abs:.6e}",
        f"  max rel diff  : {max_rel:.6e}",
        f"  max |LTD-HYB| : {max_hybrid_abs:.6e}",
    ]
    for result in results:
        lines.append(
            "    sample {sample:>2.0f}: "
            "CFF={pure_cff:+.16e}  LTD={pure_ltd:+.16e}  HYB={hybrid:+.16e}  "
            "|CFF-LTD|={abs_cff_minus_ltd:.3e}  |LTD-HYB|={abs_ltd_minus_hybrid:.3e}".format(**result)
        )
    return "\n".join(lines)


def run_tests(
    selected_case: Optional[str],
    selected_topology: Optional[str],
    n_samples: Optional[int],
    tolerance: float,
) -> Tuple[int, str]:
    cases = list(raised_cases())

    if selected_topology is not None:
        cases = [case for case in cases if case.topology.name == selected_topology]
    if selected_case is not None:
        cases = [case for case in cases if case.name == selected_case]

    if not cases:
        available = [case.name for case in raised_cases()]
        topologies = sorted({case.topology.name for case in raised_cases()})
        message_lines = ["No cases selected.", "", "Available cases:"]
        message_lines.extend([f"  - {name}" for name in available])
        message_lines.append("")
        message_lines.append("Available topologies:")
        message_lines.extend([f"  - {name}" for name in topologies])
        return 2, "\n".join(message_lines)

    reports: List[str] = []
    exit_code = 0
    for case in cases:
        results = evaluate_raised_case(case, n_samples=n_samples)
        reports.append(format_case_report(case, results))
        max_abs = max(result["abs_cff_minus_ltd"] for result in results)
        max_hybrid_abs = max(result["abs_ltd_minus_hybrid"] for result in results)
        if max_abs > tolerance or max_hybrid_abs > tolerance:
            exit_code = 1

    summary_lines = [
        "Raised-propagator cross-check summary",
        "=====================================",
        f"cases run      : {len(cases)}",
        f"tolerance      : {tolerance:.3e}",
        f"worst-case flag: {'FAIL' if exit_code else 'PASS'}",
    ]
    return exit_code, "\n".join(summary_lines + reports) + "\n"


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-tests", action="store_true", help="Run the built-in raised-propagator cross-checks")
    parser.add_argument("--case", type=str, default=None, help="Restrict to one raised-power case name")
    parser.add_argument("--topology", type=str, default=None, help="Restrict to one topology name")
    parser.add_argument("--samples", type=int, default=None, help="Override the number of random spatial points per case")
    parser.add_argument("--tolerance", type=float, default=1.0e-30, help="Absolute tolerance used for pass/fail")
    parser.add_argument("--list-cases", action="store_true", help="List all built-in raised-power cases")
    parser.add_argument("--write-log", type=str, default=None, help="Optional path to write the textual report")
    args = parser.parse_args(argv)

    if args.list_cases:
        print("Built-in raised-power cases:")
        for case in raised_cases():
            print(f"  - {case.name:30s} topology={case.topology.name:26s} powers={list(case.powers)}")
        return 0

    if args.run_tests:
        exit_code, report = run_tests(args.case, args.topology, args.samples, args.tolerance)
        print(report)
        if args.write_log is not None:
            pathlib.Path(args.write_log).write_text(report, encoding="utf-8")
        return exit_code

    parser.print_help()
    print("\nUse --list-cases to inspect the built-in dotted cases.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
