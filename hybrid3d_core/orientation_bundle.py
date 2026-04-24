#!/usr/bin/env python3
from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Dict, List, Optional, Sequence, Tuple
import itertools
import math
import mpmath as mp

from .graph_io import ParsedGraph, repeated_groups, _strip_quotes
from .ltd_cut_structure_generator import CutStructureGenerator, CLOSE_BELOW

Signature = Tuple[Tuple[int, ...], Tuple[int, ...]]
ThreeVector = Tuple[float, float, float]
FourVector = Tuple[float, float, float, float]
mp.mp.dps = 80

@dataclass(frozen=True)
class EdgeRef:
    edge_id: int
    edge_type: str

@dataclass
class CFFVertex:
    nodes: frozenset[int]
    incoming: List[EdgeRef]
    outgoing: List[EdgeRef]
    def vertex_type(self) -> str:
        is_sink = all(edge.edge_type != 'virtual' for edge in self.outgoing)
        is_source = all(edge.edge_type != 'virtual' for edge in self.incoming)
        if is_sink: return 'sink'
        if is_source: return 'source'
        return 'none'

class CFFGenerationGraphPy:
    def __init__(self, vertices: Sequence[CFFVertex]):
        self.vertices = [CFFVertex(v.nodes, list(v.incoming), list(v.outgoing)) for v in vertices]
        self.vertices.sort(key=lambda v: tuple(sorted(v.nodes)))
    def clone(self) -> 'CFFGenerationGraphPy': return CFFGenerationGraphPy(self.vertices)
    def get_vertex(self, nodes: frozenset[int]) -> CFFVertex:
        for vertex in self.vertices:
            if vertex.nodes == nodes: return vertex
        raise KeyError(nodes)
    def _directed_virtual_adjacency(self) -> Dict[frozenset[int], List[frozenset[int]]]:
        adjacency = {vertex.nodes: [] for vertex in self.vertices}
        for vertex in self.vertices:
            for edge in vertex.outgoing:
                if edge.edge_type != 'virtual': continue
                for other in self.vertices:
                    if other.nodes != vertex.nodes and edge in other.incoming:
                        adjacency[vertex.nodes].append(other.nodes); break
        return adjacency
    def has_directed_cycle(self) -> bool:
        adjacency = self._directed_virtual_adjacency(); visited=set(); stack=set()
        def dfs(node):
            if node in stack: return True
            if node in visited: return False
            visited.add(node); stack.add(node)
            for nxt in adjacency[node]:
                if dfs(nxt): return True
            stack.remove(node); return False
        return any(dfs(node) for node in list(adjacency))
    def _are_directed_adjacent(self, left, right) -> bool:
        left_vertex = self.get_vertex(left); right_vertex = self.get_vertex(right)
        return any(edge.edge_type == 'virtual' and edge in right_vertex.incoming for edge in left_vertex.outgoing)
    def are_adjacent(self, left, right) -> bool:
        return self._are_directed_adjacent(left, right) or self._are_directed_adjacent(right, left)
    def get_undirected_neighbours(self, nodes) -> List[CFFVertex]:
        return [vertex for vertex in self.vertices if vertex.nodes != nodes and self.are_adjacent(nodes, vertex.nodes)]
    def has_connected_complement(self, removed_nodes) -> bool:
        others = [vertex.nodes for vertex in self.vertices if vertex.nodes != removed_nodes]
        if not others: return True
        start=others[0]; visited={start}; frontier=[start]
        while frontier:
            cur=frontier.pop()
            for nb in self.get_undirected_neighbours(cur):
                if nb.nodes == removed_nodes: continue
                if nb.nodes not in visited:
                    visited.add(nb.nodes); frontier.append(nb.nodes)
        return len(visited) == len(others)
    def get_source_sink_greedy(self) -> Optional[CFFVertex]:
        for vertex in self.vertices:
            if vertex.vertex_type() in {'source','sink'} and self.has_connected_complement(vertex.nodes):
                return vertex
        return None
    def contract_vertices(self, left_nodes, right_nodes) -> 'CFFGenerationGraphPy':
        left = self.get_vertex(left_nodes); right = self.get_vertex(right_nodes)
        right_outgoing=set(right.outgoing); right_incoming=set(right.incoming)
        left_outgoing=set(left.outgoing); left_incoming=set(left.incoming)
        new_in = [e for e in left.incoming if e not in right_outgoing] + [e for e in right.incoming if e not in left_outgoing]
        new_out = [e for e in left.outgoing if e not in right_incoming] + [e for e in right.outgoing if e not in left_incoming]
        new_vertex=CFFVertex(left.nodes | right.nodes, sorted(new_in, key=lambda e:e.edge_id), sorted(new_out, key=lambda e:e.edge_id))
        new_vertices=[CFFVertex(v.nodes, list(v.incoming), list(v.outgoing)) for v in self.vertices if v.nodes not in {left_nodes, right_nodes}]
        new_vertices.append(new_vertex)
        return CFFGenerationGraphPy(new_vertices)

def mpf(x) -> mp.mpf:
    if isinstance(x, mp.mpf): return x
    if isinstance(x, float): return mp.mpf(repr(x))
    return mp.mpf(str(x))

def orientation_id_from_signs(signs: Sequence[int]) -> str:
    return ''.join('+' if int(s)>0 else '-' if int(s)<0 else '0' for s in signs)

def _frac_to_str(x: Fraction) -> str:
    return str(x.numerator) if x.denominator == 1 else f'{x.numerator}/{x.denominator}'

@dataclass(frozen=True)
class LinearEnergyExpr:
    internal_terms: Tuple[Tuple[int, int], ...]
    external_terms: Tuple[Tuple[int, int], ...]
    const: str = '0'
    @staticmethod
    def zero() -> 'LinearEnergyExpr': return LinearEnergyExpr(tuple(), tuple(), '0')
    @staticmethod
    def E(edge_id: int, coeff: int = 1) -> 'LinearEnergyExpr': return LinearEnergyExpr(((int(edge_id), int(coeff)),), tuple(), '0') if coeff else LinearEnergyExpr.zero()
    @staticmethod
    def X(ext_id: int, coeff: int = 1) -> 'LinearEnergyExpr': return LinearEnergyExpr(tuple(), ((int(ext_id), int(coeff)),), '0') if coeff else LinearEnergyExpr.zero()
    def _const_frac(self) -> Fraction: return Fraction(str(self.const))
    def canonical(self) -> 'LinearEnergyExpr':
        im: Dict[int, int] = {}; xm: Dict[int, int] = {}
        for i,c in self.internal_terms: im[int(i)] = im.get(int(i),0) + int(c)
        for i,c in self.external_terms: xm[int(i)] = xm.get(int(i),0) + int(c)
        return LinearEnergyExpr(tuple(sorted((i,c) for i,c in im.items() if c)), tuple(sorted((i,c) for i,c in xm.items() if c)), _frac_to_str(self._const_frac()))
    def __add__(self, other: 'LinearEnergyExpr') -> 'LinearEnergyExpr': return LinearEnergyExpr(self.internal_terms+other.internal_terms, self.external_terms+other.external_terms, _frac_to_str(self._const_frac()+other._const_frac())).canonical()
    def __neg__(self) -> 'LinearEnergyExpr': return LinearEnergyExpr(tuple((i,-c) for i,c in self.internal_terms), tuple((i,-c) for i,c in self.external_terms), _frac_to_str(-self._const_frac())).canonical()
    def __sub__(self, other: 'LinearEnergyExpr') -> 'LinearEnergyExpr': return self + (-other)
    def mul(self, n: Fraction|int) -> 'LinearEnergyExpr':
        f=Fraction(n); it=[]; xt=[]
        for i,c in self.internal_terms:
            v=Fraction(c)*f
            if v: it.append((i, int(v) if v.denominator==1 else int(v)))
        for i,c in self.external_terms:
            v=Fraction(c)*f
            if v: xt.append((i, int(v) if v.denominator==1 else int(v)))
        return LinearEnergyExpr(tuple(it), tuple(xt), _frac_to_str(self._const_frac()*f)).canonical()
    def evaluate(self, E_vals: Dict[int, mp.mpf], OSE_vals: Dict[int, mp.mpf]) -> mp.mpf:
        total=mp.mpf(self.const)
        for edge_id, coeff in self.internal_terms: total += mp.mpf(coeff)*E_vals[int(edge_id)]
        for ext_id, coeff in self.external_terms: total += mp.mpf(coeff)*OSE_vals[int(ext_id)]
        return total
    def render(self) -> str:
        parts=[]
        if self.const not in {'0','0.0'}: parts.append((self.const, False))
        def term(label,c):
            c=int(c); mag=abs(c)
            return (label if mag==1 else f'{mag}*{label}', c<0)
        for i,c in self.internal_terms: parts.append(term(f'OSE[{i}]',c))
        for i,c in self.external_terms: parts.append(term(f'E[{i}]',c))
        if not parts: return '0'
        out=('-' if parts[0][1] else '')+parts[0][0]
        for txt,neg in parts[1:]: out += (' - ' if neg else ' + ') + txt
        return out

@dataclass(frozen=True)
class SurfaceDef:
    surface_id: int; kind: str; expr: LinearEnergyExpr; label: str
@dataclass(frozen=True)
class OrientationTerm:
    orientation_id: str; family: str; branch_id: int; edge_orientations: Tuple[int,...]; prefactor_sign: Any; prefactor_half_edges: Tuple[int,...]; surface_chain: Tuple[int,...]; loop_energy_exprs: Tuple[LinearEnergyExpr,...]; edge_energy_exprs: Tuple[LinearEnergyExpr,...]; meta: Dict[str,Any]
@dataclass(frozen=True)
class ExpressionBundle:
    family: str; loop_names: Tuple[str,...]; ext_names: Tuple[str,...]; signatures: Tuple[Signature,...]; surface_cache: Tuple[SurfaceDef,...]; terms: Tuple[OrientationTerm,...]

@dataclass(frozen=True)
class _LogicalChannel:
    rep_edge: int
    members: Tuple[int, ...]
    power: int

@dataclass(frozen=True)
class _DenomFactor:
    kind: str
    ref_id: int
    power: int
    derivs: Tuple[Fraction, ...]

def classify_surface_kind(expr: LinearEnergyExpr) -> str:
    coeffs=[int(c) for _,c in expr.internal_terms if int(c)!=0]
    if len(coeffs) <= 1: return 'e'
    return 'e' if all(c>0 for c in coeffs) or all(c<0 for c in coeffs) else 'h'
class SurfaceCacheBuilder:
    def __init__(self): self._index={}; self._surfaces=[]
    def intern(self, kind: str, expr: LinearEnergyExpr, label: str) -> int:
        expr=expr.canonical(); kind=classify_surface_kind(expr) if kind=='auto' else kind; key=(kind,expr)
        if key in self._index: return self._index[key]
        idx=len(self._surfaces); self._index[key]=idx; self._surfaces.append(SurfaceDef(idx,kind,expr,label)); return idx
    def build(self): return tuple(self._surfaces)

def _rank(rows: Sequence[Sequence[int]]) -> int:
    if not rows: return 0
    A=[[Fraction(x) for x in r] for r in rows]; m=len(A); n=len(A[0]); r=0
    for c in range(n):
        piv=None
        for i in range(r,m):
            if A[i][c]!=0: piv=i; break
        if piv is None: continue
        A[r],A[piv]=A[piv],A[r]; pv=A[r][c]; A[r]=[x/pv for x in A[r]]
        for i in range(m):
            if i!=r and A[i][c]!=0:
                fac=A[i][c]; A[i]=[a-fac*b for a,b in zip(A[i],A[r])]
        r+=1
        if r==m: break
    return r

def _det_nonzero(rows): return _rank(rows)==len(rows)

def _solve_expr_system(A_int, b):
    n=len(A_int); A=[[Fraction(x) for x in row] for row in A_int]; rhs=list(b)
    for col in range(n):
        piv=None
        for r in range(col,n):
            if A[r][col]!=0: piv=r; break
        if piv is None: raise ValueError('singular basis')
        if piv != col: A[col],A[piv]=A[piv],A[col]; rhs[col],rhs[piv]=rhs[piv],rhs[col]
        pv=A[col][col]; A[col]=[x/pv for x in A[col]]; rhs[col]=rhs[col].mul(Fraction(1,1)/pv)
        for r in range(n):
            if r==col: continue
            fac=A[r][col]
            if fac!=0:
                A[r]=[a-fac*bv for a,bv in zip(A[r],A[col])]
                rhs[r]=rhs[r]-rhs[col].mul(fac)
    return [x.canonical() for x in rhs]

def _solve_fraction_system(A_int, b):
    n=len(A_int); A=[[Fraction(x) for x in row] for row in A_int]; rhs=[Fraction(x) for x in b]
    for col in range(n):
        piv=None
        for r in range(col,n):
            if A[r][col]!=0: piv=r; break
        if piv is None: raise ValueError('singular basis')
        if piv != col:
            A[col],A[piv]=A[piv],A[col]; rhs[col],rhs[piv]=rhs[piv],rhs[col]
        pv=A[col][col]; A[col]=[x/pv for x in A[col]]; rhs[col]/=pv
        for r in range(n):
            if r==col: continue
            fac=A[r][col]
            if fac:
                A[r]=[a-fac*bv for a,bv in zip(A[r],A[col])]
                rhs[r]-=fac*rhs[col]
    return rhs

def choose_basis_indices(signatures):
    n_loops=len(signatures[0][0])
    for basis in itertools.combinations(range(len(signatures)), n_loops):
        if _det_nonzero([[signatures[e][0][i] for i in range(n_loops)] for e in basis]): return tuple(basis)
    raise ValueError('Could not find a nonsingular basis')

def solve_loop_energy_from_target_edge_exprs(signatures,basis,target_edge_exprs,n_external):
    A=[]; rhs=[]
    for edge_index in basis:
        loop_coeffs, ext_coeffs=signatures[edge_index]
        A.append([int(x) for x in loop_coeffs]); r=target_edge_exprs[edge_index]
        for a,c in enumerate(ext_coeffs):
            if c: r=r-LinearEnergyExpr.X(a,int(c))
        rhs.append(r)
    return tuple(_solve_expr_system(A,rhs))

def edge_q0_from_loop_exprs(signatures, loop_exprs, n_external):
    out=[]
    for loop_coeffs, ext_coeffs in signatures:
        q=LinearEnergyExpr.zero()
        for i,c in enumerate(loop_coeffs):
            if c: q=q+loop_exprs[i].mul(int(c))
        for a,c in enumerate(ext_coeffs):
            if c: q=q+LinearEnergyExpr.X(a,int(c))
        out.append(q.canonical())
    return tuple(out)

def solve_loop_energy_substitutions(signatures,basis,cut_signs,n_external):
    targets=[LinearEnergyExpr.zero() for _ in signatures]
    for e,s in zip(basis,cut_signs): targets[e]=LinearEnergyExpr.E(e,int(s))
    loops=solve_loop_energy_from_target_edge_exprs(signatures,basis,targets,n_external)
    return loops, edge_q0_from_loop_exprs(signatures,loops,n_external)

def signed_on_shell_edge_exprs(signs,n_internal,n_external): return tuple(LinearEnergyExpr.E(i,int(s)) for i,s in enumerate(signs))

def _duplicate_signature_excess(signatures):
    counts={}
    for sig in signatures:
        loop,ext=sig
        neg=(tuple(-x for x in loop), tuple(-x for x in ext))
        key=(loop,ext) if (loop,ext) <= neg else neg
        counts[key]=counts.get(key,0)+1
    return sum(v-1 for v in counts.values() if v>1)

def _add_const(expr: LinearEnergyExpr, value: Fraction) -> LinearEnergyExpr:
    if not value:
        return expr
    return LinearEnergyExpr(expr.internal_terms, expr.external_terms, _frac_to_str(expr._const_frac()+value)).canonical()

def _multi_sub(a: Sequence[int], b: Sequence[int]) -> Tuple[int, ...]:
    return tuple(int(x)-int(y) for x,y in zip(a,b))

def _multi_factorial(a: Sequence[int]) -> int:
    out=1
    for x in a:
        out *= math.factorial(int(x))
    return out

def _iter_multiindices_leq(bounds: Sequence[int]):
    bounds=tuple(int(x) for x in bounds)
    if not bounds:
        yield tuple()
        return
    ranges=[range(x+1) for x in bounds]
    for item in itertools.product(*ranges):
        yield tuple(int(x) for x in item)

def _rising(n: int, k: int) -> int:
    out=1
    for i in range(int(k)):
        out *= int(n)+i
    return out

def _logical_channels(parsed: ParsedGraph) -> Tuple[_LogicalChannel, ...]:
    rep_groups=repeated_groups(parsed)
    group_by_member={}
    for group in rep_groups:
        for edge_id in group.edge_ids:
            group_by_member[int(edge_id)] = group
    channels=[]
    seen_groups=set()
    for edge in parsed.internal_edges:
        group=group_by_member.get(edge.edge_id)
        if group is None:
            channels.append(_LogicalChannel(edge.edge_id,(edge.edge_id,),1))
            continue
        gid=tuple(group.edge_ids)
        if gid in seen_groups:
            continue
        seen_groups.add(gid)
        channels.append(_LogicalChannel(group.edge_ids[0],tuple(int(x) for x in group.edge_ids),len(group.edge_ids)))
    return tuple(channels)

def _basis_derivative_matrices(signatures: Sequence[Signature], basis: Sequence[int]):
    n_loops=len(signatures[0][0]); n_basis=len(basis)
    A=[[Fraction(signatures[e][0][i]) for i in range(n_loops)] for e in basis]
    loop_by_basis=[[Fraction(0) for _ in range(n_basis)] for _ in range(n_loops)]
    for bpos in range(n_basis):
        rhs=[Fraction(1 if i==bpos else 0) for i in range(n_basis)]
        sol=_solve_fraction_system(A,rhs)
        for loop_id, val in enumerate(sol):
            loop_by_basis[loop_id][bpos]=val
    edge_by_basis=[]
    for loop_coeffs,_ in signatures:
        row=[]
        for bpos in range(n_basis):
            row.append(sum(Fraction(loop_coeffs[i])*loop_by_basis[i][bpos] for i in range(n_loops)))
        edge_by_basis.append(tuple(row))
    return tuple(tuple(row) for row in loop_by_basis), tuple(edge_by_basis)

def _factor_derivative_piece(factor: _DenomFactor, delta: Sequence[int]):
    order=sum(int(x) for x in delta)
    if order == 0:
        coeff=Fraction(1)
    else:
        coeff=Fraction(((-1) ** order) * _rising(factor.power, order))
        for d, a in zip(delta, factor.derivs):
            if d and not a:
                return None
            if d:
                coeff *= a ** int(d)
    power=factor.power+order
    if factor.kind == 'surface':
        return coeff, tuple(), tuple([factor.ref_id] * power)
    if factor.kind == 'cut':
        return coeff, tuple([factor.ref_id] * power), tuple()
    raise ValueError(f'Unknown denominator factor kind {factor.kind!r}')

def _denominator_derivative_terms(factors: Sequence[_DenomFactor], gamma: Sequence[int]):
    gamma=tuple(int(x) for x in gamma)
    nvar=len(gamma)
    if not factors:
        return [(Fraction(1), tuple(), tuple())] if not any(gamma) else []
    total_factor=Fraction(_multi_factorial(gamma))
    accum: Dict[Tuple[Tuple[int,...], Tuple[int,...]], Fraction] = {}
    def rec(idx: int, remaining: Tuple[int,...], coeff: Fraction, half_edges: Tuple[int,...], chain: Tuple[int,...], denom_factorial: int):
        if idx == len(factors):
            if any(remaining):
                return
            final_coeff=coeff * total_factor / denom_factorial
            if final_coeff:
                key=(tuple(sorted(half_edges)), chain)
                accum[key]=accum.get(key,Fraction(0))+final_coeff
            return
        for delta in _iter_multiindices_leq(remaining):
            piece=_factor_derivative_piece(factors[idx], delta)
            if piece is None:
                continue
            pcoeff, phalf, pchain=piece
            next_remaining=tuple(remaining[i]-delta[i] for i in range(nvar))
            rec(
                idx+1,
                next_remaining,
                coeff*pcoeff,
                half_edges+phalf,
                chain+pchain,
                denom_factorial*_multi_factorial(delta),
            )
    rec(0,gamma,Fraction(1),tuple(),tuple(),1)
    return [(coeff, half, chain) for (half,chain), coeff in accum.items() if coeff]

def _shift_exprs_by_offsets(base_exprs: Sequence[LinearEnergyExpr], derivs: Sequence[Sequence[Fraction]], offsets: Sequence[Fraction]) -> Tuple[LinearEnergyExpr, ...]:
    out=[]
    for expr, row in zip(base_exprs, derivs):
        shift=sum(Fraction(a)*Fraction(o) for a,o in zip(row, offsets))
        out.append(_add_const(expr, shift))
    return tuple(out)

def _numerator_derivative_samples(beta: Sequence[int], base_loop_exprs: Sequence[LinearEnergyExpr], base_edge_exprs: Sequence[LinearEnergyExpr], loop_derivs: Sequence[Sequence[Fraction]], edge_derivs: Sequence[Sequence[Fraction]]):
    beta=tuple(int(x) for x in beta)
    nvar=len(beta)
    if not any(beta):
        return [(Fraction(1), tuple(Fraction(0) for _ in range(nvar)), tuple(base_loop_exprs), tuple(base_edge_exprs), 'n0')]
    # The supported numerator class is affine in the EMR energy variables.  Exact
    # first derivatives are represented by a finite central interpolation stencil.
    # Higher numerator derivatives vanish in this class.
    if sum(beta) != 1 or max(beta) != 1:
        return []
    bpos=beta.index(1)
    out=[]
    for offset, coeff in ((Fraction(1), Fraction(1,2)), (Fraction(-1), Fraction(-1,2))):
        offsets=[Fraction(0) for _ in range(nvar)]
        offsets[bpos]=offset
        loops=_shift_exprs_by_offsets(base_loop_exprs, loop_derivs, offsets)
        edges=_shift_exprs_by_offsets(base_edge_exprs, edge_derivs, offsets)
        out.append((coeff, tuple(offsets), loops, edges, f'nD{bpos}{"p" if offset > 0 else "m"}'))
    return out

def _node_to_internal_id(name: str, parsed: ParsedGraph) -> Optional[int]: return parsed.node_name_to_internal.get(_strip_quotes(name).split(':',1)[0])
def build_base_graph_from_parsed(parsed: ParsedGraph) -> CFFGenerationGraphPy:
    n_vertices=max(parsed.node_name_to_internal.values())+1 if parsed.node_name_to_internal else 0
    vertices=[CFFVertex(frozenset([i]), [], []) for i in range(n_vertices)]; lookup={v.nodes:v for v in vertices}
    def get_v(i): return lookup[frozenset([i])]
    for edge in parsed.internal_edges:
        ref=EdgeRef(edge.edge_id,'virtual'); get_v(edge.tail).outgoing.append(ref); get_v(edge.head).incoming.append(ref)
    for edge in parsed.external_edges:
        ref=EdgeRef(edge.edge_id,'external'); s=_node_to_internal_id(edge.source,parsed); d=_node_to_internal_id(edge.destination,parsed)
        if s is not None: get_v(s).outgoing.append(ref)
        if d is not None: get_v(d).incoming.append(ref)
    return CFFGenerationGraphPy(vertices)

def _external_energy_sign(vertex, edge, vertex_type):
    if edge in vertex.incoming: return 1 if vertex_type=='sink' else -1
    return -1 if vertex_type=='sink' else 1

def _boundary_external_shift_from_internal_labels(parsed, node_set, vertex_type):
    """Return the external-energy part of a CFF surface.

    The DOT graph may contain dashed external half-edges solely as bookkeeping
    devices.  Counting those half-edges directly can double count momentum
    shifts.  The unambiguous source of the external shift is the internal
    momentum routing: for a contracted vertex set S we form the external part
    of the physical momentum flowing out of S through internal boundary edges.
    The CFF source/sink convention then contributes -B_S^0 for a source and
    +B_S^0 for a sink.
    """
    acc = {}
    S = set(int(x) for x in node_set)
    for edge in parsed.internal_edges:
        if edge.tail in S and edge.head not in S:
            sign = 1
        elif edge.head in S and edge.tail not in S:
            sign = -1
        else:
            continue
        for a, c in enumerate(edge.signature[1]):
            if c:
                acc[a] = acc.get(a, 0) + sign * int(c)
    if vertex_type == 'source':
        acc = {a: -c for a, c in acc.items()}
    return {a: c for a, c in acc.items() if c}

def _enumerate_cff_branches(graph,surface_builder,parsed,branch_acc):
    # Deterministic source/sink choice, exhaustive contraction sum.  The CFF
    # recursion chooses one admissible source/sink vertex and then sums over all
    # acyclic contractions of its virtual boundary edges.  Earlier revisions
    # returned after the first valid contraction, which collapsed every
    # orientation to a single product chain and lost the non-simplicial cone
    # triangulation.
    if len(graph.vertices)<2: branch_acc.append(tuple()); return
    vertex=graph.get_source_sink_greedy()
    if vertex is None: raise RuntimeError('No valid source/sink found during CFF tree generation')
    vt=vertex.vertex_type(); internal_terms=[]
    for edge in list(vertex.incoming)+list(vertex.outgoing):
        if edge.edge_type=='virtual': internal_terms.append((edge.edge_id,1))
    ext_acc=_boundary_external_shift_from_internal_labels(parsed, vertex.nodes, vt)
    sid=surface_builder.intern('auto', LinearEnergyExpr(tuple(sorted(internal_terms)), tuple(sorted((a,c) for a,c in ext_acc.items() if c)), '0'), f'{vt}@{sorted(vertex.nodes)}')
    if len(graph.vertices)==2: branch_acc.append((sid,)); return
    emitted=False
    for nb in graph.get_undirected_neighbours(vertex.nodes):
        child=graph.contract_vertices(vertex.nodes, nb.nodes)
        if child.has_directed_cycle(): continue
        sub=[]; _enumerate_cff_branches(child,surface_builder,parsed,sub)
        for chain in sub:
            branch_acc.append((sid,)+chain)
            emitted=True
    if not emitted: branch_acc.append((sid,))

def build_pure_cff_bundle(parsed):
    n_internal=len(parsed.internal_edges); signatures=tuple(e.signature for e in parsed.internal_edges); n_external=len(parsed.ext_names); basis=choose_basis_indices(signatures); sb=SurfaceCacheBuilder(); base=build_base_graph_from_parsed(parsed); terms=[]; bc=0
    overall_sign=-1 if (len(signatures[0][0])-1+_duplicate_signature_excess(signatures)) % 2 else 1
    for bitmask in range(1<<n_internal):
        graph=base.clone(); signs=[1]*n_internal
        for edge_index in range(n_internal):
            if bitmask & (1<<edge_index):
                signs[edge_index]=-1
                for vertex in graph.vertices:
                    out=[e for e in vertex.outgoing if e.edge_id==edge_index and e.edge_type=='virtual']; inc=[e for e in vertex.incoming if e.edge_id==edge_index and e.edge_type=='virtual']
                    if out:
                        e=out[0]; vertex.outgoing=[x for x in vertex.outgoing if not (x.edge_id==edge_index and x.edge_type=='virtual')]; vertex.incoming.append(e)
                    elif inc:
                        e=inc[0]; vertex.incoming=[x for x in vertex.incoming if not (x.edge_id==edge_index and x.edge_type=='virtual')]; vertex.outgoing.append(e)
        if graph.has_directed_cycle(): continue
        edge_exprs=signed_on_shell_edge_exprs(signs,n_internal,n_external); loop_exprs=solve_loop_energy_from_target_edge_exprs(signatures,basis,edge_exprs,n_external); chains=[]; _enumerate_cff_branches(graph,sb,parsed,chains)
        for chain in chains:
            terms.append(OrientationTerm(orientation_id_from_signs(signs),'pure_cff',bc,tuple(signs),overall_sign,tuple(range(n_internal)),chain,loop_exprs,edge_exprs,{'basis':list(basis),'source':'acyclic_orientation'})); bc+=1
    return ExpressionBundle('pure_cff',tuple(),tuple(),signatures,sb.build(),tuple(terms))

def build_pure_ltd_bundle(signatures,n_external_symbols=None):
    n_internal=len(signatures); n_loops=len(signatures[0][0]); n_external_symbols=n_external_symbols if n_external_symbols is not None else len(signatures[0][1]); sb=SurfaceCacheBuilder(); terms=[]; bc=0
    residues=CutStructureGenerator([sig[0] for sig in signatures]).get_residues([CLOSE_BELOW]*n_loops, simplify=True)
    for residue in residues:
        basis=tuple(int(e) for e in residue['basis'])
        cut_signs=[int(s) for s in residue['sigmas']]
        loop_exprs,edge_exprs=solve_loop_energy_substitutions(signatures,basis,cut_signs,n_external_symbols); chain=[]
        for edge_index,expr in enumerate(edge_exprs):
            if edge_index in basis: continue
            chain.append(sb.intern('auto', expr-LinearEnergyExpr.E(edge_index,1), f'dual-{edge_index}-minus'))
            chain.append(sb.intern('auto', expr+LinearEnergyExpr.E(edge_index,1), f'dual-{edge_index}-plus'))
        edge_orient=[0]*n_internal
        for e,s in zip(basis,cut_signs): edge_orient[e]=int(s)
        pref=1 if float(residue['sign']) > 0 else -1
        terms.append(OrientationTerm(orientation_id_from_signs(edge_orient),'pure_ltd',bc,tuple(edge_orient),pref,tuple(basis),tuple(chain),loop_exprs,edge_exprs,{'basis':list(basis),'cut_signs':cut_signs,'source':'canonical_ltd_residue'})); bc+=1
    return ExpressionBundle('pure_ltd',tuple(),tuple(),tuple(signatures),sb.build(),tuple(terms))

def _build_confluent_hybrid_bundle(parsed: ParsedGraph):
    signatures=tuple(e.signature for e in parsed.internal_edges)
    n_internal=len(signatures)
    n_external=len(parsed.ext_names)
    n_loops=len(signatures[0][0])
    channels=_logical_channels(parsed)
    collapsed_signatures=tuple(signatures[ch.rep_edge] for ch in channels)
    residues=CutStructureGenerator([sig[0] for sig in collapsed_signatures]).get_residues([CLOSE_BELOW]*n_loops, simplify=True)
    repeated=[ch for ch in channels if ch.power > 1]
    sb=SurfaceCacheBuilder()
    terms=[]
    branch=0
    for residue in residues:
        basis_logical=tuple(int(e) for e in residue['basis'])
        cut_signs=tuple(int(s) for s in residue['sigmas'])
        basis_orig=tuple(channels[i].rep_edge for i in basis_logical)
        powers=tuple(channels[i].power for i in basis_logical)
        alpha=tuple(power-1 for power in powers)

        targets=[LinearEnergyExpr.zero() for _ in signatures]
        for edge_id, sigma in zip(basis_orig, cut_signs):
            targets[edge_id]=LinearEnergyExpr.E(edge_id, sigma)
        loop_exprs=solve_loop_energy_from_target_edge_exprs(signatures,basis_orig,targets,n_external)
        edge_exprs=edge_q0_from_loop_exprs(signatures,loop_exprs,n_external)
        loop_derivs, edge_derivs=_basis_derivative_matrices(signatures,basis_orig)

        edge_orient=[0]*n_internal
        for logical_idx, sigma in zip(basis_logical, cut_signs):
            for edge_id in channels[logical_idx].members:
                edge_orient[edge_id]=sigma

        factors: List[_DenomFactor]=[]
        for bpos,(logical_idx,sigma,power) in enumerate(zip(basis_logical,cut_signs,powers)):
            derivs=[Fraction(0) for _ in basis_logical]
            derivs[bpos]=Fraction(sigma)
            factors.append(_DenomFactor('cut', channels[logical_idx].rep_edge, int(power), tuple(derivs)))

        basis_set=set(basis_logical)
        denominator_surface_ids=[]
        for logical_idx, channel in enumerate(channels):
            if logical_idx in basis_set:
                continue
            rep=channel.rep_edge
            q=edge_exprs[rep]
            minus=sb.intern('auto', q-LinearEnergyExpr.E(rep,1), f'hybrid-dual-{rep}-minus')
            plus=sb.intern('auto', q+LinearEnergyExpr.E(rep,1), f'hybrid-dual-{rep}-plus')
            denominator_surface_ids.extend([minus, plus])
            factors.append(_DenomFactor('surface', minus, channel.power, tuple(edge_derivs[rep])))
            factors.append(_DenomFactor('surface', plus, channel.power, tuple(edge_derivs[rep])))

        residue_sign=Fraction(1 if float(residue['sign']) > 0 else -1)
        for sigma, order in zip(cut_signs, alpha):
            if order % 2:
                residue_sign *= int(sigma)
        residue_norm=Fraction(1, _multi_factorial(alpha))
        beta_candidates=[beta for beta in _iter_multiindices_leq(alpha) if sum(beta) <= 1]
        active_basis=[basis_orig[i] for i,a in enumerate(alpha) if a]
        active_edges=[
            edge_id for edge_id,row in enumerate(edge_derivs)
            if any(a and row[i] for i,a in enumerate(alpha))
        ]
        for beta in beta_candidates:
            gamma=_multi_sub(alpha,beta)
            leibniz=Fraction(1)
            for a,b in zip(alpha,beta):
                leibniz *= Fraction(math.comb(int(a), int(b)))
            num_samples=_numerator_derivative_samples(beta, loop_exprs, edge_exprs, loop_derivs, edge_derivs)
            if not num_samples:
                continue
            den_terms=_denominator_derivative_terms(factors,gamma)
            if not den_terms:
                continue
            for ncoeff, offsets, sample_loop_exprs, sample_edge_exprs, sample_label in num_samples:
                offset_label=','.join(_frac_to_str(x) for x in offsets)
                for dcoeff, half_edges, chain in den_terms:
                    coeff=residue_sign*residue_norm*leibniz*ncoeff*dcoeff
                    if not coeff:
                        continue
                    beta_label=''.join(str(x) for x in beta) or '0'
                    gamma_label=''.join(str(x) for x in gamma) or '0'
                    label=f"{orientation_id_from_signs(edge_orient)}|B{','.join(str(x) for x in basis_orig)}|b{beta_label}|g{gamma_label}|{sample_label}"
                    terms.append(OrientationTerm(
                        label,
                        'hybrid',
                        branch,
                        tuple(edge_orient),
                        _frac_to_str(coeff),
                        tuple(int(x) for x in half_edges),
                        tuple(int(x) for x in chain),
                        tuple(sample_loop_exprs),
                        tuple(sample_edge_exprs),
                        {
                            'source':'hybrid_confluent_ltd_interpolation',
                            'basis_logical':list(basis_logical),
                            'basis':list(basis_orig),
                            'cut_signs':list(cut_signs),
                            'basis_powers':list(powers),
                            'alpha':list(alpha),
                            'beta':list(beta),
                            'gamma':list(gamma),
                            'finite_sample_offsets':offset_label,
                            'active_basis_edges':list(active_basis),
                            'active_edges':list(active_edges),
                            'denominator_surface_ids':list(denominator_surface_ids),
                            'repeated_groups':[list(ch.members) for ch in repeated],
                        },
                    ))
                    branch += 1
    return ExpressionBundle('hybrid',tuple(),tuple(),signatures,sb.build(),tuple(terms))

def build_hybrid_bundle_raw(parsed):
    rep_groups=repeated_groups(parsed); signatures=tuple(e.signature for e in parsed.internal_edges)
    if not rep_groups:
        ltd=build_pure_ltd_bundle(signatures,len(parsed.ext_names))
        return ExpressionBundle('hybrid',ltd.loop_names,ltd.ext_names,ltd.signatures,ltd.surface_cache,ltd.terms)
    return _build_confluent_hybrid_bundle(parsed)

def edge_spatial_momentum(signature, loop_spatial_momenta, external_momenta):
    loop_coeffs, ext_coeffs=signature; x=y=z=mp.mpf(0)
    for c,v in zip(loop_coeffs,loop_spatial_momenta):
        x+=mpf(c)*mpf(v[0]); y+=mpf(c)*mpf(v[1]); z+=mpf(c)*mpf(v[2])
    for c,p in zip(ext_coeffs,external_momenta):
        x+=mpf(c)*mpf(p[1]); y+=mpf(c)*mpf(p[2]); z+=mpf(c)*mpf(p[3])
    return (x,y,z)
def compute_internal_E_values(signatures,masses,loop_spatial_momenta,external_momenta):
    out={}
    for i,(sig,m) in enumerate(zip(signatures,masses)):
        q=edge_spatial_momentum(sig,loop_spatial_momenta,external_momenta); out[i]=mp.sqrt(mpf(q[0])**2+mpf(q[1])**2+mpf(q[2])**2+mpf(m)**2)
    return out
def compute_external_half_edge_energies(parsed, external_momenta): return {i: mpf(external_momenta[i][0]) for i in range(len(external_momenta))}
def evaluate_orientation_term(*args, **kwargs): raise NotImplementedError('Use structure.evaluate_minimal_bundle')
def evaluate_bundle(*args, **kwargs): raise NotImplementedError('Use structure.evaluate_minimal_bundle')
