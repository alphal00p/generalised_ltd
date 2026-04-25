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
    orientation_id: str; family: str; branch_id: int; edge_orientations: Tuple[int,...]; prefactor_sign: Any; prefactor_half_edges: Tuple[int,...]; surface_chain: Tuple[int,...]; loop_energy_exprs: Tuple[LinearEnergyExpr,...]; edge_energy_exprs: Tuple[LinearEnergyExpr,...]; meta: Dict[str,Any]; numerator_surface_chain: Tuple[int,...] = tuple()
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

@dataclass(frozen=True)
class _NumeratorSample:
    coeff: Fraction
    extra_half_edges: Tuple[int, ...]
    loop_exprs: Tuple[LinearEnergyExpr, ...]
    edge_exprs: Tuple[LinearEnergyExpr, ...]
    edge_orientations: Tuple[int, ...]
    label: str
    meta: Dict[str, Any]

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

def _sign_token(sign: int) -> str:
    return 'p' if int(sign) > 0 else 'm'

def _internal_alias(channels: Sequence[_LogicalChannel]) -> Dict[int, int]:
    alias: Dict[int, int] = {}
    for ch in channels:
        for member in ch.members:
            alias[int(member)] = int(ch.rep_edge)
    return alias

def _coeff_map(expr: LinearEnergyExpr, internal_alias: Dict[int, int]) -> Dict[Tuple[str, int], Fraction]:
    out: Dict[Tuple[str, int], Fraction] = {}
    const = Fraction(str(expr.const))
    if const:
        out[('c', 0)] = const
    for edge_id, coeff in expr.internal_terms:
        key = ('i', int(internal_alias.get(int(edge_id), int(edge_id))))
        out[key] = out.get(key, Fraction(0)) + Fraction(int(coeff))
    for ext_id, coeff in expr.external_terms:
        key = ('x', int(ext_id))
        out[key] = out.get(key, Fraction(0)) + Fraction(int(coeff))
    return {k: v for k, v in out.items() if v}

def _component_coeff_maps(loop_exprs: Sequence[LinearEnergyExpr], edge_exprs: Sequence[LinearEnergyExpr], internal_alias: Dict[int, int]) -> List[Dict[Tuple[str, int], Fraction]]:
    return [_coeff_map(expr, internal_alias) for expr in tuple(loop_exprs) + tuple(edge_exprs)]

def _independent_rows(rows: Sequence[Sequence[Fraction]]) -> List[int]:
    selected: List[int] = []
    current: List[List[Fraction]] = []
    current_rank = 0
    for idx, row in enumerate(rows):
        if not any(row):
            continue
        trial = current + [list(row)]
        trial_rank = _rank(trial)
        if trial_rank > current_rank:
            selected.append(idx)
            current = trial
            current_rank = trial_rank
    return selected

def _minimum_norm_affine_weights(rows: Sequence[Sequence[Fraction]], targets: Sequence[Fraction]) -> Tuple[Fraction, ...]:
    if not rows:
        return tuple()
    ncols = len(rows[0])
    if ncols == 0:
        return tuple()
    rows = [tuple(Fraction(x) for x in row) for row in rows]
    targets = [Fraction(x) for x in targets]
    keep = _independent_rows(rows)
    for row, target in zip(rows, targets):
        if not any(row) and target:
            raise RuntimeError('Physical numerator sample orbit cannot satisfy an affine zero-row constraint')
    if not keep:
        if any(targets):
            raise RuntimeError('Physical numerator sample orbit has no independent affine constraints')
        return tuple(Fraction(0) for _ in range(ncols))
    basis_rows = [rows[i] for i in keep]
    basis_targets = [targets[i] for i in keep]
    gram = [
        [sum(a * b for a, b in zip(row_i, row_j)) for row_j in basis_rows]
        for row_i in basis_rows
    ]
    dual = _solve_fraction_system(gram, basis_targets)
    coeffs = tuple(sum(row[j] * dual[i] for i, row in enumerate(basis_rows)) for j in range(ncols))
    for row, target in zip(rows, targets):
        got = sum(c * x for c, x in zip(coeffs, row))
        if got != target:
            raise RuntimeError(f'Physical numerator sample orbit is not affine-complete: got {got}, expected {target}')
    return coeffs

def _physical_group_options(channel: _LogicalChannel) -> Tuple[Tuple[Tuple[int, ...], int, str], ...]:
    options = []
    for tau in itertools.product((-1, 1), repeat=channel.power):
        if all(s > 0 for s in tau):
            chain_signs = (1,)
        elif all(s < 0 for s in tau):
            chain_signs = (-1,)
        else:
            chain_signs = (-1, 1)
        tau_label = ''.join('p' if s > 0 else 'm' for s in tau)
        for chain_sign in chain_signs:
            options.append((tuple(int(s) for s in tau), int(chain_sign), f't{tau_label}c{_sign_token(chain_sign)}'))
    return tuple(options)

def _physical_sample_orbit(signatures: Sequence[Signature], channels: Sequence[_LogicalChannel], basis_logical: Sequence[int], cut_signs: Sequence[int], alpha: Sequence[int], n_external: int):
    n_internal = len(signatures)
    active_positions = tuple(i for i, a in enumerate(alpha) if int(a) > 0)
    base_by_bpos = {}
    option_by_bpos = {}
    for bpos in active_positions:
        channel = channels[int(basis_logical[bpos])]
        sigma = int(cut_signs[bpos])
        options = _physical_group_options(channel)
        option_by_bpos[bpos] = options
        base_tau = tuple(sigma for _ in channel.members)
        base_by_bpos[bpos] = next(opt for opt in options if opt[0] == base_tau and opt[1] == sigma)

    # The numerator contract is affine in the EMR energies.  A full tensor
    # product of physical tau sectors is therefore redundant: mixed sign-sector
    # samples only test bilinear numerator dependence, which is outside the
    # supported class.  Use an additive affine orbit instead: the physical
    # diagonal plus one complete tau orbit per active repeated group.
    specs = [dict(base_by_bpos)]
    for bpos in active_positions:
        for opt in option_by_bpos[bpos]:
            spec = dict(base_by_bpos)
            spec[bpos] = opt
            specs.append(spec)
    seen_specs = set()
    samples = []
    for by_bpos in specs:
        spec_key = tuple((bpos, by_bpos[bpos][0], by_bpos[bpos][1]) for bpos in active_positions)
        if spec_key in seen_specs:
            continue
        seen_specs.add(spec_key)
        targets = [LinearEnergyExpr.zero() for _ in signatures]
        for bpos, (logical_idx, sigma) in enumerate(zip(basis_logical, cut_signs)):
            channel = channels[int(logical_idx)]
            chain_sign = by_bpos[bpos][1] if bpos in by_bpos else int(sigma)
            targets[channel.rep_edge] = LinearEnergyExpr.E(channel.rep_edge, chain_sign)
        basis_orig = tuple(channels[int(i)].rep_edge for i in basis_logical)
        loop_exprs = solve_loop_energy_from_target_edge_exprs(signatures, basis_orig, targets, n_external)
        edge_exprs = list(edge_q0_from_loop_exprs(signatures, loop_exprs, n_external))
        edge_orient = [0] * n_internal
        labels = []
        repeated_signs: Dict[int, List[int]] = {}
        chain_signs: Dict[int, int] = {}
        for bpos, (logical_idx, sigma) in enumerate(zip(basis_logical, cut_signs)):
            channel = channels[int(logical_idx)]
            if bpos in by_bpos:
                tau, chain_sign, label = by_bpos[bpos]
                labels.append(f'G{channel.rep_edge}{label}')
                repeated_signs[int(channel.rep_edge)] = [int(x) for x in tau]
                chain_signs[int(channel.rep_edge)] = int(chain_sign)
                for member, sign in zip(channel.members, tau):
                    edge_exprs[int(member)] = LinearEnergyExpr.E(int(member), int(sign))
                    edge_orient[int(member)] = int(sign)
            else:
                for member in channel.members:
                    edge_orient[int(member)] = int(sigma)
        label = 'phys0' if not labels else 'phys' + '_'.join(labels)
        samples.append({
            'loop_exprs': tuple(loop_exprs),
            'edge_exprs': tuple(edge_exprs),
            'edge_orientations': tuple(edge_orient),
            'label': label,
            'repeated_signs': repeated_signs,
            'chain_signs': chain_signs,
        })
    return tuple(samples)

def _numerator_derivative_samples(
    beta: Sequence[int],
    signatures: Sequence[Signature],
    channels: Sequence[_LogicalChannel],
    basis_logical: Sequence[int],
    cut_signs: Sequence[int],
    alpha: Sequence[int],
    base_loop_exprs: Sequence[LinearEnergyExpr],
    base_edge_exprs: Sequence[LinearEnergyExpr],
    loop_derivs: Sequence[Sequence[Fraction]],
    edge_derivs: Sequence[Sequence[Fraction]],
    n_external: int,
) -> Tuple[_NumeratorSample, ...]:
    beta=tuple(int(x) for x in beta)
    samples = _physical_sample_orbit(signatures, channels, basis_logical, cut_signs, alpha, n_external)
    if not samples:
        return tuple()
    internal_alias = _internal_alias(channels)
    sample_component_maps = [
        _component_coeff_maps(sample['loop_exprs'], sample['edge_exprs'], internal_alias)
        for sample in samples
    ]
    n_components = len(sample_component_maps[0])
    symbol_keys = set()
    for component_maps in sample_component_maps:
        for cmap in component_maps:
            symbol_keys.update(cmap)

    extra_half_edges: Tuple[int, ...] = tuple()
    if not any(beta):
        constant_target = Fraction(1)
        target_maps = _component_coeff_maps(base_loop_exprs, base_edge_exprs, internal_alias)
        sample_label = 'n0'
    else:
        # The supported numerator class is affine in EMR energies.  Exact first
        # derivatives are reconstructed from physical on-shell samples; higher
        # numerator derivatives vanish in this class.
        if sum(beta) != 1 or max(beta) != 1:
            return tuple()
        bpos = beta.index(1)
        logical_idx = int(basis_logical[bpos])
        rep_edge = int(channels[logical_idx].rep_edge)
        extra_half_edges = (rep_edge,)
        constant_target = Fraction(0)
        target_maps = []
        for row in loop_derivs:
            val = Fraction(row[bpos])
            target_maps.append({('i', int(internal_alias.get(rep_edge, rep_edge))): 2 * val} if val else {})
        for row in edge_derivs:
            val = Fraction(row[bpos])
            target_maps.append({('i', int(internal_alias.get(rep_edge, rep_edge))): 2 * val} if val else {})
        sample_label = f'nD{bpos}phys'
    if len(target_maps) != n_components:
        raise RuntimeError('Internal numerator interpolation component mismatch')
    for cmap in target_maps:
        symbol_keys.update(cmap)
    ordered_symbols = sorted(symbol_keys)
    rows: List[List[Fraction]] = [[Fraction(1) for _ in samples]]
    targets: List[Fraction] = [constant_target]
    for comp_idx in range(n_components):
        for sym in ordered_symbols:
            row = [component_maps[comp_idx].get(sym, Fraction(0)) for component_maps in sample_component_maps]
            target = target_maps[comp_idx].get(sym, Fraction(0))
            if any(row) or target:
                rows.append(row)
                targets.append(target)
    coeffs = _minimum_norm_affine_weights(rows, targets)
    out: List[_NumeratorSample] = []
    for coeff, sample in zip(coeffs, samples):
        if not coeff:
            continue
        out.append(_NumeratorSample(
            coeff=coeff,
            extra_half_edges=extra_half_edges,
            loop_exprs=sample['loop_exprs'],
            edge_exprs=sample['edge_exprs'],
            edge_orientations=sample['edge_orientations'],
            label=f"{sample_label}:{sample['label']}",
            meta={
                'numerator_sample_kind': 'physical_on_shell_affine_orbit',
                'physical_chain_signs': sample['chain_signs'],
                'physical_repeated_signs': sample['repeated_signs'],
            },
        ))
    return tuple(out)

def _derivative_nodes(degree: int) -> Tuple[int, ...]:
    degree = max(0, int(degree))
    nodes = [0]
    step = 1
    while len(nodes) < degree + 1:
        nodes.extend([step, -step])
        step += 1
    return tuple(nodes[:degree + 1])

def _finite_difference_weights(nodes: Sequence[int], deriv_order: int, degree: int) -> Tuple[Fraction, ...]:
    degree = int(degree)
    deriv_order = int(deriv_order)
    if deriv_order > degree:
        return tuple(Fraction(0) for _ in nodes)
    A = [[Fraction(a) ** p for a in nodes] for p in range(degree + 1)]
    rhs = [Fraction(math.factorial(deriv_order) if p == deriv_order else 0) for p in range(degree + 1)]
    return tuple(_solve_fraction_system(A, rhs))

def _basis_variable_degree_bounds(
    bounds: Sequence[int],
    edge_derivs: Sequence[Sequence[Fraction]],
    n_basis: int,
) -> Tuple[int, ...]:
    out = []
    for bpos in range(n_basis):
        degree = 0
        for edge_id, row in enumerate(edge_derivs):
            if row[bpos]:
                degree += int(bounds[edge_id])
        out.append(degree)
    return tuple(out)

def _bounded_numerator_derivative_samples(
    beta: Sequence[int],
    bounds: Sequence[int],
    signatures: Sequence[Signature],
    channels: Sequence[_LogicalChannel],
    basis_logical: Sequence[int],
    cut_signs: Sequence[int],
    edge_derivs: Sequence[Sequence[Fraction]],
    n_external: int,
) -> Tuple[_NumeratorSample, ...]:
    beta = tuple(int(x) for x in beta)
    basis_orig = tuple(channels[int(i)].rep_edge for i in basis_logical)
    if not any(beta):
        targets = [LinearEnergyExpr.zero() for _ in signatures]
        for edge_id, sigma in zip(basis_orig, cut_signs):
            targets[int(edge_id)] = LinearEnergyExpr.E(int(edge_id), int(sigma))
        loop_exprs = solve_loop_energy_from_target_edge_exprs(signatures, basis_orig, targets, n_external)
        edge_exprs = edge_q0_from_loop_exprs(signatures, loop_exprs, n_external)
        edge_orient = [0] * len(signatures)
        for edge_id, sigma in zip(basis_orig, cut_signs):
            edge_orient[int(edge_id)] = int(sigma)
        return (_NumeratorSample(
            coeff=Fraction(1),
            extra_half_edges=tuple(),
            loop_exprs=tuple(loop_exprs),
            edge_exprs=tuple(edge_exprs),
            edge_orientations=tuple(edge_orient),
            label='n0:fd0',
            meta={'numerator_sample_kind': 'bounded_degree_finite_difference', 'finite_difference_beta': list(beta)},
        ),)

    degree_by_basis = _basis_variable_degree_bounds(bounds, edge_derivs, len(basis_logical))
    per_axis = []
    for bpos, order in enumerate(beta):
        if not order:
            continue
        degree = int(degree_by_basis[bpos])
        if order > degree:
            return tuple()
        nodes = _derivative_nodes(degree)
        weights = _finite_difference_weights(nodes, order, degree)
        per_axis.append((bpos, tuple((int(node), weight) for node, weight in zip(nodes, weights) if weight)))

    out: List[_NumeratorSample] = []
    for choices in itertools.product(*(items for _, items in per_axis)):
        coeff = Fraction(1)
        offsets: Dict[int, int] = {}
        for (bpos, _), (node, weight) in zip(per_axis, choices):
            coeff *= Fraction(weight)
            offsets[int(bpos)] = int(node)
        if not coeff:
            continue
        targets = [LinearEnergyExpr.zero() for _ in signatures]
        labels = []
        for bpos, (edge_id, sigma) in enumerate(zip(basis_orig, cut_signs)):
            sample_coeff = int(sigma) + int(offsets.get(bpos, 0))
            targets[int(edge_id)] = LinearEnergyExpr.E(int(edge_id), sample_coeff)
            if bpos in offsets:
                labels.append(f'b{bpos}{sample_coeff:+d}')
        loop_exprs = solve_loop_energy_from_target_edge_exprs(signatures, basis_orig, targets, n_external)
        edge_exprs = edge_q0_from_loop_exprs(signatures, loop_exprs, n_external)
        edge_orient = [0] * len(signatures)
        for edge_id, expr in enumerate(edge_exprs):
            if expr == LinearEnergyExpr.E(edge_id, 1):
                edge_orient[edge_id] = 1
            elif expr == LinearEnergyExpr.E(edge_id, -1):
                edge_orient[edge_id] = -1
        extra_half_edges: List[int] = []
        for bpos, order in enumerate(beta):
            rep_edge = int(channels[int(basis_logical[bpos])].rep_edge)
            extra_half_edges.extend(rep_edge for _ in range(int(order)))
        coeff *= Fraction(2) ** sum(beta)
        out.append(_NumeratorSample(
            coeff=coeff,
            extra_half_edges=tuple(extra_half_edges),
            loop_exprs=tuple(loop_exprs),
            edge_exprs=tuple(edge_exprs),
            edge_orientations=tuple(edge_orient),
            label='nD' + ''.join(str(x) for x in beta) + ':fd' + ','.join(labels),
            meta={
                'numerator_sample_kind': 'bounded_degree_finite_difference',
                'finite_difference_beta': list(beta),
                'finite_difference_offsets': dict(offsets),
                'finite_difference_degree_bounds_by_basis': list(degree_by_basis),
            },
        ))
    return tuple(out)

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

def normalize_energy_degree_bounds(bounds: Any, n_internal: int) -> Optional[Tuple[int, ...]]:
    if bounds is None:
        return None
    if isinstance(bounds, (list, tuple)):
        if len(bounds) > n_internal:
            raise ValueError(f'Energy-degree bound list has {len(bounds)} entries for {n_internal} internal edges')
        out = [0] * n_internal
        for i, value in enumerate(bounds):
            out[i] = int(value)
    elif isinstance(bounds, dict):
        default = int(bounds.get('*', 0))
        out = [default] * n_internal
        for key, value in bounds.items():
            if str(key) == '*':
                continue
            idx = int(key)
            if idx < 0 or idx >= n_internal:
                raise ValueError(f'Energy-degree bound edge index {idx} is outside 0..{n_internal - 1}')
            out[idx] = int(value)
    else:
        raise TypeError('Energy-degree bounds must be a list/tuple or a dict')
    for idx, value in enumerate(out):
        if value < 0:
            raise ValueError(f'Energy-degree bound for edge {idx} is negative: {value}')
    return tuple(out)

def _energy_direction_reports(signatures: Sequence[Signature], bounds_tuple: Tuple[int, ...]) -> Tuple[Dict[str, Any], ...]:
    """Return UV degrees for all one-parameter loop-energy directions.

    The coordinate-axis test is insufficient for multiloop numerators: an
    infinity arc can live along any non-zero vector in loop-energy space.  The
    active denominator set is constant on each flat of the hyperplane
    arrangement q_e^0(v)=0, so it is enough to enumerate the closures generated
    by subsets of edge loop-signature rows.
    """
    if not signatures:
        return tuple()
    rows = [tuple(int(x) for x in sig[0]) for sig in signatures]
    n_loops = len(rows[0])
    n_edges = len(rows)
    seen_active = set()
    out: List[Dict[str, Any]] = []
    for mask in range(1 << n_edges):
        zero_rows = [rows[e] for e in range(n_edges) if mask & (1 << e)]
        zero_rank = _rank(zero_rows)
        closure = tuple(
            e for e, row in enumerate(rows)
            if _rank(zero_rows + [row]) == zero_rank
        )
        closure_rank = _rank([rows[e] for e in closure])
        if closure_rank >= n_loops:
            continue
        active = tuple(e for e in range(n_edges) if e not in closure)
        if active in seen_active:
            continue
        seen_active.add(active)
        numerator_degree = sum(bounds_tuple[e] for e in active)
        denominator_degree = 2 * len(active)
        divergence_degree = numerator_degree - denominator_degree
        out.append({
            'zero_edges': list(closure),
            'active_edges': list(active),
            'nullity': n_loops - closure_rank,
            'numerator_degree_bound': numerator_degree,
            'denominator_degree': denominator_degree,
            'divergence_degree': divergence_degree,
            'convergent': divergence_degree < -1,
        })
    out.sort(key=lambda item: (item['divergence_degree'], -len(item['active_edges']), item['active_edges']), reverse=True)
    return tuple(out)

def energy_divergence_report(signatures: Sequence[Signature], bounds: Any):
    bounds_tuple = normalize_energy_degree_bounds(bounds, len(signatures))
    if bounds_tuple is None:
        return None
    n_loops = len(signatures[0][0]) if signatures else 0
    loops = []
    for loop_id in range(n_loops):
        active_edges = [edge_id for edge_id, sig in enumerate(signatures) if int(sig[0][loop_id]) != 0]
        numerator_degree = sum(bounds_tuple[edge_id] for edge_id in active_edges)
        denominator_degree = 2 * len(active_edges)
        divergence_degree = numerator_degree - denominator_degree
        loops.append({
            'loop': loop_id,
            'active_edges': active_edges,
            'numerator_degree_bound': numerator_degree,
            'denominator_degree': denominator_degree,
            'divergence_degree': divergence_degree,
            'convergent': divergence_degree < -1,
        })
    directions = _energy_direction_reports(signatures, bounds_tuple)
    return {
        'edge_degree_bounds': list(bounds_tuple),
        'loops': loops,
        'directions': list(directions),
        'coordinate_convergent': all(item['convergent'] for item in loops),
        'directional_convergent': all(item['convergent'] for item in directions),
        'convergent': all(item['convergent'] for item in loops) and all(item['convergent'] for item in directions),
    }

def assert_energy_uv_convergent(signatures: Sequence[Signature], bounds: Any):
    report = energy_divergence_report(signatures, bounds)
    if report is None:
        return None
    bad = [('coordinate', item) for item in report['loops'] if not item['convergent']]
    bad.extend(('direction', item) for item in report.get('directions', []) if not item['convergent'])
    if bad:
        details = ', '.join(
            (
                f"k{item['loop']}^0: degree {item['divergence_degree']} from edges {item['active_edges']}"
                if kind == 'coordinate' else
                f"direction active={item['active_edges']} zero={item['zero_edges']}: degree {item['divergence_degree']}"
            )
            for kind, item in bad
        )
        raise ValueError(f'Energy-degree bounds leave non-vanishing residue at infinity ({details})')
    return report

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

def _scaled_pref(prefactor: Any, scale: Fraction) -> str:
    return _frac_to_str(Fraction(str(prefactor)) * scale)

def _append_bundle_terms(
    out_terms: List[OrientationTerm],
    sb: SurfaceCacheBuilder,
    source_bundle: ExpressionBundle,
    scale: Fraction,
    label_prefix: str,
    meta_source: str,
    branch_start: int,
) -> int:
    surface_map = {
        int(surface.surface_id): sb.intern(surface.kind, surface.expr, surface.label)
        for surface in source_bundle.surface_cache
    }
    branch = int(branch_start)
    for term in source_bundle.terms:
        pref = _scaled_pref(term.prefactor_sign, scale)
        if pref == '0':
            continue
        meta = dict(term.meta)
        meta['source'] = meta_source
        meta['original_source'] = term.meta.get('source')
        meta['finite_pole_completion_scale'] = _frac_to_str(scale)
        out_terms.append(OrientationTerm(
            f'{label_prefix}:{term.orientation_id}',
            'bounded_cff',
            branch,
            term.edge_orientations,
            pref,
            term.prefactor_half_edges,
            tuple(surface_map[int(sid)] for sid in term.surface_chain),
            term.loop_energy_exprs,
            term.edge_energy_exprs,
            meta,
            tuple(surface_map[int(sid)] for sid in term.numerator_surface_chain),
        ))
        branch += 1
    return branch

def _remap_linear_expr(expr: LinearEnergyExpr, edge_map: Dict[int, int]) -> LinearEnergyExpr:
    return LinearEnergyExpr(
        tuple((int(edge_map[int(edge_id)]), int(coeff)) for edge_id, coeff in expr.internal_terms),
        expr.external_terms,
        expr.const,
    ).canonical()

def _poly_add(a: Sequence[Fraction], b: Sequence[Fraction]) -> Tuple[Fraction, ...]:
    n = max(len(a), len(b))
    out = [Fraction(0) for _ in range(n)]
    for i, value in enumerate(a):
        out[i] += Fraction(value)
    for i, value in enumerate(b):
        out[i] += Fraction(value)
    while out and out[-1] == 0:
        out.pop()
    return tuple(out)

def _poly_scale(poly: Sequence[Fraction], scale: Fraction) -> Tuple[Fraction, ...]:
    out = [Fraction(value) * scale for value in poly]
    while out and out[-1] == 0:
        out.pop()
    return tuple(out)

def _poly_mul(a: Sequence[Fraction], b: Sequence[Fraction]) -> Tuple[Fraction, ...]:
    if not a or not b:
        return tuple()
    out = [Fraction(0) for _ in range(len(a) + len(b) - 1)]
    for i, av in enumerate(a):
        for j, bv in enumerate(b):
            out[i + j] += Fraction(av) * Fraction(bv)
    while out and out[-1] == 0:
        out.pop()
    return tuple(out)

def _contact_nodes(bound: int) -> Tuple[int, ...]:
    needed = max(0, int(bound) - 1)
    candidates = [0]
    n = 2
    while len(candidates) < needed:
        candidates.extend([n, -n])
        n += 1
    return tuple(candidates[:needed])

def _lagrange_basis(nodes: Sequence[int], idx: int) -> Tuple[Fraction, ...]:
    poly: Tuple[Fraction, ...] = (Fraction(1),)
    denom = Fraction(1)
    xj = Fraction(nodes[idx])
    for k, xk_int in enumerate(nodes):
        if k == idx:
            continue
        xk = Fraction(xk_int)
        poly = _poly_mul(poly, (-xk, Fraction(1)))
        denom *= (xj - xk)
    return _poly_scale(poly, Fraction(1, 1) / denom)

def _contact_weight_polys(bound: int) -> Dict[int, Tuple[Fraction, ...]]:
    # Contact quotient for D=(q^0)^2-E^2.  With x=q^0/E and interpolation
    # nodes a_j != +-1:
    # C(E*x) = sum_j L_j(x) [N(a_jE)-R(a_j)] / ((a_j^2-1)E^2).
    # The returned polynomials multiply 1/E^2; monomials x^r are encoded later
    # as q^r/E^(r+2) with numerator-side q factors and half-edge powers.
    if int(bound) <= 1:
        return {}
    nodes = _contact_nodes(int(bound))
    weights: Dict[int, Tuple[Fraction, ...]] = {}
    for j, a_int in enumerate(nodes):
        a = Fraction(a_int)
        denom = a * a - 1
        basis = _lagrange_basis(nodes, j)
        contributions = {
            int(a_int): Fraction(1, 1) / denom,
            1: -((a + 1) / 2) / denom,
            -1: -((1 - a) / 2) / denom,
        }
        for sample, scale in contributions.items():
            weights[sample] = _poly_add(weights.get(sample, tuple()), _poly_scale(basis, scale))
    return {sample: poly for sample, poly in weights.items() if poly}

def _linear_expr_is_zero(expr: LinearEnergyExpr) -> bool:
    return not expr.internal_terms and not expr.external_terms and expr.const in {'0', '0.0'}

def _finite_pole_contact_components(
    edge_id: int,
    bound: int,
    current_expr: LinearEnergyExpr,
    sb: SurfaceCacheBuilder,
) -> Tuple[Tuple[int, Fraction, Tuple[int, ...], Tuple[int, ...]], ...]:
    components: List[Tuple[int, Fraction, Tuple[int, ...], Tuple[int, ...]]] = []
    q_surface = None
    if not _linear_expr_is_zero(current_expr):
        q_surface = sb.intern('auto', current_expr, f'contact-q{edge_id}')
    for sample, poly in _contact_weight_polys(int(bound)).items():
        for power, coeff in enumerate(poly):
            if not coeff:
                continue
            if power and q_surface is None:
                continue
            half_edges = tuple(int(edge_id) for _ in range(power + 2))
            num_surfaces = tuple(int(q_surface) for _ in range(power)) if power else tuple()
            pref = Fraction(coeff) * (2 ** (power + 2))
            if pref:
                components.append((int(sample), pref, half_edges, num_surfaces))
    return tuple(components)

def _finite_pole_remainder_components(
    edge_id: int,
    current_expr: LinearEnergyExpr,
    sb: SurfaceCacheBuilder,
) -> Tuple[Tuple[int, Fraction, Tuple[int, ...], Tuple[int, ...]], ...]:
    plus = sb.intern('auto', (current_expr + LinearEnergyExpr.E(edge_id, 1)).canonical(), f'remainder-plus-q{edge_id}')
    minus = sb.intern('auto', (LinearEnergyExpr.E(edge_id, 1) - current_expr).canonical(), f'remainder-minus-q{edge_id}')
    return (
        (1, Fraction(1), (int(edge_id),), (int(plus),)),
        (-1, Fraction(1), (int(edge_id),), (int(minus),)),
    )

def _delete_parsed_edges(parsed: ParsedGraph, pinched_edges: Sequence[int]) -> Tuple[ParsedGraph, Tuple[int, ...]]:
    pinched = {int(e) for e in pinched_edges}
    sub_to_orig: List[int] = []
    internal_edges = []
    for edge in parsed.internal_edges:
        if edge.edge_id in pinched:
            continue
        sub_id = len(internal_edges)
        sub_to_orig.append(int(edge.edge_id))
        internal_edges.append(type(edge)(
            sub_id,
            edge.tail,
            edge.head,
            edge.label,
            edge.mass_key,
            edge.signature,
            edge.had_pow,
        ))
    return ParsedGraph(
        tuple(internal_edges),
        parsed.external_edges,
        parsed.loop_names,
        parsed.ext_names,
        parsed.node_name_to_internal,
    ), tuple(sub_to_orig)

def _contract_parsed_edges(parsed: ParsedGraph, pinched_edges: Sequence[int]) -> Tuple[ParsedGraph, Tuple[int, ...]]:
    pinched = {int(e) for e in pinched_edges}
    parent = {int(v): int(v) for v in parsed.node_name_to_internal.values()}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int):
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[rb] = ra

    for edge in parsed.internal_edges:
        if edge.edge_id in pinched:
            union(edge.tail, edge.head)

    roots = sorted({find(v) for v in parent})
    root_to_new = {root: idx for idx, root in enumerate(roots)}
    old_to_new = {old: root_to_new[find(old)] for old in parent}
    node_name_to_internal = {f'c{idx}': idx for idx in range(len(roots))}

    sub_to_orig: List[int] = []
    internal_edges = []
    for edge in parsed.internal_edges:
        if edge.edge_id in pinched:
            continue
        sub_id = len(internal_edges)
        sub_to_orig.append(int(edge.edge_id))
        internal_edges.append(type(edge)(
            sub_id,
            old_to_new[edge.tail],
            old_to_new[edge.head],
            edge.label,
            edge.mass_key,
            edge.signature,
            edge.had_pow,
        ))

    external_edges = []
    for edge in parsed.external_edges:
        source_base = _strip_quotes(edge.source).split(':', 1)[0]
        dest_base = _strip_quotes(edge.destination).split(':', 1)[0]
        source_internal = parsed.node_name_to_internal.get(source_base)
        dest_internal = parsed.node_name_to_internal.get(dest_base)
        source = f'c{old_to_new[source_internal]}' if source_internal is not None else edge.source
        dest = f'c{old_to_new[dest_internal]}' if dest_internal is not None else edge.destination
        external_edges.append(type(edge)(edge.edge_id, source, dest, edge.label, edge.ext_coeffs))

    return ParsedGraph(
        tuple(internal_edges),
        tuple(external_edges),
        parsed.loop_names,
        parsed.ext_names,
        node_name_to_internal,
    ), tuple(sub_to_orig)

def _lift_contracted_cff_terms(
    parsed: ParsedGraph,
    bounds: Tuple[int, ...],
    report: dict,
    sb: SurfaceCacheBuilder,
    terms: List[OrientationTerm],
    branch: int,
    pinched_edges: Sequence[int],
) -> int:
    n_internal = len(parsed.internal_edges)
    pinched = tuple(sorted(int(e) for e in pinched_edges))
    subparsed, sub_to_orig = _contract_parsed_edges(parsed, pinched)
    if not subparsed.internal_edges:
        return branch
    n_loops = len(parsed.loop_names)
    use_terminal_ltd = len(subparsed.internal_edges) == n_loops
    sub_bundle = (
        build_pure_ltd_bundle(tuple(e.signature for e in subparsed.internal_edges), len(subparsed.ext_names))
        if use_terminal_ltd else
        build_pure_cff_bundle(subparsed)
    )
    edge_map = {sub_id: orig_id for sub_id, orig_id in enumerate(sub_to_orig)}
    surface_map = {
        int(surface.surface_id): sb.intern(
            surface.kind,
            _remap_linear_expr(surface.expr, edge_map),
            f'pinch{list(pinched)}:{surface.label}',
        )
        for surface in sub_bundle.surface_cache
    }
    unit_surface = sb.intern('e', LinearEnergyExpr(tuple(), tuple(), '1'), 'terminal-unit') if use_terminal_ltd else None
    for term in sub_bundle.terms:
        full_loop_exprs = tuple(_remap_linear_expr(expr, edge_map) for expr in term.loop_energy_exprs)
        current_edge_exprs = edge_q0_from_loop_exprs(tuple(e.signature for e in parsed.internal_edges), full_loop_exprs, len(parsed.ext_names))
        base_edge_exprs = [LinearEnergyExpr.zero() for _ in range(n_internal)]
        full_orient = [0 for _ in range(n_internal)]
        for sub_id, orig_id in enumerate(sub_to_orig):
            base_edge_exprs[int(orig_id)] = _remap_linear_expr(term.edge_energy_exprs[sub_id], edge_map)
            full_orient[int(orig_id)] = int(term.edge_orientations[sub_id])
        remapped_half_edges = tuple(int(edge_map[int(e)]) for e in term.prefactor_half_edges)
        remapped_chain = tuple(int(surface_map[int(sid)]) for sid in term.surface_chain)
        remapped_num_chain = tuple(int(surface_map[int(sid)]) for sid in term.numerator_surface_chain)
        if use_terminal_ltd and not remapped_chain:
            remapped_chain = (int(unit_surface),)
        per_edge_choices = [
            _finite_pole_contact_components(edge_id, bounds[int(edge_id)], current_edge_exprs[int(edge_id)], sb)
            for edge_id in pinched
        ]
        sample_iter = itertools.product(*per_edge_choices) if pinched else [tuple()]
        for sample in sample_iter:
            coeff = Fraction(str(term.prefactor_sign))
            sample_label_parts = []
            edge_exprs = list(base_edge_exprs)
            edge_orient = list(full_orient)
            extra_half_edges: List[int] = []
            extra_num_surfaces: List[int] = []
            for edge_id, (sample_value, sample_pref, sample_half_edges, sample_num_surfaces) in zip(pinched, sample):
                coeff *= Fraction(sample_pref)
                extra_half_edges.extend(int(e) for e in sample_half_edges)
                extra_num_surfaces.extend(int(sid) for sid in sample_num_surfaces)
                if sample_value == 0:
                    edge_exprs[int(edge_id)] = LinearEnergyExpr.zero()
                    edge_orient[int(edge_id)] = 0
                    sample_label_parts.append(f'e{edge_id}=0')
                else:
                    edge_exprs[int(edge_id)] = LinearEnergyExpr.E(int(edge_id), int(sample_value))
                    edge_orient[int(edge_id)] = 1 if int(sample_value) > 0 else -1
                    sample_label_parts.append(f'e{edge_id}={"+" if sample_value == 1 else "-" if sample_value == -1 else sample_value}')
            if pinched and not use_terminal_ltd:
                coeff *= Fraction((-1) ** len(pinched))
            if not coeff:
                continue
            label = f'cff|{term.orientation_id}' if not pinched else 'pinch[' + ','.join(str(e) for e in pinched) + ']'
            if sample_label_parts:
                label += '|' + '|'.join(sample_label_parts)
            if pinched:
                label += f'|cff={term.orientation_id}'
            meta = dict(term.meta)
            meta.update({
                'source': 'bounded_degree_e_surface_pinch_cff',
                'original_source': term.meta.get('source'),
                'energy_degree_bounds': list(bounds),
                'energy_divergence': report,
                'finite_pole_completion': True,
                'finite_pole_completion_e_surfaces_only': True,
                'pinched_edges': list(pinched),
                'contracted_edge_map': list(sub_to_orig),
                'terminal_ltd_e_surface_sector': bool(use_terminal_ltd),
                'merge_by_numerator_map': True,
            })
            terms.append(OrientationTerm(
                label,
                'bounded_cff',
                branch,
                tuple(edge_orient),
                _frac_to_str(coeff),
                tuple(sorted(remapped_half_edges + tuple(extra_half_edges))),
                remapped_chain,
                full_loop_exprs,
                tuple(edge_exprs),
                meta,
                tuple(remapped_num_chain + tuple(extra_num_surfaces)),
            ))
            branch += 1
    return branch

def build_quadratic_e_surface_bounded_cff_bundle(parsed: ParsedGraph, bounds: Tuple[int, ...], report: dict) -> ExpressionBundle:
    quadratic_edges = tuple(idx for idx, bound in enumerate(bounds) if int(bound) > 1)
    sb = SurfaceCacheBuilder()
    terms: List[OrientationTerm] = []
    branch = 0
    for r in range(len(quadratic_edges) + 1):
        for pinched in itertools.combinations(quadratic_edges, r):
            branch = _lift_contracted_cff_terms(parsed, bounds, report, sb, terms, branch, pinched)
    enriched = []
    for idx, term in enumerate(terms):
        enriched.append(OrientationTerm(
            term.orientation_id,
            term.family,
            idx,
            term.edge_orientations,
            term.prefactor_sign,
            term.prefactor_half_edges,
            term.surface_chain,
            term.loop_energy_exprs,
            term.edge_energy_exprs,
            term.meta,
            term.numerator_surface_chain,
        ))
    return ExpressionBundle('bounded_cff', tuple(), tuple(), tuple(e.signature for e in parsed.internal_edges), sb.build(), tuple(enriched))

def _lift_deleted_ltd_quadratic_sector_terms(
    parsed: ParsedGraph,
    bounds: Tuple[int, ...],
    report: dict,
    sb: SurfaceCacheBuilder,
    terms: List[OrientationTerm],
    branch: int,
    pinched_edges: Sequence[int],
) -> int:
    n_internal = len(parsed.internal_edges)
    signatures = tuple(e.signature for e in parsed.internal_edges)
    quadratic_edges = tuple(idx for idx, bound in enumerate(bounds) if int(bound) > 1)
    pinched = tuple(sorted(int(e) for e in pinched_edges))
    subparsed, sub_to_orig = _delete_parsed_edges(parsed, pinched)
    if not subparsed.internal_edges:
        return branch
    sub_bounds = tuple(min(int(bounds[orig_id]), 1) for orig_id in sub_to_orig)
    sector_report = assert_energy_uv_convergent(tuple(e.signature for e in subparsed.internal_edges), sub_bounds)
    if pinched:
        sub_bundle = build_pure_ltd_bundle(tuple(e.signature for e in subparsed.internal_edges), len(subparsed.ext_names))
        sector_backend = 'deleted_ltd_contact_sector'
    else:
        sub_bundle = build_pure_cff_bundle(subparsed)
        sector_backend = 'ordinary_cff_remainder_sector'
    edge_map = {sub_id: orig_id for sub_id, orig_id in enumerate(sub_to_orig)}
    surface_map = {
        int(surface.surface_id): sb.intern(
            surface.kind,
            _remap_linear_expr(surface.expr, edge_map),
            f'quad{list(pinched)}:{surface.label}',
        )
        for surface in sub_bundle.surface_cache
    }
    for term in sub_bundle.terms:
        full_loop_exprs = tuple(_remap_linear_expr(expr, edge_map) for expr in term.loop_energy_exprs)
        current_edge_exprs = edge_q0_from_loop_exprs(signatures, full_loop_exprs, len(parsed.ext_names))
        base_edge_exprs = [LinearEnergyExpr.zero() for _ in range(n_internal)]
        full_orient = [0 for _ in range(n_internal)]
        for sub_id, orig_id in enumerate(sub_to_orig):
            base_edge_exprs[int(orig_id)] = _remap_linear_expr(term.edge_energy_exprs[sub_id], edge_map)
            full_orient[int(orig_id)] = int(term.edge_orientations[sub_id])
        remapped_half_edges = tuple(int(edge_map[int(e)]) for e in term.prefactor_half_edges)
        remapped_chain = tuple(int(surface_map[int(sid)]) for sid in term.surface_chain)
        remapped_num_chain = tuple(int(surface_map[int(sid)]) for sid in term.numerator_surface_chain)

        per_edge_choices = []
        active_quadratic_edges = []
        pinched_set = set(pinched)
        for edge_id in quadratic_edges:
            active_quadratic_edges.append(int(edge_id))
            if edge_id in pinched_set:
                per_edge_choices.append(_finite_pole_contact_components(edge_id, bounds[int(edge_id)], current_edge_exprs[int(edge_id)], sb))
            else:
                per_edge_choices.append(_finite_pole_remainder_components(edge_id, current_edge_exprs[int(edge_id)], sb))
        sample_iter = itertools.product(*per_edge_choices) if per_edge_choices else [tuple()]
        for sample in sample_iter:
            coeff = Fraction(str(term.prefactor_sign))
            sample_label_parts = []
            edge_exprs = list(base_edge_exprs)
            edge_orient = list(full_orient)
            extra_half_edges: List[int] = []
            extra_num_surfaces: List[int] = []
            for edge_id, (sample_value, sample_pref, sample_half_edges, sample_num_surfaces) in zip(active_quadratic_edges, sample):
                coeff *= Fraction(sample_pref)
                extra_half_edges.extend(int(e) for e in sample_half_edges)
                extra_num_surfaces.extend(int(sid) for sid in sample_num_surfaces)
                if sample_value == 0:
                    edge_exprs[int(edge_id)] = LinearEnergyExpr.zero()
                    edge_orient[int(edge_id)] = 0
                    sample_label_parts.append(f'e{edge_id}=0')
                else:
                    edge_exprs[int(edge_id)] = LinearEnergyExpr.E(int(edge_id), int(sample_value))
                    edge_orient[int(edge_id)] = 1 if int(sample_value) > 0 else -1
                    sample_label_parts.append(f'e{edge_id}={"+" if sample_value == 1 else "-" if sample_value == -1 else sample_value}')
            if not coeff:
                continue
            if pinched:
                label = 'qpinch[' + ','.join(str(e) for e in pinched) + ']'
            else:
                label = f'cff-rem|{term.orientation_id}'
            if sample_label_parts:
                label += '|' + '|'.join(sample_label_parts)
            if pinched:
                label += f'|ltd={term.orientation_id}'
            meta = dict(term.meta)
            meta.update({
                'source': 'bounded_degree_quadratic_sector',
                'original_source': term.meta.get('source'),
                'energy_degree_bounds': list(bounds),
                'energy_divergence': report,
                'sector_energy_divergence': sector_report,
                'finite_pole_completion': True,
                'finite_pole_completion_e_surfaces_only': False,
                'quadratic_remainder_contact_decomposition': True,
                'quadratic_sector_backend': sector_backend,
                'pinched_edges': list(pinched),
                'deleted_edge_map': list(sub_to_orig),
                'merge_by_numerator_map': True,
            })
            terms.append(OrientationTerm(
                label,
                'bounded_cff',
                branch,
                tuple(edge_orient),
                _frac_to_str(coeff),
                tuple(sorted(remapped_half_edges + tuple(extra_half_edges))),
                remapped_chain,
                full_loop_exprs,
                tuple(edge_exprs),
                meta,
                tuple(remapped_num_chain + tuple(extra_num_surfaces)),
            ))
            branch += 1
    return branch

def build_quadratic_general_bounded_cff_bundle(parsed: ParsedGraph, bounds: Tuple[int, ...], report: dict) -> ExpressionBundle:
    quadratic_edges = tuple(idx for idx, bound in enumerate(bounds) if int(bound) > 1)
    sb = SurfaceCacheBuilder()
    terms: List[OrientationTerm] = []
    branch = 0
    for r in range(len(quadratic_edges) + 1):
        for pinched in itertools.combinations(quadratic_edges, r):
            branch = _lift_deleted_ltd_quadratic_sector_terms(parsed, bounds, report, sb, terms, branch, pinched)
    enriched = []
    for idx, term in enumerate(terms):
        enriched.append(OrientationTerm(
            term.orientation_id,
            term.family,
            idx,
            term.edge_orientations,
            term.prefactor_sign,
            term.prefactor_half_edges,
            term.surface_chain,
            term.loop_energy_exprs,
            term.edge_energy_exprs,
            term.meta,
            term.numerator_surface_chain,
        ))
    return ExpressionBundle('bounded_cff', tuple(), tuple(), tuple(e.signature for e in parsed.internal_edges), sb.build(), tuple(enriched))

def build_bounded_degree_cff_bundle(parsed, energy_degree_bounds):
    signatures = tuple(e.signature for e in parsed.internal_edges)
    bounds = normalize_energy_degree_bounds(energy_degree_bounds, len(signatures))
    report = assert_energy_uv_convergent(signatures, bounds)
    if bounds is None or max(bounds) <= 1:
        bundle = build_pure_cff_bundle(parsed)
        if report is None:
            return bundle
        terms = []
        for idx, term in enumerate(bundle.terms):
            meta = dict(term.meta)
            meta['energy_degree_bounds'] = list(bounds)
            meta['energy_divergence'] = report
            terms.append(OrientationTerm(
                term.orientation_id,
                term.family,
                idx,
                term.edge_orientations,
                term.prefactor_sign,
                term.prefactor_half_edges,
                term.surface_chain,
                term.loop_energy_exprs,
                term.edge_energy_exprs,
                meta,
                term.numerator_surface_chain,
            ))
        return ExpressionBundle(bundle.family, bundle.loop_names, bundle.ext_names, bundle.signatures, bundle.surface_cache, tuple(terms))

    high_edges = [idx for idx, bound in enumerate(bounds) if int(bound) > 2]
    if repeated_groups(parsed):
        bundle = build_hybrid_bundle_raw(parsed, energy_degree_bounds=bounds)
        terms = []
        for idx, term in enumerate(bundle.terms):
            meta = dict(term.meta)
            meta['source'] = 'bounded_degree_repeated_cff_confluent_limit'
            meta['original_source'] = term.meta.get('source')
            meta['energy_degree_bounds'] = list(bounds)
            meta['energy_divergence'] = report
            meta['merge_by_numerator_map'] = True
            terms.append(OrientationTerm(
                term.orientation_id,
                'bounded_cff',
                idx,
                term.edge_orientations,
                term.prefactor_sign,
                term.prefactor_half_edges,
                term.surface_chain,
                term.loop_energy_exprs,
                term.edge_energy_exprs,
                meta,
                term.numerator_surface_chain,
            ))
        return ExpressionBundle('bounded_cff', tuple(), tuple(), signatures, bundle.surface_cache, tuple(terms))

    if (
        len(parsed.loop_names) == 1
    ):
        if high_edges:
            isolated_cubic = len(high_edges) == 1 and int(bounds[high_edges[0]]) == 3 and sum(int(x) for x in bounds) == 3
            if not isolated_cubic:
                raise NotImplementedError(
                    'The bounded CFF lift supports arbitrary quadratic caps.  Higher caps or mixed '
                    'cubic sectors need the next recursive reduction of known numerator-side '
                    'polynomial factors.'
                )
        return build_quadratic_e_surface_bounded_cff_bundle(parsed, bounds, report)

    if not high_edges:
        return build_quadratic_general_bounded_cff_bundle(parsed, bounds, report)

    raise NotImplementedError(
        'Bounded-degree pure CFF supports arbitrary quadratic caps.  Higher caps in multiloop '
        'or repeated sectors need the recursive known-polynomial contact lift.'
    )

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

def _build_confluent_hybrid_bundle(parsed: ParsedGraph, energy_degree_bounds=None):
    signatures=tuple(e.signature for e in parsed.internal_edges)
    bounds = normalize_energy_degree_bounds(energy_degree_bounds, len(signatures)) if energy_degree_bounds is not None else None
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
        if bounds is None or max(bounds) <= 1:
            beta_candidates=[beta for beta in _iter_multiindices_leq(alpha) if sum(beta) <= 1]
        else:
            beta_candidates=[beta for beta in _iter_multiindices_leq(alpha)]
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
            if bounds is None or max(bounds) <= 1:
                num_samples=_numerator_derivative_samples(
                    beta,
                    signatures,
                    channels,
                    basis_logical,
                    cut_signs,
                    alpha,
                    loop_exprs,
                    edge_exprs,
                    loop_derivs,
                    edge_derivs,
                    n_external,
                )
            else:
                num_samples=_bounded_numerator_derivative_samples(
                    beta,
                    bounds,
                    signatures,
                    channels,
                    basis_logical,
                    cut_signs,
                    edge_derivs,
                    n_external,
                )
            if not num_samples:
                continue
            den_terms=_denominator_derivative_terms(factors,gamma)
            if not den_terms:
                continue
            for num_sample in num_samples:
                for dcoeff, half_edges, chain in den_terms:
                    merged_half_edges=tuple(sorted(tuple(int(x) for x in half_edges)+tuple(num_sample.extra_half_edges)))
                    coeff=residue_sign*residue_norm*leibniz*num_sample.coeff*dcoeff
                    if not coeff:
                        continue
                    beta_label=''.join(str(x) for x in beta) or '0'
                    gamma_label=''.join(str(x) for x in gamma) or '0'
                    label=f"{orientation_id_from_signs(num_sample.edge_orientations)}|B{','.join(str(x) for x in basis_orig)}|b{beta_label}|g{gamma_label}|{num_sample.label}"
                    meta={
                        'source':'hybrid_confluent_ltd_interpolation',
                        'basis_logical':list(basis_logical),
                        'basis':list(basis_orig),
                        'cut_signs':list(cut_signs),
                        'basis_powers':list(powers),
                        'alpha':list(alpha),
                        'beta':list(beta),
                        'gamma':list(gamma),
                        'active_basis_edges':list(active_basis),
                        'active_edges':list(active_edges),
                        'denominator_surface_ids':list(denominator_surface_ids),
                        'repeated_groups':[list(ch.members) for ch in repeated],
                        'merge_by_numerator_map': True,
                    }
                    if bounds is not None:
                        meta['energy_degree_bounds'] = list(bounds)
                    meta.update(num_sample.meta)
                    terms.append(OrientationTerm(
                        label,
                        'hybrid',
                        branch,
                        tuple(num_sample.edge_orientations),
                        _frac_to_str(coeff),
                        merged_half_edges,
                        tuple(int(x) for x in chain),
                        tuple(num_sample.loop_exprs),
                        tuple(num_sample.edge_exprs),
                        meta,
                    ))
                    branch += 1
    return ExpressionBundle('hybrid',tuple(),tuple(),signatures,sb.build(),tuple(terms))

def build_hybrid_bundle_raw(parsed, energy_degree_bounds=None):
    rep_groups=repeated_groups(parsed); signatures=tuple(e.signature for e in parsed.internal_edges)
    if not rep_groups:
        ltd=build_pure_ltd_bundle(signatures,len(parsed.ext_names))
        return ExpressionBundle('hybrid',ltd.loop_names,ltd.ext_names,ltd.signatures,ltd.surface_cache,ltd.terms)
    return _build_confluent_hybrid_bundle(parsed, energy_degree_bounds=energy_degree_bounds)

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
