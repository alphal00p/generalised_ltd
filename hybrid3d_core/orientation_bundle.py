#!/usr/bin/env python3
from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import itertools
import mpmath as mp

from .graph_io import ParsedGraph, ParsedGraphExternalEdge, repeated_groups, _strip_quotes
from .ltd_cut_structure_generator import CutStructureGenerator, CLOSE_ABOVE, CLOSE_BELOW

Signature = Tuple[Tuple[int, ...], Tuple[int, ...]]
ThreeVector = Tuple[float, float, float]
FourVector = Tuple[float, float, float, float]
NumeratorFn = Callable[[Sequence[FourVector], Sequence[FourVector]], float]
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
    orientation_id: str; family: str; branch_id: int; edge_orientations: Tuple[int,...]; prefactor_sign: int; prefactor_half_edges: Tuple[int,...]; surface_chain: Tuple[int,...]; loop_energy_exprs: Tuple[LinearEnergyExpr,...]; edge_energy_exprs: Tuple[LinearEnergyExpr,...]; meta: Dict[str,Any]
@dataclass(frozen=True)
class ExpressionBundle:
    family: str; loop_names: Tuple[str,...]; ext_names: Tuple[str,...]; signatures: Tuple[Signature,...]; surface_cache: Tuple[SurfaceDef,...]; terms: Tuple[OrientationTerm,...]

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

def _det_fraction(rows):
    n=len(rows); A=[[Fraction(x) for x in row] for row in rows]; det=Fraction(1); sign=1
    for col in range(n):
        piv=None
        for r in range(col,n):
            if A[r][col] != 0: piv=r; break
        if piv is None: return Fraction(0)
        if piv != col:
            A[col],A[piv]=A[piv],A[col]; sign *= -1
        pv=A[col][col]; det *= pv
        for r in range(col+1,n):
            if A[r][col] != 0:
                fac=A[r][col]/pv
                for c in range(col,n): A[r][c] -= fac*A[col][c]
    return det*sign

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
def _label_with_chain(edge_signs, chain_signs): return orientation_id_from_signs(edge_signs)+('|' + orientation_id_from_signs(chain_signs) if chain_signs else '')
def _group_rank_basis(signatures, rep_groups, chain_signs):
    chosen=[]; rows=[]; target=[]
    for grp,chi in zip(rep_groups,chain_signs):
        eid=grp.edge_ids[0]; rel=grp.rel_signs[0]; row=list(signatures[eid][0]); cand=rows+[row]
        if _rank(cand)>_rank(rows): chosen.append(eid); rows.append(row); target.append(int(rel)*int(chi))
    return chosen, rows, target
def _complete_basis(signatures, chosen, rows, avoid):
    n_loops=len(signatures[0][0]); out=list(chosen)
    for eid,sig in enumerate(signatures):
        if eid in out or eid in avoid: continue
        row=list(sig[0]); cand=rows+[row]
        if _rank(cand)>_rank(rows): out.append(eid); rows.append(row)
        if len(out)==n_loops: break
    if len(out)!=n_loops: raise ValueError('Could not complete hybrid basis')
    return out
def _chain_choices(vals):
    vals=[int(v) for v in vals]
    return [-vals[0]] if all(v==vals[0] for v in vals) else [-1,1]
def _make_hybrid_substitution_maps(signatures,n_external_symbols,rep_groups,edge_signs,chain_signs):
    n_internal=len(signatures); rep_ids={e for g in rep_groups for e in g.edge_ids}; chosen,rows,target_sgns=_group_rank_basis(signatures,rep_groups,chain_signs); basis=_complete_basis(signatures,chosen,rows,rep_ids); targets=[LinearEnergyExpr.zero() for _ in range(n_internal)]
    for e,s in zip(chosen,target_sgns): targets[e]=LinearEnergyExpr.E(e,s)
    for e in basis:
        if targets[e].render()=='0': targets[e]=LinearEnergyExpr.E(e,1)
    loop_exprs=solve_loop_energy_from_target_edge_exprs(signatures,basis,targets,n_external_symbols); edge_exprs=list(edge_q0_from_loop_exprs(signatures,loop_exprs,n_external_symbols))
    for g in rep_groups:
        for eid in g.edge_ids:
            if int(edge_signs[eid])!=0: edge_exprs[eid]=LinearEnergyExpr.E(eid,int(edge_signs[eid]))
    return loop_exprs,tuple(edge_exprs),tuple(basis)
def _skeleton_surfaces(signatures,rep_groups,edge_exprs,sb,basis=()):
    rep_ids={e for g in rep_groups for e in g.edge_ids}; basis_ids=set(int(x) for x in basis); out=[]
    for eid,expr in enumerate(edge_exprs):
        if eid in rep_ids or eid in basis_ids: continue
        out.append(sb.intern('auto', expr-LinearEnergyExpr.E(eid,1), f'hybrid-skeleton-{eid}-minus'))
        out.append(sb.intern('auto', expr+LinearEnergyExpr.E(eid,1), f'hybrid-skeleton-{eid}-plus'))
    return tuple(out)

def _local_repeated_surfaces(rep_groups,edge_signs,chain_signs,sb):
    out=[]
    for g,chi in zip(rep_groups,chain_signs):
        anchor=g.edge_ids[0]
        for eid in g.edge_ids[1:]:
            if int(edge_signs[eid]) != int(chi):
                out.append(sb.intern('auto', LinearEnergyExpr.E(anchor,int(chi)) - LinearEnergyExpr.E(eid,int(edge_signs[eid])), f'local-repeat-{anchor}-{eid}'))
    return tuple(out)

def _unit_ray(n,index):
    return tuple(1 if i==index else 0 for i in range(n))

def _ray_rank(rays):
    return _rank(rays)

def _embedded_cones(cones,keep,n):
    out=[]
    for coeff,rays in cones:
        embedded=[]
        for ray in rays:
            full=[0]*n
            for local_idx,global_idx in enumerate(keep):
                full[global_idx]=int(ray[local_idx])
            embedded.append(tuple(full))
        out.append((coeff,tuple(embedded)))
    return out

def _equality_facet_cones(same,opposite,n):
    same=list(same); opposite=list(opposite)
    if not same or not opposite:
        return []
    paths=[]
    def rec(i,j,path):
        path=path+[(i,j)]
        if i==len(same)-1 and j==len(opposite)-1:
            paths.append(path); return
        if i<len(same)-1: rec(i+1,j,path)
        if j<len(opposite)-1: rec(i,j+1,path)
    rec(0,0,[])
    out=[]
    for path in paths:
        rays=[]
        for i,j in path:
            ray=[0]*n
            ray[same[i]]=1
            ray[opposite[j]]=1
            rays.append(tuple(ray))
        out.append((1,tuple(rays)))
    return out

def _det_int(rows):
    return int(_det_fraction(rows))

def _cone_from_rays(rays):
    rays=tuple(tuple(int(x) for x in ray) for ray in rays)
    det=abs(_det_int(rays))
    if det == 0:
        return ()
    return ((det,rays),)

def _halfspace_cone_simplices_dim4(same,opposite):
    n=4; same=tuple(same); opposite=tuple(opposite)
    def e(i): return _unit_ray(n,i)
    def pair(a,b):
        ray=[0]*n; ray[int(a)]=1; ray[int(b)]=1
        return tuple(ray)
    m=len(same); p=len(opposite)
    if p == 1:
        p0=opposite[0]
        return _cone_from_rays((e(p0),)+tuple(pair(a,p0) for a in same))
    if m == 1 and p == 3:
        a=same[0]; p0,p1,p2=opposite
        cones=[
            (pair(a,p2), e(p0), e(p1), e(p2)),
            (pair(a,p0), pair(a,p2), e(p0), e(p1)),
            (pair(a,p0), pair(a,p2), pair(a,p1), e(p1)),
        ]
        out=[]
        for rays in cones:
            out.extend(_cone_from_rays(rays))
        return tuple(out)
    if m == 2 and p == 2:
        a0,a1=same; p0,p1=opposite
        cones=[
            (pair(a0,p0), pair(a1,p0), e(p1), e(p0)),
            (pair(a0,p1), pair(a0,p0), pair(a1,p0), e(p1)),
            (pair(a0,p1), pair(a1,p1), pair(a1,p0), e(p1)),
        ]
        out=[]
        for rays in cones:
            out.extend(_cone_from_rays(rays))
        return tuple(out)
    raise ValueError(f'Unsupported four-dimensional halfspace split {m}+{p}')

def _halfspace_cone_simplices(same,opposite,n):
    same=tuple(sorted(int(x) for x in same)); opposite=tuple(sorted(int(x) for x in opposite))
    if not same:
        return [(1,tuple(_unit_ray(n,i) for i in range(n)))]
    if not opposite:
        return []
    if n == 4:
        return _halfspace_cone_simplices_dim4(same,opposite)
    if n==1:
        return []
    r0=[1]*n
    for i in opposite:
        r0[i]=len(same)+1
    r0=tuple(r0)
    boundary=[]
    for removed in range(n):
        keep=[i for i in range(n) if i!=removed]
        sub_same=[keep.index(i) for i in same if i!=removed]
        sub_opposite=[keep.index(i) for i in opposite if i!=removed]
        sub=_halfspace_cone_simplices(sub_same,sub_opposite,n-1)
        for coeff,rays in _embedded_cones(sub,keep,n):
            if len(rays)==n-1 and _ray_rank(rays)==n-1:
                boundary.append((coeff,rays))
    boundary.extend(_equality_facet_cones(same,opposite,n))
    out=[]; seen=set()
    for coeff,rays in boundary:
        full=(r0,)+tuple(rays)
        if len(full)!=n or _ray_rank(full)!=n:
            continue
        det=abs(_det_int(full))
        if det==0:
            continue
        key=tuple(sorted(full))
        if key in seen:
            continue
        seen.add(key)
        out.append((coeff*det,full))
    return out

def _linear_combo_expr(coeffs,exprs):
    out=LinearEnergyExpr.zero()
    for c,expr in zip(coeffs,exprs):
        if int(c):
            out=out+expr.mul(int(c))
    return out.canonical()

def _local_repeated_cone_terms(rep_group,edge_signs,chain_sign,channel_expr,sb):
    edge_ids=tuple(rep_group.edge_ids); rels=tuple(int(x) for x in rep_group.rel_signs)
    chi=int(chain_sign); anchor=edge_ids[0]; canonical_signs=[rels[i]*int(edge_signs[eid]) for i,eid in enumerate(edge_ids)]
    same=[i for i,s in enumerate(canonical_signs) if int(s)==chi]
    opposite=[i for i,s in enumerate(canonical_signs) if int(s)==-chi]
    cones=_halfspace_cone_simplices(same,opposite,len(edge_ids))
    a_exprs=[]
    for eid,s in zip(edge_ids,canonical_signs):
        a_exprs.append((LinearEnergyExpr.E(eid,1)-channel_expr.mul(int(s))).canonical())
    out=[]
    for coeff,rays in cones:
        chain=[]
        for ray in rays:
            expr=_linear_combo_expr(ray,a_exprs)
            chain.append(sb.intern('auto',expr,f'local-cone-{anchor}-{orientation_id_from_signs(canonical_signs)}-{chi}'))
        out.append((int(coeff)*((-1)**len(edge_ids)),tuple(chain)))
    agg={}
    for coeff,chain in out:
        agg[chain]=agg.get(chain,0)+int(coeff)
    return tuple((coeff,chain) for chain,coeff in agg.items() if coeff)

def _local_repeated_cone_product(rep_groups,edge_signs,chain_signs,channel_exprs,sb):
    acc=[(1,tuple())]
    for group,chi,channel_expr in zip(rep_groups,chain_signs,channel_exprs):
        group_terms=_local_repeated_cone_terms(group,edge_signs,chi,channel_expr,sb)
        nxt=[]
        for coeff_a,chain_a in acc:
            for coeff_b,chain_b in group_terms:
                nxt.append((coeff_a*coeff_b,chain_a+chain_b))
        acc=nxt
    agg={}
    for coeff,chain in acc:
        agg[chain]=agg.get(chain,0)+int(coeff)
    return tuple((coeff,chain) for chain,coeff in agg.items() if coeff)

def _invert_fraction_matrix(rows):
    n=len(rows); A=[[Fraction(x) for x in row] for row in rows]; I=[[Fraction(int(i==j)) for j in range(n)] for i in range(n)]
    for col in range(n):
        piv=None
        for r in range(col,n):
            if A[r][col] != 0:
                piv=r; break
        if piv is None:
            raise ValueError('singular matrix')
        if piv != col:
            A[col],A[piv]=A[piv],A[col]; I[col],I[piv]=I[piv],I[col]
        pv=A[col][col]
        A[col]=[x/pv for x in A[col]]; I[col]=[x/pv for x in I[col]]
        for r in range(n):
            if r==col: continue
            fac=A[r][col]
            if fac:
                A[r]=[a-fac*b for a,b in zip(A[r],A[col])]
                I[r]=[a-fac*b for a,b in zip(I[r],I[col])]
    return I

def _complete_hybrid_energy_basis(signatures,rep_groups):
    n_loops=len(signatures[0][0]); rows=[]; independent_groups=[]
    for idx,group in enumerate(rep_groups):
        row=list(group.key[0][0])
        if _rank(rows+[row])>_rank(rows):
            rows.append(row); independent_groups.append(idx)
    for i in range(n_loops):
        row=[0]*n_loops; row[i]=1
        if _rank(rows+[row])>_rank(rows):
            rows.append(row)
        if len(rows)==n_loops:
            break
    if len(rows)!=n_loops:
        raise ValueError('Could not complete hybrid energy basis')
    return tuple(tuple(int(x) for x in row) for row in rows),tuple(independent_groups)

def _loop_coeffs_in_energy_basis(loop_coeffs,basis_rows):
    inv=_invert_fraction_matrix(basis_rows)
    out=[]
    for col in range(len(basis_rows)):
        val=sum(Fraction(loop_coeffs[i])*inv[i][col] for i in range(len(basis_rows)))
        out.append(val)
    return tuple(out)

def _fraction_to_generator_number(x):
    x=Fraction(x)
    if x.denominator==1:
        return int(x)
    return float(x)

def _canonical_group_exprs(rep_groups,loop_exprs,n_external):
    out=[]
    for group in rep_groups:
        out.append(edge_q0_from_loop_exprs((group.key[0],),loop_exprs,n_external)[0])
    return tuple(out)

def _build_coupled_cff_hybrid_bundle(parsed, reason):
    cff=build_pure_cff_bundle(parsed)
    rep_groups=repeated_groups(parsed)
    hybrid_terms=[]
    for branch,term in enumerate(reversed(cff.terms)):
        meta=dict(term.meta)
        meta.update({
            'source':'hybrid_coupled_cff_cone_kernel',
            'coupled_reason':reason,
            'repeated_groups':[list(g.edge_ids) for g in rep_groups],
        })
        hybrid_terms.append(OrientationTerm(
            term.orientation_id + '|coupled',
            'hybrid',
            branch,
            term.edge_orientations,
            term.prefactor_sign,
            term.prefactor_half_edges,
            term.surface_chain,
            term.loop_energy_exprs,
            term.edge_energy_exprs,
            meta,
        ))
    return ExpressionBundle('hybrid',cff.loop_names,cff.ext_names,cff.signatures,cff.surface_cache,tuple(hybrid_terms))

def _fixed_tau_local_cone_supported(signatures,rep_groups):
    # The factorized fixed-tau implementation below has been validated for a
    # single loop-energy variable.  In multi-loop graphs the Fourier closure
    # signs are linear combinations of the repeated-sector tau sums; treating
    # each repeated group as an independent local cone misses coupled residues.
    return len(signatures[0][0]) == 1

def _build_fixed_tau_local_hybrid_bundle(parsed):
    rep_groups=repeated_groups(parsed); signatures=tuple(e.signature for e in parsed.internal_edges); n_external=len(parsed.ext_names)
    if not rep_groups:
        ltd=build_pure_ltd_bundle(signatures,n_external); return ExpressionBundle('hybrid',ltd.loop_names,ltd.ext_names,ltd.signatures,ltd.surface_cache,ltd.terms)
    sb=SurfaceCacheBuilder(); terms=[]; bc=0; rep_ids={e for g in rep_groups for e in g.edge_ids}
    nonrep_edges=[eid for eid in range(len(signatures)) if eid not in rep_ids]
    energy_basis,independent_group_indices=_complete_hybrid_energy_basis(signatures,rep_groups)
    transformed_nonrep=[
        tuple(_fraction_to_generator_number(x) for x in _loop_coeffs_in_energy_basis(signatures[eid][0],energy_basis))
        for eid in nonrep_edges
    ]
    group_sign_choices=[list(itertools.product([-1,1],repeat=len(g.edge_ids))) for g in rep_groups]
    for group_edge_signs in itertools.product(*group_sign_choices):
        edge_signs=[0]*len(signatures); canonical_group_signs=[]
        for group,signs in zip(rep_groups,group_edge_signs):
            vals=[]
            for eid,rel,sign in zip(group.edge_ids,group.rel_signs,signs):
                edge_signs[eid]=int(sign); vals.append(int(rel)*int(sign))
            canonical_group_signs.append(vals)
        chain_choices=[_chain_choices(vals) for vals in canonical_group_signs]
        for chain_signs in itertools.product(*chain_choices):
            closure=[CLOSE_BELOW]*len(energy_basis)
            for basis_pos,group_idx in enumerate(independent_group_indices):
                closure[basis_pos]=CLOSE_BELOW if int(chain_signs[group_idx])>0 else CLOSE_ABOVE
            residues=CutStructureGenerator(transformed_nonrep).get_residues(closure,simplify=True)
            for residue in residues:
                residual_basis=tuple(nonrep_edges[int(i)] for i in residue['basis'])
                cut_signs=[int(s) for s in residue['sigmas']]
                loop_exprs,edge_exprs_base=solve_loop_energy_substitutions(signatures,residual_basis,cut_signs,n_external)
                edge_exprs=list(edge_exprs_base)
                for group in rep_groups:
                    for eid in group.edge_ids:
                        edge_exprs[eid]=LinearEnergyExpr.E(eid,int(edge_signs[eid]))
                edge_exprs=tuple(edge_exprs)
                channel_exprs=_canonical_group_exprs(rep_groups,loop_exprs,n_external)
                skeleton=_skeleton_surfaces(signatures,rep_groups,edge_exprs,sb,residual_basis)
                residue_sign=1 if float(residue['sign']) > 0 else -1
                orient_edge_signs=list(edge_signs)
                for eid,sign in zip(residual_basis,cut_signs):
                    orient_edge_signs[eid]=int(sign)
                for cone_coeff,cone_chain in _local_repeated_cone_product(rep_groups,edge_signs,chain_signs,channel_exprs,sb):
                    chain=tuple(skeleton)+tuple(cone_chain)
                    half_edges=tuple(sorted(set(residual_basis)|rep_ids))
                    terms.append(OrientationTerm(
                        _label_with_chain(orient_edge_signs,chain_signs),
                        'hybrid',
                        bc,
                        tuple(orient_edge_signs),
                        int(residue_sign*cone_coeff),
                        half_edges,
                        chain,
                        loop_exprs,
                        edge_exprs,
                        {'basis':list(residual_basis),'cut_signs':cut_signs,'chain_signs':list(chain_signs),'energy_basis':[list(r) for r in energy_basis],'source':'hybrid_ltd_local_cone','repeated_groups':[list(g.edge_ids) for g in rep_groups]},
                    )); bc+=1
    return ExpressionBundle('hybrid',tuple(),tuple(),signatures,sb.build(),tuple(terms))

def build_hybrid_bundle_raw(parsed):
    rep_groups=repeated_groups(parsed); signatures=tuple(e.signature for e in parsed.internal_edges)
    if not rep_groups:
        ltd=build_pure_ltd_bundle(signatures,len(parsed.ext_names))
        return ExpressionBundle('hybrid',ltd.loop_names,ltd.ext_names,ltd.signatures,ltd.surface_cache,ltd.terms)
    if _fixed_tau_local_cone_supported(signatures,rep_groups):
        return _build_fixed_tau_local_hybrid_bundle(parsed)
    return _build_coupled_cff_hybrid_bundle(parsed,'multi_loop_repeated_tau_closures_are_coupled')

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
