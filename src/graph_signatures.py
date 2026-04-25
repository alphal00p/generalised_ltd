#!/usr/bin/env python3
"""
Reconstruct a loop graph (directed multigraph) from edge momentum signatures
and emit Graphviz DOT using pydot.

Input:
    signatures = [
        ( (loop_coeffs...), (ext_coeffs...) ),
        ...
    ]
where each coefficient is in {-1, 0, +1}.

Assumptions (default):
  - The internal graph is connected.
  - The number of loop momenta L equals the cyclomatic number of the internal graph.
    Then V = E_int - L + 1 is used as the internal vertex count.

Notes:
  - The reconstruction is not unique in general; this code returns one valid realization if it finds any.
  - Runtime is exponential in worst case (graph realization is closely related to matroid/graphic-matrix realization).
"""

from __future__ import annotations

import argparse
import ast
import itertools
import json
import random
import re
import subprocess
import sys
import unittest
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pydot

LoopVec = Tuple[int, ...]
ExtVec = Tuple[int, ...]
Signature = Tuple[LoopVec, ExtVec]


def _check_coeffs(vec: Sequence[int]) -> None:
    for x in vec:
        if x not in (-1, 0, 1):
            raise ValueError(f"Coefficient {x} not in {{-1,0,1}}.")


def _format_lincomb(coeffs: Sequence[int], names: Sequence[str]) -> str:
    """Format sum_i coeffs[i]*names[i] with coeffs in {-1,0,1}."""
    terms: List[str] = []
    for c, n in zip(coeffs, names):
        if c == 0:
            continue
        if c == 1:
            terms.append(n)
        elif c == -1:
            terms.append(f"-{n}")
        else:
            terms.append(f"{c}*{n}")
    if not terms:
        return "0"
    # Join nicely: "k - l + p" rather than "k + -l + p"
    out = terms[0]
    for t in terms[1:]:
        if t.startswith("-"):
            out += " - " + t[1:]
        else:
            out += " + " + t
    return out


def format_momentum(loop: LoopVec, ext: ExtVec, loop_names: Sequence[str], ext_names: Sequence[str]) -> str:
    lpart = _format_lincomb(loop, loop_names)
    epart = _format_lincomb(ext, ext_names)
    if lpart == "0":
        return epart
    if epart == "0":
        return lpart
    return f"{lpart} + {epart}"


_SYMBOLICA_FORMAT_PLAIN_CACHE: Dict[str, str] = {}


def _dot_quote_label(label: str) -> str:
    escaped = label.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _normalize_sign_runs(expr_text: str) -> str:
    expr = expr_text.replace(" ", "")
    while True:
        normalized = expr.replace("+-", "-").replace("-+", "-").replace("--", "+").replace("++", "+")
        if normalized == expr:
            return normalized
        expr = normalized


def _format_plain_with_symbolica_subprocess(expr_text: str) -> str:
    cached = _SYMBOLICA_FORMAT_PLAIN_CACHE.get(expr_text)
    if cached is not None:
        return cached

    child_code = "import sys\nfrom symbolica import E\nprint(E(sys.argv[1]).format_plain())\n"
    proc = subprocess.run(
        [sys.executable, "-c", child_code, expr_text],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        stdout = (proc.stdout or "").strip()
        raise RuntimeError(stderr or stdout or "Unknown Symbolica formatting subprocess failure.")

    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("Symbolica formatter produced no output.")
    result = lines[-1]
    _SYMBOLICA_FORMAT_PLAIN_CACHE[expr_text] = result
    return result


def _normalize_momentum_label(label: str) -> str:
    text = label.strip()
    if not text:
        return "0"
    try:
        return _format_plain_with_symbolica_subprocess(text)
    except Exception:
        normalized = _normalize_sign_runs(text)
        return normalized or "0"


_MOMENTUM_SYMBOL_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SYMBOL_WITH_INT_SUFFIX_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*?)(\d+)$")


def _parse_e_wrapper_text(expr_text: str) -> str:
    text = expr_text.strip()
    if not (text.startswith("E(") and text.endswith(")")):
        return text

    try:
        parsed = ast.parse(text, mode="eval")
    except SyntaxError:
        return text

    call = parsed.body
    if not isinstance(call, ast.Call):
        return text
    if not isinstance(call.func, ast.Name) or call.func.id != "E":
        return text
    if not call.args:
        return text
    first_arg = call.args[0]
    if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
        return first_arg.value
    return text


def _split_linear_terms(momentum_expr_text: str) -> List[Tuple[int, str]]:
    text = momentum_expr_text.replace(" ", "")
    if not text:
        raise ValueError("Empty momentum expression.")

    # Normalize sign runs to keep parsing simple.
    while True:
        normalized = text.replace("+-", "-").replace("-+", "-").replace("--", "+").replace("++", "+")
        if normalized == text:
            break
        text = normalized

    if text[0] not in "+-":
        text = "+" + text

    terms: List[Tuple[int, str]] = []
    idx = 0
    while idx < len(text):
        sign = 1 if text[idx] == "+" else -1
        idx += 1
        start = idx
        while idx < len(text) and text[idx] not in "+-":
            idx += 1
        token = text[start:idx]
        if not token:
            raise ValueError(f"Malformed momentum expression '{momentum_expr_text}'.")
        if any(ch in token for ch in "*/^()"):
            raise ValueError(f"Only linear sums/differences of bare momentum symbols are supported; got token '{token}' in '{momentum_expr_text}'.")
        symbol = token.split("::")[-1]
        if not _MOMENTUM_SYMBOL_RE.fullmatch(symbol):
            raise ValueError(f"Unsupported momentum symbol '{symbol}' in '{momentum_expr_text}'.")
        terms.append((sign, symbol))
    return terms


def _symbol_sort_key(name: str) -> Tuple[int, str, int, str]:
    m = _SYMBOL_WITH_INT_SUFFIX_RE.fullmatch(name)
    if m:
        return (0, m.group(1), int(m.group(2)), name)
    return (1, name, -1, name)


def _extract_pattern_value(match_obj: Any, key: str) -> Any:
    # Symbolica matches are dict-like: m["q_"].
    if hasattr(match_obj, "__getitem__"):
        return match_obj[key]
    if hasattr(match_obj, "get"):
        return match_obj.get(key)
    raise TypeError("Unsupported Symbolica match object; could not access wildcard captures.")


def _find_matching_paren(text: str, open_idx: int) -> int:
    depth = 0
    for idx in range(open_idx, len(text)):
        ch = text[idx]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return idx
    raise ValueError("Unbalanced parentheses while parsing expression.")


def _split_top_level_args(args_text: str) -> List[str]:
    parts: List[str] = []
    current: List[str] = []
    paren_depth = 0
    bracket_depth = 0
    brace_depth = 0

    for ch in args_text:
        if ch == "," and paren_depth == 0 and bracket_depth == 0 and brace_depth == 0:
            parts.append("".join(current).strip())
            current = []
            continue
        current.append(ch)
        if ch == "(":
            paren_depth += 1
        elif ch == ")":
            paren_depth -= 1
        elif ch == "[":
            bracket_depth += 1
        elif ch == "]":
            bracket_depth -= 1
        elif ch == "{":
            brace_depth += 1
        elif ch == "}":
            brace_depth -= 1

    if paren_depth != 0 or bracket_depth != 0 or brace_depth != 0:
        raise ValueError("Unbalanced delimiters in function argument list.")
    parts.append("".join(current).strip())
    return parts


_PROP_CALL_RE = re.compile(r"(?:[A-Za-z_][A-Za-z0-9_]*::)*prop\s*\(")


def _extract_qm_pairs_from_text_fallback(expr_text: str) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    pos = 0
    while True:
        m = _PROP_CALL_RE.search(expr_text, pos)
        if m is None:
            break
        open_idx = m.end() - 1
        close_idx = _find_matching_paren(expr_text, open_idx)
        args = _split_top_level_args(expr_text[open_idx + 1 : close_idx])
        if len(args) < 2:
            raise ValueError(f"Malformed propagator call '{expr_text[m.start() : close_idx + 1]}'.")
        pairs.append((args[0], args[1]))
        pos = close_idx + 1
    return pairs


def _extract_qm_pairs_from_symbolica_subprocess(
    expr_text: str,
    prop_pattern: str,
) -> List[Tuple[str, str]]:
    child_code = (
        "import json,sys\n"
        "from symbolica import E\n"
        "expr=E(sys.argv[1])\n"
        "pat=E(sys.argv[2])\n"
        "out=[]\n"
        "for m in expr.match(pat):\n"
        "    out.append((str(m['q_']), str(m['m_'])))\n"
        "print(json.dumps(out))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", child_code, expr_text, prop_pattern],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        stdout = (proc.stdout or "").strip()
        raise RuntimeError(stderr or stdout or "Unknown Symbolica subprocess failure.")
    try:
        raw = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Could not decode Symbolica subprocess output.") from exc
    return [(str(q), str(m)) for q, m in raw]


def _build_signatures_from_qm_pairs(
    qm_pairs: Sequence[Tuple[str, str]],
    loop_prefix: str = "k",
    external_prefixes: Sequence[str] = ("p", "q"),
) -> Tuple[List[Signature], List[str], List[str], List[str]]:
    if not loop_prefix:
        raise ValueError("loop_prefix must be non-empty.")
    if not external_prefixes:
        raise ValueError("external_prefixes must be non-empty.")

    edge_loop: List[Dict[str, int]] = []
    edge_ext: List[Dict[str, int]] = []
    masses: List[str] = []
    loop_names_set = set()
    ext_names_set = set()

    for momentum_expr, mass_expr in qm_pairs:
        coeffs: Dict[str, int] = {}
        for sign, symbol in _split_linear_terms(str(momentum_expr)):
            coeffs[symbol] = coeffs.get(symbol, 0) + sign

        loop_coeffs: Dict[str, int] = {}
        ext_coeffs: Dict[str, int] = {}
        for symbol, coeff in coeffs.items():
            if coeff == 0:
                continue
            if coeff not in (-1, 1):
                raise ValueError(f"Symbol '{symbol}' has coefficient {coeff} in '{momentum_expr}', but only {{-1,0,+1}} are supported.")
            if symbol.startswith(loop_prefix):
                loop_coeffs[symbol] = coeff
                loop_names_set.add(symbol)
            elif any(symbol.startswith(pfx) for pfx in external_prefixes):
                ext_coeffs[symbol] = coeff
                ext_names_set.add(symbol)
            else:
                allowed = ", ".join([loop_prefix, *external_prefixes])
                raise ValueError(f"Momentum symbol '{symbol}' does not match allowed prefixes [{allowed}] in '{momentum_expr}'.")

        edge_loop.append(loop_coeffs)
        edge_ext.append(ext_coeffs)
        masses.append(str(mass_expr).strip())

    loop_names = sorted(loop_names_set, key=_symbol_sort_key)
    ext_names = sorted(ext_names_set, key=_symbol_sort_key)

    signatures: List[Signature] = []
    for loop_coeffs, ext_coeffs in zip(edge_loop, edge_ext):
        loop_vec = tuple(loop_coeffs.get(name, 0) for name in loop_names)
        ext_vec = tuple(ext_coeffs.get(name, 0) for name in ext_names)
        _check_coeffs(loop_vec)
        _check_coeffs(ext_vec)
        signatures.append((loop_vec, ext_vec))

    return signatures, loop_names, ext_names, masses


def extract_signatures_and_masses_from_symbolica_expression(
    expr_or_text: Any,
    loop_prefix: str = "k",
    external_prefixes: Sequence[str] = ("p", "q"),
    prop_pattern: str = "prop(q_,m_)",
) -> Tuple[List[Signature], List[str], List[str], List[str]]:
    """
    Extract propagator momentum signatures (+ masses) from a Symbolica expression.

    The primary path uses Symbolica pattern matching (`prop(q_,m_)`) in a subprocess.
    If Symbolica cannot be used locally (for example, unlicensed single-instance limits),
    it falls back to a direct textual parser for `prop(momentum,mass)` factors.
    """
    if hasattr(expr_or_text, "match"):
        try:
            from symbolica import E
        except Exception as exc:
            raise RuntimeError("Could not import Symbolica for in-process expression matching.") from exc
        matches = expr_or_text.match(E(prop_pattern))
        qm_pairs = [
            (
                str(_extract_pattern_value(m, "q_")),
                str(_extract_pattern_value(m, "m_")),
            )
            for m in matches
        ]
    else:
        if not isinstance(expr_or_text, str):
            raise TypeError("Expected a Symbolica expression object or a string.")
        expr_text = _parse_e_wrapper_text(expr_or_text)
        qm_pairs: List[Tuple[str, str]] = []
        symbolica_error: Optional[str] = None
        try:
            qm_pairs = _extract_qm_pairs_from_symbolica_subprocess(expr_text, prop_pattern)
        except Exception as exc:
            symbolica_error = str(exc)

        if not qm_pairs:
            if prop_pattern != "prop(q_,m_)":
                detail = f" Symbolica error: {symbolica_error}" if symbolica_error else ""
                raise RuntimeError("Pattern extraction without Symbolica only supports prop(q_,m_)." + detail)
            qm_pairs = _extract_qm_pairs_from_text_fallback(expr_text)

        if not qm_pairs:
            detail = f" Symbolica error: {symbolica_error}" if symbolica_error else ""
            raise ValueError(f"No propagators matched pattern '{prop_pattern}'. Expected something like prop(momentum, mass) factors." + detail)

    return _build_signatures_from_qm_pairs(
        qm_pairs=qm_pairs,
        loop_prefix=loop_prefix,
        external_prefixes=external_prefixes,
    )


def extract_signatures_from_symbolica_expression(
    expr_or_text: Any,
    loop_prefix: str = "k",
    external_prefixes: Sequence[str] = ("p", "q"),
    prop_pattern: str = "prop(q_,m_)",
) -> Tuple[List[Signature], List[str], List[str]]:
    signatures, loop_names, ext_names, _masses = extract_signatures_and_masses_from_symbolica_expression(
        expr_or_text,
        loop_prefix=loop_prefix,
        external_prefixes=external_prefixes,
        prop_pattern=prop_pattern,
    )
    return signatures, loop_names, ext_names


def reconstruct_dot_from_symbolica_expression(
    expr_or_text: Any,
    loop_prefix: str = "k",
    external_prefixes: Sequence[str] = ("p", "q"),
    prop_pattern: str = "prop(q_,m_)",
    num_vertices: Optional[int] = None,
    require_connected: bool = True,
    max_degree: Optional[int] = None,
) -> Tuple[pydot.Dot, List[Signature], List[str], List[str], List[str]]:
    signatures, loop_names, ext_names, masses = extract_signatures_and_masses_from_symbolica_expression(
        expr_or_text,
        loop_prefix=loop_prefix,
        external_prefixes=external_prefixes,
        prop_pattern=prop_pattern,
    )
    dot = reconstruct_dot(
        signatures,
        loop_names=list(loop_names),
        ext_names=list(ext_names),
        num_vertices=num_vertices,
        require_connected=require_connected,
        max_degree=max_degree,
        edge_masses=masses,
    )
    return dot, signatures, loop_names, ext_names, masses


class DSURollback:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.size = [1] * n
        self.changes: List[Tuple[int, int, int]] = []  # (b_root, old_parent, old_size_a)

    def find(self, a: int) -> int:
        while self.parent[a] != a:
            a = self.parent[a]
        return a

    def snapshot(self) -> int:
        return len(self.changes)

    def rollback(self, snap: int) -> None:
        while len(self.changes) > snap:
            b, old_parent, old_size_a = self.changes.pop()
            a = self.parent[b]  # current parent (the root it was attached to)
            self.parent[b] = old_parent
            self.size[a] = old_size_a

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        # attach rb under ra
        self.changes.append((rb, self.parent[rb], self.size[ra]))
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        return True


@dataclass(frozen=True)
class EdgePlacement:
    tail: int
    head: int


@dataclass(frozen=True)
class GraphCase:
    num_vertices: int
    edges: Tuple[Tuple[int, int], ...]
    basis_edge_indices: Tuple[int, ...]
    vertex_potentials: Tuple[Tuple[int, ...], ...]
    loop_names: Tuple[str, ...]
    ext_names: Tuple[str, ...]


def _find_lmb_edge_indices(loop_vecs: Sequence[LoopVec], ext_vecs: Sequence[ExtVec]) -> Dict[int, int]:
    lmb_edges: Dict[int, int] = {}
    if not loop_vecs:
        return lmb_edges
    n_loops = len(loop_vecs[0])
    for edge_idx, (lv, ev) in enumerate(zip(loop_vecs, ext_vecs)):
        if any(x != 0 for x in ev):
            continue
        nonzero = [i for i, x in enumerate(lv) if x != 0]
        if len(nonzero) != 1:
            continue
        loop_id = nonzero[0]
        if abs(lv[loop_id]) != 1:
            continue
        if loop_id not in lmb_edges:
            lmb_edges[loop_id] = edge_idx
        if len(lmb_edges) == n_loops:
            break
    return lmb_edges


def _build_ordered_external_half_edges(
    num_vertices: int,
    ext_names: Sequence[str],
    incoming_prefix: str = "p",
    outgoing_prefix: str = "q",
) -> List[Tuple[str, str, str]]:
    incoming_names: List[str] = []
    outgoing_names: List[str] = []
    incoming_set = set()
    outgoing_set = set()

    for name in ext_names:
        if incoming_prefix and name.startswith(incoming_prefix):
            incoming_names.append(name)
            incoming_set.add(name)
        elif outgoing_prefix and name.startswith(outgoing_prefix):
            outgoing_names.append(name)
            outgoing_set.add(name)
        else:
            outgoing_names.append(name)
            outgoing_set.add(name)

    plan: List[Tuple[str, str, str]] = []
    incoming_count = min(num_vertices, len(incoming_names))
    for node_id in range(incoming_count):
        name = incoming_names[node_id]
        plan.append(("ext", f"{node_id}:{node_id}", name))

    remaining = num_vertices - incoming_count
    outgoing_labels = list(outgoing_names)
    dependent_label = "0"
    if remaining > len(outgoing_labels):
        dependent_coeffs: List[int] = []
        for name in ext_names:
            if name in incoming_set:
                dependent_coeffs.append(1)
            elif name in outgoing_set:
                dependent_coeffs.append(-1)
            else:
                dependent_coeffs.append(-1)
        dependent_label = _format_lincomb(dependent_coeffs, ext_names)
        outgoing_labels.append(dependent_label)

    while len(outgoing_labels) < remaining:
        outgoing_labels.append(dependent_label)
    outgoing_labels = outgoing_labels[:remaining]

    for offset, label in enumerate(outgoing_labels):
        node_id = incoming_count + offset
        plan.append((f"{node_id}:{node_id}", "ext", label))
    return plan


def _match_external_slots_to_vertices(
    ext_balance: Sequence[Sequence[int]],
    slot_targets: Sequence[ExtVec],
) -> Optional[List[int]]:
    if not slot_targets:
        return []
    if len(ext_balance) < len(slot_targets):
        return None
    n_slots = len(slot_targets)
    balance_rows = [tuple(row) for row in ext_balance]
    candidates: List[List[int]] = []
    for target in slot_targets:
        candidates.append([v for v, row in enumerate(balance_rows) if row == target])
    if any(len(c) == 0 for c in candidates):
        return None

    order = sorted(range(n_slots), key=lambda s: len(candidates[s]))
    slot_to_vertex = [-1] * n_slots
    used = [False] * len(ext_balance)

    def backtrack(pos: int) -> bool:
        if pos == n_slots:
            return True
        slot_idx = order[pos]
        for v in candidates[slot_idx]:
            if used[v]:
                continue
            used[v] = True
            slot_to_vertex[slot_idx] = v
            if backtrack(pos + 1):
                return True
            slot_to_vertex[slot_idx] = -1
            used[v] = False
        return False

    if not backtrack(0):
        return None
    return slot_to_vertex


def reconstruct_dot(
    signatures: List[Signature],
    loop_names: Optional[List[str]] = None,
    ext_names: Optional[List[str]] = None,
    num_vertices: Optional[int] = None,
    require_connected: bool = True,
    max_degree: Optional[int] = None,
    edge_masses: Optional[List[str]] = None,
    emit_vakint_dot_format: bool = False,
    incoming_external_prefix: str = "p",
    outgoing_external_prefix: str = "q",
    graph_engine: Optional[str] = None,
    minimize_external_legs: bool = False,
) -> pydot.Dot:
    if not signatures:
        raise ValueError("Empty signature list.")

    L = len(signatures[0][0])
    Eext = len(signatures[0][1])

    for lv, ev in signatures:
        if len(lv) != L or len(ev) != Eext:
            raise ValueError("All loop/ext vectors must have consistent lengths.")
        _check_coeffs(lv)
        _check_coeffs(ev)

    n_edges = len(signatures)
    if edge_masses is not None and len(edge_masses) != n_edges:
        raise ValueError("edge_masses must have the same length as signatures.")

    if loop_names is None:
        loop_names = [f"k{i + 1}" for i in range(L)]
    if ext_names is None:
        ext_names = [f"p{j + 1}" for j in range(Eext)]
    if len(loop_names) != L or len(ext_names) != Eext:
        raise ValueError("loop_names/ext_names have wrong length.")

    # If connected and L is cyclomatic number: V = E - L + 1
    if num_vertices is None:
        guessed = n_edges - L + 1
        if guessed < 2:
            guessed = 2
        num_vertices = guessed

    V = num_vertices

    original_loop_vecs: List[LoopVec] = [sig[0] for sig in signatures]
    original_ext_vecs: List[ExtVec] = [sig[1] for sig in signatures]
    original_edge_masses: Optional[List[str]] = [str(m) for m in edge_masses] if edge_masses is not None else None

    if emit_vakint_dot_format:
        # Vakint-format reconstruction benefits from more aggressive ordering.
        order = sorted(
            range(n_edges),
            key=lambda i: (
                sum(abs(x) for x in original_loop_vecs[i]),
                sum(abs(x) for x in original_ext_vecs[i]),
                sum(1 for x in original_loop_vecs[i] if x),
                sum(1 for x in original_ext_vecs[i] if x),
            ),
            reverse=True,
        )
    else:
        # Keep historical ordering in the default mode to preserve legacy behavior.
        order = sorted(range(n_edges), key=lambda i: sum(abs(x) for x in original_loop_vecs[i]), reverse=True)
    loop_vecs = [original_loop_vecs[i] for i in order]
    ext_vecs = [original_ext_vecs[i] for i in order]
    if edge_masses is not None:
        edge_masses = [str(edge_masses[i]) for i in order]
    original_to_sorted = {original_idx: sorted_idx for sorted_idx, original_idx in enumerate(order)}

    # Precompute remaining abs sums per loop component for pruning
    rem_abs = [[0] * (n_edges + 1) for _ in range(L)]
    for i in range(L):
        for idx in range(n_edges - 1, -1, -1):
            rem_abs[i][idx] = rem_abs[i][idx + 1] + abs(loop_vecs[idx][i])
    rem_ext_abs = [[0] * (n_edges + 1) for _ in range(Eext)]
    for j in range(Eext):
        for idx in range(n_edges - 1, -1, -1):
            rem_ext_abs[j][idx] = rem_ext_abs[j][idx + 1] + abs(ext_vecs[idx][j])

    placements: List[Optional[EdgePlacement]] = [None] * n_edges
    imb = [[0] * L for _ in range(V)]  # loop imbalance per vertex: incoming - outgoing
    ext_imb = [[0] * Eext for _ in range(V)]  # external imbalance per vertex: incoming - outgoing
    deg = [0] * V  # undirected degree count

    required_external_targets: List[ExtVec] = []
    if emit_vakint_dot_format:
        ext_pos = {name: i for i, name in enumerate(ext_names)}
        incoming_names = [name for name in ext_names if incoming_external_prefix and name.startswith(incoming_external_prefix)]
        outgoing_names = [name for name in ext_names if outgoing_external_prefix and name.startswith(outgoing_external_prefix)]
        for name in incoming_names:
            if len(required_external_targets) >= V:
                break
            target = [0] * Eext
            target[ext_pos[name]] = -1
            required_external_targets.append(tuple(target))
        for name in outgoing_names:
            if len(required_external_targets) >= V:
                break
            target = [0] * Eext
            target[ext_pos[name]] = 1
            required_external_targets.append(tuple(target))

    # In legacy vakint output every internal vertex has one external half-edge,
    # so a total-valence cap translates to an internal-degree cap of max_degree - 1.
    # In minimized-external mode, keep the full cap during search and validate
    # total valence at the leaf when we know which vertices actually need an external.
    max_internal_degree = max_degree
    if emit_vakint_dot_format and max_degree is not None and not minimize_external_legs:
        max_internal_degree = max_degree - 1

    dsu = DSURollback(V)
    components = V
    pair_cache: Dict[int, List[Tuple[int, int]]] = {n: [(u, v) for u in range(n) for v in range(n) if u != v] for n in range(2, V + 1)}
    optimize_external_legs = emit_vakint_dot_format and minimize_external_legs
    best_external_vertices = V + 1
    best_placements: Optional[List[EdgePlacement]] = None
    minimum_external_vertices_lower_bound = 0
    if optimize_external_legs:
        minimum_external_vertices_lower_bound = len(required_external_targets)
        target_sum = [0] * Eext
        for target in required_external_targets:
            for j in range(Eext):
                target_sum[j] += target[j]
        # External balance over all vertices must sum to zero, so if mandatory
        # slots do not already do that, at least one extra external is required.
        if any(x != 0 for x in target_sum):
            minimum_external_vertices_lower_bound += 1

    def _minimum_possible_external_vertices(idx: int) -> int:
        forced_nonzero = 0
        for v in range(V):
            can_cancel_to_zero = True
            for j in range(Eext):
                if abs(ext_imb[v][j]) > rem_ext_abs[j][idx]:
                    can_cancel_to_zero = False
                    break
            if not can_cancel_to_zero:
                forced_nonzero += 1
        forced_nonzero = max(forced_nonzero, minimum_external_vertices_lower_bound)
        return forced_nonzero

    def prune_possible(idx: int, components_now: int) -> bool:
        # Connectivity prune
        remaining = n_edges - idx
        if require_connected and remaining < (components_now - 1):
            return True

        # Loop imbalance prune
        for v in range(V):
            for i in range(L):
                if abs(imb[v][i]) > rem_abs[i][idx]:
                    return True

        # Degree prune
        if max_internal_degree is not None:
            for v in range(V):
                if deg[v] > max_internal_degree:
                    return True

        # Optimization prune: once we have a solution, abandon branches that
        # cannot beat the current best external-leg count.
        if optimize_external_legs and best_external_vertices <= V:
            if _minimum_possible_external_vertices(idx) >= best_external_vertices:
                return True

        return False

    def backtrack(idx: int, components_now: int, used_vertices_now: int) -> bool:
        nonlocal best_external_vertices
        nonlocal best_placements
        if prune_possible(idx, components_now):
            return False
        if idx == n_edges:
            # Must have zero loop imbalance everywhere
            for v in range(V):
                if any(imb[v][i] != 0 for i in range(L)):
                    return False
            if require_connected and components_now != 1:
                return False
            final_ext_balance = ext_imb
            if emit_vakint_dot_format and max_degree is not None and minimize_external_legs:
                for v in range(V):
                    has_external = any(final_ext_balance[v][j] != 0 for j in range(Eext))
                    if deg[v] + (1 if has_external else 0) > max_degree:
                        return False
            if required_external_targets:
                if _match_external_slots_to_vertices(final_ext_balance, required_external_targets) is None:
                    return False

            if optimize_external_legs:
                nonzero_external_vertices = sum(1 for v in range(V) if any(final_ext_balance[v][j] != 0 for j in range(Eext)))
                if nonzero_external_vertices < best_external_vertices:
                    best_external_vertices = nonzero_external_vertices
                    best_placements = []
                    for pl in placements:
                        assert pl is not None
                        best_placements.append(pl)
                # This is the theoretical lower bound in vakint mode, so we can stop.
                return best_external_vertices == minimum_external_vertices_lower_bound

            return True

        vec = loop_vecs[idx]
        ex = ext_vecs[idx]

        # Symmetry breaking: fix first edge to connect (0 -> 1) if possible
        if idx == 0:
            cand_pairs = [(0, 1)]
        else:
            frontier_size = used_vertices_now + (1 if used_vertices_now < V else 0)
            cand_pairs = pair_cache[frontier_size]

        for tail, head in cand_pairs:
            # Apply placement: tail -> head carries vec
            snap = dsu.snapshot()
            old_components = components_now
            merged = dsu.union(tail, head)
            if merged:
                components_now -= 1

            placements[idx] = EdgePlacement(tail=tail, head=head)

            # Update imbalances (incoming - outgoing)
            for i in range(L):
                imb[tail][i] -= vec[i]
                imb[head][i] += vec[i]
            for j in range(Eext):
                ext_imb[tail][j] -= ex[j]
                ext_imb[head][j] += ex[j]

            # Update degrees
            deg[tail] += 1
            deg[head] += 1

            next_used_vertices = used_vertices_now
            if used_vertices_now < V and (tail == used_vertices_now or head == used_vertices_now):
                next_used_vertices += 1

            if backtrack(idx + 1, components_now, next_used_vertices):
                return True

            # Undo
            deg[tail] -= 1
            deg[head] -= 1
            for i in range(L):
                imb[tail][i] += vec[i]
                imb[head][i] -= vec[i]
            for j in range(Eext):
                ext_imb[tail][j] += ex[j]
                ext_imb[head][j] -= ex[j]
            placements[idx] = None

            dsu.rollback(snap)
            components_now = old_components

        return False

    search_found = backtrack(0, components, min(V, 2))
    if optimize_external_legs:
        if best_placements is not None:
            placements = list(best_placements)
        ok = best_placements is not None
    else:
        ok = search_found
    if not ok:
        raise RuntimeError(
            "No realization found under current assumptions/constraints. "
            "Try adjusting num_vertices, disabling require_connected, or relaxing max_degree."
        )

    # Undo edge ordering to match original indices for labeling if desired
    # For DOT we will keep an internal sequential e0,e1,... in the found (ordered) solution,
    # but label includes the momentum expression, which is what matters.

    # Compute external imbalance at each vertex from internal edges:
    # ext_balance[v][j] = sum_in x - sum_out x
    ext_balance = [[0] * Eext for _ in range(V)]
    for idx in range(n_edges):
        pl = placements[idx]
        assert pl is not None
        tail, head = pl.tail, pl.head
        ex = ext_vecs[idx]
        for j in range(Eext):
            ext_balance[tail][j] -= ex[j]
            ext_balance[head][j] += ex[j]

    if emit_vakint_dot_format:
        g = pydot.Dot("G", graph_type="digraph", strict=False)
        g.set_layout(graph_engine or "dot")

        # Keep a single hub node for all half-edges.
        g.add_node(pydot.Node("ext", style="invis"))

        ext_pos = {name: i for i, name in enumerate(ext_names)}
        incoming_names = [name for name in ext_names if incoming_external_prefix and name.startswith(incoming_external_prefix)]
        outgoing_names = [name for name in ext_names if outgoing_external_prefix and name.startswith(outgoing_external_prefix)]

        mandatory_slots: List[Tuple[str, str, str, ExtVec]] = []
        slot_idx = 0
        for name in incoming_names:
            if slot_idx >= V:
                break
            target = [0] * Eext
            target[ext_pos[name]] = -1
            mandatory_slots.append(("ext", f"{slot_idx}:{slot_idx}", name, tuple(target)))
            slot_idx += 1
        for name in outgoing_names:
            if slot_idx >= V:
                break
            target = [0] * Eext
            target[ext_pos[name]] = 1
            mandatory_slots.append((f"{slot_idx}:{slot_idx}", "ext", name, tuple(target)))
            slot_idx += 1

        mandatory_targets = [slot[3] for slot in mandatory_slots]
        slot_to_vertex = _match_external_slots_to_vertices(ext_balance, mandatory_targets)
        if slot_to_vertex is None:
            slot_to_vertex = []
            used_vertices = set()
            for _src, _dst, _label, target in mandatory_slots:
                found = None
                for v, row in enumerate(ext_balance):
                    if v in used_vertices:
                        continue
                    if tuple(row) == target:
                        found = v
                        break
                if found is None:
                    for v in range(V):
                        if v not in used_vertices:
                            found = v
                            break
                assert found is not None
                slot_to_vertex.append(found)
                used_vertices.add(found)
        used_vertices = set(slot_to_vertex)

        external_plan: List[Tuple[str, str, str]] = [(src, dst, label) for src, dst, label, _target in mandatory_slots]
        for old_vertex in range(V):
            if old_vertex in used_vertices:
                continue
            new_slot = len(slot_to_vertex)
            slot_to_vertex.append(old_vertex)
            ext_vec = tuple(ext_balance[old_vertex])
            if minimize_external_legs and not any(ext_vec):
                continue
            external_plan.append((f"{new_slot}:{new_slot}", "ext", _format_lincomb(ext_vec, ext_names)))

        old_to_slot = {old_vertex: slot for slot, old_vertex in enumerate(slot_to_vertex)}

        # Keep numeric node names to support "n:n" port syntax in edge endpoints.
        for v in range(V):
            g.add_node(
                pydot.Node(
                    str(v),
                    fontsize="4",
                    shape='"circle"',
                    width="0.1",
                    height="0.1",
                    fixedsize="true",
                )
            )

        lmb_edges = _find_lmb_edge_indices(original_loop_vecs, original_ext_vecs)
        edge_to_lmb = {edge_idx: loop_id for loop_id, edge_idx in lmb_edges.items()}

        next_edge_id = 0
        for src, dst, ext_label in external_plan:
            normalized_ext_label = _normalize_momentum_label(ext_label)
            attrs: Dict[str, str] = {
                "id": str(next_edge_id),
                "mass": "0",
                "label": _dot_quote_label(normalized_ext_label),
                "fontsize": "10",
            }
            g.add_edge(pydot.Edge(src, dst, **attrs))
            next_edge_id += 1

        for idx in range(n_edges):
            sorted_idx = original_to_sorted[idx]
            pl = placements[sorted_idx]
            assert pl is not None
            mom = _normalize_momentum_label(format_momentum(original_loop_vecs[idx], original_ext_vecs[idx], loop_names, ext_names))
            mass_value = original_edge_masses[idx] if original_edge_masses is not None else "0"
            edge_attrs: Dict[str, str] = {
                "label": _dot_quote_label(mom),
                "fontsize": "10",
                "mass": mass_value,
                "id": str(next_edge_id),
            }
            if idx in edge_to_lmb:
                edge_attrs["lmb_id"] = f'"{edge_to_lmb[idx]}"'
            g.add_edge(
                pydot.Edge(
                    str(old_to_slot.get(pl.tail, pl.tail)),
                    str(old_to_slot.get(pl.head, pl.head)),
                    **edge_attrs,
                )
            )
            next_edge_id += 1
        return g

    # Build DOT graph
    g = pydot.Dot("G", graph_type="digraph", strict=False, rankdir="LR")
    if graph_engine is not None:
        g.set_layout(graph_engine)

    node_masses: Optional[List[List[str]]] = None
    if edge_masses is not None:
        node_masses = [[] for _ in range(V)]
        for idx in range(n_edges):
            pl = placements[idx]
            assert pl is not None
            mass = edge_masses[idx]
            node_masses[pl.tail].append(mass)
            node_masses[pl.head].append(mass)

    # Internal vertices
    for v in range(V):
        attrs: Dict[str, str] = {"label": f"v{v}", "shape": "circle"}
        if node_masses is not None:
            attrs["mass"] = ",".join(node_masses[v])
        g.add_node(pydot.Node(f"v{v}", **attrs))

    # External momentum nodes (one per basis momentum)
    ext_nodes = []
    for j, name in enumerate(ext_names):
        n = pydot.Node(f"ext_{name}", label=name, shape="box")
        g.add_node(n)
        ext_nodes.append(n)

    # Internal edges
    for idx in range(n_edges):
        pl = placements[idx]
        assert pl is not None
        mom = format_momentum(loop_vecs[idx], ext_vecs[idx], loop_names, ext_names)
        edge_attrs: Dict[str, str] = {
            "label": mom,
            "fontsize": "10",
        }
        if edge_masses is not None:
            edge_attrs["mass"] = edge_masses[idx]
        g.add_edge(
            pydot.Edge(
                f"v{pl.tail}",
                f"v{pl.head}",
                **edge_attrs,
            )
        )

    # Attach external legs to satisfy conservation at each vertex:
    # If ext_balance[v][j] = +n, add n legs carrying "-p_j" (in all-incoming convention).
    # If ext_balance[v][j] = -n, add n legs carrying "+p_j".
    for v in range(V):
        for j, name in enumerate(ext_names):
            b = ext_balance[v][j]
            if b == 0:
                continue
            if b > 0:
                # need -b incoming => b outgoing p => label "-p"
                for k in range(b):
                    g.add_edge(
                        pydot.Edge(
                            f"v{v}",
                            f"ext_{name}",
                            label=f"-{name}",
                            fontsize="10",
                            style="dashed",
                        )
                    )
            else:
                # b < 0 => need +(-b) incoming p => edge ext -> v label "+p"
                for k in range(-b):
                    g.add_edge(
                        pydot.Edge(
                            f"ext_{name}",
                            f"v{v}",
                            label=f"{name}",
                            fontsize="10",
                            style="dashed",
                        )
                    )

    return g


def _strip_quotes(s: Optional[str]) -> str:
    if s is None:
        return ""
    out = s.strip()
    if len(out) >= 2 and ((out[0] == '"' and out[-1] == '"') or (out[0] == "'" and out[-1] == "'")):
        out = out[1:-1]
    return out


def _find_tree_path(num_vertices: int, tree_edges: Sequence[Tuple[int, int]], start: int, goal: int) -> List[Tuple[int, int]]:
    if start == goal:
        return []

    adj: List[List[Tuple[int, int, int]]] = [[] for _ in range(num_vertices)]
    for idx, (tail, head) in enumerate(tree_edges):
        # sign = +1 means traversing in the stored orientation tail->head
        adj[tail].append((head, idx, +1))
        adj[head].append((tail, idx, -1))

    parent: List[Optional[Tuple[int, int, int]]] = [None] * num_vertices
    seen = [False] * num_vertices
    queue = [start]
    seen[start] = True

    q_idx = 0
    while q_idx < len(queue):
        cur = queue[q_idx]
        q_idx += 1
        if cur == goal:
            break
        for nxt, edge_idx, sign in adj[cur]:
            if seen[nxt]:
                continue
            seen[nxt] = True
            parent[nxt] = (cur, edge_idx, sign)
            queue.append(nxt)

    if not seen[goal]:
        raise ValueError("Tree path not found; tree edges are not connected.")

    path: List[Tuple[int, int]] = []
    cur = goal
    while cur != start:
        prev, edge_idx, sign = parent[cur]  # type: ignore[misc]
        path.append((edge_idx, sign))
        cur = prev
    path.reverse()
    return path


def signatures_from_graph_case(case: GraphCase) -> List[Signature]:
    V = case.num_vertices
    E = len(case.edges)
    L = len(case.basis_edge_indices)
    Eext = len(case.ext_names)

    if E != (V - 1 + L):
        raise ValueError("Expected E = V - 1 + L for connected graph with chosen loop basis.")
    if len(case.vertex_potentials) != V:
        raise ValueError("vertex_potentials must provide one row per internal vertex.")
    for row in case.vertex_potentials:
        if len(row) != Eext:
            raise ValueError("Each vertex potential row must match number of external basis momenta.")
        for x in row:
            if x not in (0, 1):
                raise ValueError("vertex_potentials entries must be 0 or 1.")

    basis_set = set(case.basis_edge_indices)
    if len(basis_set) != L:
        raise ValueError("basis_edge_indices must be unique.")

    tree_indices = [i for i in range(E) if i not in basis_set]
    if len(tree_indices) != V - 1:
        raise ValueError("Non-basis edges must form a spanning tree (exactly V-1 edges).")

    tree_edges = [case.edges[i] for i in tree_indices]
    loop_coeffs = [[0] * L for _ in range(E)]

    for loop_id, edge_idx in enumerate(case.basis_edge_indices):
        tail, head = case.edges[edge_idx]
        loop_coeffs[edge_idx][loop_id] = 1

        # Fundamental cycle flow: compensate basis edge tail->head with tree flow head->tail.
        path = _find_tree_path(V, tree_edges, head, tail)
        for tree_local_idx, sign in path:
            global_idx = tree_indices[tree_local_idx]
            loop_coeffs[global_idx][loop_id] = sign

    signatures: List[Signature] = []
    for edge_idx, (tail, head) in enumerate(case.edges):
        lv = tuple(loop_coeffs[edge_idx])
        ev = tuple(case.vertex_potentials[tail][j] - case.vertex_potentials[head][j] for j in range(Eext))
        _check_coeffs(lv)
        _check_coeffs(ev)
        signatures.append((lv, ev))
    return signatures


def dot_from_graph_case(case: GraphCase, signatures: Sequence[Signature]) -> pydot.Dot:
    if len(signatures) != len(case.edges):
        raise ValueError("signatures must match number of edges.")

    g = pydot.Dot("G", graph_type="digraph", strict=False, rankdir="LR")
    V = case.num_vertices
    Eext = len(case.ext_names)

    for v in range(V):
        g.add_node(pydot.Node(f"v{v}", label=f"v{v}", shape="circle"))

    for name in case.ext_names:
        g.add_node(pydot.Node(f"ext_{name}", label=name, shape="box"))

    ext_balance = [[0] * Eext for _ in range(V)]
    for (tail, head), (loop_vec, ext_vec) in zip(case.edges, signatures):
        mom = format_momentum(loop_vec, ext_vec, case.loop_names, case.ext_names)
        g.add_edge(
            pydot.Edge(
                f"v{tail}",
                f"v{head}",
                label=mom,
                fontsize="10",
            )
        )
        for j in range(Eext):
            ext_balance[tail][j] -= ext_vec[j]
            ext_balance[head][j] += ext_vec[j]

    for v in range(V):
        for j, name in enumerate(case.ext_names):
            b = ext_balance[v][j]
            if b > 0:
                for _ in range(b):
                    g.add_edge(
                        pydot.Edge(
                            f"v{v}",
                            f"ext_{name}",
                            label=f"-{name}",
                            fontsize="10",
                            style="dashed",
                        )
                    )
            elif b < 0:
                for _ in range(-b):
                    g.add_edge(
                        pydot.Edge(
                            f"ext_{name}",
                            f"v{v}",
                            label=name,
                            fontsize="10",
                            style="dashed",
                        )
                    )

    return g


def _dot_signature(dot: pydot.Dot) -> Tuple[List[str], List[str], Counter[Tuple[str, str, str, str]]]:
    nodes = set()
    edges: Counter[Tuple[str, str, str, str]] = Counter()

    for n in dot.get_nodes():
        name = _strip_quotes(n.get_name())
        if name and name not in {"graph", "node", "edge"}:
            nodes.add(name)

    for e in dot.get_edges():
        src = _strip_quotes(e.get_source())
        dst = _strip_quotes(e.get_destination())
        label = _strip_quotes(e.get("label"))
        style = _strip_quotes(e.get("style"))
        nodes.add(src)
        nodes.add(dst)
        edges[(src, dst, label, style)] += 1

    internal = sorted(n for n in nodes if n.startswith("v"))
    external = sorted(n for n in nodes if n.startswith("ext_"))
    return internal, external, edges


def dots_are_isomorphic(expected: pydot.Dot, actual: pydot.Dot) -> bool:
    exp_internal, exp_external, exp_edges = _dot_signature(expected)
    act_internal, act_external, act_edges = _dot_signature(actual)

    if exp_external != act_external:
        return False
    if len(exp_internal) != len(act_internal):
        return False

    for perm in itertools.permutations(act_internal):
        mapping = dict(zip(exp_internal, perm))
        mapped_edges: Counter[Tuple[str, str, str, str]] = Counter()
        for (src, dst, label, style), cnt in exp_edges.items():
            mapped_src = mapping.get(src, src)
            mapped_dst = mapping.get(dst, dst)
            mapped_edges[(mapped_src, mapped_dst, label, style)] += cnt
        if mapped_edges == act_edges:
            return True
    return False


def _parse_momentum_label(label: str, loop_names: Sequence[str], ext_names: Sequence[str]) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    loop_pos = {name: i for i, name in enumerate(loop_names)}
    ext_pos = {name: j for j, name in enumerate(ext_names)}
    lv = [0] * len(loop_names)
    ev = [0] * len(ext_names)

    text = label.strip()
    if not text or text == "0":
        return tuple(lv), tuple(ev)

    expr = text.replace(" ", "")
    while True:
        normalized = expr.replace("+-", "-").replace("-+", "-").replace("--", "+").replace("++", "+")
        if normalized == expr:
            break
        expr = normalized

    if not expr:
        return tuple(lv), tuple(ev)
    if expr[0] not in "+-":
        expr = "+" + expr

    idx = 0
    while idx < len(expr):
        sign_char = expr[idx]
        if sign_char not in "+-":
            raise ValueError(f"Unexpected token in label '{label}'.")
        sign = 1 if sign_char == "+" else -1
        idx += 1

        start = idx
        while idx < len(expr) and expr[idx] not in "+-":
            idx += 1
        tok = expr[start:idx]
        if not tok:
            raise ValueError(f"Malformed momentum label '{label}'.")

        if tok in loop_pos:
            lv[loop_pos[tok]] += sign
        elif tok in ext_pos:
            ev[ext_pos[tok]] += sign
        else:
            raise ValueError(f"Unrecognized momentum term '{tok}' in label '{label}'.")

    _check_coeffs(lv)
    _check_coeffs(ev)
    return tuple(lv), tuple(ev)


def dot_is_valid_realization(
    dot: pydot.Dot,
    signatures: Sequence[Signature],
    loop_names: Sequence[str],
    ext_names: Sequence[str],
    num_vertices: int,
    require_connected: bool = True,
) -> bool:
    internal, external, _ = _dot_signature(dot)
    if len(internal) != num_vertices:
        return False
    if sorted(external) != sorted(f"ext_{name}" for name in ext_names):
        return False

    # Track balances implied by internal edges.
    v_index = {f"v{i}": i for i in range(num_vertices)}
    balance_loop = [[0] * len(loop_names) for _ in range(num_vertices)]
    internal_ext_balance = [[0] * len(ext_names) for _ in range(num_vertices)]

    internal_edge_labels: Counter[str] = Counter()
    internal_edges: List[Tuple[int, int]] = []
    external_leg_counts: Counter[Tuple[str, int, int]] = Counter()
    # key: (direction, vertex, ext_idx), direction in {"out_neg", "in_pos"}

    for e in dot.get_edges():
        src = _strip_quotes(e.get_source())
        dst = _strip_quotes(e.get_destination())
        label = _strip_quotes(e.get("label"))

        src_is_internal = src in v_index
        dst_is_internal = dst in v_index
        if not src_is_internal and not dst_is_internal:
            return False

        try:
            lv, ev = _parse_momentum_label(label, loop_names, ext_names)
        except ValueError:
            return False

        if src_is_internal and dst_is_internal:
            internal_edge_labels[label] += 1
            internal_edges.append((v_index[src], v_index[dst]))
            s = v_index[src]
            t = v_index[dst]
            for i, x in enumerate(lv):
                balance_loop[s][i] -= x
                balance_loop[t][i] += x
            for j, x in enumerate(ev):
                internal_ext_balance[s][j] -= x
                internal_ext_balance[t][j] += x
        else:
            # External legs should not carry loop momentum.
            if any(x != 0 for x in lv):
                return False
            nonzero_ext = [j for j, x in enumerate(ev) if x != 0]
            if len(nonzero_ext) != 1:
                return False
            j = nonzero_ext[0]
            coeff = ev[j]
            name = ext_names[j]

            if src_is_internal and dst == f"ext_{name}" and coeff == -1:
                external_leg_counts[("out_neg", v_index[src], j)] += 1
            elif dst_is_internal and src == f"ext_{name}" and coeff == 1:
                external_leg_counts[("in_pos", v_index[dst], j)] += 1
            else:
                return False

    if len(internal_edges) != len(signatures):
        return False

    expected_labels = Counter(format_momentum(lv, ev, loop_names, ext_names) for lv, ev in signatures)
    if internal_edge_labels != expected_labels:
        return False

    for v in range(num_vertices):
        if any(balance_loop[v][i] != 0 for i in range(len(loop_names))):
            return False

        for j in range(len(ext_names)):
            b = internal_ext_balance[v][j]
            out_neg = external_leg_counts[("out_neg", v, j)]
            in_pos = external_leg_counts[("in_pos", v, j)]
            if b > 0 and not (out_neg == b and in_pos == 0):
                return False
            if b < 0 and not (in_pos == -b and out_neg == 0):
                return False
            if b == 0 and not (out_neg == 0 and in_pos == 0):
                return False

    # Internal graph consistency checks.
    parent = list(range(num_vertices))

    def find(a: int) -> int:
        while parent[a] != a:
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in internal_edges:
        union(a, b)

    components = len({find(i) for i in range(num_vertices)})
    if require_connected and components != 1:
        return False

    loops = len(internal_edges) - num_vertices + components
    if loops != len(loop_names):
        return False

    return True


def _max_undirected_degree(num_vertices: int, edges: Sequence[Tuple[int, int]]) -> int:
    deg = [0] * num_vertices
    for tail, head in edges:
        deg[tail] += 1
        deg[head] += 1
    return max(deg) if deg else 0


def _handcrafted_5_loop_case() -> GraphCase:
    return GraphCase(
        num_vertices=6,
        edges=(
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (1, 0),
            (2, 0),
            (4, 1),
            (5, 2),
            (3, 0),
        ),
        basis_edge_indices=(5, 6, 7, 8, 9),
        vertex_potentials=(
            (1, 0, 1, 0),
            (0, 1, 1, 0),
            (1, 1, 0, 1),
            (0, 0, 1, 1),
            (1, 0, 0, 1),
            (0, 1, 0, 0),
        ),
        loop_names=("k1", "k2", "k3", "k4", "k5"),
        ext_names=("p1", "p2", "p3", "p4"),
    )


def _handcrafted_6_loop_case() -> GraphCase:
    return GraphCase(
        num_vertices=6,
        edges=(
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 0),
            (2, 0),
            (4, 1),
            (5, 2),
            (3, 0),
            (1, 4),
        ),
        basis_edge_indices=(5, 6, 7, 8, 9, 10),
        vertex_potentials=(
            (1, 0, 1, 0, 1),
            (0, 1, 1, 0, 0),
            (1, 1, 0, 1, 0),
            (0, 0, 1, 1, 1),
            (1, 0, 0, 1, 1),
            (0, 1, 0, 0, 1),
        ),
        loop_names=("k1", "k2", "k3", "k4", "k5", "k6"),
        ext_names=("p1", "p2", "p3", "p4", "p5"),
    )


def _generate_random_5_loop_case(rng: random.Random) -> GraphCase:
    loops = 5
    V = 6
    Eext = 5

    for _ in range(200):
        tree_edges: List[Tuple[int, int]] = []
        tree_undirected = set()

        for v in range(1, V):
            parent = rng.randrange(v)
            a, b = sorted((v, parent))
            tree_undirected.add((a, b))
            if rng.random() < 0.5:
                tree_edges.append((parent, v))
            else:
                tree_edges.append((v, parent))

        candidates = []
        for a in range(V):
            for b in range(a + 1, V):
                if (a, b) not in tree_undirected:
                    candidates.append((a, b))
        chord_pairs = rng.sample(candidates, loops)

        chords: List[Tuple[int, int]] = []
        for a, b in chord_pairs:
            if rng.random() < 0.5:
                chords.append((a, b))
            else:
                chords.append((b, a))

        edges = tuple(tree_edges + chords)
        basis = tuple(range(V - 1, V - 1 + loops))

        potentials: List[List[int]] = [[rng.randrange(2) for _ in range(Eext)] for _ in range(V)]
        # Make each external basis momentum active on at least one edge.
        for j in range(Eext):
            col = [potentials[v][j] for v in range(V)]
            if len(set(col)) == 1:
                v = rng.randrange(V)
                potentials[v][j] = 1 - potentials[v][j]

        case = GraphCase(
            num_vertices=V,
            edges=edges,
            basis_edge_indices=basis,
            vertex_potentials=tuple(tuple(row) for row in potentials),
            loop_names=tuple(f"k{i + 1}" for i in range(loops)),
            ext_names=tuple(f"p{j + 1}" for j in range(Eext)),
        )
        sigs = signatures_from_graph_case(case)
        if len(set(sigs)) == len(sigs):
            return case

    raise RuntimeError("Failed to generate a sufficiently distinctive random 5-loop case.")


class ReconstructDotTests(unittest.TestCase):
    def _assert_reconstructs_to_isomorphic_dot(self, case: GraphCase) -> None:
        signatures = signatures_from_graph_case(case)
        expected = dot_from_graph_case(case, signatures)
        reconstructed = reconstruct_dot(
            signatures,
            loop_names=list(case.loop_names),
            ext_names=list(case.ext_names),
            num_vertices=case.num_vertices,
            require_connected=True,
            max_degree=_max_undirected_degree(case.num_vertices, case.edges),
        )
        self.assertTrue(
            dots_are_isomorphic(expected, reconstructed),
            msg=(
                "Reconstructed graph is not isomorphic to source graph.\n"
                f"Expected:\n{expected.to_string()}\n"
                f"Reconstructed:\n{reconstructed.to_string()}"
            ),
        )

    def test_sunrise_example(self) -> None:
        case = GraphCase(
            num_vertices=2,
            edges=((0, 1), (1, 0), (0, 1)),
            basis_edge_indices=(0, 1),
            vertex_potentials=((1,), (0,)),
            loop_names=("k", "l"),
            ext_names=("p",),
        )
        self._assert_reconstructs_to_isomorphic_dot(case)

    def test_symbolica_style_extraction_from_bare_expression(self) -> None:
        expr = "prop(k1+p1,0)*prop(k1+p1-q1,0)*prop(k1-p2+q2,0)*prop(k1-p2,0)*prop(k1,m1)"
        signatures, loop_names, ext_names, masses = extract_signatures_and_masses_from_symbolica_expression(
            expr,
            loop_prefix="k",
            external_prefixes=("p", "q"),
        )
        self.assertEqual(loop_names, ["k1"])
        self.assertEqual(ext_names, ["p1", "p2", "q1", "q2"])
        self.assertEqual(masses, ["0", "0", "0", "0", "m1"])
        self.assertEqual(
            signatures,
            [
                ((1,), (1, 0, 0, 0)),
                ((1,), (1, 0, -1, 0)),
                ((1,), (0, -1, 0, 1)),
                ((1,), (0, -1, 0, 0)),
                ((1,), (0, 0, 0, 0)),
            ],
        )

    def test_symbolica_style_extraction_two_loop_from_bare_expression(self) -> None:
        expr = "prop(k1+p1,0)*prop(k1+k2+p1,0)*prop(k1+k2+p1-q1,0)*prop(k1+k2-p2+q2,0)*prop(k1+k2-p2,0)*prop(k1-p2,0)*prop(k1,m1)*prop(k2,m1)"
        signatures, loop_names, ext_names, masses = extract_signatures_and_masses_from_symbolica_expression(
            expr,
            loop_prefix="k",
            external_prefixes=("p", "q"),
        )
        self.assertEqual(loop_names, ["k1", "k2"])
        self.assertEqual(ext_names, ["p1", "p2", "q1", "q2"])
        self.assertEqual(masses, ["0", "0", "0", "0", "0", "0", "m1", "m1"])
        self.assertEqual(len(signatures), 8)

    def test_reconstruct_dot_from_symbolica_keeps_mass_attributes(self) -> None:
        expr = "prop(k1+p1,0)*prop(k1+p1-q1,0)*prop(k1-p2+q2,0)*prop(k1-p2,0)*prop(k1,m1)"
        dot, signatures, _loop_names, _ext_names, masses = reconstruct_dot_from_symbolica_expression(
            expr,
            loop_prefix="k",
            external_prefixes=("p", "q"),
        )

        internal_edges = []
        for e in dot.get_edges():
            src = _strip_quotes(e.get_source())
            dst = _strip_quotes(e.get_destination())
            if src.startswith("v") and dst.startswith("v"):
                internal_edges.append(e)

        self.assertEqual(len(internal_edges), len(signatures))
        self.assertEqual(
            sorted(_strip_quotes(e.get("mass")) for e in internal_edges),
            sorted(masses),
        )

    def test_reconstruct_dot_stores_mass_attributes(self) -> None:
        signatures = [
            ((1, 0), (0,)),
            ((0, 1), (0,)),
            ((1, 1), (1,)),
        ]
        masses = ["mA", "mB", "mC"]
        dot = reconstruct_dot(
            signatures,
            loop_names=["k", "l"],
            ext_names=["p"],
            require_connected=True,
            edge_masses=masses,
        )

        internal_edges = []
        for e in dot.get_edges():
            src = _strip_quotes(e.get_source())
            dst = _strip_quotes(e.get_destination())
            if src.startswith("v") and dst.startswith("v"):
                internal_edges.append(e)
        self.assertEqual(len(internal_edges), len(masses))
        self.assertEqual(
            sorted(_strip_quotes(e.get("mass")) for e in internal_edges),
            sorted(masses),
        )

        internal_nodes = [n for n in dot.get_nodes() if _strip_quotes(n.get_name()).startswith("v")]
        self.assertTrue(internal_nodes)
        for n in internal_nodes:
            self.assertIsNotNone(n.get("mass"))

    def test_unsatisfiable_signatures_raise_clean_error(self) -> None:
        signatures = [
            ((-1, 1), ()),
            ((0, 1), ()),
            ((1, 1), ()),
            ((-1, 0), ()),
        ]
        with self.assertRaisesRegex(RuntimeError, "No realization found under current assumptions/constraints"):
            reconstruct_dot(
                signatures,
                loop_names=["k1", "k2"],
                ext_names=[],
                require_connected=True,
            )

    def test_vakint_total_valence_cap_rejects_pentagon_at_2(self) -> None:
        expr = "prop(k1+p1,0)*prop(k1+p1-q1,0)*prop(k1-p2+q2,0)*prop(k1-p2,0)*prop(k1,m1)"
        signatures, loop_names, ext_names, masses = extract_signatures_and_masses_from_symbolica_expression(
            expr,
            loop_prefix="k",
            external_prefixes=("p", "q"),
        )
        with self.assertRaisesRegex(RuntimeError, "No realization found under current assumptions/constraints"):
            reconstruct_dot(
                signatures,
                loop_names=list(loop_names),
                ext_names=list(ext_names),
                edge_masses=masses,
                max_degree=2,
                emit_vakint_dot_format=True,
                incoming_external_prefix="p",
                outgoing_external_prefix="q",
                graph_engine="dot",
            )

    def test_reconstruct_dot_vakint_format_ordered_external_ids_and_lmb(self) -> None:
        expr = "prop(k1+p1,0)*prop(k1+p1-q1,0)*prop(k1-p2+q2,0)*prop(k1-p2,0)*prop(k1,m1)"
        signatures, loop_names, ext_names, masses = extract_signatures_and_masses_from_symbolica_expression(
            expr,
            loop_prefix="k",
            external_prefixes=("p", "q"),
        )
        dot = reconstruct_dot(
            signatures,
            loop_names=list(loop_names),
            ext_names=list(ext_names),
            edge_masses=masses,
            emit_vakint_dot_format=True,
            incoming_external_prefix="p",
            outgoing_external_prefix="q",
            graph_engine="dot",
        )

        self.assertEqual(_strip_quotes(dot.get("layout")), "dot")
        ext_node = next(n for n in dot.get_nodes() if _strip_quotes(n.get_name()) == "ext")
        self.assertEqual(_strip_quotes(ext_node.get("style")), "invis")
        internal_node_zero = next(n for n in dot.get_nodes() if _strip_quotes(n.get_name()) == "0")
        self.assertEqual(_strip_quotes(internal_node_zero.get("fontsize")), "4")
        self.assertEqual(_strip_quotes(internal_node_zero.get("shape")), "circle")
        self.assertEqual(_strip_quotes(internal_node_zero.get("width")), "0.1")
        self.assertEqual(_strip_quotes(internal_node_zero.get("height")), "0.1")
        self.assertEqual(_strip_quotes(internal_node_zero.get("fixedsize")), "true")

        edges_by_id = sorted(dot.get_edges(), key=lambda e: int(_strip_quotes(e.get("id"))))
        self.assertEqual([int(_strip_quotes(e.get("id"))) for e in edges_by_id], list(range(len(edges_by_id))))

        external_edges = []
        internal_edges = []
        for e in edges_by_id:
            src = _strip_quotes(e.get_source())
            dst = _strip_quotes(e.get_destination())
            src_base = src.split(":")[0]
            dst_base = dst.split(":")[0]
            if src_base == "ext" or dst_base == "ext":
                external_edges.append(e)
            else:
                internal_edges.append(e)

        self.assertEqual(len(external_edges), 5)
        self.assertEqual(_strip_quotes(external_edges[0].get_source()), "ext")
        self.assertEqual(_strip_quotes(external_edges[0].get_destination()), "0:0")
        self.assertEqual(_strip_quotes(external_edges[1].get_source()), "ext")
        self.assertEqual(_strip_quotes(external_edges[1].get_destination()), "1:1")
        for e in external_edges[2:]:
            self.assertEqual(_strip_quotes(e.get_destination()), "ext")
        for e in edges_by_id:
            self.assertIsNotNone(e.get("mass"))

        lmb_edges = [e for e in internal_edges if e.get("lmb_id") is not None]
        self.assertEqual(len(lmb_edges), 1)
        self.assertEqual(_strip_quotes(lmb_edges[0].get("lmb_id")), "0")
        self.assertEqual(_strip_quotes(lmb_edges[0].get("label")), "k1")

        p1_node = _strip_quotes(external_edges[0].get_destination()).split(":")[0]
        p2_node = _strip_quotes(external_edges[1].get_destination()).split(":")[0]
        q1_node = _strip_quotes(external_edges[2].get_source()).split(":")[0]
        q2_node = _strip_quotes(external_edges[3].get_source()).split(":")[0]

        edge_k1_p1 = next(e for e in internal_edges if _strip_quotes(e.get("label")) == "k1+p1")
        self.assertEqual(_strip_quotes(edge_k1_p1.get_source()), p1_node)
        self.assertEqual(_strip_quotes(edge_k1_p1.get_destination()), q1_node)

        edge_k1_p1_q1 = next(e for e in internal_edges if _strip_quotes(e.get("label")) == "k1+p1-q1")
        self.assertEqual(_strip_quotes(edge_k1_p1_q1.get_source()), q1_node)
        self.assertNotEqual(_strip_quotes(edge_k1_p1_q1.get_source()), p2_node)

        edge_k1_p2_q2 = next(e for e in internal_edges if _strip_quotes(e.get("label")) == "k1-p2+q2")
        self.assertEqual(_strip_quotes(edge_k1_p2_q2.get_destination()), q2_node)

        internal_with_ext = [e for e in internal_edges if "p2" in _strip_quotes(e.get("label"))]
        self.assertTrue(internal_with_ext)
        for e in internal_with_ext:
            self.assertNotIn("+ -", _strip_quotes(e.get("label")))

    def test_reconstruct_dot_vakint_minimizes_externals_for_pentabox(self) -> None:
        expr = "prop(k1+p1,0)*prop(k1+k2+p1,0)*prop(k1+k2+p1-q1,0)*prop(k1+k2-p2+q2,0)*prop(k1+k2-p2,0)*prop(k1-p2,0)*prop(k1,m1)*prop(k2,m1)"
        signatures, loop_names, ext_names, masses = extract_signatures_and_masses_from_symbolica_expression(
            expr,
            loop_prefix="k",
            external_prefixes=("p", "q"),
        )
        dot = reconstruct_dot(
            signatures,
            loop_names=list(loop_names),
            ext_names=list(ext_names),
            edge_masses=masses,
            max_degree=4,
            emit_vakint_dot_format=True,
            incoming_external_prefix="p",
            outgoing_external_prefix="q",
            graph_engine="dot",
            minimize_external_legs=True,
        )

        edges_by_id = sorted(dot.get_edges(), key=lambda e: int(_strip_quotes(e.get("id"))))
        external_labels = []
        for e in edges_by_id:
            src = _strip_quotes(e.get_source()).split(":")[0]
            dst = _strip_quotes(e.get_destination()).split(":")[0]
            if src == "ext" or dst == "ext":
                external_labels.append(_strip_quotes(e.get("label")))

        self.assertEqual(len(external_labels), 5)
        self.assertIn("p1", external_labels)
        self.assertIn("p2", external_labels)
        self.assertIn("q1", external_labels)
        self.assertIn("q2", external_labels)
        self.assertIn("p1+p2-q1-q2", external_labels)

    def test_handcrafted_5_loop_many_externals(self) -> None:
        self._assert_reconstructs_to_isomorphic_dot(_handcrafted_5_loop_case())

    def test_handcrafted_6_loop_many_externals(self) -> None:
        self._assert_reconstructs_to_isomorphic_dot(_handcrafted_6_loop_case())

    def test_random_5_loop_cases_fixed_seed(self) -> None:
        rng = random.Random(20260220)
        saw_non_isomorphic = False
        for idx in range(5):
            with self.subTest(case=idx):
                case = _generate_random_5_loop_case(rng)
                signatures = signatures_from_graph_case(case)
                expected = dot_from_graph_case(case, signatures)
                reconstructed = reconstruct_dot(
                    signatures,
                    loop_names=list(case.loop_names),
                    ext_names=list(case.ext_names),
                    num_vertices=case.num_vertices,
                    require_connected=True,
                    max_degree=_max_undirected_degree(case.num_vertices, case.edges),
                )
                if not dots_are_isomorphic(expected, reconstructed):
                    saw_non_isomorphic = True
                self.assertTrue(
                    dot_is_valid_realization(
                        reconstructed,
                        signatures=signatures,
                        loop_names=case.loop_names,
                        ext_names=case.ext_names,
                        num_vertices=case.num_vertices,
                        require_connected=True,
                    ),
                    msg=f"Reconstructed graph is not a valid realization for random case {idx}.",
                )
        self.assertTrue(
            saw_non_isomorphic,
            msg="Expected at least one random case to exhibit non-unique (non-isomorphic) reconstruction.",
        )


def _print_example() -> None:
    # Example: 2-loop sunrise with one external basis momentum p
    signatures = [
        ((1, 0), (0,)),
        ((0, 1), (0,)),
        ((1, 1), (1,)),
    ]

    dot = reconstruct_dot(
        signatures,
        loop_names=["k", "l"],
        ext_names=["p"],
        require_connected=True,
        max_degree=None,
    )
    print(dot.to_string())


def _print_symbolica_signatures(
    expr_text: str,
    loop_prefix: str,
    external_prefixes: Sequence[str],
    prop_pattern: str,
    chain_to_dot: bool,
    dot_output: str,
    max_degree: Optional[int],
    minimize_externals: bool,
) -> None:
    signatures, loop_names, ext_names, masses = extract_signatures_and_masses_from_symbolica_expression(
        expr_text,
        loop_prefix=loop_prefix,
        external_prefixes=external_prefixes,
        prop_pattern=prop_pattern,
    )

    print(f"loop_names = {loop_names}")
    print(f"ext_names = {ext_names}")
    print(f"masses = {masses}")
    print("signatures = [")
    for loop_vec, ext_vec in signatures:
        print(f"  ({loop_vec}, {ext_vec}),")
    print("]")

    if not chain_to_dot:
        return

    incoming_prefix = external_prefixes[0] if external_prefixes else "p"
    outgoing_prefix = external_prefixes[1] if len(external_prefixes) >= 2 else "q"
    dot = reconstruct_dot(
        signatures,
        loop_names=list(loop_names),
        ext_names=list(ext_names),
        edge_masses=masses,
        max_degree=max_degree,
        emit_vakint_dot_format=True,
        incoming_external_prefix=incoming_prefix,
        outgoing_external_prefix=outgoing_prefix,
        graph_engine="dot",
        minimize_external_legs=minimize_externals,
    )
    dot_text = dot.to_string()
    if dot_output == "-":
        print(dot_text)
        return

    with open(dot_output, "w", encoding="utf-8") as handle:
        handle.write(dot_text)
    print(f"dot_output = {dot_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reconstruct loop graphs from momentum signatures.")
    parser.add_argument(
        "--run-tests",
        action="store_true",
        help="Run in-file reconstruction tests.",
    )
    parser.add_argument(
        "--extract-from-symbolica",
        type=str,
        help=('Symbolica expression (or E("...")) from which to extract momentum signatures via pattern matching of prop(q_,m_).'),
    )
    parser.add_argument(
        "--loop-prefix",
        type=str,
        default="k",
        help="Prefix used for loop momenta when extracting from Symbolica input (default: k).",
    )
    parser.add_argument(
        "--external-prefixes",
        type=str,
        default="p,q",
        help="Comma-separated prefixes used for external momenta (default: p,q).",
    )
    parser.add_argument(
        "--prop-pattern",
        type=str,
        default="prop(q_,m_)",
        help="Symbolica pattern used to extract each propagator momentum (default: prop(q_,m_)).",
    )
    parser.add_argument(
        "--chain-to-dot",
        dest="chain_to_dot",
        action="store_true",
        help="After Symbolica extraction, reconstruct a graph and emit DOT (default: enabled).",
    )
    parser.add_argument(
        "--no-chain-to-dot",
        dest="chain_to_dot",
        action="store_false",
        help="Disable DOT reconstruction when using --extract-from-symbolica.",
    )
    parser.add_argument(
        "--dot-output",
        type=str,
        default="reconstructed_from_symbolica.dot",
        help="Path for DOT output when chaining is enabled (default: reconstructed_from_symbolica.dot). Use '-' to print DOT.",
    )
    parser.add_argument(
        "--max-degree",
        type=int,
        default=None,
        help=(
            "Optional maximum total valence cap used to prune reconstruction. "
            "In legacy vakint output every vertex gets one external half-edge "
            "(internal degree <= max_degree-1)."
        ),
    )
    parser.add_argument(
        "--minimize-externals",
        action="store_true",
        help=("In vakint DOT output, search for the realization with the minimum number of external half-edges."),
    )
    parser.set_defaults(chain_to_dot=True)
    args = parser.parse_args()

    if args.run_tests:
        unittest.main(argv=["graph_signatures.py"], verbosity=2)
    elif args.extract_from_symbolica:
        ext_prefixes = [p.strip() for p in args.external_prefixes.split(",") if p.strip()]
        _print_symbolica_signatures(
            args.extract_from_symbolica,
            loop_prefix=args.loop_prefix,
            external_prefixes=ext_prefixes,
            prop_pattern=args.prop_pattern,
            chain_to_dot=args.chain_to_dot,
            dot_output=args.dot_output,
            max_degree=args.max_degree,
            minimize_externals=args.minimize_externals,
        )
    else:
        _print_example()
