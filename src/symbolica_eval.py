from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
import pathlib
import time
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .graph_io import parse_dot_graph, resolve_edge_masses


@dataclass(frozen=True)
class VectorRef:
    kind: str
    index: int


def _expr_key(expr: Mapping[str, Any]) -> Tuple[Tuple[Tuple[int, int], ...], Tuple[Tuple[int, int], ...], str]:
    return (
        tuple((int(a), int(b)) for a, b in expr.get("i", [])),
        tuple((int(a), int(b)) for a, b in expr.get("x", [])),
        str(expr.get("c", "0")),
    )


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def structure_fingerprint(data: Mapping[str, Any]) -> str:
    clean = copy.deepcopy(dict(data))
    clean.pop("evaluator", None)
    clean.pop("evaluators", None)
    return hashlib.sha256(_canonical_json(clean).encode()).hexdigest()


def graph_fingerprint(dot: Any) -> str:
    parsed = parse_dot_graph(dot)
    payload = {
        "loop_names": list(parsed.loop_names),
        "ext_names": list(parsed.ext_names),
        "internal": [
            {
                "tail": edge.tail,
                "head": edge.head,
                "label": edge.label,
                "mass_key": edge.mass_key,
                "signature": edge.signature,
                "had_pow": edge.had_pow,
            }
            for edge in parsed.internal_edges
        ],
        "external": [
            {
                "source": edge.source,
                "destination": edge.destination,
                "label": edge.label,
                "ext_coeffs": edge.ext_coeffs,
            }
            for edge in parsed.external_edges
        ],
    }
    return hashlib.sha256(_canonical_json(payload).encode()).hexdigest()


def _require_symbolica():
    try:
        from symbolica import E, S, Expression, Evaluator, CompiledComplexEvaluator, CompiledRealEvaluator
    except Exception as exc:  # pragma: no cover - exercised only when optional dependency is missing.
        raise RuntimeError("Symbolica is required for this command but could not be imported.") from exc
    return E, S, Expression, Evaluator, CompiledComplexEvaluator, CompiledRealEvaluator


def _num(value: Any):
    _E, _S, Expression, _Evaluator, _CC, _CR = _require_symbolica()
    frac = Fraction(str(value))
    out = Expression.num(frac.numerator)
    if frac.denominator != 1:
        out = out / Expression.num(frac.denominator)
    return out


def _tagged(name: str, *tags: int):
    E, _S, _Expression, _Evaluator, _CC, _CR = _require_symbolica()
    return E(f"{name}({','.join(str(int(t)) for t in tags)})")


def _zero():
    return _num(0)


def _linear_expr(expr: Mapping[str, Any]):
    out = _num(expr.get("c", "0"))
    for edge_id, coeff in expr.get("i", []):
        out += _num(coeff) * _tagged("iE", int(edge_id))
    for ext_id, coeff in expr.get("x", []):
        out += _num(coeff) * _tagged("extE", int(ext_id))
    return out


def _component(ref: VectorRef, mu: int, map_id: int, edge_count: int, loop_count: int, ext_count: int):
    mu = int(mu)
    if mu < 0:
        mu += 4
    if mu < 0 or mu > 3:
        raise ValueError(f"Four-vector component index {mu} is out of range.")
    if ref.kind in {"edges", "src_edges"}:
        idx = ref.index if ref.index >= 0 else edge_count + ref.index
        if idx < 0 or idx >= edge_count:
            raise ValueError(f"Edge index {ref.index} is out of range for {edge_count} edges.")
        return _tagged("edgeQ", map_id, idx, mu)
    if ref.kind == "loops":
        idx = ref.index if ref.index >= 0 else loop_count + ref.index
        if idx < 0 or idx >= loop_count:
            raise ValueError(f"Loop index {ref.index} is out of range for {loop_count} loops.")
        return _tagged("loopQ", map_id, idx, mu)
    if ref.kind == "ext":
        idx = ref.index if ref.index >= 0 else ext_count + ref.index
        if idx < 0 or idx >= ext_count:
            raise ValueError(f"External index {ref.index} is out of range for {ext_count} external momenta.")
        return _tagged("extQ", idx, mu)
    raise ValueError(f"Unsupported vector kind {ref.kind!r}.")


def _dot4(a: VectorRef, b: VectorRef, map_id: int, edge_count: int, loop_count: int, ext_count: int):
    return (
        _component(a, 0, map_id, edge_count, loop_count, ext_count)
        * _component(b, 0, map_id, edge_count, loop_count, ext_count)
        - _component(a, 1, map_id, edge_count, loop_count, ext_count)
        * _component(b, 1, map_id, edge_count, loop_count, ext_count)
        - _component(a, 2, map_id, edge_count, loop_count, ext_count)
        * _component(b, 2, map_id, edge_count, loop_count, ext_count)
        - _component(a, 3, map_id, edge_count, loop_count, ext_count)
        * _component(b, 3, map_id, edge_count, loop_count, ext_count)
    )


class NumeratorTranslator(ast.NodeVisitor):
    def __init__(self, map_id: int, edge_count: int, loop_count: int, ext_count: int):
        self.map_id = int(map_id)
        self.edge_count = int(edge_count)
        self.loop_count = int(loop_count)
        self.ext_count = int(ext_count)

    def translate(self, expr: str):
        tree = ast.parse(expr, mode="eval")
        return self.visit(tree.body)

    def _index(self, node: ast.AST) -> int:
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return int(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -self._index(node.operand)
        raise ValueError("Only integer literal indices are supported in Symbolica numerators.")

    def visit_Constant(self, node: ast.Constant):
        if isinstance(node.value, (int, float)):
            return _num(repr(node.value))
        raise ValueError(f"Unsupported numerator constant {node.value!r}.")

    def visit_UnaryOp(self, node: ast.UnaryOp):
        val = self.visit(node.operand)
        if isinstance(val, VectorRef):
            raise ValueError("Unary operators cannot be applied to vector references.")
        if isinstance(node.op, ast.USub):
            return -val
        if isinstance(node.op, ast.UAdd):
            return val
        raise ValueError("Unsupported unary operator in numerator.")

    def visit_BinOp(self, node: ast.BinOp):
        left = self.visit(node.left)
        right = self.visit(node.right)
        if isinstance(left, VectorRef) or isinstance(right, VectorRef):
            raise ValueError("Vector references must be indexed or passed to dot(...).")
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Pow):
            if isinstance(node.right, ast.Constant) and isinstance(node.right.value, int):
                return left ** int(node.right.value)
            raise ValueError("Symbolica numerator powers must use integer literal exponents.")
        raise ValueError("Unsupported binary operator in numerator.")

    def visit_Name(self, node: ast.Name):
        if node.id in {"edges", "src_edges", "loops", "ext"}:
            return node.id
        raise ValueError(f"Unsupported name {node.id!r} in Symbolica numerator.")

    def visit_Subscript(self, node: ast.Subscript):
        base = self.visit(node.value)
        idx = self._index(node.slice)
        if isinstance(base, str):
            if base not in {"edges", "src_edges", "loops", "ext"}:
                raise ValueError(f"Unsupported indexed object {base!r}.")
            return VectorRef(base, idx)
        if isinstance(base, VectorRef):
            return _component(base, idx, self.map_id, self.edge_count, self.loop_count, self.ext_count)
        raise ValueError("Unsupported subscript in Symbolica numerator.")

    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id == "dot":
            if len(node.args) != 2 or node.keywords:
                raise ValueError("dot(...) expects exactly two positional arguments.")
            a = self.visit(node.args[0])
            b = self.visit(node.args[1])
            if not isinstance(a, VectorRef) or not isinstance(b, VectorRef):
                raise ValueError("dot(...) arguments must be vector references such as edges[0] or ext[1].")
            return _dot4(a, b, self.map_id, self.edge_count, self.loop_count, self.ext_count)
        raise ValueError("Only dot(...) calls are currently supported in Symbolica numerators.")

    def generic_visit(self, node: ast.AST):
        raise ValueError(f"Unsupported numerator syntax: {node.__class__.__name__}.")


def _input_layout(parsed: Any) -> List[Dict[str, Any]]:
    layout: List[Dict[str, Any]] = []
    for ext_id, name in enumerate(parsed.ext_names):
        for component, comp_name in enumerate(("e", "x", "y", "z")):
            layout.append({"kind": "external", "index": ext_id, "component": component, "name": f"ext_{name}_{comp_name}"})
    for loop_id, name in enumerate(parsed.loop_names):
        for spatial_component, comp_name in enumerate(("x", "y", "z")):
            layout.append({"kind": "loop3", "index": loop_id, "component": spatial_component, "name": f"loop_{name}_{comp_name}"})
    for edge in parsed.internal_edges:
        layout.append({"kind": "mass", "edge": edge.edge_id, "mass_key": edge.mass_key, "name": f"mass_edge_{edge.edge_id}"})
    return layout


def _params_from_layout(layout: Sequence[Mapping[str, Any]]):
    _E, S, _Expression, _Evaluator, _CC, _CR = _require_symbolica()
    return [S(str(entry["name"])) for entry in layout]


def _param_lookup(layout: Sequence[Mapping[str, Any]], params: Sequence[Any]) -> Dict[Tuple[Any, ...], Any]:
    lookup: Dict[Tuple[Any, ...], Any] = {}
    for entry, param in zip(layout, params):
        kind = entry["kind"]
        if kind == "external":
            lookup[(kind, int(entry["index"]), int(entry["component"]))] = param
        elif kind == "loop3":
            lookup[(kind, int(entry["index"]), int(entry["component"]))] = param
        elif kind == "mass":
            lookup[(kind, int(entry["edge"]))] = param
    return lookup


def _surface_call(surface: Mapping[str, Any]):
    kind = str(surface.get("k", "e"))
    name = "etaE" if kind == "e" else "etaH"
    return _tagged(name, int(surface["id"]))


def _tree_expr(tree: Mapping[str, Any], node_idx: int, surfaces: Mapping[int, Mapping[str, Any]]):
    node = tree["nodes"][int(node_idx)]
    factor = _num(1)
    for sid in node.get("surfaces", []):
        factor /= _surface_call(surfaces[int(sid)])
    children = node.get("children", [])
    if not children:
        return factor
    subtotal = _zero()
    for child in children:
        subtotal += _tree_expr(tree, int(child), surfaces)
    return factor * subtotal


def _variant_tree_sum(variant: Mapping[str, Any], surfaces: Mapping[int, Mapping[str, Any]]):
    total = _zero()
    for root in variant["tree"].get("roots", []):
        total += _tree_expr(variant["tree"], int(root), surfaces)
    return total


def _map_key(loop_q0: Sequence[Mapping[str, Any]], edge_q0: Sequence[Mapping[str, Any]]):
    return (
        tuple(_expr_key(x) for x in loop_q0),
        tuple(_expr_key(x) for x in edge_q0),
    )


def build_symbolica_expression(data: Mapping[str, Any], dot: Any, numerator_expr: str):
    E, S, _Expression, _Evaluator, _CC, _CR = _require_symbolica()
    parsed = parse_dot_graph(dot)
    layout = _input_layout(parsed)
    params = _params_from_layout(layout)
    param = _param_lookup(layout, params)
    surfaces = {int(surface["id"]): surface for surface in data.get("surfaces", [])}
    signatures = tuple(edge.signature for edge in parsed.internal_edges)
    edge_count = len(parsed.internal_edges)
    loop_count = len(parsed.loop_names)
    ext_count = len(parsed.ext_names)
    functions: Dict[Tuple[Any, str, Sequence[Any]], Any] = {}

    ose_x, ose_y, ose_z, ose_m = S("ose_x", "ose_y", "ose_z", "ose_m")
    functions[(S("OSE"), "OSE", (ose_x, ose_y, ose_z, ose_m))] = (ose_x * ose_x + ose_y * ose_y + ose_z * ose_z + ose_m * ose_m).sqrt()

    for ext_id in range(ext_count):
        for mu in range(4):
            functions[(_tagged("extQ", ext_id, mu), f"extQ_{ext_id}_{mu}", tuple())] = param[("external", ext_id, mu)]
        functions[(_tagged("extE", ext_id), f"extE_{ext_id}", tuple())] = param[("external", ext_id, 0)]

    for loop_id in range(loop_count):
        for spatial_component, mu in enumerate((1, 2, 3)):
            functions[(_tagged("loopSpatial", loop_id, mu), f"loopSpatial_{loop_id}_{mu}", tuple())] = param[("loop3", loop_id, spatial_component)]

    for edge_id, sig in enumerate(signatures):
        loop_coeffs, ext_coeffs = sig
        for spatial_component, axis_name in enumerate(("x", "y", "z")):
            body = _zero()
            for loop_id, coeff in enumerate(loop_coeffs):
                if coeff:
                    body += _num(coeff) * param[("loop3", loop_id, spatial_component)]
            for ext_id, coeff in enumerate(ext_coeffs):
                if coeff:
                    body += _num(coeff) * param[("external", ext_id, spatial_component + 1)]
            functions[(_tagged(f"q{axis_name}", edge_id), f"q{axis_name}_{edge_id}", tuple())] = body
        functions[(_tagged("mass", edge_id), f"mass_{edge_id}", tuple())] = param[("mass", edge_id)]
        functions[(_tagged("iE", edge_id), f"iE_{edge_id}", tuple())] = E(f"OSE(qx({edge_id}),qy({edge_id}),qz({edge_id}),mass({edge_id}))")

    for surface in surfaces.values():
        call = _surface_call(surface)
        functions[(call, f"{str(surface.get('k', 'e'))}_{int(surface['id'])}", tuple())] = _linear_expr(surface["e"])

    unique_maps: Dict[Tuple[Any, Any], int] = {}
    map_defs: List[Tuple[Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]]]] = []
    for orient in data.get("orientations", []):
        for variant in orient.get("variants") or [orient]:
            key = _map_key(variant["loop_q0"], variant["edge_q0"])
            if key not in unique_maps:
                unique_maps[key] = len(unique_maps)
                map_defs.append((variant["loop_q0"], variant["edge_q0"]))

    numerator_by_map: Dict[int, Any] = {}
    for map_id, (loop_q0, edge_q0) in enumerate(map_defs):
        for loop_id in range(loop_count):
            functions[(_tagged("loopQ", map_id, loop_id, 0), f"loopQ_{map_id}_{loop_id}_0", tuple())] = _linear_expr(loop_q0[loop_id])
            for spatial_component, mu in enumerate((1, 2, 3)):
                functions[(_tagged("loopQ", map_id, loop_id, mu), f"loopQ_{map_id}_{loop_id}_{mu}", tuple())] = param[("loop3", loop_id, spatial_component)]
        for edge_id in range(edge_count):
            functions[(_tagged("edgeQ", map_id, edge_id, 0), f"edgeQ_{map_id}_{edge_id}_0", tuple())] = _linear_expr(edge_q0[edge_id])
            functions[(_tagged("edgeQ", map_id, edge_id, 1), f"edgeQ_{map_id}_{edge_id}_1", tuple())] = _tagged("qx", edge_id)
            functions[(_tagged("edgeQ", map_id, edge_id, 2), f"edgeQ_{map_id}_{edge_id}_2", tuple())] = _tagged("qy", edge_id)
            functions[(_tagged("edgeQ", map_id, edge_id, 3), f"edgeQ_{map_id}_{edge_id}_3", tuple())] = _tagged("qz", edge_id)
        numerator_by_map[map_id] = NumeratorTranslator(map_id, edge_count, loop_count, ext_count).translate(numerator_expr)

    total = _zero()
    for orient in data.get("orientations", []):
        variants = orient.get("variants") or [orient]
        for variant in variants:
            key = _map_key(variant["loop_q0"], variant["edge_q0"])
            map_id = unique_maps[key]
            term = _num(variant["pref"])
            for edge_id in variant.get("half_edges", []):
                term /= _num(2) * _tagged("iE", int(edge_id))
            for sid in variant.get("num_surfaces", []):
                term *= _surface_call(surfaces[int(sid)])
            term *= _variant_tree_sum(variant, surfaces)
            term *= numerator_by_map[map_id]
            total += term

    return {
        "expression": total,
        "functions": functions,
        "params": params,
        "layout": layout,
        "map_count": len(unique_maps),
    }


def default_output_path(json_path: pathlib.Path, data: Mapping[str, Any], dot: Any, numerator_expr: str, value_type: str) -> pathlib.Path:
    payload = {
        "structure": structure_fingerprint(data),
        "graph": graph_fingerprint(dot),
        "numerator": numerator_expr,
        "value_type": value_type,
    }
    digest = hashlib.sha256(_canonical_json(payload).encode()).hexdigest()[:12]
    return json_path.with_name(f"{json_path.stem}__symbolica_{digest}.so")


def default_eager_output_path(json_path: pathlib.Path, data: Mapping[str, Any], dot: Any, numerator_expr: str, value_type: str) -> pathlib.Path:
    return default_output_path(json_path, data, dot, numerator_expr, value_type).with_suffix(".sev")


def _common_metadata(
    data: Mapping[str, Any],
    dot: Any,
    built: Mapping[str, Any],
    numerator_expr: str,
    value_type: str,
    n_cores: int,
    iterations: int,
    cpe_iterations: Optional[int],
) -> Dict[str, Any]:
    return {
        "kind": "symbolica",
        "version": 2,
        "value_type": value_type,
        "input_len": len(built["params"]),
        "output_len": 1,
        "numerator_expr": numerator_expr,
        "input_layout": built["layout"],
        "map_count": built["map_count"],
        "graph_fingerprint": graph_fingerprint(dot),
        "structure_fingerprint": structure_fingerprint(data),
        "symbolica": {
            "n_cores": n_cores,
            "iterations": iterations,
            "cpe_iterations": cpe_iterations,
        },
    }


def compile_symbolica_evaluator(
    data: Mapping[str, Any],
    dot: Any,
    numerator_expr: str,
    json_path: pathlib.Path,
    output_path: Optional[pathlib.Path] = None,
    eager_output_path: Optional[pathlib.Path] = None,
    value_type: str = "real",
    function_name: Optional[str] = None,
    inline_asm: str = "default",
    optimization_level: int = 3,
    native: bool = True,
    n_cores: int = 4,
    iterations: int = 1,
    cpe_iterations: Optional[int] = None,
    compiler_path: Optional[str] = None,
    compiler_flags: Optional[Sequence[str]] = None,
    keep_cpp: bool = False,
) -> Dict[str, Any]:
    _E, _S, _Expression, _Evaluator, _CompiledComplexEvaluator, _CompiledRealEvaluator = _require_symbolica()
    if value_type not in {"real", "complex"}:
        raise ValueError("value_type must be 'real' or 'complex'.")
    json_path = pathlib.Path(json_path)
    out_path = pathlib.Path(output_path) if output_path is not None else default_output_path(json_path, data, dot, numerator_expr, value_type)
    if out_path.suffix != ".so":
        out_path = out_path.with_suffix(".so")
    eager_path = pathlib.Path(eager_output_path) if eager_output_path is not None else default_eager_output_path(json_path, data, dot, numerator_expr, value_type)
    if eager_path.suffix == "":
        eager_path = eager_path.with_suffix(".sev")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    eager_path.parent.mkdir(parents=True, exist_ok=True)
    cpp_path = out_path.with_suffix(".cpp")
    function_name = function_name or f"hybrid3d_eval_{hashlib.sha256(str(out_path).encode()).hexdigest()[:10]}"

    built = build_symbolica_expression(data, dot, numerator_expr)
    evaluator = built["expression"].evaluator(
        {},
        built["functions"],
        built["params"],
        iterations=iterations,
        cpe_iterations=cpe_iterations,
        n_cores=n_cores,
        jit_compile=False,
        direct_translation=True,
    )
    if value_type == "complex":
        evaluator.set_real_params(list(range(len(built["params"]))), sqrt_real=True)
    eager_path.write_bytes(evaluator.save())
    evaluator.compile(
        function_name,
        str(cpp_path),
        str(out_path),
        value_type,
        inline_asm=inline_asm,
        optimization_level=optimization_level,
        native=native,
        compiler_path=compiler_path,
        compiler_flags=list(compiler_flags) if compiler_flags else None,
    )
    if not keep_cpp:
        try:
            cpp_path.unlink()
        except FileNotFoundError:
            pass
    common = _common_metadata(data, dot, built, numerator_expr, value_type, n_cores, iterations, cpe_iterations)
    rel_library = os.path.relpath(out_path, json_path.parent)
    rel_eager = os.path.relpath(eager_path, json_path.parent)
    compiled = {
        **common,
        "mode": "symbolica_compiled",
        "library": rel_library,
        "function_name": function_name,
        "symbolica": {
            **common["symbolica"],
            "inline_asm": inline_asm,
            "optimization_level": optimization_level,
            "native": native,
        },
    }
    eager = {
        **common,
        "mode": "symbolica_eager",
        "state": rel_eager,
        "symjit": False,
    }
    eager_symjit = {
        **common,
        "mode": "symbolica_eager_symjit",
        "state": rel_eager,
        "symjit": True,
    }
    return {
        "symbolica_compiled": compiled,
        "symbolica_eager": eager,
        "symbolica_eager_symjit": eager_symjit,
    }


def prepare_symbolica_inputs(metadata: Mapping[str, Any], dot: Any, ext4: Sequence[Sequence[Any]], loop3: Sequence[Sequence[Any]], mass_map: Optional[Mapping[str, Any]]):
    parsed = parse_dot_graph(dot)
    masses = resolve_edge_masses(parsed, dict(mass_map or {}))
    values: List[float] = []
    for entry in metadata["input_layout"]:
        kind = entry["kind"]
        if kind == "external":
            values.append(float(ext4[int(entry["index"])][int(entry["component"])]))
        elif kind == "loop3":
            values.append(float(loop3[int(entry["index"])][int(entry["component"])]))
        elif kind == "mass":
            values.append(float(masses[int(entry["edge"])]))
        else:
            raise ValueError(f"Unsupported Symbolica input layout kind {kind!r}.")
    return values


def _metadata_library_path(json_path: pathlib.Path, metadata: Mapping[str, Any]) -> pathlib.Path:
    lib = pathlib.Path(str(metadata["library"]))
    if not lib.is_absolute():
        lib = pathlib.Path(json_path).parent / lib
    return lib


def _metadata_state_path(json_path: pathlib.Path, metadata: Mapping[str, Any]) -> pathlib.Path:
    state = pathlib.Path(str(metadata["state"]))
    if not state.is_absolute():
        state = pathlib.Path(json_path).parent / state
    return state


def canonical_symbolica_backend(backend: str) -> str:
    return "symbolica_compiled" if backend == "symbolica" else backend


def available_symbolica_evaluator_modes(data: Mapping[str, Any]) -> List[str]:
    modes: List[str] = []
    evaluators = data.get("evaluators")
    if isinstance(evaluators, Mapping):
        for mode in ("symbolica_compiled", "symbolica_eager", "symbolica_eager_symjit"):
            if mode in evaluators:
                modes.append(mode)
    elif "evaluator" in data:
        modes.append("symbolica_compiled")
    return modes


def symbolica_evaluator_metadata(data: Mapping[str, Any], backend: str) -> Mapping[str, Any]:
    mode = canonical_symbolica_backend(backend)
    evaluators = data.get("evaluators")
    if isinstance(evaluators, Mapping) and mode in evaluators:
        metadata = evaluators[mode]
    elif mode == "symbolica_compiled" and "evaluator" in data:
        metadata = data["evaluator"]
    else:
        available = ", ".join(available_symbolica_evaluator_modes(data)) or "none"
        if available == "none":
            raise ValueError("JSON has no evaluator metadata. Run the compile subcommand first.")
        raise ValueError(f"JSON has no {mode!r} evaluator metadata. Available modes: {available}.")
    if metadata.get("kind") != "symbolica":
        raise ValueError(f"JSON evaluator metadata for {mode!r} is not a Symbolica evaluator.")
    return metadata


def load_compiled_evaluator(json_path: pathlib.Path, metadata: Mapping[str, Any]):
    _E, _S, _Expression, _Evaluator, CompiledComplexEvaluator, CompiledRealEvaluator = _require_symbolica()
    lib = _metadata_library_path(json_path, metadata)
    if not lib.exists():
        raise FileNotFoundError(f"Symbolica evaluator library does not exist: {lib}")
    value_type = metadata["value_type"]
    cls = CompiledComplexEvaluator if value_type == "complex" else CompiledRealEvaluator
    return cls.load(str(lib), str(metadata["function_name"]), int(metadata["input_len"]), int(metadata["output_len"]))


def load_eager_evaluator(json_path: pathlib.Path, metadata: Mapping[str, Any]):
    _E, _S, _Expression, Evaluator, _CompiledComplexEvaluator, _CompiledRealEvaluator = _require_symbolica()
    state = _metadata_state_path(json_path, metadata)
    if not state.exists():
        raise FileNotFoundError(f"Symbolica eager evaluator state does not exist: {state}")
    evaluator = Evaluator.load(state.read_bytes())
    evaluator.jit_compile(bool(metadata.get("symjit", False)))
    return evaluator


def evaluate_symbolica(
    data: Mapping[str, Any],
    dot: Any,
    json_path: pathlib.Path,
    ext4: Sequence[Sequence[Any]],
    loop3: Sequence[Sequence[Any]],
    mass_map: Optional[Mapping[str, Any]],
    batch_size: int = 1,
    profile: bool = False,
    backend: str = "symbolica_compiled",
) -> Tuple[Any, Optional[Dict[str, Any]]]:
    metadata = symbolica_evaluator_metadata(data, backend)
    if metadata.get("structure_fingerprint") != structure_fingerprint(data):
        raise ValueError("Symbolica evaluator is stale: structure fingerprint does not match this JSON.")
    if metadata.get("graph_fingerprint") != graph_fingerprint(dot):
        raise ValueError("Symbolica evaluator is stale: graph fingerprint does not match the DOT graph.")

    import numpy as np

    mode = canonical_symbolica_backend(backend)
    if mode == "symbolica_compiled":
        evaluator = load_compiled_evaluator(pathlib.Path(json_path), metadata)
    elif mode in {"symbolica_eager", "symbolica_eager_symjit"}:
        evaluator = load_eager_evaluator(pathlib.Path(json_path), metadata)
    else:
        raise ValueError(f"Unsupported Symbolica evaluator backend {backend!r}.")
    row = prepare_symbolica_inputs(metadata, dot, ext4, loop3, mass_map)
    batch_size = int(batch_size)
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    dtype = np.complex128 if metadata["value_type"] == "complex" else np.float64
    inputs = np.asarray([row] * batch_size, dtype=dtype)
    eval_fn = evaluator.evaluate_complex if metadata["value_type"] == "complex" and mode in {"symbolica_eager", "symbolica_eager_symjit"} else evaluator.evaluate
    if profile:
        eval_fn(inputs[:1])
    start = time.perf_counter()
    result = eval_fn(inputs)
    elapsed = time.perf_counter() - start
    value = result[0][0]
    if not profile:
        return value, None
    return value, {
        "batch_size": batch_size,
        "total_seconds": elapsed,
        "seconds_per_sample": elapsed / batch_size,
    }


def format_duration(seconds: float) -> str:
    if seconds < 1e-3:
        return f"{seconds * 1e6:.3g} us"
    if seconds < 1:
        return f"{seconds * 1e3:.3g} ms"
    return f"{seconds:.3g} s"


def _pretty_expr(expr: Any, max_line_length: int = 100) -> str:
    if hasattr(expr, "format"):
        return expr.format(
            max_line_length=max_line_length,
            indentation=2,
            terms_on_new_line=True,
            color_top_level_sum=False,
            color_builtin_symbols=False,
            bracket_level_colors=None,
            max_terms=None,
        )
    return str(expr)


def format_symbolica_evaluator_inputs(
    data: Mapping[str, Any],
    dot: Any,
    numerator_expr: str,
    value_type: str = "real",
    n_cores: int = 4,
    iterations: int = 1,
    cpe_iterations: Optional[int] = None,
    max_line_length: int = 100,
) -> str:
    built = build_symbolica_expression(data, dot, numerator_expr)
    lines: List[str] = []
    lines.append("Symbolica evaluator input")
    lines.append("========================")
    lines.append("")
    lines.append(f"value_type = {value_type}")
    lines.append(f"numerator_expr = {numerator_expr}")
    lines.append(f"input_len = {len(built['params'])}")
    lines.append("output_len = 1")
    lines.append(f"n_cores = {n_cores}")
    lines.append(f"iterations = {iterations}")
    lines.append(f"cpe_iterations = {cpe_iterations}")
    lines.append("")
    lines.append("Evaluator call")
    lines.append("--------------")
    lines.append(
        "expression.evaluator(constants, functions, params, "
        f"iterations={iterations}, cpe_iterations={cpe_iterations}, "
        f"n_cores={n_cores}, jit_compile=False, direct_translation=True)"
    )
    lines.append("")
    lines.append("Parameters")
    lines.append("----------")
    for idx, (param, entry) in enumerate(zip(built["params"], built["layout"])):
        lines.append(f"[{idx:03d}] {_pretty_expr(param, max_line_length)}  # {json.dumps(entry, sort_keys=True)}")
    lines.append("")
    lines.append("Constants")
    lines.append("---------")
    lines.append("(none)")
    lines.append("")
    lines.append("Function map")
    lines.append("------------")
    for idx, ((name, printable, args), body) in enumerate(built["functions"].items()):
        arg_text = ", ".join(_pretty_expr(arg, max_line_length) for arg in args)
        header = f"[{idx:03d}] {_pretty_expr(name, max_line_length)}"
        header += f"  printable={printable!r}"
        header += f"  args=({arg_text})"
        lines.append(header)
        rendered = _pretty_expr(body, max_line_length)
        for body_line in rendered.splitlines():
            lines.append(f"      {body_line}")
    lines.append("")
    lines.append("Top-level expression")
    lines.append("--------------------")
    lines.append(_pretty_expr(built["expression"], max_line_length))
    return "\n".join(lines)
