from __future__ import annotations
from typing import Any, Dict, List, Optional, Set

try:
    from prettytable import PrettyTable
except Exception:
    class PrettyTable:  # type: ignore
        def __init__(self): self.field_names=[]; self._rows=[]
        def add_row(self,row): self._rows.append(row)
        def get_string(self):
            out=[' | '.join(map(str,self.field_names))]
            out += [' | '.join(map(str,r)) for r in self._rows]
            return '\n'.join(out)
        def __str__(self): return self.get_string()

try:
    from colorama import Fore, Back, Style, init as colorama_init
    colorama_init()
except Exception:
    class _Dummy:
        def __getattr__(self, _): return ''
    Fore = Back = Style = _Dummy()  # type: ignore


def c(text: str, code: str, use_color: bool = True) -> str:
    return f"{code}{text}{Style.RESET_ALL}" if use_color else text


def _color_sign(ch: str, use_color: bool = True) -> str:
    if ch == '+': return c('+', Fore.GREEN + Style.BRIGHT, use_color)
    if ch == '-': return c('-', Fore.RED + Style.BRIGHT, use_color)
    return c('0', Fore.WHITE + Style.BRIGHT, use_color)


def _render_signs(signs: List[int], use_color: bool = True) -> str:
    return ''.join(_color_sign('+' if s > 0 else '-' if s < 0 else '0', use_color) for s in signs)


def _render_orientation_label(label: str, use_color: bool = True) -> str:
    out = []
    for ch in str(label):
        if ch in '+-0':
            out.append(_color_sign(ch, use_color))
        elif ch == '|':
            out.append(c('|', Fore.YELLOW + Style.BRIGHT, use_color))
        else:
            out.append(c(ch, Fore.WHITE + Style.BRIGHT, use_color))
    return ''.join(out)


def _collect_ose_ids(data: Dict[str, Any]) -> List[int]:
    ids: Set[int] = set()
    for surf in data['surfaces']:
        for ext_id, _ in surf['e'].get('x', []): ids.add(int(ext_id))
    for orient in data['orientations']:
        for expr in orient.get('loop_q0', []) + orient.get('edge_q0', []):
            for ext_id, _ in expr.get('x', []): ids.add(int(ext_id))
    return sorted(ids)


def _ose_display_map(data: Dict[str, Any]) -> Dict[int, int]:
    return {ose_id: i for i, ose_id in enumerate(_collect_ose_ids(data))}


def _token(base: str, idx: int, palette: str, use_color: bool = True) -> str:
    return c(base, palette + Style.BRIGHT, use_color) + '[' + c(str(idx), Fore.GREEN + Style.BRIGHT, use_color) + ']'


def _render_surface_expr(expr_container: Dict[str, Any], use_color: bool = True, ose_map: Optional[Dict[int, int]] = None) -> str:
    expr = expr_container['e'] if 'e' in expr_container else expr_container
    parts: List[tuple[str, bool]] = []
    cst = expr.get('c', '0')
    if cst not in {'0', '0.0'}:
        parts.append((c(str(cst), Fore.YELLOW + Style.BRIGHT, use_color), False))
    for edge_id, coeff in expr.get('i', []):
        name = _token('OSE', int(edge_id), Fore.CYAN, use_color)
        if int(coeff) == -1:
            parts.append((name, True))
        elif int(coeff) == 1:
            parts.append((name, False))
        else:
            mult = c(str(abs(int(coeff))), Fore.YELLOW + Style.BRIGHT, use_color) + c('*', Fore.WHITE, use_color) + name
            parts.append((mult, int(coeff) < 0))
    for ext_id, coeff in expr.get('x', []):
        disp = ose_map.get(int(ext_id), int(ext_id)) if ose_map is not None else int(ext_id)
        name = _token('E', disp, Fore.MAGENTA, use_color)
        if int(coeff) == -1:
            parts.append((name, True))
        elif int(coeff) == 1:
            parts.append((name, False))
        else:
            mult = c(str(abs(int(coeff))), Fore.YELLOW + Style.BRIGHT, use_color) + c('*', Fore.WHITE, use_color) + name
            parts.append((mult, int(coeff) < 0))
    if not parts:
        return c('0', Fore.YELLOW + Style.BRIGHT, use_color)
    out = (c('-', Fore.RED + Style.BRIGHT, use_color) + ' ' if parts[0][1] else '') + parts[0][0]
    for txt, is_neg in parts[1:]:
        out += ' ' + (c('-', Fore.RED + Style.BRIGHT, use_color) if is_neg else c('+', Fore.GREEN + Style.BRIGHT, use_color)) + ' ' + txt
    return out


def _collect_surface_ids(tree: Dict[str, Any], roots: List[int], surface_kinds: Optional[Dict[int, str]] = None, only_kind: Optional[str] = None) -> List[int]:
    seen: Set[int] = set(); used: Set[int] = set()
    def walk(node_id: int):
        if node_id in seen: return
        seen.add(node_id)
        node = tree['nodes'][node_id]
        for sid in node['surfaces']:
            sid_i = int(sid)
            if only_kind is None or (surface_kinds or {}).get(sid_i) == only_kind:
                used.add(sid_i)
        for child in node['children']:
            walk(child)
    for r in roots: walk(r)
    return sorted(used)


def _mass_desc(parsed, edge_idx: int, use_color: bool = True) -> str:
    e = parsed.internal_edges[edge_idx]
    if getattr(e, 'mass_key', None) is None:
        return c('m=0', Fore.YELLOW + Style.BRIGHT, use_color)
    return c('m', Fore.YELLOW + Style.BRIGHT, use_color) + '(' + c(str(e.mass_key), Fore.GREEN + Style.BRIGHT, use_color) + ')'


def _matches(orient: Dict[str, Any], selector: Optional[str]) -> bool:
    if selector is None:
        return True
    return str(orient['id']) == str(selector) or str(orient.get('orient_label', '')) == str(selector)


def render_pretty(
    data: Dict[str, Any],
    parsed=None,
    use_color: bool = True,
    orientation_id: Optional[str] = None,
    show_details: bool = False,
    details_for: Optional[str] = None,
) -> str:
    lines: List[str] = []
    ose_map = _ose_display_map(data)
    kind_map = {int(s['id']): s['k'] for s in data['surfaces']}

    title = f"{data['family'].upper()} structure"
    subtitle = f"backend={data['backend']}"
    lines.append(c(title, Fore.BLUE + Style.BRIGHT, use_color) + '  ' + c(subtitle, Fore.WHITE + Style.BRIGHT, use_color))
    lines.append(c('=' * max(16, len(title) + len(subtitle) + 2), Fore.BLUE + Style.BRIGHT, use_color))

    summary = PrettyTable()
    summary.field_names = [c('family', Fore.CYAN + Style.BRIGHT, use_color), c('backend', Fore.CYAN + Style.BRIGHT, use_color), c('internal edges', Fore.CYAN + Style.BRIGHT, use_color), c('surfaces', Fore.CYAN + Style.BRIGHT, use_color), c('orientations', Fore.CYAN + Style.BRIGHT, use_color)]
    summary.add_row([c(data['family'], Fore.GREEN + Style.BRIGHT, use_color), c(data['backend'], Fore.MAGENTA + Style.BRIGHT, use_color), data['graph']['n_internal_edges'], len(data['surfaces']), len(data['orientations'])])
    lines.append(summary.get_string())

    if parsed is not None:
        ch_tbl = PrettyTable()
        ch_tbl.field_names = [c('edge', Fore.CYAN + Style.BRIGHT, use_color), c('tail', Fore.CYAN + Style.BRIGHT, use_color), c('head', Fore.CYAN + Style.BRIGHT, use_color), c('label', Fore.CYAN + Style.BRIGHT, use_color), c('mass', Fore.CYAN + Style.BRIGHT, use_color)]
        for e in parsed.internal_edges:
            ch_tbl.add_row([c(str(e.edge_id), Fore.GREEN + Style.BRIGHT, use_color), e.tail, e.head, c(e.label, Fore.WHITE + Style.BRIGHT, use_color), _mass_desc(parsed, e.edge_id, use_color)])
        lines.append('\n' + c('Internal edges', Fore.BLUE + Style.BRIGHT, use_color))
        lines.append(ch_tbl.get_string())

    surf_tbl = PrettyTable()
    surf_tbl.field_names = [c('sid', Fore.CYAN + Style.BRIGHT, use_color), c('kind', Fore.CYAN + Style.BRIGHT, use_color), c('expr', Fore.CYAN + Style.BRIGHT, use_color)]
    for surf in data['surfaces']:
        surf_tbl.add_row([c(str(surf['id']), Fore.GREEN + Style.BRIGHT, use_color), c(str(surf['k']), Fore.MAGENTA + Style.BRIGHT, use_color), _render_surface_expr(surf, use_color, ose_map)])
    lines.append('\n' + c('Surface cache', Fore.BLUE + Style.BRIGHT, use_color))
    lines.append(surf_tbl.get_string())

    shown = [o for o in data['orientations'] if _matches(o, orientation_id)]
    or_tbl = PrettyTable()
    or_tbl.field_names = [c('id', Fore.CYAN + Style.BRIGHT, use_color), c('orient', Fore.CYAN + Style.BRIGHT, use_color), c('pref', Fore.CYAN + Style.BRIGHT, use_color), c('half_edges', Fore.CYAN + Style.BRIGHT, use_color), c('e-surface ids', Fore.CYAN + Style.BRIGHT, use_color), c('root_nodes', Fore.CYAN + Style.BRIGHT, use_color)]
    for orient in shown:
        eids = _collect_surface_ids(orient['tree'], orient['tree']['roots'], kind_map, only_kind='e')
        or_tbl.add_row([c(str(orient['id']), Fore.GREEN + Style.BRIGHT, use_color), _render_orientation_label(orient.get('orient_label', ''), use_color), c(str(orient['pref']), Fore.YELLOW + Style.BRIGHT, use_color), orient['half_edges'], eids, orient['tree']['roots']])
    lines.append('\n' + c('Orientations', Fore.BLUE + Style.BRIGHT, use_color))
    lines.append(or_tbl.get_string())

    if not show_details and details_for is None:
        return '\n'.join(lines)

    detailed = [o for o in shown if _matches(o, details_for)] if details_for is not None else shown
    for orient in detailed:
        lines.append('\n' + c(f"Orientation #{orient['id']}", Fore.BLUE + Style.BRIGHT, use_color) + '  ' + _render_orientation_label(orient.get('orient_label', ''), use_color))
        q_tbl = PrettyTable(); q_tbl.field_names = [c('loop', Fore.CYAN + Style.BRIGHT, use_color), c('q0 substitution', Fore.CYAN + Style.BRIGHT, use_color)]
        for i, expr in enumerate(orient['loop_q0']):
            q_tbl.add_row([c(str(i), Fore.GREEN + Style.BRIGHT, use_color), _render_surface_expr(expr, use_color, ose_map)])
        lines.append(q_tbl.get_string())
        e_tbl = PrettyTable(); e_tbl.field_names = [c('edge', Fore.CYAN + Style.BRIGHT, use_color), c('q_e^0 substitution', Fore.CYAN + Style.BRIGHT, use_color)]
        for i, expr in enumerate(orient['edge_q0']):
            e_tbl.add_row([c(str(i), Fore.GREEN + Style.BRIGHT, use_color), _render_surface_expr(expr, use_color, ose_map)])
        lines.append(e_tbl.get_string())
        skel = orient.get('meta', {}).get('skeleton_surface_ids', [])
        if skel:
            lines.append(c('Hybrid LTD skeleton surfaces', Fore.BLUE + Style.BRIGHT, use_color) + ': ' + ', '.join(c(str(s), Fore.GREEN + Style.BRIGHT, use_color) + ':' + c(kind_map.get(int(s), '?'), Fore.MAGENTA + Style.BRIGHT, use_color) for s in skel))
        lines.append(c('Factorization tree', Fore.BLUE + Style.BRIGHT, use_color))
        def walk(node_id: int, depth: int):
            node = orient['tree']['nodes'][node_id]
            prefix = '  ' * depth
            surf_text = ', '.join(c(str(s), Fore.GREEN + Style.BRIGHT, use_color) for s in node['surfaces'])
            child_text = ', '.join(c(str(ch), Fore.YELLOW + Style.BRIGHT, use_color) for ch in node['children'])
            lines.append(f"{prefix}{c('node', Fore.MAGENTA + Style.BRIGHT, use_color)} {c(str(node_id), Fore.GREEN + Style.BRIGHT, use_color)}: surfaces=[{surf_text}] children=[{child_text}]")
            for child in node['children']:
                walk(child, depth + 1)
        for root in orient['tree']['roots']:
            walk(root, 0)
    return '\n'.join(lines)
