"""LaTeX tables (booktabs): objects reconstructed per configuration, Chamfer distance per object, failures."""

from typing import Optional

import numpy as np

from ..core.metrics import compute_chamfer
from .cells import Cell, Results
from .pooling import load_distances, pooled_fscore
from .spec import ChamferTable, FiguresSpec


def _end(lines: list[str]) -> str:
    return "\n".join(lines + [""])


def coverage_table(spec: FiguresSpec, results: Results) -> str:
    """Objects reconstructed per method, for each view subset (views tier) and each resolution tier (all views)."""
    s = spec.tables.coverage
    vt, nv = spec.cells.views_tier, spec.cells.full_views

    def n(cell):
        k = results.n_ok(cell)
        return "--" if k is None else str(k)

    views, tiers = s.views, s.tiers
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\footnotesize",
        f"\\caption{{{s.caption}}}",
        f"\\label{{{s.label}}}",
        "\\begin{tabular}{l" + "c" * len(views) + "c" * len(tiers) + "}",
        "\\toprule",
        f" & \\multicolumn{{{len(views)}}}{{c}}{{Views ({vt})}} & "
        f"\\multicolumn{{{len(tiers)}}}{{c}}{{Resolution ({nv} views)}} \\\\",
        f"\\cmidrule(lr){{2-{1 + len(views)}}} \\cmidrule(lr){{{2 + len(views)}-{1 + len(views) + len(tiers)}}}",
        "Method & " + " & ".join(str(nv) if v == "all" else str(v) for v in views) + " & " + " & ".join(tiers)
        + " \\\\",
        "\\midrule",
    ]
    for m in s.methods:
        cells = [n(Cell(m, vt, None if v == "all" else int(v))) for v in views] + [n(Cell(m, t)) for t in tiers]
        lines.append(f"{spec.methods[m].display} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return _end(lines)


def _ranked(vals: list, fmt: str, lowest: bool = True, rank: bool = True) -> list[str]:
    """\\textbf the best value (lowest, or highest), \\underline the second best, -- for a missing one."""
    ok = sorted((v for v in vals if v is not None), reverse=not lowest) if rank else []
    best = ok[0] if ok else None
    second = ok[1] if len(ok) > 1 else None
    out = []
    for v in vals:
        if v is None:
            out.append("--")
        elif v == best:
            out.append(f"\\textbf{{{v:{fmt}}}}")
        elif v == second:
            out.append(f"\\underline{{{v:{fmt}}}}")
        else:
            out.append(f"{v:{fmt}}")
    return out


def _row(label: str, cells: list[str]) -> str:
    return f"{label} & " + " & ".join(cells) + " \\\\"


def _mean(vals: list) -> Optional[float]:
    ok = [v for v in vals if v is not None]
    return sum(ok) / len(ok) if ok else None


def _pooled(spec: FiguresSpec, results: Results, cell: Cell, fscore: Optional[float]):
    """(CD, F %) of the point distances of every reconstructed object concatenated, as if they were one mesh."""
    pairs = [load_distances(results.config, o, spec.cell_dir(cell)) for o in results.objects if results.ok(o, cell)]
    if not pairs:
        return None, None
    d2g = np.concatenate([d.ravel() for d, _ in pairs])
    g2d = np.concatenate([g.ravel() for _, g in pairs])
    cd = compute_chamfer(d2g, g2d, max_dist=results.config.evaluation.max_dist)["chamfer"]
    return float(cd), None if fscore is None else pooled_fscore(pairs, fscore)


def chamfer_table(spec: FiguresSpec, results: Results, table: Optional[ChamferTable] = None) -> str:
    """Chamfer distance per object (rows) and method x resolution tier with all views (columns that were run), and
    optionally the F-score at one threshold: 'CD / F' in each cell, or a CD block above an F block. Mean row: mean
    over the reconstructed objects, or their point distances pooled (CD of Eq. chamfer and F of the pooled points)."""
    s = table or spec.tables.chamfer
    env = "sidewaystable" if s.sideways else "table"
    cols = [Cell(m, t) for m in s.methods for t in s.method_tiers.get(m, s.tiers) if spec.is_declared(Cell(m, t))]
    groups = []
    for c in cols:
        if groups and groups[-1][0] == c.method:
            groups[-1][1].append(c.tier)
        else:
            groups.append((c.method, [c.tier]))
    head = "  & " + " & ".join(f"\\multicolumn{{{len(ts)}}}{{c}}{{{spec.methods[m].display}}}"
                               for m, ts in groups) + " \\\\"
    rules, k = [], 2
    for _, ts in groups:
        rules.append(f"\\cmidrule(lr){{{k}-{k + len(ts) - 1}}}")
        k += len(ts)
    lines = [
        f"\\begin{{{env}}}" + ("" if s.sideways else "[htbp]"),
        "\\centering",
        "\\footnotesize",
        f"\\caption{{{s.caption}}}",
        f"\\resizebox{{\\text{'height' if s.sideways else 'width'}}}{{!}}{{%",
        "\\begin{tabular}{l" + "r" * len(cols) + "}",
        "\\toprule",
        head,
        " ".join(rules),
        "Object & " + " & ".join(c.tier for c in cols) + " \\\\",
        "\\midrule",
    ]
    objs = results.objects
    cd = {o: [results.chamfer(o, c) for c in cols] for o in objs}
    counts = [f"{sum(cd[o][i] is not None for o in objs)}/{len(objs)}" for i in range(len(cols))]
    if s.mean == "pooled":
        cd_mean, f_mean = map(list, zip(*(_pooled(spec, results, c, s.fscore) for c in cols)))
    else:
        cd_mean = [_mean([cd[o][i] for o in objs]) for i in range(len(cols))]
    blocks = [(cd, cd_mean, ".3f", True)]
    if s.fscore is not None:
        f = {o: [None if v is None else 100 * v for v in (results.fscore_at(o, c, s.fscore) for c in cols)]
             for o in objs}
        if s.mean != "pooled":
            f_mean = [_mean([f[o][i] for o in objs]) for i in range(len(cols))]
        blocks.append((f, f_mean, ".1f", False))

    def ranked(vals_by_block, rank=True):
        """Cells of one row: one ranked list per metric, joined as 'CD / F' in the cell layout."""
        per = [_ranked(vals, fmt, lowest, rank) for vals, (_, _, fmt, lowest) in zip(vals_by_block, blocks)]
        return [" / ".join(c) if any(x != "--" for x in c) else "--" for c in zip(*per)]

    if s.fscore is None or s.layout == "cell":
        lines += [_row(spec.object_label(o), ranked([b[0][o] for b in blocks])) for o in objs]
        lines += ["\\midrule", _row("\\textit{Mean}", ranked([b[1] for b in blocks], rank=False)),
                  _row("\\textit{Reconstructed}", counts)]
    else:
        n = len(cols) + 1
        titles = ["Chamfer distance (mm) $\\downarrow$", f"F-score at {s.fscore:g}~mm (\\%) $\\uparrow$"]
        for i, ((vals, mean, fmt, lowest), title) in enumerate(zip(blocks, titles)):
            lines += ([] if i == 0 else ["\\midrule"]) + [f"\\multicolumn{{{n}}}{{c}}{{{title}}} \\\\", "\\midrule"]
            lines += [_row(spec.object_label(o), _ranked(vals[o], fmt, lowest)) for o in objs]
            lines += ["\\midrule", _row("\\textit{Mean}", _ranked(mean, fmt, rank=False)),
                      _row("\\textit{Reconstructed}", counts)]
    lines += ["\\bottomrule", "\\end{tabular}", "}", f"\\label{{{s.label}}}", f"\\end{{{env}}}"]
    return _end(lines)


def failures_table(spec: FiguresSpec, results: Results) -> str:
    """Every declared cell with at least one failure, and the objects it failed on."""
    s = spec.tables.failures
    methods = s.methods or list(spec.methods)
    cells = [c for c in spec.declared_cells() if c.method in methods and results.failures(c)]
    cells.sort(key=lambda c: methods.index(c.method))  # stable: tier order and dense-to-sparse views kept
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\footnotesize",
        f"\\caption{{{s.caption}}}",
        f"\\label{{{s.label}}}",
        "\\begin{tabular}{lcp{0.62\\textwidth}}",
        "\\toprule",
        "Configuration & Failed & Objects \\\\",
        "\\midrule",
    ]
    for c in cells:
        conf = f"{spec.methods[c.method].display} {c.tier}" + (f", {c.views} views" if c.views else "")
        fails = results.failures(c)
        lines.append(f"{conf} & {len(fails)}/{len(results.objects)} & "
                     + ", ".join(spec.object_label(o) for o in fails) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return _end(lines)
