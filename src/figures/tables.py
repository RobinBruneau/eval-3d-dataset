"""LaTeX tables (booktabs): objects reconstructed per configuration, Chamfer distance per object, failures."""

from .cells import Cell, Results
from .spec import FiguresSpec


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


def _ranked_row(label: str, vals: list, rank: bool = True) -> str:
    """\\textbf the lowest value, \\underline the second lowest, -- for a missing one."""
    ok = sorted(v for v in vals if v is not None) if rank else []
    best = ok[0] if ok else None
    second = ok[1] if len(ok) > 1 else None
    out = []
    for v in vals:
        if v is None:
            out.append("--")
        elif v == best:
            out.append(f"\\textbf{{{v:.3f}}}")
        elif v == second:
            out.append(f"\\underline{{{v:.3f}}}")
        else:
            out.append(f"{v:.3f}")
    return f"{label} & " + " & ".join(out) + " \\\\"


def chamfer_table(spec: FiguresSpec, results: Results) -> str:
    """Chamfer distance per object (rows) and method x resolution tier with all views (columns that were run)."""
    s = spec.tables.chamfer
    cols = [Cell(m, t) for m in s.methods for t in s.tiers if spec.is_declared(Cell(m, t))]
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
        "\\begin{table}[htbp]",
        "\\centering",
        "\\footnotesize",
        f"\\caption{{{s.caption}}}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{l" + "r" * len(cols) + "}",
        "\\toprule",
        head,
        " ".join(rules),
        "Object & " + " & ".join(c.tier for c in cols) + " \\\\",
        "\\midrule",
    ]
    for obj in results.objects:
        lines.append(_ranked_row(spec.object_label(obj), [results.chamfer(obj, c) for c in cols]))
    means, counts = [], []
    for c in cols:
        vals = [v for v in (results.chamfer(o, c) for o in results.objects) if v is not None]
        means.append(sum(vals) / len(vals) if vals else None)
        counts.append(f"{len(vals)}/{len(results.objects)}")
    lines += ["\\midrule", _ranked_row("\\textit{Mean}", means, rank=False),
              "\\textit{Reconstructed} & " + " & ".join(counts) + " \\\\",
              "\\bottomrule", "\\end{tabular}", "}", f"\\label{{{s.label}}}", "\\end{table}"]
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
