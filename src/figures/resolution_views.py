"""F-score of one object against the number of views and against the image resolution, failures drawn as crosses.

The F-scores are those of metrics.json. A declared cell without result is a failure (cross at F = 0); a cell
never run is not drawn, and a method without view subsets has no views curve.
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

from .cells import Cell, Results
from .spec import FiguresSpec
from .style import apply_style, method_style


def _point(spec: FiguresSpec, results: Results, x, cell: Cell):
    f = results.fscore_at(spec.resolution_views.object, cell, spec.resolution_views.threshold)
    return (x, f * 100 if f is not None else 0.0, f is not None)


def series(spec: FiguresSpec, results: Results) -> dict:
    """{method: {"views": [(n_views, F %, ok)], "res": [(tier index, F %, ok)]}} over the cells that were run."""
    s = spec.resolution_views
    vt = spec.cells.views_tier
    out = {}
    for m in s.methods:
        has_views = bool(spec.cells.declared.get(m) and spec.cells.declared[m].views)
        views = [_point(spec, results, v, Cell(m, vt, v)) for v in s.views
                 if has_views and spec.is_declared(Cell(m, vt, v))]
        res = [_point(spec, results, i, Cell(m, t)) for i, t in enumerate(s.tiers) if spec.is_declared(Cell(m, t))]
        out[m] = {"views": views, "res": res}
    return out


def plot(spec: FiguresSpec, data: dict, out_dir: Path) -> Path:
    s = spec.resolution_views
    apply_style(spec)
    fig, (ax_v, ax_r) = plt.subplots(2, 1, figsize=(3.4, 4.5))
    # failures of several methods fall on the same x: spread the crosses so that none hides another
    spread = {m: (i - (len(s.methods) - 1) / 2) * 0.06 for i, m in enumerate(s.methods)}
    for m in s.methods:
        style = method_style(spec, m)
        views, res = data[m]["views"], data[m]["res"]
        for ax, pts in ((ax_v, views), (ax_r, res)):
            ok = [(x, y) for x, y, good in pts if good]
            if ok:
                ax.plot(*zip(*ok), markersize=5,
                        label=spec.methods[m].display if ax is ax_v or not views else None, **style)
            fail = [x * (1 + spread[m]) if ax is ax_v else x + spread[m] for x, _, good in pts if not good]
            if fail:
                ax.plot(fail, [0] * len(fail), linestyle="none", marker="x", markersize=6, markeredgewidth=1.4,
                        color=style["color"], clip_on=False, zorder=5)
            for x, y, good in pts:
                print(f"{spec.methods[m].display:12s} {'views' if ax is ax_v else 'res'} x={x}: "
                      f"{'F=%.1f' % y if good else 'FAILURE'}")
    ax_v.plot([], [], linestyle="none", marker="x", color="#555555", label="Failure")
    ax_v.set_xscale("log")
    ax_v.set_xticks(s.views)
    ax_v.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax_v.minorticks_off()
    ax_v.set_xlim(4, 100)
    ax_v.set_xlabel("Number of views (log scale)")
    ax_r.set_xticks(range(len(s.tiers)))
    ax_r.set_xticklabels([s.tier_labels.get(t, t) for t in s.tiers])
    ax_r.set_xlabel("Image resolution")
    for ax in (ax_v, ax_r):
        ax.set_ylabel("F-score (%)")
        ax.set_ylim(0, 100)
        ax.grid(True, linestyle=":", alpha=0.5)
    handles, labels = ax_v.get_legend_handles_labels()
    h_r, l_r = ax_r.get_legend_handles_labels()
    by_label = dict(zip(labels + l_r, handles + h_r))
    order = [spec.methods[m].display for m in s.methods] + ["Failure"]
    ax_v.legend([by_label[k] for k in order if k in by_label], [k for k in order if k in by_label],
                fontsize=6.5, loc="lower right", ncol=2)
    fig.tight_layout()
    path = Path(out_dir) / s.output
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=200)
    plt.close(fig)
    return path
