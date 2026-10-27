"""Precision, recall and F-score curves pooled over all objects, one curve per method."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..config import Config
from .cells import Cell, Results
from .pooling import load_distances, prf_curves
from .spec import FiguresSpec
from .style import apply_style, method_style


def compute(config: Config, spec: FiguresSpec, results: Results) -> dict:
    s = spec.prf
    thresholds = np.linspace(0, s.max_threshold, s.n_thresholds)
    out = {"thresholds": thresholds.tolist(), "methods": {}}
    for m in s.methods:
        cell = Cell(m, s.tier)
        pairs = [load_distances(config, o, spec.cell_dir(cell)) for o in results.objects if results.ok(o, cell)]
        if not pairs:
            print(f"  {spec.methods[m].display:20s}  -- no reconstruction, skipped")
            continue
        data2gt = np.concatenate([d.ravel() for d, _ in pairs])
        gt2data = np.concatenate([g.ravel() for _, g in pairs])
        p, r, f = prf_curves(data2gt, gt2data, thresholds)
        out["methods"][m] = {"n_objects": len(pairs), "n_data": len(data2gt), "n_gt": len(gt2data),
                             "precision": p.tolist(), "recall": r.tolist(), "fscore": f.tolist()}
        print(f"  {spec.methods[m].display:20s}  objects={len(pairs):2d}/{len(results.objects)}  "
              f"data_pts={len(data2gt):>10,}  gt_pts={len(gt2data):>10,}")
    for tau in s.report:
        i = int(np.argmin(np.abs(thresholds - tau)))
        print(f"[pooled tau={thresholds[i]:.2f}] " + "  ".join(
            f"{spec.methods[m].display}: P={r['precision'][i] * 100:.1f} R={r['recall'][i] * 100:.1f} "
            f"F={r['fscore'][i] * 100:.1f} (n={r['n_objects']})" for m, r in out["methods"].items()))
    return out


def plot(spec: FiguresSpec, numbers: dict, out_dir: Path) -> Path:
    s = spec.prf
    apply_style(spec, s.rc)
    thresholds = np.asarray(numbers["thresholds"])
    width, height = s.figsize
    fig, axes = plt.subplots(1, 3, figsize=(width, height))
    curve_keys = ["precision", "recall", "fscore"]
    titles = ["Precision (%)", "Recall (%)", "F-score (%)"]
    for i, (ax, key, title) in enumerate(zip(axes, curve_keys, titles)):
        for m, res in numbers["methods"].items():
            # sparse markers, shifted per panel: redundant encoding for print
            style = method_style(spec, m, markevery=(i * 7 % 20, 20), markersize=3.6, markeredgewidth=0.8)
            ax.plot(thresholds, np.asarray(res[key]) * 100.0,
                    label=spec.methods[m].display if key == curve_keys[0] else None, **style)
        if i > 0:
            ax.set_yticklabels([])
        ax.set_ylim(0, 100)
        ax.set_xlim(0, s.max_threshold)
        ax.grid(True, linestyle=":", alpha=0.8)
        ax.set_yticks(np.arange(0, 101, 25))
        ax.tick_params(axis="y", labelsize=7, pad=1)
        ax.tick_params(axis="x", labelsize=7, pad=1)
        ax.set_title(title)
    axes[1].set_xlabel(r"$\tau$ (mm)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    # vertical margins fixed in inches (titles above, legend and tau label below), whatever the height
    plt.subplots_adjust(wspace=0.08, left=0.06, right=0.99, top=1 - 0.22 / height, bottom=0.616 / height)
    path = Path(out_dir) / s.output
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=200)
    plt.close(fig)
    return path
