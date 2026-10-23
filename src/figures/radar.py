"""Radar of the F-score per surface property, pooled over the objects that have the property.

Property axes: an object counts on an axis if and only if its GT has challenges/<property>.npy (an object whose
whole surface has the property ships a mask of ones). Attribute axes select GT points by a per-point attribute
(attributes/<name>.npy). The GT `excluded` points are always removed.
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from ..config import Config
from .cells import Cell, Results
from .pooling import load_distances, pooled_fscore
from .spec import AttributeAxis, FiguresSpec
from .style import apply_style, method_style

# GT-wide exclusion in challenges/, not a property.
NOT_PROPERTIES = {"excluded"}


def discover_properties(config: Config, objects: list[str], properties: list[str]) -> dict[str, list[str]]:
    """Property -> objects, from the masks present. An unknown mask or a missing challenges/ is an error."""
    found = {p: [] for p in properties}
    for obj in objects:
        ch = config.get_gt_artifacts_dir(obj) / "challenges"
        if not ch.is_dir():
            raise FileNotFoundError(f"{obj}: no challenges directory {ch}")
        for path in sorted(ch.glob("*.npy")):
            if path.stem in NOT_PROPERTIES:
                continue
            if path.stem not in found:
                raise ValueError(f"{obj}: {path.name} is not a radar axis ({', '.join(properties)})")
            found[path.stem].append(obj)
    return found


def _attribute_mask(axis: AttributeAxis, values: np.ndarray) -> np.ndarray:
    mask = np.ones(len(values), bool)
    if axis.lt is not None:
        mask &= values < axis.lt
    if axis.ge is not None:
        mask &= values >= axis.ge
    if axis.le is not None:
        mask &= values <= axis.le
    return mask


def _gt_file(config: Config, obj: str, sub: str, name: str) -> Path:
    path = config.get_gt_artifacts_dir(obj) / sub / f"{name}.npy"
    if not path.is_file():
        raise FileNotFoundError(f"{obj}: expected GT file {path} is missing")
    return path


def radar_values(config: Config, spec: FiguresSpec, results: Results,
                 properties: Optional[dict] = None) -> tuple[dict, dict]:
    """({method: {axis: F in % or None}}, {(method, axis): pooled objects})."""
    s = spec.radar
    if properties is None:
        properties = discover_properties(config, results.objects, s.properties)
    values, pooled = {}, {}
    for m in s.methods:
        cell = Cell(m, s.tier)
        values[m] = {}
        axes = [(p, properties[p], lambda obj, p=p: np.load(_gt_file(config, obj, "challenges", p)).astype(bool))
                for p in s.properties]
        axes += [(a.key, results.objects,
                  lambda obj, a=a: _attribute_mask(a, np.load(_gt_file(config, obj, "attributes", a.attribute))))
                 for a in s.attributes]
        for key, objects, select in axes:
            pairs = []
            for obj in objects:
                if not results.ok(obj, cell):
                    continue  # failure of the method: out of every pool, declared in the failures table
                pairs.append(load_distances(config, obj, spec.cell_dir(cell), select=select(obj)))
                pooled.setdefault((m, key), []).append(obj)
            values[m][key] = pooled_fscore(pairs, s.threshold)
        print(f"  {spec.methods[m].display:15s}: "
              + ", ".join("--" if v is None else f"{v:.1f}" for v in values[m].values()))
    return values, pooled


def plot(spec: FiguresSpec, values: dict, out_dir: Path) -> Path:
    s = spec.radar
    apply_style(spec)
    axes_keys = list(s.properties) + [a.key for a in s.attributes]
    angles = np.linspace(0, 2 * np.pi, len(axes_keys), endpoint=False).tolist()
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "polar"})
    ax.set_theta_direction(-1)
    ax.set_theta_zero_location("N")
    max_val = 100
    for m in s.methods:
        scores = [values[m][k] if values[m][k] is not None else 0.0 for k in axes_keys]
        # outlines only: stacked translucent fills merge and hide the polygons underneath
        ax.plot(angles + angles[:1], scores + scores[:1], linewidth=2, markersize=7,
                label=spec.methods[m].display, **method_style(spec, m))
    ax.set_xticks(angles)
    ax.set_xticklabels([])
    for angle, key in zip(angles, axes_keys):
        deg = np.rad2deg(angle)
        rotation = -deg - 180 if 90 < deg <= 270 else -deg
        ax.text(angle, max_val + 8, s.labels.get(key, key), fontsize=15, ha="center", va="center",
                rotation=rotation, rotation_mode="anchor")
    ax.set_ylim(0, max_val)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(["25", "50", "75", "100"], fontsize=10)
    # legend on two rows, the second one centred
    handles, labels = ax.get_legend_handles_labels()
    n1 = (len(handles) + 1) // 2
    fig.add_artist(fig.legend(handles[:n1], labels[:n1], loc="lower center", bbox_to_anchor=(0.5, -0.04),
                              ncol=n1, fontsize=15, frameon=False, columnspacing=1.5))
    fig.legend(handles[n1:], labels[n1:], loc="lower center", bbox_to_anchor=(0.5, -0.09),
               ncol=len(handles) - n1, fontsize=15, frameon=False, columnspacing=1.5)
    plt.subplots_adjust(bottom=0.12)
    path = Path(out_dir) / s.output
    fig.savefig(path, bbox_inches="tight", dpi=300)
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    return path
