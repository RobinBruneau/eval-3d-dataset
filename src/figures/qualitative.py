"""Qualitative figure: error renders of `eval-pipeline visualize`, padded with a white margin, and their colorbar.

The renders themselves come from `visualize` (e.g. --metrics accuracy --cmap cividis --max-dist 4 --exclude-mode
remove --no-integrated-colorbar); this module only crops them for the paper and draws the shared colorbar.
"""

from pathlib import Path

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from ..config import Config
from .cells import Cell, Results
from .spec import FiguresSpec
from .style import apply_style


def pad_render(img: Image.Image, margin: int) -> Image.Image:
    arr = np.array(img.convert("RGB"))
    padded = np.full((arr.shape[0] + 2 * margin, arr.shape[1] + 2 * margin, 3), 255, dtype=np.uint8)
    padded[margin:margin + arr.shape[0], margin:margin + arr.shape[1]] = arr
    return Image.fromarray(padded)


def crop_renders(config: Config, spec: FiguresSpec, results: Results, out_dir: Path) -> list[Path]:
    """<output_dir>/<object>_<method>.png for every object x method; a missing render is an error."""
    s = spec.qualitative
    out = Path(out_dir) / s.output_dir
    out.mkdir(parents=True, exist_ok=True)
    written, missing = [], []
    for obj, view in s.views.items():
        for m in s.methods:
            path = config.get_method_dir(obj, spec.cell_dir(Cell(m, s.tier))) / s.render_dir / view
            if not path.is_file():
                missing.append(str(path))
                continue
            dest = out / f"{obj}_{m}.png"
            pad_render(Image.open(path), s.margin).save(dest)
            written.append(dest)
            print(f"{dest.name} {Image.open(dest).size}")
    if missing:
        raise FileNotFoundError("missing renders (run `eval-pipeline visualize`): " + ", ".join(missing))
    return written


def colorbar(spec: FiguresSpec, out_dir: Path) -> Path:
    c = spec.qualitative.colorbar
    apply_style(spec)
    fig, ax = plt.subplots(figsize=(8, 0.12))
    sm = plt.cm.ScalarMappable(cmap=matplotlib.colormaps.get_cmap(c.cmap),
                               norm=mcolors.Normalize(vmin=c.vmin, vmax=c.vmax))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=ax, orientation="horizontal")
    cb.set_label(c.label, fontsize=8, labelpad=2)
    cb.ax.tick_params(labelsize=7)
    path = Path(out_dir) / c.output
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return path
