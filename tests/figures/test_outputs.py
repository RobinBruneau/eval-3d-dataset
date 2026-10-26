"""Views/resolution series, qualitative crops, and the `figures` command end to end."""

import json

import numpy as np
import pytest
from click.testing import CliRunner
from PIL import Image

from src.cli import main
from src.figures.cells import Results
from src.figures.qualitative import pad_render
from src.figures.resolution_views import series

from .helpers import make_cell, make_config, make_gt, make_spec

RV = {"object": "obj1", "methods": ["alpha", "beta"], "threshold": 1.0, "views": [10, 84],
      "tiers": ["d1", "d2", "d4"], "tier_labels": {"d1": "Full", "d2": "Half", "d4": "Quarter"},
      "output": "rv.pdf"}


def test_series_marks_failures_and_skips_cells_never_run(tmp_path):
    config = make_config(tmp_path)
    make_cell(config, "obj1", "alpha_d4", fscore=(0.5, 0.8))
    make_cell(config, "obj1", "beta_d4", fscore=(0.5, 0.6))
    spec = make_spec(resolution_views=RV)
    s = series(spec, Results(config, spec))
    assert s["alpha"]["views"] == [(10, 0.0, False), (84, pytest.approx(80.0), True)]
    assert s["alpha"]["res"] == [(1, 0.0, False), (2, pytest.approx(80.0), True)]  # d1 never run: no point
    assert s["beta"]["views"] == []  # no view subset declared: no views curve at all
    assert s["beta"]["res"] == [(2, pytest.approx(60.0), True)]


def test_pad_render_adds_white_margin():
    img = Image.fromarray(np.zeros((2, 3, 3), np.uint8))
    out = np.array(pad_render(img, 10))
    assert out.shape == (22, 23, 3)
    assert out[0, 0].tolist() == [255, 255, 255] and out[10, 10].tolist() == [0, 0, 0]


def test_figures_command_writes_tables_and_numbers(tmp_path):
    config = make_config(tmp_path, objects=["obj1"])
    make_gt(config, "obj1", 2, challenges={"lambertian": [1, 1]})
    make_cell(config, "obj1", "alpha_d4", d2g=[0.1], g2d=[0.1, 0.1], idx=[0], chamfer=0.2)
    cfg = tmp_path / "eval.yaml"
    cfg.write_text(json.dumps(config.model_dump(mode="json")))
    spec = tmp_path / "spec.yaml"
    s = make_spec(tables={"failures": {"caption": "F.", "label": "tab:f", "output": "failures.tex"}},
                  prf={"methods": ["alpha"], "tier": "d4", "max_threshold": 1.0, "n_thresholds": 3,
                       "output": "prf.pdf"})
    spec.write_text(json.dumps(s.model_dump(mode="json")))
    out = tmp_path / "out"
    r = CliRunner().invoke(main, ["-c", str(cfg), "figures", "--spec", str(spec), "--out", str(out)])
    assert r.exit_code == 0, r.output
    assert (out / "failures.tex").is_file() and (out / "prf.pdf").is_file()
    numbers = json.loads((out / "numbers.json").read_text())
    assert numbers["prf"]["methods"]["alpha"]["fscore"] == [0.0, 1.0, 1.0]


def _prf_numbers():
    curve = [0.0, 0.5, 1.0]
    return {"thresholds": [0.0, 0.5, 1.0],
            "methods": {"alpha": {"precision": curve, "recall": curve, "fscore": curve}}}


def test_prf_figure_size_from_the_spec(tmp_path):
    """figsize (inches) is read from the spec; the default keeps the paper's 10 x 2.2 strip."""
    from PIL import Image

    from src.figures import prf
    base = {"methods": ["alpha"], "tier": "d4", "max_threshold": 1.0, "n_thresholds": 3, "output": "prf.pdf"}
    prf.plot(make_spec(prf=base), _prf_numbers(), tmp_path)
    assert Image.open(tmp_path / "prf.png").size == (2000, 440)
    tall = tmp_path / "tall"
    tall.mkdir()
    prf.plot(make_spec(prf={**base, "figsize": [10, 5]}), _prf_numbers(), tall)
    assert Image.open(tall / "prf.png").size == (2000, 1000)
