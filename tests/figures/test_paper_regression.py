"""Non-regression against the published paper (opt-in: needs the method eval outputs, which are not distributed).

    EVAL_FIGURES_REF=<dir> EVAL_FIGURES_CONFIG=<eval config> EVAL_FIGURES_VIZ_CONFIG=<config of the renders> \
        pytest tests/figures/test_paper_regression.py

<dir> holds reference.json (the numbers of the paper's figures at full precision, written by the code that made
them), tex/ (the paper's tables) and qualitative/ (the paper's crops).
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.config import load_config
from src.figures.run import run_figures
from src.figures.spec import load_spec

REF = os.environ.get("EVAL_FIGURES_REF")
pytestmark = pytest.mark.skipif(not REF, reason="EVAL_FIGURES_REF not set (paper eval outputs not available)")
SPEC = Path(__file__).resolve().parents[2] / "config" / "figures" / "martine_neurips2026.yaml"
TOL = 1e-9


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("figures")
    spec = load_spec(SPEC)
    numbers = run_figures(load_config(os.environ["EVAL_FIGURES_CONFIG"]), spec, out,
                          ["prf", "radar", "resolution_views", "tables"])
    run_figures(load_config(os.environ["EVAL_FIGURES_VIZ_CONFIG"]), spec, out, ["qualitative"])
    return out, numbers, json.loads((Path(REF) / "reference.json").read_text())


def test_prf_curves(built):
    _, new, ref = built
    assert new["prf"]["thresholds"] == ref["prf"]["thresholds"]
    assert list(new["prf"]["methods"]) == list(ref["prf"]["methods"])
    for m, r in ref["prf"]["methods"].items():
        n = new["prf"]["methods"][m]
        assert (n["n_objects"], n["n_data"], n["n_gt"]) == (r["n_objects"], r["n_data"], r["n_gt"]), m
        for key in ("precision", "recall", "fscore"):
            assert np.max(np.abs(np.subtract(n[key], r[key]))) <= TOL, (m, key)


def test_radar(built):
    _, new, ref = built
    for m, axes in ref["radar"]["values"].items():
        for axis, v in axes.items():
            assert abs(new["radar"]["values"][m][axis] - v) <= TOL, (m, axis)
    # the old code keyed the attribute axes' objects by attribute name
    old_names = {"high_curvature": "curvature_values", "low_visibility": "visibility_count"}
    for key, objs in new["radar"]["objects"].items():
        m, axis = key.split("/")
        assert objs == ref["radar"]["objects"][f"{m}/{old_names.get(axis, axis)}"], key


def test_resolution_views(built):
    """The paper's values were read from a table of metrics.json rounded to 6 significant digits."""
    _, new, ref = built
    for m, r in ref["resolution_views"]["methods"].items():
        for key in ("views", "res"):
            got = new["resolution_views"]["methods"][m][key]
            assert [(x, ok) for x, _, ok in got] == [(x, ok) for x, _, ok in r[key]], (m, key)
            for (_, f_new, _), (_, f_ref, _) in zip(got, r[key]):
                assert abs(float(f"{f_new / 100:.6g}") * 100 - f_ref) <= TOL, (m, key)


@pytest.mark.parametrize("name", ["acquisition_coverage.tex", "all_cd.tex", "failures.tex"])
def test_tables_byte_identical(built, name):
    out, _, _ = built
    assert (out / name).read_bytes() == (Path(REF) / "tex" / name).read_bytes()


def test_qualitative_crops_pixel_identical(built):
    out, _, _ = built
    ref = sorted((Path(REF) / "qualitative").glob("*.png"))
    assert len(ref) == 20
    for f in ref:
        assert np.array_equal(np.array(Image.open(out / "qualitative" / f.name)), np.array(Image.open(f))), f.name
