"""Spec loading and cell bookkeeping: a declared cell without metrics.json is a failure."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from src.figures.cells import Cell, Results
from src.figures.spec import load_spec

from .helpers import make_cell, make_config, make_spec

REPO = Path(__file__).resolve().parents[2]


def test_martine_example_spec_loads():
    spec = load_spec(REPO / "config" / "figures" / "martine_neurips2026.yaml")
    # legend order: RealityScan leads the photogrammetry family; the tables keep their own order
    assert list(spec.methods)[:2] == ["realityscan", "meshroom"]
    paper_order = ["meshroom", "realityscan", "colmap", "pgsr", "neus2", "neuralangelo", "rnbneus2", "supernormal"]
    assert spec.tables.chamfer.methods == paper_order and spec.tables.failures.methods == paper_order
    assert spec.radar.threshold == 0.5 and spec.prf.n_thresholds == 201
    # RealityScan view subsets at d4 (Linux rerun, 25/10/2026), as for COLMAP, PGSR, NeuS2 and RNb-NeuS2
    assert spec.cells.declared["realityscan"].views == [40, 20, 10, 5]


def test_unknown_method_in_a_figure_is_rejected():
    with pytest.raises(ValidationError, match="gamma"):
        make_spec(prf={"methods": ["alpha", "gamma"], "tier": "d4", "output": "prf.pdf"})


def test_declared_cells_and_directory_names():
    spec = make_spec()
    assert spec.cell_dir(Cell("alpha", "d4")) == "alpha_d4"
    assert spec.cell_dir(Cell("alpha", "d4", 10)) == "alpha_d4_v10"
    assert spec.is_declared(Cell("alpha", "d2")) and spec.is_declared(Cell("alpha", "d4", 10))
    assert spec.is_declared(Cell("alpha", "d4", 84))  # all views is the plain d4 cell
    assert not spec.is_declared(Cell("beta", "d2")) and not spec.is_declared(Cell("beta", "d4", 10))


def test_results_ok_failure_and_not_declared(tmp_path):
    config = make_config(tmp_path)
    spec = make_spec()
    make_cell(config, "obj1", "alpha_d4", chamfer=0.25, fscore=(0.5, 0.75))
    res = Results(config, spec)
    c = Cell("alpha", "d4")
    assert res.ok("obj1", c) and not res.ok("obj2", c)
    assert res.n_ok(c) == 1 and res.failures(c) == ["obj2"]
    assert res.n_ok(Cell("beta", "d2")) is None  # never run: not a failure
    assert res.chamfer("obj1", c) == 0.25 and res.chamfer("obj2", c) is None
    assert res.fscore_at("obj1", c, 1.0) == 0.75


def test_nan_chamfer_is_missing(tmp_path):
    config = make_config(tmp_path)
    make_cell(config, "obj1", "alpha_d4", chamfer=float("nan"))
    assert Results(config, make_spec()).chamfer("obj1", Cell("alpha", "d4")) is None


def test_empty_reconstruction_is_a_failure(tmp_path):
    """evaluate writes metrics.json for a mesh with no vertices (empty_reconstruction: true): still a failure."""
    import json
    config = make_config(tmp_path)
    make_cell(config, "obj1", "alpha_d4")
    ev = make_cell(config, "obj2", "alpha_d4", chamfer=float("nan"))
    m = json.loads((ev / "metrics.json").read_text())
    (ev / "metrics.json").write_text(json.dumps({**m, "empty_reconstruction": True}))
    res = Results(config, make_spec())
    c = Cell("alpha", "d4")
    assert res.ok("obj1", c) and not res.ok("obj2", c)
    assert res.n_ok(c) == 1 and res.failures(c) == ["obj2"]
    assert res.metrics("obj2", c) is None and res.fscore_at("obj2", c, 0.5) is None
