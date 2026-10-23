"""Radar: property axes come from the challenges/*.npy present, attribute axes from attributes/."""

import pytest

from src.figures.cells import Results
from src.figures.radar import discover_properties, radar_values

from .helpers import make_cell, make_config, make_gt, make_spec

RADAR = {
    "methods": ["alpha"], "tier": "d4", "threshold": 0.5,
    "properties": ["lambertian", "specular"],
    "attributes": [{"key": "low_vis", "attribute": "visibility_count", "ge": 1, "le": 3},
                   {"key": "convex", "attribute": "curvature_values", "lt": -0.2}],
    "output": "radar.pdf",
}


def test_axes_listed_from_npy_present(tmp_path):
    config = make_config(tmp_path)
    make_gt(config, "obj1", 3, excluded=[0, 0, 0], challenges={"lambertian": [1, 1, 1]})
    make_gt(config, "obj2", 3, challenges={"specular": [1, 0, 0]})
    props = discover_properties(config, ["obj1", "obj2"], ["lambertian", "specular"])
    assert props == {"lambertian": ["obj1"], "specular": ["obj2"]}


def test_unknown_property_is_an_error(tmp_path):
    config = make_config(tmp_path)
    make_gt(config, "obj1", 3, challenges={"glossy": [1, 1, 1]})
    with pytest.raises(ValueError, match="glossy"):
        discover_properties(config, ["obj1"], ["lambertian"])


def test_object_without_challenges_is_an_error(tmp_path):
    config = make_config(tmp_path)
    make_gt(config, "obj1", 3)
    with pytest.raises(FileNotFoundError, match="obj2"):
        discover_properties(config, ["obj1", "obj2"], ["lambertian"])


def test_values_pool_masked_points_and_skip_failures(tmp_path):
    config = make_config(tmp_path)
    # obj1: 4 GT points, lambertian on the first three, the second one excluded.
    make_gt(config, "obj1", 4, excluded=[0, 1, 0, 0], challenges={"lambertian": [1, 1, 1, 0]},
            attributes={"visibility_count": [0, 2, 3, 5], "curvature_values": [-1, -1, 0, -0.2]})
    make_cell(config, "obj1", "alpha_d4", d2g=[0.1, 0.9, 0.2, 0.1], g2d=[0.1, 0.1, 0.9, 0.1], idx=[0, 1, 2, 3])
    make_gt(config, "obj2", 2, challenges={"lambertian": [1, 1]})  # alpha failed on obj2: no cell
    spec = make_spec(radar=RADAR)
    values, objects = radar_values(config, spec, Results(config, spec))
    # lambertian keeps GT 0 and 2: recall 1/2; data points 0 and 2 (nearest GT 0, 2): precision 2/2.
    assert values["alpha"]["lambertian"] == pytest.approx(100 * 2 * 1 * 0.5 / 1.5)
    assert values["alpha"]["specular"] is None
    assert objects[("alpha", "lambertian")] == ["obj1"]
    # low_vis = 1 <= v <= 3 and not excluded: GT 2 only (recall 0); data point 2 (precision 1).
    assert values["alpha"]["low_vis"] == 0.0
    # convex = curvature < -0.2 strictly, not excluded: GT 0 (recall 1, precision 1).
    assert values["alpha"]["convex"] == pytest.approx(100.0)


def test_expected_mask_missing_is_an_error(tmp_path):
    config = make_config(tmp_path, objects=["obj1"])
    make_gt(config, "obj1", 2, challenges={"lambertian": [1, 1]})
    make_cell(config, "obj1", "alpha_d4", d2g=[0.1], g2d=[0.1, 0.1], idx=[0])
    spec = make_spec(radar=RADAR)
    props = discover_properties(config, ["obj1"], ["lambertian", "specular"])
    (config.get_gt_artifacts_dir("obj1") / "challenges" / "lambertian.npy").unlink()
    with pytest.raises(FileNotFoundError, match="lambertian"):
        radar_values(config, spec, Results(config, spec), properties=props)
