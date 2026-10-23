"""Pooled precision / recall / F-score from the saved distances."""

import numpy as np
import pytest

from src.figures.pooling import load_distances, pooled_fscore, prf_curves

from .helpers import make_cell, make_config, make_gt


def test_prf_counts_strictly_below_threshold_in_float64():
    d2g = np.array([0.0, 0.5, 1.0, 2.0])
    g2d = np.array([0.25, 0.5, 3.0])
    p, r, f = prf_curves(d2g, g2d, np.array([0.0, 0.5, 1.0]))
    assert p.tolist() == [0.0, 0.25, 0.5]           # d < t, never d <= t
    assert r.tolist() == [0.0, 1 / 3, 2 / 3]
    assert f[0] == 0.0 and f[2] == pytest.approx(2 * 0.5 * (2 / 3) / (0.5 + 2 / 3))
    assert p.dtype == np.float64


def test_excluded_points_leave_both_directions(tmp_path):
    config = make_config(tmp_path)
    make_gt(config, "obj1", 4, excluded=[False, True, False, False])
    make_cell(config, "obj1", "alpha_d4", d2g=[0.1, 0.2, 0.3], g2d=[1.0, 2.0, 3.0, 4.0], idx=[0, 1, 3])
    d2g, g2d = load_distances(config, "obj1", "alpha_d4")
    assert g2d.tolist() == [1.0, 3.0, 4.0]
    assert d2g.tolist() == pytest.approx([0.1, 0.3])  # the point whose nearest GT is excluded goes too


def test_selection_mask_restricts_and_must_match_gt(tmp_path):
    config = make_config(tmp_path)
    make_gt(config, "obj1", 4)
    make_cell(config, "obj1", "alpha_d4", d2g=[0.1, 0.2, 0.3], g2d=[1.0, 2.0, 3.0, 4.0], idx=[0, 1, 3])
    d2g, g2d = load_distances(config, "obj1", "alpha_d4", select=np.array([True, True, False, False]))
    assert g2d.tolist() == [1.0, 2.0] and d2g.tolist() == pytest.approx([0.1, 0.2])
    with pytest.raises(ValueError, match="length"):
        load_distances(config, "obj1", "alpha_d4", select=np.ones(3, bool))


def test_pooled_fscore_concatenates_objects():
    pairs = [(np.array([0.1, 0.9]), np.array([0.1])), (np.array([]), np.array([0.9, 0.1]))]
    p, r = 1 / 2, 2 / 3
    assert pooled_fscore(pairs, 0.5) == pytest.approx(200 * p * r / (p + r))
    assert pooled_fscore([], 0.5) is None
