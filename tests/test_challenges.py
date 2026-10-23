"""Tests for the challenge (property) path: apply_challenge_masks + aggregate_challenges.

Rule (Baptiste, 2026-10-22): a property counts for an object iff its
`challenges/<property>.npy` exists; a whole-object property ships as an all-ones mask.
`excluded.npy` marks unreliable GT zones: never evaluated, never a property.
"""

from pathlib import Path

import numpy as np

from src.config import Config
from src.pipeline.aggregate import aggregate_challenges
from src.pipeline.apply_masks import apply_challenge_masks

N_GT = 40
N_DATA = 30


def _make_config(tmp_path: Path) -> Config:
    return Config(
        dataset={"name": "t", "objects": ["obj1"], "methods": ["m"]},
        paths={
            "data_root": str(tmp_path / "data" / "{object}"),
            "eval_root": str(tmp_path / "eval" / "{object}"),
            "gt_artifacts_root": str(tmp_path / "gt" / "{object}"),
        },
        evaluation={"max_dist": 4.0, "fscore_thresholds": [0.5, 1.0]},
    )


def _setup(tmp_path: Path, excluded: np.ndarray | None) -> tuple[Config, Path]:
    """One object, one method: distances + an all-ones `lambertian.npy` (+ `excluded.npy`)."""
    config = _make_config(tmp_path)
    challenges = config.get_gt_artifacts_dir("obj1") / "challenges"
    challenges.mkdir(parents=True)
    np.save(challenges / "lambertian.npy", np.ones(N_GT, dtype=bool))
    if excluded is not None:
        np.save(challenges / "excluded.npy", excluded)

    method_dir = config.get_method_dir("obj1", "m")
    (method_dir / "results_raw").mkdir(parents=True)
    (method_dir / "results_raw" / "mesh.ply").write_text("ply\n")
    eval_dir = config.get_eval_dir("obj1", "m")
    dist = eval_dir / "distances"
    dist.mkdir(parents=True)
    rng = np.random.default_rng(0)
    np.save(dist / "gt2data_dist.npy", rng.uniform(0, 2, N_GT).astype(np.float32))
    np.save(dist / "gt2data_idx.npy", rng.integers(0, N_DATA, N_GT))
    np.save(dist / "data2gt_dist.npy", rng.uniform(0, 2, N_DATA).astype(np.float32))
    # data point j's nearest GT point is j: the first N_DATA GT points own one data point each
    np.save(dist / "data2gt_idx.npy", np.arange(N_DATA))
    np.save(dist / "gt_points.npy", rng.uniform(0, 1, (N_GT, 3)))
    np.save(dist / "data_points.npy", rng.uniform(0, 1, (N_DATA, 3)))
    (eval_dir / "metrics.json").write_text("{}")
    return config, eval_dir


def test_all_ones_property_never_evaluates_excluded_points(tmp_path):
    excluded = np.zeros(N_GT, dtype=bool)
    excluded[:10] = True  # 10 GT points, 10 of them owning a data point
    config, eval_dir = _setup(tmp_path, excluded)

    apply_challenge_masks(config, "obj1", "m")

    zone = np.load(eval_dir / "zones" / "challenges" / "lambertian.npz")
    dist = eval_dir / "distances"
    np.testing.assert_array_equal(zone["gt2data"], np.load(dist / "gt2data_dist.npy")[10:])
    np.testing.assert_array_equal(zone["data2gt"], np.load(dist / "data2gt_dist.npy")[10:])


def test_excluded_is_not_a_property(tmp_path):
    excluded = np.zeros(N_GT, dtype=bool)
    excluded[:10] = True
    config, eval_dir = _setup(tmp_path, excluded)

    apply_challenge_masks(config, "obj1", "m")
    results = aggregate_challenges(config, methods=["m"])

    assert not (eval_dir / "zones" / "challenges" / "excluded.npz").exists()
    assert set(results["m"]) == {"lambertian"}
    assert results["m"]["lambertian"]["n_gt_points"] == N_GT - 10


def test_without_excluded_the_whole_mask_counts(tmp_path):
    config, eval_dir = _setup(tmp_path, excluded=None)

    apply_challenge_masks(config, "obj1", "m")

    zone = np.load(eval_dir / "zones" / "challenges" / "lambertian.npz")
    assert len(zone["gt2data"]) == N_GT
    assert len(zone["data2gt"]) == N_DATA
