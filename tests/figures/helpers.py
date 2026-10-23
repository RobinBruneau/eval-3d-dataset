"""Synthetic eval roots and specs for the figures tests."""

import json
from pathlib import Path

import numpy as np

from src.config import Config
from src.figures.spec import FiguresSpec


def make_config(tmp_path: Path, objects=("obj1", "obj2")) -> Config:
    return Config(
        dataset={"name": "t", "objects": list(objects)},
        paths={
            "data_root": str(tmp_path / "data" / "{object}"),
            "eval_root": str(tmp_path / "eval" / "{object}"),
            "gt_artifacts_root": str(tmp_path / "gt" / "{object}"),
        },
        evaluation={"max_dist": 4.0, "fscore_thresholds": [0.5, 1.0]},
    )


def make_gt(config: Config, obj: str, n: int, excluded=None, challenges=None, attributes=None) -> Path:
    gt = config.get_gt_artifacts_dir(obj)
    (gt / "challenges").mkdir(parents=True, exist_ok=True)
    (gt / "attributes").mkdir(parents=True, exist_ok=True)
    np.save(gt / "gt_pcd.npy", np.zeros((n, 3), np.float32))
    if excluded is not None:
        np.save(gt / "challenges" / "excluded.npy", np.asarray(excluded, bool))
    for name, mask in (challenges or {}).items():
        np.save(gt / "challenges" / f"{name}.npy", np.asarray(mask, bool))
    for name, values in (attributes or {}).items():
        np.save(gt / "attributes" / f"{name}.npy", np.asarray(values, np.float32))
    return gt


def make_cell(config: Config, obj: str, dirname: str, d2g=None, g2d=None, idx=None, chamfer=0.1,
              fscore=(0.5, 0.6)) -> Path:
    """An evaluated cell: metrics.json (thresholds 0.5, 1.0) and, if given, the distance arrays."""
    ev = config.get_eval_dir(obj, dirname)
    (ev / "distances").mkdir(parents=True)
    (ev / "metrics.json").write_text(json.dumps({"chamfer": chamfer, "thresholds": [0.5, 1.0],
                                                 "fscore": list(fscore)}))
    if d2g is not None:
        np.save(ev / "distances" / "data2gt_dist.npy", np.asarray(d2g, np.float32))
        np.save(ev / "distances" / "gt2data_dist.npy", np.asarray(g2d, np.float32))
        np.save(ev / "distances" / "data2gt_idx.npy", np.asarray(idx, np.int64))
    return ev


def make_spec(**sections) -> FiguresSpec:
    base = {
        "methods": {
            "alpha": {"display": "Alpha", "color": "#111111", "marker": "o"},
            "beta": {"display": "Beta", "color": "#222222", "marker": "s", "linestyle": "--", "hollow": True},
        },
        "cells": {
            "tiers": ["d1", "d2", "d4"],
            "views_tier": "d4",
            "full_views": 84,
            "declared": {"alpha": {"tiers": ["d2", "d4"], "views": [10]}, "beta": {"tiers": ["d4"]}},
        },
    }
    base.update(sections)
    return FiguresSpec(**base)
