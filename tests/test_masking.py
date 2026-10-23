"""GT exclusion masks: `excluded.npy` is the only implicit one; extras are explicit.

`invisible.npy` (auto-discovered until 2026-10) is no longer a feature: a file of that
name on disk must not change any score.
"""

import numpy as np
import pytest

from src.core.masking import load_gt_exclude

N_GT = 1000


def _make_gt(tmp_path, **masks):
    gt = tmp_path / "Groundtruth"
    (gt / "challenges").mkdir(parents=True)
    for name, arr in masks.items():
        np.save(gt / "challenges" / f"{name}.npy", arr)
    return gt


def test_invisible_on_disk_is_ignored(tmp_path):
    inv = np.zeros(N_GT, dtype=bool)
    inv[:80] = True
    gt = _make_gt(tmp_path, invisible=inv)
    exclude, applied = load_gt_exclude(gt, n_gt=N_GT, verbose=False)
    assert exclude is None and applied == []


def test_excluded_is_applied_without_any_flag(tmp_path):
    """23_whisk/excluded.npy legitimately drops ~50%: no fraction guard on it."""
    exc = np.zeros(N_GT, dtype=bool)
    exc[:507] = True
    gt = _make_gt(tmp_path, excluded=exc)
    exclude, applied = load_gt_exclude(gt, n_gt=N_GT, verbose=False)
    assert exclude.sum() == 507
    assert [p.rsplit("/", 1)[-1] for p in applied] == ["excluded.npy"]


def test_absent_excluded_is_not_an_error(tmp_path):
    gt = _make_gt(tmp_path)
    assert load_gt_exclude(gt, n_gt=N_GT, verbose=False) == (None, [])


def test_extra_exclude_is_ored_with_excluded(tmp_path):
    exc = np.zeros(N_GT, dtype=bool)
    exc[:10] = True
    extra = np.zeros(N_GT, dtype=bool)
    extra[5:20] = True
    gt = _make_gt(tmp_path, excluded=exc, holes=extra)
    exclude, applied = load_gt_exclude(gt, n_gt=N_GT, extra_exclude=["holes"], verbose=False)
    assert exclude.sum() == 20
    assert [p.rsplit("/", 1)[-1] for p in applied] == ["excluded.npy", "holes.npy"]


def test_length_mismatch_refuses_to_guess(tmp_path):
    gt = _make_gt(tmp_path, excluded=np.zeros(N_GT - 1, dtype=bool))
    with pytest.raises(ValueError, match="refusing to guess"):
        load_gt_exclude(gt, n_gt=N_GT, verbose=False)


def test_requested_but_missing_mask_raises(tmp_path):
    gt = _make_gt(tmp_path)
    with pytest.raises(FileNotFoundError):
        load_gt_exclude(gt, n_gt=N_GT, extra_exclude=["nope"], verbose=False)
