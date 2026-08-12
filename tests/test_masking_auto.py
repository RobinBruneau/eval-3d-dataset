"""Auto-discovered exclusion masks: polarity, guard, dedup, declaration.

These tests exist because the failure mode they cover is silent. An inverted mask does
not crash: it drops 92% of the ground truth and returns plausible-looking numbers. The
guard must therefore fail *closed* (raise), and this file is what proves it does.

Run: eval_pipeline/venv/bin/python -m pytest tests/test_masking_auto.py -v
"""

import numpy as np
import pytest

from src.core.masking import (
    AUTO_EXCLUDE,
    MAX_AUTO_DROP_FRACTION,
    data_in_voxels,
    load_gt_exclude,
)

N_GT = 1000


def _make_gt(tmp_path, *, invisible=None, excluded=None, extra=None):
    gt = tmp_path / "Groundtruth"
    (gt / "challenges").mkdir(parents=True)
    if invisible is not None:
        np.save(gt / "challenges" / "invisible.npy", invisible)
    if excluded is not None:
        np.save(gt / "challenges" / "excluded.npy", excluded)
    for name, arr in (extra or {}).items():
        np.save(gt / "challenges" / name, arr)
    return gt


def test_invisible_is_auto_discovered():
    assert "invisible.npy" in AUTO_EXCLUDE


def test_data_in_voxels_marks_points_inside_any_voxel():
    points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [5.0, 5.0, 5.0]])
    voxels = np.array([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 3.0, 3.0, 3.0]])

    inside = data_in_voxels(points, voxels)

    assert inside.dtype == bool
    assert inside.tolist() == [True, True, True, False]


def test_applied_without_any_flag(tmp_path):
    """The whole point: no caller passes anything, the mask still applies."""
    inv = np.zeros(N_GT, dtype=bool)
    inv[:80] = True
    gt = _make_gt(tmp_path, invisible=inv)

    exclude, applied = load_gt_exclude(gt, n_gt=N_GT, verbose=False)

    assert exclude is not None
    assert exclude.sum() == 80
    assert [p.rsplit("/", 1)[-1] for p in applied] == ["invisible.npy"]


def test_ored_with_excluded(tmp_path):
    inv = np.zeros(N_GT, dtype=bool)
    inv[:80] = True
    exc = np.zeros(N_GT, dtype=bool)
    exc[70:100] = True
    gt = _make_gt(tmp_path, invisible=inv, excluded=exc)

    exclude, applied = load_gt_exclude(gt, n_gt=N_GT, verbose=False)

    assert exclude.sum() == 100  # union, not sum
    assert [p.rsplit("/", 1)[-1] for p in applied] == ["excluded.npy", "invisible.npy"]


def test_absent_mask_is_not_an_error(tmp_path):
    """Objects predating the mask must still evaluate, just without it."""
    gt = _make_gt(tmp_path)
    exclude, applied = load_gt_exclude(gt, n_gt=N_GT, verbose=False)
    assert exclude is None
    assert applied == []


def test_no_double_declaration_when_also_requested(tmp_path):
    """A caller still passing --extra-exclude invisible must not list it twice."""
    inv = np.zeros(N_GT, dtype=bool)
    inv[:80] = True
    gt = _make_gt(tmp_path, invisible=inv)

    exclude, applied = load_gt_exclude(
        gt, n_gt=N_GT, extra_exclude=["invisible"], verbose=False
    )

    assert exclude.sum() == 80
    assert applied == [p for p in applied if p.endswith("invisible.npy")]
    assert len(applied) == 1, f"declared twice: {applied}"


# --- the guards, which must FAIL CLOSED -------------------------------------------

def test_inverted_count_array_is_refused(tmp_path):
    """The trap: a raw visibility_count array cast to bool drops every VISIBLE point.

    ~92% of a real GT. Must raise, not silently evaluate on the 8% nobody can see.
    """
    counts = np.random.default_rng(0).integers(0, 85, size=N_GT).astype(np.int32)
    counts[:80] = 0  # 8% genuinely invisible -> 92% would be dropped if inverted
    gt = _make_gt(tmp_path, invisible=counts)

    with pytest.raises(ValueError, match="inverted mask"):
        load_gt_exclude(gt, n_gt=N_GT, verbose=False)


def test_guard_threshold_is_the_documented_one(tmp_path):
    """Just under the limit passes, just over raises. No silent clamp."""
    n_under = int(N_GT * MAX_AUTO_DROP_FRACTION) - 1
    inv = np.zeros(N_GT, dtype=bool)
    inv[:n_under] = True
    gt = _make_gt(tmp_path, invisible=inv)
    exclude, _ = load_gt_exclude(gt, n_gt=N_GT, verbose=False)
    assert exclude.sum() == n_under

    inv2 = np.zeros(N_GT, dtype=bool)
    inv2[: int(N_GT * MAX_AUTO_DROP_FRACTION) + 1] = True
    gt2 = _make_gt(tmp_path / "b", invisible=inv2)
    with pytest.raises(ValueError, match="inverted mask"):
        load_gt_exclude(gt2, n_gt=N_GT, verbose=False)


def test_guard_does_not_second_guess_excluded(tmp_path):
    """23_whisk/excluded.npy legitimately drops 50.65%. The guard is auto-only."""
    exc = np.zeros(N_GT, dtype=bool)
    exc[:507] = True  # 50.7%, like 23_whisk
    gt = _make_gt(tmp_path, excluded=exc)
    exclude, applied = load_gt_exclude(gt, n_gt=N_GT, verbose=False)
    assert exclude.sum() == 507


def test_length_mismatch_still_refuses_to_guess(tmp_path):
    inv = np.zeros(N_GT - 1, dtype=bool)
    gt = _make_gt(tmp_path, invisible=inv)
    with pytest.raises(ValueError, match="refusing to guess"):
        load_gt_exclude(gt, n_gt=N_GT, verbose=False)


def test_requested_but_missing_mask_still_raises(tmp_path):
    """An auto mask may be absent; one explicitly asked for may not."""
    gt = _make_gt(tmp_path)
    with pytest.raises(FileNotFoundError):
        load_gt_exclude(gt, n_gt=N_GT, extra_exclude=["nope"], verbose=False)


def test_use_auto_false_reproduces_the_unmasked_baseline(tmp_path):
    """The reversal path must stay available for before/after comparison."""
    inv = np.zeros(N_GT, dtype=bool)
    inv[:80] = True
    gt = _make_gt(tmp_path, invisible=inv)
    exclude, applied = load_gt_exclude(gt, n_gt=N_GT, use_auto=False, verbose=False)
    assert exclude is None and applied == []
