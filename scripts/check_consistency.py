#!/usr/bin/env python3
"""Check consistency of all .npy arrays against gt_pcd for each object.

Verifies:
- gt_pcd.npy exists
- visibility_count.npy: same length as gt_pcd, max <= 84
- curvature_values.npy: same length as gt_pcd
- challenges/*.npy (excluded, properties): same length as gt_pcd
- Per-method eval_results/distances/: gt2data_dist.npy and data2gt_dist.npy exist
"""

import json
import sys
from pathlib import Path

import numpy as np

EVAL_ROOT = Path("/home/babrument/dev/alicevision/dataset_temp/eval")

SKIP_METHODS = {"Groundtruth", "camera_debug", "slurm_logs", "slurm_scripts", "tables", "debug_cleanup",
                "curvature_tests"}


def check_object(obj_dir: Path) -> list[str]:
    obj = obj_dir.name
    gt = obj_dir / "Groundtruth"
    issues = []

    # gt_pcd
    pcd_path = gt / "gt_pcd.npy"
    if not pcd_path.exists():
        issues.append(f"MISSING gt_pcd.npy")
        return issues
    pcd_len = len(np.load(pcd_path))

    # Visibility
    vis_path = gt / "attributes" / "visibility_count.npy"
    if vis_path.exists():
        v = np.load(vis_path)
        if len(v) != pcd_len:
            issues.append(f"visibility: len={len(v)} != pcd={pcd_len}")
        elif v.max() > 84:
            issues.append(f"visibility: max={v.max()} > 84")
        elif v.max() == 0:
            issues.append(f"visibility: all zeros")
    else:
        issues.append(f"MISSING visibility_count.npy")

    # Curvature
    curv_path = gt / "attributes" / "curvature_values.npy"
    if curv_path.exists():
        c = np.load(curv_path)
        if len(c) != pcd_len:
            issues.append(f"curvature: len={len(c)} != pcd={pcd_len}")
    else:
        issues.append(f"MISSING curvature_values.npy")

    # Excluded
    exc_path = gt / "challenges" / "excluded.npy"
    if exc_path.exists():
        e = np.load(exc_path)
        if len(e) != pcd_len:
            issues.append(f"excluded: len={len(e)} != pcd={pcd_len}")

    # Properties: every challenges/*.npy must match gt_pcd
    for npy_path in sorted((gt / "challenges").glob("*.npy")):
        if npy_path.stem == "excluded":
            continue
        arr = np.load(npy_path)
        if len(arr) != pcd_len:
            issues.append(f"{npy_path.stem}: len={len(arr)} != pcd={pcd_len}")

    # Per-method distances
    for method_dir in sorted(obj_dir.iterdir()):
        if not method_dir.is_dir() or method_dir.name in SKIP_METHODS:
            continue
        if method_dir.name.startswith("normals"):
            continue
        m = method_dir.name
        dist_dir = method_dir / "eval_results" / "distances"
        metrics_path = method_dir / "eval_results" / "metrics.json"
        has_raw = (method_dir / "results_raw").is_dir() and any((method_dir / "results_raw").iterdir()) if (method_dir / "results_raw").is_dir() else False

        if not has_raw:
            continue

        if not metrics_path.exists():
            has_clean = (method_dir / "results_cleaned").is_dir()
            lock = list(method_dir.glob(".*_failed"))
            lock_str = f" ({lock[0].name})" if lock else ""
            stage = "cleaned" if has_clean else "raw"
            issues.append(f"  {m}: no metrics.json (stage={stage}){lock_str}")
            continue

        # Check distances exist and sizes are compatible
        for dist_name in ["gt2data_dist.npy", "data2gt_dist.npy"]:
            dp = dist_dir / dist_name
            if not dp.exists():
                issues.append(f"  {m}: MISSING {dist_name}")

        # Check gt_points.npy matches gt_pcd (same sampling)
        gt_pts_path = dist_dir / "gt_points.npy"
        if gt_pts_path.exists():
            gt_pts_len = len(np.load(gt_pts_path))
            if gt_pts_len != pcd_len:
                issues.append(f"  {m}: gt_points({gt_pts_len}) != gt_pcd({pcd_len}) — STALE EVAL")

            gt2data_path = dist_dir / "gt2data_dist.npy"
            if gt2data_path.exists():
                gt2data_len = len(np.load(gt2data_path))
                if gt_pts_len != gt2data_len:
                    issues.append(f"  {m}: gt_points({gt_pts_len}) != gt2data_dist({gt2data_len})")

        # Check data_points.npy compatible with data2gt_dist
        data_pts_path = dist_dir / "data_points.npy"
        if data_pts_path.exists():
            data_pts_len = len(np.load(data_pts_path))
            data2gt_path = dist_dir / "data2gt_dist.npy"
            if data2gt_path.exists():
                data2gt_len = len(np.load(data2gt_path))
                if data_pts_len != data2gt_len:
                    issues.append(f"  {m}: data_points({data_pts_len}) != data2gt_dist({data2gt_len})")

    return issues


def main():
    total_issues = 0
    for obj_dir in sorted(EVAL_ROOT.iterdir()):
        if not obj_dir.is_dir() or not obj_dir.name[0].isdigit():
            continue

        issues = check_object(obj_dir)
        status = "OK" if not issues else f"{len(issues)} issue(s)"
        print(f"{'✓' if not issues else '✗'} {obj_dir.name:<22} {status}")
        for issue in issues:
            print(f"    {issue}")
        total_issues += len(issues)

    print(f"\nTotal: {total_issues} issues")
    return 1 if total_issues > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
