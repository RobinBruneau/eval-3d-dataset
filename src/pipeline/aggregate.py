"""Cross-dataset aggregation with dynamic binning."""

import json
from pathlib import Path
from typing import Optional

import numpy as np

from ..config import Config
from ..io.masks import list_challenge_masks
from ..core.metrics import compute_chamfer, compute_fscore_curve


def discover_methods(config: Config) -> list[str]:
    """List the evaluated methods on disk, for configs that leave `dataset.methods` empty.

    Mirrors the watcher's selection (src/watcher/scanner.py): every `results_raw/`
    under each object's eval root (outside `Groundtruth/`) names a method by its
    path relative to that root, `dataset.exclude_methods` substrings are skipped,
    and only combos with an `eval_results/metrics.json` count.

    Returns:
        Sorted, deduplicated method names across all objects.
    """
    found = set()
    for obj in config.dataset.objects:
        eval_root = config.get_eval_root(obj)
        if not eval_root.exists():
            continue
        for results_raw in eval_root.rglob("results_raw"):
            if not results_raw.is_dir() or "Groundtruth" in results_raw.parts:
                continue
            method = str(results_raw.parent.relative_to(eval_root))
            if any(ex in method for ex in config.dataset.exclude_methods):
                continue
            if (config.get_eval_dir(obj, method) / "metrics.json").exists():
                found.add(method)
    return sorted(found)


def _resolve_methods(config: Config, methods: list[str] | None) -> list[str]:
    """Explicit argument > `dataset.methods` > methods discovered on disk."""
    return methods or config.dataset.methods or discover_methods(config)


def _to_builtin(obj):
    """Recursively convert numpy scalars/arrays (and tuples) to JSON-native Python types."""
    if isinstance(obj, dict):
        return {str(k): _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _to_builtin(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _dump_json(obj, path: Path) -> None:
    # Serialize fully before touching the file: a failure must not leave a truncated JSON.
    text = json.dumps(_to_builtin(obj), indent=2)
    path.write_text(text)


def aggregate_global(
    config: Config,
    methods: list[str] | None = None,
) -> dict:
    """Aggregate global metrics across all objects.

    Concatenates all distances and computes combined metrics
    (not mean of means).

    Args:
        config: Pipeline configuration.
        methods: Methods to aggregate (default: all from config).

    Returns:
        Dictionary with per-method aggregated metrics.
    """
    methods = _resolve_methods(config, methods)
    results = {}

    for method in methods:
        all_gt2data = []
        all_data2gt = []

        for obj in config.dataset.objects:
            eval_dir = config.get_eval_dir(obj, method)
            dist_dir = eval_dir / "distances"

            gt2data_path = dist_dir / "gt2data_dist.npy"
            data2gt_path = dist_dir / "data2gt_dist.npy"

            if not gt2data_path.exists():
                continue

            all_gt2data.append(np.load(gt2data_path))
            all_data2gt.append(np.load(data2gt_path))

        if not all_gt2data:
            continue

        gt2data = np.concatenate(all_gt2data)
        data2gt = np.concatenate(all_data2gt)

        chamfer = compute_chamfer(data2gt, gt2data, config.evaluation.max_dist)
        curves = compute_fscore_curve(
            data2gt, gt2data,
            config.evaluation.fscore_thresholds,
        )

        results[method] = {
            **chamfer,
            "thresholds": curves["thresholds"].tolist(),
            "precision": curves["precision"].tolist(),
            "recall": curves["recall"].tolist(),
            "fscore": curves["fscore"].tolist(),
            "n_objects": len(all_gt2data),
            "n_gt_points": len(gt2data),
            "n_data_points": len(data2gt),
        }

    return results


def aggregate_visibility(
    config: Config,
    methods: list[str] | None = None,
    grouping: dict[str, list[int]] | None = None,
) -> dict:
    """Aggregate visibility zone metrics.

    Args:
        config: Pipeline configuration.
        methods: Methods to aggregate.
        grouping: Optional grouping, e.g., {"low": [1, 5], "high": [6, 84]}.
            Each value is [min, max] inclusive.

    Returns:
        Dictionary with per-method, per-zone metrics.
    """
    methods = _resolve_methods(config, methods)
    grouping = grouping or config.aggregation.visibility_groups
    results = {}

    for method in methods:
        if grouping:
            group_gt2data = {g: [] for g in grouping}
            group_data2gt = {g: [] for g in grouping}
        else:
            cam_gt2data = {}
            cam_data2gt = {}

        for obj in config.dataset.objects:
            eval_dir = config.get_eval_dir(obj, method)
            zones_dir = eval_dir / "zones" / "visibility"

            if not zones_dir.exists():
                continue

            for npz_path in zones_dir.glob("cam_*.npz"):
                cam_num = int(npz_path.stem.split("_")[1])
                data = np.load(npz_path)

                if grouping:
                    for group_name, (min_cam, max_cam) in grouping.items():
                        if min_cam <= cam_num <= max_cam:
                            group_gt2data[group_name].append(data["gt2data"])
                            group_data2gt[group_name].append(data["data2gt"])
                            break
                else:
                    if cam_num not in cam_gt2data:
                        cam_gt2data[cam_num] = []
                        cam_data2gt[cam_num] = []
                    cam_gt2data[cam_num].append(data["gt2data"])
                    cam_data2gt[cam_num].append(data["data2gt"])

        method_results = {}

        if grouping:
            for group_name in grouping:
                if not group_gt2data[group_name]:
                    continue
                gt2data = np.concatenate(group_gt2data[group_name])
                data2gt = np.concatenate(group_data2gt[group_name])
                chamfer = compute_chamfer(data2gt, gt2data, config.evaluation.max_dist)
                method_results[group_name] = {
                    **chamfer,
                    "n_gt_points": len(gt2data),
                    "n_data_points": len(data2gt),
                }
        else:
            for cam_num in sorted(cam_gt2data.keys()):
                gt2data = np.concatenate(cam_gt2data[cam_num])
                data2gt = np.concatenate(cam_data2gt[cam_num])
                chamfer = compute_chamfer(data2gt, gt2data, config.evaluation.max_dist)
                method_results[f"cam_{cam_num:02d}"] = {
                    **chamfer,
                    "n_gt_points": len(gt2data),
                    "n_data_points": len(data2gt),
                }

        results[method] = method_results

    return results


def aggregate_curvature(
    config: Config,
    methods: list[str] | None = None,
    thresholds: dict[str, float] | None = None,
    percentiles: tuple[float, float] = (33.3, 66.7),
) -> dict:
    """Aggregate curvature zone metrics with dynamic binning.

    Args:
        config: Pipeline configuration.
        methods: Methods to aggregate.
        thresholds: Optional explicit thresholds, e.g., {"concave": -0.1, "convex": 0.1}.
        percentiles: Percentiles for automatic threshold computation if no explicit thresholds.

    Returns:
        Dictionary with per-method, per-category metrics.
    """
    methods = _resolve_methods(config, methods)
    thresholds = thresholds or config.aggregation.curvature_thresholds
    results = {}

    all_curvatures = []
    for obj in config.dataset.objects:
        gt_dir = config.get_gt_artifacts_dir(obj)
        curv_path = gt_dir / "attributes" / "curvature_values.npy"
        if curv_path.exists():
            all_curvatures.append(np.load(curv_path))

    if not all_curvatures:
        return results

    all_curv = np.concatenate(all_curvatures)

    if thresholds is None:
        # GT curvature ships as float32 -> np.percentile returns np.float32,
        # which json.dump rejects: cast to builtin floats at the source.
        low_thresh = float(np.percentile(all_curv, percentiles[0]))
        high_thresh = float(np.percentile(all_curv, percentiles[1]))
        thresholds = {"concave": low_thresh, "convex": high_thresh}

    results["thresholds"] = thresholds

    for method in methods:
        category_gt2data = {"concave": [], "flat": [], "convex": []}
        category_data2gt = {"concave": [], "flat": [], "convex": []}

        for obj in config.dataset.objects:
            eval_dir = config.get_eval_dir(obj, method)
            zones_dir = eval_dir / "zones" / "curvature"

            if not zones_dir.exists():
                continue

            curv = np.load(zones_dir / "curvature_values.npy")
            gt2data = np.load(zones_dir / "gt2data_dist.npy")
            data2gt = np.load(zones_dir / "data2gt_dist.npy")
            data2gt_curv = np.load(zones_dir / "data2gt_curvature.npy")

            concave_gt = curv < thresholds["concave"]
            convex_gt = curv > thresholds["convex"]
            flat_gt = ~concave_gt & ~convex_gt

            category_gt2data["concave"].append(gt2data[concave_gt])
            category_gt2data["flat"].append(gt2data[flat_gt])
            category_gt2data["convex"].append(gt2data[convex_gt])

            concave_data = data2gt_curv < thresholds["concave"]
            convex_data = data2gt_curv > thresholds["convex"]
            flat_data = ~concave_data & ~convex_data

            category_data2gt["concave"].append(data2gt[concave_data])
            category_data2gt["flat"].append(data2gt[flat_data])
            category_data2gt["convex"].append(data2gt[convex_data])

        method_results = {}
        for cat in ["concave", "flat", "convex"]:
            if not category_gt2data[cat]:
                continue
            gt2data = np.concatenate(category_gt2data[cat])
            data2gt = np.concatenate(category_data2gt[cat])
            if len(gt2data) == 0 or len(data2gt) == 0:
                continue
            chamfer = compute_chamfer(data2gt, gt2data, config.evaluation.max_dist)
            method_results[cat] = {
                **chamfer,
                "n_gt_points": len(gt2data),
                "n_data_points": len(data2gt),
            }

        results[method] = method_results

    return results


def aggregate_challenges(
    config: Config,
    methods: list[str] | None = None,
) -> dict:
    """Aggregate challenge mask metrics.

    Args:
        config: Pipeline configuration.
        methods: Methods to aggregate.

    Returns:
        Dictionary with per-method, per-challenge metrics.
    """
    methods = _resolve_methods(config, methods)
    results = {}

    challenge_names = set()
    for obj in config.dataset.objects:
        challenge_names.update(list_challenge_masks(config.get_gt_artifacts_dir(obj)))

    if not challenge_names:
        return results

    for method in methods:
        challenge_gt2data = {c: [] for c in challenge_names}
        challenge_data2gt = {c: [] for c in challenge_names}

        for obj in config.dataset.objects:
            eval_dir = config.get_eval_dir(obj, method)
            zones_dir = eval_dir / "zones" / "challenges"

            if not zones_dir.exists():
                continue

            for npz_path in zones_dir.glob("*.npz"):
                name = npz_path.stem
                if name not in challenge_names:
                    continue
                data = np.load(npz_path)
                challenge_gt2data[name].append(data["gt2data"])
                challenge_data2gt[name].append(data["data2gt"])

        method_results = {}
        for name in challenge_names:
            if not challenge_gt2data[name]:
                continue
            gt2data = np.concatenate(challenge_gt2data[name])
            data2gt = np.concatenate(challenge_data2gt[name])
            if len(gt2data) == 0 or len(data2gt) == 0:
                continue
            chamfer = compute_chamfer(data2gt, gt2data, config.evaluation.max_dist)
            method_results[name] = {
                **chamfer,
                "n_gt_points": len(gt2data),
                "n_data_points": len(data2gt),
            }

        results[method] = method_results

    return results


def save_aggregated_results(
    config: Config,
    global_metrics: dict | None = None,
    visibility_metrics: dict | None = None,
    curvature_metrics: dict | None = None,
    challenge_metrics: dict | None = None,
) -> None:
    """Save all aggregated results to JSON files.

    Numpy scalars/arrays anywhere in the payloads are converted to builtin
    types (a single np.float32 used to abort the whole save mid-way).

    Args:
        config: Pipeline configuration.
        global_metrics: Global aggregated metrics.
        visibility_metrics: Visibility zone metrics.
        curvature_metrics: Curvature zone metrics.
        challenge_metrics: Challenge zone metrics.
    """
    output_dir = config.paths.output_root / "aggregated"
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, payload in (
        ("global", global_metrics),
        ("visibility", visibility_metrics),
        ("curvature", curvature_metrics),
        ("challenge", challenge_metrics),
    ):
        if payload:
            _dump_json(payload, output_dir / f"{name}_metrics.json")
