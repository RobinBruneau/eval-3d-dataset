"""Global evaluation: compute distances, Chamfer, and F-score metrics."""

import json
from pathlib import Path

import numpy as np

from ..config import Config
from ..core.mesh import load_mesh
from ..core.sampling import upsample_mesh, downsample_pcd, remove_nan_points
from ..core.metrics import compute_distances, compute_metrics
from ..core.masking import load_gt_exclude, apply_exclude
import cv2


def evaluate(
    config: Config,
    object_name: str,
    method_name: str,
    force: bool = False,
    seed: int = 42,
    extra_exclude: list[str] | None = None,
) -> dict:
    """Evaluate a reconstructed mesh against ground truth.

    Computes bidirectional distances and stores:
        - distances/gt2data_dist.npy, gt2data_idx.npy
        - distances/data2gt_dist.npy, data2gt_idx.npy
        - distances/gt_points.npy, data_points.npy
        - curves/thresholds.npy, precision.npy, recall.npy, fscore.npy
        - metrics.json

    Args:
        config: Pipeline configuration.
        object_name: Object name.
        method_name: Method name.
        force: Overwrite existing output.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with computed metrics.
    """
    gt_dir = config.get_gt_artifacts_dir(object_name)
    eval_dir = config.get_eval_dir(object_name, method_name)
    cleaned_mesh_path = config.get_cleaned_mesh_path(object_name, method_name)

    gt_pcd_path = gt_dir / "gt_pcd.npy"
    metrics_path = eval_dir / "metrics.json"

    if metrics_path.exists() and not force:
        print(f"  Metrics exist, loading: {metrics_path}")
        with open(metrics_path) as f:
            return json.load(f)

    print(f"Evaluating: {object_name}/{method_name}")

    if not gt_pcd_path.exists():
        raise FileNotFoundError(f"GT point cloud not found: {gt_pcd_path}")
    if not cleaned_mesh_path.exists():
        raise FileNotFoundError(f"Cleaned mesh not found: {cleaned_mesh_path}")

    gt_pcd = np.load(gt_pcd_path)
    print(f"  Loaded GT point cloud: {len(gt_pcd)} points")

    data_mesh = load_mesh(cleaned_mesh_path)
    print(f"  Loaded cleaned mesh: {len(data_mesh.vertices)} vertices")

    if len(data_mesh.vertices) == 0 or len(data_mesh.faces) == 0:
        # Empty reconstruction (failed method) -> failure cell. Skip NearestNeighbors,
        # which cannot fit 0 points; build degenerate distances directly so compute_metrics
        # yields NaN (its documented empty-set contract), keeping this out of means/rankings.
        print("  Empty cleaned mesh -> recording failure metrics (NaN)")
        dist_data2gt = np.empty(0, dtype=np.float32)
        dist_gt2data = np.full(len(gt_pcd), np.inf, dtype=np.float32)
        metrics = compute_metrics(
            dist_data2gt, dist_gt2data,
            config.evaluation.fscore_thresholds, config.evaluation.max_dist,
        )
        metrics["seed"] = seed
        metrics["n_gt_points"] = len(gt_pcd)
        metrics["n_data_points"] = 0
        metrics["exclude_masks"] = []
        metrics["max_dist"] = config.evaluation.max_dist
        metrics["empty_reconstruction"] = True
        eval_dir.mkdir(parents=True, exist_ok=True)
        curves_dir = eval_dir / "curves"
        curves_dir.mkdir(exist_ok=True)
        np.save(curves_dir / "thresholds.npy", np.array(metrics["thresholds"]))
        np.save(curves_dir / "precision.npy", np.array(metrics["precision"]))
        np.save(curves_dir / "recall.npy", np.array(metrics["recall"]))
        np.save(curves_dir / "fscore.npy", np.array(metrics["fscore"]))
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print("  Chamfer: nan (empty reconstruction, scored as failure)")
        return metrics

    density = config.evaluation.downsample_density

    print(f"  Upsampling data mesh (density={density})")
    data_pcd = upsample_mesh(data_mesh.vertices, data_mesh.faces, density)
    print(f"  Upsampled to {len(data_pcd)} points")

    # Numerical hygiene only: drop NaN points. No relative height cut is
    # applied any more, so reconstructed points below the GT floor (bases,
    # tables, ...) are kept and count in the distance/F-score metrics.
    data_pcd = remove_nan_points(data_pcd)
    print(f"  After NaN removal: {len(data_pcd)} points")

    print(f"  Downsampling data point cloud")
    data_down = downsample_pcd(data_pcd, density, shuffle=True, seed=seed)
    print(f"  Downsampled to {len(data_down)} points")

    print(f"  Computing distances")
    dist_data2gt, idx_data2gt = compute_distances(data_down, gt_pcd)
    dist_gt2data, idx_gt2data = compute_distances(gt_pcd, data_down)

    # Apply GT exclusion mask(s). Shared with recompute/curves via core.masking so the
    # three code paths can no longer diverge; `metrics["exclude_masks"]` below records
    # exactly which files were used.
    exclude, applied_masks = load_gt_exclude(
        gt_dir, n_gt=len(gt_pcd), extra_exclude=extra_exclude
    )
    dist_data2gt_filtered, dist_gt2data_filtered = apply_exclude(
        dist_data2gt, dist_gt2data, idx_data2gt, exclude
    )

    print(f"  Computing metrics")
    metrics = compute_metrics(
        dist_data2gt_filtered,
        dist_gt2data_filtered,
        config.evaluation.fscore_thresholds,
        config.evaluation.max_dist,
    )
    metrics["seed"] = seed
    metrics["n_gt_points"] = len(gt_pcd)
    metrics["n_data_points"] = len(data_down)
    metrics["exclude_masks"] = [Path(p).name for p in applied_masks]
    metrics["max_dist"] = config.evaluation.max_dist

    eval_dir.mkdir(parents=True, exist_ok=True)
    distances_dir = eval_dir / "distances"
    curves_dir = eval_dir / "curves"
    distances_dir.mkdir(exist_ok=True)
    curves_dir.mkdir(exist_ok=True)

    np.save(distances_dir / "gt2data_dist.npy", dist_gt2data)
    np.save(distances_dir / "gt2data_idx.npy", idx_gt2data)
    np.save(distances_dir / "data2gt_dist.npy", dist_data2gt)
    np.save(distances_dir / "data2gt_idx.npy", idx_data2gt)
    np.save(distances_dir / "gt_points.npy", gt_pcd)
    np.save(distances_dir / "data_points.npy", data_down)

    np.save(curves_dir / "thresholds.npy", np.array(metrics["thresholds"]))
    np.save(curves_dir / "precision.npy", np.array(metrics["precision"]))
    np.save(curves_dir / "recall.npy", np.array(metrics["recall"]))
    np.save(curves_dir / "fscore.npy", np.array(metrics["fscore"]))

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"  Chamfer: {metrics['chamfer']:.4f}")
    # Find closest threshold to 1.0 for display
    thresholds = metrics["thresholds"]
    if 1.0 in thresholds:
        fscore_1 = metrics["fscore"][thresholds.index(1.0)]
    else:
        # Find closest threshold
        closest_idx = min(range(len(thresholds)), key=lambda i: abs(thresholds[i] - 1.0))
        fscore_1 = metrics["fscore"][closest_idx]
    print(f"  F-score@1.0: {fscore_1:.4f}")
    print(f"  Saved results to {eval_dir}")

    return metrics
