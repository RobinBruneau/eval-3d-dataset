"""Global evaluation: compute distances, Chamfer, and F-score metrics."""

import json
from pathlib import Path

import numpy as np

from ..config import Config
from ..core.mesh import load_mesh
from ..core.sampling import upsample_mesh, downsample_pcd
from ..core.metrics import compute_distances, compute_metrics
from ..core.masking import load_gt_exclude, apply_exclude, load_gt_exclude_voxels, apply_exclude_voxels
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
    gt_dir = config.get_gt_dir(object_name)
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

    # randomly take 10% of the GT points for evaluation to speed up the process
    if len(gt_pcd) > 100000:
        np.random.seed(seed)
        gt_pcd = gt_pcd[np.random.choice(len(gt_pcd), size=int(len(gt_pcd) * 0.1), replace=False)]
        print(f"  Downsampled GT point cloud to {len(gt_pcd)} points for evaluation")

    data_mesh = load_mesh(cleaned_mesh_path)
    print(f"  Loaded cleaned mesh: {len(data_mesh.vertices)} vertices")

    density = config.evaluation.downsample_density

    print(f"  Upsampling data mesh (density={density})")
    data_pcd = upsample_mesh(data_mesh.vertices, data_mesh.faces, density)
    print(f"  Upsampled to {len(data_pcd)} points")

    gt_min_z = gt_pcd[:, 2].min()
    z_mask = data_pcd[:, 2] > gt_min_z
    data_pcd = data_pcd[z_mask]
    print(f"  After z-filter: {len(data_pcd)} points")

    nan_mask = ~np.isnan(data_pcd).any(axis=1)
    data_pcd = data_pcd[nan_mask]

    print(f"  Downsampling data point cloud")
    data_down = downsample_pcd(data_pcd, density, shuffle=True, seed=seed)
    print(f"  Downsampled to {len(data_down)} points")

    print(f"  Computing distances")
    dist_data2gt, idx_data2gt = compute_distances(data_down, gt_pcd)
    dist_gt2data, idx_gt2data = compute_distances(gt_pcd, data_down)
    print(f"  Computed distances: data2gt {len(dist_data2gt)} points, gt2data {len(dist_gt2data)} points")

    # Apply GT exclusion mask(s). Shared with recompute/curves via core.masking so the
    # three code paths can no longer diverge.
    #
    # `challenges/invisible.npy` (visibility_count == 0) is in `core.masking.AUTO_EXCLUDE`
    # and is therefore applied here **unconditionally when the file exists**, with no flag
    # to pass and no follow-up pass to run. Every cell this function writes carries the
    # mask by construction; `metrics["exclude_masks"]` below records exactly which files
    # were used, so the claim is checkable per cell rather than assumed corpus-wide.

    #exclude, applied_masks = load_gt_exclude(
    #    gt_dir, n_gt=len(gt_pcd), extra_exclude=extra_exclude
    #)

    #dist_data2gt_filtered, dist_gt2data_filtered = apply_exclude_voxels(
    #       dist_data2gt, dist_gt2data, idx_data2gt, exclude
    #    )

    print(f"  Loading voxels exclusion masks")
    exclude_gt, exclude_data, applied_masks = load_gt_exclude_voxels(
            gt_dir, data_down, gt_pcd, extra_exclude=extra_exclude
    )

    dist_data2gt_filtered, dist_gt2data_filtered = apply_exclude_voxels(
        dist_data2gt, dist_gt2data, exclude_gt, exclude_data
    )

    debug = True
    if debug:
        debug_dir = eval_dir / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)

        def write_point_cloud_obj(path: Path, points: np.ndarray) -> None:
            points = np.asarray(points, dtype=float)
            if points.ndim == 1:
                points = points.reshape(1, -1)
            with open(path, "w") as f:
                for point in points:
                    if point.shape[0] != 3:
                        raise ValueError(f"Expected 3D point, got shape {point.shape}")
                    f.write(f"v {point[0]} {point[1]} {point[2]}\n")

        write_point_cloud_obj(debug_dir / "data_points_before.obj", data_down)
        write_point_cloud_obj(
            debug_dir / "data_points_after.obj",
            data_down[~exclude_data] if exclude_data is not None else data_down,
        )
        write_point_cloud_obj(debug_dir / "gt_points_before.obj", gt_pcd)
        write_point_cloud_obj(
            debug_dir / "gt_points_after.obj",
            gt_pcd[~exclude_gt] if exclude_gt is not None else gt_pcd,
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
