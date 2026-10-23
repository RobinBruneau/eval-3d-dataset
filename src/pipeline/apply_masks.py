"""Apply GT attribute masks to extract per-zone distances."""

from pathlib import Path

import numpy as np

from ..config import Config
from ..io.masks import load_visibility_count, load_curvature_values, load_challenge_mask, list_challenge_masks
from ..io.results import load_distances


def apply_gt_mask(
    gt_mask: np.ndarray,
    gt2data_dist: np.ndarray,
    data2gt_dist: np.ndarray,
    data2gt_idx: np.ndarray,
) -> dict:
    """Extract bidirectional distances for points matching a GT mask.

    For gt2data: select GT points where mask is True
    For data2gt: select data points whose nearest GT is in the masked set

    Args:
        gt_mask: Boolean mask on GT points, shape (N,).
        gt2data_dist: Distances from GT to data, shape (N,).
        data2gt_dist: Distances from data to GT, shape (M,).
        data2gt_idx: Indices of nearest GT for each data point, shape (M,).

    Returns:
        Dictionary with:
            - gt2data: Distances from masked GT points to data
            - data2gt: Distances from data points whose nearest GT is masked
    """
    gt2data = gt2data_dist[gt_mask]

    masked_gt_indices = np.where(gt_mask)[0]
    data_mask = np.isin(data2gt_idx, masked_gt_indices)
    data2gt = data2gt_dist[data_mask]

    return {"gt2data": gt2data, "data2gt": data2gt}


def apply_visibility_masks(
    config: Config,
    object_name: str,
    method_name: str,
    force: bool = False,
) -> None:
    """Apply visibility masks to create per-camera-count distance files.

    Creates: eval/zones/visibility/cam_{01..N}.npz

    Args:
        config: Pipeline configuration.
        object_name: Object name.
        method_name: Method name.
        force: Overwrite existing files.
    """
    gt_dir = config.get_gt_artifacts_dir(object_name)
    eval_dir = config.get_eval_dir(object_name, method_name)
    zones_dir = eval_dir / "zones" / "visibility"

    print(f"  Applying visibility masks: {object_name}/{method_name}")

    visibility = load_visibility_count(gt_dir)
    distances = load_distances(eval_dir)

    unique_counts = np.unique(visibility)
    print(f"  Visibility range: {visibility.min()}-{visibility.max()}")

    zones_dir.mkdir(parents=True, exist_ok=True)

    for count in unique_counts:
        out_path = zones_dir / f"cam_{count:02d}.npz"
        if out_path.exists() and not force:
            continue

        gt_mask = visibility == count
        zone_dist = apply_gt_mask(
            gt_mask,
            distances["gt2data_dist"],
            distances["data2gt_dist"],
            distances["data2gt_idx"],
        )

        np.savez(out_path, gt2data=zone_dist["gt2data"], data2gt=zone_dist["data2gt"])

    print(f"  Saved {len(unique_counts)} visibility zone files")


def apply_curvature_masks(
    config: Config,
    object_name: str,
    method_name: str,
    force: bool = False,
) -> None:
    """Store curvature values alongside distances for deferred binning.

    Creates:
        - eval/zones/curvature/curvature_values.npy (per GT point)
        - eval/zones/curvature/gt2data_dist.npy
        - eval/zones/curvature/data2gt_dist.npy
        - eval/zones/curvature/data2gt_curvature.npy (curvature of nearest GT)

    Args:
        config: Pipeline configuration.
        object_name: Object name.
        method_name: Method name.
        force: Overwrite existing files.
    """
    gt_dir = config.get_gt_artifacts_dir(object_name)
    eval_dir = config.get_eval_dir(object_name, method_name)
    zones_dir = eval_dir / "zones" / "curvature"

    print(f"  Applying curvature masks: {object_name}/{method_name}")

    curvature = load_curvature_values(gt_dir)
    distances = load_distances(eval_dir)

    zones_dir.mkdir(parents=True, exist_ok=True)

    np.save(zones_dir / "curvature_values.npy", curvature)
    np.save(zones_dir / "gt2data_dist.npy", distances["gt2data_dist"])
    np.save(zones_dir / "data2gt_dist.npy", distances["data2gt_dist"])

    data2gt_curvature = curvature[distances["data2gt_idx"]]
    np.save(zones_dir / "data2gt_curvature.npy", data2gt_curvature)

    print(f"  Saved curvature zone data")


def apply_challenge_masks(
    config: Config,
    object_name: str,
    method_name: str,
    force: bool = False,
) -> None:
    """Apply user-provided challenge masks.

    Creates: eval/zones/challenges/{name}.npz

    Args:
        config: Pipeline configuration.
        object_name: Object name.
        method_name: Method name.
        force: Overwrite existing files.
    """
    gt_dir = config.get_gt_artifacts_dir(object_name)
    eval_dir = config.get_eval_dir(object_name, method_name)
    zones_dir = eval_dir / "zones" / "challenges"

    print(f"  Applying challenge masks: {object_name}/{method_name}")

    challenge_names = list_challenge_masks(gt_dir)
    if not challenge_names:
        print(f"  No challenge masks found")
        return

    distances = load_distances(eval_dir)
    zones_dir.mkdir(parents=True, exist_ok=True)

    # `excluded.npy` marks unreliable GT zones: never evaluated, whatever the property.
    excluded_path = gt_dir / "challenges" / "excluded.npy"
    excluded = np.load(excluded_path).astype(bool) if excluded_path.exists() else None

    for name in challenge_names:
        out_path = zones_dir / f"{name}.npz"
        if out_path.exists() and not force:
            continue

        gt_mask = load_challenge_mask(gt_dir, name).astype(bool)
        if excluded is not None:
            if len(excluded) != len(gt_mask):
                raise ValueError(
                    f"{gt_dir}/challenges: excluded.npy has {len(excluded)} points, "
                    f"{name}.npy has {len(gt_mask)}. Refusing to guess."
                )
            gt_mask &= ~excluded
        zone_dist = apply_gt_mask(
            gt_mask,
            distances["gt2data_dist"],
            distances["data2gt_dist"],
            distances["data2gt_idx"],
        )

        np.savez(out_path, gt2data=zone_dist["gt2data"], data2gt=zone_dist["data2gt"])
        print(f"    Saved {name}.npz")

    print(f"  Saved {len(challenge_names)} challenge zone files")


def apply_all_masks(
    config: Config,
    object_name: str,
    method_name: str,
    visibility: bool = True,
    curvature: bool = True,
    challenges: bool = True,
    force: bool = False,
) -> None:
    """Apply all mask types.

    Args:
        config: Pipeline configuration.
        object_name: Object name.
        method_name: Method name.
        visibility: Apply visibility masks.
        curvature: Apply curvature masks.
        challenges: Apply challenge masks.
        force: Overwrite existing files.
    """
    gt_dir = config.get_gt_artifacts_dir(object_name)

    if visibility:
        vis_path = gt_dir / "attributes" / "visibility_count.npy"
        if vis_path.exists():
            apply_visibility_masks(config, object_name, method_name, force)
        else:
            print(f"  Skipping visibility (no visibility_count.npy)")

    if curvature:
        curv_path = gt_dir / "attributes" / "curvature_values.npy"
        if curv_path.exists():
            apply_curvature_masks(config, object_name, method_name, force)
        else:
            print(f"  Skipping curvature (no curvature_values.npy)")

    if challenges:
        apply_challenge_masks(config, object_name, method_name, force)
