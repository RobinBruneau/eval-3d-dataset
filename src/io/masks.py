"""I/O utilities for GT masks and attributes."""

from pathlib import Path

import numpy as np


def load_gt_pcd(gt_dir: str | Path) -> np.ndarray:
    """Load ground truth point cloud.

    Args:
        gt_dir: Path to Groundtruth directory.

    Returns:
        GT point cloud, shape (N, 3).
    """
    return np.load(Path(gt_dir) / "gt_pcd.npy")


def load_visibility_count(gt_dir: str | Path) -> np.ndarray:
    """Load visibility count attribute.

    Args:
        gt_dir: Path to Groundtruth directory.

    Returns:
        Visibility count per point, shape (N,), dtype int.
    """
    return np.load(Path(gt_dir) / "attributes" / "visibility_count.npy")


def load_curvature_values(gt_dir: str | Path) -> np.ndarray:
    """Load curvature values attribute.

    Args:
        gt_dir: Path to Groundtruth directory.

    Returns:
        Curvature values per point, shape (N,), dtype float.
    """
    return np.load(Path(gt_dir) / "attributes" / "curvature_values.npy")


def load_challenge_mask(gt_dir: str | Path, name: str) -> np.ndarray:
    """Load a challenge mask by name.

    Args:
        gt_dir: Path to Groundtruth directory.
        name: Challenge name (without .npy extension).

    Returns:
        Boolean mask, shape (N,).
    """
    return np.load(Path(gt_dir) / "challenges" / f"{name}.npy")


#: GT exclusion masks stored next to the properties in `challenges/`. They drop GT points
#: from every evaluation (see `core/masking.py`) and are never scored as a property.
EXCLUSION_MASKS: tuple[str, ...] = ("excluded",)


def list_challenge_masks(gt_dir: str | Path) -> list[str]:
    """List the properties (challenges) of an object.

    A property counts for an object iff its `challenges/<property>.npy` exists; a property
    covering the whole object ships as an all-ones mask. Exclusion masks are not properties.

    Args:
        gt_dir: Path to Groundtruth directory.

    Returns:
        Sorted challenge names (without .npy extension).
    """
    challenges_dir = Path(gt_dir) / "challenges"
    if not challenges_dir.exists():
        return []
    return sorted(p.stem for p in challenges_dir.glob("*.npy") if p.stem not in EXCLUSION_MASKS)


def save_visibility_count(gt_dir: str | Path, visibility: np.ndarray) -> None:
    """Save visibility count attribute.

    Args:
        gt_dir: Path to Groundtruth directory.
        visibility: Visibility count per point, shape (N,).
    """
    attr_dir = Path(gt_dir) / "attributes"
    attr_dir.mkdir(parents=True, exist_ok=True)
    np.save(attr_dir / "visibility_count.npy", visibility)


def save_curvature_values(gt_dir: str | Path, curvature: np.ndarray) -> None:
    """Save curvature values attribute.

    Args:
        gt_dir: Path to Groundtruth directory.
        curvature: Curvature values per point, shape (N,).
    """
    attr_dir = Path(gt_dir) / "attributes"
    attr_dir.mkdir(parents=True, exist_ok=True)
    np.save(attr_dir / "curvature_values.npy", curvature)


def save_challenge_mask(gt_dir: str | Path, name: str, mask: np.ndarray) -> None:
    """Save a challenge mask.

    Args:
        gt_dir: Path to Groundtruth directory.
        name: Challenge name (without .npy extension).
        mask: Boolean mask, shape (N,).
    """
    challenges_dir = Path(gt_dir) / "challenges"
    challenges_dir.mkdir(parents=True, exist_ok=True)
    np.save(challenges_dir / f"{name}.npy", mask.astype(bool))
