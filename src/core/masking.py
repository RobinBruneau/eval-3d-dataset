"""GT-side exclusion masks, shared by evaluate / recompute / curves.

Single source of truth for "which GT points count, and which reconstructed points
inherit that decision". Before this module the mask logic lived inline in
`evaluate.py` only — `recompute.py` and `curves.py` silently skipped it, which made
the curves on disk inconsistent with the `metrics.json` sitting next to them for
every object carrying a `challenges/excluded.npy`.

Semantics: every mask array is boolean, indexed over the **GT point cloud**, and True
means *drop this point*. Masks combine with logical OR. The reconstruction side inherits
the decision through the nearest-neighbour index (`idx_data2gt`), exactly as
`evaluate.py` has always done.

`challenges/excluded.npy` (unreliable GT zones) is the only mask applied implicitly,
whenever the file exists. Any other mask must be requested explicitly (`extra_exclude`).
The applied set is reported per cell through `metrics["exclude_masks"]`.
"""

from pathlib import Path

import numpy as np


def resolve_mask_path(gt_dir: Path, name: str) -> Path:
    """Resolve a mask name to a path.

    A bare name resolves under `<gt_dir>/challenges/<name>.npy`; anything containing a
    path separator, or ending in .npy, is taken as given.
    """
    p = Path(name)
    if p.suffix == ".npy" or len(p.parts) > 1:
        return p if p.is_absolute() else (gt_dir / p)
    return gt_dir / "challenges" / f"{name}.npy"


def load_gt_exclude(
    gt_dir: Path,
    n_gt: int,
    use_excluded: bool = True,
    extra_exclude: list[str] | None = None,
    verbose: bool = True,
) -> tuple[np.ndarray | None, list[str]]:
    """Build the combined boolean *exclusion* mask over GT points.

    Args:
        gt_dir: Groundtruth directory for the object.
        n_gt: Number of GT points; every mask must match this length.
        use_excluded: Load `challenges/excluded.npy` when it exists (default on).
        extra_exclude: Additional mask names/paths, OR-ed in. True = drop.
        verbose: Print what was applied.

    Returns:
        (exclude, applied) where `exclude` is a bool array of length n_gt (True = drop)
        or None if nothing applies, and `applied` lists the sources used.
    """
    exclude: np.ndarray | None = None
    applied: list[str] = []

    sources: list[Path] = []
    if use_excluded:
        sources.append(gt_dir / "challenges" / "excluded.npy")
    for name in extra_exclude or []:
        sources.append(resolve_mask_path(gt_dir, name))

    # The same file requested twice is applied (and declared) once.
    seen: set[str] = set()

    for path in sources:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        if not path.exists():
            # An absent excluded.npy is normal; an absent *requested* mask is a typo or a
            # broken deployment — fail loudly.
            if path.name == "excluded.npy":
                continue
            raise FileNotFoundError(f"Exclusion mask not found: {path}")
        seen.add(key)
        m = np.load(path)
        if m.dtype != bool:
            m = m.astype(bool)
        if m.shape[0] != n_gt:
            raise ValueError(
                f"Mask {path} has length {m.shape[0]} but GT has {n_gt} points. "
                "Stale mask or stale GT — refusing to guess."
            )
        exclude = m.copy() if exclude is None else (exclude | m)
        applied.append(str(path))
        if verbose:
            print(f"  Exclusion mask {path.name}: {int(m.sum()):,} / {n_gt:,} GT points dropped")

    if verbose and exclude is not None and len(applied) > 1:
        print(f"  Combined exclusion: {int(exclude.sum()):,} / {n_gt:,} GT points dropped")

    return exclude, applied


def apply_exclude(
    dist_data2gt: np.ndarray,
    dist_gt2data: np.ndarray,
    idx_data2gt: np.ndarray,
    exclude: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a GT exclusion mask to both distance directions.

    The reconstruction side inherits exclusion through its nearest-GT index, which is
    what `evaluate.py` has always done. Reproduces that behaviour bit-for-bit so that
    recomputing from saved `.npy` equals a full re-evaluation.
    """
    if exclude is None:
        return dist_data2gt, dist_gt2data

    gt_keep = ~exclude
    data_keep = gt_keep[idx_data2gt]
    return dist_data2gt[data_keep], dist_gt2data[gt_keep]
