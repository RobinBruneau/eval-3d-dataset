"""Mesh sampling utilities for upsampling and downsampling."""

import multiprocessing as mp
from typing import Optional

import numpy as np
import sklearn.neighbors as skln


def sample_single_tri(args: tuple) -> np.ndarray:
    """Sample points inside a triangle using barycentric coordinates.

    Args:
        args: Tuple of (n1, n2, v1, v2, tri_vert) where:
            - n1: Number of samples along edge 1
            - n2: Number of samples along edge 2
            - v1: Edge vector 1, shape (1, 3)
            - v2: Edge vector 2, shape (1, 3)
            - tri_vert: Triangle vertex (origin), shape (1, 3)

    Returns:
        Sampled points inside the triangle, shape (M, 3).
    """
    n1, n2, v1, v2, tri_vert = args

    # Create grid of barycentric coordinates
    c = np.mgrid[: n1 + 1, : n2 + 1].astype(np.float32)
    c += 0.5
    c[0] /= max(n1, 1e-7)
    c[1] /= max(n2, 1e-7)
    c = np.transpose(c, (1, 2, 0))

    # Keep only points inside the triangle (barycentric sum < 1)
    k = c[c.sum(axis=-1) < 1]

    # Convert barycentric to Cartesian
    q = v1 * k[:, :1] + v2 * k[:, 1:] + tri_vert

    return q


def upsample_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    density: float,
    n_jobs: int = -1,
) -> np.ndarray:
    """Upsample a mesh to a point cloud with given density.

    Uses multiprocessing to sample points inside triangles based on
    their area and the target density.

    Args:
        vertices: Mesh vertices, shape (V, 3).
        faces: Mesh faces, shape (F, 3).
        density: Target point spacing.
        n_jobs: Number of parallel workers (-1 = all CPUs).

    Returns:
        Upsampled point cloud including original vertices, shape (N, 3).
    """
    # Get triangle vertices
    tri_vert = vertices[faces]  # (F, 3, 3)

    # Compute edge vectors
    v1 = tri_vert[:, 1] - tri_vert[:, 0]  # (F, 3)
    v2 = tri_vert[:, 2] - tri_vert[:, 0]  # (F, 3)

    # Compute edge lengths
    l1 = np.linalg.norm(v1, axis=-1, keepdims=True)  # (F, 1)
    l2 = np.linalg.norm(v2, axis=-1, keepdims=True)  # (F, 1)

    # Compute triangle area (2x area = cross product magnitude)
    area2 = np.linalg.norm(np.cross(v1, v2), axis=-1, keepdims=True)  # (F, 1)

    # Filter out degenerate triangles
    non_zero_area = (area2 > 0)[:, 0]
    l1 = l1[non_zero_area]
    l2 = l2[non_zero_area]
    area2 = area2[non_zero_area]
    v1 = v1[non_zero_area]
    v2 = v2[non_zero_area]
    tri_vert = tri_vert[non_zero_area]

    # Compute sampling density per triangle
    print(f"[upsample_mesh] begin triangle grid estimation: faces={len(tri_vert)}, density={density}")
    thr = density * np.sqrt(l1 * l2 / area2)
    n1 = np.floor(l1 / thr).astype(int)
    n2 = np.floor(l2 / thr).astype(int)
    print(f"[upsample_mesh] max n1={int(n1.max()) if len(n1) else 0}, max n2={int(n2.max()) if len(n2) else 0}")

    # Prepare arguments for parallel processing
    args = (
        (n1[i, 0], n2[i, 0], v1[i : i + 1], v2[i : i + 1], tri_vert[i : i + 1, 0])
        for i in range(len(n1))
    )

    # Handle edge case: no valid triangles
    if len(n1) == 0:
        print("[upsample_mesh] no valid triangles after degenerate filtering")
        return vertices.astype(np.float32)

    # Sample points in parallel using imap for memory efficiency
    if n_jobs == -1:
        n_jobs = mp.cpu_count()
    print(f"[upsample_mesh] launching pool with n_jobs={n_jobs}")

    # Collect results incrementally to reduce memory peak
    all_pts = [vertices]
    batch_size = 10000

    with mp.Pool(n_jobs) as pool:
        # Use imap for lazy evaluation
        args_list = [
            (n1[i, 0], n2[i, 0], v1[i : i + 1], v2[i : i + 1], tri_vert[i : i + 1, 0])
            for i in range(len(n1))
        ]
        print(f"[upsample_mesh] built {len(args_list)} tasks for multiprocessing")

        for i, pts in enumerate(pool.imap(sample_single_tri, args_list, chunksize=1024)):
            if len(pts) > 0:
                all_pts.append(pts)

            # Periodically concatenate to free memory
            if len(all_pts) > batch_size:
                print(f"[upsample_mesh] compacting accumulated batch at iteration {i}")
                all_pts = [np.concatenate(all_pts, axis=0)]

    # Final concatenation
    if len(all_pts) == 0:
        print("[upsample_mesh] no points collected after multiprocessing")
        return vertices.astype(np.float32)

    print(f"[upsample_mesh] final concatenation: total points={sum(len(x) for x in all_pts)}")
    pcd = np.concatenate(all_pts, axis=0).astype(np.float32)
    return pcd


def downsample_pcd(pcd: np.ndarray, density: float, shuffle: bool = True, seed: Optional[int] = None) -> np.ndarray:
    """Downsample a point cloud to a given density using radius-based filtering.

    For each point, removes all neighbors within the radius, keeping the
    current point. This creates a roughly uniform sampling.

    Args:
        pcd: Input point cloud, shape (N, 3).
        density: Target point spacing (radius for neighbor search).
        shuffle: Whether to shuffle points before downsampling.
        seed: Random seed for reproducibility.

    Returns:
        Downsampled point cloud, shape (M, 3) where M <= N.
    """
    pcd = pcd.copy()

    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(pcd, axis=0)

    # Build KD-tree
    nn_engine = skln.NearestNeighbors(
        n_neighbors=1, radius=density, algorithm="kd_tree", n_jobs=-1
    )
    nn_engine.fit(pcd)

    # Find neighbors within radius for each point
    rnn_idxs = nn_engine.radius_neighbors(pcd, radius=density, return_distance=False)

    # Greedy selection: keep point, remove its neighbors
    mask = np.ones(pcd.shape[0], dtype=bool)
    for curr, idxs in enumerate(rnn_idxs):
        if mask[curr]:
            mask[idxs] = False
            mask[curr] = True

    return pcd[mask].astype(np.float32)
