"""Voronoi Covariance Measure (VCM) curvature on the connected ball (Mérigot, Ovsjanikov, Guibas 2011).

1. Cell of each vertex v: points y within R of the vertex cloud whose nearest vertex is v,
   VCM_v = ∫ (y - v)(y - v)ᵀ dy, by stratified sampling of the tube of thickness R.
2. Sum of the VCM_v over the connected ball of radius r (triangles linked to the vertex without
   leaving the ball, so the other side of a thin wall is not mixed in).
3. |k_i| = (2/r) sqrt(λ_i / λ_n); the sign comes from the variation of the triangle normals in the ball.

Validated on the Martine GT meshes (mm) with r = 1, R = 0.5 and |k| > 0.75 as the high-curvature zone.
"""
import time

import numba as nb
import numpy as np
from scipy.spatial import cKDTree


# ---------- connected ball ----------

def _csr(rows, cols, n):
    order = np.argsort(rows, kind="stable")
    ptr = np.zeros(n + 1, np.int64)
    np.add.at(ptr, rows + 1, 1)
    return np.cumsum(ptr), cols[order].astype(np.int64)


def _face_adjacency(F, nv):
    """Triangles sharing an edge (non-manifold edges included)."""
    nf = len(F)
    e = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    key = e[:, 0] * np.int64(nv) + e[:, 1]
    del e
    order = np.argsort(key)
    key = key[order]
    fid = (order % nf).astype(np.int64)
    del order
    starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    del key
    sizes = np.diff(np.r_[starts, len(fid)])
    rows, cols = [], []
    for s in np.unique(sizes[sizes > 1]):
        g = fid[starts[sizes == s, None] + np.arange(s)]
        for a in range(s):
            for b in range(s):
                if a != b:
                    rows.append(g[:, a])
                    cols.append(g[:, b])
    if not rows:
        return np.zeros(nf + 1, np.int64), np.zeros(0, np.int64)
    return _csr(np.concatenate(rows), np.concatenate(cols), nf)


@nb.njit(cache=True)
def _bfs(i, P, C, r2, vf_ptr, vf_idx, ff_ptr, ff_idx, stamp, mark, queue):
    """Put in queue[:n] the triangles of the connected ball of vertex i and return n (-1 if queue is too small)."""
    tail = 0
    for k in range(vf_ptr[i], vf_ptr[i + 1]):
        f = vf_idx[k]
        if stamp[f] != mark:
            stamp[f] = mark
            if tail == len(queue):
                return -1
            queue[tail] = f
            tail += 1
    head = 0
    while head < tail:
        f = queue[head]
        head += 1
        for k in range(ff_ptr[f], ff_ptr[f + 1]):
            g = ff_idx[k]
            if stamp[g] != mark and stamp[g] != -mark:
                d2 = 0.0
                for a in range(3):
                    d2 += (C[g, a] - P[i, a]) ** 2
                if d2 > r2:
                    stamp[g] = -mark  # neighbour outside the ball
                    continue
                stamp[g] = mark
                if tail == len(queue):
                    return -1
                queue[tail] = g
                tail += 1
    return tail


@nb.njit(parallel=True, cache=True)
def _ball_sums(P, C, W, r, vf_ptr, vf_idx, ff_ptr, ff_idx, ids, cap, nchunks):
    """Sum of the rows of W over the connected ball of each vertex of ids (NaN if more than cap triangles)."""
    n, nf, m = len(ids), len(C), W.shape[1]
    r2 = r * r
    out = np.zeros((n, m))
    step = (n + nchunks - 1) // nchunks
    for c in nb.prange(nchunks):
        stamp = np.zeros(nf, np.int32)
        queue = np.empty(cap, np.int64)
        for ii in range(c * step, min(n, (c + 1) * step)):
            nq = _bfs(ids[ii], P, C, r2, vf_ptr, vf_idx, ff_ptr, ff_idx, stamp, np.int32(ii + 1), queue)
            if nq < 0:
                for j in range(m):
                    out[ii, j] = np.nan
            for q in range(max(nq, 0)):
                for j in range(m):
                    out[ii, j] += W[queue[q], j]
    return out


def _ball_sums_batched(V, F, W, C, r, finish, batch=2_000_000, verbose=False, t0=None):
    """Sum W over the connected ball of each vertex, in batches; finish(S, slice) turns a batch into a dict.
    Vertices whose ball exceeds cap triangles are recomputed with an 8x larger cap."""
    t0 = time.perf_counter() if t0 is None else t0
    vf_ptr, vf_idx = _csr(F.ravel(), np.repeat(np.arange(len(F)), 3), len(V))
    ff_ptr, ff_idx = _face_adjacency(F, len(V))
    n, m, parts = len(V), W.shape[1], []
    for i0 in range(0, n, batch):
        i1 = min(n, i0 + batch)
        ids, cap = np.arange(i0, i1), 16384
        S = np.full((len(ids), m), np.nan)
        todo = np.arange(len(ids))
        while len(todo):
            S[todo] = _ball_sums(V, C, W, float(r), vf_ptr, vf_idx, ff_ptr, ff_idx, ids[todo], cap,
                                 nb.get_num_threads() * 2)
            todo, cap = todo[np.isnan(S[todo, 0])], cap * 8
        parts.append(finish(S, slice(i0, i1)))
        if verbose:
            el = time.perf_counter() - t0
            print(f"  VCM curvature {i1:,}/{n:,} vertices, {el:.0f} s elapsed, ~{el / i1 * (n - i1):.0f} s left", flush=True)
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


# ---------- VCM ----------

def _vcm_cells(V, area_v, R, spv=20, seed=0, verbose=False):
    """VCM of the Voronoi cell ∩ ball B(v, R) of each vertex, (nv, 6): xx, yy, zz, xy, xz, yz.
    spv: mean number of samples per cell (sets the step δ)."""
    rng = np.random.default_rng(seed)
    delta = (2 * R * area_v.sum() / (spv * len(V))) ** (1 / 3)
    m = max(1, int(round(R / 2 / delta)))  # coarse cells of m³ samples, side c ≈ R/2
    c = m * delta
    lo = V.min(0) - 2 * R
    tree = cKDTree(V)
    occ = np.unique(((V - lo) / c).astype(np.int64), axis=0)
    k = int(np.ceil(R / c))
    # dilation by the cube [-k, k]³ = three 1D dilations of the keys
    dims = occ.max(0) + k + 1
    keys = np.ravel_multi_index(occ.T, dims)
    for st in (dims[1] * dims[2], dims[2], 1):
        keys = np.unique((keys[:, None] + np.arange(-k, k + 1) * st).ravel())
    cells = np.column_stack(np.unravel_index(keys, dims))
    del occ, keys
    d, _ = tree.query((cells + 0.5) * c + lo, distance_upper_bound=R + c * np.sqrt(3) / 2, workers=-1)
    cells = cells[np.isfinite(d)]
    if verbose:
        print(f"  VCM: δ = {delta:.3f}, {len(cells) * m**3:,} samples", flush=True)

    sub = np.stack(np.meshgrid(*[np.arange(m)] * 3, indexing="ij"), -1).reshape(-1, 3)
    out = np.zeros((len(V), 6))
    pairs = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))
    step = max(1, 4_000_000 // m**3)
    for s in range(0, len(cells), step):
        y = ((cells[s:s + step, None, :] * m + sub[None]).reshape(-1, 3)
             + rng.random((len(cells[s:s + step]) * m**3, 3))) * delta + lo
        dist, idx = tree.query(y, distance_upper_bound=R, workers=-1)
        ok = np.isfinite(dist)
        idx, dy = idx[ok], y[ok] - V[idx[ok]]
        for j, (a, b) in enumerate(pairs):
            out[:, j] += np.bincount(idx, dy[:, a] * dy[:, b], minlength=len(V))
    return out * delta**3


def _rank1(cell):
    """Keep the normal part λ_max ν νᵀ of each cell (drops the tangent spread due to meshing and noise)."""
    i, j = np.array([0, 1, 2, 0, 0, 1]), np.array([0, 1, 2, 1, 2, 2])
    M = np.zeros((len(cell), 3, 3))
    M[:, i, j] = cell
    M[:, j, i] = cell
    w, vec = np.linalg.eigh(M)
    nu = vec[:, :, 2]
    return w[:, 2:] * nu[:, i] * nu[:, j]


def _principal(S, V, r, o):
    area, Vc, An, AnC = S[:, 0], S[:, 1:7], S[:, 7:10], S[:, 10:19].reshape(-1, 3, 3)
    i, j = np.array([0, 1, 2, 0, 0, 1]), np.array([0, 1, 2, 1, 2, 2])
    M = np.zeros((len(S), 3, 3))
    M[:, i, j] = Vc
    M[:, j, i] = Vc
    w, vec = np.linalg.eigh(np.nan_to_num(M))  # ascending: w[:, 2] = normal
    N = vec[:, :, 2] * np.sign(np.einsum("ni,ni->n", vec[:, :, 2], An))[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        mag = 2 / r * np.sqrt(np.clip(w[:, :2] / w[:, 2:], 0, None))  # |k| along vec[:, :, 0] and 1
    G = AnC - An[:, :, None] * (V - o)[:, None, :]  # Σ A n (c - p)ᵀ
    sg = np.sign(np.einsum("nia,nij,nja->na", vec[:, :, :2], G, vec[:, :, :2]))
    k = mag * np.where(sg == 0, 1, sg)
    first = k[:, 1] >= k[:, 0]
    k1, k2 = np.where(first, k[:, 1], k[:, 0]), np.where(first, k[:, 0], k[:, 1])
    d1 = np.where(first[:, None], vec[:, :, 1], vec[:, :, 0])
    d2 = np.where(first[:, None], vec[:, :, 0], vec[:, :, 1])
    bad = ~(area > 0)
    k1[bad] = k2[bad] = np.nan
    return dict(k1=k1, k2=k2, d1=d1, d2=d2, H=(k1 + k2) / 2, K=k1 * k2, normal=N, area=area)


def vcm_principal_curvatures(V, F, r=1.0, R=0.5, spv=20, batch=2_000_000, verbose=False):
    """VCM principal curvatures (convex > 0): dict k1 >= k2, d1, d2 (directions), H, K, normal, area, one row per vertex."""
    t0 = time.perf_counter()
    V = np.ascontiguousarray(V, np.float64)
    F = np.ascontiguousarray(F, np.int64)
    nrm = 0.5 * np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])  # normal x area
    A = np.linalg.norm(nrm, axis=1)
    area_v = np.bincount(F.ravel(), np.repeat(A / 3, 3), minlength=len(V))
    cell = _rank1(_vcm_cells(V, area_v, R, spv, verbose=verbose))
    if verbose:
        print(f"  VCM cells {time.perf_counter() - t0:.0f} s", flush=True)
    # share of each triangle in the VCM of its vertices: (A_f / 3) / A_v
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.nan_to_num((A / 3)[:, None] / area_v[F])
    o = V.mean(0)  # centred coordinates for float32 precision
    C = V[F].mean(axis=1)
    W = np.column_stack([A, np.einsum("fk,fkj->fj", share, cell[F]), nrm,
                         (nrm[:, :, None] * (C - o)[:, None, :]).reshape(-1, 9)]).astype(np.float32)
    del cell, share
    return _ball_sums_batched(V, F, W, C.astype(np.float32), r, lambda S, sl: _principal(S, V[sl], r, o),
                           batch, verbose, t0)


def compute_vertex_curvature(mesh, radius: float = 1.0, tube: float = 0.5, verbose: bool = False) -> np.ndarray:
    """Signed principal curvature of largest magnitude at each vertex, by VCM.

    Sign convention of the former quadric fit, kept for the downstream thresholds: convex < 0, concave > 0.
    Isolated vertices (no triangle) get 0, as vertices without neighbours did.

    Args:
        mesh: Input mesh (vertex order is kept).
        radius: Radius r of the connected ball (mesh units).
        tube: Thickness R of the tube around the surface (mesh units).
        verbose: Print progress.

    Returns:
        Array of shape (N,), float32.
    """
    res = vcm_principal_curvatures(mesh.vertices, mesh.faces, radius, tube, verbose=verbose)
    k = np.where(np.abs(res["k1"]) >= np.abs(res["k2"]), res["k1"], res["k2"])
    return np.nan_to_num(-k).astype(np.float32)
