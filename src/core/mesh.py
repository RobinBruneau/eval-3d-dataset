"""Mesh utilities using trimesh."""

from pathlib import Path

import numpy as np
import trimesh


def load_mesh(path: str | Path) -> trimesh.Trimesh:
    """Load a mesh from a file.

    Args:
        path: Path to the mesh file (PLY, OBJ, etc.).

    Returns:
        Loaded trimesh object.
    """
    mesh = trimesh.load(str(path), force="mesh")
    mesh.update_faces(mesh.nondegenerate_faces())
    return mesh


def save_mesh(mesh: trimesh.Trimesh, path: str | Path) -> None:
    """Save a mesh to a file.

    Args:
        mesh: Trimesh object to save.
        path: Output path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(path))


def get_vertices_faces(mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray]:
    """Get vertices and faces from a mesh.

    Args:
        mesh: Trimesh object.

    Returns:
        Tuple of (vertices, faces) as numpy arrays.
    """
    return mesh.vertices.astype(np.float32), mesh.faces.astype(np.int32)


def filter_mesh_by_vertex_mask(mesh: trimesh.Trimesh, vertex_mask: np.ndarray) -> trimesh.Trimesh:
    """Filter a mesh keeping only vertices where mask is True.

    Uses cumsum-based reindexing to update face indices after vertex removal.

    Args:
        mesh: Input mesh.
        vertex_mask: Boolean array of shape (N,) where N is number of vertices.
            True = keep vertex, False = remove vertex.

    Returns:
        New mesh with filtered vertices and updated face indices.
    """
    vertex_mask = np.asarray(vertex_mask, dtype=bool)

    # Keep faces where all vertices are in the mask
    valid_faces_mask = vertex_mask[mesh.faces].all(axis=1)
    valid_faces = mesh.faces[valid_faces_mask]

    # Compute index shift using cumsum
    # For each vertex, shift[i] = number of removed vertices before index i
    shift = np.cumsum(~vertex_mask)

    # Adjust face indices: subtract the shift for each vertex index
    adjusted_faces = valid_faces - shift[valid_faces]

    # Create new mesh with filtered vertices
    new_vertices = mesh.vertices[vertex_mask]
    new_mesh = trimesh.Trimesh(vertices=new_vertices, faces=adjusted_faces, process=False)

    # Clean up degenerate faces
    new_mesh.update_faces(new_mesh.nondegenerate_faces())

    return new_mesh

def filter_mesh_by_vertex_mask(mesh: trimesh.Trimesh, vertex_mask: np.ndarray) -> trimesh.Trimesh:
    """Filter a mesh keeping only vertices where mask is True.

    Uses cumsum-based reindexing to update face indices after vertex removal.

    Args:
        mesh: Input mesh.
        vertex_mask: Boolean array of shape (N,) where N is number of vertices.
            True = keep vertex, False = remove vertex.

    Returns:
        New mesh with filtered vertices and updated face indices.
    """
    vertex_mask = np.asarray(vertex_mask, dtype=bool)

    # Keep faces where all vertices are in the mask
    valid_faces_mask = vertex_mask[mesh.faces].all(axis=1)
    valid_faces = mesh.faces[valid_faces_mask]

    # Compute index shift using cumsum
    # For each vertex, shift[i] = number of removed vertices before index i
    shift = np.cumsum(~vertex_mask)

    # Adjust face indices: subtract the shift for each vertex index
    adjusted_faces = valid_faces - shift[valid_faces]

    # Create new mesh with filtered vertices
    new_vertices = mesh.vertices[vertex_mask]
    new_mesh = trimesh.Trimesh(vertices=new_vertices, faces=adjusted_faces, process=False)

    # Clean up degenerate faces
    new_mesh.update_faces(new_mesh.nondegenerate_faces())

    return new_mesh


def compute_vertex_curvature(mesh: trimesh.Trimesh, radius: float | None = None) -> np.ndarray:
    """Compute maximum principal curvature at each vertex.

    Uses a neighborhood-based estimation to highlight concave/convex regions.
    If radius is None, it defaults to 5x the average edge length.

    Args:
        mesh: Input mesh.
        radius: Spatial radius for neighborhood search.

    Returns:
        Array of signed maximum principal curvature values, shape (N,).
    """
    from scipy.spatial import cKDTree
    
    n_vertices = len(mesh.vertices)
    if n_vertices == 0:
        return np.array([], dtype=np.float32)

    # Estimate default radius if not provided (5x avg edge length)
    if radius is None:
        edges = mesh.edges_unique
        edge_lengths = np.linalg.norm(mesh.vertices[edges[:, 0]] - mesh.vertices[edges[:, 1]], axis=1)
        avg_edge = np.mean(edge_lengths)
        radius = avg_edge * 5.0
        print(f"  Estimating curvature with radius={radius:.3f} mm (5x avg edge)")

    tree = cKDTree(mesh.vertices)
    curvature = np.zeros(n_vertices, dtype=np.float32)
    normals = mesh.vertex_normals

    batch_size = 500_000
    n_batches = (n_vertices + batch_size - 1) // batch_size

    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, n_vertices)
        if n_batches > 1:
            print(f"  Batch {batch_idx+1}/{n_batches} (vertices {start:,}-{end:,})")

        neighbors_list = tree.query_ball_point(mesh.vertices[start:end], r=radius, workers=-1)

        for local_i, idx in enumerate(neighbors_list):
            if len(idx) < 6:
                continue

            i = start + local_i
            pts = mesh.vertices[idx]
            center = mesh.vertices[i]
            pts_centered = pts - center

            normal = normals[i]
            if abs(normal[0]) < 0.9:
                tangent1 = np.cross(normal, [1, 0, 0])
            else:
                tangent1 = np.cross(normal, [0, 1, 0])
            tangent1 /= np.linalg.norm(tangent1)
            tangent2 = np.cross(normal, tangent1)

            u = pts_centered @ tangent1
            v = pts_centered @ tangent2
            w = pts_centered @ normal

            A = np.column_stack([u**2, u*v, v**2])
            try:
                params, _, _, _ = np.linalg.lstsq(A, w, rcond=None)
                a, b, c = params
                tr = 2*a + 2*c
                det = 4*a*c - b**2
                disc = max(0, tr**2 - 4*det)
                k1 = (tr + np.sqrt(disc)) / 2
                k2 = (tr - np.sqrt(disc)) / 2
                curvature[i] = k1 if abs(k1) > abs(k2) else k2
            except np.linalg.LinAlgError:
                continue

    return curvature
