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
    # An empty or multi-geometry file makes trimesh return a Scene / list / PointCloud
    # (e.g. a failed COLMAP run writes a PLY with `element vertex 0`). Normalise to a
    # (possibly empty) Trimesh so callers can rely on .vertices/.faces and guard on
    # len()==0 rather than crash on a bare list having no `.update_faces`.
    if isinstance(mesh, trimesh.Scene):
        geoms = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        mesh = trimesh.util.concatenate(geoms) if geoms else trimesh.Trimesh()
    elif isinstance(mesh, (list, tuple)):
        geoms = [g for g in mesh if isinstance(g, trimesh.Trimesh)]
        mesh = trimesh.util.concatenate(geoms) if geoms else trimesh.Trimesh()
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = trimesh.Trimesh()
    if len(mesh.faces) > 0:
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


# Vertex curvature: VCM estimator (src/core/vcm.py), importable from here as before.
from .vcm import compute_vertex_curvature  # noqa: E402,F401
