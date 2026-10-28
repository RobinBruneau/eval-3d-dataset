"""VCM curvature: value and sign on spheres (convex < 0, the convention of the former quadric fit),
GT-point transfer, defaults."""

import numpy as np
import pytest
import trimesh

from src.core.mesh import compute_vertex_curvature
from src.pipeline.preprocess_gt import compute_gt_curvature

SPHERE_RADIUS = 10.0


def _sphere(subdivisions=7):
    # subdivisions 7: 163 842 vertices, edges ~0.09. The estimate is biased upwards when the edge is not
    # small next to the ball radius (r = 2: +8 % at edge 0.19, +3 % at 0.09), hence the fine mesh.
    return trimesh.creation.icosphere(subdivisions=subdivisions, radius=SPHERE_RADIUS)


def test_sphere_is_convex_with_magnitude_one_over_radius():
    k = compute_vertex_curvature(_sphere(), radius=2.0, tube=0.5)
    assert k.dtype == np.float32 and len(k) == 163842
    assert np.median(k) == pytest.approx(-1 / SPHERE_RADIUS, rel=0.05)
    assert (k < 0).mean() > 0.99


def test_inverted_sphere_is_concave():
    m = _sphere()
    m.faces = m.faces[:, ::-1]  # normals pointing inwards: the sphere is now a cavity
    k = compute_vertex_curvature(m, radius=2.0, tube=0.5)
    assert np.median(k) == pytest.approx(1 / SPHERE_RADIUS, rel=0.05)


def test_gt_curvature_transfers_vertex_values_to_points():
    m = _sphere(5)
    pts = m.vertices[::7] * 1.001  # just off the surface: nearest vertex is the source vertex
    k = compute_gt_curvature(m, pts, radius=1.0, tube=0.5)
    assert k.dtype == np.float32
    assert np.array_equal(k, compute_vertex_curvature(m, radius=1.0, tube=0.5)[::7])


def test_defaults_are_r1_R05_and_none_radius_means_default():
    m = _sphere(4)
    ref = compute_vertex_curvature(m, radius=1.0, tube=0.5)
    assert np.array_equal(compute_vertex_curvature(m), ref)
    assert np.array_equal(compute_gt_curvature(m, m.vertices, radius=None), ref)
