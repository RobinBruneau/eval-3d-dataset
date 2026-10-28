"""The fast GT-attribute path (in-view-only chunked visibility) must give the same values as the
former all-points implementation. (The curvature is now the VCM, tested in test_vcm_curvature.py.)"""

import numpy as np
import trimesh
from PIL import Image

from src.core.visibility import compute_visibility_count, ray_visibility_check, check_points_in_mask
from src.core.camera import get_camera_centers


def _scene(tmp_path):
    """Sphere + an occluding plate in front of camera 0; three cameras around."""
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    plate = trimesh.creation.box(extents=(0.8, 0.8, 0.05)); plate.apply_translation((0.3, 0.2, 2.0))
    mesh = trimesh.util.concatenate([sphere, plate])
    K = np.array([[300, 0, 160, 0], [0, 300, 120, 0], [0, 0, 1, 0], [0, 0, 0, 1]], float)
    Ps, Rts, names = [], [], []
    for k, ang in enumerate((0.0, 2.0, 4.0)):
        C = np.array([5 * np.sin(ang), 0.3, 5 * np.cos(ang)])
        z = -C / np.linalg.norm(C); x = np.cross([0, 1, 0], z); x /= np.linalg.norm(x); y = np.cross(z, x)
        R = np.stack([x, y, z]); Rt = np.eye(4); Rt[:3, :3] = R; Rt[:3, 3] = -R @ C
        Ps.append(K @ Rt); Rts.append(Rt)
        mask = np.zeros((240, 320), np.uint8); mask[40:200, 60:260] = 255            # partial mask
        Image.fromarray(mask).save(tmp_path / f"{k:03d}.png"); names.append(f"{k:03d}.png")
    pts = trimesh.sample.sample_surface(sphere, 4000, seed=1)[0]
    return pts, mesh, {"P": np.array(Ps), "Rt": np.array(Rts), "mask_names": names}


def test_visibility_count_matches_all_points_reference(tmp_path):
    pts, mesh, cams = _scene(tmp_path)
    centres = get_camera_centers(cams["Rt"])
    ref = np.zeros(len(pts), np.int32)
    for k in range(3):
        mask = np.array(Image.open(tmp_path / cams["mask_names"][k])) > 127
        _, in_view = check_points_in_mask(pts, cams["P"][k], mask, 0)
        ref += (in_view & ray_visibility_check(pts, mesh, centres[k])).astype(np.int32)
    fast = compute_visibility_count(pts, mesh, cams, tmp_path, dilation_radius=0, show_progress=False,
                                    ray_chunk=333)
    assert np.array_equal(fast, ref)
    assert 0 < ref.sum() < 3 * len(pts)          # the scene does exercise occlusion and the masks
