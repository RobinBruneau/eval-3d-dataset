"""Visibility computation for point clouds using ray tracing."""

from pathlib import Path

import cv2
import numpy as np
import trimesh
from PIL import Image
from tqdm import tqdm

from .camera import get_camera_centers, project_points


def load_mask(mask_path: str | Path) -> np.ndarray:
    """Load a binary mask image.

    Args:
        mask_path: Path to mask image (PNG).

    Returns:
        Binary mask array, shape (H, W).
    """
    img = Image.open(mask_path)
    mask = np.array(img)
    if mask.ndim == 3:
        mask = mask[..., 0]
    # Handle both uint8 (0/255) and boolean masks
    if mask.dtype == bool:
        return mask
    return mask > 127


def create_dilation_kernel(radius: int) -> np.ndarray:
    """Create a circular dilation kernel.

    Args:
        radius: Dilation radius in pixels.

    Returns:
        Binary kernel array.
    """
    diameter = 2 * radius + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))


def ray_visibility_check(
    points: np.ndarray,
    mesh: trimesh.Trimesh,
    camera_center: np.ndarray,
    ray_offset_factor: float = 0.01 / 100,
) -> np.ndarray:
    """Check visibility of points from a camera using ray tracing.

    Args:
        points: 3D points to check, shape (N, 3).
        mesh: Mesh for occlusion testing.
        camera_center: Camera center position, shape (3,).
        ray_offset_factor: Factor for ray origin offset to avoid self-intersection.

    Returns:
        Boolean array, True if point is visible (not occluded), shape (N,).
    """
    ray_directions = camera_center - points
    ray_norms = np.linalg.norm(ray_directions, axis=-1, keepdims=True)

    # Use per-point offset based on individual ray length (not global max)
    # This prevents numerical issues for points at varying distances
    ray_offsets = ray_offset_factor * ray_norms
    ray_directions_normalized = ray_directions / np.maximum(ray_norms, 1e-10)
    ray_origins = points + ray_offsets * ray_directions_normalized

    hit = mesh.ray.intersects_any(ray_origins, ray_directions_normalized)

    return ~hit


def check_points_in_bounds(
    points: np.ndarray,
    P: np.ndarray,
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Check if 3D points project inside image bounds.

    Args:
        points: 3D points, shape (N, 3).
        P: Projection matrix (4x4).
        image_shape: (height, width) of the image.

    Returns:
        Boolean array, True if point projects inside image bounds, shape (N,).
    """
    projected, depth_valid = project_points(points, P)

    h, w = image_shape
    x = projected[:, 0]
    y = projected[:, 1]

    in_bounds = depth_valid & (x >= 0) & (x < w) & (y >= 0) & (y < h) & ~np.isnan(x) & ~np.isnan(y)

    return in_bounds


def check_points_in_mask(
    points: np.ndarray,
    P: np.ndarray,
    mask: np.ndarray,
    dilation_radius: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Check if 3D points project inside a 2D mask.

    Args:
        points: 3D points, shape (N, 3).
        P: Projection matrix (4x4).
        mask: Binary mask, shape (H, W).
        dilation_radius: Optional mask dilation radius.

    Returns:
        Tuple of:
            - in_bounds: Boolean array, True if point is inside image bounds, shape (N,).
            - in_mask: Boolean array, True if point projects inside mask, shape (N,).
    """
    if dilation_radius is not None and dilation_radius > 0:
        kernel = create_dilation_kernel(dilation_radius)
        mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)

    projected, depth_valid = project_points(points, P)

    h, w = mask.shape
    x = projected[:, 0]
    y = projected[:, 1]

    # Check bounds (also handles NaN from invalid depth)
    in_bounds = depth_valid & (x >= 0) & (x < w) & (y >= 0) & (y < h) & ~np.isnan(x) & ~np.isnan(y)

    in_mask = np.zeros(len(points), dtype=bool)
    valid_idx = np.where(in_bounds)[0]

    if len(valid_idx) > 0:
        xi = x[valid_idx].astype(int)
        yi = y[valid_idx].astype(int)
        in_mask[valid_idx] = mask[yi, xi]

    return in_bounds, in_mask


def compute_mask_face_culling(
    vertices: np.ndarray,
    mesh: trimesh.Trimesh,
    cameras: dict,
    output_dir: str | Path | None = None,
) -> None:
    """Create binary masks per view by culling geometry from vertex orientation.

    The function first filters vertices whose normals do not point toward the camera,
    checks that those vertices are visible and inside the image, and finally rasterizes
    only the faces incident to the surviving vertices. This is typically faster than
    looping over every face for every view.
    """
    print("[visibility] compute_mask_face_culling: starting")
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    if len(faces) == 0:
        print("[visibility] compute_mask_face_culling: no faces to process")
        return None

    print(
        f"[visibility] compute_mask_face_culling: processing {len(faces)} faces "
        f"across {len(cameras['P'])} views"
    )
    image_shape = (int(cameras["image_height"]), int(cameras["image_width"]))
    h, w = image_shape

    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"[visibility] compute_mask_face_culling: saving masks to {output_path}")

    if hasattr(mesh, "vertex_normals") and mesh.vertex_normals is not None and len(mesh.vertex_normals) == len(vertices):
        vertex_normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
        vertex_normals = vertex_normals / np.maximum(np.linalg.norm(vertex_normals, axis=1, keepdims=True), 1e-10)
    else:
        vertex_normals = np.zeros_like(vertices)
        print("[visibility] compute_mask_face_culling: no vertex normals available, using zero normals")

    camera_centers = get_camera_centers(cameras["Rt"])
    projection_matrices = cameras["P"]
    print(f"[visibility] compute_mask_face_culling: image shape = {image_shape}")

    for view_idx in range(len(projection_matrices)):
        print(f"[visibility] compute_mask_face_culling: processing view {view_idx + 1}/{len(projection_matrices)}")
        mask = np.zeros((h, w), dtype=bool)
        P = projection_matrices[view_idx]
        camera_center = camera_centers[view_idx]

        vertex_dirs = camera_center - vertices
        vertex_dirs_norm = vertex_dirs / np.maximum(np.linalg.norm(vertex_dirs, axis=1, keepdims=True), 1e-10)
        normal_dot = np.einsum("ij,ij->i", vertex_normals, vertex_dirs_norm)
        candidate_vertices = np.where(normal_dot <= 0)[0]
        print(f"[visibility] compute_mask_face_culling: {len(candidate_vertices)} candidate vertices after normal culling")

        if len(candidate_vertices) == 0:
            if output_dir is not None:
                out_file = output_path / f"{view_idx:06d}.png"
                cv2.imwrite(str(out_file), mask.astype(np.uint8) * 255)
            continue

        projected, depth_valid = project_points(vertices[candidate_vertices], P)
        x = projected[:, 0]
        y = projected[:, 1]
        in_bounds = depth_valid & (x >= 0) & (x < w) & (y >= 0) & (y < h) & ~np.isnan(x) & ~np.isnan(y)
        candidate_vertices = candidate_vertices[in_bounds]
        print(f"[visibility] compute_mask_face_culling: {len(candidate_vertices)} vertices inside image bounds")

        if len(candidate_vertices) == 0:
            if output_dir is not None:
                out_file = output_path / f"{view_idx:06d}.png"
                cv2.imwrite(str(out_file), mask.astype(np.uint8) * 255)
            continue

        visible_vertices = ray_visibility_check(vertices[candidate_vertices], mesh, camera_center)
        candidate_vertices = candidate_vertices[visible_vertices]
        print(f"[visibility] compute_mask_face_culling: {len(candidate_vertices)} visible candidate vertices")

        if len(candidate_vertices) == 0:
            if output_dir is not None:
                out_file = output_path / f"{view_idx:06d}.png"
                cv2.imwrite(str(out_file), mask.astype(np.uint8) * 255)
            continue

        face_candidates: list[int] = []
        try:
            print("[visibility] compute_mask_face_culling: attempting to use mesh.vertex_faces for face lookup")
            vertex_faces = mesh.vertex_faces
            print("[visibility] compute_mask_face_culling: mesh.vertex_faces loaded successfully")
        except Exception:
            vertex_faces = None

        if vertex_faces is not None and hasattr(vertex_faces, "__len__") and len(vertex_faces) == len(vertices):
            print("[visibility] compute_mask_face_culling: using mesh.vertex_faces for face lookup")
            candidate_vertex_set = set(candidate_vertices.tolist())
            for vertex_idx in candidate_vertices:
                faces_for_vertex = np.asarray(vertex_faces[vertex_idx], dtype=np.int32).ravel()
                if faces_for_vertex.size > 0:
                    for face_idx in faces_for_vertex:
                        face = faces[face_idx]
                        if all(int(v_idx) in candidate_vertex_set for v_idx in face):
                            face_candidates.append(int(face_idx))
        else:
            print("[visibility] compute_mask_face_culling: using vectorized face membership fallback")
            face_candidates = np.flatnonzero(
                np.isin(faces[:, 0], candidate_vertices)
                & np.isin(faces[:, 1], candidate_vertices)
                & np.isin(faces[:, 2], candidate_vertices)
            ).astype(np.int32).tolist()

        if len(face_candidates) == 0:
            if output_dir is not None:
                out_file = output_path / f"{view_idx:06d}.png"
                cv2.imwrite(str(out_file), mask.astype(np.uint8) * 255)
            continue

        unique_faces = np.unique(np.asarray(face_candidates, dtype=np.int32))
        print(f"[visibility] compute_mask_face_culling: rasterizing {len(unique_faces)} candidate faces")

        for face_idx in unique_faces:
            face = faces[face_idx]
            projected, depth_valid = project_points(vertices[face], P)
            if not np.all(depth_valid):
                continue

            projected = np.round(projected).astype(np.int32)[:, :2]
            pts = projected.reshape(-1, 1, 2)
            if np.any(pts[:, 0, 0] < 0) or np.any(pts[:, 0, 0] >= w) or np.any(pts[:, 0, 1] < 0) or np.any(pts[:, 0, 1] >= h):
                pts = np.clip(pts, [0, 0], [w - 1, h - 1])

            cv2.fillPoly(mask, [pts], 1)

        if output_dir is not None:
            out_file = output_path / f"{view_idx:06d}.png"
            cv2.imwrite(str(out_file), mask.astype(np.uint8) * 255)
            print(f"[visibility] compute_mask_face_culling: wrote {out_file}")

    print("[visibility] compute_mask_face_culling: finished")



def compute_visibility_count(
    points: np.ndarray,
    mesh: trimesh.Trimesh,
    cameras: dict,
    masks_dir: str | Path,
    dilation_radius: int = 0,
    show_progress: bool = True,
    use_masks: bool = True,
) -> np.ndarray:
    """Compute per-point visibility count across all cameras.

    A point is visible from a camera if:
    1. It projects inside the (dilated) mask (if use_masks=True) or image bounds
    2. It is not occluded by the mesh

    Args:
        points: 3D points, shape (N, 3).
        mesh: Mesh for occlusion testing.
        cameras: Camera dict with 'P', 'Rt' arrays.
        masks_dir: Directory containing mask images (000.png, 001.png, ...).
        dilation_radius: Mask dilation radius in pixels.
        show_progress: Show progress bar.
        use_masks: Whether to use 2D masks for filtering.

    Returns:
        Visibility count per point, shape (N,), dtype int.
    """
    masks_dir = Path(masks_dir)
    n_views = len(cameras["P"])
    camera_centers = get_camera_centers(cameras["Rt"])

    visibility_count = np.zeros(len(points), dtype=np.int32)

    iterator = range(n_views)
    if show_progress:
        iterator = tqdm(iterator, desc="Computing visibility")

    # Use mask_names from cameras dict if available (SfMData format)
    mask_names = cameras.get("mask_names")

    for view_idx in iterator:
        mask_path = None
        if mask_names is not None:
            mask_path = masks_dir / mask_names[view_idx]
        else:
            # Fallback: try sequential naming conventions
            for pattern in [f"{view_idx:03d}.png", f"V{view_idx:02d}.png"]:
                p = masks_dir / pattern
                if p.exists():
                    mask_path = p
                    break
        
        if not mask_path or not mask_path.exists():
            continue

        mask = load_mask(mask_path)
        P = cameras["P"][view_idx]
        camera_center = camera_centers[view_idx]

        if use_masks:
            _, in_view = check_points_in_mask(points, P, mask, dilation_radius)
        else:
            in_view = check_points_in_bounds(points, P, mask.shape)

        visible_from_cam = ray_visibility_check(points, mesh, camera_center)

        visible = in_view & visible_from_cam
        visibility_count += visible.astype(np.int32)

    return visibility_count

def _resolve_mask_path_for_view(cameras: dict, masks_dir: str | Path, view_idx: int) -> Path | None:
    """Resolve the view mask path for a camera index, following the same naming rules as visibility."""
    masks_dir = Path(masks_dir)
    mask_names = cameras.get("mask_names")
    if mask_names is not None:
        if view_idx < len(mask_names):
            mask_path = masks_dir / mask_names[view_idx]
            if mask_path.exists():
                return mask_path
    for pattern in [f"{view_idx:03d}.png", f"V{view_idx:02d}.png"]:
        mask_path = masks_dir / pattern
        if mask_path.exists():
            return mask_path
    return None


def filter_mesh_by_issue_watertight(
    vertices: np.ndarray,
    mesh: trimesh.Trimesh,
    cameras: dict,
    masks_dir: str | Path,
    masks_issue_watertight_dir: str | Path,
    sample_count: int = 100,
    orientation_ratio_threshold: float = 0.7,
    visible_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Count how many watertight-issue masks flag each vertex.

    The function first estimates whether mesh vertex normals are already oriented toward
    the camera by sampling visible points on the mesh from the first camera view and
    checking the sign of the dot product between the sampled vertex normal and the
    viewing direction. If less than 70% of the sampled points face the camera, the
    normals are flipped.

    Then, for each issue mask found in ``masks_issue_watertight_dir``, the function
    projects all vertices into the associated camera view. A vertex contributes one count
    when it projects inside the mask and its (possibly flipped) normal points toward the
    camera.

    Returns:
        An integer array of shape (N,), where each value is the number of masks that
        flagged the corresponding vertex.
    """
    print("[visibility] filter_mesh_by_issue_watertight: starting")
    vertices = np.asarray(vertices, dtype=np.float32)
    if len(vertices) == 0:
        print("[visibility] filter_mesh_by_issue_watertight: no vertices to process")
        return np.zeros(0, dtype=np.int32)

    print(f"[visibility] filter_mesh_by_issue_watertight: processing {len(vertices)} vertices")
    if visible_mask is not None:
        visible_mask = np.asarray(visible_mask, dtype=bool)
        if visible_mask.shape[0] != len(vertices):
            raise ValueError(
                f"Expected visible_mask of length {len(vertices)}, got {visible_mask.shape[0]}"
            )
        active_vertices = visible_mask
    else:
        active_vertices = np.ones(len(vertices), dtype=bool)

    if not np.any(active_vertices):
        print("[visibility] filter_mesh_by_issue_watertight: no visible vertices to process")
        return np.zeros(len(vertices), dtype=np.int32)

    normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
    if normals.shape[0] != len(vertices):
        if len(normals) == 0:
            return np.zeros(len(vertices), dtype=np.int32)
        raise ValueError(
            f"Expected {len(vertices)} vertex normals, got {normals.shape[0]}"
        )

    normals = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-10)

    n_views = len(cameras.get("P", []))
    if n_views == 0:
        print("[visibility] filter_mesh_by_issue_watertight: no camera views available")
        return np.zeros(len(vertices), dtype=np.int32)

    print(f"[visibility] filter_mesh_by_issue_watertight: using {n_views} views")
    camera_centers = get_camera_centers(cameras["Rt"])
    camera_center0 = camera_centers[0]

    # Determine the orientation of the normals by sampling visible points from camera 0.
    print("[visibility] filter_mesh_by_issue_watertight: estimating normal orientation")
    oriented_normals = normals
    P0 = cameras["P"][0]

    sample_points = mesh.sample(max(sample_count * 5, sample_count))
    valid_samples = []
    for point in sample_points:
        if len(valid_samples) >= sample_count:
            break
        point = np.asarray(point, dtype=np.float32)
        
        visible = ray_visibility_check(point[None, :], mesh, camera_center0)[0]
        if visible:
            valid_samples.append(point)

    print(f"[visibility] filter_mesh_by_issue_watertight: sampled {len(valid_samples)} visible points")
    if valid_samples:
        sample_points = np.asarray(valid_samples, dtype=np.float32)
        sample_points = sample_points[:sample_count]
        sample_normals = normals[mesh.kdtree.query(sample_points)[1]]
        camera_dirs = sample_points - camera_center0
        camera_dirs_norm = camera_dirs / np.maximum(np.linalg.norm(camera_dirs, axis=1, keepdims=True), 1e-10)
        dot_products = np.einsum("ij,ij->i", sample_normals, camera_dirs_norm)
        toward_camera_ratio = np.mean(dot_products < 0)
        print(f"[visibility] filter_mesh_by_issue_watertight: toward-camera ratio={toward_camera_ratio:.3f}")
        if toward_camera_ratio <= orientation_ratio_threshold:
            print("[visibility] filter_mesh_by_issue_watertight: flipping normals")
            oriented_normals = -normals

    issue_mask_paths = sorted(Path(masks_issue_watertight_dir).glob("*.png"))
    if not issue_mask_paths:
        issue_mask_paths = sorted(Path(masks_issue_watertight_dir).glob("*.npy"))
    if not issue_mask_paths:
        print("[visibility] filter_mesh_by_issue_watertight: no issue masks found")
        return np.zeros(len(vertices), dtype=np.int32)

    print(f"[visibility] filter_mesh_by_issue_watertight: processing {len(issue_mask_paths)} issue masks")
    counts = np.zeros(len(vertices), dtype=np.int32)
    camera_centers = get_camera_centers(cameras["Rt"])
    for view_idx, mask_path in enumerate(issue_mask_paths):
        if view_idx >= n_views:
            break
        if not mask_path.exists():
            continue

        mask = load_mask(mask_path)
        P = cameras["P"][view_idx]
        _, in_mask = check_points_in_mask(vertices, P, mask)

        camera_center = camera_centers[view_idx]
        cam_dirs = camera_center - vertices
        cam_dirs_norm = cam_dirs / np.maximum(np.linalg.norm(cam_dirs, axis=1, keepdims=True), 1e-10)
        normal_dot = np.einsum("ij,ij->i", oriented_normals, cam_dirs_norm)
        facing_camera = normal_dot > 0

        candidate_mask = active_vertices & in_mask & facing_camera
        candidate_indices = np.flatnonzero(candidate_mask)
        if len(candidate_indices) > 0:
            visible_from_cam = ray_visibility_check(vertices[candidate_indices], mesh, camera_center)
            counts[candidate_indices[visible_from_cam]] += 1

    print(f"[visibility] filter_mesh_by_issue_watertight: finished, {int(counts.sum())} vertex-mask associations")
    return counts


def filter_by_visibility(
    vertices: np.ndarray,
    mesh: trimesh.Trimesh,
    cameras: dict,
    masks_dir: str | Path,
    dilation_radius: int = 12,
    show_progress: bool = True,
    use_masks: bool = True,
) -> np.ndarray:
    """Create a vertex mask keeping only vertices visible from at least one view.

    Uses strict mask filtering if use_masks=True: if a vertex projects inside a view's 
    image bounds but outside the mask, it is removed.
    Otherwise (use_masks=False), it only checks for occlusion and camera bounds.

    Args:
        vertices: Mesh vertices, shape (N, 3).
        mesh: Mesh for occlusion testing.
        cameras: Camera dict.
        masks_dir: Masks directory.
        dilation_radius: Mask dilation radius.
        show_progress: Show progress bar.
        use_masks: Whether to use 2D masks for filtering.

    Returns:
        Boolean mask, True for visible vertices, shape (N,).
    """
    masks_dir = Path(masks_dir)
    n_views = len(cameras["P"])
    camera_centers = get_camera_centers(cameras["Rt"])
    n_vertices = len(vertices)

    # Pre-load and dilate all masks
    masks = []
    image_shapes = []
    kernel = create_dilation_kernel(dilation_radius) if dilation_radius > 0 else None

    mask_names = cameras.get("mask_names")
    for view_idx in range(n_views):
        mask_path = None
        if mask_names is not None:
            mask_path = masks_dir / mask_names[view_idx]
        else:
            # Try multiple naming conventions
            for pattern in [f"{view_idx:03d}.png", f"V{view_idx:02d}.png"]:
                p = masks_dir / pattern
                if p.exists():
                    mask_path = p
                    break
        
        if mask_path and mask_path.exists():
            mask = load_mask(mask_path)
            image_shapes.append(mask.shape)
            if use_masks:
                if kernel is not None:
                    mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
                masks.append(mask)
            else:
                masks.append(None)
        else:
            masks.append(None)
            image_shapes.append(None)


    # Pass 1: Find vertices inside at least one image's bounds
    if show_progress:
        print("  Pass 1: Finding vertices inside image bounds...")
    outside_all_images = np.ones(n_vertices, dtype=bool)

    for view_idx in range(n_views):
        if image_shapes[view_idx] is None:
            continue
        P = cameras["P"][view_idx]
        in_bounds = check_points_in_bounds(vertices, P, image_shapes[view_idx])
        outside_all_images[in_bounds] = False

    inside_any_image = ~outside_all_images
    if show_progress:
        print(f"    Vertices inside at least one image: {inside_any_image.sum()}/{n_vertices}")

    # Pass 2: Strict mask filtering
    # For each view, if vertex is inside bounds but outside mask, remove it
    if show_progress:
        print(f"  Pass 2: Mask filtering (use_masks={use_masks})...")
    keep_mask = inside_any_image.copy()

    if use_masks:
        iterator = range(n_views)
        if show_progress:
            iterator = tqdm(iterator, desc="  Filtering by masks")

        for view_idx in iterator:
            if masks[view_idx] is None:
                continue

            P = cameras["P"][view_idx]
            in_bounds, in_mask = check_points_in_mask(vertices, P, masks[view_idx], dilation_radius=0)

            # Keep if: (outside this view's bounds) OR (inside this view's mask)
            keep_this_view = ~in_bounds | in_mask
            keep_mask &= keep_this_view
    else:
        if show_progress:
            print("    Skipping mask filtering as use_masks=False")

    if show_progress:
        print(f"    Vertices after mask filtering: {keep_mask.sum()}/{n_vertices}")

    # Pass 3: Ray tracing for occlusion
    # Keep vertices visible (not occluded) from at least one view
    if show_progress:
        print("  Pass 3: Ray tracing for occlusion...")

    # Pre-compute which vertices are in mask for each view (needed for ray tracing)
    in_view_all_views = np.zeros((n_views, n_vertices), dtype=bool)
    for view_idx in range(n_views):
        if image_shapes[view_idx] is None:
            continue
        P = cameras["P"][view_idx]
        if use_masks and masks[view_idx] is not None:
            _, in_view = check_points_in_mask(vertices, P, masks[view_idx], dilation_radius=0)
        else:
            in_view = check_points_in_bounds(vertices, P, image_shapes[view_idx])
        in_view_all_views[view_idx] = in_view

    # Compute ray visibility for all views
    hits_all_views = np.zeros((n_views, n_vertices), dtype=bool)

    iterator = range(n_views)
    if show_progress:
        iterator = tqdm(iterator, desc="  Ray tracing")

    for view_idx in iterator:
        if image_shapes[view_idx] is None:
            continue
        camera_center = camera_centers[view_idx]
        visible_from_cam = ray_visibility_check(vertices, mesh, camera_center)
        hits_all_views[view_idx] = ~visible_from_cam

    # A vertex is visible if: not hit (not occluded) AND in mask (or bounds) for that view
    visible_per_view = ~hits_all_views & in_view_all_views
    visible_any = visible_per_view.any(axis=0)

    # Final mask: passed mask filtering AND visible from at least one view
    final_mask = keep_mask & visible_any

    if show_progress:
        print(f"    Final visible vertices: {final_mask.sum()}/{n_vertices}")

    return final_mask
