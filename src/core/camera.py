"""Camera loading utilities.

Loads cameras directly from npz files without decomposition to avoid
numerical uncertainty from cv2.decomposeProjectionMatrix.
"""

from pathlib import Path

import cv2
import numpy as np


def load_K_Rt_from_P(P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Decompose projection matrix P into intrinsics K and pose [R|t].
    
    Uses cv2.decomposeProjectionMatrix as fallback when inverse matrices
    are not available in cameras.npz.
    
    Args:
        P: Projection matrix (3x4 or 4x4).
        
    Returns:
        Tuple of:
            - K: Intrinsics (4x4)
            - pose: Camera-to-world pose (4x4), where pose[:3, :3] = R^T and pose[:3, 3] = -R^T @ t
    """
    # Use only 3x4 part for decomposition
    P_3x4 = P[:3, :4] if P.shape[0] == 4 else P
    
    out = cv2.decomposeProjectionMatrix(P_3x4)
    K_3x3 = out[0]
    R = out[1]
    t = out[2]
    
    # Normalize K
    K_3x3 = K_3x3 / K_3x3[2, 2]
    intrinsics = np.eye(4, dtype=np.float32)
    intrinsics[:3, :3] = K_3x3.astype(np.float32)
    
    # Build camera-to-world pose
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose().astype(np.float32)
    pose[:3, 3] = (t[:3] / t[3])[:, 0].astype(np.float32)
    
    return intrinsics, pose


def load_cameras(npz_path: str | Path) -> dict[str, np.ndarray]:
    """Load cameras from npz file.

    The npz file is expected to contain:
        - world_mat_{i}: P = K @ [R|t] (4x4 or 3x4) - projection matrix
        - scale_mat_{i}: S (4x4) - scaling matrix
        
    Optional (preferred for numerical stability):
        - camera_mat_{i}: K (4x4) - intrinsics
        - camera_mat_inv_{i}: K^-1 (4x4)
        - world_mat_inv_{i}: P^-1 (4x4)
        - scale_mat_inv_{i}: S^-1 (4x4)

    Args:
        npz_path: Path to cameras.npz file.

    Returns:
        Dictionary with camera arrays:
            - K: Intrinsics (n_views, 4, 4)
            - K_inv: Intrinsics inverse (n_views, 4, 4)
            - P: Projection matrices (n_views, 4, 4)
            - P_inv: Projection inverse (n_views, 4, 4)
            - Rt: Extrinsics [R|t] = K_inv @ P (n_views, 4, 4)
            - scale: Scale matrices (n_views, 4, 4)
            - scale_inv: Scale inverse (n_views, 4, 4)
    """
    data = np.load(str(npz_path))

    # Count number of views
    n_views = sum(1 for k in data.keys() if k.startswith("world_mat_") and "inv" not in k)
    
    # Check if inverse matrices are available
    has_inverses = f"camera_mat_inv_0" in data and f"world_mat_inv_0" in data

    cameras = {
        "K": [],
        "K_inv": [],
        "P": [],
        "P_inv": [],
        "Rt": [],
        "scale": [],
        "scale_inv": [],
    }

    for i in range(n_views):
        P = data[f"world_mat_{i}"].astype(np.float32)
        
        # Handle scale matrices
        if f"scale_mat_{i}" in data:
            S = data[f"scale_mat_{i}"].astype(np.float32)
        else:
            S = np.eye(4, dtype=np.float32)
            
        if f"scale_mat_inv_{i}" in data:
            S_inv = data[f"scale_mat_inv_{i}"].astype(np.float32)
        else:
            S_inv = np.eye(4, dtype=np.float32)
        
        if has_inverses:
            # Use precomputed matrices (preferred for stability)
            K = data[f"camera_mat_{i}"].astype(np.float32)
            K_inv = data[f"camera_mat_inv_{i}"].astype(np.float32)
            P_inv = data[f"world_mat_inv_{i}"].astype(np.float32)
            
            # Compute [R|t] = K^-1 @ P (no decomposition needed!)
            Rt = K_inv @ P
        else:
            # Fallback: decompose P using cv2
            K, pose = load_K_Rt_from_P(P)
            K_inv = np.linalg.inv(K).astype(np.float32)
            
            # Convert camera-to-world pose to [R|t] format
            # pose has R^T and camera center, need to convert to [R|t]
            R_c2w = pose[:3, :3]  # This is R^T (world to camera rotation transposed)
            C = pose[:3, 3]  # Camera center
            
            # Build [R|t] in camera frame
            Rt = np.eye(4, dtype=np.float32)
            Rt[:3, :3] = R_c2w.T  # R (world to camera)
            Rt[:3, 3] = -R_c2w.T @ C  # t = -R @ C
            
            # Compute P_inv (approximate)
            P_inv = np.linalg.inv(P).astype(np.float32) if P.shape[0] == 4 else np.eye(4, dtype=np.float32)

        cameras["K"].append(K)
        cameras["K_inv"].append(K_inv)
        cameras["P"].append(P)
        cameras["P_inv"].append(P_inv)
        cameras["Rt"].append(Rt)
        cameras["scale"].append(S)
        cameras["scale_inv"].append(S_inv)

    # Convert to numpy arrays
    for key in cameras:
        cameras[key] = np.array(cameras[key])

    return cameras


def load_cameras_from_sfmdata(sfm_path: str | Path) -> dict[str, np.ndarray]:
    """Load cameras from an AliceVision SfMData JSON file.

    Follows the same conventions as pyalicevisionlib:
    - Intrinsics: fx = focal_mm * width / sensor_width
    - World correction: WORLD_CORR @ R_c2w and WORLD_CORR @ center
    - Projection: P = K @ [R_w2c | -R_w2c @ center]

    Args:
        sfm_path: Path to SfMData .json or .sfm file.

    Returns:
        Camera dict with arrays:
            K, K_inv, P, P_inv, Rt, scale, scale_inv, mask_names
    """
    import json

    # AliceVision world correction (flip Y and Z) — same as pyalicevisionlib
    WORLD_CORR = np.diag([1.0, -1.0, -1.0])

    with open(sfm_path) as f:
        sfm = json.load(f)

    intrinsics_dict = {str(i["intrinsicId"]): i for i in sfm.get("intrinsics", [])}
    poses_dict = {str(p["poseId"]): p["pose"]["transform"]
                  for p in sfm.get("poses", [])}

    cameras = {
        "K": [], "K_inv": [], "P": [], "P_inv": [],
        "Rt": [], "scale": [], "scale_inv": [],
    }
    pose_ids_collected = []
    mask_names_collected = []
    seen_pose_ids = set()

    for view in sfm.get("views", []):
        pose_id = str(view.get("poseId", ""))
        intr_id = str(view.get("intrinsicId", ""))
        mask_id = str(view.get("path", ""))
        if pose_id not in poses_dict or intr_id not in intrinsics_dict:
            continue
        # Deduplicate by poseId (MVPS has 30 views per pose with same camera)
        if pose_id in seen_pose_ids:
            continue
        seen_pose_ids.add(pose_id)

        intr = intrinsics_dict[intr_id]
        transform = poses_dict[pose_id]

        # --- Intrinsics (same formula as pyalicevisionlib.Camera) ---
        w = int(intr["width"])
        h = int(intr["height"])
        focal_mm = float(intr["focalLength"])
        sensor_width = float(intr.get("sensorWidth", 36.0))
        fx = fy = focal_mm * w / sensor_width

        pp = intr.get("principalPoint", ["0", "0"])
        cx = w / 2.0 + float(pp[0])
        cy = h / 2.0 + float(pp[1])

        K = np.eye(4, dtype=np.float64)
        K[0, 0] = fx
        K[1, 1] = fy
        K[0, 2] = cx
        K[1, 2] = cy
        K_inv = np.linalg.inv(K)

        # --- Extrinsics (same as pyalicevisionlib._apply_world_correction) ---
        R_c2w_raw = np.array([float(r) for r in transform["rotation"]]).reshape(3, 3)
        center_raw = np.array([float(c) for c in transform["center"]])

        # Apply world correction
        R_c2w = WORLD_CORR @ R_c2w_raw
        center = WORLD_CORR @ center_raw

        # Build [R|t] (same as pyalicevisionlib.Camera.get_projection_matrix)
        R_w2c = R_c2w.T
        t = -R_w2c @ center

        Rt = np.eye(4, dtype=np.float64)
        Rt[:3, :3] = R_w2c
        Rt[:3, 3] = t

        # Projection P = K @ Rt
        P = K @ Rt
        P_inv = np.linalg.inv(P)

        # No scale matrix for SfMData
        S = np.eye(4, dtype=np.float64)

        cameras["K"].append(K.astype(np.float32))
        cameras["K_inv"].append(K_inv.astype(np.float32))
        cameras["P"].append(P.astype(np.float32))
        cameras["P_inv"].append(P_inv.astype(np.float32))
        cameras["Rt"].append(Rt.astype(np.float32))
        cameras["scale"].append(S.astype(np.float32))
        cameras["scale_inv"].append(S.astype(np.float32))
        pose_ids_collected.append(pose_id)

        mask_id = mask_id.split("/")[-1].split(".")[0]
        mask_names_collected.append(mask_id)




    cameras["mask_names"] = [f"{pid}.png" for pid in mask_names_collected]
    cameras["image_width"] = w
    cameras["image_height"] = h

    for key in cameras:
        if key in ("mask_names", "image_width", "image_height"):
            continue
        cameras[key] = np.array(cameras[key])

    return cameras


def load_cameras_auto(cameras_path: str | Path) -> dict[str, np.ndarray]:
    """Load cameras from any supported format (.npz, .json, .sfm).

    Auto-detects format from file extension and dispatches to the
    appropriate loader.

    Args:
        cameras_path: Path to camera file (.npz, .json, or .sfm).

    Returns:
        Camera dict (same format as load_cameras).
        For SfMData files, also includes 'mask_names' key.
    """
    cameras_path = Path(cameras_path)
    ext = cameras_path.suffix.lower()

    if ext == ".npz":
        return load_cameras(cameras_path)
    elif ext in (".json", ".sfm"):
        return load_cameras_from_sfmdata(cameras_path)
    else:
        raise ValueError(f"Unsupported camera file format: {ext}. Expected .npz, .json, or .sfm")


def get_camera_centers(Rt_array: np.ndarray) -> np.ndarray:
    """Extract camera centers from [R|t] matrices.

    Camera center C = -R^T @ t

    Args:
        Rt_array: Array of [R|t] matrices, shape (n_views, 4, 4).

    Returns:
        Camera centers, shape (n_views, 3).
    """
    centers = []
    for Rt in Rt_array:
        R = Rt[:3, :3]
        t = Rt[:3, 3]
        C = -R.T @ t
        centers.append(C)
    return np.array(centers, dtype=np.float32)


def project_points(points: np.ndarray, P: np.ndarray, min_depth: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """Project 3D points to 2D using projection matrix.

    Args:
        points: 3D points, shape (N, 3).
        P: Projection matrix (4x4), includes intrinsics.
        min_depth: Minimum depth threshold (points behind camera are invalid).

    Returns:
        Tuple of:
            - 2D points, shape (N, 2). Invalid points have NaN values.
            - valid_mask: Boolean array, True for points in front of camera.
    """
    N = points.shape[0]
    points_h = np.hstack([points, np.ones((N, 1), dtype=points.dtype)])

    projected = points_h @ P.T

    # Check for points behind or on camera plane
    depth = projected[:, 2]
    valid_mask = depth > min_depth

    # Initialize with NaN for invalid points
    projected_2d = np.full((N, 2), np.nan, dtype=np.float32)

    # Only divide for valid points
    if valid_mask.any():
        projected_2d[valid_mask] = (
            projected[valid_mask, :2] / projected[valid_mask, 2:3]
        ).astype(np.float32)

    return projected_2d, valid_mask
