"""Real-data Layer 2 backend: assemble MergedPrediction from NIR ground truth.

Instead of running (or mock-fabricating) the 5-stage HaWoR/MoGe/GeoCalib/MegaSAM
chain, this reads the real signals that Layer 1 already extracted into the NIR
directory and assembles a :class:`MergedPrediction` directly:

  - real 3D hand keypoints (camera frame, metres) → wrist + 21 joints, world frame
  - real head 6DOF trajectory → camera-to-world (c2w) per frame
  - real camera intrinsics (Kalibr) → K

Coordinate convention: keypoints arrive in CAMERA frame. We transform them to
WORLD frame with the per-frame head pose c2w (p_world = R(q) @ p_cam + t), which
is exactly what Layer 3 expects (it later applies extrinsics_w2c to recover the
camera frame for the state vector). The round-trip world→camera should reproduce
the original camera-frame coordinates, giving a built-in correctness check.

Activated by ``MMPIPE_HANDPOSE_BACKEND=real_ingest``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from .schemas import (
    CameraIntrinsics,
    CameraTrajectory,
    MergedPrediction,
    VideoMeta,
)

HAND_LEFT = 0
HAND_RIGHT = 1
N_KP = 21  # MANO-style keypoints per hand


# ---------------------------------------------------------------------------
# Small rotation helpers (numpy only)
# ---------------------------------------------------------------------------


def quat_to_rotmat(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Unit quaternion (x, y, z, w) → 3x3 rotation matrix."""
    n = qx * qx + qy * qy + qz * qz + qw * qw
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    s = 2.0 / n
    xx, yy, zz = qx * qx * s, qy * qy * s, qz * qz * s
    xy, xz, yz = qx * qy * s, qx * qz * s, qy * qz * s
    wx, wy, wz = qw * qx * s, qw * qy * s, qw * qz * s
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=np.float64,
    )


def rotmat_to_axis_angle(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix → (3,) axis-angle (Rodrigues), numerically robust."""
    # Clamp trace for arccos domain.
    cos_theta = (np.trace(R) - 1.0) * 0.5
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    theta = np.arccos(cos_theta)
    if theta < 1e-6:
        return np.zeros(3, dtype=np.float64)
    if abs(np.pi - theta) < 1e-3:
        # Near 180°: axis from the largest diagonal of (R+I)/2.
        A = (R + np.eye(3)) * 0.5
        k = int(np.argmax(np.diag(A)))
        axis = A[:, k] / (np.sqrt(max(A[k, k], 1e-12)))
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        return (axis * theta).astype(np.float64)
    rx = R[2, 1] - R[1, 2]
    ry = R[0, 2] - R[2, 0]
    rz = R[1, 0] - R[0, 1]
    axis = np.array([rx, ry, rz], dtype=np.float64) / (2.0 * np.sin(theta))
    return (axis * theta).astype(np.float64)


def axis_angle_to_rotmat(aa: np.ndarray) -> np.ndarray:
    """(3,) axis-angle → 3×3 rotation matrix (Rodrigues exp map)."""
    theta = float(np.linalg.norm(aa))
    if theta < 1e-8:
        return np.eye(3, dtype=np.float64)
    k = aa / theta
    K = np.array(
        [[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]],
        dtype=np.float64,
    )
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def _wrist_orientation_aa(kp_world: np.ndarray) -> np.ndarray:
    """Derive a wrist axis-angle orientation from world-frame keypoint geometry.

    kp_world: (21, 3). Builds an orthonormal hand frame:
      x = wrist → index_mcp  (radial / side)
      z = wrist → middle_mcp (forward along the hand)
      y = z × x              (palm normal)
    then Gram-Schmidt orthonormalises and converts the resulting basis to AA.
    Returns zeros if the keypoints are degenerate (e.g. NaN).
    """
    wrist = kp_world[0]
    index_mcp = kp_world[5]
    middle_mcp = kp_world[9]
    if not np.all(np.isfinite([wrist, index_mcp, middle_mcp])):
        return np.zeros(3, dtype=np.float64)
    x = index_mcp - wrist
    z = middle_mcp - wrist
    nx = np.linalg.norm(x)
    nz = np.linalg.norm(z)
    if nx < 1e-6 or nz < 1e-6:
        return np.zeros(3, dtype=np.float64)
    x = x / nx
    z = z - np.dot(z, x) * x          # orthogonalise z against x
    nz = np.linalg.norm(z)
    if nz < 1e-6:
        return np.zeros(3, dtype=np.float64)
    z = z / nz
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=1)   # columns are basis vectors
    return rotmat_to_axis_angle(R)


# ---------------------------------------------------------------------------
# NIR readers
# ---------------------------------------------------------------------------


def _read_intrinsics(nir_dir: Path, video: VideoMeta) -> CameraIntrinsics:
    """Build CameraIntrinsics from calibration.json (rgb block). Scales K if the
    decoded video resolution differs from the calibrated resolution."""
    calib = json.loads((nir_dir / "calibration.json").read_text(encoding="utf-8"))
    rgb = calib.get("rgb") or calib.get("depth") or {}
    fx = float(rgb.get("fx", video.width))
    fy = float(rgb.get("fy", video.height))
    cx = float(rgb.get("cx", video.width / 2.0))
    cy = float(rgb.get("cy", video.height / 2.0))
    calib_w = int(rgb.get("width", video.width)) or video.width
    calib_h = int(rgb.get("height", video.height)) or video.height
    # Scale to the actual decoded resolution if it was resized.
    sx = video.width / calib_w if calib_w else 1.0
    sy = video.height / calib_h if calib_h else 1.0
    fx, cx = fx * sx, cx * sx
    fy, cy = fy * sy, cy * sy
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    import math

    hfov = 2.0 * math.degrees(math.atan((video.width / 2.0) / fx)) if fx else 0.0
    vfov = 2.0 * math.degrees(math.atan((video.height / 2.0) / fy)) if fy else 0.0
    return CameraIntrinsics(
        video_id=video.video_id,
        K=K,
        dist=np.zeros(5, dtype=np.float32),
        width=video.width,
        height=video.height,
        vfov_deg=vfov,
        hfov_deg=hfov,
    )


def _read_c2w_per_frame(nir_dir: Path, n_frames: int) -> np.ndarray:
    """Read head_pose.parquet → (N, 4, 4) camera-to-world transforms.

    Missing frames fall back to the nearest available pose (identity if none).
    """
    c2w = np.tile(np.eye(4, dtype=np.float64), (n_frames, 1, 1))
    p = nir_dir / "head_pose.parquet"
    if not p.exists():
        return c2w
    d = pq.read_table(p).to_pydict()
    idx = d.get("frame_index", [])
    last_T: np.ndarray | None = None
    by_frame: dict[int, np.ndarray] = {}
    for j, fi in enumerate(idx):
        R = quat_to_rotmat(d["qx"][j], d["qy"][j], d["qz"][j], d["qw"][j])
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = [d["tx"][j], d["ty"][j], d["tz"][j]]
        by_frame[int(fi)] = T
    for i in range(n_frames):
        if i in by_frame:
            last_T = by_frame[i]
        if last_T is not None:
            c2w[i] = last_T
    return c2w


def _read_keypoints(nir_dir: Path, n_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """Read hand_keypoints.parquet → (kp_cam (2,N,21,3), valid (2,N) bool).

    kp_cam is in CAMERA-frame metres; NaN where a hand/keypoint is missing.
    valid is True where the hand is present with finite keypoints.
    """
    kp = np.full((2, n_frames, N_KP, 3), np.nan, dtype=np.float64)
    valid = np.zeros((2, n_frames), dtype=bool)
    p = nir_dir / "hand_keypoints.parquet"
    if not p.exists():
        return kp, valid
    d = pq.read_table(p).to_pydict()
    n = len(d["frame_index"])
    for j in range(n):
        fi = int(d["frame_index"][j])
        if not (0 <= fi < n_frames):
            continue
        for hand, present_key, flat_key in (
            (HAND_LEFT, "left_present", "left_keypoints_3d_flat"),
            (HAND_RIGHT, "right_present", "right_keypoints_3d_flat"),
        ):
            flat = d.get(flat_key, [None] * n)[j]
            if not d.get(present_key, [False] * n)[j] or flat is None:
                continue
            arr = np.asarray(flat, dtype=np.float64)
            if arr.shape[0] != N_KP * 3 or not np.all(np.isfinite(arr)):
                continue
            kp[hand, fi] = arr.reshape(N_KP, 3)
            valid[hand, fi] = True
    return kp, valid


def _read_mano(
    nir_dir: Path, n_frames: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read real MANO params from hand_keypoints.parquet.

    Returns ``(pose (2,N,45), betas (2,N,10), global_orient (2,N,3))``,
    NaN-filled where a frame/hand has no MANO block (or the column is absent,
    e.g. NIR produced before this field existed).
    """
    pose = np.full((2, n_frames, 45), np.nan, dtype=np.float64)
    betas = np.full((2, n_frames, 10), np.nan, dtype=np.float64)
    gorient = np.full((2, n_frames, 3), np.nan, dtype=np.float64)
    p = nir_dir / "hand_keypoints.parquet"
    if not p.exists():
        return pose, betas, gorient
    d = pq.read_table(p).to_pydict()
    cols = set(d.keys())
    if "left_mano_pose_aa" not in cols:  # older NIR without MANO columns
        return pose, betas, gorient
    idx = d["frame_index"]
    for j, fi in enumerate(idx):
        fi = int(fi)
        if not (0 <= fi < n_frames):
            continue
        for hand, pre, po, be, go in (
            (HAND_LEFT, "left_present", "left_mano_pose_aa", "left_mano_betas", "left_global_orient"),
            (HAND_RIGHT, "right_present", "right_mano_pose_aa", "right_mano_betas", "right_global_orient"),
        ):
            for arr, key, dim in ((pose, po, 45), (betas, be, 10), (gorient, go, 3)):
                v = d.get(key, [None] * len(idx))[j]
                if v is None:
                    continue
                a = np.asarray(v, dtype=np.float64)
                if a.shape[0] == dim and np.all(np.isfinite(a)):
                    arr[hand, fi] = a
    return pose, betas, gorient


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def build_merged_from_nir(nir_dir: str | Path, video: VideoMeta) -> MergedPrediction:
    """Assemble a MergedPrediction from real NIR ground truth (no mock chain)."""
    nir_dir = Path(nir_dir)
    n = int(video.n_frames)

    intrinsics = _read_intrinsics(nir_dir, video)
    c2w = _read_c2w_per_frame(nir_dir, n)              # (N,4,4)
    kp_cam, valid = _read_keypoints(nir_dir, n)        # (2,N,21,3), (2,N)

    # Camera → world for every keypoint: p_w = R @ p_c + t.
    R = c2w[:, :3, :3]                                 # (N,3,3)
    t = c2w[:, :3, 3]                                  # (N,3)
    kp_world = np.full_like(kp_cam, np.nan)            # (2,N,21,3)
    for hand in (HAND_LEFT, HAND_RIGHT):
        # einsum over frames: (N,3,3) @ (N,21,3) → (N,21,3)
        pc = kp_cam[hand]                              # (N,21,3)
        finite = np.all(np.isfinite(pc), axis=(1, 2))  # (N,)
        pw = np.einsum("nij,nkj->nki", R, np.nan_to_num(pc))
        pw = pw + t[:, None, :]
        pw[~finite] = np.nan
        kp_world[hand] = pw

    pred_trans = kp_world[:, :, 0, :].astype(np.float32)        # wrist = kp 0
    pred_trans = np.nan_to_num(pred_trans)                      # zero-fill invalid

    # Real MANO from the source tracker (camera frame).
    mano_pose, mano_betas, mano_gorient = _read_mano(nir_dir, n)  # (2,N,45),(2,N,10),(2,N,3)

    # MANO 15-joint finger pose is wrist-local (frame-independent) → use directly.
    pred_hand_pose = np.nan_to_num(mano_pose).astype(np.float32)   # zero where absent
    pred_betas = np.nan_to_num(mano_betas).astype(np.float32)

    # Wrist orientation (world frame). Prefer real MANO global_orient (camera
    # frame) rotated into world by the per-frame head pose: R_world = R_c2w @
    # R(global_orient). Fall back to keypoint-geometry estimate when the MANO
    # block is absent for that frame/hand.
    pred_rot = np.zeros((2, n, 3), dtype=np.float32)
    for hand in (HAND_LEFT, HAND_RIGHT):
        for i in range(n):
            if not valid[hand, i]:
                continue
            g = mano_gorient[hand, i]
            if np.all(np.isfinite(g)):
                R_world = R[i] @ axis_angle_to_rotmat(g)
                pred_rot[hand, i] = rotmat_to_axis_angle(R_world).astype(np.float32)
            else:
                pred_rot[hand, i] = _wrist_orientation_aa(kp_world[hand, i]).astype(np.float32)

    pred_kept = valid.copy()

    trajectory = CameraTrajectory(
        video_id=video.video_id,
        clip_idx=-1,
        frame_start=0,
        cam_c2w=c2w.astype(np.float32),
        K=intrinsics.K,
        slam_hw=(video.height, video.width),
        valid=np.ones(n, dtype=bool),
    )

    return MergedPrediction(
        video_id=video.video_id,
        pred_trans=pred_trans,
        pred_rot=pred_rot,
        pred_hand_pose=pred_hand_pose,
        pred_betas=pred_betas,
        pred_valid=valid,
        pred_kept=pred_kept,
        trajectory=trajectory,
        intrinsics=intrinsics,
        hand_keypoints_world=kp_world.astype(np.float32),
    )


__all__ = ["build_merged_from_nir", "quat_to_rotmat", "rotmat_to_axis_angle"]
