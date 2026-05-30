from __future__ import annotations

import numpy as np


def axis_angle_to_rotmat(aa: np.ndarray) -> np.ndarray:
    """Rodrigues' formula. `aa` is (..., 3); returns (..., 3, 3) float32."""
    aa = np.asarray(aa, dtype=np.float32)
    leading = aa.shape[:-1]
    flat = aa.reshape(-1, 3)
    out = np.empty((flat.shape[0], 3, 3), dtype=np.float32)
    for i, v in enumerate(flat):
        theta = float(np.linalg.norm(v))
        if theta < 1e-8:
            out[i] = np.eye(3, dtype=np.float32)
            continue
        k = v / theta
        K = np.array(
            [[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]],
            dtype=np.float32,
        )
        out[i] = (
            np.eye(3, dtype=np.float32)
            + np.sin(theta) * K
            + (1.0 - np.cos(theta)) * (K @ K)
        )
    return out.reshape(*leading, 3, 3)


def rotmat_to_axis_angle(R: np.ndarray) -> np.ndarray:
    """3×3 rotation matrix → (3,) axis-angle (Rodrigues log map), robust near 0/π."""
    R = np.asarray(R, dtype=np.float64)
    cos_theta = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    theta = np.arccos(cos_theta)
    if theta < 1e-6:
        return np.zeros(3, dtype=np.float32)
    if abs(np.pi - theta) < 1e-3:
        A = (R + np.eye(3)) * 0.5
        k = int(np.argmax(np.diag(A)))
        axis = A[:, k] / np.sqrt(max(A[k, k], 1e-12))
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        return (axis * theta).astype(np.float32)
    rx, ry, rz = R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]
    axis = np.array([rx, ry, rz], dtype=np.float64) / (2.0 * np.sin(theta))
    return (axis * theta).astype(np.float32)


def single_aa_per_frame_to_flat_rotmat(aa_seq: np.ndarray) -> np.ndarray:
    """`(T, 3)` axis-angle per frame → `(T, 9)` flattened rotation matrix per frame."""
    aa_seq = np.asarray(aa_seq, dtype=np.float32)
    if aa_seq.ndim != 2 or aa_seq.shape[-1] != 3:
        raise ValueError(
            f"single_aa_per_frame_to_flat_rotmat expects (T,3), got {aa_seq.shape}"
        )
    rotmats = axis_angle_to_rotmat(aa_seq)  # (T, 3, 3)
    return rotmats.reshape(aa_seq.shape[0], 9).astype(np.float32)


def joint_aa_per_frame_to_flat_rotmat(aa_seq: np.ndarray) -> np.ndarray:
    """`(T, N_joints, 3)` axis-angle per joint per frame → `(T, N_joints*9)` flattened."""
    aa_seq = np.asarray(aa_seq, dtype=np.float32)
    if aa_seq.ndim != 3 or aa_seq.shape[-1] != 3:
        raise ValueError(
            f"joint_aa_per_frame_to_flat_rotmat expects (T,N,3), got {aa_seq.shape}"
        )
    T, N, _ = aa_seq.shape
    rotmats = axis_angle_to_rotmat(aa_seq.reshape(-1, 3))  # (T*N, 3, 3)
    return rotmats.reshape(T, N * 9).astype(np.float32)


__all__ = [
    "axis_angle_to_rotmat",
    "rotmat_to_axis_angle",
    "single_aa_per_frame_to_flat_rotmat",
    "joint_aa_per_frame_to_flat_rotmat",
]
