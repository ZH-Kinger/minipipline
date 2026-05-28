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
    "single_aa_per_frame_to_flat_rotmat",
    "joint_aa_per_frame_to_flat_rotmat",
]
