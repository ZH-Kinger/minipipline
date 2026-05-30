"""Canonical action-vector computation from the per-frame state sequence.

The ``action`` feature (102-dim) is *derived* from ``observation.state`` rather
than stored as raw columns. This module is the **single source of truth** for
that derivation, used both to compute real ``meta/stats.json`` action statistics
at write time and as the reference implementation any dataloader should mirror.

Convention (per hand, camera frame; see ``schema.ACTION_LAYOUT``):
    delta_transl[t]  = transl[t+1] - transl[t]                     (last frame → 0)
    delta_orient[t]  = log( R(orient[t+1]) @ R(orient[t])^T )       (last frame → 0)
    next_mano_pose[t]= mano_pose[t+1]                              (last frame → mano_pose[t])

i.e. the action is the transition that carries the current state to the next
frame's state — the standard next-step target for imitation learning. The final
frame has no successor, so deltas are zero and ``next_mano_pose`` repeats.
"""

from __future__ import annotations

import numpy as np

from .rotations import axis_angle_to_rotmat, rotmat_to_axis_angle
from .schema import ACTION_DIM, ACTION_LAYOUT, STATE_LAYOUT


def _hand_action(
    transl: np.ndarray, orient_aa: np.ndarray, mano_pose: np.ndarray
) -> np.ndarray:
    """Compute one hand's (T, 51) action block from its (T,*) state slices."""
    T = transl.shape[0]
    out = np.zeros((T, 51), dtype=np.float32)
    if T == 0:
        return out
    # delta translation (3): next - current, last frame zero.
    out[:-1, 0:3] = (transl[1:] - transl[:-1]).astype(np.float32)
    # delta orientation (3): relative rotation next ∘ current^-1 → axis-angle.
    for t in range(T - 1):
        R_t = axis_angle_to_rotmat(orient_aa[t]).astype(np.float64)
        R_n = axis_angle_to_rotmat(orient_aa[t + 1]).astype(np.float64)
        out[t, 3:6] = rotmat_to_axis_angle(R_n @ R_t.T)
    # next-frame MANO pose (45): shift by one, last frame repeats.
    out[:-1, 6:51] = mano_pose[1:].astype(np.float32)
    out[-1, 6:51] = mano_pose[-1].astype(np.float32)
    return out


def compute_action_from_state(state: np.ndarray) -> np.ndarray:
    """(T, 122) state sequence → (T, 102) action sequence (camera frame).

    See module docstring for the exact convention. The two hand blocks are laid
    out per ``ACTION_LAYOUT`` (left 0:51, right 51:102).
    """
    state = np.asarray(state, dtype=np.float64)
    T = state.shape[0]
    action = np.zeros((T, ACTION_DIM), dtype=np.float32)
    sl = STATE_LAYOUT
    for side, base in (("left", 0), ("right", 51)):
        transl = state[:, sl[f"{side}_wrist_transl_cam"][0]: sl[f"{side}_wrist_transl_cam"][1]]
        orient = state[:, sl[f"{side}_wrist_orient_cam_aa"][0]: sl[f"{side}_wrist_orient_cam_aa"][1]]
        pose = state[:, sl[f"{side}_mano_pose_aa"][0]: sl[f"{side}_mano_pose_aa"][1]]
        action[:, base: base + 51] = _hand_action(transl, orient, pose)
    return action


__all__ = ["compute_action_from_state"]
