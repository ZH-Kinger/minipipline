"""Episode-level retargeting: human MANO keypoints → robot qpos + EE pose.

For each hand we (1) build a per-frame wrist-local frame from the keypoints
(so global hand motion is removed and only finger shape remains), (2) estimate a
single Kabsch alignment + size scale from the episode's mean fingertip layout
(absorbing the MANO↔Wuji convention difference, data-driven not hard-coded), and
(3) solve the hand joints per frame to match the aligned, scaled fingertip
targets. Arm joints (if the URDF has them) are reserved for arm IK; until an arm
URDF is supplied they stay NaN — honest "not computed", never a placeholder.
"""

from __future__ import annotations

import numpy as np

from ..lerobot_v3.schema import (
    ROBOT_EE_LAYOUT, ROBOT_EE_POSE_DIM, ROBOT_QPOS_DIM, ROBOT_QPOS_LAYOUT,
)
from .optimize import kabsch_rotation, solve_arm_qpos, solve_hand_qpos
from .robot import MANO_FINGERTIPS, MANO_WRIST, RobotModel

# MANO MCP (knuckle) keypoints used to build a stable wrist-local frame.
_MCP_INDEX, _MCP_MIDDLE = 5, 9

_MODEL_CACHE: dict[tuple[str, tuple[int, ...]], RobotModel] = {}


def _get_model(urdf: str, finger_map: tuple[int, ...]) -> RobotModel:
    key = (urdf, tuple(finger_map))
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = RobotModel(urdf, finger_map=finger_map)
    return _MODEL_CACHE[key]


def _hand_basis(kp: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Wrist-local orthonormal frame R (cols ex,ey,ez) + origin from 21 keypoints.

    ez = wrist→middle-MCP (finger-extension axis); ex completes the index plane;
    ey = ez × ex. Returns None if degenerate (collinear / zero keypoints).
    """
    wrist = kp[MANO_WRIST]
    ez = kp[_MCP_MIDDLE] - wrist
    nz = np.linalg.norm(ez)
    if nz < 1e-6:
        return None
    ez = ez / nz
    x0 = kp[_MCP_INDEX] - wrist
    x0 = x0 - (x0 @ ez) * ez
    nx = np.linalg.norm(x0)
    if nx < 1e-6:
        return None
    ex = x0 / nx
    ey = np.cross(ez, ex)
    R = np.column_stack([ex, ey, ez])
    return R, wrist


def _rotmat_to_aa(R: np.ndarray) -> np.ndarray:
    """Rotation matrix → axis-angle (3,)."""
    tr = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    theta = float(np.arccos(tr))
    if theta < 1e-8:
        return np.zeros(3)
    s = 2.0 * np.sin(theta)
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / s
    return axis * theta


def _human_fingertip_vectors(kp: np.ndarray, finger_map: tuple[int, ...]) -> np.ndarray | None:
    """Per-frame fingertip vectors in the wrist-local frame, reordered to robot
    finger order. Returns (5,3) or None if the frame is degenerate."""
    basis = _hand_basis(kp)
    if basis is None:
        return None
    R, wrist = basis
    tips = np.array([kp[MANO_FINGERTIPS[finger_map[i]]] for i in range(5)])  # (5,3)
    return (tips - wrist) @ R  # = R.T @ (tip - wrist) per row, in wrist-local frame


def _retarget_hand(
    kp_seq: np.ndarray, kept: np.ndarray, model: RobotModel,
    finger_map: tuple[int, ...], smooth_weight: float, n_iters: int,
    scale_override: float | None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Retarget one hand's (T,21,3) keypoints. Returns (qpos_hand (T,n_hand),
    ee_pose (T,6), diagnostics). NaN rows where not kept / degenerate."""
    T = kp_seq.shape[0]
    qpos = np.full((T, model.n_hand), np.nan, dtype=np.float64)
    qpos_arm = np.full((T, model.n_arm), np.nan, dtype=np.float64)
    ee = np.full((T, 6), np.nan, dtype=np.float64)

    valid_vecs, valid_t = [], []
    bases = [None] * T
    for t in range(T):
        if not kept[t]:
            continue
        v = _human_fingertip_vectors(kp_seq[t], finger_map)
        if v is None:
            continue
        valid_vecs.append(v)
        valid_t.append(t)
        bases[t] = _hand_basis(kp_seq[t])

    diag = {"n_valid": len(valid_t), "fingertip_err_mm": [], "align_residual_mm": None,
            "limit_ok": [], "wrist_resid_mm": []}
    if not valid_vecs:
        return qpos, qpos_arm, ee, diag

    # Under-actuated arm IK: anchor the base so the rest-pose palm lands on the
    # wrist-trajectory centroid (base = centroid − rest_palm_offset), so the arm
    # sweeps around the actual wrist workspace. Synthetic-arm convention; replace
    # with the real base calibration + arm URDF.
    arm_base = None
    if model.n_arm:
        wrist_centroid = np.mean([kp_seq[t][MANO_WRIST] for t in valid_t], axis=0)
        palm_rest = model.palm_pose(model.tree.midpoint())[:3, 3]
        arm_base = wrist_centroid - palm_rest

    V = np.stack(valid_vecs)                 # (Tk,5,3)
    Vbar = V.mean(axis=0)                    # (5,3) mean human layout (robot order)
    V_robot = model.fingertips_local(model.tree.midpoint())  # (5,3)
    # Per-finger scale: human vs robot finger-length ratios differ, so one global
    # scale can't match all 5 fingers. Scale each finger to its robot length,
    # then align orientation in that scaled space (tighter than a single scale).
    robot_len = np.linalg.norm(V_robot, axis=1)            # (5,)
    human_len = np.linalg.norm(Vbar, axis=1)               # (5,)
    if scale_override is not None:
        scale_vec = np.full(5, float(scale_override))
    else:
        scale_vec = np.where(human_len > 1e-6, robot_len / human_len, 1.0)
    R_align = kabsch_rotation(scale_vec[:, None] * Vbar, V_robot)
    diag["align_residual_mm"] = float(
        np.mean(np.linalg.norm(scale_vec[:, None] * (Vbar @ R_align.T) - V_robot, axis=1)) * 1000)
    diag["scale"] = float(np.mean(scale_vec))

    lo, hi = model.hand_limits[:, 0], model.hand_limits[:, 1]
    prev = None
    prev_arm = None
    for t in valid_t:
        v = _human_fingertip_vectors(kp_seq[t], finger_map)
        target = scale_vec[:, None] * (v @ R_align.T)   # (5,3) palm-local, per-finger scaled
        q = solve_hand_qpos(model, target, q_init=prev, smooth_ref=prev,
                            smooth_weight=smooth_weight, n_iters=n_iters)
        qpos[t] = q
        prev = q
        # diagnostics
        full = model.tree.midpoint().copy(); full[model.hand_idx] = q
        fk_tips = model.fingertips_local(full)
        diag["fingertip_err_mm"].append(float(np.mean(np.linalg.norm(fk_tips - target, axis=1)) * 1000))
        diag["limit_ok"].append(bool(np.all((q >= lo - 1e-6) & (q <= hi + 1e-6))))
        R, wrist = bases[t]
        ee[t, :3] = wrist
        ee[t, 3:] = _rotmat_to_aa(R)
        # Under-actuated arm IK (only if the URDF has arm joints).
        if model.n_arm:
            qa, resid = solve_arm_qpos(model, wrist - arm_base, q_arm_init=prev_arm,
                                       smooth_ref=prev_arm, smooth_weight=smooth_weight)
            qpos_arm[t] = qa
            prev_arm = qa
            diag["wrist_resid_mm"].append(resid)
    return qpos, qpos_arm, ee, diag


def retarget_episode(
    keypoints_world: np.ndarray, kept: np.ndarray, cfg=None,
    *, return_diagnostics: bool = False,
):
    """Retarget an episode's two hands.

    Args:
        keypoints_world: (2, T, 21, 3) — left/right MANO keypoints (world frame).
        kept: (2, T) bool — per-frame per-hand validity.
        cfg: RetargetConfig (defaults to from_env()).

    Returns dict with ``robot_qpos`` (T,46) and ``robot_ee_pose`` (T,12),
    NaN-filled where unavailable. With ``return_diagnostics`` also ``diag``.
    """
    if cfg is None:
        from ..config.retarget import RetargetConfig
        cfg = RetargetConfig.from_env()

    kp = np.asarray(keypoints_world, np.float64)
    kept = np.asarray(kept, bool)
    T = kp.shape[1]
    qpos = np.full((T, ROBOT_QPOS_DIM), np.nan, dtype=np.float32)
    ee = np.full((T, ROBOT_EE_POSE_DIM), np.nan, dtype=np.float32)
    diags = {}

    for h, hand in enumerate(("left", "right")):
        model = _get_model(cfg.urdf_for(hand), cfg.finger_map)
        q_hand, q_arm, ee_hand, diag = _retarget_hand(
            kp[h], kept[h], model, cfg.finger_map, cfg.smooth_weight,
            cfg.n_iters, cfg.scale)
        hslice = ROBOT_QPOS_LAYOUT[f"{hand}_hand"]
        qpos[:, hslice[0]:hslice[1]] = q_hand
        # Arm joints: filled when the URDF has them, else left NaN.
        if model.n_arm:
            aslice = ROBOT_QPOS_LAYOUT[f"{hand}_arm"]
            qpos[:, aslice[0]:aslice[1]] = q_arm
        ee_t = ROBOT_EE_LAYOUT[f"{hand}_ee_transl"]
        ee_o = ROBOT_EE_LAYOUT[f"{hand}_ee_orient_aa"]
        ee[:, ee_t[0]:ee_t[1]] = ee_hand[:, :3]
        ee[:, ee_o[0]:ee_o[1]] = ee_hand[:, 3:]
        diags[hand] = diag

    out = {"robot_qpos": qpos, "robot_ee_pose": ee}
    if return_diagnostics:
        out["diag"] = diags
    return out


def self_check(keypoints_world: np.ndarray, kept: np.ndarray, cfg=None) -> dict:
    """Run retargeting and return per-hand fit quality (mm / %), for QC."""
    res = retarget_episode(keypoints_world, kept, cfg, return_diagnostics=True)
    summary = {}
    for hand, d in res["diag"].items():
        errs = d["fingertip_err_mm"]
        ok = d["limit_ok"]
        wr = d.get("wrist_resid_mm") or []
        summary[hand] = {
            "n_valid_frames": d["n_valid"],
            "mean_fingertip_err_mm": float(np.mean(errs)) if errs else None,
            "max_fingertip_err_mm": float(np.max(errs)) if errs else None,
            "align_residual_mm": d["align_residual_mm"],
            "scale": d.get("scale"),
            "limit_compliance": (float(np.mean(ok)) if ok else None),
            "arm_wrist_resid_mm": (float(np.mean(wr)) if wr else None),
        }
    return summary


__all__ = ["retarget_episode", "self_check"]
