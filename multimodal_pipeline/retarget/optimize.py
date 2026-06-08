"""Retargeting solvers (pure-numpy Levenberg–Marquardt; no scipy / nlopt).

Two pieces:

* :func:`kabsch_rotation` — the single rotation that best aligns the human
  wrist-frame fingertip layout to the robot palm-frame layout. The MANO wrist
  convention and the Wuji palm convention differ by an unknown fixed rotation;
  rather than hard-code a guess, we estimate it per-hand from the data (the
  mean fingertip layout), so only the *relative* per-frame finger motion drives
  the solve. The residual is reported by the self-check, never hidden.

* :func:`solve_hand_qpos` — solves the 20 hand joints so the robot's palm-local
  fingertips match the (aligned, scaled) human fingertip targets, with a small
  pull toward a warm-start pose for temporal smoothness, clamped to joint limits.

(Arm IK for the under-actuated 3-DOF wrist lands once the arm URDF is supplied;
its scaffold mirrors :func:`solve_hand_qpos`.)
"""

from __future__ import annotations

import numpy as np


def kabsch_rotation(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Best-fit rotation R (3×3, det=+1) minimising ‖R·aᵢ − bᵢ‖ over rows.

    Vectors emanate from a common origin (palm/wrist), so we rotate about that
    origin — no centroid subtraction.
    """
    A = np.asarray(A, np.float64).reshape(-1, 3)
    B = np.asarray(B, np.float64).reshape(-1, 3)
    H = A.T @ B
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    return Vt.T @ D @ U.T


def solve_hand_qpos(
    model,
    target_local: np.ndarray,
    q_init: np.ndarray | None = None,
    smooth_ref: np.ndarray | None = None,
    smooth_weight: float = 0.01,
    n_iters: int = 30,
    lm_damping: float = 1e-3,
    tol: float = 1e-8,
) -> np.ndarray:
    """Solve the 20 hand joints to match palm-local fingertip targets (5,3).

    Levenberg–Marquardt on the stacked residual
    ``[fingertips_local(q) − target ; √w·(q − smooth_ref)]`` with joint-limit
    clamping each step. Returns the hand-joint vector (n_hand,).
    """
    n_hand = model.n_hand
    lo, hi = model.hand_limits[:, 0], model.hand_limits[:, 1]
    if q_init is None:
        q_init = 0.5 * (lo + hi)
    q = np.clip(np.asarray(q_init, np.float64).copy(), lo, hi)
    if smooth_ref is None:
        smooth_ref = 0.5 * (lo + hi)
    sw = float(smooth_weight)
    target = np.asarray(target_local, np.float64).reshape(5, 3)

    # Build a full-DOF qpos with the (fixed) arm part from q_init's neutral; the
    # hand columns are what we optimise.
    full = model.tree.midpoint()

    def residual_and_jac(qh: np.ndarray):
        full[model.hand_idx] = qh
        tips, jac = model.fingertips_local_with_jac(full)  # (5,3), (5,3,n_hand)
        r_tip = (tips - target).reshape(-1)                # (15,)
        J_tip = jac.reshape(-1, n_hand)                    # (15, n_hand)
        if sw > 0:
            r_sm = np.sqrt(sw) * (qh - smooth_ref)
            J_sm = np.sqrt(sw) * np.eye(n_hand)
            return np.concatenate([r_tip, r_sm]), np.concatenate([J_tip, J_sm], axis=0)
        return r_tip, J_tip

    r, J = residual_and_jac(q)
    cost = float(r @ r)
    lam = lm_damping
    for _ in range(n_iters):
        H = J.T @ J + lam * np.eye(n_hand)
        g = J.T @ r
        try:
            dq = np.linalg.solve(H, -g)
        except np.linalg.LinAlgError:
            break
        q_new = np.clip(q + dq, lo, hi)
        r_new, J_new = residual_and_jac(q_new)
        cost_new = float(r_new @ r_new)
        if cost_new < cost:
            if cost - cost_new < tol:
                q, r, J, cost = q_new, r_new, J_new, cost_new
                break
            q, r, J, cost = q_new, r_new, J_new, cost_new
            lam = max(lam * 0.5, 1e-9)
        else:
            lam = min(lam * 4.0, 1e6)
    return q


def solve_arm_qpos(
    model,
    target_palm_pos: np.ndarray,
    q_arm_init: np.ndarray | None = None,
    smooth_ref: np.ndarray | None = None,
    smooth_weight: float = 0.01,
    n_iters: int = 40,
    lm_damping: float = 1e-3,
) -> tuple[np.ndarray, float]:
    """Under-actuated arm IK: solve the arm joints so the palm origin reaches
    ``target_palm_pos`` (in the robot base frame). Returns ``(q_arm, resid_mm)``.

    A 3-DOF rotational arm cannot reach an arbitrary 3-D point (fixed radius),
    so the residual is reported, not hidden — it quantifies the under-actuation.
    """
    n_arm = model.n_arm
    if n_arm == 0:
        return np.zeros(0), 0.0
    lo, hi = model.arm_limits[:, 0], model.arm_limits[:, 1]
    if q_arm_init is None:
        q_arm_init = 0.5 * (lo + hi)
    qa = np.clip(np.asarray(q_arm_init, np.float64).copy(), lo, hi)
    if smooth_ref is None:
        smooth_ref = 0.5 * (lo + hi)
    sw = float(smooth_weight)
    target = np.asarray(target_palm_pos, np.float64).reshape(3)

    full = model.tree.midpoint().copy()

    def res_jac(q):
        full[model.arm_idx] = q
        palm, J = model.palm_position_jac(full)  # (4,4), (3,n_arm)
        r = palm[:3, 3] - target
        if sw > 0:
            r = np.concatenate([r, np.sqrt(sw) * (q - smooth_ref)])
            J = np.concatenate([J, np.sqrt(sw) * np.eye(n_arm)], axis=0)
        return r, J

    r, J = res_jac(qa)
    cost = float(r @ r)
    lam = lm_damping
    for _ in range(n_iters):
        H = J.T @ J + lam * np.eye(n_arm)
        try:
            dq = np.linalg.solve(H, -J.T @ r)
        except np.linalg.LinAlgError:
            break
        qn = np.clip(qa + dq, lo, hi)
        rn, Jn = res_jac(qn)
        cn = float(rn @ rn)
        if cn < cost:
            qa, r, J, cost = qn, rn, Jn, cn
            lam = max(lam * 0.5, 1e-9)
        else:
            lam = min(lam * 4.0, 1e6)
    full[model.arm_idx] = qa
    resid_mm = float(np.linalg.norm(model.palm_pose(full)[:3, 3] - target) * 1000)
    return qa, resid_mm


__all__ = ["kabsch_rotation", "solve_hand_qpos", "solve_arm_qpos"]
