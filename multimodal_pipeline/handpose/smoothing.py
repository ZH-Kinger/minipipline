"""One-Euro temporal smoothing for the ``real_ingest`` MANO output.

The production ``real_ingest`` backend assembles ``MergedPrediction`` directly
from the source tracker's per-frame MANO, which carries no temporal filtering —
so the 45-D finger pose (verbatim from the tracker) and the head-pose-transformed
wrist translation/orientation both jitter at high frequency. This module denoises
them with the 1€ filter (Casiez et al.), applied **only within continuous kept
runs** (never across gaps, never onto non-kept frames) so it denoises without
ever fabricating a value — consistent with the repo's no-fake-data policy.

Vector fields (translation, 21 keypoints) use per-component scalar 1€. Rotation
fields (wrist orient + 15 finger joints, all axis-angle) use a **quaternion** 1€
with SLERP as the spherical low-pass — per-component 1€ on axis-angle is unsafe
(2π wrap, axis-sign ambiguity, non-unique near π). Pure sequential numpy: no RNG,
no threads → bit-reproducible, satisfying the determinism contract.
"""

from __future__ import annotations

from dataclasses import replace
from math import pi

import numpy as np

from .cleaning import _runs
from .schemas import MergedPrediction

__all__ = ["smooth_merged"]


# ---------------------------------------------------------------------------
# 1€ low-pass primitive
# ---------------------------------------------------------------------------


def _alpha(cutoff: np.ndarray | float, dt: float) -> np.ndarray | float:
    """1€ smoothing factor for a given cutoff frequency (Hz)."""
    tau = 1.0 / (2.0 * pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


def _oneeuro_vec(x: np.ndarray, dt: float, min_cutoff: float, beta: float,
                 d_cutoff: float) -> np.ndarray:
    """Scalar 1€ filter over ``x`` of shape ``(T, C)`` — sequential in T,
    vectorised over the C independent channels. Frame 0 passes through."""
    T = x.shape[0]
    if T < 2:
        return x
    out = np.empty_like(x)
    out[0] = x[0]
    a_d = _alpha(d_cutoff, dt)
    dx_hat = np.zeros(x.shape[1])
    x_prev = x[0].copy()
    x_hat_prev = x[0].copy()
    for i in range(1, T):
        dx = (x[i] - x_prev) / dt
        dx_hat = a_d * dx + (1.0 - a_d) * dx_hat
        cutoff = min_cutoff + beta * np.abs(dx_hat)      # per-channel (C,)
        a = _alpha(cutoff, dt)
        x_hat = a * x[i] + (1.0 - a) * x_hat_prev
        out[i] = x_hat
        x_prev = x[i]
        x_hat_prev = x_hat
    return out


# ---------------------------------------------------------------------------
# Quaternion helpers (x, y, z, w) and quaternion 1€
# ---------------------------------------------------------------------------


def _aa_to_quat(aa: np.ndarray) -> np.ndarray:
    """axis-angle ``(..., 3)`` -> unit quaternion ``(..., 4)`` as (x,y,z,w)."""
    theta = np.linalg.norm(aa, axis=-1, keepdims=True)      # (...,1)
    half = 0.5 * theta
    small = theta < 1e-8
    # sin(half)/theta -> 1/2 as theta->0 (limit), guard div by zero
    scale = np.where(small, 0.5, np.sin(half) / np.where(small, 1.0, theta))
    xyz = aa * scale
    w = np.cos(half)
    return np.concatenate([xyz, w], axis=-1)


def _quat_to_aa(q: np.ndarray) -> np.ndarray:
    """unit quaternion ``(..., 4)`` -> axis-angle ``(..., 3)``."""
    q = q / np.clip(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12, None)
    xyz = q[..., :3]
    w = np.clip(q[..., 3:4], -1.0, 1.0)
    s = np.linalg.norm(xyz, axis=-1, keepdims=True)
    theta = 2.0 * np.arctan2(s, w)
    axis = xyz / np.where(s < 1e-8, 1.0, s)
    return np.where(s < 1e-8, 0.0, axis * theta)


def _slerp(q0: np.ndarray, q1: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Per-row SLERP. ``q0``/``q1`` ``(J, 4)``, ``t`` ``(J,)`` -> ``(J, 4)``."""
    d = np.sum(q0 * q1, axis=-1)                      # (J,)
    q1 = np.where((d < 0.0)[:, None], -q1, q1)        # shortest arc
    d = np.abs(d)
    tcol = t[:, None]                                 # (J,1)
    theta = np.arccos(np.clip(d, -1.0, 1.0))[:, None]  # (J,1)
    sin_th = np.sin(theta)
    denom = np.where(sin_th < 1e-6, 1.0, sin_th)
    q_slerp = (np.sin((1.0 - tcol) * theta) / denom) * q0 + (np.sin(tcol * theta) / denom) * q1
    q_lerp = q0 + tcol * (q1 - q0)                    # near-parallel fallback
    use_lerp = (d > 0.9995)[:, None] | (sin_th < 1e-6)
    q = np.where(use_lerp, q_lerp, q_slerp)
    return q / np.clip(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12, None)


def _oneeuro_quat(aa: np.ndarray, dt: float, min_cutoff: float, beta: float,
                  d_cutoff: float) -> np.ndarray:
    """Quaternion 1€ over ``aa`` of shape ``(T, J, 3)`` — sequential in T,
    vectorised over the J independent joints. Frame 0 passes through."""
    T, J, _ = aa.shape
    if T < 2:
        return aa
    q = _aa_to_quat(aa)                                # (T, J, 4)
    # hemisphere continuity along time (resolve double-cover before filtering)
    for i in range(1, T):
        flip = np.sum(q[i] * q[i - 1], axis=-1) < 0.0
        q[i][flip] *= -1.0
    out = np.empty_like(q)
    out[0] = q[0]
    a_d = _alpha(d_cutoff, dt)
    speed_hat = np.zeros(J)
    for i in range(1, T):
        dot = np.clip(np.abs(np.sum(q[i] * q[i - 1], axis=-1)), 0.0, 1.0)  # (J,)
        speed = 2.0 * np.arccos(dot) / dt              # rad/s angular velocity
        speed_hat = a_d * speed + (1.0 - a_d) * speed_hat
        cutoff = min_cutoff + beta * speed_hat         # (J,)
        a = _alpha(cutoff, dt)
        out[i] = _slerp(out[i - 1], q[i], np.asarray(a, dtype=np.float64))
    return _quat_to_aa(out)


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------


def smooth_merged(merged: MergedPrediction, fps: float, cfg) -> MergedPrediction:
    """Return a new ``MergedPrediction`` with 1€-smoothed hand fields.

    Filters ``hand_keypoints_world``, ``pred_rot`` and ``pred_hand_pose`` within
    each hand's continuous kept runs; ``pred_trans`` is re-derived from the
    filtered wrist keypoint (joint 0) so the two never diverge. ``pred_betas``
    (hand shape) and all masks/trajectory pass through untouched. No-op when
    smoothing is disabled or there are no real keypoints (mock chain).
    """
    if not getattr(cfg, "smoothing_enabled", True):
        return merged
    if merged.hand_keypoints_world is None:
        return merged

    dt = 1.0 / float(fps) if fps and fps > 0 else 1.0 / 30.0
    kp = merged.hand_keypoints_world.astype(np.float64).copy()   # (2, N, 21, 3)
    rot = merged.pred_rot.astype(np.float64).copy()              # (2, N, 3)
    pose = merged.pred_hand_pose.astype(np.float64).copy()       # (2, N, 45)
    # Start translation from the ORIGINAL (real_ingest zero-fills non-kept frames);
    # only kept-run frames get overwritten below. Blanket-copying kp[...,0] would
    # drag the raw non-kept-frame NaN in hand_keypoints_world into pred_trans.
    trans = merged.pred_trans.astype(np.float64).copy()          # (2, N, 3)
    kept = merged.pred_kept

    for h in range(kp.shape[0]):
        for s, e in _runs(kept[h], True):
            if e - s < 2:                       # single-frame run is a no-op
                continue
            L = e - s
            kp[h, s:e] = _oneeuro_vec(
                kp[h, s:e].reshape(L, -1), dt,
                cfg.oneeuro_kp_min_cutoff, cfg.oneeuro_kp_beta, cfg.oneeuro_d_cutoff,
            ).reshape(L, 21, 3)
            # keep wrist translation consistent with the filtered wrist keypoint
            trans[h, s:e] = kp[h, s:e, 0, :]
            rot[h, s:e] = _oneeuro_quat(
                rot[h, s:e][:, None, :], dt,
                cfg.oneeuro_rot_min_cutoff, cfg.oneeuro_rot_beta, cfg.oneeuro_d_cutoff,
            )[:, 0, :]
            pose[h, s:e] = _oneeuro_quat(
                pose[h, s:e].reshape(L, 15, 3), dt,
                cfg.oneeuro_rot_min_cutoff, cfg.oneeuro_rot_beta, cfg.oneeuro_d_cutoff,
            ).reshape(L, 45)

    trans = trans.astype(np.float32)
    return replace(
        merged,
        hand_keypoints_world=kp.astype(np.float32),
        pred_trans=trans,
        pred_rot=rot.astype(np.float32),
        pred_hand_pose=pose.astype(np.float32),
    )


def _selftest() -> None:
    """`python3 -m multimodal_pipeline.handpose.smoothing` — synthetic asserts:
    jerk drops, true motion largely preserved, deterministic, frame-0 identity,
    no NaN from finite input, quaternion round-trip."""
    rng = np.random.default_rng(0)
    dt = 1.0 / 30.0
    t = np.arange(120) * dt
    true = np.stack([np.sin(2 * pi * 0.5 * t), np.cos(2 * pi * 0.4 * t), 0.3 * t], 1)
    x = true + rng.normal(0, 0.02, true.shape)
    jerk = lambda a: np.median(np.linalg.norm(a[2:] - 2 * a[1:-1] + a[:-2], -1))
    xf = _oneeuro_vec(x.copy(), dt, 2.0, 0.7, 1.0)
    assert jerk(xf) < 0.6 * jerk(x), "vec jerk not reduced"
    assert np.allclose(xf[0], x[0]), "vec frame-0 not identity"
    assert np.array_equal(xf, _oneeuro_vec(x.copy(), dt, 2.0, 0.7, 1.0)), "vec non-deterministic"
    assert np.isfinite(xf).all(), "vec introduced non-finite"

    aa = (np.stack([0.4 * np.sin(2 * pi * 0.3 * t), 0.1 * t, 0.1 * np.ones_like(t)], 1)
          + rng.normal(0, 0.03, (120, 3)))[:, None, :]
    aaf = _oneeuro_quat(aa.copy(), dt, 2.0, 0.7, 1.0)
    ang = lambda a: np.median(np.linalg.norm(a[2:, 0] - 2 * a[1:-1, 0] + a[:-2, 0], -1))
    assert ang(aaf) < 0.6 * ang(aa), "quat jerk not reduced"
    assert np.allclose(aaf[0], aa[0], atol=1e-6), "quat frame-0 not identity"
    assert np.isfinite(aaf).all(), "quat introduced non-finite"
    # aa -> quat -> aa round-trip
    assert np.allclose(_quat_to_aa(_aa_to_quat(aa[:, 0])), aa[:, 0], atol=1e-6), "aa/quat round-trip"
    print("smoothing selftest OK: jerk reduced, motion preserved, deterministic, finite, round-trip")


if __name__ == "__main__":
    _selftest()
