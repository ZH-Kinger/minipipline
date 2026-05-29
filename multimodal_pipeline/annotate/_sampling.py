"""Frame index selection for the VLM annotator.

Uniform sampling treats every part of a clip as equally informative — but
the action-defining moments (hand closing on object, lift-off, release)
typically have peaks in wrist velocity. Sampling those moments lets the
VLM see the discriminating instants instead of slow approach / settle frames.

Falls back to uniform sampling when motion signal is unavailable, the clip
is too short, or all frames are below a "moving" threshold.
"""

from __future__ import annotations

import numpy as np


def uniform_indices(fs: int, fe: int, n: int) -> list[int]:
    """N indices uniformly spread over [fs, fe). Always non-empty when fe>fs."""
    if fe <= fs:
        return []
    n = max(1, min(n, fe - fs))
    return [int(round(fs + (fe - fs - 1) * i / max(1, n - 1))) for i in range(n)]


def pick_motion_peak_indices(
    wrist_speed: np.ndarray,
    fs: int,
    fe: int,
    n: int,
    *,
    static_threshold: float = 1e-3,
) -> list[int]:
    """Pick ``n`` frame indices in ``[fs, fe)`` biased toward velocity peaks.

    Algorithm:
      1. Slice ``wrist_speed`` to the clip window.
      2. Take the top-``2n`` highest-velocity frames.
      3. NMS with minimum spacing ``(fe - fs) / n / 2`` to spread them out.
      4. Sort in time order; pad with uniform fillers if NMS culled too many.

    Fallback to uniform sampling when:
      - clip is shorter than ``2 * n`` frames (too few candidates)
      - the whole clip's max velocity is below ``static_threshold``
        (everything is static; uniform is at least unbiased)
    """
    if fe <= fs or n <= 0:
        return []
    window = fe - fs
    if window < 2 * n or wrist_speed is None or wrist_speed.size == 0:
        return uniform_indices(fs, fe, n)

    # Slice into clip window, clamp to available length.
    end = min(fe, fs + wrist_speed.shape[0]) if wrist_speed.shape[0] < fe else fe
    if end - fs < 2 * n:
        return uniform_indices(fs, fe, n)

    seg = wrist_speed[fs:end].astype(np.float32, copy=False)
    if float(seg.max()) < static_threshold:
        return uniform_indices(fs, fe, n)

    # Top-2n candidates.
    k = min(2 * n, seg.shape[0])
    cand_local = np.argpartition(-seg, k - 1)[:k]
    cand_local = cand_local[np.argsort(-seg[cand_local])]  # sort by speed desc

    # NMS: keep highest-speed first; reject anything within `min_gap` frames
    # of an already-kept index. Goal: n well-spread peaks.
    min_gap = max(1, (end - fs) // (n * 2))
    kept_local: list[int] = []
    for idx in cand_local.tolist():
        if all(abs(idx - k_) >= min_gap for k_ in kept_local):
            kept_local.append(idx)
            if len(kept_local) >= n:
                break

    # If NMS culled too aggressively, fill with uniform fillers (skipping
    # frames too close to already-kept ones).
    if len(kept_local) < n:
        for u in uniform_indices(0, end - fs, n):
            if all(abs(u - k_) >= min_gap for k_ in kept_local):
                kept_local.append(u)
                if len(kept_local) >= n:
                    break

    kept_global = sorted(fs + i for i in kept_local[:n])
    return kept_global


def double_hand_max_speed(pred_trans: np.ndarray) -> np.ndarray:
    """Per-frame max(|Δ left|, |Δ right|) wrist speed in 3D space.

    Input: ``pred_trans`` shape (2, N, 3) from MergedPrediction.
    Output: shape (N,) float32; first frame is duplicated so length matches N.
    """
    if pred_trans is None or pred_trans.ndim != 3 or pred_trans.shape[1] < 2:
        return np.zeros(pred_trans.shape[1] if pred_trans is not None else 0, dtype=np.float32)
    # delta shape: (2, N-1, 3)
    delta = np.diff(pred_trans, axis=1)
    speed = np.linalg.norm(delta, axis=-1)  # (2, N-1)
    per_frame = speed.max(axis=0)            # (N-1,)
    return np.concatenate([per_frame[:1], per_frame]).astype(np.float32)


__all__ = ["uniform_indices", "pick_motion_peak_indices", "double_hand_max_speed"]
