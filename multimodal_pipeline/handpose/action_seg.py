from __future__ import annotations

import numpy as np

from ..config import HandPoseConfig
from .schemas import AtomicAction, LabelSegment


def _hand_for_label(hint: str) -> str:
    if hint in ("left", "right", "both"):
        return hint
    return "both"


def _smoothed_speed(positions: np.ndarray, win: int) -> np.ndarray:
    """Per-frame 3-D speed (Euclidean delta) with simple moving-average smoothing."""
    if positions.shape[0] < 2:
        return np.zeros(positions.shape[0], dtype=np.float32)
    delta = np.diff(positions, axis=0)
    speed = np.linalg.norm(delta, axis=1).astype(np.float32)
    speed = np.concatenate([speed[:1], speed])  # pad to length T
    if win >= 2:
        kernel = np.ones(win, dtype=np.float32) / win
        padded = np.pad(speed, (win // 2, win // 2), mode="edge")
        speed = np.convolve(padded, kernel, mode="valid")[: positions.shape[0]]
    return speed


def _local_minima(speed: np.ndarray, eps: float) -> list[int]:
    out: list[int] = []
    for i in range(1, speed.shape[0] - 1):
        if speed[i] < speed[i - 1] and speed[i] < speed[i + 1] and speed[i] < eps:
            out.append(i)
    return out


def atomic_split_for_label(
    label: LabelSegment,
    pred_trans: np.ndarray,    # (2, T, 3) world-space wrist translation
    pred_valid: np.ndarray,    # (2, T) bool
    cfg: HandPoseConfig,
    label_index: int,
) -> list[AtomicAction]:
    """Split a label segment into atomic actions by hand-3D velocity minima.

    Falls back to a uniform split (cfg.fallback_split_count) when no minima
    are found within the segment, guaranteeing at least one atomic action.
    """
    hand = _hand_for_label(label.hand_hint)
    seg_start = label.frame_start if label.frame_start >= 0 else 0
    seg_end = label.frame_end if label.frame_end >= 0 else pred_trans.shape[1]
    seg_end = min(seg_end, pred_trans.shape[1])
    seg_start = max(seg_start, 0)
    if seg_end - seg_start < cfg.min_atomic_frames:
        return []

    # Pick velocity series for the primary hand (or average over both for "both").
    if hand == "left":
        speed = _smoothed_speed(pred_trans[0, seg_start:seg_end], cfg.velocity_min_window)
        valid_mask = pred_valid[0, seg_start:seg_end]
    elif hand == "right":
        speed = _smoothed_speed(pred_trans[1, seg_start:seg_end], cfg.velocity_min_window)
        valid_mask = pred_valid[1, seg_start:seg_end]
    else:  # both
        sp_l = _smoothed_speed(pred_trans[0, seg_start:seg_end], cfg.velocity_min_window)
        sp_r = _smoothed_speed(pred_trans[1, seg_start:seg_end], cfg.velocity_min_window)
        speed = (sp_l + sp_r) * 0.5
        valid_mask = pred_valid[0, seg_start:seg_end] | pred_valid[1, seg_start:seg_end]

    if not valid_mask.any():
        return []  # no hand presence at all in this label

    minima_local = _local_minima(speed, cfg.velocity_min_eps)
    minima_global = sorted({m + seg_start for m in minima_local})

    cuts = [seg_start] + minima_global + [seg_end]
    cuts = sorted(set(cuts))

    actions: list[AtomicAction] = []
    atomic_idx = 0
    for i in range(len(cuts) - 1):
        a = cuts[i]
        b = cuts[i + 1]
        if b - a < cfg.min_atomic_frames:
            continue
        actions.append(
            AtomicAction(
                label_id=label.label_id,
                atomic_idx=atomic_idx,
                frame_start=a,
                frame_end=b,
                hand=hand,  # type: ignore[arg-type]
                parent_label_text=label.action_text,
                source="velocity_min" if minima_global else "label_json",
            )
        )
        atomic_idx += 1

    if not actions:
        # Fallback: evenly split into cfg.fallback_split_count pieces.
        n_split = max(1, cfg.fallback_split_count)
        edges = np.linspace(seg_start, seg_end, n_split + 1).astype(int)
        for i in range(n_split):
            a, b = int(edges[i]), int(edges[i + 1])
            if b - a < cfg.min_atomic_frames:
                continue
            actions.append(
                AtomicAction(
                    label_id=label.label_id,
                    atomic_idx=i,
                    frame_start=a,
                    frame_end=b,
                    hand=hand,  # type: ignore[arg-type]
                    parent_label_text=label.action_text,
                    source="fallback",
                )
            )

    return actions


__all__ = ["atomic_split_for_label"]
