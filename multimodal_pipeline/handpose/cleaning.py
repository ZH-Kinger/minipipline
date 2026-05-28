from __future__ import annotations

import numpy as np

from ..config import HandPoseConfig


def clean_pred_result(
    pred_trans: np.ndarray,        # (2, T, 3)
    pred_rot_aa: np.ndarray,       # (2, T, 3)
    pred_hand_pose: np.ndarray,    # (2, T, 45)
    pred_betas: np.ndarray,        # (2, T, 10)
    pred_valid: np.ndarray,        # (2, T) bool
    cfg: HandPoseConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Linear-interpolate over invalid runs and smooth.

    Returns (trans, rot_aa, hand_pose, betas, valid, kept), where:
    - `valid` reflects post-interpolation validity (frames in long invalid
      runs > cfg.max_invalid_run_frames remain invalid)
    - `kept` is the original validity before interpolation
    """
    pred_trans = np.array(pred_trans, copy=True)
    pred_rot_aa = np.array(pred_rot_aa, copy=True)
    pred_hand_pose = np.array(pred_hand_pose, copy=True)
    pred_betas = np.array(pred_betas, copy=True)
    pred_valid = np.array(pred_valid, copy=True)

    kept = pred_valid.copy()
    new_valid = pred_valid.copy()

    for slot in range(pred_valid.shape[0]):
        valid_mask = pred_valid[slot]
        T = valid_mask.shape[0]

        # Identify invalid runs.
        runs = _runs(valid_mask, want=False)
        for run_start, run_end in runs:  # [run_start, run_end)
            run_len = run_end - run_start
            if run_len > cfg.max_invalid_run_frames:
                # Don't fill long gaps; mark as still invalid.
                continue
            # Need anchors on both sides for linear interpolation.
            left_anchor = run_start - 1
            right_anchor = run_end
            if left_anchor < 0 or right_anchor >= T:
                continue  # edge run; leave invalid
            alphas = np.linspace(0, 1, run_len + 2)[1:-1]  # exclude anchors
            for arr in (pred_trans, pred_rot_aa, pred_hand_pose, pred_betas):
                arr[slot, run_start:run_end] = (
                    (1 - alphas[:, None]) * arr[slot, left_anchor][None, :]
                    + alphas[:, None] * arr[slot, right_anchor][None, :]
                ).astype(arr.dtype)
            new_valid[slot, run_start:run_end] = True

        # Light smoothing of translation only (preserve rotation/pose semantics).
        win = max(1, cfg.smoothing_window_frames)
        if win >= 2:
            kernel = np.ones(win, dtype=np.float32) / win
            for dim in range(3):
                pred_trans[slot, :, dim] = _conv_same(pred_trans[slot, :, dim], kernel)

    return pred_trans, pred_rot_aa, pred_hand_pose, pred_betas, new_valid, kept


def _runs(mask: np.ndarray, want: bool) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    i = 0
    n = mask.shape[0]
    while i < n:
        if bool(mask[i]) != want:
            i += 1
            continue
        j = i
        while j < n and bool(mask[j]) == want:
            j += 1
        out.append((i, j))
        i = j
    return out


def _conv_same(arr: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    pad = kernel.shape[0] // 2
    padded = np.pad(arr, (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: arr.shape[0]]


__all__ = ["clean_pred_result"]
