from __future__ import annotations

from pathlib import Path

import numpy as np

from ..annotate.schemas import ClipAnnotation
from ..handpose.schemas import AtomicAction, MergedPrediction, VideoMeta
from .writer import EpisodeInput


def _invert_se3(cam_c2w: np.ndarray) -> np.ndarray:
    """Invert a batch of 4×4 rigid transforms `(T, 4, 4)`."""
    out = np.broadcast_to(np.eye(4, dtype=np.float32), cam_c2w.shape).copy()
    R = cam_c2w[:, :3, :3]
    t = cam_c2w[:, :3, 3]
    Rt = np.transpose(R, (0, 2, 1))
    out[:, :3, :3] = Rt
    out[:, :3, 3] = -np.einsum("tij,tj->ti", Rt, t)
    return out


def _hand_to_main_type(hand: str) -> int:
    return {"left": 0, "right": 1}.get(hand, -1)


def build_episode_inputs(
    video: VideoMeta,
    merged: MergedPrediction,
    atomic_actions: list[AtomicAction],
    source_video_path: Path,
    clip_annotations: list[ClipAnnotation] | None = None,
) -> list[EpisodeInput]:
    """Slice a `MergedPrediction` into per-`AtomicAction` `EpisodeInput`s ready for the writer.

    When ``clip_annotations`` is provided (from Layer 1.5), each episode's
    ``task_text`` becomes the per-clip natural-language description and the
    per-frame ``action_label`` / ``action_score`` are populated. The mapping
    keys on ``ClipAnnotation.clip_idx == enumerate(atomic_actions)`` order.
    """
    w2c_all = _invert_se3(merged.trajectory.cam_c2w)
    fov = (float(merged.intrinsics.hfov_deg), float(merged.intrinsics.vfov_deg))

    ann_by_idx: dict[int, ClipAnnotation] = {}
    if clip_annotations is not None:
        ann_by_idx = {c.clip_idx: c for c in clip_annotations}

    episodes: list[EpisodeInput] = []
    for ep_idx, action in enumerate(atomic_actions):
        a, b = action.frame_start, action.frame_end
        if b <= a:
            continue
        ann = ann_by_idx.get(ep_idx)
        if ann is not None:
            task_text = ann.language_text or action.parent_label_text
            action_label = ann.action_label
            action_score = ann.action_score
        else:
            task_text = action.parent_label_text
            action_label = ""
            action_score = 0.0
        episodes.append(
            EpisodeInput(
                episode_index=ep_idx,
                task_text=task_text,
                main_type=_hand_to_main_type(action.hand),
                source_video_path=Path(source_video_path),
                source_fps=video.fps,
                frame_start=a,
                frame_end=b,
                pred_trans=merged.pred_trans[:, a:b],
                pred_rot_aa=merged.pred_rot[:, a:b],
                pred_hand_pose_aa=merged.pred_hand_pose[:, a:b],
                pred_betas=merged.pred_betas[:, a:b],
                pred_kept=merged.pred_kept[:, a:b],
                extrinsics_w2c=w2c_all[a:b],
                intrinsics_fov=fov,
                action_label=action_label,
                action_score=action_score,
            )
        )
    return episodes


__all__ = ["build_episode_inputs"]
