from __future__ import annotations

from ..config import HandPoseConfig
from .schemas import ClipPlan, LabelSegment, VideoMeta


def plan_clips(
    video: VideoMeta,
    label_segments: list[LabelSegment],
    cfg: HandPoseConfig,
) -> list[ClipPlan]:
    """Split a video into overlapping clips when it exceeds the threshold.

    Short videos return a single clip covering [0, n_frames). Labels are
    distributed to all clips they intersect (duplicated when spanning a boundary).
    """
    if video.n_frames <= 0:
        return []
    fps = video.fps if video.fps > 0 else cfg.target_fps

    if video.duration_s <= cfg.long_video_threshold_s:
        return [
            ClipPlan(
                video_id=video.video_id,
                clip_idx=0,
                frame_start=0,
                frame_end=video.n_frames,
                t_start_s=0.0,
                t_end_s=video.n_frames / fps,
                overlap_left=0,
                overlap_right=0,
                label_segments=tuple(label_segments),
            )
        ]

    clip_len = max(1, int(round(cfg.clip_len_s * fps)))
    overlap = max(0, int(round(cfg.clip_overlap_s * fps)))
    step = max(1, clip_len - overlap)

    plans: list[ClipPlan] = []
    cursor = 0
    clip_idx = 0
    while cursor < video.n_frames:
        end = min(cursor + clip_len, video.n_frames)
        clip_segments = tuple(
            _clip_label(seg, cursor, end, fps) for seg in label_segments if _label_overlaps(seg, cursor, end, fps)
        )
        plans.append(
            ClipPlan(
                video_id=video.video_id,
                clip_idx=clip_idx,
                frame_start=cursor,
                frame_end=end,
                t_start_s=cursor / fps,
                t_end_s=end / fps,
                overlap_left=overlap if clip_idx > 0 else 0,
                overlap_right=overlap if end < video.n_frames else 0,
                label_segments=clip_segments,
            )
        )
        if end == video.n_frames:
            break
        cursor += step
        clip_idx += 1
    return plans


def _label_overlaps(seg: LabelSegment, frame_start: int, frame_end: int, fps: float) -> bool:
    seg_start = seg.frame_start if seg.frame_start >= 0 else int(seg.t_start_s * fps)
    seg_end = seg.frame_end if seg.frame_end >= 0 else int(seg.t_end_s * fps)
    return seg_end > frame_start and seg_start < frame_end


def _clip_label(seg: LabelSegment, frame_start: int, frame_end: int, fps: float) -> LabelSegment:
    seg_start = seg.frame_start if seg.frame_start >= 0 else int(seg.t_start_s * fps)
    seg_end = seg.frame_end if seg.frame_end >= 0 else int(seg.t_end_s * fps)
    clipped_start = max(seg_start, frame_start)
    clipped_end = min(seg_end, frame_end)
    return LabelSegment(
        label_id=seg.label_id,
        action_text=seg.action_text,
        t_start_s=clipped_start / fps,
        t_end_s=clipped_end / fps,
        hand_hint=seg.hand_hint,
        frame_start=clipped_start,
        frame_end=clipped_end,
    )


__all__ = ["plan_clips"]
