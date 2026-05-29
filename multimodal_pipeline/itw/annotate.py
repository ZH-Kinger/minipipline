from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .align import AlignedData
from ..config import ITWConfig
from .schemas import (
    AnnotationSummary,
    DiscoveryResult,
    TaskInfo,
)


@dataclass(frozen=True)
class HandPerFrame:
    """Per-frame hand annotation row (for hand_keypoints.parquet)."""

    frame_index: int
    rgb_timestamp_s: float
    left_present: bool
    right_present: bool
    excluded: bool
    exclude_reason: str
    left_confidence: str  # "high" / "low" / ""
    right_confidence: str
    left_wrist_xyz: tuple[float, float, float] | None
    right_wrist_xyz: tuple[float, float, float] | None
    left_keypoints_3d: list[tuple[float, float, float]] | None
    right_keypoints_3d: list[tuple[float, float, float]] | None


@dataclass(frozen=True)
class AnnotatedData:
    """In-memory annotation output for the normalize stage."""

    task_text: str
    task: TaskInfo
    hand_frames: tuple[HandPerFrame, ...]
    is_gap_anomaly_per_frame: np.ndarray  # (N,) bool
    summary: AnnotationSummary


# Canonical order of MANO-style 21 hand keypoints from the producer.
# Names match the actual `keypoints_3d_cam_m` keys emitted by the depth-fusion
# hand tracker: thumb_cmc/mcp/ip, {index,middle,ring,pinky}_mcp/pip/dip, *_tip.
# (The earlier thumb_1/index_1/little_* naming never matched any real key, so
# _extract_keypoints rejected every hand and all coords were written as nan.)
_KP_ORDER = (
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
)


def _extract_keypoints(hand_entry: dict) -> list[tuple[float, float, float]] | None:
    kp = hand_entry.get("keypoints_3d_cam_m")
    if not isinstance(kp, dict):
        return None
    out: list[tuple[float, float, float]] = []
    for name in _KP_ORDER:
        coord = kp.get(name)
        if not isinstance(coord, (list, tuple)) or len(coord) < 3:
            return None  # missing a canonical joint → reject this hand
        out.append((float(coord[0]), float(coord[1]), float(coord[2])))
    return out


def annotate(
    discovery: DiscoveryResult,
    aligned: AlignedData,
    cfg: ITWConfig,
) -> AnnotatedData:
    hand_path = discovery.found_files.get("hands_keypoint_3d")
    rgb_n = aligned.rgb_timestamps_s.shape[0]

    if hand_path is None:
        # No hand keypoints — return empty annotation but don't crash.
        empty_frames = tuple(
            HandPerFrame(
                frame_index=i,
                rgb_timestamp_s=float(aligned.rgb_timestamps_s[i]),
                left_present=False,
                right_present=False,
                excluded=True,
                exclude_reason="no_hand_file",
                left_confidence="",
                right_confidence="",
                left_wrist_xyz=None,
                right_wrist_xyz=None,
                left_keypoints_3d=None,
                right_keypoints_3d=None,
            )
            for i in range(rgb_n)
        )
        gap_mask = np.zeros(rgb_n, dtype=bool)
        for g in aligned.summary.gap_anomalies:
            if 0 <= g.frame_index < rgb_n:
                gap_mask[g.frame_index] = True
        summary = AnnotationSummary(
            left_present_frames=0,
            right_present_frames=0,
            both_present_frames=0,
            tail_excluded_frames=0,
            any_excluded_frames=rgb_n,
            first_left_frame=None,
            first_right_frame=None,
            last_valid_frame=None,
        )
        return AnnotatedData(
            task_text=discovery.task_info.name,
            task=discovery.task_info,
            hand_frames=empty_frames,
            is_gap_anomaly_per_frame=gap_mask,
            summary=summary,
        )

    with hand_path.open("r", encoding="utf-8") as fh:
        hand_raw = json.load(fh)

    frames_dict = hand_raw.get("frames", {})

    # Build a per-frame entry by looking up the rgb timestamp string.
    # rgb_head.csv writes timestamps with 6-decimal precision, and hand json
    # uses the same exact strings as keys; we format to 6 decimals to match.
    hand_frames: list[HandPerFrame] = []
    left_count = 0
    right_count = 0
    both_count = 0
    tail_count = 0
    any_excluded = 0
    first_left: int | None = None
    first_right: int | None = None
    last_valid: int | None = None

    for i in range(rgb_n):
        ts = float(aligned.rgb_timestamps_s[i])
        ts_key = f"{ts:.6f}"
        entry = frames_dict.get(ts_key)
        excluded = False
        exclude_reason = ""
        left_present = False
        right_present = False
        left_conf = ""
        right_conf = ""
        left_wrist: tuple[float, float, float] | None = None
        right_wrist: tuple[float, float, float] | None = None
        left_kpts: list[tuple[float, float, float]] | None = None
        right_kpts: list[tuple[float, float, float]] | None = None

        if entry is None:
            excluded = True
            exclude_reason = "no_keypoint_entry"
        else:
            excluded = bool(entry.get("excluded", False))
            exclude_reason = str(entry.get("exclude_reason") or "")
            hands = entry.get("hands", []) or []
            for hand_entry in hands:
                is_right = bool(hand_entry.get("is_right", False))
                conf = str(hand_entry.get("confidence", ""))
                kpts = _extract_keypoints(hand_entry)
                wrist_xyz = tuple(kpts[0]) if kpts else None
                if is_right:
                    right_present = True
                    right_conf = conf
                    right_wrist = wrist_xyz
                    right_kpts = kpts
                else:
                    left_present = True
                    left_conf = conf
                    left_wrist = wrist_xyz
                    left_kpts = kpts

        # When cfg.tail_exclude is False, ignore tail exclusion.
        if not cfg.tail_exclude and exclude_reason == "tail":
            excluded = False
            exclude_reason = ""

        hand_frames.append(
            HandPerFrame(
                frame_index=i,
                rgb_timestamp_s=ts,
                left_present=left_present and not excluded,
                right_present=right_present and not excluded,
                excluded=excluded,
                exclude_reason=exclude_reason,
                left_confidence=left_conf,
                right_confidence=right_conf,
                left_wrist_xyz=left_wrist,
                right_wrist_xyz=right_wrist,
                left_keypoints_3d=left_kpts,
                right_keypoints_3d=right_kpts,
            )
        )

        if excluded:
            any_excluded += 1
            if exclude_reason == "tail":
                tail_count += 1
        else:
            if left_present:
                left_count += 1
                if first_left is None:
                    first_left = i
            if right_present:
                right_count += 1
                if first_right is None:
                    first_right = i
            if left_present and right_present:
                both_count += 1
            if left_present or right_present:
                last_valid = i

    gap_mask = np.zeros(rgb_n, dtype=bool)
    for g in aligned.summary.gap_anomalies:
        if 0 <= g.frame_index < rgb_n:
            gap_mask[g.frame_index] = True

    summary = AnnotationSummary(
        left_present_frames=left_count,
        right_present_frames=right_count,
        both_present_frames=both_count,
        tail_excluded_frames=tail_count,
        any_excluded_frames=any_excluded,
        first_left_frame=first_left,
        first_right_frame=first_right,
        last_valid_frame=last_valid,
    )

    return AnnotatedData(
        task_text=discovery.task_info.name,
        task=discovery.task_info,
        hand_frames=tuple(hand_frames),
        is_gap_anomaly_per_frame=gap_mask,
        summary=summary,
    )


__all__ = ["AnnotatedData", "HandPerFrame", "annotate"]
