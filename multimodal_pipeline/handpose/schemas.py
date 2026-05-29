from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np


# Convention across all per-hand arrays: index 0 = left, index 1 = right.
HAND_LEFT = 0
HAND_RIGHT = 1


# ---------------------------------------------------------------------------
# Video + clip planning (stdlib only, JSON-serializable)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VideoMeta:
    video_id: str
    path: Path
    fps: float
    n_frames: int
    width: int
    height: int
    duration_s: float
    sha1_short: str


@dataclass(frozen=True)
class LabelSegment:
    label_id: str
    action_text: str
    t_start_s: float
    t_end_s: float
    hand_hint: Literal["left", "right", "both", "unknown"] = "unknown"
    frame_start: int = -1
    frame_end: int = -1


@dataclass(frozen=True)
class ClipPlan:
    video_id: str
    clip_idx: int
    frame_start: int
    frame_end: int  # exclusive
    t_start_s: float
    t_end_s: float
    overlap_left: int
    overlap_right: int
    label_segments: tuple[LabelSegment, ...] = ()


# ---------------------------------------------------------------------------
# Per-stage tensor contracts (numpy-carrying)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CameraIntrinsics:
    video_id: str
    K: np.ndarray  # (3, 3) float32
    dist: np.ndarray  # (5,) float32
    width: int
    height: int
    vfov_deg: float
    hfov_deg: float


@dataclass(frozen=True)
class DepthBatch:
    video_id: str
    clip_idx: int
    frame_start: int
    disparity: np.ndarray  # (T, H, W) float16
    depth_metric: np.ndarray  # (T, H, W) float16


@dataclass(frozen=True)
class HandsBatchS1:
    """HaWoR Stage 1 output: per-frame detection + tracking IDs."""

    video_id: str
    clip_idx: int
    frame_start: int
    det_2d: np.ndarray  # (T, 2, 4) float32  -- xyxy boxes, slot 0=left, 1=right
    det_score: np.ndarray  # (T, 2) float32
    hand_present: np.ndarray  # (T, 2) bool
    track_id: np.ndarray  # (T, 2) int32


@dataclass(frozen=True)
class CameraTrajectory:
    video_id: str
    clip_idx: int  # -1 means already merged at video level
    frame_start: int
    cam_c2w: np.ndarray  # (T, 4, 4) float32  -- camera-to-world transform
    K: np.ndarray  # (3, 3) float32
    slam_hw: tuple[int, int]  # (H, W) used during tracking
    valid: np.ndarray  # (T,) bool


@dataclass(frozen=True)
class PredResult:
    """HaWoR Stage 2 raw output, in WORLD coordinates."""

    video_id: str
    clip_idx: int
    frame_start: int
    pred_trans: np.ndarray  # (2, T, 3) float32
    pred_rot: np.ndarray  # (2, T, 3) float32 axis-angle
    pred_hand_pose: np.ndarray  # (2, T, 45) float32 axis-angle, MANO 15 joints × 3
    pred_betas: np.ndarray  # (2, T, 10) float32
    pred_valid: np.ndarray  # (2, T) bool

    @property
    def n_frames(self) -> int:
        return self.pred_trans.shape[1]


@dataclass(frozen=True)
class MergedPrediction:
    """Video-level merged + cleaned predictions, in WORLD coordinates."""

    video_id: str
    pred_trans: np.ndarray  # (2, N, 3) float32
    pred_rot: np.ndarray  # (2, N, 3) float32 axis-angle
    pred_hand_pose: np.ndarray  # (2, N, 45) float32 axis-angle
    pred_betas: np.ndarray  # (2, N, 10) float32
    pred_valid: np.ndarray  # (2, N) bool  -- after cleaning, post-interp
    pred_kept: np.ndarray  # (2, N) bool   -- original validity before fill
    trajectory: CameraTrajectory  # full-video, clip_idx = -1
    intrinsics: CameraIntrinsics
    # Real 21-keypoint world-space hand pose, when sourced from the
    # real_ingest backend; (2, N, 21, 3) float32. None for the mock chain
    # (which only fabricates MANO params, not raw keypoints).
    hand_keypoints_world: np.ndarray | None = None

    @property
    def n_frames(self) -> int:
        return self.pred_trans.shape[1]


# ---------------------------------------------------------------------------
# Atomic action segmentation / episode planning
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AtomicAction:
    label_id: str
    atomic_idx: int
    frame_start: int
    frame_end: int  # exclusive
    hand: Literal["left", "right", "both"]
    parent_label_text: str
    source: Literal["label_json", "velocity_min", "fallback"] = "label_json"


@dataclass(frozen=True)
class EpisodePlan:
    episode_id: int
    video_id: str
    atomic_action: AtomicAction
    fps: float
    frame_start: int
    frame_end: int  # exclusive

    @property
    def length(self) -> int:
        return self.frame_end - self.frame_start


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StageMetrics:
    stage: str
    wall_seconds: float
    ok: bool
    error: str | None = None
    extras: dict[str, float] = field(default_factory=dict)
