"""Layer 2 — hand-pose feature extraction (GeoCalib + MoGe-2 + HaWoR + MegaSAM).

This layer is currently mocked. All model inference returns deterministic
random tensors with correct shapes. Orchestration, IO, scheduling, cleaning,
clip merging, and action segmentation are real.
"""

from ..config import HandPoseConfig
from .orchestrator import HandPoseRunResult, run_handpose_pipeline
from .schemas import (
    AtomicAction,
    CameraIntrinsics,
    CameraTrajectory,
    ClipPlan,
    DepthBatch,
    EpisodePlan,
    HandsBatchS1,
    LabelSegment,
    MergedPrediction,
    PredResult,
    StageMetrics,
    VideoMeta,
)
from .video_io import probe_video

__all__ = [
    "HandPoseConfig",
    "VideoMeta",
    "LabelSegment",
    "ClipPlan",
    "CameraIntrinsics",
    "DepthBatch",
    "HandsBatchS1",
    "CameraTrajectory",
    "PredResult",
    "MergedPrediction",
    "AtomicAction",
    "EpisodePlan",
    "StageMetrics",
    "HandPoseRunResult",
    "run_handpose_pipeline",
    "probe_video",
]
