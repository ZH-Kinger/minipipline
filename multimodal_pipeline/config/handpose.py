"""Layer 2 (hand-pose feature extraction) algorithm / orchestration parameters.

Backend selection (``mock`` vs ``real``) and weights paths are environment
concerns and live in ``.env``; see ``multimodal_pipeline.config.settings``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


ExecutorMode = Literal["thread", "sequential"]


@dataclass(frozen=True)
class HandPoseConfig:
    """Layer 2 (hand-pose feature extraction) configuration.

    All `mock_*` knobs only apply when the backend dispatch resolves to mock
    implementations (the current default). Model selection happens via
    ``MMPIPE_<MODEL>_BACKEND`` env vars rather than this config.
    """

    # Mock determinism.
    mock_scheme_version: str = "v1"
    mock_seed_salt: str = "multimodal_pipeline.handpose.mock"

    # Orchestration.
    executor_mode: ExecutorMode = "thread"
    max_workers: int = 4
    max_clip_parallel: int = 2

    # Clip planning (long-video splitting).
    clip_len_s: float = 30.0
    clip_overlap_s: float = 2.0
    blend_window_frames: int = 30
    long_video_threshold_s: float = 30.0

    # Frame rate handling.
    target_fps: float = 30.0
    allow_resample: bool = False  # if True, re-encodes VFR to CFR target_fps

    # MANO dims.
    mano_pose_dim: int = 45
    mano_betas_dim: int = 10

    # Cleaning.
    max_invalid_run_frames: int = 5  # invalid runs longer than this dropped
    smoothing_window_frames: int = 5

    # Action segmentation.
    velocity_min_window: int = 5  # frames for local-minima detection
    velocity_min_eps: float = 1e-2
    min_atomic_frames: int = 5
    fallback_split_count: int = 1

    # Output formats.
    save_raw_predictions: bool = True
    save_cleaned_predictions: bool = True

    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_file(cls, path: str | Path | None) -> "HandPoseConfig":
        if path is None:
            return cls()
        with Path(path).open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
        return cls.from_mapping(raw)

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "HandPoseConfig":
        allowed = {f.name for f in cls.__dataclass_fields__.values()}
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ValueError(f"Unknown HandPose config key(s): {', '.join(unknown)}")
        return cls(**raw)


__all__ = ["HandPoseConfig", "ExecutorMode"]
