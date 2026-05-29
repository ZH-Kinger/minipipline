"""Data contracts for Layer 1.5 (Annotate)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class ClipAnnotation:
    """Language description + action label for one contiguous clip."""

    clip_idx: int                    # 0-indexed within the session
    frame_start: int                 # inclusive
    frame_end: int                   # exclusive
    t_start_s: float
    t_end_s: float
    language_text: str               # natural-language description
    action_label: str                # categorical (e.g. "grasp", "place")
    action_score: float              # confidence in [0, 1]
    language_source: str             # backend name (e.g. "qwen2.5-vl-72b" or "mock")
    actions_source: str


@dataclass(frozen=True)
class FrameQualityRow:
    """Per-frame data-cleaning signals."""

    frame_idx: int
    blur_score: float                # higher = sharper; rule-based Laplacian var
    exposure_score: float            # higher = better; mean luminance distance from mid
    hand_visible: bool               # echo of upstream hand validity
    overall_quality: float           # in [0, 1]
    kept: bool                       # True iff this frame survives all filters


@dataclass(frozen=True)
class AnnotateRunResult:
    """Return value of run_annotate_pipeline()."""

    clips: list[ClipAnnotation]
    frame_quality: list[FrameQualityRow]
    stage_timings: dict[str, float] = field(default_factory=dict)
    nir_dir: str = ""
    cache_stats: dict[str, float] = field(default_factory=dict)  # {} when no cache


__all__ = ["ClipAnnotation", "FrameQualityRow", "AnnotateRunResult"]
