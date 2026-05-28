"""Layer 1.5 (Annotate) algorithm / orchestration parameters.

Backend selection (``mock`` / ``dashscope`` / ``local`` / ``rule_based``) for
each of the 3 components is environment-driven; see ``.env`` and
``multimodal_pipeline.config.settings``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AnnotateConfig:
    """Layer 1.5 configuration."""

    # Mock determinism (shared seed scheme across the 3 components).
    mock_scheme_version: str = "v1"
    mock_seed_salt: str = "multimodal_pipeline.annotate.mock"

    # Clip planning. When Layer 2 atomic actions are provided, they are used
    # as clips; otherwise the orchestrator falls back to single-clip
    # (whole-video) or fixed-stride windows.
    fixed_window_clip_len_s: float = 4.0   # used only when no atomic_actions

    # Frame quality filtering.
    quality_blur_threshold: float = 60.0   # Laplacian variance; lower => blurry
    quality_overall_threshold: float = 0.4 # frames below this get kept=False
    quality_drop_on_invalid_hand: bool = False  # if True, kept = kept and hand_visible

    # Output toggles.
    write_clip_annotations: bool = True
    write_frame_quality: bool = True

    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_file(cls, path: str | Path | None) -> "AnnotateConfig":
        if path is None:
            return cls()
        with Path(path).open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
        return cls.from_mapping(raw)

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "AnnotateConfig":
        allowed = {f.name for f in cls.__dataclass_fields__.values()}
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ValueError(f"Unknown Annotate config key(s): {', '.join(unknown)}")
        return cls(**raw)


__all__ = ["AnnotateConfig"]
