"""Algorithm hyperparameters for the MegaSAM (stage 4.4) backend."""

from __future__ import annotations

from dataclasses import dataclass

from ._base import ModelHyperMixin


@dataclass(frozen=True)
class MegaSamHyper(ModelHyperMixin):
    """Hyperparameters for MegaSAM / DROID-SLAM-style camera tracking."""

    variant: str = "megasam-base"

    # Real backends only.
    keyframe_stride: int = 5
    optimizer_iters: int = 12
    max_frames: int = 4096
    fp16: bool = True

    # Mock-only.
    mock_translation_step_m: float = 0.005
    mock_rotation_step_rad: float = 0.009


__all__ = ["MegaSamHyper"]
