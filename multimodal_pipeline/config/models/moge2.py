"""Algorithm hyperparameters for the MoGe-2 (stage 4.2) backend."""

from __future__ import annotations

from dataclasses import dataclass

from ._base import ModelHyperMixin


@dataclass(frozen=True)
class MoGe2Hyper(ModelHyperMixin):
    """Hyperparameters for MoGe-2 monocular depth / disparity estimation."""

    variant: str = "moge2-base"

    # Real backends only.
    batch_size: int = 4
    max_image_size: int = 1024  # longest edge for resize before inference
    depth_clip_m: tuple[float, float] = (0.3, 5.0)
    fp16: bool = True

    # Mock-only.
    mock_height: int = 24
    mock_width: int = 32
    mock_disparity_range: tuple[float, float] = (0.1, 1.0)
    mock_depth_clip_m: tuple[float, float] = (0.3, 1.5)


__all__ = ["MoGe2Hyper"]
