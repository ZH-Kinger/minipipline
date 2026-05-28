"""Algorithm hyperparameters for the frame quality scorer."""

from __future__ import annotations

from dataclasses import dataclass

from ._base import ModelHyperMixin


@dataclass(frozen=True)
class QualityHyper(ModelHyperMixin):
    """Hyperparameters for rule-based + mock frame quality scoring."""

    # Sampling: don't score every frame for cost (rule-based backend).
    sample_every_n_frames: int = 1

    # Real (rule-based) backend thresholds.
    blur_min_laplacian_var: float = 60.0   # below = blurry → low blur_score
    exposure_target_mean: float = 128.0    # 0-255 grayscale
    exposure_max_dev: float = 80.0         # |mean - target| > this → low score

    # Mock backend.
    mock_blur_range: tuple[float, float] = (40.0, 300.0)
    mock_exposure_range: tuple[float, float] = (0.3, 1.0)
    mock_overall_quality_floor: float = 0.5  # mock keeps most frames "good"


__all__ = ["QualityHyper"]
