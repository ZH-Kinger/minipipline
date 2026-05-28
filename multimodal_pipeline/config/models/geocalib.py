"""Algorithm hyperparameters for the GeoCalib (stage 4.1) backend."""

from __future__ import annotations

from dataclasses import dataclass

from ._base import ModelHyperMixin


@dataclass(frozen=True)
class GeoCalibHyper(ModelHyperMixin):
    """Hyperparameters for GeoCalib camera-intrinsic estimation."""

    # Variant / weights identifier (real backend reads this; mock ignores).
    variant: str = "geocalib-v1"

    # Real backends only.
    batch_size: int = 1
    input_size: tuple[int, int] = (384, 512)  # (H, W)
    fp16: bool = False

    # Mock-only.
    mock_hfov_deg: float = 76.0
    mock_vfov_deg: float = 55.0
    mock_fov_jitter_deg: float = 0.5


__all__ = ["GeoCalibHyper"]
