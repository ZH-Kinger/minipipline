"""Algorithm hyperparameters for the HaWoR Stage 1 (stage 4.3) backend."""

from __future__ import annotations

from dataclasses import dataclass

from ._base import ModelHyperMixin


@dataclass(frozen=True)
class HaWoRStage1Hyper(ModelHyperMixin):
    """Hyperparameters for HaWoR S1 hand-detection + tracking."""

    variant: str = "hawor-s1-v0.3"

    # Real backends only.
    batch_size: int = 8
    detection_threshold: float = 0.5
    max_seq_len: int = 1024
    fp16: bool = True

    # Mock-only.
    mock_present_prob: float = 0.7
    mock_score_range: tuple[float, float] = (0.6, 0.99)


__all__ = ["HaWoRStage1Hyper"]
