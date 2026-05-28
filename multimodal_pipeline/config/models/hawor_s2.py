"""Algorithm hyperparameters for the HaWoR Stage 2 (stage 4.5) backend."""

from __future__ import annotations

from dataclasses import dataclass

from ._base import ModelHyperMixin


@dataclass(frozen=True)
class HaWoRStage2Hyper(ModelHyperMixin):
    """Hyperparameters for HaWoR S2 world-space MANO reconstruction."""

    variant: str = "hawor-s2-v0.3"

    # Real backends only.
    refinement_iters: int = 50
    smoothing_window: int = 7
    batch_size: int = 8
    fp16: bool = True

    # Mock-only.
    mock_baseline_left_xyz: tuple[float, float, float] = (-0.15, 0.05, 0.45)
    mock_baseline_right_xyz: tuple[float, float, float] = (0.15, 0.05, 0.45)
    mock_translation_jitter_m: float = 0.01
    mock_rot_std_rad: float = 0.3
    mock_pose_std_rad: float = 0.15
    mock_betas_std: float = 0.2


__all__ = ["HaWoRStage2Hyper"]
