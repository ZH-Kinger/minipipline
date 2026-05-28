from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...config import HandPoseConfig
from ...config.models import HaWoRStage2Hyper
from ..schemas import CameraTrajectory, HandsBatchS1, PredResult, VideoMeta
from ._mock import mock_rng


@dataclass
class HaWoRStage2Backend:
    """Stage 4.5 — world-space MANO reconstruction (mock).

    Per-frame translation drifts slowly around a hand-region center (slot 0 left,
    slot 1 right); rotation/pose/betas are random gaussians with plausible
    scale (controlled by ``hyper.mock_*``). Validity follows the upstream
    ``HandsBatchS1.hand_present`` mask.
    """

    cfg: HandPoseConfig
    hyper: HaWoRStage2Hyper

    def infer(
        self,
        video: VideoMeta,
        hands_s1: HandsBatchS1,
        trajectory: CameraTrajectory,
        frame_start: int,
        frame_end: int,
        clip_idx: int = 0,
    ) -> PredResult:
        T = frame_end - frame_start
        pred_trans = np.zeros((2, T, 3), dtype=np.float32)
        pred_rot = np.zeros((2, T, 3), dtype=np.float32)
        pred_hand_pose = np.zeros((2, T, self.cfg.mano_pose_dim), dtype=np.float32)
        pred_betas = np.zeros((2, T, self.cfg.mano_betas_dim), dtype=np.float32)
        pred_valid = hands_s1.hand_present.T.copy()  # (2, T) bool

        baselines = (
            np.array(self.hyper.mock_baseline_left_xyz, dtype=np.float32),
            np.array(self.hyper.mock_baseline_right_xyz, dtype=np.float32),
        )

        for slot in (0, 1):
            base_rng = mock_rng(
                scheme_version=self.cfg.mock_scheme_version,
                salt=self.cfg.mock_seed_salt,
                video_id=video.sha1_short,
                stage="hawor_s2",
                slot=slot,
            )
            betas_base = base_rng.normal(0.0, self.hyper.mock_betas_std, size=self.cfg.mano_betas_dim).astype(np.float32)
            for i in range(T):
                rng = mock_rng(
                    scheme_version=self.cfg.mock_scheme_version,
                    salt=self.cfg.mock_seed_salt,
                    video_id=video.sha1_short,
                    stage="hawor_s2",
                    frame_idx=frame_start + i,
                    slot=slot,
                )
                jitter = rng.normal(0.0, self.hyper.mock_translation_jitter_m, size=3).astype(np.float32)
                pred_trans[slot, i] = baselines[slot] + jitter
                pred_rot[slot, i] = rng.normal(0.0, self.hyper.mock_rot_std_rad, size=3).astype(np.float32)
                pred_hand_pose[slot, i] = rng.normal(
                    0.0, self.hyper.mock_pose_std_rad, size=self.cfg.mano_pose_dim
                ).astype(np.float32)
                pred_betas[slot, i] = betas_base

        return PredResult(
            video_id=video.video_id,
            clip_idx=clip_idx,
            frame_start=frame_start,
            pred_trans=pred_trans,
            pred_rot=pred_rot,
            pred_hand_pose=pred_hand_pose,
            pred_betas=pred_betas,
            pred_valid=pred_valid,
        )


__all__ = ["HaWoRStage2Backend"]
