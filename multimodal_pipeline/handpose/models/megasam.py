from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...config import HandPoseConfig
from ...config.models import MegaSamHyper
from ..schemas import CameraIntrinsics, CameraTrajectory, DepthBatch, HandsBatchS1, VideoMeta
from ._mock import mock_rng


@dataclass
class MegaSAMBackend:
    """Stage 4.4 — camera trajectory tracking (MegaSAM stand-in, mock).

    Translation drifts at ``hyper.mock_translation_step_m`` m/frame, rotation
    at ``hyper.mock_rotation_step_rad`` rad/frame (random walk).
    """

    cfg: HandPoseConfig
    hyper: MegaSamHyper

    def infer(
        self,
        video: VideoMeta,
        intrinsics: CameraIntrinsics,
        depth: DepthBatch,
        hands_s1: HandsBatchS1,
        frame_start: int,
        frame_end: int,
        clip_idx: int = 0,
    ) -> CameraTrajectory:
        T = frame_end - frame_start
        rng = mock_rng(
            scheme_version=self.cfg.mock_scheme_version,
            salt=self.cfg.mock_seed_salt,
            video_id=video.sha1_short,
            stage="megasam",
            frame_idx=frame_start,
        )
        trans_step = self.hyper.mock_translation_step_m
        rot_step = self.hyper.mock_rotation_step_rad
        delta_trans = rng.normal(0.0, trans_step, size=(T, 3)).astype(np.float32)
        delta_rot_aa = rng.normal(0.0, rot_step, size=(T, 3)).astype(np.float32)
        cum_trans = np.cumsum(delta_trans, axis=0)

        cam_c2w = np.broadcast_to(np.eye(4, dtype=np.float32), (T, 4, 4)).copy()
        cam_c2w[:, :3, 3] = cum_trans
        for i in range(T):
            v = delta_rot_aa[i]
            theta = float(np.linalg.norm(v))
            if theta < 1e-8:
                continue
            k = v / theta
            K = np.array(
                [[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]],
                dtype=np.float32,
            )
            R = np.eye(3, dtype=np.float32) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)
            cam_c2w[i, :3, :3] = R

        valid = np.ones(T, dtype=bool)
        return CameraTrajectory(
            video_id=video.video_id,
            clip_idx=clip_idx,
            frame_start=frame_start,
            cam_c2w=cam_c2w,
            K=intrinsics.K.astype(np.float32),
            slam_hw=(video.height, video.width),
            valid=valid,
        )


__all__ = ["MegaSAMBackend"]
