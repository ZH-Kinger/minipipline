from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...config import HandPoseConfig
from ...config.models import MoGe2Hyper
from ..schemas import CameraIntrinsics, DepthBatch, VideoMeta
from ._mock import mock_rng


@dataclass
class MoGe2Backend:
    """Stage 4.2 — monocular depth + disparity (mock implementation).

    Returns small-resolution depth/disparity tensors with plausible value ranges.
    Real MoGe-2 would return per-pixel arrays at input resolution; the mock
    down-samples heavily (``hyper.mock_height`` × ``hyper.mock_width``) to keep
    memory sane.
    """

    cfg: HandPoseConfig
    hyper: MoGe2Hyper

    def infer(
        self,
        video: VideoMeta,
        intrinsics: CameraIntrinsics,
        frame_start: int,
        frame_end: int,
        clip_idx: int = 0,
    ) -> DepthBatch:
        T = frame_end - frame_start
        rng = mock_rng(
            scheme_version=self.cfg.mock_scheme_version,
            salt=self.cfg.mock_seed_salt,
            video_id=video.sha1_short,
            stage="moge2",
            frame_idx=frame_start,
        )
        H, W = self.hyper.mock_height, self.hyper.mock_width
        d_lo, d_hi = self.hyper.mock_disparity_range
        depth_lo, depth_hi = self.hyper.mock_depth_clip_m
        disp = rng.uniform(d_lo, d_hi, size=(T, H, W)).astype(np.float16)
        depth = (
            (1.0 / np.maximum(disp.astype(np.float32), 1e-3))
            .clip(depth_lo, depth_hi)
            .astype(np.float16)
        )
        return DepthBatch(
            video_id=video.video_id,
            clip_idx=clip_idx,
            frame_start=frame_start,
            disparity=disp,
            depth_metric=depth,
        )


__all__ = ["MoGe2Backend"]
