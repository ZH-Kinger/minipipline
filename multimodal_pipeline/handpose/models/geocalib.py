from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ...config import HandPoseConfig
from ...config.models import GeoCalibHyper
from ..schemas import CameraIntrinsics, VideoMeta
from ._mock import mock_rng


@dataclass
class GeoCalibBackend:
    """Stage 4.1 — camera-intrinsic estimation (mock implementation).

    Returns intrinsics derived from the video's resolution (with the FOV sampled
    around the values in ``hyper.mock_*``), so downstream stages see plausible K.
    """

    cfg: HandPoseConfig
    hyper: GeoCalibHyper

    def infer(self, video: VideoMeta) -> CameraIntrinsics:
        rng = mock_rng(
            scheme_version=self.cfg.mock_scheme_version,
            salt=self.cfg.mock_seed_salt,
            video_id=video.sha1_short,
            stage="geocalib",
        )
        jitter = self.hyper.mock_fov_jitter_deg
        hfov_deg = float(self.hyper.mock_hfov_deg + rng.normal(0.0, jitter))
        vfov_deg = float(self.hyper.mock_vfov_deg + rng.normal(0.0, jitter))
        hfov_rad = math.radians(hfov_deg)
        vfov_rad = math.radians(vfov_deg)
        fx = (video.width / 2.0) / math.tan(hfov_rad / 2.0)
        fy = (video.height / 2.0) / math.tan(vfov_rad / 2.0)
        cx = video.width / 2.0
        cy = video.height / 2.0
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
        dist = np.zeros(5, dtype=np.float32)
        return CameraIntrinsics(
            video_id=video.video_id,
            K=K,
            dist=dist,
            width=video.width,
            height=video.height,
            vfov_deg=vfov_deg,
            hfov_deg=hfov_deg,
        )


__all__ = ["GeoCalibBackend"]
