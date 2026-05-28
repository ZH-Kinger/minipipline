from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...config import HandPoseConfig
from ...config.models import HaWoRStage1Hyper
from ..schemas import CameraIntrinsics, HandsBatchS1, VideoMeta
from ._mock import mock_rng


@dataclass
class HaWoRStage1Backend:
    """Stage 4.3 — 2-D hand detection + cross-frame tracking IDs (mock).

    ``hyper.mock_present_prob`` controls hand-present rate; per-frame detection
    box is centered near image center with random jitter; confidence is sampled
    from ``hyper.mock_score_range``.
    """

    cfg: HandPoseConfig
    hyper: HaWoRStage1Hyper

    def infer(
        self,
        video: VideoMeta,
        intrinsics: CameraIntrinsics,
        frame_start: int,
        frame_end: int,
        clip_idx: int = 0,
    ) -> HandsBatchS1:
        T = frame_end - frame_start
        det_2d = np.zeros((T, 2, 4), dtype=np.float32)
        det_score = np.zeros((T, 2), dtype=np.float32)
        hand_present = np.zeros((T, 2), dtype=bool)
        track_id = np.full((T, 2), -1, dtype=np.int32)

        score_lo, score_hi = self.hyper.mock_score_range

        for slot in (0, 1):
            for i in range(T):
                rng = mock_rng(
                    scheme_version=self.cfg.mock_scheme_version,
                    salt=self.cfg.mock_seed_salt,
                    video_id=video.sha1_short,
                    stage="hawor_s1",
                    frame_idx=frame_start + i,
                    slot=slot,
                )
                if rng.random() < self.hyper.mock_present_prob:
                    hand_present[i, slot] = True
                    cx = video.width * 0.5 + rng.normal(0.0, video.width * 0.1)
                    cy = video.height * 0.5 + rng.normal(0.0, video.height * 0.1)
                    box_w = max(40.0, video.width * 0.12 + rng.normal(0.0, 8.0))
                    box_h = max(40.0, video.height * 0.16 + rng.normal(0.0, 8.0))
                    det_2d[i, slot] = [
                        cx - box_w / 2,
                        cy - box_h / 2,
                        cx + box_w / 2,
                        cy + box_h / 2,
                    ]
                    det_score[i, slot] = float(rng.uniform(score_lo, score_hi))
                    track_id[i, slot] = slot
        return HandsBatchS1(
            video_id=video.video_id,
            clip_idx=clip_idx,
            frame_start=frame_start,
            det_2d=det_2d,
            det_score=det_score,
            hand_present=hand_present,
            track_id=track_id,
        )


__all__ = ["HaWoRStage1Backend"]
