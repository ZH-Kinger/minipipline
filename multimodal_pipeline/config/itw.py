"""Layer 1 (ITW ingest) algorithm / orchestration parameters.

Environment-specific values (device, paths, secrets) belong in ``.env``; see
``multimodal_pipeline.config.settings``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ITWConfig:
    """Layer 1 (ITW ingest) configuration.

    Defaults match the on-disk reality of `00010a33-...` sample sessions:
    RGB at 30fps as the reference clock, depth + head 6-DoF 1:1 with RGB,
    IMU at 200Hz potentially longer than the video, audio cropped to RGB window.
    """

    # File names inside a session directory.
    manifest_filename: str = "config.json"
    task_info_filename: str = "task_info.json"
    rgb_video_filename: str = "rgb_head.mp4"
    rgb_timestamps_filename: str = "rgb_head.csv"
    depth_video_filename: str = "depth_head.mkv"
    depth_timestamps_filename: str = "depth_head.csv"
    imu_filename: str = "imu.txt"
    audio_filename: str = "mic.wav"
    hand_keypoints_filename: str = "hands_keypoint_3d.json"
    head_6dof_candidates: tuple[str, ...] = (
        "head_hands_sixdof.csv",
        "head_hands_sixdof2.csv",
    )
    kalibr_filename: str = "kalibr_parameters.yaml"
    head_param_path: str = "camera_params/head_param.json"

    # Pipeline parameters.
    reference_stream: str = "rgb_head"
    nominal_fps: float = 30.0
    gap_anomaly_factor: float = 1.6  # interval > nominal * factor → anomaly
    audio_pad_ms: float = 0.0
    tail_exclude: bool = True

    # Calibration cross-check tolerance.
    intrinsics_match_atol: float = 1e-3

    # NIR output behaviour.
    copy_video: bool = False  # False = reference original (best on Windows: copy fallback)
    shard_max_size_mb: int = 512

    # Misc.
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_file(cls, path: str | Path | None) -> "ITWConfig":
        if path is None:
            return cls()
        with Path(path).open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
        return cls.from_mapping(raw)

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "ITWConfig":
        allowed = {f.name for f in cls.__dataclass_fields__.values()}
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ValueError(f"Unknown ITW config key(s): {', '.join(unknown)}")
        normalized: dict[str, Any] = dict(raw)
        for tuple_key in ("head_6dof_candidates",):
            if tuple_key in normalized and isinstance(normalized[tuple_key], list):
                normalized[tuple_key] = tuple(normalized[tuple_key])
        return cls(**normalized)

    @property
    def shard_max_size_bytes(self) -> int:
        return max(1, self.shard_max_size_mb) * 1024 * 1024


__all__ = ["ITWConfig"]
