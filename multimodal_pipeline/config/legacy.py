"""Legacy sensor-ETL pipeline configuration (pre-3-layer architecture).

Retained because the legacy ``run`` / ``init-config`` CLI commands still
ingest sensor files from the older code paths in
``multimodal_pipeline/{records,pipeline,stages}.py``. New code should consume
``ITWConfig`` / ``HandPoseConfig`` / ``LeRobotConfig`` instead.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_MODALITY_EXTENSIONS = {
    "rgb": [".jpg", ".jpeg", ".png", ".bmp", ".webp"],
    "depth": [".npy", ".npz", ".pfm", ".exr", ".tiff", ".tif"],
    "audio": [".wav", ".flac", ".mp3", ".aac"],
    "imu": [".json", ".jsonl", ".csv"],
    "pose": [".json", ".jsonl", ".csv", ".txt"],
    "pointcloud": [".pcd", ".ply", ".bin", ".las", ".laz"],
}


@dataclass(frozen=True)
class PipelineConfig:
    session_id: str = "default"
    manifest_file: str | None = None
    reference_stream: str = "rgb"
    expected_streams: list[str] = field(default_factory=list)
    alignment_tolerance_ms: float = 50.0
    annotation_confidence_threshold: float = 0.85
    shard_max_size_mb: int = 512
    timestamp_regex: str = r"(?P<timestamp>\d{10,})"
    modality_extensions: dict[str, list[str]] = field(
        default_factory=lambda: dict(DEFAULT_MODALITY_EXTENSIONS)
    )
    ignore_dirs: list[str] = field(
        default_factory=lambda: [".git", "__pycache__", ".venv", "venv", "processed"]
    )

    @classmethod
    def from_file(cls, path: str | Path | None) -> "PipelineConfig":
        if path is None:
            return cls()
        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
        return cls.from_mapping(raw)

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "PipelineConfig":
        allowed = {field.name for field in cls.__dataclass_fields__.values()}
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ValueError(f"Unknown config key(s): {', '.join(unknown)}")
        return cls(**raw)

    @property
    def shard_max_size_bytes(self) -> int:
        return max(1, self.shard_max_size_mb) * 1024 * 1024


__all__ = ["PipelineConfig", "DEFAULT_MODALITY_EXTENSIONS"]
