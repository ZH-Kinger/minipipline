"""Layer 3 (LeRobot v3 dataset packing) algorithm / orchestration parameters."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

VIDEO_KEY_DEFAULT = "observation.images.ego"


@dataclass(frozen=True)
class LeRobotConfig:
    """Layer 3 (LeRobot v3 dataset packing) configuration."""

    # Dataset identity.
    codebase_version: str = "v3.0"
    robot_type: str = "ego_hand_vitra"
    video_key: str = VIDEO_KEY_DEFAULT

    # Sharding behaviour.
    chunks_size: int = 1000
    rows_per_shard: int = 100_000  # data parquet shard rollover threshold
    episodes_per_shard: int = 1000

    # Video encoding (per-episode .mp4).
    video_codec: str = "h264"
    video_pix_fmt: str = "yuv420p"
    video_crf: int = 23
    video_preset: str = "medium"

    # Stats: keep occupied-but-placeholder values per spec; real Welford runs
    # only when enabled.
    enable_real_stats: bool = False

    # Misc.
    strict_dim_validation: bool = True
    main_type_default: int = -1  # -1 = unknown when atomic_action.hand is missing

    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_file(cls, path: str | Path | None) -> "LeRobotConfig":
        if path is None:
            return cls()
        with Path(path).open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
        return cls.from_mapping(raw)

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "LeRobotConfig":
        allowed = {f.name for f in cls.__dataclass_fields__.values()}
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ValueError(f"Unknown LeRobot config key(s): {', '.join(unknown)}")
        return cls(**raw)


__all__ = ["LeRobotConfig"]
