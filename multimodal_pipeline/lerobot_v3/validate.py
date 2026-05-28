from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from .._system import check_ffmpeg
from ..config import LeRobotConfig
from .schema import (
    ACTION_DIM,
    STATE_DIM,
    build_data_schema,
    build_episodes_schema,
    build_tasks_schema,
)


@dataclass
class ValidationIssue:
    code: str
    severity: str  # "warn" | "error"
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    root: Path
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(i.severity == "error" for i in self.issues)

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "warn")


def _ffprobe_frame_count(ffprobe: Path, video: Path) -> int:
    cmd = [
        str(ffprobe),
        "-v", "error",
        "-select_streams", "v:0",
        "-count_frames",
        "-show_entries", "stream=nb_read_frames",
        "-of", "json",
        str(video),
    ]
    res = subprocess.run(cmd, check=True, capture_output=True, text=True)
    info = json.loads(res.stdout)
    stream = info.get("streams", [{}])[0]
    return int(stream.get("nb_read_frames", 0))


def validate_dataset(root: Path, cfg: LeRobotConfig | None = None) -> ValidationReport:
    cfg = cfg or LeRobotConfig()
    root = Path(root)
    report = ValidationReport(root=root)

    def add(code: str, severity: str, message: str, **details: Any) -> None:
        report.issues.append(
            ValidationIssue(code=code, severity=severity, message=message, details=details)
        )

    # Required files.
    info_path = root / "meta" / "info.json"
    stats_path = root / "meta" / "stats.json"
    tasks_path = root / "meta" / "tasks.parquet"
    episodes_path = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    data_path = root / "data" / "chunk-000" / "file-000.parquet"

    for p in (info_path, stats_path, tasks_path, episodes_path, data_path):
        if not p.exists():
            add("MISSING_FILE", "error", f"required file not found: {p.relative_to(root)}")

    if not info_path.exists():
        return report

    info = json.loads(info_path.read_text(encoding="utf-8"))
    fps_declared = float(info.get("fps", 0.0))

    # Schemas.
    if data_path.exists():
        data_table = pq.read_table(data_path)
        expected = {f.name for f in build_data_schema()}
        actual = set(data_table.column_names)
        missing = expected - actual
        extra = actual - expected
        if missing:
            add("DATA_SCHEMA_MISSING_COLUMNS", "error", f"data parquet missing: {sorted(missing)}")
        if extra:
            add("DATA_SCHEMA_EXTRA_COLUMNS", "warn", f"data parquet has unexpected: {sorted(extra)}")
        # State dimension check.
        if "observation.state" in data_table.column_names:
            sample = data_table["observation.state"][0].as_py()
            if len(sample) != STATE_DIM:
                add(
                    "STATE_DIM_MISMATCH",
                    "error",
                    f"observation.state has {len(sample)} dims, expected {STATE_DIM}",
                )

    if episodes_path.exists():
        ep_table = pq.read_table(episodes_path)
        expected_ep = {f.name for f in build_episodes_schema(cfg.video_key)}
        missing = expected_ep - set(ep_table.column_names)
        if missing:
            add("EPISODES_SCHEMA_MISSING", "error", f"episodes parquet missing: {sorted(missing)}")
        # Cross-check video frame counts.
        try:
            ffmpeg = check_ffmpeg()
        except Exception as exc:
            add("FFMPEG_MISSING", "warn", f"cannot verify video frame counts: {exc}")
            ffmpeg = None
        if ffmpeg is not None:
            for i in range(ep_table.num_rows):
                ep_idx = ep_table["episode_index"][i].as_py()
                ep_length = ep_table["length"][i].as_py()
                video_rel = (
                    f"videos/{cfg.video_key}/chunk-000/episode_{ep_idx:06d}.mp4"
                )
                video_path = root / video_rel
                if not video_path.exists():
                    add("EPISODE_VIDEO_MISSING", "error", f"video missing: {video_rel}")
                    continue
                actual = _ffprobe_frame_count(ffmpeg.ffprobe, video_path)
                # Allow ±1 frame tolerance for ffmpeg encoding rounding.
                if abs(actual - ep_length) > 1:
                    add(
                        "EPISODE_FRAME_COUNT_MISMATCH",
                        "error",
                        f"episode {ep_idx}: parquet length={ep_length}, video has {actual} frames",
                    )

    if tasks_path.exists():
        t_table = pq.read_table(tasks_path)
        if "task_index" not in t_table.column_names or "task" not in t_table.column_names:
            add("TASKS_SCHEMA_INVALID", "error", "tasks.parquet missing task_index or task columns")

    # Info sanity.
    if "splits" not in info:
        add("INFO_SPLITS_MISSING", "error", "info.json missing 'splits'")
    if fps_declared <= 0:
        add("INFO_FPS_INVALID", "error", f"info.json fps={fps_declared}")

    return report


__all__ = ["ValidationIssue", "ValidationReport", "validate_dataset"]
