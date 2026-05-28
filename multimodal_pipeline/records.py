from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .config import PipelineConfig


@dataclass(frozen=True)
class SensorRecord:
    session_id: str
    stream: str
    modality: str
    sensor_id: str
    timestamp_ns: int
    path: str
    frame_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def stream_key(self) -> str:
        return self.stream

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def read_manifest(path: Path, input_dir: Path, config: PipelineConfig) -> list[SensorRecord]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
    else:
        rows = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))

    return [record_from_mapping(row, input_dir, config) for row in rows]


def discover_records(input_dir: Path, config: PipelineConfig) -> list[SensorRecord]:
    records: list[SensorRecord] = []
    timestamp_pattern = re.compile(config.timestamp_regex)
    ignored = {name.lower() for name in config.ignore_dirs}

    for path in sorted(input_dir.rglob("*")):
        if not path.is_file():
            continue
        if any(part.lower() in ignored for part in path.relative_to(input_dir).parts):
            continue
        modality = infer_modality(path, config)
        if modality is None:
            continue
        metadata = read_sidecar_metadata(path)
        timestamp_ns = infer_timestamp_ns(path, timestamp_pattern)
        if timestamp_ns is None:
            timestamp_ns = infer_timestamp_from_metadata(metadata)
        if timestamp_ns is None:
            continue
        sensor_id = infer_sensor_id(path, input_dir, modality)
        rel_path = path.relative_to(input_dir).as_posix()
        stream = f"{modality}:{sensor_id}"
        metadata = {**metadata, "discovered": True}
        records.append(
            SensorRecord(
                session_id=config.session_id,
                stream=stream,
                modality=modality,
                sensor_id=sensor_id,
                timestamp_ns=timestamp_ns,
                path=rel_path,
                frame_id=path.stem,
                metadata=metadata,
            )
        )
    return records


def record_from_mapping(
    row: dict[str, Any], input_dir: Path, config: PipelineConfig
) -> SensorRecord:
    path_value = str(row.get("path") or row.get("file") or row.get("uri") or "").strip()
    if not path_value:
        raise ValueError(f"Manifest row is missing path/file/uri: {row}")

    source_path = Path(path_value)
    if source_path.is_absolute():
        try:
            stored_path = source_path.relative_to(input_dir).as_posix()
        except ValueError:
            stored_path = source_path.as_posix()
    else:
        stored_path = source_path.as_posix()

    materialized_path = source_path if source_path.is_absolute() else input_dir / source_path
    modality = str(row.get("modality") or infer_modality(materialized_path, config) or "unknown")
    sensor_id = str(row.get("sensor_id") or row.get("sensor") or modality)
    stream = str(row.get("stream") or f"{modality}:{sensor_id}")
    timestamp_ns = parse_timestamp_ns(row)
    session_id = str(row.get("session_id") or config.session_id)
    frame_id = row.get("frame_id")
    metadata = row.get("metadata", {})
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            metadata = {"raw_metadata": metadata}
    if not isinstance(metadata, dict):
        metadata = {"raw_metadata": metadata}

    extra = {
        key: value
        for key, value in row.items()
        if key
        not in {
            "session_id",
            "stream",
            "modality",
            "sensor_id",
            "sensor",
            "timestamp_ns",
            "timestamp_us",
            "timestamp_ms",
            "timestamp_s",
            "timestamp",
            "path",
            "file",
            "uri",
            "frame_id",
            "metadata",
        }
    }
    if extra:
        metadata = {**metadata, "manifest_extra": extra}

    return SensorRecord(
        session_id=session_id,
        stream=stream,
        modality=modality,
        sensor_id=sensor_id,
        timestamp_ns=timestamp_ns,
        path=stored_path,
        frame_id=str(frame_id) if frame_id is not None else materialized_path.stem,
        metadata=metadata,
    )


def parse_timestamp_ns(row: dict[str, Any]) -> int:
    if row.get("timestamp_ns") not in (None, ""):
        return int(float(row["timestamp_ns"]))
    if row.get("timestamp_us") not in (None, ""):
        return int(float(row["timestamp_us"]) * 1_000)
    if row.get("timestamp_ms") not in (None, ""):
        return int(float(row["timestamp_ms"]) * 1_000_000)
    if row.get("timestamp_s") not in (None, ""):
        return int(float(row["timestamp_s"]) * 1_000_000_000)
    if row.get("timestamp") not in (None, ""):
        value = float(row["timestamp"])
        if value > 1_000_000_000_000_000:
            return int(value)
        if value > 1_000_000_000_000:
            return int(value * 1_000)
        if value > 1_000_000_000:
            return int(value * 1_000_000)
        return int(value * 1_000_000_000)
    raise ValueError(f"Manifest row is missing timestamp: {row}")


def infer_timestamp_from_metadata(metadata: dict[str, Any]) -> int | None:
    if not metadata:
        return None
    try:
        return parse_timestamp_ns(metadata)
    except (TypeError, ValueError):
        return None


def read_sidecar_metadata(path: Path) -> dict[str, Any]:
    if path.suffix.lower() != ".json" or path.stat().st_size > 5 * 1024 * 1024:
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def infer_modality(path: Path, config: PipelineConfig) -> str | None:
    lower = path.as_posix().lower()
    suffix = path.suffix.lower()
    if "depth" in lower:
        return "depth"
    if "imu" in lower:
        return "imu"
    if "pose" in lower or "6dof" in lower or "tf" in lower:
        return "pose"
    if "pointcloud" in lower or "point_cloud" in lower or "lidar" in lower:
        return "pointcloud"
    for modality, suffixes in config.modality_extensions.items():
        if suffix in {item.lower() for item in suffixes}:
            return modality
    return None


def infer_timestamp_ns(path: Path, pattern: re.Pattern[str]) -> int | None:
    match = pattern.search(path.stem)
    if not match:
        match = pattern.search(path.name)
    if not match:
        return None
    value = match.groupdict().get("timestamp") or match.group(0)
    number = int(value)
    digits = len(value)
    if digits >= 18:
        return number
    if digits >= 16:
        return number * 1_000
    if digits >= 13:
        return number * 1_000_000
    return number * 1_000_000_000


def infer_sensor_id(path: Path, input_dir: Path, modality: str) -> str:
    relative = path.relative_to(input_dir)
    parts = list(relative.parts)
    if len(parts) >= 2:
        for part in parts[:-1]:
            normalized = part.lower()
            if normalized not in {modality, "data", "raw", "frames"}:
                return part
    return modality


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            fh.write("\n")
            count += 1
    return count


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
