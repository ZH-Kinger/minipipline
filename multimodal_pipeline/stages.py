from __future__ import annotations

import bisect
import hashlib
import json
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import PipelineConfig
from .records import (
    SensorRecord,
    discover_records,
    read_jsonl,
    read_manifest,
    write_jsonl,
)


@dataclass(frozen=True)
class StageResult:
    name: str
    output: Path
    metrics: dict[str, Any]


def stage_discover(input_dir: Path, output_dir: Path, config: PipelineConfig) -> StageResult:
    manifest_path = input_dir / config.manifest_file if config.manifest_file else None
    if manifest_path is not None:
        records = read_manifest(manifest_path, input_dir, config)
        mode = "manifest"
    else:
        records = discover_records(input_dir, config)
        mode = "directory_scan"

    output = output_dir / "manifests" / "raw_manifest.jsonl"
    count = write_jsonl(output, (record.to_dict() for record in records))
    return StageResult(
        name="discover",
        output=output,
        metrics={"records": count, "mode": mode, "manifest": str(manifest_path or "")},
    )


def stage_validate(input_dir: Path, output_dir: Path) -> StageResult:
    source = output_dir / "manifests" / "raw_manifest.jsonl"
    rows = read_jsonl(source)
    valid: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []

    for row in rows:
        reasons: list[str] = []
        path_value = row.get("path")
        timestamp_ns = row.get("timestamp_ns")
        if not path_value:
            reasons.append("missing_path")
        else:
            file_path = resolve_data_path(input_dir, path_value)
            if not file_path.exists():
                reasons.append("missing_file")
        if timestamp_ns is None:
            reasons.append("missing_timestamp_ns")
        else:
            try:
                int(timestamp_ns)
            except (TypeError, ValueError):
                reasons.append("invalid_timestamp_ns")
        if not row.get("stream"):
            reasons.append("missing_stream")
        if not reasons:
            valid.append(row)
        else:
            invalid.append({**row, "validation_errors": reasons})

    valid_output = output_dir / "manifests" / "validated_manifest.jsonl"
    invalid_output = output_dir / "manifests" / "invalid_manifest.jsonl"
    write_jsonl(valid_output, valid)
    write_jsonl(invalid_output, invalid)
    return StageResult(
        name="validate",
        output=valid_output,
        metrics={"valid": len(valid), "invalid": len(invalid), "invalid_output": str(invalid_output)},
    )


def stage_align(output_dir: Path, config: PipelineConfig) -> StageResult:
    rows = read_jsonl(output_dir / "manifests" / "validated_manifest.jsonl")
    records = [record_from_json_row(row) for row in rows]
    samples = align_records(records, config)

    output = output_dir / "manifests" / "aligned_samples.jsonl"
    write_jsonl(output, samples)
    missing_total = sum(len(sample["missing_streams"]) for sample in samples)
    return StageResult(
        name="align",
        output=output,
        metrics={"samples": len(samples), "missing_stream_refs": missing_total},
    )


def stage_annotate(output_dir: Path, config: PipelineConfig) -> StageResult:
    samples = read_jsonl(output_dir / "manifests" / "aligned_samples.jsonl")
    accepted: list[dict[str, Any]] = []
    review: list[dict[str, Any]] = []

    for sample in samples:
        confidence = estimate_annotation_confidence(sample)
        label = infer_label(sample)
        annotated = {
            **sample,
            "annotation": {
                "label": label,
                "confidence": confidence,
                "source": "rules",
            },
        }
        if confidence >= config.annotation_confidence_threshold:
            accepted.append(annotated)
        else:
            review.append(
                {
                    **annotated,
                    "review_reason": "confidence_below_threshold",
                    "threshold": config.annotation_confidence_threshold,
                }
            )

    accepted_output = output_dir / "annotation" / "accepted_samples.jsonl"
    review_output = output_dir / "annotation" / "review_queue.jsonl"
    write_jsonl(accepted_output, accepted)
    write_jsonl(review_output, review)
    return StageResult(
        name="annotate",
        output=accepted_output,
        metrics={
            "accepted": len(accepted),
            "review": len(review),
            "review_output": str(review_output),
            "threshold": config.annotation_confidence_threshold,
        },
    )


def stage_pack(input_dir: Path, output_dir: Path, config: PipelineConfig) -> StageResult:
    samples = read_jsonl(output_dir / "annotation" / "accepted_samples.jsonl")
    shards_dir = output_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    index_rows: list[dict[str, Any]] = []
    index_output = output_dir / "shards" / "package_index.jsonl"
    if not samples:
        write_jsonl(index_output, index_rows)
        return StageResult(
            name="pack",
            output=index_output,
            metrics={"samples_packed": 0, "shards": 0},
        )

    shard_id = 0
    shard_size = 0
    tar: tarfile.TarFile | None = None
    tar_path: Path | None = None

    def open_next_shard() -> tarfile.TarFile:
        nonlocal shard_id, shard_size, tar_path
        tar_path = shards_dir / f"shard-{shard_id:06d}.tar"
        shard_id += 1
        shard_size = 0
        return tarfile.open(tar_path, "w")

    try:
        tar = open_next_shard()
        for sample in samples:
            payload_files = [
                resolve_data_path(input_dir, stream["path"])
                for stream in sample.get("streams", {}).values()
            ]
            estimated_size = sum(path.stat().st_size for path in payload_files if path.exists())
            if shard_size > 0 and shard_size + estimated_size > config.shard_max_size_bytes:
                tar.close()
                tar = open_next_shard()

            sample_key = sample["sample_id"]
            metadata_name = f"{sample_key}.json"
            metadata_bytes = json.dumps(sample, ensure_ascii=False, sort_keys=True).encode("utf-8")
            metadata_info = tarfile.TarInfo(metadata_name)
            metadata_info.size = len(metadata_bytes)
            tar.addfile(metadata_info, fileobj=BytesReader(metadata_bytes))
            shard_size += metadata_info.size

            for stream_name, stream in sorted(sample.get("streams", {}).items()):
                source_path = resolve_data_path(input_dir, stream["path"])
                if not source_path.exists():
                    continue
                extension = source_path.suffix or ".bin"
                member_name = f"{sample_key}.{safe_member_name(stream_name)}{extension}"
                tar.add(source_path, arcname=member_name, recursive=False)
                shard_size += source_path.stat().st_size

            index_rows.append(
                {
                    "sample_id": sample_key,
                    "shard": str(tar_path.relative_to(output_dir)) if tar_path else "",
                    "streams": sorted(sample.get("streams", {}).keys()),
                }
            )
    finally:
        if tar is not None:
            tar.close()

    write_jsonl(index_output, index_rows)
    return StageResult(
        name="pack",
        output=index_output,
        metrics={"samples_packed": len(index_rows), "shards": shard_id},
    )


def align_records(records: list[SensorRecord], config: PipelineConfig) -> list[dict[str, Any]]:
    if not records:
        return []

    by_session: dict[str, list[SensorRecord]] = {}
    for record in records:
        by_session.setdefault(record.session_id, []).append(record)

    samples: list[dict[str, Any]] = []
    for session_id, session_records in sorted(by_session.items()):
        by_stream = group_by_stream(session_records)
        expected_streams = config.expected_streams or sorted(by_stream)
        reference_records = [
            record
            for record in session_records
            if record.stream_key == config.reference_stream
            or record.modality == config.reference_stream
        ]
        if not reference_records:
            reference_stream = expected_streams[0]
            reference_records = by_stream.get(reference_stream, [])
        reference_records = sorted(reference_records, key=lambda item: item.timestamp_ns)

        for anchor in reference_records:
            streams: dict[str, dict[str, Any]] = {}
            missing: list[str] = []
            for stream in expected_streams:
                nearest = nearest_record(by_stream.get(stream, []), anchor.timestamp_ns)
                if nearest is None:
                    missing.append(stream)
                    continue
                delta_ms = abs(nearest.timestamp_ns - anchor.timestamp_ns) / 1_000_000
                if delta_ms > config.alignment_tolerance_ms:
                    missing.append(stream)
                    continue
                streams[stream] = {
                    "path": nearest.path,
                    "modality": nearest.modality,
                    "sensor_id": nearest.sensor_id,
                    "timestamp_ns": nearest.timestamp_ns,
                    "delta_ms": round(delta_ms, 6),
                    "frame_id": nearest.frame_id,
                    "metadata": nearest.metadata,
                }

            sample_id = stable_sample_id(session_id, anchor.timestamp_ns, anchor.path)
            samples.append(
                {
                    "sample_id": sample_id,
                    "session_id": session_id,
                    "anchor_stream": anchor.stream_key,
                    "timestamp_ns": anchor.timestamp_ns,
                    "streams": streams,
                    "missing_streams": missing,
                }
            )
    return samples


def group_by_stream(records: list[SensorRecord]) -> dict[str, list[SensorRecord]]:
    grouped: dict[str, list[SensorRecord]] = {}
    for record in records:
        grouped.setdefault(record.stream_key, []).append(record)
    for stream_records in grouped.values():
        stream_records.sort(key=lambda item: item.timestamp_ns)
    return grouped


def nearest_record(records: list[SensorRecord], timestamp_ns: int) -> SensorRecord | None:
    if not records:
        return None
    timestamps = [record.timestamp_ns for record in records]
    index = bisect.bisect_left(timestamps, timestamp_ns)
    candidates = []
    if index < len(records):
        candidates.append(records[index])
    if index > 0:
        candidates.append(records[index - 1])
    return min(candidates, key=lambda item: abs(item.timestamp_ns - timestamp_ns))


def record_from_json_row(row: dict[str, Any]) -> SensorRecord:
    return SensorRecord(
        session_id=str(row["session_id"]),
        stream=str(row["stream"]),
        modality=str(row["modality"]),
        sensor_id=str(row["sensor_id"]),
        timestamp_ns=int(row["timestamp_ns"]),
        path=str(row["path"]),
        frame_id=str(row["frame_id"]) if row.get("frame_id") is not None else None,
        metadata=row.get("metadata", {}) if isinstance(row.get("metadata", {}), dict) else {},
    )


def estimate_annotation_confidence(sample: dict[str, Any]) -> float:
    streams = sample.get("streams", {})
    missing = sample.get("missing_streams", [])
    if not streams:
        return 0.0
    completeness = len(streams) / (len(streams) + len(missing))
    max_delta = max((stream.get("delta_ms", 0.0) for stream in streams.values()), default=0.0)
    timing_score = max(0.0, 1.0 - (max_delta / 100.0))
    metadata_score = 1.0 if any(stream.get("metadata", {}).get("label") for stream in streams.values()) else 0.5
    metadata_confidences = []
    for stream in streams.values():
        metadata = stream.get("metadata", {})
        parsed_confidence = parse_confidence(metadata.get("confidence"))
        if parsed_confidence is not None:
            metadata_confidences.append(parsed_confidence)
    if metadata_confidences:
        metadata_score = min(metadata_score, max(metadata_confidences))
    return round(min(completeness, timing_score, metadata_score), 4)


def parse_confidence(value: Any) -> float | None:
    if value is None:
        return None
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, confidence))


def infer_label(sample: dict[str, Any]) -> str:
    for stream in sample.get("streams", {}).values():
        metadata = stream.get("metadata", {})
        label = metadata.get("label") or metadata.get("action") or metadata.get("state")
        if label:
            return str(label)
    return "unlabeled"


def stable_sample_id(session_id: str, timestamp_ns: int, path: str) -> str:
    digest = hashlib.sha1(f"{session_id}:{timestamp_ns}:{path}".encode("utf-8")).hexdigest()[:10]
    return f"{safe_member_name(session_id)}-{timestamp_ns}-{digest}"


def safe_member_name(value: str) -> str:
    allowed = []
    for char in value:
        if char.isalnum() or char in {"-", "_", "."}:
            allowed.append(char)
        else:
            allowed.append("_")
    return "".join(allowed).strip("._") or "item"


def resolve_data_path(input_dir: Path, path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else input_dir / path


class BytesReader:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        self._offset = 0

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = len(self._payload) - self._offset
        chunk = self._payload[self._offset : self._offset + size]
        self._offset += len(chunk)
        return chunk
