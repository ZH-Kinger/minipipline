from __future__ import annotations

import dataclasses
import json
import shutil
import wave
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .align import AlignedData
from .annotate import AnnotatedData, HandPerFrame
from ..config import ITWConfig
from .schemas import (
    Calibration,
    DiscoveryResult,
    IngestReport,
    NIRSession,
    ValidationResult,
)


def _path_to_str(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    return value


def _dataclass_to_jsonable(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj):
        out: dict[str, Any] = {}
        for f in dataclasses.fields(obj):
            out[f.name] = _dataclass_to_jsonable(getattr(obj, f.name))
        return out
    if isinstance(obj, dict):
        return {k: _dataclass_to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_dataclass_to_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")
    tmp.replace(path)


def _write_frame_index_parquet(
    out_path: Path,
    aligned: AlignedData,
    annotated: AnnotatedData,
) -> None:
    n = aligned.rgb_timestamps_s.shape[0]
    frame_idx = np.arange(n, dtype=np.int64)
    rgb_ts_ns = (aligned.rgb_timestamps_s * 1e9).astype(np.int64)
    depth_ts_ns = np.where(
        np.isfinite(aligned.depth_timestamps_s),
        (np.nan_to_num(aligned.depth_timestamps_s, nan=0.0) * 1e9),
        0,
    ).astype(np.int64)
    depth_present = np.isfinite(aligned.depth_timestamps_s)

    excluded = np.fromiter((f.excluded for f in annotated.hand_frames), dtype=bool, count=n)
    exclude_reason = pa.array([f.exclude_reason for f in annotated.hand_frames], type=pa.string())
    left_present = np.fromiter((f.left_present for f in annotated.hand_frames), dtype=bool, count=n)
    right_present = np.fromiter((f.right_present for f in annotated.hand_frames), dtype=bool, count=n)

    table = pa.table(
        {
            "frame_index": pa.array(frame_idx, type=pa.int64()),
            "rgb_timestamp_ns": pa.array(rgb_ts_ns, type=pa.int64()),
            "depth_timestamp_ns": pa.array(depth_ts_ns, type=pa.int64()),
            "depth_present": pa.array(depth_present, type=pa.bool_()),
            "depth_offset_ms": pa.array(aligned.depth_offsets_ms, type=pa.float32()),
            "left_hand_present": pa.array(left_present, type=pa.bool_()),
            "right_hand_present": pa.array(right_present, type=pa.bool_()),
            "excluded": pa.array(excluded, type=pa.bool_()),
            "exclude_reason": exclude_reason,
            "is_gap_anomaly": pa.array(
                annotated.is_gap_anomaly_per_frame, type=pa.bool_()
            ),
        }
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_path, compression="zstd")


def _write_head_pose_parquet(out_path: Path, aligned: AlignedData) -> None:
    hp = aligned.head_pose
    table = pa.table(
        {
            "frame_index": pa.array(np.arange(hp.shape[0], dtype=np.int64), type=pa.int64()),
            "tx": pa.array(hp[:, 0], type=pa.float32()),
            "ty": pa.array(hp[:, 1], type=pa.float32()),
            "tz": pa.array(hp[:, 2], type=pa.float32()),
            "qx": pa.array(hp[:, 3], type=pa.float32()),
            "qy": pa.array(hp[:, 4], type=pa.float32()),
            "qz": pa.array(hp[:, 5], type=pa.float32()),
            "qw": pa.array(hp[:, 6], type=pa.float32()),
        }
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_path, compression="zstd")


def _flat_keypoints(kpts: list[tuple[float, float, float]] | None, kp_count: int) -> list[float]:
    if kpts is None or len(kpts) != kp_count:
        return [float("nan")] * (kp_count * 3)
    flat: list[float] = []
    for x, y, z in kpts:
        flat.extend((float(x), float(y), float(z)))
    return flat


def _write_hand_keypoints_parquet(out_path: Path, annotated: AnnotatedData) -> None:
    kp_count = 21
    n = len(annotated.hand_frames)
    left_kpts = [_flat_keypoints(f.left_keypoints_3d, kp_count) for f in annotated.hand_frames]
    right_kpts = [_flat_keypoints(f.right_keypoints_3d, kp_count) for f in annotated.hand_frames]

    def _wrist(f: HandPerFrame, side: str) -> list[float]:
        coord = f.left_wrist_xyz if side == "L" else f.right_wrist_xyz
        if coord is None:
            return [float("nan"), float("nan"), float("nan")]
        return [float(coord[0]), float(coord[1]), float(coord[2])]

    table = pa.table(
        {
            "frame_index": pa.array(np.arange(n, dtype=np.int64), type=pa.int64()),
            "left_present": pa.array(
                [f.left_present for f in annotated.hand_frames], type=pa.bool_()
            ),
            "right_present": pa.array(
                [f.right_present for f in annotated.hand_frames], type=pa.bool_()
            ),
            "excluded": pa.array(
                [f.excluded for f in annotated.hand_frames], type=pa.bool_()
            ),
            "exclude_reason": pa.array(
                [f.exclude_reason for f in annotated.hand_frames], type=pa.string()
            ),
            "left_confidence": pa.array(
                [f.left_confidence for f in annotated.hand_frames], type=pa.string()
            ),
            "right_confidence": pa.array(
                [f.right_confidence for f in annotated.hand_frames], type=pa.string()
            ),
            "left_wrist_xyz": pa.array(
                [_wrist(f, "L") for f in annotated.hand_frames], type=pa.list_(pa.float32())
            ),
            "right_wrist_xyz": pa.array(
                [_wrist(f, "R") for f in annotated.hand_frames], type=pa.list_(pa.float32())
            ),
            "left_keypoints_3d_flat": pa.array(left_kpts, type=pa.list_(pa.float32())),
            "right_keypoints_3d_flat": pa.array(right_kpts, type=pa.list_(pa.float32())),
        }
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_path, compression="zstd")


def _write_imu_npz(out_path: Path, aligned: AlignedData) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        timestamps_s=aligned.imu_block[:, 0],
        acc=aligned.imu_block[:, 1:4],
        gyr=aligned.imu_block[:, 4:7],
    )


def _write_audio_wav(out_path: Path, aligned: AlignedData) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    samples = aligned.audio_block
    if samples.ndim == 1:
        samples = samples.reshape(-1, 1)
    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(aligned.audio_channels)
        wf.setsampwidth(2)  # int16 PCM
        wf.setframerate(aligned.audio_sample_rate)
        wf.writeframes(samples.astype(np.int16).tobytes())


def _materialize_video(src: Path, dst: Path, copy: bool) -> tuple[str, str]:
    """Return (kind, target_path_string). kind ∈ {'copied', 'referenced'}."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        shutil.copyfile(src, dst)
        return "copied", str(dst)
    return "referenced", str(src.resolve())


def normalize(
    discovery: DiscoveryResult,
    aligned: AlignedData,
    annotated: AnnotatedData,
    calibration: Calibration,
    validation: ValidationResult,
    out_root: Path,
    cfg: ITWConfig,
) -> tuple[NIRSession, IngestReport]:
    """Write the NIR layout under `out_root/<session_id>/` and return the report."""
    nir_dir = (out_root / discovery.session_id).resolve()
    nir_dir.mkdir(parents=True, exist_ok=True)

    output_files: list[str] = []

    # 1. calibration.json
    calib_path = nir_dir / "calibration.json"
    _write_json(calib_path, _dataclass_to_jsonable(calibration))
    output_files.append("calibration.json")

    # 2. task.json — task_info + a summary block useful for downstream consumers.
    task_payload = {
        "task_info": _dataclass_to_jsonable(discovery.task_info),
        "task_text": annotated.task_text,
        "annotation_summary": _dataclass_to_jsonable(annotated.summary),
    }
    _write_json(nir_dir / "task.json", task_payload)
    output_files.append("task.json")

    # 3. frame_index.parquet
    _write_frame_index_parquet(nir_dir / "frame_index.parquet", aligned, annotated)
    output_files.append("frame_index.parquet")

    # 4. head_pose.parquet
    _write_head_pose_parquet(nir_dir / "head_pose.parquet", aligned)
    output_files.append("head_pose.parquet")

    # 5. hand_keypoints.parquet
    _write_hand_keypoints_parquet(nir_dir / "hand_keypoints.parquet", annotated)
    output_files.append("hand_keypoints.parquet")

    # 6. imu_cropped.npz
    _write_imu_npz(nir_dir / "imu_cropped.npz", aligned)
    output_files.append("imu_cropped.npz")

    # 7. audio_cropped.wav
    _write_audio_wav(nir_dir / "audio_cropped.wav", aligned)
    output_files.append("audio_cropped.wav")

    # 8. RGB + Depth videos (copy or reference)
    rgb_src = discovery.found_files["rgb_head"]
    depth_src = discovery.found_files.get("depth_head")
    rgb_kind, rgb_target = _materialize_video(rgb_src, nir_dir / "rgb.mp4", cfg.copy_video)
    media_paths: dict[str, str] = {"rgb": rgb_target}
    if cfg.copy_video:
        output_files.append("rgb.mp4")
    if depth_src is not None:
        depth_kind, depth_target = _materialize_video(
            depth_src, nir_dir / "depth.mkv", cfg.copy_video
        )
        media_paths["depth"] = depth_target
        if cfg.copy_video:
            output_files.append("depth.mkv")
    _write_json(
        nir_dir / "media_paths.json",
        {
            "rgb_kind": rgb_kind,
            "media": media_paths,
            "fps": aligned.summary.fps_nominal,
            "fps_actual": aligned.summary.fps_actual,
            "frame_count": aligned.summary.rgb_frame_count,
        },
    )
    output_files.append("media_paths.json")

    # 9. ingest_report.json — last so any earlier IO errors are visible if it's missing.
    report = IngestReport(
        session_id=discovery.session_id,
        session_dir=discovery.session_dir,
        nir_dir=nir_dir,
        discovery=discovery,
        validation=validation,
        alignment=aligned.summary,
        annotation=annotated.summary,
        output_files=tuple(output_files),
        calibration_ok=calibration.cross_check_passed,
    )
    _write_json(nir_dir / "ingest_report.json", _dataclass_to_jsonable(report))
    output_files.append("ingest_report.json")

    nir_session = NIRSession(
        root=nir_dir,
        session_id=discovery.session_id,
        frame_count=aligned.summary.rgb_frame_count,
        fps=aligned.summary.fps_nominal,
        duration_s=aligned.summary.duration_s,
        has_depth=depth_src is not None,
        has_audio=aligned.audio_block.size > 0,
        has_imu=aligned.imu_block.shape[0] > 0,
        has_hand_keypoints=any(
            f.left_present or f.right_present for f in annotated.hand_frames
        ),
    )
    return nir_session, report


__all__ = ["normalize"]
