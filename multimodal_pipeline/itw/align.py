from __future__ import annotations

import csv
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import ITWConfig
from .schemas import AlignmentResult, DiscoveryResult, GapAnomaly


@dataclass(frozen=True)
class AlignedData:
    """In-memory aligned arrays produced by `align()` and consumed by `normalize()`.

    Not JSON-serializable; this struct just passes between stages within the
    same pipeline run. The serializable summary lives in `AlignmentResult`.
    """

    rgb_timestamps_s: np.ndarray         # (N,) float64
    depth_timestamps_s: np.ndarray       # (N,) float64
    depth_offsets_ms: np.ndarray         # (N,) float32
    head_pose: np.ndarray                # (N, 7) float32: tx,ty,tz,qx,qy,qz,qw
    imu_block: np.ndarray                # (M, 7) float32: ts,ax,ay,az,gx,gy,gz
    audio_block: np.ndarray              # (S, C) int16
    audio_sample_rate: int
    audio_channels: int
    summary: AlignmentResult


def _read_two_col_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a `frame_index,timestamp_s` CSV and return (frame_idx, timestamp_s)."""
    frames: list[int] = []
    times: list[float] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        next(reader, None)  # header
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            frames.append(int(row[0]))
            times.append(float(row[1]))
    return np.asarray(frames, dtype=np.int64), np.asarray(times, dtype=np.float64)


def _read_head_pose_csv(path: Path) -> np.ndarray:
    """Read `timestamp_s,head_tx,...,head_qw` and return (N, 7) float32 of tx..qw."""
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader, None) or []
        # Expect 8 columns: timestamp + 3 transl + 4 quat
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            rows.append([float(x) for x in row[1:8]])
    return np.asarray(rows, dtype=np.float32)


def _read_imu_txt(path: Path) -> np.ndarray:
    """Space-delimited imu.txt → (M, 7) float32."""
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            rows.append([float(x) for x in parts[:7]])
    return np.asarray(rows, dtype=np.float64)


def _read_wav_pcm(path: Path) -> tuple[np.ndarray, int, int]:
    """Read a PCM WAV file into (frames, channels) int16 + sample rate + channels."""
    with wave.open(str(path), "rb") as wf:
        sample_rate = wf.getframerate()
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)
    if sample_width != 2:
        raise ValueError(
            f"audio crop currently expects 16-bit PCM (sample_width=2), got {sample_width}"
        )
    samples = np.frombuffer(raw, dtype=np.int16)
    if channels > 1:
        samples = samples.reshape(-1, channels)
    else:
        samples = samples.reshape(-1, 1)
    return samples, sample_rate, channels


def _detect_gap_anomalies(
    rgb_times: np.ndarray, nominal_fps: float, factor: float
) -> list[GapAnomaly]:
    if rgb_times.size < 2:
        return []
    intervals_ms = np.diff(rgb_times) * 1000.0
    nominal_ms = 1000.0 / nominal_fps
    threshold_ms = nominal_ms * factor
    out: list[GapAnomaly] = []
    for i, dt in enumerate(intervals_ms):
        if dt > threshold_ms:
            out.append(
                GapAnomaly(
                    frame_index=int(i + 1),
                    interval_ms=float(dt),
                    nominal_ms=float(nominal_ms),
                )
            )
    return out


def align(discovery: DiscoveryResult, cfg: ITWConfig) -> AlignedData:
    rgb_csv = discovery.found_files["rgb_head_timestamps"]
    depth_csv = discovery.found_files.get("depth_head_timestamps")
    head_pose_csv = discovery.found_files["head_hands_sixdof"]
    imu_path = discovery.found_files["imu"]
    audio_path = discovery.found_files["mic"]

    rgb_idx, rgb_times = _read_two_col_csv(rgb_csv)
    if rgb_idx.size == 0:
        raise ValueError("rgb_head.csv is empty; cannot align.")
    t0_s = float(rgb_times[0])
    t1_s = float(rgb_times[-1])
    rgb_n = int(rgb_idx.size)

    if depth_csv is not None:
        _, depth_times = _read_two_col_csv(depth_csv)
        if depth_times.size != rgb_n:
            # Truncate or pad to match (1:1 by frame_index).
            depth_times = depth_times[:rgb_n] if depth_times.size > rgb_n else np.pad(
                depth_times,
                (0, rgb_n - depth_times.size),
                constant_values=np.nan,
            )
    else:
        depth_times = np.full(rgb_n, np.nan, dtype=np.float64)

    depth_offsets_ms = ((depth_times - rgb_times) * 1000.0).astype(np.float32)
    finite = np.isfinite(depth_offsets_ms)
    if finite.any():
        offset_mean = float(np.mean(depth_offsets_ms[finite]))
        offset_max = float(np.max(np.abs(depth_offsets_ms[finite])))
    else:
        offset_mean = 0.0
        offset_max = 0.0

    head_pose = _read_head_pose_csv(head_pose_csv)
    if head_pose.shape[0] != rgb_n:
        head_pose = head_pose[:rgb_n] if head_pose.shape[0] > rgb_n else np.pad(
            head_pose, ((0, rgb_n - head_pose.shape[0]), (0, 0))
        )

    # IMU: keep samples in [t0_s, t1_s + 1/fps).
    imu_all = _read_imu_txt(imu_path)
    nominal_dt = 1.0 / cfg.nominal_fps
    upper_bound = t1_s + nominal_dt
    if imu_all.size:
        ts = imu_all[:, 0]
        keep_mask = (ts >= t0_s) & (ts < upper_bound)
        imu_block = imu_all[keep_mask]
        imu_dropped = int(imu_all.shape[0] - imu_block.shape[0])
    else:
        imu_block = np.zeros((0, 7), dtype=np.float64)
        imu_dropped = 0

    # Audio: assume sample 0 corresponds to t0_s; crop to RGB duration.
    audio_samples, sample_rate, n_channels = _read_wav_pcm(audio_path)
    audio_pad_s = cfg.audio_pad_ms / 1000.0
    target_duration_s = (t1_s - t0_s) + nominal_dt + audio_pad_s
    target_frames = int(round(target_duration_s * sample_rate))
    if target_frames < audio_samples.shape[0]:
        audio_block = audio_samples[:target_frames]
        audio_dropped = audio_samples.shape[0] - target_frames
    else:
        audio_block = audio_samples
        audio_dropped = 0

    gaps = _detect_gap_anomalies(rgb_times, cfg.nominal_fps, cfg.gap_anomaly_factor)
    fps_actual = (rgb_n - 1) / (t1_s - t0_s) if rgb_n > 1 and t1_s > t0_s else cfg.nominal_fps

    summary = AlignmentResult(
        reference_stream=cfg.reference_stream,
        fps_nominal=cfg.nominal_fps,
        fps_actual=float(fps_actual),
        t0_s=t0_s,
        t1_s=t1_s,
        duration_s=t1_s - t0_s + nominal_dt,
        rgb_frame_count=rgb_n,
        depth_frame_count=int(np.isfinite(depth_times).sum()),
        depth_offset_mean_ms=offset_mean,
        depth_offset_max_ms=offset_max,
        imu_kept_samples=int(imu_block.shape[0]),
        imu_dropped_samples=imu_dropped,
        audio_kept_samples=int(audio_block.shape[0]),
        audio_dropped_samples=audio_dropped,
        gap_anomalies=tuple(gaps),
    )

    return AlignedData(
        rgb_timestamps_s=rgb_times,
        depth_timestamps_s=depth_times,
        depth_offsets_ms=depth_offsets_ms,
        head_pose=head_pose,
        imu_block=imu_block.astype(np.float32),
        audio_block=audio_block,
        audio_sample_rate=sample_rate,
        audio_channels=n_channels,
        summary=summary,
    )


__all__ = ["AlignedData", "align"]
