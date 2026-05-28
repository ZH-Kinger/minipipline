from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .._system import check_ffmpeg
from ..config import LeRobotConfig
from .rotations import (
    axis_angle_to_rotmat,
    joint_aa_per_frame_to_flat_rotmat,
    single_aa_per_frame_to_flat_rotmat,
)
from .schema import (
    ACTION_DIM,
    BETAS_DIM,
    EXTRINSICS_FLAT_DIM,
    FOV_DIM,
    HAND_POSE_AXISANGLE_DIM,
    HAND_POSE_ROTMAT_DIM,
    HAND_STATE_DIM,
    ORIENT_ROTMAT_DIM,
    STATE_DIM,
    STATE_LAYOUT,
    STATE_MASK_DIM,
    WRIST_TRANSL_DIM,
    build_data_schema,
    build_episodes_schema,
    build_info_dict,
    build_tasks_schema,
)
from .stats import StatsAccumulator


HAND_LEFT = 0
HAND_RIGHT = 1


@dataclass
class EpisodeInput:
    """Per-episode payload assembled by callers (typically Layer 2 outputs)."""

    episode_index: int
    task_text: str
    main_type: int          # 0=left, 1=right, -1=unknown
    source_video_path: Path
    source_fps: float
    frame_start: int        # inclusive, in the source-video frame indexing
    frame_end: int          # exclusive
    # Per-frame world-coord predictions (slice from MergedPrediction).
    pred_trans: np.ndarray       # (2, T, 3)
    pred_rot_aa: np.ndarray      # (2, T, 3) axis-angle
    pred_hand_pose_aa: np.ndarray  # (2, T, 45)
    pred_betas: np.ndarray       # (2, T, 10)
    pred_kept: np.ndarray        # (2, T) bool
    # Camera per-frame (world->camera) extrinsics, plus intrinsics.
    extrinsics_w2c: np.ndarray   # (T, 4, 4)
    intrinsics_fov: tuple[float, float]   # (hfov_deg, vfov_deg)
    # Layer 1.5 annotation (per-episode, repeated to each row).
    action_label: str = ""
    action_score: float = 0.0

    @property
    def length(self) -> int:
        return self.frame_end - self.frame_start


@dataclass
class DatasetReport:
    root: Path
    total_episodes: int
    total_frames: int
    total_tasks: int
    total_videos: int
    data_files: list[str] = field(default_factory=list)
    video_files: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)


def _apply_extrinsic_to_translation(
    transl_world: np.ndarray, w2c: np.ndarray
) -> np.ndarray:
    """Apply (T,4,4) extrinsics to (T,3) world-translations → (T,3) camera-space."""
    homo = np.concatenate([transl_world, np.ones((transl_world.shape[0], 1), dtype=np.float32)], axis=1)
    out = np.einsum("tij,tj->ti", w2c, homo)
    return out[:, :3]


def _apply_extrinsic_to_orient_aa(
    orient_aa_world: np.ndarray, w2c: np.ndarray
) -> np.ndarray:
    """Rotate world-frame axis-angle orientations into camera frame.

    Strategy: convert AA→R_w, apply R_c = R_extr @ R_w, then back to AA.
    """
    T = orient_aa_world.shape[0]
    R_w = axis_angle_to_rotmat(orient_aa_world)  # (T,3,3)
    R_extr = w2c[:, :3, :3]  # (T,3,3)
    R_c = np.einsum("tij,tjk->tik", R_extr, R_w)  # (T,3,3)
    # Convert R_c back to axis-angle using log map.
    out = np.empty((T, 3), dtype=np.float32)
    for i in range(T):
        R = R_c[i]
        tr = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
        theta = float(np.arccos(tr))
        if theta < 1e-8:
            out[i] = np.zeros(3, dtype=np.float32)
            continue
        # axis from off-diagonal
        rx = (R[2, 1] - R[1, 2]) / (2.0 * np.sin(theta))
        ry = (R[0, 2] - R[2, 0]) / (2.0 * np.sin(theta))
        rz = (R[1, 0] - R[0, 1]) / (2.0 * np.sin(theta))
        out[i] = np.array([rx, ry, rz], dtype=np.float32) * theta
    return out


class LeRobotV3DatasetWriter:
    """Writes a LeRobot v3 dataset to disk.

    Lifecycle:
        with LeRobotV3DatasetWriter(root, cfg) as w:
            for ep in episodes:
                idx = w.register_task(ep.task_text)
                w.add_episode(ep, task_index=idx)
            report = w.finalize()
    """

    def __init__(self, root: Path, cfg: LeRobotConfig, fps: float):
        self.root = Path(root)
        self.cfg = cfg
        self.fps = float(fps)
        self._ffmpeg = check_ffmpeg()

        self._tasks: dict[str, int] = {}        # task_text → task_index
        self._episodes: list[dict[str, Any]] = []
        self._data_rows: list[dict[str, Any]] = []
        self._data_files: list[str] = []
        self._video_files: list[str] = []
        self._next_global_index = 0
        self._total_frames = 0
        self._video_size: tuple[int, int] | None = None  # (H, W)

        self._state_stats = StatsAccumulator(
            dim=STATE_DIM, name="observation.state", enabled=cfg.enable_real_stats
        )
        self._action_stats = StatsAccumulator(
            dim=ACTION_DIM, name="action", enabled=cfg.enable_real_stats
        )

    def __enter__(self) -> "LeRobotV3DatasetWriter":
        (self.root / "meta").mkdir(parents=True, exist_ok=True)
        (self.root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
        (self.root / "videos" / self.cfg.video_key / "chunk-000").mkdir(parents=True, exist_ok=True)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def register_task(self, text: str) -> int:
        if text not in self._tasks:
            self._tasks[text] = len(self._tasks)
        return self._tasks[text]

    def add_episode(self, ep: EpisodeInput, task_index: int) -> None:
        T = ep.length
        if T <= 0:
            raise ValueError(f"episode {ep.episode_index} has non-positive length {T}")
        if ep.pred_trans.shape != (2, T, 3):
            raise ValueError(f"pred_trans expected (2,{T},3), got {ep.pred_trans.shape}")

        # 1. World-frame fields (per-frame, flattened to 1-D for parquet).
        left_transl_world = ep.pred_trans[HAND_LEFT].astype(np.float32)   # (T,3)
        right_transl_world = ep.pred_trans[HAND_RIGHT].astype(np.float32) # (T,3)
        left_orient_world_rotmat = single_aa_per_frame_to_flat_rotmat(ep.pred_rot_aa[HAND_LEFT])
        right_orient_world_rotmat = single_aa_per_frame_to_flat_rotmat(ep.pred_rot_aa[HAND_RIGHT])

        # 2. MANO 15-joint pose: axis-angle (45,) → rotation matrices flattened (135,).
        left_pose_aa = ep.pred_hand_pose_aa[HAND_LEFT].astype(np.float32)  # (T, 45)
        right_pose_aa = ep.pred_hand_pose_aa[HAND_RIGHT].astype(np.float32)
        left_pose_flat = joint_aa_per_frame_to_flat_rotmat(left_pose_aa.reshape(T, 15, 3))
        right_pose_flat = joint_aa_per_frame_to_flat_rotmat(right_pose_aa.reshape(T, 15, 3))

        # 3. observation.state = camera-frame compositional vector (122,).
        left_transl_cam = _apply_extrinsic_to_translation(left_transl_world, ep.extrinsics_w2c)
        right_transl_cam = _apply_extrinsic_to_translation(right_transl_world, ep.extrinsics_w2c)
        left_orient_cam_aa = _apply_extrinsic_to_orient_aa(
            ep.pred_rot_aa[HAND_LEFT].astype(np.float32), ep.extrinsics_w2c
        )
        right_orient_cam_aa = _apply_extrinsic_to_orient_aa(
            ep.pred_rot_aa[HAND_RIGHT].astype(np.float32), ep.extrinsics_w2c
        )
        left_betas = ep.pred_betas[HAND_LEFT].astype(np.float32)   # (T, 10)
        right_betas = ep.pred_betas[HAND_RIGHT].astype(np.float32)

        state = np.empty((T, STATE_DIM), dtype=np.float32)
        sl = STATE_LAYOUT
        state[:, sl["left_wrist_transl_cam"][0]:sl["left_wrist_transl_cam"][1]] = left_transl_cam
        state[:, sl["left_wrist_orient_cam_aa"][0]:sl["left_wrist_orient_cam_aa"][1]] = left_orient_cam_aa
        state[:, sl["left_mano_pose_aa"][0]:sl["left_mano_pose_aa"][1]] = left_pose_aa
        state[:, sl["left_mano_betas"][0]:sl["left_mano_betas"][1]] = left_betas
        state[:, sl["right_wrist_transl_cam"][0]:sl["right_wrist_transl_cam"][1]] = right_transl_cam
        state[:, sl["right_wrist_orient_cam_aa"][0]:sl["right_wrist_orient_cam_aa"][1]] = right_orient_cam_aa
        state[:, sl["right_mano_pose_aa"][0]:sl["right_mano_pose_aa"][1]] = right_pose_aa
        state[:, sl["right_mano_betas"][0]:sl["right_mano_betas"][1]] = right_betas

        # 4. State mask: per-frame, [left_kept, right_kept].
        state_mask = np.stack(
            [ep.pred_kept[HAND_LEFT], ep.pred_kept[HAND_RIGHT]], axis=1
        ).astype(bool)

        # 5. Per-frame rows.
        ep_global_start = self._next_global_index
        extrinsics_flat = ep.extrinsics_w2c.reshape(T, EXTRINSICS_FLAT_DIM).astype(np.float32)
        fov_row = np.array(ep.intrinsics_fov, dtype=np.float32)

        for t in range(T):
            row = {
                "index": int(ep_global_start + t),
                "episode_index": int(ep.episode_index),
                "frame_index": int(t),
                "timestamp": float(t / self.fps),
                "task_index": int(task_index),
                "main_type": int(ep.main_type),
                "observation.state": state[t].tolist(),
                "state_mask": state_mask[t].tolist(),
                "fov": fov_row.tolist(),
                "extrinsics_w2c": extrinsics_flat[t].tolist(),
                "left_transl_world": left_transl_world[t].tolist(),
                "left_orient_world": left_orient_world_rotmat[t].tolist(),
                "left_hand_pose": left_pose_flat[t].tolist(),
                "left_kept": bool(ep.pred_kept[HAND_LEFT, t]),
                "left_seg_start": -1,
                "left_seg_end": -1,
                "right_transl_world": right_transl_world[t].tolist(),
                "right_orient_world": right_orient_world_rotmat[t].tolist(),
                "right_hand_pose": right_pose_flat[t].tolist(),
                "right_kept": bool(ep.pred_kept[HAND_RIGHT, t]),
                "right_seg_start": -1,
                "right_seg_end": -1,
                "action_label": ep.action_label,
                "action_score": float(ep.action_score),
            }
            self._data_rows.append(row)

        self._state_stats.update(state)
        # Action stats placeholder when disabled — pass a zeros block of correct shape.
        if self._action_stats.enabled:
            self._action_stats.update(np.zeros((T, ACTION_DIM), dtype=np.float32))

        # 6. Episode video re-encode via ffmpeg.
        video_relpath = (
            f"videos/{self.cfg.video_key}/chunk-000/"
            f"episode_{ep.episode_index:06d}.mp4"
        )
        video_path = self.root / video_relpath
        height, width = self._encode_episode_video(ep, video_path)
        self._video_files.append(video_relpath)
        if self._video_size is None:
            self._video_size = (height, width)

        # 7. Episode index row.
        ep_global_end = ep_global_start + T
        self._episodes.append(
            {
                "episode_index": int(ep.episode_index),
                "length": int(T),
                "tasks": [ep.task_text],
                "dataset_from_index": int(ep_global_start),
                "dataset_to_index": int(ep_global_end),
                "data/chunk_index": 0,
                "data/file_index": int(ep.episode_index),
                "meta/episodes/chunk_index": 0,
                "meta/episodes/file_index": 0,
                f"videos/{self.cfg.video_key}/chunk_index": 0,
                f"videos/{self.cfg.video_key}/file_index": int(ep.episode_index),
                f"videos/{self.cfg.video_key}/from_timestamp": 0.0,
                f"videos/{self.cfg.video_key}/to_timestamp": float(T / self.fps),
            }
        )
        self._next_global_index = ep_global_end
        self._total_frames += T

    def _encode_episode_video(self, ep: EpisodeInput, dst: Path) -> tuple[int, int]:
        t_start = ep.frame_start / ep.source_fps
        duration = ep.length / ep.source_fps
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(".partial.mp4")
        cmd = [
            str(self._ffmpeg.ffmpeg),
            "-y",
            "-loglevel", "error",
            "-ss", f"{t_start:.6f}",
            "-i", str(ep.source_video_path),
            "-t", f"{duration:.6f}",
            "-vf", f"fps={self.fps}",
            "-c:v", "libx264",
            "-preset", self.cfg.video_preset,
            "-crf", str(self.cfg.video_crf),
            "-pix_fmt", self.cfg.video_pix_fmt,
            "-an",
            str(tmp),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        tmp.replace(dst)
        # ffprobe to fill height/width for info.json.
        probe = subprocess.run(
            [
                str(self._ffmpeg.ffprobe),
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height",
                "-of", "json",
                str(dst),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        info = json.loads(probe.stdout)
        stream = info.get("streams", [{}])[0]
        return int(stream.get("height", 0)), int(stream.get("width", 0))

    def _write_data_shard(self) -> None:
        if not self._data_rows:
            return
        # All rows go to a single chunk-000/file-000.parquet for now; sharding
        # is a future iteration controlled by cfg.rows_per_shard.
        shard_path = self.root / "data" / "chunk-000" / "file-000.parquet"
        # Build column arrays by name to preserve schema order.
        schema = build_data_schema()
        columns: dict[str, list[Any]] = {f.name: [] for f in schema}
        for row in self._data_rows:
            for f in schema:
                columns[f.name].append(row[f.name])
        table = pa.table(columns, schema=schema)
        pq.write_table(table, shard_path, compression="zstd")
        self._data_files.append(str(shard_path.relative_to(self.root)))

    def _write_episodes_index(self) -> None:
        if not self._episodes:
            return
        schema = build_episodes_schema(self.cfg.video_key)
        columns: dict[str, list[Any]] = {f.name: [] for f in schema}
        for ep in self._episodes:
            for f in schema:
                columns[f.name].append(ep[f.name])
        table = pa.table(columns, schema=schema)
        dst = self.root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        dst.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, dst, compression="zstd")

    def _write_tasks_table(self) -> None:
        schema = build_tasks_schema()
        text_to_idx = sorted(self._tasks.items(), key=lambda kv: kv[1])
        columns = {
            "task_index": [idx for _, idx in text_to_idx],
            "task": [text for text, _ in text_to_idx],
        }
        table = pa.table(columns, schema=schema)
        dst = self.root / "meta" / "tasks.parquet"
        pq.write_table(table, dst, compression="zstd")

    def _write_info_json(self) -> None:
        height, width = self._video_size or (0, 0)
        info = build_info_dict(
            codebase_version=self.cfg.codebase_version,
            robot_type=self.cfg.robot_type,
            fps=self.fps,
            total_episodes=len(self._episodes),
            total_frames=self._total_frames,
            total_tasks=len(self._tasks),
            total_videos=len(self._video_files),
            chunks_size=self.cfg.chunks_size,
            video_key=self.cfg.video_key,
            video_height=height,
            video_width=width,
            video_codec=self.cfg.video_codec,
            video_pix_fmt=self.cfg.video_pix_fmt,
        )
        with (self.root / "meta" / "info.json").open("w", encoding="utf-8") as fh:
            json.dump(info, fh, ensure_ascii=False, indent=2, sort_keys=True)
            fh.write("\n")

    def _write_stats_json(self) -> None:
        stats = {
            "observation.state": self._state_stats.finalize(),
            "action": self._action_stats.finalize(),
        }
        with (self.root / "meta" / "stats.json").open("w", encoding="utf-8") as fh:
            json.dump(stats, fh, ensure_ascii=False, indent=2, sort_keys=True)
            fh.write("\n")

    def finalize(self) -> DatasetReport:
        if not self._episodes:
            raise RuntimeError("LeRobotV3DatasetWriter.finalize(): no episodes were added.")
        self._write_data_shard()
        self._write_episodes_index()
        self._write_tasks_table()
        self._write_info_json()
        self._write_stats_json()
        return DatasetReport(
            root=self.root,
            total_episodes=len(self._episodes),
            total_frames=self._total_frames,
            total_tasks=len(self._tasks),
            total_videos=len(self._video_files),
            data_files=list(self._data_files),
            video_files=list(self._video_files),
            stats={
                "state_count": self._state_stats._count,
                "action_count": self._action_stats._count,
            },
        )


__all__ = ["EpisodeInput", "DatasetReport", "LeRobotV3DatasetWriter"]
