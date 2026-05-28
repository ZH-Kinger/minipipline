from __future__ import annotations

from typing import Any

import pyarrow as pa


# ---------------------------------------------------------------------------
# Dimensional constants
# ---------------------------------------------------------------------------


HAND_STATE_DIM = 61          # 3 transl + 3 axis-angle orient + 45 MANO pose + 10 betas
HAND_ACTION_DIM = 51         # 3 Δt + 3 Δr + 45 next-frame MANO pose
STATE_DIM = HAND_STATE_DIM * 2  # 122
ACTION_DIM = HAND_ACTION_DIM * 2  # 102

MANO_JOINT_COUNT = 15
HAND_POSE_ROTMAT_DIM = MANO_JOINT_COUNT * 9  # 135
HAND_POSE_AXISANGLE_DIM = MANO_JOINT_COUNT * 3  # 45
BETAS_DIM = 10
EXTRINSICS_FLAT_DIM = 16  # 4×4 row-major
ORIENT_ROTMAT_DIM = 9  # 3×3 row-major
WRIST_TRANSL_DIM = 3
FOV_DIM = 2
STATE_MASK_DIM = 2


# State vector layout: [left 61 | right 61]. Each slice is (start, end_exclusive).
STATE_LAYOUT: dict[str, tuple[int, int]] = {
    "left_wrist_transl_cam":    (0, 3),
    "left_wrist_orient_cam_aa": (3, 6),
    "left_mano_pose_aa":        (6, 51),
    "left_mano_betas":           (51, 61),
    "right_wrist_transl_cam":   (61, 64),
    "right_wrist_orient_cam_aa":(64, 67),
    "right_mano_pose_aa":       (67, 112),
    "right_mano_betas":         (112, 122),
}


# Action vector layout: [left 51 | right 51].
ACTION_LAYOUT: dict[str, tuple[int, int]] = {
    "left_wrist_delta_transl":  (0, 3),
    "left_wrist_delta_orient":  (3, 6),
    "left_next_mano_pose_aa":   (6, 51),
    "right_wrist_delta_transl": (51, 54),
    "right_wrist_delta_orient": (54, 57),
    "right_next_mano_pose_aa":  (57, 102),
}


from ..config.lerobot import VIDEO_KEY_DEFAULT  # canonical home in config layer


# ---------------------------------------------------------------------------
# Parquet schemas
# ---------------------------------------------------------------------------


def _vec(n: int) -> pa.DataType:
    """Variable-length list of float32 used to store a fixed-shape vector.

    LeRobot's downstream dataloaders treat these as numpy arrays of length n;
    info.json records the shape so consumers can reshape correctly.
    """
    return pa.list_(pa.float32())


def _vec_bool(_n: int) -> pa.DataType:
    return pa.list_(pa.bool_())


def build_data_schema() -> pa.Schema:
    """Per-frame parquet schema (one row per frame within an episode)."""
    fields = [
        pa.field("index", pa.int64()),
        pa.field("episode_index", pa.int64()),
        pa.field("frame_index", pa.int64()),
        pa.field("timestamp", pa.float32()),
        pa.field("task_index", pa.int64()),
        pa.field("main_type", pa.int64()),
        pa.field("observation.state", _vec(STATE_DIM)),
        pa.field("state_mask", _vec_bool(STATE_MASK_DIM)),
        pa.field("fov", _vec(FOV_DIM)),
        pa.field("extrinsics_w2c", _vec(EXTRINSICS_FLAT_DIM)),
        pa.field("left_transl_world", _vec(WRIST_TRANSL_DIM)),
        pa.field("left_orient_world", _vec(ORIENT_ROTMAT_DIM)),
        pa.field("left_hand_pose", _vec(HAND_POSE_ROTMAT_DIM)),
        pa.field("left_kept", pa.bool_()),
        pa.field("left_seg_start", pa.int64()),
        pa.field("left_seg_end", pa.int64()),
        pa.field("right_transl_world", _vec(WRIST_TRANSL_DIM)),
        pa.field("right_orient_world", _vec(ORIENT_ROTMAT_DIM)),
        pa.field("right_hand_pose", _vec(HAND_POSE_ROTMAT_DIM)),
        pa.field("right_kept", pa.bool_()),
        pa.field("right_seg_start", pa.int64()),
        pa.field("right_seg_end", pa.int64()),
        # Layer 1.5 annotation fields (constant across an episode's frames).
        pa.field("action_label", pa.string()),
        pa.field("action_score", pa.float32()),
    ]
    return pa.schema(fields)


def build_episodes_schema(video_key: str = VIDEO_KEY_DEFAULT) -> pa.Schema:
    """Episodes index parquet schema (one row per episode)."""
    fields = [
        pa.field("episode_index", pa.int64()),
        pa.field("length", pa.int64()),
        pa.field("tasks", pa.list_(pa.string())),
        pa.field("dataset_from_index", pa.int64()),
        pa.field("dataset_to_index", pa.int64()),
        pa.field("data/chunk_index", pa.int64()),
        pa.field("data/file_index", pa.int64()),
        pa.field("meta/episodes/chunk_index", pa.int64()),
        pa.field("meta/episodes/file_index", pa.int64()),
        pa.field(f"videos/{video_key}/chunk_index", pa.int64()),
        pa.field(f"videos/{video_key}/file_index", pa.int64()),
        pa.field(f"videos/{video_key}/from_timestamp", pa.float64()),
        pa.field(f"videos/{video_key}/to_timestamp", pa.float64()),
    ]
    return pa.schema(fields)


def build_tasks_schema() -> pa.Schema:
    """Task-text deduplication table (one row per unique task prompt)."""
    return pa.schema(
        [
            pa.field("task_index", pa.int64()),
            pa.field("task", pa.string()),
        ]
    )


# ---------------------------------------------------------------------------
# info.json builder
# ---------------------------------------------------------------------------


def build_info_dict(
    *,
    codebase_version: str,
    robot_type: str,
    fps: float,
    total_episodes: int,
    total_frames: int,
    total_tasks: int,
    total_videos: int,
    chunks_size: int,
    video_key: str,
    video_height: int,
    video_width: int,
    video_codec: str,
    video_pix_fmt: str,
) -> dict[str, Any]:
    return {
        "codebase_version": codebase_version,
        "robot_type": robot_type,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": total_tasks,
        "total_videos": total_videos,
        "chunks_size": chunks_size,
        "fps": fps,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/episode_{episode_index:06d}.mp4",
        "splits": {"train": f"0:{total_episodes}"},
        "features": {
            "observation.state": {"dtype": "float32", "shape": [STATE_DIM]},
            "action": {"dtype": "float32", "shape": [ACTION_DIM]},
            "state_mask": {"dtype": "bool", "shape": [STATE_MASK_DIM]},
            "fov": {"dtype": "float32", "shape": [FOV_DIM]},
            "extrinsics_w2c": {"dtype": "float32", "shape": [EXTRINSICS_FLAT_DIM]},
            "left_transl_world": {"dtype": "float32", "shape": [WRIST_TRANSL_DIM]},
            "left_orient_world": {"dtype": "float32", "shape": [ORIENT_ROTMAT_DIM]},
            "left_hand_pose": {"dtype": "float32", "shape": [HAND_POSE_ROTMAT_DIM]},
            "left_kept": {"dtype": "bool", "shape": [1]},
            "left_seg_start": {"dtype": "int64", "shape": [1]},
            "left_seg_end": {"dtype": "int64", "shape": [1]},
            "right_transl_world": {"dtype": "float32", "shape": [WRIST_TRANSL_DIM]},
            "right_orient_world": {"dtype": "float32", "shape": [ORIENT_ROTMAT_DIM]},
            "right_hand_pose": {"dtype": "float32", "shape": [HAND_POSE_ROTMAT_DIM]},
            "right_kept": {"dtype": "bool", "shape": [1]},
            "right_seg_start": {"dtype": "int64", "shape": [1]},
            "right_seg_end": {"dtype": "int64", "shape": [1]},
            "main_type": {"dtype": "int64", "shape": [1]},
            "frame_index": {"dtype": "int64", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "index": {"dtype": "int64", "shape": [1]},
            "task_index": {"dtype": "int64", "shape": [1]},
            "timestamp": {"dtype": "float32", "shape": [1]},
            "action_label": {"dtype": "string", "shape": [1]},
            "action_score": {"dtype": "float32", "shape": [1]},
            video_key: {
                "dtype": "video",
                "shape": [video_height, video_width, 3],
                "info": {
                    "video.fps": fps,
                    "video.height": video_height,
                    "video.width": video_width,
                    "video.codec": video_codec,
                    "video.pix_fmt": video_pix_fmt,
                },
            },
        },
    }
