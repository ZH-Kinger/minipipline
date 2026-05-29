from __future__ import annotations

import json
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config import HandPoseConfig
from .action_seg import atomic_split_for_label
from .cleaning import clean_pred_result
from .clip_merge import merge_clip_predictions
from .clip_split import plan_clips
from .models import (
    GeoCalibBackend,
    HaWoRStage1Backend,
    HaWoRStage2Backend,
    MegaSAMBackend,
    MoGe2Backend,
    build_geocalib_backend,
    build_hawor_s1_backend,
    build_hawor_s2_backend,
    build_megasam_backend,
    build_moge2_backend,
)
from .schemas import (
    AtomicAction,
    CameraIntrinsics,
    CameraTrajectory,
    ClipPlan,
    LabelSegment,
    MergedPrediction,
    PredResult,
    VideoMeta,
)
from .video_io import probe_video


@dataclass
class HandPoseRunResult:
    video: VideoMeta
    intrinsics: CameraIntrinsics
    merged: MergedPrediction
    clip_plans: list[ClipPlan]
    label_segments: list[LabelSegment]
    atomic_actions: list[AtomicAction]
    stage_timings: dict[str, float] = field(default_factory=dict)


def _process_clip(
    clip: ClipPlan,
    video: VideoMeta,
    intr_future: Future[CameraIntrinsics],
    cfg: HandPoseConfig,
    geo: GeoCalibBackend,
    moge: MoGe2Backend,
    s1: HaWoRStage1Backend,
    megasam: MegaSAMBackend,
    s2: HaWoRStage2Backend,
    executor: ThreadPoolExecutor,
) -> tuple[CameraTrajectory, PredResult]:
    """Process a single clip through the 4-stage GPU chain.

    Stages 4.2 (depth) and 4.3 (S1 detection) run concurrently. Stage 4.4
    (SLAM) waits for both, then 4.5 (S2) waits for SLAM + S1.
    """
    intr = intr_future.result()

    depth_fut = executor.submit(
        moge.infer, video, intr, clip.frame_start, clip.frame_end, clip.clip_idx
    )
    s1_fut = executor.submit(
        s1.infer, video, intr, clip.frame_start, clip.frame_end, clip.clip_idx
    )
    depth = depth_fut.result()
    s1_out = s1_fut.result()

    traj = megasam.infer(video, intr, depth, s1_out, clip.frame_start, clip.frame_end, clip.clip_idx)
    pred = s2.infer(video, s1_out, traj, clip.frame_start, clip.frame_end, clip.clip_idx)
    return traj, pred


def run_handpose_pipeline(
    nir_dir: str | Path,
    cfg: HandPoseConfig | None = None,
    label_segments: list[LabelSegment] | None = None,
) -> HandPoseRunResult:
    """Run Layer 2 on a NIR session.

    `nir_dir` points at a single NIR session directory (e.g.
    `.../processed/nir/00010a33-.../`). Reads `media_paths.json` to find
    the source RGB video. If `label_segments` is None, a default segment
    covering the entire video is constructed from `task.json`.
    """
    cfg = cfg or HandPoseConfig()
    nir_dir = Path(nir_dir)
    media_paths = json.loads((nir_dir / "media_paths.json").read_text(encoding="utf-8"))
    rgb_path = Path(media_paths["media"]["rgb"])

    timings: dict[str, float] = {}

    t = time.perf_counter()
    video = probe_video(rgb_path)
    timings["probe"] = time.perf_counter() - t

    # Default label segment (single segment covering the whole video).
    if label_segments is None:
        task = json.loads((nir_dir / "task.json").read_text(encoding="utf-8"))
        task_text = task.get("task_text") or task.get("task_info", {}).get("name", "")
        label_segments = [
            LabelSegment(
                label_id="L0",
                action_text=task_text,
                t_start_s=0.0,
                t_end_s=video.duration_s,
                hand_hint="both",
                frame_start=0,
                frame_end=video.n_frames,
            )
        ]

    t = time.perf_counter()
    clips = plan_clips(video, label_segments, cfg)
    timings["clip_split"] = time.perf_counter() - t

    # Real-ingest fast path: when the handpose backend is "real_ingest", skip
    # the entire 5-stage mock/GPU chain and assemble MergedPrediction directly
    # from the NIR ground truth (real keypoints + head 6DOF + intrinsics).
    from ..config import settings as _settings

    try:
        _hp_backend = _settings.backend("hawor_s2", fallback_key="HANDPOSE_BACKEND")
    except Exception:
        _hp_backend = "mock"
    if _hp_backend == "real_ingest":
        from .real_ingest import build_merged_from_nir

        t = time.perf_counter()
        merged = build_merged_from_nir(nir_dir, video)
        timings["real_ingest"] = time.perf_counter() - t

        t = time.perf_counter()
        atomic = []
        for idx, label in enumerate(label_segments):
            atomic.extend(
                atomic_split_for_label(label, merged.pred_trans, merged.pred_valid, cfg, idx)
            )
        timings["action_seg"] = time.perf_counter() - t

        return HandPoseRunResult(
            video=video,
            intrinsics=merged.intrinsics,
            merged=merged,
            clip_plans=clips,
            label_segments=label_segments,
            atomic_actions=atomic,
            stage_timings=timings,
        )

    geo = build_geocalib_backend(cfg)
    moge = build_moge2_backend(cfg)
    s1 = build_hawor_s1_backend(cfg)
    megasam = build_megasam_backend(cfg)
    s2 = build_hawor_s2_backend(cfg)

    t = time.perf_counter()
    if cfg.executor_mode == "sequential":
        intrinsics = geo.infer(video)
        clip_results: list[tuple[CameraTrajectory, PredResult]] = []
        for clip in clips:
            depth = moge.infer(video, intrinsics, clip.frame_start, clip.frame_end, clip.clip_idx)
            s1_out = s1.infer(video, intrinsics, clip.frame_start, clip.frame_end, clip.clip_idx)
            traj = megasam.infer(video, intrinsics, depth, s1_out, clip.frame_start, clip.frame_end, clip.clip_idx)
            pred = s2.infer(video, s1_out, traj, clip.frame_start, clip.frame_end, clip.clip_idx)
            clip_results.append((traj, pred))
    else:
        with ThreadPoolExecutor(max_workers=cfg.max_workers) as ex:
            intr_fut = ex.submit(geo.infer, video)
            clip_results = [
                _process_clip(clip, video, intr_fut, cfg, geo, moge, s1, megasam, s2, ex)
                for clip in clips
            ]
            intrinsics = intr_fut.result()
    timings["models"] = time.perf_counter() - t

    trajectories = [t for t, _ in clip_results]
    predictions = [p for _, p in clip_results]

    t = time.perf_counter()
    trans, rot, pose, betas, valid, merged_traj = merge_clip_predictions(
        video, clips, predictions, trajectories
    )
    timings["clip_merge"] = time.perf_counter() - t

    t = time.perf_counter()
    trans_c, rot_c, pose_c, betas_c, valid_c, kept = clean_pred_result(
        trans, rot, pose, betas, valid, cfg
    )
    timings["clean"] = time.perf_counter() - t

    merged = MergedPrediction(
        video_id=video.video_id,
        pred_trans=trans_c,
        pred_rot=rot_c,
        pred_hand_pose=pose_c,
        pred_betas=betas_c,
        pred_valid=valid_c,
        pred_kept=kept,
        trajectory=merged_traj,
        intrinsics=intrinsics,
    )

    t = time.perf_counter()
    atomic: list[AtomicAction] = []
    for idx, label in enumerate(label_segments):
        atomic.extend(atomic_split_for_label(label, trans_c, valid_c, cfg, idx))
    timings["action_seg"] = time.perf_counter() - t

    return HandPoseRunResult(
        video=video,
        intrinsics=intrinsics,
        merged=merged,
        clip_plans=clips,
        label_segments=label_segments,
        atomic_actions=atomic,
        stage_timings=timings,
    )


__all__ = ["HandPoseRunResult", "run_handpose_pipeline"]
