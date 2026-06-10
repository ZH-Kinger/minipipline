"""Command-line interface.

Main entry point is the shortcut form::

    mmpipe <session_dir> [output_dir]

which is equivalent to ``mmpipe run-all <session_dir> <output_dir>``. If the
first positional argument is not a recognised subcommand and not a flag, it
is treated as a session directory.

Output defaults to ``$MMPIPE_OUTPUT_ROOT/<session_basename>/`` if that env var
is set (handy when mounting OSS / cloud storage), otherwise
``./output/<session_basename>/``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import __version__
from .config import PipelineConfig
from .pipeline import run_pipeline


# Subcommand names recognised at the top level. Anything else as argv[0] is
# treated as the implicit `run-all <session>` shortcut.
_KNOWN_COMMANDS = {
    "run-all",
    "ingest",
    "handpose",
    "annotate",
    "lerobot",
    "retarget-check",
    "validate",
    "lerobot-validate",  # legacy alias
    "doctor",
    "info",
    "visualize",
    "quality-report",
    "run",                # legacy ETL
    "init-config",        # legacy ETL
}


# ---------------------------------------------------------------------------
# Subparser definitions
# ---------------------------------------------------------------------------


def _add_run_all(sp: argparse._SubParsersAction) -> None:
    p = sp.add_parser(
        "run-all",
        help="Run all 3 layers on a session (Layer 1 -> 2 -> 3 -> validate).",
    )
    p.add_argument("session_dir", type=Path, help="ITW session directory.")
    p.add_argument(
        "output_root",
        type=Path,
        nargs="?",
        default=None,
        help=(
            "Output directory. Defaults to $MMPIPE_OUTPUT_ROOT/<session_basename>/ "
            "if MMPIPE_OUTPUT_ROOT is set, otherwise ./output/<session_basename>/."
        ),
    )
    p.add_argument("--itw-config", type=Path)
    p.add_argument("--handpose-config", type=Path)
    p.add_argument("--lerobot-config", type=Path)
    p.add_argument(
        "--force", action="store_true",
        help="Re-process sessions even if <out>/<session>/lerobot_dataset/meta/info.json "
             "already exists (default: skip such sessions for crash-resume friendliness).",
    )
    p.add_argument(
        "--no-quality-report", action="store_true",
        help="Skip the auto quality-report run at the end (default: ON; writes quality_report.json).",
    )
    p.add_argument(
        "--no-quality-psnr", action="store_true",
        help="In the auto quality-report, skip per-episode PSNR/SSIM (faster: ~30s vs ~200s on 40 sess).",
    )
    p.add_argument(
        "--no-cache", action="store_true",
        help="Disable persistent SQLite cache for DashScope responses "
             "(default: enabled at <output_root>/.annotation_cache.db).",
    )
    p.add_argument(
        "--parallel-sessions", type=int, default=1, metavar="N",
        help="Run N sessions concurrently via ProcessPool (default 1 = serial). "
             "N=4 is a good batch default; goes through DashScope ~3-4x faster. "
             "On Windows the first batch waits ~10s for process spawn.",
    )


def _add_ingest(sp: argparse._SubParsersAction) -> None:
    p = sp.add_parser("ingest", help="Layer 1 only: ITW session -> NIR + tar.")
    p.add_argument("session_dir", type=Path)
    p.add_argument("output_dir", type=Path, nargs="?", default=None)
    p.add_argument("--config", type=Path)
    p.add_argument("--no-pack", action="store_true")


def _add_handpose(sp: argparse._SubParsersAction) -> None:
    p = sp.add_parser("handpose", help="Layer 2 only: NIR -> MergedPrediction (mock).")
    p.add_argument("nir_dir", type=Path)
    p.add_argument("--config", type=Path)


def _add_annotate(sp: argparse._SubParsersAction) -> None:
    p = sp.add_parser(
        "annotate",
        help="Layer 1.5 only: NIR -> language + action labels + frame quality.",
    )
    p.add_argument("nir_dir", type=Path)
    p.add_argument("--config", type=Path)


def _add_lerobot(sp: argparse._SubParsersAction) -> None:
    p = sp.add_parser(
        "lerobot",
        help="Layer 2+3: NIR -> LeRobot v3 dataset (skips Layer 1).",
    )
    p.add_argument("nir_dir", type=Path)
    p.add_argument("dataset_root", type=Path, nargs="?", default=None)
    p.add_argument("--config", type=Path)
    p.add_argument("--handpose-config", type=Path)


def _add_retarget_check(sp: argparse._SubParsersAction) -> None:
    p = sp.add_parser(
        "retarget-check",
        help="Layer 2.5: retarget human hands -> Wuji robot and report fit quality (mm/%).",
    )
    p.add_argument("nir_dir", type=Path, help="A NIR directory (Layer 1 output).")
    p.add_argument("--handpose-config", type=Path)


def _add_validate(sp: argparse._SubParsersAction) -> None:
    for name in ("validate", "lerobot-validate"):
        p = sp.add_parser(name, help="Read back a LeRobot v3 dataset and verify.")
        p.add_argument("dataset_root", type=Path)
        p.add_argument("--config", type=Path)


def _add_doctor(sp: argparse._SubParsersAction) -> None:
    sp.add_parser("doctor", help="Check environment (ffmpeg, deps, backends, weights).")


def _add_info(sp: argparse._SubParsersAction) -> None:
    p = sp.add_parser("info", help="Print summary of a LeRobot v3 dataset.")
    p.add_argument("dataset_root", type=Path)


def _add_visualize(sp: argparse._SubParsersAction) -> None:
    p = sp.add_parser(
        "visualize",
        help="Render keypoint-overlay (and optional depth) MP4s for human check.",
    )
    p.add_argument("dataset_root", type=Path, help="A packed lerobot_dataset directory.")
    p.add_argument("--episode", type=int, action="append", default=None,
                   help="Episode index to render (repeatable). Default: first 3.")
    p.add_argument("--depth", action="store_true",
                   help="Also render the depth stream as a side-by-side colormap.")
    p.add_argument("--compare", action="store_true",
                   help="Prepend the raw (unannotated) RGB panel for side-by-side comparison.")
    p.add_argument("--pointcloud", action="store_true",
                   help="Render a 3D depth point-cloud + hand-keypoint world view (PNG) instead of the overlay MP4.")
    p.add_argument("--mano", action="store_true",
                   help="Render the MANO hand-mesh overlay (PNG). Needs torch+smplx and the "
                        "MANO model files in $MMPIPE_MANO_DIR (default ./models/mano/).")
    p.add_argument("--all", action="store_true", dest="combined",
                   help="Render one synced 2x2 MP4: raw | keypoints+axes / MANO mesh | depth. "
                        "Everything in a single VLC window. Needs MANO (see --mano).")
    p.add_argument("--3d", action="store_true", dest="threed",
                   help="GPU-render a crisp 3D video (Open3D/EGL): scene point-cloud + MANO "
                        "hand meshes from an orbiting camera. Needs open3d + MANO.")
    p.add_argument("--world", action="store_true", dest="world",
                   help="One frame-locked MP4 with EVERY view in a single VLC window (3x2 grid): "
                        "raw | keypoints+axes | depth / MANO mesh | 3D world | 3D keypoint fit. "
                        "Needs open3d + MANO.")
    p.add_argument("--out", type=Path, default=None,
                   help="Output dir (default: <dataset_root>/viz).")


def _add_quality_report(sp: argparse._SubParsersAction) -> None:
    p = sp.add_parser(
        "quality-report",
        help="4-dim data quality report (efficiency/video/annotation/compliance) over an output root.",
    )
    p.add_argument("output_root", type=Path, nargs="?", default=None,
                   help="Pipeline output root. Defaults to $MMPIPE_OUTPUT_ROOT or ./output.")
    p.add_argument("--json", type=Path, default=None,
                   help="Write JSON report to this path (default: <output_root>/quality_report.json).")
    p.add_argument("--json-only", action="store_true",
                   help="Skip human-readable summary; only write JSON.")
    p.add_argument("--no-psnr", action="store_true",
                   help="Skip per-episode PSNR/SSIM (much faster but loses video quality metric).")
    p.add_argument("--parallelism", type=int, default=4,
                   help="Per-session parallel workers (default 4).")


def _add_legacy(sp: argparse._SubParsersAction) -> None:
    # Kept for backward compatibility; not advertised in main --help.
    p = sp.add_parser("run", help=argparse.SUPPRESS)
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--config", type=Path)
    q = sp.add_parser("init-config", help=argparse.SUPPRESS)
    q.add_argument("--output", required=True, type=Path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mmpipe",
        description=(
            "Multimodal human-demonstration data pipeline.\n\n"
            "Quick start:\n"
            "  mmpipe <session_dir>            run all 3 layers, output to ./output/\n"
            "  mmpipe <session_dir> <out>      same with explicit output dir\n"
            "  mmpipe doctor                   check environment\n"
            "  mmpipe info <dataset_root>      summarise a packed dataset\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sp = parser.add_subparsers(dest="command", required=True, metavar="<command>")
    _add_run_all(sp)
    _add_doctor(sp)
    _add_info(sp)
    _add_validate(sp)
    _add_ingest(sp)
    _add_handpose(sp)
    _add_annotate(sp)
    _add_lerobot(sp)
    _add_retarget_check(sp)
    _add_visualize(sp)
    _add_quality_report(sp)
    _add_legacy(sp)
    return parser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _print_json(payload: dict) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str))


def _resolve_output_root() -> Path:
    """Resolve the parent output directory.

    Honours ``MMPIPE_OUTPUT_ROOT`` so ECS/DSW deployments mounting OSS for
    output can set it once in ``.env``. Falls back to ``./output`` otherwise.
    Per-session subdirectories live under this root.
    """

    base_env = os.environ.get("MMPIPE_OUTPUT_ROOT")
    return Path(base_env).expanduser() if base_env else Path("output")


def _default_output_for(session_dir: Path) -> Path:
    """Default per-session output directory: ``<root>/<session_basename>/``."""

    return _resolve_output_root() / session_dir.resolve().name


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------


def _cmd_init_config(args: argparse.Namespace) -> int:
    config = PipelineConfig()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(config.__dict__, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"Wrote config: {args.output}")
    return 0


def _cmd_run_legacy(args: argparse.Namespace) -> int:
    config = PipelineConfig.from_file(args.config)
    report = run_pipeline(args.input, args.output, config)
    _print_json(report)
    return 0


def _cmd_ingest(args: argparse.Namespace) -> int:
    from .itw import ITWConfig, run_itw_ingest

    if args.output_dir is None:
        args.output_dir = _default_output_for(args.session_dir)
    cfg = ITWConfig.from_file(args.config)
    result = run_itw_ingest(args.session_dir, args.output_dir, cfg, pack_tar=not args.no_pack)
    summary = {
        "session_id": result.session.session_id,
        "nir_dir": str(result.session.root),
        "frames": result.session.frame_count,
        "duration_s": result.session.duration_s,
        "stage_timings_ms": {k: round(v * 1000, 1) for k, v in result.stage_timings.items()},
        "calibration_ok": result.report.calibration_ok,
        "validation_errors": result.report.validation.error_count,
        "validation_warnings": result.report.validation.warning_count,
        "gap_anomalies": len(result.report.alignment.gap_anomalies),
        "pack_shard": str(result.pack.shard_path) if result.pack else None,
    }
    _print_json(summary)
    return 0


def _cmd_handpose(args: argparse.Namespace) -> int:
    from .handpose import HandPoseConfig, run_handpose_pipeline

    cfg = HandPoseConfig.from_file(args.config)
    result = run_handpose_pipeline(args.nir_dir, cfg)
    summary = {
        "video_id": result.video.video_id,
        "frames": result.video.n_frames,
        "fps": result.video.fps,
        "stage_timings_ms": {k: round(v * 1000, 1) for k, v in result.stage_timings.items()},
        "clip_plans": len(result.clip_plans),
        "label_segments": len(result.label_segments),
        "atomic_actions": len(result.atomic_actions),
        "merged_valid_frames": int(result.merged.pred_valid.sum()),
        "merged_kept_frames": int(result.merged.pred_kept.sum()),
    }
    _print_json(summary)
    return 0


def _cmd_annotate(args: argparse.Namespace) -> int:
    from .annotate import AnnotateConfig, run_annotate_pipeline

    cfg = AnnotateConfig.from_file(args.config)
    result = run_annotate_pipeline(args.nir_dir, cfg)
    summary = {
        "nir_dir": result.nir_dir,
        "clip_count": len(result.clips),
        "frame_quality_rows": len(result.frame_quality),
        "frames_kept": sum(1 for r in result.frame_quality if r.kept),
        "language_source": result.clips[0].language_source if result.clips else None,
        "actions_source": result.clips[0].actions_source if result.clips else None,
        "stage_timings_ms": {k: round(v * 1000, 1) for k, v in result.stage_timings.items()},
        "first_clip_example": (
            {
                "language_text": result.clips[0].language_text,
                "action_label": result.clips[0].action_label,
                "action_score": round(result.clips[0].action_score, 3),
            }
            if result.clips
            else None
        ),
    }
    _print_json(summary)
    return 0


def _cmd_lerobot(args: argparse.Namespace) -> int:
    from .handpose import HandPoseConfig, run_handpose_pipeline
    from .lerobot_v3 import (
        LeRobotConfig,
        LeRobotV3DatasetWriter,
        build_episode_inputs,
    )

    if args.dataset_root is None:
        args.dataset_root = _default_output_for(args.nir_dir) / "lerobot_dataset"
    hp_cfg = HandPoseConfig.from_file(args.handpose_config)
    lr_cfg = LeRobotConfig.from_file(args.config)

    nir_dir = args.nir_dir
    handpose_result = run_handpose_pipeline(nir_dir, hp_cfg)

    media_paths = json.loads((nir_dir / "media_paths.json").read_text(encoding="utf-8"))
    source_video = Path(media_paths["media"]["rgb"])

    episodes = build_episode_inputs(
        handpose_result.video,
        handpose_result.merged,
        handpose_result.atomic_actions,
        source_video,
    )

    args.dataset_root.mkdir(parents=True, exist_ok=True)
    with LeRobotV3DatasetWriter(args.dataset_root, lr_cfg, fps=handpose_result.video.fps) as w:
        for ep in episodes:
            idx = w.register_task(ep.task_text)
            w.add_episode(ep, task_index=idx)
        report = w.finalize()

    _print_json(
        {
            "dataset_root": str(report.root),
            "total_episodes": report.total_episodes,
            "total_frames": report.total_frames,
            "total_tasks": report.total_tasks,
            "total_videos": report.total_videos,
            "data_files": report.data_files,
        }
    )
    return 0


def _cmd_retarget_check(args: argparse.Namespace) -> int:
    from .handpose import HandPoseConfig, run_handpose_pipeline
    from .config.retarget import RetargetConfig
    from .retarget import self_check

    cfg = HandPoseConfig.from_file(args.handpose_config)
    result = run_handpose_pipeline(args.nir_dir, cfg)
    m = result.merged
    if m.hand_keypoints_world is None:
        print("No real hand keypoints (mock chain) — retarget needs a real "
              "keypoint source (MMPIPE_HANDPOSE_BACKEND=real_ingest).")
        return 1
    summary = self_check(m.hand_keypoints_world, m.pred_kept, RetargetConfig.from_env())
    _print_json(summary)
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    from .lerobot_v3 import LeRobotConfig, validate_dataset

    cfg = LeRobotConfig.from_file(args.config)
    report = validate_dataset(args.dataset_root, cfg)
    payload = {
        "ok": report.ok,
        "errors": report.error_count,
        "warnings": report.warning_count,
        "issues": [
            {"code": i.code, "severity": i.severity, "message": i.message, "details": i.details}
            for i in report.issues
        ],
    }
    _print_json(payload)
    return 0 if report.ok else 1


def _run_one_session(
    session_dir: Path,
    per_session_output: Path,
    itw_cfg,
    hp_cfg,
    lr_cfg,
    annotate_cfg=None,
) -> bool:
    """Run Layer 1 -> 2 -> 1.5 -> 3 -> validate for a single session.

    Returns ``True`` iff the final validation passes. Prints per-layer status
    lines for visibility.
    """

    from .annotate import AnnotateConfig, run_annotate_pipeline
    from .handpose import run_handpose_pipeline
    from .itw import run_itw_ingest
    from .lerobot_v3 import (
        LeRobotV3DatasetWriter,
        build_episode_inputs,
        validate_dataset,
    )

    if annotate_cfg is None:
        annotate_cfg = AnnotateConfig()

    layer1_root = per_session_output
    layer3_root = per_session_output / "lerobot_dataset"

    from .config import settings

    def _backend_summary(slots: tuple[str, ...], fallback_key: str) -> str:
        seen = []
        for s in slots:
            try:
                seen.append(settings.backend(s, fallback_key=fallback_key))
            except ValueError:
                seen.append("?")
        if all(b == seen[0] for b in seen):
            return seen[0]
        return ",".join(f"{s}={b}" for s, b in zip(slots, seen))

    hp_backends = _backend_summary(
        ("geocalib", "moge2", "hawor_s1", "megasam", "hawor_s2"),
        fallback_key="HANDPOSE_BACKEND",
    )
    an_backends = _backend_summary(
        ("language", "actions", "quality"),
        fallback_key="ANNOTATOR_BACKEND",
    )

    print("  [Layer 1] ingest ...")
    l1 = run_itw_ingest(session_dir, layer1_root, itw_cfg, pack_tar=True)
    print(
        f"    -> {l1.session.frame_count} frames, "
        f"calibration_ok={l1.report.calibration_ok}, "
        f"warnings={l1.report.validation.warning_count}, "
        f"errors={l1.report.validation.error_count}"
    )

    print(f"  [Layer 2] handpose [{hp_backends}] ...")
    l2 = run_handpose_pipeline(l1.session.root, hp_cfg)
    print(
        f"    -> {l2.video.n_frames} frames, {len(l2.atomic_actions)} atomic actions, "
        f"merged_valid={int(l2.merged.pred_valid.sum())}/{l2.video.n_frames * 2}"
    )

    print(f"  [Layer 1.5] annotate [{an_backends}] ...")
    la = run_annotate_pipeline(
        l1.session.root, annotate_cfg,
        atomic_actions=l2.atomic_actions,
        merged_prediction=l2.merged,
    )
    kept = sum(1 for r in la.frame_quality if r.kept)
    cache_note = ""
    if la.cache_stats:
        cs = la.cache_stats
        cache_note = (
            f", cache hits={int(cs['hits'])}/{int(cs['hits']) + int(cs['misses'])} "
            f"({cs['hit_rate'] * 100:.0f}%)"
        )
    print(
        f"    -> {len(la.clips)} clip annotations, "
        f"{kept}/{len(la.frame_quality)} frames kept by quality filter{cache_note}"
    )

    print("  [Layer 3] lerobot v3 pack ...")
    media_paths = json.loads((l1.session.root / "media_paths.json").read_text(encoding="utf-8"))
    source_video = Path(media_paths["media"]["rgb"])
    depth_src = media_paths.get("media", {}).get("depth")
    depth_video = Path(depth_src) if depth_src else None

    # Per-frame auxiliary modalities (IMU + audio contact phase) aligned to the
    # RGB timeline, computed once per video and sliced per-episode downstream.
    import pyarrow.parquet as _pq
    from .itw.multimodal_features import contact_phase_per_frame, imu_per_frame
    _fi = _pq.read_table(l1.session.root / "frame_index.parquet")
    _frame_ts = None
    if "rgb_timestamp_ns" in _fi.column_names:
        _frame_ts = _fi.column("rgb_timestamp_ns").to_numpy().astype("float64") / 1e9
    elif "rgb_timestamp_s" in _fi.column_names:
        _frame_ts = _fi.column("rgb_timestamp_s").to_numpy().astype("float64")
    imu_arr = imu_per_frame(l1.session.root, _frame_ts) if _frame_ts is not None else None
    phase_arr = contact_phase_per_frame(l1.session.root, _frame_ts) if _frame_ts is not None else None

    episodes = build_episode_inputs(
        l2.video, l2.merged, l2.atomic_actions, source_video,
        clip_annotations=la.clips, depth_video_path=depth_video,
        imu_per_frame=imu_arr, contact_phase=phase_arr,
    )
    layer3_root.mkdir(parents=True, exist_ok=True)
    with LeRobotV3DatasetWriter(layer3_root, lr_cfg, fps=l2.video.fps) as w:
        for ep in episodes:
            idx = w.register_task(ep.task_text)
            w.add_episode(ep, task_index=idx)
        l3 = w.finalize()
    print(
        f"    -> {l3.total_episodes} episodes, {l3.total_frames} frames, "
        f"{l3.total_tasks} tasks, {l3.total_videos} videos"
    )

    print("  [Validate] ...")
    v = validate_dataset(layer3_root, lr_cfg)
    print(f"    -> ok={v.ok}, errors={v.error_count}, warnings={v.warning_count}")
    for issue in v.issues:
        print(f"    [{issue.severity}] {issue.code}: {issue.message}")
    return v.ok


def _run_one_session_worker(
    session_path: Path,
    per_out: Path,
    itw_cfg,
    hp_cfg,
    lr_cfg,
    annotate_cfg,
    worker_id: int,
    total: int,
    idx: int,
) -> tuple[str, str, Path]:
    """Top-level entry point for ProcessPool workers.

    Must be importable + picklable (no closures). Each worker re-enters this
    function with its own process — annotation cache opens a fresh per-process
    SQLite connection (WAL mode handles concurrent writes).
    """
    name = session_path.name
    prefix = f"[w{worker_id}|{idx}/{total}] {name}"
    print(f"{prefix}: processing -> {per_out}")
    try:
        ok = _run_one_session(
            session_path, per_out, itw_cfg, hp_cfg, lr_cfg,
            annotate_cfg=annotate_cfg,
        )
        return ("ok" if ok else "fail", name, per_out)
    except Exception as exc:
        print(f"{prefix}: ERROR: {type(exc).__name__}: {exc}")
        return ("fail", name, per_out)


def _cmd_run_all(args: argparse.Namespace) -> int:
    from dataclasses import replace
    from .annotate import AnnotateConfig
    from .config import HandPoseConfig, ITWConfig, LeRobotConfig
    from .itw import is_session_dir

    itw_cfg = ITWConfig.from_file(args.itw_config)
    hp_cfg = HandPoseConfig.from_file(args.handpose_config)
    lr_cfg = LeRobotConfig.from_file(args.lerobot_config)

    output_root = Path(args.output_root) if args.output_root is not None else _resolve_output_root()

    # Persistent annotation cache: enabled by default at
    # <output_root>/.annotation_cache.db; --no-cache disables.
    annotate_cfg = AnnotateConfig()
    if not getattr(args, "no_cache", False):
        cache_db_path = output_root / ".annotation_cache.db"
        annotate_cfg = replace(annotate_cfg, cache_db=cache_db_path)

    # Single-session vs batch detection.
    if is_session_dir(args.session_dir, itw_cfg):
        sessions = [args.session_dir]
        mode = "single"
    else:
        candidates = sorted(c for c in args.session_dir.iterdir() if c.is_dir())
        sessions = [c for c in candidates if is_session_dir(c, itw_cfg)]
        if not sessions:
            print(
                f"No ITW session directories found at {args.session_dir} or one level down."
            )
            print(
                f"  A directory counts as a session iff it contains both "
                f"`{itw_cfg.manifest_filename}` and `{itw_cfg.rgb_video_filename}`."
            )
            return 2
        mode = "batch"

    src = "$MMPIPE_OUTPUT_ROOT" if os.environ.get("MMPIPE_OUTPUT_ROOT") else "default"
    explicit = " (explicit)" if args.output_root is not None else f" (from {src})"
    print(f"[{mode}] {len(sessions)} session(s) -> {output_root}/{explicit}")

    # Pre-flight: classify into skip-able vs needs-processing. Skip / cleanup
    # always runs in main process so ProcessPool workers only see fresh state.
    results: list[tuple[str, str, Path]] = []  # (status, name, per_out)
    to_process: list[tuple[int, Path, Path]] = []  # (idx, session, per_out)
    for i, session in enumerate(sessions, 1):
        per_out = output_root / session.name
        info_path = per_out / "lerobot_dataset" / "meta" / "info.json"

        prefix = f"[{i}/{len(sessions)}] {session.name}"
        if info_path.exists() and not args.force:
            print(f"{prefix}: SKIP (already processed, info.json found; pass --force to re-process)")
            results.append(("skip", session.name, per_out))
            continue
        if info_path.exists() and args.force:
            import shutil
            shutil.rmtree(per_out, ignore_errors=True)
            print(f"{prefix}: FORCE re-process (cleaned old output)")
        to_process.append((i, session, per_out))

    parallel_n = max(1, int(getattr(args, "parallel_sessions", 1)))
    if parallel_n > 1 and len(to_process) > 1:
        import platform
        from concurrent.futures import ProcessPoolExecutor, as_completed
        if platform.system() == "Windows":
            print(
                f"[parallel] launching {parallel_n} worker processes "
                f"(Windows spawn: ~10s warmup)..."
            )
        else:
            print(f"[parallel] launching {parallel_n} worker processes...")
        with ProcessPoolExecutor(max_workers=parallel_n) as ex:
            futures = []
            for k, (i, session, per_out) in enumerate(to_process):
                worker_id = (k % parallel_n) + 1
                futures.append(ex.submit(
                    _run_one_session_worker,
                    session, per_out, itw_cfg, hp_cfg, lr_cfg, annotate_cfg,
                    worker_id, len(sessions), i,
                ))
            for fut in as_completed(futures):
                results.append(fut.result())
    else:
        for i, session, per_out in to_process:
            prefix = f"[{i}/{len(sessions)}] {session.name}"
            print(f"{prefix}: processing -> {per_out}")
            try:
                ok = _run_one_session(
                    session, per_out, itw_cfg, hp_cfg, lr_cfg,
                    annotate_cfg=annotate_cfg,
                )
                results.append(("ok" if ok else "fail", session.name, per_out))
            except Exception as exc:
                print(f"  ERROR: {type(exc).__name__}: {exc}")
                results.append(("fail", session.name, per_out))

    n_ok = sum(1 for r, _, _ in results if r == "ok")
    n_skip = sum(1 for r, _, _ in results if r == "skip")
    n_fail = sum(1 for r, _, _ in results if r == "fail")
    print(
        f"\n[summary] total={len(results)}  ok={n_ok}  skip={n_skip}  fail={n_fail}"
    )
    if n_fail:
        print("Failed sessions:")
        for r, name, _ in results:
            if r == "fail":
                print(f"  - {name}")
    if n_ok == 1 and n_fail == 0 and n_skip == 0:
        # Only one new session processed — show convenience hint
        only = next(p for r, _, p in results if r == "ok")
        print(f"\nInspect with: mmpipe info {only / 'lerobot_dataset'}")

    # Auto-generate quality-report unless the user opted out. Runs over the
    # whole output_root so it picks up both newly-processed and previously
    # skipped sessions — gives a current snapshot of the canonical dataset.
    if not getattr(args, "no_quality_report", False) and len(results) > 0:
        try:
            from .quality import (
                print_human_summary, report_to_dict, run_quality_report,
            )
            print(f"\n[quality-report] generating (this may take ~30-200s)...")
            report = run_quality_report(
                output_root,
                skip_psnr=bool(getattr(args, "no_quality_psnr", False)),
                parallelism=4,
            )
            print_human_summary(report)
            json_path = output_root / "quality_report.json"
            with json_path.open("w", encoding="utf-8") as fh:
                json.dump(report_to_dict(report), fh, ensure_ascii=False, indent=2, default=str)
                fh.write("\n")
            print(f"\nfull JSON: {json_path}")
        except Exception as exc:
            # Quality report failure should never poison the batch exit code —
            # the dataset itself is fine; just warn and continue.
            print(f"\n[quality-report] WARNING: failed to generate ({type(exc).__name__}: {exc})")

    return 0 if n_fail == 0 else 1


def _cmd_doctor(args: argparse.Namespace) -> int:
    from .config import settings

    fails = 0
    print("Environment:")

    def _row(ok: bool, label: str, detail: str) -> None:
        nonlocal fails
        if not ok:
            fails += 1
        mark = " ok  " if ok else "FAIL "
        print(f"  [{mark}] {label:<14} {detail}")

    _row(True, "Python", sys.version.split()[0])

    try:
        from ._system import FFmpegMissingError, check_ffmpeg

        bundle = check_ffmpeg()
        ver_parts = bundle.ffmpeg_version.split()
        ver = ver_parts[2] if len(ver_parts) > 2 else bundle.ffmpeg_version
        _row(True, "ffmpeg", f"{bundle.ffmpeg} (v{ver})")
        _row(True, "ffprobe", str(bundle.ffprobe))
    except FFmpegMissingError as e:
        _row(False, "ffmpeg", str(e).splitlines()[0])

    for mod in ("numpy", "pyarrow", "yaml"):
        try:
            m = __import__(mod)
            _row(True, mod, getattr(m, "__version__", "(unknown)"))
        except ImportError:
            _row(False, mod, "NOT INSTALLED — `pip install -e .`")

    _row(True, "device", settings.device)
    out_root = os.environ.get("MMPIPE_OUTPUT_ROOT")
    _row(True, "output root", out_root if out_root else "./output (set MMPIPE_OUTPUT_ROOT to override)")

    print("\nHandpose backends (Layer 2):")
    for name in ("geocalib", "moge2", "hawor_s1", "megasam", "hawor_s2"):
        try:
            backend = settings.backend(name, fallback_key="HANDPOSE_BACKEND")
        except ValueError as e:
            print(f"  [FAIL ] {name:<10} {e}")
            fails += 1
            continue
        weights = settings.weights(name)
        if backend == "mock":
            print(f"  [ ok  ] {name:<10} backend=mock  (deterministic random tensors)")
        else:
            wstatus = f"weights={weights}" if weights else "WEIGHTS UNSET"
            ok = weights is not None
            mark = " ok  " if ok else "FAIL "
            if not ok:
                fails += 1
            print(f"  [{mark}] {name:<10} backend=real  {wstatus}")

    print("\nAnnotator backends (Layer 1.5):")
    for name in ("language", "actions", "quality"):
        try:
            backend = settings.backend(name, fallback_key="ANNOTATOR_BACKEND")
        except ValueError as e:
            print(f"  [FAIL ] {name:<10} {e}")
            fails += 1
            continue
        if backend == "mock":
            print(f"  [ ok  ] {name:<10} backend=mock  (deterministic placeholder text/labels)")
        else:
            extra = ""
            if name in ("language", "actions") and backend == "dashscope":
                api_key = os.environ.get("MMPIPE_DASHSCOPE_API_KEY", "").strip()
                if not api_key:
                    extra = " (MMPIPE_DASHSCOPE_API_KEY unset)"
                    fails += 1
            mark = " ok  " if not extra else "FAIL "
            print(f"  [{mark}] {name:<10} backend={backend}{extra}")

    print()
    if fails:
        print(f"{fails} problem(s) detected.")
        return 1
    print("All checks passed. To run:")
    print("  mmpipe <session_dir>")
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    root = Path(args.dataset_root)
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        print(f"Not a LeRobot v3 dataset (missing {info_path}).")
        return 1
    info = json.loads(info_path.read_text(encoding="utf-8"))
    print(f"LeRobot {info.get('codebase_version', '?')} dataset")
    print(f"  root        {root.resolve()}")
    print(f"  robot_type  {info.get('robot_type')}")
    print(f"  fps         {info.get('fps')}")
    print(f"  episodes    {info.get('total_episodes', '?')}")
    print(f"  frames      {info.get('total_frames', '?')}")
    print(f"  tasks       {info.get('total_tasks', '?')}")
    print(f"  videos      {info.get('total_videos', '?')}")

    features = info.get("features") or {}
    for feat_name in ("observation.state", "action", "observation.images.ego"):
        feat = features.get(feat_name)
        if feat is None:
            continue
        shape = feat.get("shape")
        dtype = feat.get("dtype")
        extra = ""
        if dtype == "video":
            v = feat.get("info") or {}
            extra = f"  codec={v.get('video.codec')} {v.get('video.width')}x{v.get('video.height')}@{v.get('video.fps')}fps"
        print(f"  {feat_name}: shape={shape} dtype={dtype}{extra}")

    def _du(paths: list[Path]) -> int:
        return sum(p.stat().st_size for p in paths if p.is_file())

    vids = (
        list((root / "videos").rglob("*.mp4")) + list((root / "videos").rglob("*.mkv"))
        if (root / "videos").exists()
        else []
    )
    shards = list((root / "data").rglob("*.parquet")) if (root / "data").exists() else []
    meta_files = list((root / "meta").rglob("*")) if (root / "meta").exists() else []

    print(f"  size breakdown:")
    print(f"    videos    {len(vids):>4d} files  {_du(vids)/1024/1024:>7.2f} MB")
    print(f"    data      {len(shards):>4d} files  {_du(shards)/1024/1024:>7.2f} MB")
    print(f"    meta                {_du(meta_files)/1024:>7.1f} KB")
    return 0


def _cmd_visualize(args: argparse.Namespace) -> int:
    from .viz.core import visualize_dataset

    root = Path(args.dataset_root)
    if not (root / "meta" / "info.json").exists():
        print(f"Not a LeRobot v3 dataset (missing {root / 'meta' / 'info.json'}).")
        return 1
    if getattr(args, "world", False):
        from .viz.scene3d import render_world_synced
        out_dir = args.out or (root / "viz")
        eps = args.episode if args.episode else list(range(min(3, json.loads((root / "meta" / "info.json").read_text())["total_episodes"])))
        written = [render_world_synced(root, ep, out_dir) for ep in eps]
        print(f"Rendered {len(written)} synced world video(s):")
        for p in written:
            print(f"  {p}")
        return 0
    if getattr(args, "threed", False):
        from .viz.scene3d import render_3d_video
        out_dir = args.out or (root / "viz")
        eps = args.episode if args.episode else list(range(min(3, json.loads((root / "meta" / "info.json").read_text())["total_episodes"])))
        written = [render_3d_video(root, ep, out_dir) for ep in eps]
        print(f"Rendered {len(written)} 3D video(s):")
        for p in written:
            print(f"  {p}")
        return 0

    if args.mano or args.combined:
        from .viz.mano import render_mano_overlay, render_combined
        out_dir = args.out or (root / "viz")
        eps = args.episode if args.episode else list(range(min(3, json.loads((root / "meta" / "info.json").read_text())["total_episodes"])))
        fn = render_combined if args.combined else render_mano_overlay
        written = [fn(root, ep, out_dir) for ep in eps]
        kind = "combined 2x2" if args.combined else "MANO-mesh overlay"
        print(f"Rendered {len(written)} {kind}:")
        for p in written:
            print(f"  {p}")
        return 0

    written = visualize_dataset(
        root, episodes=args.episode, with_depth=bool(args.depth),
        with_raw=bool(args.compare), pointcloud=bool(args.pointcloud), out_dir=args.out,
    )
    print(f"Rendered {len(written)} overlay video(s):")
    for p in written:
        print(f"  {p}")
    return 0


def _cmd_quality_report(args: argparse.Namespace) -> int:
    from .quality import print_human_summary, report_to_dict, run_quality_report

    output_root = Path(args.output_root) if args.output_root is not None else _resolve_output_root()
    if not output_root.exists():
        print(f"output_root does not exist: {output_root}")
        return 2

    report = run_quality_report(
        output_root,
        skip_psnr=bool(args.no_psnr),
        parallelism=max(1, int(args.parallelism)),
    )

    json_path = Path(args.json) if args.json else output_root / "quality_report.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = report_to_dict(report)
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, default=str)
        fh.write("\n")

    if not args.json_only:
        print_human_summary(report)
        print(f"\nfull JSON written to: {json_path}")
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


_DISPATCH = {
    "run-all": _cmd_run_all,
    "ingest": _cmd_ingest,
    "handpose": _cmd_handpose,
    "annotate": _cmd_annotate,
    "lerobot": _cmd_lerobot,
    "retarget-check": _cmd_retarget_check,
    "validate": _cmd_validate,
    "lerobot-validate": _cmd_validate,
    "doctor": _cmd_doctor,
    "info": _cmd_info,
    "visualize": _cmd_visualize,
    "quality-report": _cmd_quality_report,
    "run": _cmd_run_legacy,
    "init-config": _cmd_init_config,
}


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:]) if argv is None else list(argv)
    # Shortcut: if argv[0] isn't a known subcommand and doesn't look like a flag,
    # treat it as `run-all <session>`.
    if raw and not raw[0].startswith("-") and raw[0] not in _KNOWN_COMMANDS:
        raw = ["run-all", *raw]

    parser = build_parser()
    args = parser.parse_args(raw)
    handler = _DISPATCH.get(args.command)
    if handler is None:
        parser.error(f"Unknown command: {args.command}")
        return 2
    return handler(args)
