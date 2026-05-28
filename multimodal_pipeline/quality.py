"""Data quality verification — produce a 4-dimensional report on a finished pipeline output.

Dimensions:
1. Conversion efficiency — sizes, compression ratios, episode/frame counts
2. Video quality — sampled PSNR-Y / SSIM vs source, frame alignment, first-frame
   keyframe coverage
3. Annotation quality — task uniqueness, action label distribution + entropy,
   action_score statistics, backend provenance
4. Structural compliance — reuses lerobot_v3.validate.validate_dataset for each
   session, info.json required fields

Designed for incremental runs: cheap to invoke after `mmpipe`, so before/after
diffs across pipeline iterations are easy.
"""

from __future__ import annotations

import json
import math
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from ._system import check_ffmpeg
from .config import LeRobotConfig
from .lerobot_v3.validate import validate_dataset


# ---------------------------------------------------------------------------
# Dataclasses (per-session + aggregate)
# ---------------------------------------------------------------------------


@dataclass
class EfficiencyMetric:
    source_size_bytes: int                 # source rgb_head.mp4 size
    output_total_bytes: int                # whole per-session output dir
    output_videos_bytes: int               # lerobot_dataset/videos/**
    output_parquet_bytes: int              # lerobot_dataset/data/** + meta/*.parquet
    output_nir_bytes: int                  # nir/** (Layer 1 + 1.5 intermediates)
    compression_ratio: float               # output_videos / source (smaller = better compressed)
    episodes: int
    frames: int


@dataclass
class VideoQualityMetric:
    sampled_episodes: list[int]            # which episode indices were measured
    psnr_y_per_ep: list[float]             # one PSNR per sampled episode (avg of its frames)
    ssim_per_ep: list[float]               # one SSIM per sampled episode (avg of its frames)
    psnr_y_mean: float
    ssim_mean: float
    frame_alignment_ok: bool               # episode mp4 nb_frames == parquet length, all episodes
    frame_alignment_failures: list[int]    # episode indices where mismatch
    keyframe_ok_count: int                 # episodes whose first frame is I-frame
    keyframe_total: int


@dataclass
class AnnotationQualityMetric:
    unique_tasks: int
    total_clips: int
    task_dedup_ratio: float                # unique_tasks / total_clips
    action_label_counts: dict[str, int]
    action_label_entropy_bits: float       # Shannon entropy in bits
    action_score_mean: float
    action_score_std: float
    language_source: str                   # "mock" | "dashscope" | ...
    actions_source: str
    quality_source: str


@dataclass
class ComplianceMetric:
    validate_ok: bool
    validate_errors: int
    validate_warnings: int
    first_issues: list[str]                # first 3 issue messages
    info_required_fields_ok: bool          # codebase_version / fps / features / total_episodes etc.
    parquet_readable: bool                 # can pyarrow open data parquet without error


@dataclass
class SessionQuality:
    session_id: str
    lerobot_root: str
    nir_dir: str
    efficiency: EfficiencyMetric
    video: VideoQualityMetric
    annotation: AnnotationQualityMetric
    compliance: ComplianceMetric


@dataclass
class QualityReport:
    output_root: str
    session_count: int
    sessions: list[SessionQuality] = field(default_factory=list)
    aggregate: dict[str, Any] = field(default_factory=dict)
    elapsed_s: float = 0.0
    generated_at: str = ""


# ---------------------------------------------------------------------------
# ffmpeg / ffprobe helpers
# ---------------------------------------------------------------------------


def _ffprobe_frame_count(ffprobe: Path, video: Path) -> int:
    cmd = [
        str(ffprobe), "-v", "error", "-select_streams", "v:0",
        "-count_frames", "-show_entries", "stream=nb_read_frames",
        "-of", "csv=p=0", str(video),
    ]
    res = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return int(res.stdout.strip() or 0)


def _ffprobe_first_keyframe(ffprobe: Path, video: Path) -> bool:
    cmd = [
        str(ffprobe), "-v", "error", "-select_streams", "v:0",
        "-read_intervals", "%+#1",
        "-show_entries", "frame=key_frame",
        "-of", "csv=p=0", str(video),
    ]
    res = subprocess.run(cmd, check=True, capture_output=True, text=True)
    out = res.stdout.strip().rstrip(",").strip()
    return out == "1"


def _ffprobe_width_height(ffprobe: Path, video: Path) -> tuple[int, int]:
    cmd = [
        str(ffprobe), "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0", str(video),
    ]
    res = subprocess.run(cmd, check=True, capture_output=True, text=True)
    parts = [p.strip() for p in res.stdout.strip().split(",") if p.strip()]
    w, h = int(parts[0]), int(parts[1])
    return w, h


def _decode_frames_rawvideo(
    ffmpeg: Path, video: Path, width: int, height: int,
    select_filter: str | None = None, max_frames: int | None = None,
) -> np.ndarray:
    """Decode video frames as raw RGB24 → (T, H, W, 3) uint8 numpy array.

    select_filter: an ffmpeg filter expression like
        "select='between(n,23,30)',setpts=N/30/TB"
    used to subselect frames before output.
    """
    cmd: list[str] = [str(ffmpeg), "-v", "error", "-i", str(video)]
    if select_filter:
        cmd += ["-vf", select_filter]
    if max_frames is not None:
        cmd += ["-frames:v", str(max_frames)]
    cmd += ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    res = subprocess.run(cmd, check=True, capture_output=True)
    buf = res.stdout
    frame_bytes = width * height * 3
    n = len(buf) // frame_bytes
    if n == 0:
        return np.zeros((0, height, width, 3), dtype=np.uint8)
    return np.frombuffer(buf, dtype=np.uint8, count=n * frame_bytes).reshape(
        n, height, width, 3
    ).copy()


# ---------------------------------------------------------------------------
# PSNR / SSIM (numpy only, luma channel)
# ---------------------------------------------------------------------------


def _psnr_y(a: np.ndarray, b: np.ndarray) -> float:
    """PSNR on luma (BT.601 weights). a, b: (H, W, 3) uint8."""
    ay = (0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]).astype(np.float64)
    by = (0.299 * b[..., 0] + 0.587 * b[..., 1] + 0.114 * b[..., 2]).astype(np.float64)
    mse = ((ay - by) ** 2).mean()
    if mse <= 1e-9:
        return float("inf")
    return float(10.0 * math.log10(255.0 * 255.0 / mse))


def _ssim_simple(a: np.ndarray, b: np.ndarray) -> float:
    """Single-window SSIM on luma — fast, slightly looser than skimage SSIM."""
    ay = (0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]).astype(np.float64)
    by = (0.299 * b[..., 0] + 0.587 * b[..., 1] + 0.114 * b[..., 2]).astype(np.float64)
    mu1, mu2 = ay.mean(), by.mean()
    sig1, sig2 = ay.std(), by.std()
    cov = ((ay - mu1) * (by - mu2)).mean()
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    num = (2 * mu1 * mu2 + c1) * (2 * cov + c2)
    den = (mu1 ** 2 + mu2 ** 2 + c1) * (sig1 ** 2 + sig2 ** 2 + c2)
    return float(num / den)


# ---------------------------------------------------------------------------
# Per-dimension measurement
# ---------------------------------------------------------------------------


def _dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _measure_efficiency(session_dir: Path, lerobot_root: Path, source_video: Path,
                        info: dict) -> EfficiencyMetric:
    src_size = source_video.stat().st_size if source_video.exists() else 0
    videos_size = _dir_size_bytes(lerobot_root / "videos")
    parquet_size = (
        _dir_size_bytes(lerobot_root / "data")
        + _dir_size_bytes(lerobot_root / "meta")
    )
    nir_size = _dir_size_bytes(session_dir / "nir")
    out_total = _dir_size_bytes(session_dir)
    ratio = (videos_size / src_size) if src_size > 0 else 0.0
    return EfficiencyMetric(
        source_size_bytes=src_size,
        output_total_bytes=out_total,
        output_videos_bytes=videos_size,
        output_parquet_bytes=parquet_size,
        output_nir_bytes=nir_size,
        compression_ratio=round(ratio, 4),
        episodes=int(info.get("total_episodes", 0)),
        frames=int(info.get("total_frames", 0)),
    )


def _measure_video(
    lerobot_root: Path, video_key: str, source_video: Path,
    clip_anns: dict[int, tuple[int, int]], episode_lengths: dict[int, int],
    ffmpeg: Path, ffprobe: Path,
) -> VideoQualityMetric:
    videos_dir = lerobot_root / "videos" / video_key / "chunk-000"
    eps_sorted = sorted(episode_lengths.keys())
    if not eps_sorted:
        return VideoQualityMetric(
            sampled_episodes=[], psnr_y_per_ep=[], ssim_per_ep=[],
            psnr_y_mean=0.0, ssim_mean=0.0,
            frame_alignment_ok=True, frame_alignment_failures=[],
            keyframe_ok_count=0, keyframe_total=0,
        )

    # Frame alignment + first-frame keyframe on ALL episodes (cheap ffprobe).
    align_failures: list[int] = []
    keyframe_ok = 0
    for ep_idx in eps_sorted:
        ep_mp4 = videos_dir / f"episode_{ep_idx:06d}.mp4"
        if not ep_mp4.exists():
            align_failures.append(ep_idx)
            continue
        try:
            n = _ffprobe_frame_count(ffprobe, ep_mp4)
        except subprocess.CalledProcessError:
            align_failures.append(ep_idx)
            continue
        if n != episode_lengths[ep_idx]:
            align_failures.append(ep_idx)
        try:
            if _ffprobe_first_keyframe(ffprobe, ep_mp4):
                keyframe_ok += 1
        except subprocess.CalledProcessError:
            pass

    # PSNR/SSIM on sampled episodes (first, middle, last).
    sampled: list[int] = []
    if len(eps_sorted) >= 1:
        sampled.append(eps_sorted[0])
    if len(eps_sorted) >= 3:
        sampled.append(eps_sorted[len(eps_sorted) // 2])
    if len(eps_sorted) >= 2:
        sampled.append(eps_sorted[-1])
    sampled = sorted(set(sampled))

    psnr_per_ep: list[float] = []
    ssim_per_ep: list[float] = []
    if source_video.exists() and clip_anns:
        try:
            src_w, src_h = _ffprobe_width_height(ffprobe, source_video)
        except subprocess.CalledProcessError:
            src_w, src_h = 0, 0
        for ep_idx in sampled:
            ep_mp4 = videos_dir / f"episode_{ep_idx:06d}.mp4"
            if ep_idx not in clip_anns or not ep_mp4.exists():
                continue
            fs, fe = clip_anns[ep_idx]
            try:
                src_frames = _decode_frames_rawvideo(
                    ffmpeg, source_video, src_w, src_h,
                    select_filter=f"select='between(n,{fs},{fe - 1})',setpts=N/30/TB",
                    max_frames=(fe - fs),
                )
                ep_frames = _decode_frames_rawvideo(ffmpeg, ep_mp4, src_w, src_h)
            except subprocess.CalledProcessError:
                continue
            n = min(src_frames.shape[0], ep_frames.shape[0])
            if n == 0:
                continue
            psnrs = [_psnr_y(src_frames[i], ep_frames[i]) for i in range(n)]
            ssims = [_ssim_simple(src_frames[i], ep_frames[i]) for i in range(n)]
            psnr_per_ep.append(round(sum(psnrs) / n, 3))
            ssim_per_ep.append(round(sum(ssims) / n, 5))

    psnr_mean = round(sum(psnr_per_ep) / len(psnr_per_ep), 3) if psnr_per_ep else 0.0
    ssim_mean = round(sum(ssim_per_ep) / len(ssim_per_ep), 5) if ssim_per_ep else 0.0

    return VideoQualityMetric(
        sampled_episodes=sampled,
        psnr_y_per_ep=psnr_per_ep,
        ssim_per_ep=ssim_per_ep,
        psnr_y_mean=psnr_mean,
        ssim_mean=ssim_mean,
        frame_alignment_ok=len(align_failures) == 0,
        frame_alignment_failures=align_failures[:10],
        keyframe_ok_count=keyframe_ok,
        keyframe_total=len(eps_sorted),
    )


def _shannon_entropy_bits(counts: dict[str, int]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    H = 0.0
    for c in counts.values():
        if c <= 0:
            continue
        p = c / total
        H -= p * math.log2(p)
    return round(H, 4)


def _measure_annotation(session_dir: Path, info: dict, video_key: str) -> AnnotationQualityMetric:
    nir_dir = session_dir / "nir"
    sess_nir = None
    if nir_dir.exists():
        for child in nir_dir.iterdir():
            if child.is_dir() and (child / "clip_annotations.parquet").exists():
                sess_nir = child
                break

    clip_anns_path = (sess_nir / "clip_annotations.parquet") if sess_nir else None
    tasks_path = session_dir / "lerobot_dataset" / "meta" / "tasks.parquet"

    if not clip_anns_path or not clip_anns_path.exists():
        return AnnotationQualityMetric(
            unique_tasks=int(info.get("total_tasks", 0)) if info else 0,
            total_clips=int(info.get("total_episodes", 0)) if info else 0,
            task_dedup_ratio=0.0,
            action_label_counts={},
            action_label_entropy_bits=0.0,
            action_score_mean=0.0,
            action_score_std=0.0,
            language_source="?",
            actions_source="?",
            quality_source="?",
        )

    ca = pq.read_table(clip_anns_path).to_pydict()
    total = len(ca["clip_idx"])
    labels = ca.get("action_label", [])
    scores = ca.get("action_score", []) or [0.0] * total
    label_counts: dict[str, int] = {}
    for L in labels:
        label_counts[L] = label_counts.get(L, 0) + 1
    score_arr = np.array(scores, dtype=np.float64) if scores else np.zeros(0)
    unique_tasks = int(pq.read_table(tasks_path).num_rows) if tasks_path.exists() else 0

    language_source = ca.get("language_source", ["?"])[0] if ca.get("language_source") else "?"
    actions_source = ca.get("actions_source", ["?"])[0] if ca.get("actions_source") else "?"
    # quality_source comes from frame_quality.parquet — peek schema if possible
    fq_path = sess_nir / "frame_quality.parquet" if sess_nir else None
    quality_source = "rule_based" if fq_path and fq_path.exists() else "?"

    return AnnotationQualityMetric(
        unique_tasks=unique_tasks,
        total_clips=total,
        task_dedup_ratio=round(unique_tasks / total, 4) if total > 0 else 0.0,
        action_label_counts=label_counts,
        action_label_entropy_bits=_shannon_entropy_bits(label_counts),
        action_score_mean=round(float(score_arr.mean()), 4) if score_arr.size else 0.0,
        action_score_std=round(float(score_arr.std()), 4) if score_arr.size else 0.0,
        language_source=language_source,
        actions_source=actions_source,
        quality_source=quality_source,
    )


_INFO_REQUIRED_FIELDS = (
    "codebase_version", "robot_type", "fps",
    "total_episodes", "total_frames", "total_tasks", "total_videos",
    "features", "chunks_size",
)


def _measure_compliance(lerobot_root: Path, info: dict) -> ComplianceMetric:
    report = validate_dataset(lerobot_root, LeRobotConfig())
    info_ok = all(k in info for k in _INFO_REQUIRED_FIELDS)
    parquet_ok = True
    try:
        data_p = lerobot_root / "data" / "chunk-000" / "file-000.parquet"
        pq.read_table(data_p, columns=["index"])
    except Exception:
        parquet_ok = False
    issues = [f"[{i.severity}] {i.code}: {i.message}" for i in report.issues[:3]]
    return ComplianceMetric(
        validate_ok=report.ok,
        validate_errors=report.error_count,
        validate_warnings=report.warning_count,
        first_issues=issues,
        info_required_fields_ok=info_ok,
        parquet_readable=parquet_ok,
    )


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------


def _measure_session(session_dir: Path, ffmpeg: Path, ffprobe: Path,
                     skip_psnr: bool) -> SessionQuality | None:
    lerobot_root = session_dir / "lerobot_dataset"
    info_path = lerobot_root / "meta" / "info.json"
    if not info_path.exists():
        return None
    info = json.loads(info_path.read_text(encoding="utf-8"))
    video_key = info.get("video_key", "observation.images.ego")

    # Resolve source video via media_paths.json under nir/<sess>/.
    nir_root = session_dir / "nir"
    sess_nir = next((c for c in nir_root.iterdir() if c.is_dir()), None) if nir_root.exists() else None
    media_paths_path = (sess_nir / "media_paths.json") if sess_nir else None
    source_video = Path()
    if media_paths_path and media_paths_path.exists():
        mp = json.loads(media_paths_path.read_text(encoding="utf-8"))
        source_video = Path(mp.get("media", {}).get("rgb", ""))

    # Episode lengths from episodes manifest.
    eps_path = lerobot_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    episode_lengths: dict[int, int] = {}
    if eps_path.exists():
        eps_t = pq.read_table(eps_path).to_pydict()
        for i, ep_idx in enumerate(eps_t["episode_index"]):
            episode_lengths[int(ep_idx)] = int(eps_t["length"][i])

    # Clip frame ranges from NIR clip_annotations.
    clip_anns: dict[int, tuple[int, int]] = {}
    if sess_nir and (sess_nir / "clip_annotations.parquet").exists():
        ca = pq.read_table(sess_nir / "clip_annotations.parquet").to_pydict()
        for i, ci in enumerate(ca["clip_idx"]):
            clip_anns[int(ci)] = (int(ca["frame_start"][i]), int(ca["frame_end"][i]))

    efficiency = _measure_efficiency(session_dir, lerobot_root, source_video, info)
    if skip_psnr:
        video = VideoQualityMetric(
            sampled_episodes=[], psnr_y_per_ep=[], ssim_per_ep=[],
            psnr_y_mean=0.0, ssim_mean=0.0,
            frame_alignment_ok=True, frame_alignment_failures=[],
            keyframe_ok_count=0, keyframe_total=len(episode_lengths),
        )
    else:
        video = _measure_video(
            lerobot_root, video_key, source_video, clip_anns, episode_lengths,
            ffmpeg, ffprobe,
        )
    annotation = _measure_annotation(session_dir, info, video_key)
    compliance = _measure_compliance(lerobot_root, info)

    return SessionQuality(
        session_id=session_dir.name,
        lerobot_root=str(lerobot_root),
        nir_dir=str(sess_nir) if sess_nir else "",
        efficiency=efficiency,
        video=video,
        annotation=annotation,
        compliance=compliance,
    )


def _aggregate(sessions: list[SessionQuality]) -> dict[str, Any]:
    n = len(sessions)
    if n == 0:
        return {}
    psnr_vals = [s.video.psnr_y_mean for s in sessions if s.video.psnr_y_mean > 0]
    ssim_vals = [s.video.ssim_mean for s in sessions if s.video.ssim_mean > 0]
    src_total = sum(s.efficiency.source_size_bytes for s in sessions)
    vid_total = sum(s.efficiency.output_videos_bytes for s in sessions)
    out_total = sum(s.efficiency.output_total_bytes for s in sessions)
    eps_total = sum(s.efficiency.episodes for s in sessions)
    fr_total = sum(s.efficiency.frames for s in sessions)
    valid_ok = sum(1 for s in sessions if s.compliance.validate_ok)
    align_ok = sum(1 for s in sessions if s.video.frame_alignment_ok)
    kf_ok_total = sum(s.video.keyframe_ok_count for s in sessions)
    kf_total_total = sum(s.video.keyframe_total for s in sessions)
    # Aggregate action label counts.
    agg_labels: dict[str, int] = {}
    for s in sessions:
        for k, v in s.annotation.action_label_counts.items():
            agg_labels[k] = agg_labels.get(k, 0) + v
    unique_tasks_total = sum(s.annotation.unique_tasks for s in sessions)

    def _p(arr: list[float], q: float) -> float:
        if not arr:
            return 0.0
        return round(float(np.percentile(arr, q * 100)), 3)

    return {
        "n_sessions": n,
        "efficiency": {
            "source_total_MB": round(src_total / 1024 / 1024, 2),
            "output_videos_MB": round(vid_total / 1024 / 1024, 2),
            "output_total_MB": round(out_total / 1024 / 1024, 2),
            "video_compression_ratio": round(vid_total / src_total, 4) if src_total else 0.0,
            "episodes_total": eps_total,
            "frames_total": fr_total,
            "avg_episode_size_KB": round(vid_total / 1024 / max(1, eps_total), 2),
        },
        "video": {
            "psnr_y_mean": round(sum(psnr_vals) / max(1, len(psnr_vals)), 3) if psnr_vals else 0.0,
            "psnr_y_p5": _p(psnr_vals, 0.05),
            "psnr_y_p95": _p(psnr_vals, 0.95),
            "ssim_mean": round(sum(ssim_vals) / max(1, len(ssim_vals)), 5) if ssim_vals else 0.0,
            "frame_alignment_pass_rate": round(align_ok / n, 4),
            "first_keyframe_pass_rate": round(kf_ok_total / max(1, kf_total_total), 4),
        },
        "annotation": {
            "unique_tasks_sum": unique_tasks_total,
            "action_label_dist": dict(sorted(agg_labels.items(), key=lambda kv: -kv[1])),
            "dominant_label_pct": round(
                100.0 * max(agg_labels.values()) / sum(agg_labels.values()), 2
            ) if agg_labels else 0.0,
        },
        "compliance": {
            "validate_pass_rate": round(valid_ok / n, 4),
            "validate_pass_count": valid_ok,
            "validate_fail_count": n - valid_ok,
        },
    }


def run_quality_report(
    output_root: str | Path,
    *,
    skip_psnr: bool = False,
    parallelism: int = 4,
) -> QualityReport:
    """Generate a 4-dimensional quality report over every session under output_root.

    A "session" is a child directory of output_root containing
    ``lerobot_dataset/meta/info.json``.
    """
    output_root = Path(output_root)
    ffb = check_ffmpeg()
    t0 = time.perf_counter()

    sess_dirs = sorted(
        c for c in output_root.iterdir()
        if c.is_dir() and (c / "lerobot_dataset" / "meta" / "info.json").exists()
    )

    sessions: list[SessionQuality] = []
    if parallelism <= 1 or len(sess_dirs) <= 1:
        for d in sess_dirs:
            r = _measure_session(d, ffb.ffmpeg, ffb.ffprobe, skip_psnr)
            if r is not None:
                sessions.append(r)
    else:
        with ThreadPoolExecutor(max_workers=parallelism) as ex:
            futures = [
                ex.submit(_measure_session, d, ffb.ffmpeg, ffb.ffprobe, skip_psnr)
                for d in sess_dirs
            ]
            for fut in futures:
                r = fut.result()
                if r is not None:
                    sessions.append(r)
    sessions.sort(key=lambda s: s.session_id)

    return QualityReport(
        output_root=str(output_root),
        session_count=len(sessions),
        sessions=sessions,
        aggregate=_aggregate(sessions),
        elapsed_s=round(time.perf_counter() - t0, 2),
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def report_to_dict(report: QualityReport) -> dict[str, Any]:
    """Convert to a JSON-serialisable dict."""
    return asdict(report)


def print_human_summary(report: QualityReport) -> None:
    """Print a compact, human-readable summary to stdout."""
    a = report.aggregate
    print(f"\n=== Quality Report — {report.output_root} ===")
    print(f"sessions: {report.session_count}   generated: {report.generated_at}   elapsed: {report.elapsed_s}s\n")

    if not a:
        print("(no sessions found)")
        return

    e = a["efficiency"]
    print("[efficiency]")
    print(f"  source total      : {e['source_total_MB']} MB")
    print(f"  output videos     : {e['output_videos_MB']} MB  (compression {e['video_compression_ratio']:.3f} of source)")
    print(f"  output total      : {e['output_total_MB']} MB")
    print(f"  episodes / frames : {e['episodes_total']} / {e['frames_total']}")
    print(f"  avg episode size  : {e['avg_episode_size_KB']} KB")

    v = a["video"]
    print("\n[video quality]")
    print(f"  PSNR-Y mean / p5 / p95 : {v['psnr_y_mean']} / {v['psnr_y_p5']} / {v['psnr_y_p95']} dB")
    print(f"  SSIM mean              : {v['ssim_mean']}")
    print(f"  frame alignment pass   : {v['frame_alignment_pass_rate'] * 100:.2f}%")
    print(f"  first-keyframe pass    : {v['first_keyframe_pass_rate'] * 100:.2f}%")

    n = a["annotation"]
    print("\n[annotation quality]")
    print(f"  unique tasks (sum across sessions) : {n['unique_tasks_sum']}")
    print(f"  dominant action label pct          : {n['dominant_label_pct']}%")
    print(f"  action label distribution          :")
    for label, count in n["action_label_dist"].items():
        print(f"    {label:<10} {count}")

    c = a["compliance"]
    print("\n[structural compliance]")
    print(f"  validate pass rate : {c['validate_pass_rate'] * 100:.2f}%  ({c['validate_pass_count']}/{c['validate_pass_count'] + c['validate_fail_count']})")
    if c["validate_fail_count"] > 0:
        print(f"  ⚠ {c['validate_fail_count']} session(s) failed validate")
        for s in report.sessions:
            if not s.compliance.validate_ok:
                print(f"    - {s.session_id}: errors={s.compliance.validate_errors} {s.compliance.first_issues}")


__all__ = [
    "EfficiencyMetric",
    "VideoQualityMetric",
    "AnnotationQualityMetric",
    "ComplianceMetric",
    "SessionQuality",
    "QualityReport",
    "run_quality_report",
    "report_to_dict",
    "print_human_summary",
]