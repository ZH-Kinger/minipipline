"""Run all Layer 1.5 annotation backends on a NIR session.

Consumes:
- NIR session directory (mandatory) — provides ``media_paths.json``,
  ``frame_index.parquet`` (frame count + fps), ``task.json``, optional
  ``hand_keypoints.parquet`` for hand-visibility signals.
- Optional ``atomic_actions`` list from Layer 2 — used as clip boundaries.
  When absent, the orchestrator falls back to fixed-stride clipping (every
  ``cfg.fixed_window_clip_len_s`` seconds), with at least 1 clip covering
  the whole video.

Produces:
- ``clip_annotations.parquet`` next to the existing NIR parquets
- ``frame_quality.parquet`` ditto
- :class:`AnnotateRunResult` for callers that want in-memory access.
"""

from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..config import AnnotateConfig, settings
from ..config.models import QwenVlHyper
from ._sampling import double_hand_max_speed
from .actions import DashScopeActionLabeler, build_action_labeler
from .language import DashScopeLanguageAnnotator, build_language_annotator
from .quality import build_quality_scorer
from .schemas import AnnotateRunResult, ClipAnnotation, FrameQualityRow


def _video_id_from_path(path: Path) -> str:
    """Deterministic short id used as a seed component for mock backends."""

    h = hashlib.sha1()
    h.update(str(path.resolve()).encode("utf-8"))
    try:
        st = path.stat()
        h.update(str(st.st_size).encode("utf-8"))
        h.update(str(int(st.st_mtime)).encode("utf-8"))
    except OSError:
        pass
    return h.hexdigest()[:12]


def _read_frame_index(nir_dir: Path) -> tuple[int, float]:
    """Return (n_frames, fps) from frame_index.parquet."""

    table = pq.read_table(nir_dir / "frame_index.parquet")
    n_frames = table.num_rows
    if "rgb_timestamp_s" in table.column_names and n_frames >= 2:
        ts = table.column("rgb_timestamp_s").to_numpy()
        dt = float(ts[-1] - ts[0])
        fps = (n_frames - 1) / dt if dt > 1e-9 else 30.0
    else:
        fps = 30.0
    return n_frames, fps


def _read_hand_visible(nir_dir: Path, n_frames: int) -> np.ndarray:
    """Best-effort: read hand validity (either hand) per frame; else all True."""

    p = nir_dir / "hand_keypoints.parquet"
    if not p.exists():
        return np.ones(n_frames, dtype=bool)
    t = pq.read_table(p)
    cols = t.column_names
    out = np.zeros(n_frames, dtype=bool)
    # `left_valid` / `right_valid` columns produced by Layer 1 normalize step.
    left = t.column("left_valid").to_numpy().astype(bool) if "left_valid" in cols else None
    right = t.column("right_valid").to_numpy().astype(bool) if "right_valid" in cols else None
    if left is None and right is None:
        return np.ones(n_frames, dtype=bool)
    if left is not None:
        out[: len(left)] |= left
    if right is not None:
        out[: len(right)] |= right
    return out


def _plan_clips_from_atomics(
    atomic_actions: Iterable[Any], fps: float
) -> list[tuple[int, int, float, float]]:
    """Convert Layer 2 atomic_actions into ``(fs, fe, ts, te)`` tuples.

    ``AtomicAction`` only carries frame indices (Layer 2 schema); we derive
    timestamps from the NIR ``frame_index.parquet`` reported fps.
    """

    safe_fps = fps if fps > 1e-6 else 30.0
    out: list[tuple[int, int, float, float]] = []
    for a in atomic_actions:
        fs = int(a.frame_start)
        fe = int(a.frame_end)
        out.append((fs, fe, fs / safe_fps, fe / safe_fps))
    return out


def _plan_clips_fixed_window(
    n_frames: int, fps: float, window_s: float
) -> list[tuple[int, int, float, float]]:
    """Fall-back clip planner when no atomic_actions are available."""

    if n_frames <= 0:
        return []
    stride = max(1, int(round(fps * window_s)))
    clips: list[tuple[int, int, float, float]] = []
    fs = 0
    while fs < n_frames:
        fe = min(n_frames, fs + stride)
        ts = fs / fps if fps > 0 else 0.0
        te = fe / fps if fps > 0 else 0.0
        clips.append((fs, fe, ts, te))
        if fe == n_frames:
            break
        fs = fe
    return clips


def _write_clip_annotations_parquet(path: Path, clips: list[ClipAnnotation]) -> None:
    if not clips:
        path.write_bytes(b"")  # leave an empty marker
        return
    table = pa.table(
        {
            "clip_idx": [c.clip_idx for c in clips],
            "frame_start": [c.frame_start for c in clips],
            "frame_end": [c.frame_end for c in clips],
            "t_start_s": [c.t_start_s for c in clips],
            "t_end_s": [c.t_end_s for c in clips],
            "language_text": [c.language_text for c in clips],
            "action_label": [c.action_label for c in clips],
            "action_score": [c.action_score for c in clips],
            "language_source": [c.language_source for c in clips],
            "actions_source": [c.actions_source for c in clips],
        }
    )
    pq.write_table(table, path)


def _write_frame_quality_parquet(path: Path, rows: list[FrameQualityRow]) -> None:
    if not rows:
        path.write_bytes(b"")
        return
    table = pa.table(
        {
            "frame_idx": [r.frame_idx for r in rows],
            "blur_score": [r.blur_score for r in rows],
            "exposure_score": [r.exposure_score for r in rows],
            "hand_visible": [r.hand_visible for r in rows],
            "overall_quality": [r.overall_quality for r in rows],
            "kept": [r.kept for r in rows],
        }
    )
    pq.write_table(table, path)


def run_annotate_pipeline(
    nir_dir: str | Path,
    cfg: AnnotateConfig | None = None,
    atomic_actions: Iterable[Any] | None = None,
    merged_prediction: Any | None = None,
) -> AnnotateRunResult:
    """Run language + actions + quality backends on a NIR session.

    ``atomic_actions`` is the list of :class:`handpose.schemas.AtomicAction`
    produced by Layer 2. When provided, each atomic action becomes one clip
    for the language / actions backends. When None, the orchestrator falls
    back to fixed-stride clipping driven by ``cfg.fixed_window_clip_len_s``.

    ``merged_prediction`` is the Layer 2 :class:`MergedPrediction`. When
    provided AND ``cfg.frame_sampling == "motion_peak"``, the VLM sees
    velocity-peak frames instead of uniform samples — substantially better
    grounding on the action-defining moments.
    """

    cfg = cfg or AnnotateConfig()
    nir_dir = Path(nir_dir)
    timings: dict[str, float] = {}

    media_paths = json.loads((nir_dir / "media_paths.json").read_text(encoding="utf-8"))
    video_path = Path(media_paths["media"]["rgb"])
    task = json.loads((nir_dir / "task.json").read_text(encoding="utf-8"))
    task_text = task.get("task_text") or task.get("task_info", {}).get("name", "")

    t = time.perf_counter()
    n_frames, fps = _read_frame_index(nir_dir)
    hand_visible = _read_hand_visible(nir_dir, n_frames)
    timings["nir_read"] = time.perf_counter() - t

    # Clip planning.
    if atomic_actions is not None:
        clip_tuples = _plan_clips_from_atomics(atomic_actions, fps)
    else:
        clip_tuples = _plan_clips_fixed_window(n_frames, fps, cfg.fixed_window_clip_len_s)

    video_id = _video_id_from_path(video_path)

    # Build backends and decide whether we can fuse language + actions into
    # one Qwen-VL call. The combined path halves cost + latency when both
    # components are dashscope; everything else uses the separate-calls path.
    language_backend = build_language_annotator(cfg)
    action_backend = build_action_labeler(cfg)

    combined_backend = None
    if isinstance(language_backend, DashScopeLanguageAnnotator) and isinstance(
        action_backend, DashScopeActionLabeler
    ):
        from ._combined import DashScopeCombinedAnnotator

        combined_backend = DashScopeCombinedAnnotator(cfg=cfg, hyper=QwenVlHyper())

    # Per-video wrist speed for motion-peak sampling (computed once, sliced
    # per-clip inside _annotate_one). None when Layer 2 prediction is absent.
    wrist_speed = (
        double_hand_max_speed(merged_prediction.pred_trans)
        if merged_prediction is not None and getattr(merged_prediction, "pred_trans", None) is not None
        else None
    )

    def _annotate_one(clip_idx: int, fs: int, fe: int, ts: float, te: float) -> ClipAnnotation:
        if combined_backend is not None:
            text, label, score = combined_backend.annotate_clip(
                video_id=video_id,
                clip_idx=clip_idx,
                frame_start=fs,
                frame_end=fe,
                t_start_s=ts,
                t_end_s=te,
                task_text=task_text,
                video_path=video_path,
                wrist_speed=wrist_speed,
            )
            lang_src = combined_backend.name
            act_src = combined_backend.name
        else:
            text = language_backend.annotate_clip(
                video_id=video_id,
                clip_idx=clip_idx,
                frame_start=fs,
                frame_end=fe,
                t_start_s=ts,
                t_end_s=te,
                task_text=task_text,
                video_path=video_path,
            )
            label, score = action_backend.classify_clip(
                video_id=video_id,
                clip_idx=clip_idx,
                frame_start=fs,
                frame_end=fe,
                t_start_s=ts,
                t_end_s=te,
                task_text=task_text,
                video_path=video_path,
            )
            lang_src = language_backend.name
            act_src = action_backend.name
        return ClipAnnotation(
            clip_idx=clip_idx,
            frame_start=fs,
            frame_end=fe,
            t_start_s=ts,
            t_end_s=te,
            language_text=text,
            action_label=label,
            action_score=score,
            language_source=lang_src,
            actions_source=act_src,
        )

    t = time.perf_counter()
    parallelism = max(1, int(getattr(cfg, "clip_parallelism", 1)))
    if parallelism > 1 and len(clip_tuples) > 1:
        with ThreadPoolExecutor(max_workers=parallelism) as ex:
            futures = [
                ex.submit(_annotate_one, i, fs, fe, ts, te)
                for i, (fs, fe, ts, te) in enumerate(clip_tuples)
            ]
            clips: list[ClipAnnotation] = [f.result() for f in futures]
    else:
        clips = [
            _annotate_one(i, fs, fe, ts, te)
            for i, (fs, fe, ts, te) in enumerate(clip_tuples)
        ]
    timings["language_actions"] = time.perf_counter() - t

    # Per-frame quality.
    t = time.perf_counter()
    quality_backend = build_quality_scorer(cfg)
    frame_quality = quality_backend.score_frames(
        video_id=video_id,
        n_frames=n_frames,
        video_path=video_path,
        hand_visible=hand_visible,
    )
    timings["quality"] = time.perf_counter() - t

    # Persist into NIR.
    t = time.perf_counter()
    if cfg.write_clip_annotations:
        _write_clip_annotations_parquet(nir_dir / "clip_annotations.parquet", clips)
    if cfg.write_frame_quality:
        _write_frame_quality_parquet(nir_dir / "frame_quality.parquet", frame_quality)
    timings["write"] = time.perf_counter() - t

    cache_stats: dict[str, float] = {}
    if combined_backend is not None and combined_backend._cache is not None:
        cache_stats = combined_backend._cache.stats()

    return AnnotateRunResult(
        clips=clips,
        frame_quality=frame_quality,
        stage_timings=timings,
        nir_dir=str(nir_dir),
        cache_stats=cache_stats,
    )


__all__ = ["run_annotate_pipeline", "AnnotateRunResult"]
