"""Combined Qwen-VL annotator: one call returns description + action.

Used when both ``MMPIPE_LANGUAGE_BACKEND`` and ``MMPIPE_ACTIONS_BACKEND`` are
``dashscope`` — halves the per-clip API cost and roughly halves wall time
compared to two separate calls.

Falls back to ``("", "move", 0.5)`` on JSON parse failure rather than
aborting the whole batch; the failure is logged via a returned warning so
the orchestrator can summarise.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import AnnotateConfig
from ..config.models import QwenVlHyper
from ._cache import AnnotationCache
from ._dashscope import (
    DashScopeError,
    call_qwen_vl,
    extract_clip_frames,
    require_api_key,
)
from ._mock import MOCK_ACTION_VOCAB
from ._sampling import pick_motion_peak_indices
from .actions import _normalise_action_output


_log = logging.getLogger(__name__)

# Placeholder score returned by the backend; the orchestrator overrides it with
# a measured per-clip reliability score (hand-detection coverage × frame
# quality), so the VLM's poorly-calibrated self-confidence is not used.
_DEFAULT_VLM_CONFIDENCE = 0.5
# Greedy {...} match — supports nested objects (CoT reasoning may contain
# braces in quoted strings, but tolerant parsing below handles it).
_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
_DESC_FIELD_RE = re.compile(r'"description"\s*:\s*"((?:[^"\\]|\\.)*)"', re.DOTALL)
_ACTION_FIELD_RE = re.compile(r'"action"\s*:\s*"((?:[^"\\]|\\.)*)"', re.DOTALL)


def _strip_markdown_fences(text: str) -> str:
    """Remove ```json ... ``` / ``` ... ``` wrapping if present."""
    t = text.strip()
    if t.startswith("```"):
        # Drop leading fence + optional language tag, then trailing fence.
        t = re.sub(r"^```(?:json|JSON)?\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def _parse_combined_response(raw: str, *, clip_idx: int | None = None) -> tuple[str | None, str | None]:
    """Parse Qwen-VL's combined-call output. Returns (description, action).

    Strategy ladder:
      1. Strip markdown fences, json.loads the whole thing.
      2. Find the first balanced {...} substring, json.loads it.
      3. Field-level regex fallback on description / action.

    Falls all the way through to (None, None) only when neither field can be
    located. Logs a WARNING with clip_idx + raw truncated when all strategies
    fail, so silent parse failures stop being invisible.
    """
    text = _strip_markdown_fences(raw)

    parsed: dict | None = None

    # Strategy 1: parse whole stripped text.
    try:
        candidate = json.loads(text)
        if isinstance(candidate, dict):
            parsed = candidate
    except json.JSONDecodeError:
        pass

    # Strategy 2: grab first {...} block.
    if parsed is None:
        match = _JSON_BLOCK_RE.search(text)
        if match:
            try:
                candidate = json.loads(match.group(0))
                if isinstance(candidate, dict):
                    parsed = candidate
            except json.JSONDecodeError:
                pass

    desc_str: str | None = None
    action_str: str | None = None
    if parsed is not None:
        desc = parsed.get("description")
        action = parsed.get("action")
        if isinstance(desc, (str, int, float)):
            desc_str = str(desc).strip()
        if isinstance(action, (str, int, float)):
            action_str = str(action).strip()

    # Strategy 3: field-level regex for either field that's still missing.
    if desc_str is None:
        m = _DESC_FIELD_RE.search(text)
        if m:
            desc_str = m.group(1).encode("utf-8").decode("unicode_escape").strip()
    if action_str is None:
        m = _ACTION_FIELD_RE.search(text)
        if m:
            action_str = m.group(1).strip()

    if desc_str is None and action_str is None:
        truncated = raw.replace("\n", " ")[:200]
        _log.warning(
            "VLM parse-fail (clip_idx=%s): raw[:200]=%r", clip_idx, truncated
        )

    return desc_str, action_str


@dataclass
class DashScopeCombinedAnnotator:
    """One Qwen-VL call → (description, action_label, action_score).

    Exposes the same ``name`` attribute the orchestrator already reads to
    populate ``ClipAnnotation.language_source`` / ``actions_source``.
    """

    cfg: AnnotateConfig
    hyper: QwenVlHyper
    name: str = "dashscope"

    def __post_init__(self) -> None:
        self._api_key = require_api_key()
        self._model = (
            os.environ.get("MMPIPE_DASHSCOPE_MODEL", "").strip() or self.hyper.variant
        )
        self._cache: AnnotationCache | None = (
            AnnotationCache(self.cfg.cache_db) if self.cfg.cache_db is not None else None
        )

    def annotate_clip(
        self,
        *,
        video_id: str,
        clip_idx: int,
        frame_start: int,
        frame_end: int,
        t_start_s: float,
        t_end_s: float,
        task_text: str,
        video_path: Path,
        wrist_speed: np.ndarray | None = None,
    ) -> tuple[str, str, float]:
        # Adaptive frame budget: scale by clip duration so short clips don't ship
        # redundant frames (quality-neutral — caps at max_clip_frames for rich
        # clips). Falls back to the fixed max when adaptive_frames is off.
        n_target = self.hyper.max_clip_frames
        if getattr(self.hyper, "adaptive_frames", False):
            dur = max(0.0, t_end_s - t_start_s)
            n_target = int(round(dur * self.hyper.adaptive_sample_fps))
            n_target = max(self.hyper.min_clip_frames, min(self.hyper.max_clip_frames, n_target))

        frame_indices: list[int] | None = None
        if (
            getattr(self.cfg, "frame_sampling", "uniform") == "motion_peak"
            and wrist_speed is not None
        ):
            frame_indices = pick_motion_peak_indices(
                wrist_speed, frame_start, frame_end, n_target,
            )
        frames = extract_clip_frames(
            video_path,
            t_start_s=t_start_s,
            t_end_s=t_end_s,
            max_frames=n_target,
            long_edge_px=self.hyper.max_frame_resize_long_edge,
            frame_indices=frame_indices,
        )
        prompt = self.hyper.combined_prompt_template.format(
            n_frames=len(frames),
            duration_s=max(0.0, t_end_s - t_start_s),
            task_text=task_text or "未指定",
            vocab=" / ".join(MOCK_ACTION_VOCAB),
        )
        raw = call_qwen_vl(
            api_key=self._api_key,
            model=self._model,
            prompt=prompt,
            frame_bytes_list=frames,
            temperature=self.hyper.temperature,
            top_p=self.hyper.top_p,
            max_tokens=self.hyper.combined_max_tokens,
            cache=self._cache,
        )

        desc, action_raw = _parse_combined_response(raw, clip_idx=clip_idx)
        if desc is None and action_raw is None:
            # Total parse failure — keep the clip alive with a marker so the
            # batch doesn't die.
            return (raw.replace("\n", " ").strip()[:60] or "[parse-fail]", "move", _DEFAULT_VLM_CONFIDENCE)

        text = (desc or "").replace("\n", " ").strip() or "[empty]"
        label = _normalise_action_output(action_raw or "") if action_raw else "move"
        # Score is a placeholder here; the orchestrator overrides action_score
        # with a measured reliability proxy (coverage × frame quality).
        return text, label, _DEFAULT_VLM_CONFIDENCE


__all__ = ["DashScopeCombinedAnnotator", "_parse_combined_response"]
