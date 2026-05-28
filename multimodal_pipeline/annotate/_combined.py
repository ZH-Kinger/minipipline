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
import os
import re
from dataclasses import dataclass
from pathlib import Path

from ..config import AnnotateConfig
from ..config.models import QwenVlHyper
from ._dashscope import (
    DashScopeError,
    call_qwen_vl,
    extract_clip_frames,
    require_api_key,
)
from ._mock import MOCK_ACTION_VOCAB
from .actions import _normalise_action_output


_DEFAULT_VLM_CONFIDENCE = 0.8
_JSON_BLOCK_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _parse_combined_response(raw: str) -> tuple[str | None, str | None]:
    """Parse Qwen-VL's combined-call output. Returns (description, action).

    Tolerates: markdown code fences, extra text around the JSON, single-line
    or pretty-printed JSON. Returns ``(None, None)`` if no JSON object can be
    located at all.
    """
    text = raw.strip()
    # Strip ```json ... ``` or ``` ... ``` markdown fences if present.
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()

    # First try the full string as JSON.
    parsed: dict | None = None
    try:
        candidate = json.loads(text)
        if isinstance(candidate, dict):
            parsed = candidate
    except json.JSONDecodeError:
        pass

    if parsed is None:
        # Fall back: grab the first {...} substring.
        match = _JSON_BLOCK_RE.search(text)
        if match:
            try:
                candidate = json.loads(match.group(0))
                if isinstance(candidate, dict):
                    parsed = candidate
            except json.JSONDecodeError:
                pass

    if parsed is None:
        return None, None

    desc = parsed.get("description")
    action = parsed.get("action")
    desc_str = str(desc).strip() if isinstance(desc, (str, int, float)) else None
    action_str = str(action).strip() if isinstance(action, (str, int, float)) else None
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
    ) -> tuple[str, str, float]:
        frames = extract_clip_frames(
            video_path,
            t_start_s=t_start_s,
            t_end_s=t_end_s,
            max_frames=self.hyper.max_clip_frames,
            long_edge_px=self.hyper.max_frame_resize_long_edge,
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
        )

        desc, action_raw = _parse_combined_response(raw)
        if desc is None and action_raw is None:
            # Total parse failure — keep the clip alive with a marker so the
            # batch doesn't die.
            return (raw.replace("\n", " ").strip()[:60] or "[parse-fail]", "move", 0.5)

        text = (desc or "").replace("\n", " ").strip() or "[empty]"
        label = _normalise_action_output(action_raw or "") if action_raw else "move"
        return text, label, _DEFAULT_VLM_CONFIDENCE


__all__ = ["DashScopeCombinedAnnotator", "_parse_combined_response"]
