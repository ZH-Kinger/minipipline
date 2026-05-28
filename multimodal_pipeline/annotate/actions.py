"""Action category labeller (Layer 1.5 — clip → atomic action class + score)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from ..config import AnnotateConfig, settings
from ..config.models import QwenVlHyper
from ._dashscope import (
    DashScopeError,
    call_qwen_vl,
    extract_clip_frames,
    require_api_key,
)
from ._mock import MOCK_ACTION_VOCAB, mock_rng


@dataclass
class MockActionLabeler:
    """Deterministic placeholder. No external calls."""

    cfg: AnnotateConfig
    hyper: QwenVlHyper
    name: str = "mock"

    def classify_clip(
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
    ) -> tuple[str, float]:
        rng = mock_rng(
            scheme_version=self.cfg.mock_scheme_version,
            salt=self.cfg.mock_seed_salt,
            video_id=video_id,
            stage="actions",
            clip_idx=clip_idx,
        )
        label = str(rng.choice(MOCK_ACTION_VOCAB))
        score = float(rng.uniform(0.6, 0.95))
        return label, score


_DEFAULT_VLM_CONFIDENCE = 0.8


def _normalise_action_output(text: str) -> str:
    """Map a free-form VLM string to one of MOCK_ACTION_VOCAB (best effort)."""
    t = text.strip().lower()
    for v in MOCK_ACTION_VOCAB:
        if v in t:
            return v
    # Common Chinese aliases — small map, kept inline for now.
    aliases: dict[str, str] = {
        "伸手": "reach", "靠近": "reach",
        "抓": "grasp", "握": "grasp", "捏": "grasp",
        "抬": "lift", "举": "lift",
        "移动": "move", "推": "move", "拉": "move",
        "旋转": "rotate", "拧": "rotate", "转": "rotate",
        "放": "place", "放置": "place",
        "松": "release", "放开": "release", "释放": "release",
    }
    for cn, en in aliases.items():
        if cn in text:
            return en
    return "move"  # fall-back default


@dataclass
class DashScopeActionLabeler:
    """Real backend: VLM 0-shot action classification."""

    cfg: AnnotateConfig
    hyper: QwenVlHyper
    name: str = "dashscope"

    def __post_init__(self) -> None:
        self._api_key = require_api_key()
        self._model = (
            os.environ.get("MMPIPE_DASHSCOPE_MODEL", "").strip() or self.hyper.variant
        )

    def classify_clip(
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
    ) -> tuple[str, float]:
        frames = extract_clip_frames(
            video_path,
            t_start_s=t_start_s,
            t_end_s=t_end_s,
            max_frames=self.hyper.max_clip_frames,
            long_edge_px=self.hyper.max_frame_resize_long_edge,
        )
        prompt = self.hyper.action_prompt_template.format(
            vocab=" / ".join(MOCK_ACTION_VOCAB),
        )
        raw = call_qwen_vl(
            api_key=self._api_key,
            model=self._model,
            prompt=prompt,
            frame_bytes_list=frames,
            temperature=self.hyper.temperature,
            top_p=self.hyper.top_p,
            max_tokens=8,  # category names are short
        )
        label = _normalise_action_output(raw)
        return label, _DEFAULT_VLM_CONFIDENCE


def _real_unwired(backend: str) -> NotImplementedError:
    return NotImplementedError(
        f"Actions backend '{backend}' is not wired. "
        f"Available: mock | dashscope. Set MMPIPE_ACTIONS_BACKEND accordingly."
    )


def build_action_labeler(
    cfg: AnnotateConfig, hyper: QwenVlHyper | None = None
):
    backend = settings.backend("actions", fallback_key="ANNOTATOR_BACKEND")
    hyper = hyper or QwenVlHyper()
    if backend == "mock":
        return MockActionLabeler(cfg=cfg, hyper=hyper)
    if backend == "dashscope":
        return DashScopeActionLabeler(cfg=cfg, hyper=hyper)
    raise _real_unwired(backend)


__all__ = [
    "MockActionLabeler",
    "DashScopeActionLabeler",
    "build_action_labeler",
    "MOCK_ACTION_VOCAB",
]
