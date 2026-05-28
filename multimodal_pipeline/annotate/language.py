"""Language description backend (Layer 1.5 — clip → natural-language text)."""

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
from ._mock import mock_rng


# Mock vocabulary of plausible Chinese descriptions. A real VLM would
# return free-form text; we keep templates short and varied so downstream
# code that does string ops doesn't break on weird outputs.
_MOCK_TEMPLATES: tuple[str, ...] = (
    "右手伸向目标物体并触碰",
    "左手稳定物体右手开始操作",
    "双手协作抬起目标",
    "右手旋转物体调整朝向",
    "左手移开干扰物",
    "右手把物体放到目标位置",
    "双手松开完成动作",
    "右手抓握物体准备移动",
    "左手扶住右手插入",
    "双手保持稳定等待",
)


@dataclass
class MockLanguageAnnotator:
    """Deterministic placeholder. No external calls."""

    cfg: AnnotateConfig
    hyper: QwenVlHyper
    name: str = "mock"

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
    ) -> str:
        rng = mock_rng(
            scheme_version=self.cfg.mock_scheme_version,
            salt=self.cfg.mock_seed_salt,
            video_id=video_id,
            stage="language",
            clip_idx=clip_idx,
        )
        return str(rng.choice(_MOCK_TEMPLATES))


@dataclass
class DashScopeLanguageAnnotator:
    """Real backend: calls Aliyun DashScope Qwen-VL multimodal API.

    The model identifier comes from ``MMPIPE_DASHSCOPE_MODEL`` (falls back to
    ``hyper.variant``). The endpoint is overridable via
    ``MMPIPE_DASHSCOPE_ENDPOINT``.
    """

    cfg: AnnotateConfig
    hyper: QwenVlHyper
    name: str = "dashscope"

    def __post_init__(self) -> None:
        # Resolve API key eagerly so the user gets a clear error before any
        # ffmpeg work is done.
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
    ) -> str:
        frames = extract_clip_frames(
            video_path,
            t_start_s=t_start_s,
            t_end_s=t_end_s,
            max_frames=self.hyper.max_clip_frames,
            long_edge_px=self.hyper.max_frame_resize_long_edge,
        )
        prompt = self.hyper.language_prompt_template.format(
            n_frames=len(frames),
            duration_s=max(0.0, t_end_s - t_start_s),
            task_text=task_text or "未指定",
        )
        text = call_qwen_vl(
            api_key=self._api_key,
            model=self._model,
            prompt=prompt,
            frame_bytes_list=frames,
            temperature=self.hyper.temperature,
            top_p=self.hyper.top_p,
            max_tokens=self.hyper.max_tokens,
        )
        # Collapse multi-line responses to a single line.
        return text.replace("\n", " ").strip()


def _real_unwired(backend: str) -> NotImplementedError:
    return NotImplementedError(
        f"Language backend '{backend}' is not wired. "
        f"Available: mock | dashscope. Set MMPIPE_LANGUAGE_BACKEND accordingly."
    )


def build_language_annotator(
    cfg: AnnotateConfig, hyper: QwenVlHyper | None = None
):
    backend = settings.backend("language", fallback_key="ANNOTATOR_BACKEND")
    hyper = hyper or QwenVlHyper()
    if backend == "mock":
        return MockLanguageAnnotator(cfg=cfg, hyper=hyper)
    if backend == "dashscope":
        return DashScopeLanguageAnnotator(cfg=cfg, hyper=hyper)
    raise _real_unwired(backend)


__all__ = [
    "MockLanguageAnnotator",
    "DashScopeLanguageAnnotator",
    "build_language_annotator",
]
