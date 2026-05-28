"""Algorithm hyperparameters for the Qwen-VL annotator (language + actions)."""

from __future__ import annotations

from dataclasses import dataclass

from ._base import ModelHyperMixin


_LANGUAGE_PROMPT = (
    "你是一个视频内容描述专家。给你一个人类示范任务的视频片段（{n_frames} 帧，"
    "时长 {duration_s:.1f} 秒，主任务：{task_text}），用一句话精炼描述这一段"
    "片段里发生了什么动作（重点写双手做什么）。不要超过 30 个字。"
)


_ACTION_PROMPT = (
    "从下列原子动作类别里选出最符合该视频片段的一个："
    "{vocab}。只输出类别名，不要其它内容。"
)


# Combined prompt used when both language + actions backends are dashscope:
# one API call returns both fields in a single JSON object, halving cost and
# latency vs. two separate calls.
_COMBINED_PROMPT = (
    "你是人类示范视频分析助手。给你一个片段（{n_frames} 帧，时长 "
    "{duration_s:.1f} 秒，主任务：{task_text}）。请输出严格 JSON："
    "{{\"description\": \"<一句话中文描述这一段里双手做了什么，不超过 30 个字>\", "
    "\"action\": \"<从以下类别选一个：{vocab}>\"}}。"
    "只输出 JSON 对象本身，不要 markdown 代码块或其它解释文字。"
)


@dataclass(frozen=True)
class QwenVlHyper(ModelHyperMixin):
    """Hyperparameters for the Qwen-VL annotator."""

    # Variant identifier. For DashScope this is the model name string.
    variant: str = "qwen2.5-vl-72b-instruct"

    # Generation parameters.
    temperature: float = 0.2
    max_tokens: int = 96
    top_p: float = 0.8

    # Video downsampling before sending (controls cost + latency).
    max_clip_frames: int = 16              # send at most this many frames per clip
    max_frame_resize_long_edge: int = 640  # downscale to this long edge

    # Prompt templates.
    language_prompt_template: str = _LANGUAGE_PROMPT
    action_prompt_template: str = _ACTION_PROMPT
    combined_prompt_template: str = _COMBINED_PROMPT

    # Combined-call needs more tokens for the JSON envelope + description.
    combined_max_tokens: int = 128


__all__ = ["QwenVlHyper"]
