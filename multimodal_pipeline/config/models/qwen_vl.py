"""Algorithm hyperparameters for the Qwen-VL annotator (language + actions)."""

from __future__ import annotations

from dataclasses import dataclass

from ._base import ModelHyperMixin


_LANGUAGE_PROMPT = (
    "You are a video captioning expert for human-demonstration clips. You are "
    "given a clip ({n_frames} frames, {duration_s:.1f}s, overall task: {task_text}). "
    "Write ONE concise English sentence describing what happens in this clip "
    "(focus on what the hands do), present tense, at most 12 words. "
    "Always answer in English, regardless of the task text language."
)


_ACTION_PROMPT = (
    "Pick the single atomic-action category that best matches this clip from: "
    "{vocab}. Output only the category name, nothing else."
)


# Combined prompt used when both language + actions backends are dashscope:
# one API call returns both fields in a single JSON object, halving cost and
# latency vs. two separate calls.
#
# CoT variant: ask the model to reason about objects + hand state + best
# matching action category before emitting the final JSON. Forces stricter
# grounding and dramatically reduces parse failures vs. the original prompt.
_COMBINED_PROMPT = (
    "You are a video analysis assistant for human-demonstration clips. You are "
    "given a clip ({n_frames} frames, {duration_s:.1f}s, overall task: {task_text}).\n\n"
    "**Output a STRICT JSON object only** (no markdown fences ``` and no extra "
    "text), with this structure:\n"
    "{{\n"
    "  \"reasoning\": \"Reason briefly step by step: (1) what objects are visible; "
    "(2) what state the hands are in and what they are doing; (3) which category "
    "below the action best matches.\",\n"
    "  \"description\": \"<one English sentence describing what the hands do, "
    "present tense, third person, at most 12 words>\",\n"
    "  \"action\": \"<pick exactly one from: {vocab}>\"\n"
    "}}\n"
    "Always write the description in English, regardless of the task text language."
)

# Legacy non-CoT prompt kept for A/B rollback. Mirrors the pre-CoT behaviour.
_COMBINED_PROMPT_LEGACY = (
    "You are a video analysis assistant for human-demonstration clips. You are "
    "given a clip ({n_frames} frames, {duration_s:.1f}s, overall task: {task_text}). "
    "Output strict JSON: "
    "{{\"description\": \"<one English sentence describing what the hands do, at most 12 words>\", "
    "\"action\": \"<pick one from: {vocab}>\"}}. "
    "Output only the JSON object itself, no markdown fences or other text."
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
    max_frame_resize_long_edge: int = 512  # downscale to this long edge (640->512: ~36% fewer image tokens, small-object recall mostly preserved)

    # Adaptive frame count: short / low-duration clips carry fewer distinct
    # frames, so sending the full `max_clip_frames` is redundant. Scale the
    # frame count by clip duration (frames ≈ duration_s × adaptive_sample_fps),
    # clamped to [min_clip_frames, max_clip_frames]. Quality-neutral: only trims
    # frames a short clip never had distinct information for. Set False to always
    # send max_clip_frames.
    adaptive_frames: bool = True
    min_clip_frames: int = 4
    adaptive_sample_fps: float = 4.0

    # Prompt templates.
    language_prompt_template: str = _LANGUAGE_PROMPT
    action_prompt_template: str = _ACTION_PROMPT
    combined_prompt_template: str = _COMBINED_PROMPT

    # Combined-call needs more tokens for the CoT reasoning + JSON envelope
    # + description. ~250 tokens fits the schema with room to spare.
    combined_max_tokens: int = 256


__all__ = ["QwenVlHyper"]
