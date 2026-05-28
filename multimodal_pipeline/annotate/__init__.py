"""Layer 1.5 — automatic annotation (language description, action labels, frame quality).

All three components default to deterministic ``mock`` backends. To wire real
implementations:

- ``language``  → DashScope Qwen-VL or local Qwen-VL via ``language.py``
- ``actions``   → VLM 0-shot classifier via ``actions.py``
- ``quality``   → OpenCV Laplacian-variance rule-based scorer via ``quality.py``

Set ``MMPIPE_LANGUAGE_BACKEND`` / ``MMPIPE_ACTIONS_BACKEND`` /
``MMPIPE_QUALITY_BACKEND`` (or the umbrella ``MMPIPE_ANNOTATOR_BACKEND``) to
switch between mock and real at runtime.
"""

from __future__ import annotations

from ..config import AnnotateConfig
from .actions import MockActionLabeler, build_action_labeler
from .language import MockLanguageAnnotator, build_language_annotator
from .orchestrator import AnnotateRunResult, run_annotate_pipeline
from .quality import MockQualityScorer, build_quality_scorer
from .schemas import ClipAnnotation, FrameQualityRow

__all__ = [
    "AnnotateConfig",
    "AnnotateRunResult",
    "ClipAnnotation",
    "FrameQualityRow",
    "run_annotate_pipeline",
    "build_language_annotator",
    "build_action_labeler",
    "build_quality_scorer",
    "MockLanguageAnnotator",
    "MockActionLabeler",
    "MockQualityScorer",
]
