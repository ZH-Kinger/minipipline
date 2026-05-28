"""Configuration package.

- ``settings`` : runtime values sourced from environment (`.env` supported).
- ``itw`` / ``handpose`` / ``lerobot`` : per-layer algorithm parameter dataclasses.
- ``models``  : per-model algorithm hyperparameter dataclasses.

Layer dataclasses are re-exported at the package level for ergonomic imports::

    from multimodal_pipeline.config import settings, ITWConfig, HandPoseConfig, LeRobotConfig
"""

from __future__ import annotations

from .annotate import AnnotateConfig
from .handpose import ExecutorMode, HandPoseConfig
from .itw import ITWConfig
from .legacy import DEFAULT_MODALITY_EXTENSIONS, PipelineConfig
from .lerobot import LeRobotConfig
from .settings import Settings, load_dotenv, reload, settings

__all__ = [
    "settings",
    "Settings",
    "load_dotenv",
    "reload",
    "ITWConfig",
    "HandPoseConfig",
    "ExecutorMode",
    "LeRobotConfig",
    "AnnotateConfig",
    "PipelineConfig",
    "DEFAULT_MODALITY_EXTENSIONS",
]
