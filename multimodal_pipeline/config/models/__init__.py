"""Per-model algorithm hyperparameters.

Each model slot in Layer 2 has a frozen dataclass that captures its private
algorithm knobs (variant, batch size, sequence length, thresholds, etc.).
Real backends read these to build/configure their inference graph. Mock
backends largely ignore them but may consume the ``mock_*`` fields.

Environment-specific values (device, weights path) are sourced from
``settings`` and threaded into each backend at construction time.
"""

from __future__ import annotations

from .geocalib import GeoCalibHyper
from .hawor_s1 import HaWoRStage1Hyper
from .hawor_s2 import HaWoRStage2Hyper
from .megasam import MegaSamHyper
from .moge2 import MoGe2Hyper
from .quality import QualityHyper
from .qwen_vl import QwenVlHyper

__all__ = [
    "GeoCalibHyper",
    "MoGe2Hyper",
    "HaWoRStage1Hyper",
    "MegaSamHyper",
    "HaWoRStage2Hyper",
    "QwenVlHyper",
    "QualityHyper",
]
