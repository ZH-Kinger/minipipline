"""Layer 2 model backends.

Each model slot in the hand-pose pipeline exposes a ``XBackend`` class with an
``infer(...)`` method whose input/output contract is fixed by the dataclasses
in ``handpose.schemas``.

Backend selection is environment-driven:

- ``MMPIPE_<MODEL>_BACKEND=mock`` (default) → the deterministic mock in this
  package.
- ``MMPIPE_<MODEL>_BACKEND=real`` → not yet wired; raises ``NotImplementedError``
  with a clear message at construction time.

The ``build_*_backend`` helpers below centralise that dispatch. The orchestrator
calls them instead of constructing classes directly.
"""

from __future__ import annotations

from ...config import HandPoseConfig
from ...config.models import (
    GeoCalibHyper,
    HaWoRStage1Hyper,
    HaWoRStage2Hyper,
    MegaSamHyper,
    MoGe2Hyper,
)
from ...config.settings import settings as _settings
from ._mock import derive_seed, mock_rng
from .geocalib import GeoCalibBackend
from .hawor_s1 import HaWoRStage1Backend
from .hawor_s2 import HaWoRStage2Backend
from .megasam import MegaSAMBackend
from .moge2 import MoGe2Backend


def _real_unwired(model: str) -> NotImplementedError:
    env_key = f"MMPIPE_{model.upper()}_BACKEND"
    return NotImplementedError(
        f"Real backend for '{model}' is not wired in this build. "
        f"Set {env_key}=mock (or unset it) to use the mock implementation, "
        f"or implement and register the real backend in "
        f"multimodal_pipeline/handpose/models/{model}.py."
    )


def build_geocalib_backend(
    cfg: HandPoseConfig, hyper: GeoCalibHyper | None = None
) -> GeoCalibBackend:
    chosen = _settings.backend("geocalib")
    hyper = hyper or GeoCalibHyper()
    if chosen == "mock":
        return GeoCalibBackend(cfg=cfg, hyper=hyper)
    raise _real_unwired("geocalib")


def build_moge2_backend(
    cfg: HandPoseConfig, hyper: MoGe2Hyper | None = None
) -> MoGe2Backend:
    chosen = _settings.backend("moge2")
    hyper = hyper or MoGe2Hyper()
    if chosen == "mock":
        return MoGe2Backend(cfg=cfg, hyper=hyper)
    raise _real_unwired("moge2")


def build_hawor_s1_backend(
    cfg: HandPoseConfig, hyper: HaWoRStage1Hyper | None = None
) -> HaWoRStage1Backend:
    chosen = _settings.backend("hawor_s1")
    hyper = hyper or HaWoRStage1Hyper()
    if chosen == "mock":
        return HaWoRStage1Backend(cfg=cfg, hyper=hyper)
    raise _real_unwired("hawor_s1")


def build_megasam_backend(
    cfg: HandPoseConfig, hyper: MegaSamHyper | None = None
) -> MegaSAMBackend:
    chosen = _settings.backend("megasam")
    hyper = hyper or MegaSamHyper()
    if chosen == "mock":
        return MegaSAMBackend(cfg=cfg, hyper=hyper)
    raise _real_unwired("megasam")


def build_hawor_s2_backend(
    cfg: HandPoseConfig, hyper: HaWoRStage2Hyper | None = None
) -> HaWoRStage2Backend:
    chosen = _settings.backend("hawor_s2")
    hyper = hyper or HaWoRStage2Hyper()
    if chosen == "mock":
        return HaWoRStage2Backend(cfg=cfg, hyper=hyper)
    raise _real_unwired("hawor_s2")


__all__ = [
    "derive_seed",
    "mock_rng",
    "GeoCalibBackend",
    "MoGe2Backend",
    "HaWoRStage1Backend",
    "MegaSAMBackend",
    "HaWoRStage2Backend",
    "build_geocalib_backend",
    "build_moge2_backend",
    "build_hawor_s1_backend",
    "build_megasam_backend",
    "build_hawor_s2_backend",
]
