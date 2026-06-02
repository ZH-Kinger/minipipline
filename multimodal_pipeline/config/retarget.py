"""Layer 2.5 (MANO → Wuji robot retargeting) configuration.

Sourced from ``MMPIPE_RETARGET_*`` environment variables (see ``.env.example``).
The URDF search dir is shared with the model wrapper via ``MMPIPE_WUJI_URDF_DIR``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

_ENV = "MMPIPE_RETARGET_"


def _b(name: str, default: bool) -> bool:
    raw = os.environ.get(_ENV + name)
    return default if raw is None else raw.strip().lower() in {"1", "true", "yes", "on"}


def _f(name: str, default: float) -> float:
    raw = os.environ.get(_ENV + name)
    try:
        return float(raw) if raw and raw.strip() else default
    except ValueError:
        return default


def _i(name: str, default: int) -> int:
    raw = os.environ.get(_ENV + name)
    try:
        return int(raw) if raw and raw.strip() else default
    except ValueError:
        return default


def _s(name: str, default: str) -> str:
    raw = os.environ.get(_ENV + name)
    return raw.strip() if raw and raw.strip() else default


@dataclass(frozen=True)
class RetargetConfig:
    """Retargeting parameters.

    ``left_urdf`` / ``right_urdf`` accept either a vendored key (``"left"`` /
    ``"right"`` → hand-only URDF) or an explicit ``.urdf`` path (e.g. a dual-arm
    URDF once supplied). ``scale`` is auto-estimated per hand when ``None``.
    """

    enabled: bool = True
    left_urdf: str = "left"
    right_urdf: str = "right"
    smooth_weight: float = 0.01     # temporal pull toward previous-frame qpos
    n_iters: int = 30               # LM iterations per frame
    finger_map: tuple[int, ...] = (0, 1, 2, 3, 4)  # robot finger N → MANO finger
    scale: float | None = None      # None → auto (robot_span / human_span)

    @classmethod
    def from_env(cls) -> "RetargetConfig":
        scale_raw = os.environ.get(_ENV + "SCALE", "").strip()
        return cls(
            enabled=_b("ENABLED", True),
            left_urdf=_s("LEFT_URDF", "left"),
            right_urdf=_s("RIGHT_URDF", "right"),
            smooth_weight=_f("SMOOTH_WEIGHT", 0.01),
            n_iters=_i("N_ITERS", 30),
            scale=float(scale_raw) if scale_raw else None,
        )

    def urdf_for(self, hand: str) -> str:
        return self.left_urdf if hand == "left" else self.right_urdf


__all__ = ["RetargetConfig"]
