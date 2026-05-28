"""Deterministic mock helpers for Layer 1.5 backends.

Same BLAKE2b-derived seed pattern as ``handpose.models._mock``: given a
fixed (scheme_version, salt, video_id, stage, clip_idx) tuple, two runs
produce byte-identical mock outputs.
"""

from __future__ import annotations

import hashlib

import numpy as np


# A small canonical action vocabulary for the mock action labeller.
MOCK_ACTION_VOCAB: tuple[str, ...] = (
    "reach",
    "grasp",
    "lift",
    "move",
    "rotate",
    "place",
    "release",
)


def derive_seed(
    *,
    scheme_version: str,
    salt: str,
    video_id: str,
    stage: str,
    clip_idx: int | None = None,
    frame_idx: int | None = None,
) -> int:
    h = hashlib.blake2b(digest_size=8)
    for part in (scheme_version, salt, video_id, stage):
        h.update(part.encode("utf-8"))
        h.update(b"|")
    h.update((str(clip_idx) if clip_idx is not None else "*").encode("utf-8"))
    h.update(b"|")
    h.update((str(frame_idx) if frame_idx is not None else "*").encode("utf-8"))
    return int.from_bytes(h.digest(), "little", signed=False)


def mock_rng(**kwargs) -> np.random.Generator:
    return np.random.default_rng(derive_seed(**kwargs))


__all__ = ["MOCK_ACTION_VOCAB", "derive_seed", "mock_rng"]
