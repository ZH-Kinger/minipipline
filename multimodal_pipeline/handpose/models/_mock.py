from __future__ import annotations

import hashlib

import numpy as np


def derive_seed(
    *,
    scheme_version: str,
    salt: str,
    video_id: str,
    stage: str,
    frame_idx: int | None = None,
    slot: int | None = None,
) -> int:
    """Derive a deterministic uint64 seed via BLAKE2b.

    The seed is order-independent across threads: as long as the (scheme, salt,
    video, stage, frame, slot) tuple is identical, the same seed is produced.
    """
    h = hashlib.blake2b(digest_size=8)
    h.update(scheme_version.encode("utf-8"))
    h.update(b"|")
    h.update(salt.encode("utf-8"))
    h.update(b"|")
    h.update(video_id.encode("utf-8"))
    h.update(b"|")
    h.update(stage.encode("utf-8"))
    h.update(b"|")
    h.update((str(frame_idx) if frame_idx is not None else "*").encode("utf-8"))
    h.update(b"|")
    h.update((str(slot) if slot is not None else "*").encode("utf-8"))
    return int.from_bytes(h.digest(), "little", signed=False)


def mock_rng(
    *,
    scheme_version: str,
    salt: str,
    video_id: str,
    stage: str,
    frame_idx: int | None = None,
    slot: int | None = None,
) -> np.random.Generator:
    seed = derive_seed(
        scheme_version=scheme_version,
        salt=salt,
        video_id=video_id,
        stage=stage,
        frame_idx=frame_idx,
        slot=slot,
    )
    return np.random.default_rng(seed)


__all__ = ["derive_seed", "mock_rng"]
