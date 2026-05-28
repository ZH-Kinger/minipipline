from __future__ import annotations

import json
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import ITWConfig
from .schemas import NIRSession


@dataclass(frozen=True)
class PackResult:
    shard_path: Path
    index_path: Path
    bytes_written: int
    files_packed: int


def _make_tarinfo(name: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = size
    info.mode = 0o644
    info.mtime = 0  # deterministic for byte-identical reruns
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    return info


def pack_session_to_tar(
    nir_session: NIRSession,
    shards_root: Path,
    cfg: ITWConfig,
    shard_index: int = 0,
) -> PackResult:
    """Bundle a NIR session into a WebDataset-style tar shard.

    Inside the tar each file is prefixed with `<session_id>.` so multiple
    sessions can later coexist in the same shard (when batch mode is enabled).
    """
    shards_root.mkdir(parents=True, exist_ok=True)
    shard_path = shards_root / f"shard-{shard_index:06d}.tar"
    index_path = shards_root / "package_index.jsonl"

    bytes_written = 0
    files_packed = 0
    packed_names: list[str] = []

    with tarfile.open(shard_path, "w") as tar:
        # Pack everything inside the NIR session dir, deterministic name order.
        for source_path in sorted(nir_session.root.iterdir()):
            if not source_path.is_file():
                continue
            arcname = f"{nir_session.session_id}.{source_path.name}"
            payload = source_path.read_bytes()
            info = _make_tarinfo(arcname, len(payload))
            import io
            tar.addfile(info, io.BytesIO(payload))
            bytes_written += len(payload)
            files_packed += 1
            packed_names.append(arcname)

    index_entry: dict[str, Any] = {
        "shard": shard_path.name,
        "session_id": nir_session.session_id,
        "files": packed_names,
        "frame_count": nir_session.frame_count,
        "fps": nir_session.fps,
        "duration_s": nir_session.duration_s,
        "has_depth": nir_session.has_depth,
        "has_imu": nir_session.has_imu,
        "has_audio": nir_session.has_audio,
        "has_hand_keypoints": nir_session.has_hand_keypoints,
    }
    # Append-style: one JSONL row per session.
    with index_path.open("a", encoding="utf-8") as fh:
        json.dump(index_entry, fh, ensure_ascii=False, sort_keys=True)
        fh.write("\n")

    return PackResult(
        shard_path=shard_path,
        index_path=index_path,
        bytes_written=bytes_written,
        files_packed=files_packed,
    )


__all__ = ["PackResult", "pack_session_to_tar"]
