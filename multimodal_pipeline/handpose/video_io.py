from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from .._system import check_ffmpeg
from .schemas import VideoMeta


def _hash_short(path: Path) -> str:
    h = hashlib.sha1()
    h.update(str(path.resolve()).encode("utf-8"))
    try:
        stat = path.stat()
        h.update(str(stat.st_size).encode("utf-8"))
        h.update(str(int(stat.st_mtime)).encode("utf-8"))
    except OSError:
        pass
    return h.hexdigest()[:12]


def _parse_fps(rate_str: str) -> float:
    if not rate_str:
        return 0.0
    if "/" in rate_str:
        num, denom = rate_str.split("/", 1)
        denom_f = float(denom)
        if denom_f <= 0:
            return 0.0
        return float(num) / denom_f
    try:
        return float(rate_str)
    except ValueError:
        return 0.0


def probe_video(path: Path) -> VideoMeta:
    """Use ffprobe to extract VideoMeta. Falls back to nb_read_frames when nb_frames is absent."""
    ffmpeg = check_ffmpeg()
    cmd = [
        str(ffmpeg.ffprobe),
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration,nb_read_frames",
        "-show_entries", "format=duration",
        "-of", "json",
        str(path),
    ]
    res = subprocess.run(cmd, check=True, capture_output=True, text=True)
    info = json.loads(res.stdout)
    streams = info.get("streams") or []
    if not streams:
        raise ValueError(f"ffprobe found no video streams in {path}")
    s = streams[0]
    width = int(s.get("width", 0))
    height = int(s.get("height", 0))
    fps = _parse_fps(str(s.get("r_frame_rate") or s.get("avg_frame_rate") or ""))
    n_frames_raw = s.get("nb_frames") or s.get("nb_read_frames")
    n_frames = int(n_frames_raw) if n_frames_raw else 0
    duration_raw = s.get("duration") or (info.get("format") or {}).get("duration")
    duration_s = float(duration_raw) if duration_raw else (n_frames / fps if fps else 0.0)
    if n_frames == 0 and fps > 0 and duration_s > 0:
        n_frames = int(round(fps * duration_s))
    return VideoMeta(
        video_id=path.stem,
        path=Path(path),
        fps=fps,
        n_frames=n_frames,
        width=width,
        height=height,
        duration_s=duration_s,
        sha1_short=_hash_short(path),
    )


__all__ = ["probe_video"]
