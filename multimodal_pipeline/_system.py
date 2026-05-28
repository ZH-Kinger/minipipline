from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .config import settings


_INSTALL_HINT = (
    "ffmpeg binary not found on PATH.\n"
    "  Windows : scoop install ffmpeg   (or download from ffmpeg.org and add to PATH)\n"
    "  macOS   : brew install ffmpeg\n"
    "  Linux   : apt install ffmpeg     (or your distro's package manager)\n"
    "  Or set MMPIPE_FFMPEG_PATH / MMPIPE_FFPROBE_PATH in .env to override."
)


class FFmpegMissingError(RuntimeError):
    pass


@dataclass(frozen=True)
class FFmpegBundle:
    ffmpeg: Path
    ffprobe: Path
    ffmpeg_version: str
    ffprobe_version: str


def find_binary(name: str, override: Path | None = None) -> Path:
    if override is not None:
        p = override.expanduser()
        if not p.exists():
            raise FFmpegMissingError(
                f"{name} override path does not exist: {override}\n{_INSTALL_HINT}"
            )
        return p
    located = shutil.which(name)
    if not located:
        raise FFmpegMissingError(_INSTALL_HINT)
    return Path(located)


def _probe_version(binary: Path) -> str:
    try:
        result = subprocess.run(
            [str(binary), "-version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except subprocess.SubprocessError as exc:
        raise FFmpegMissingError(f"{binary.name} found but failed to execute: {exc}") from exc
    first_line = result.stdout.splitlines()[0] if result.stdout else ""
    return first_line.strip()


def check_ffmpeg() -> FFmpegBundle:
    ffmpeg = find_binary("ffmpeg", override=settings.ffmpeg_path)
    ffprobe = find_binary("ffprobe", override=settings.ffprobe_path)
    return FFmpegBundle(
        ffmpeg=ffmpeg,
        ffprobe=ffprobe,
        ffmpeg_version=_probe_version(ffmpeg),
        ffprobe_version=_probe_version(ffprobe),
    )
