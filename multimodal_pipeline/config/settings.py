"""Runtime settings sourced from environment variables (+ optional `.env`).

Layered config strategy:

- Algorithmic / orchestration parameters → dataclass defaults in
  ``config/itw.py``, ``config/handpose.py``, ``config/lerobot.py``. Optionally
  overridable via ``--*-config <path>.json``.
- Per-model algorithm hyperparameters → ``config/models/<name>.py``.
- Environment-specific values (device, weights paths, cache dirs, backend
  switches, secrets) → ``.env`` file at repo root, exposed through the
  ``Settings`` singleton below.

The ``.env`` parser is intentionally tiny (stdlib only). It supports lines of
the form ``KEY=value``, comments starting with ``#``, blank lines, and quoted
values (single or double). It does **not** support variable interpolation or
shell-style export prefixes — keep `.env` simple.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


_ENV_PREFIX = "MMPIPE_"
# Valid backend identifiers. Open-ended: each model dispatch site is
# responsible for raising a helpful NotImplementedError for backends it
# doesn't yet wire.
_KNOWN_BACKENDS = ("mock", "real", "rule_based", "dashscope", "local")
_KNOWN_MODELS = ("geocalib", "moge2", "hawor_s1", "megasam", "hawor_s2")


def _strip_quotes(v: str) -> str:
    if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
        return v[1:-1]
    return v


def load_dotenv(path: Path | None = None, *, override: bool = False) -> dict[str, str]:
    """Parse a ``.env`` file and populate ``os.environ``.

    Returns the dict of keys that were applied (useful for diagnostics).
    By default existing env vars take precedence (``override=False``), so the
    OS environment always wins over file contents.
    """

    if path is None:
        path = _find_dotenv()
    applied: dict[str, str] = {}
    if path is None or not path.exists():
        return applied
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = _strip_quotes(value.strip())
        if not key:
            continue
        if override or key not in os.environ:
            os.environ[key] = value
            applied[key] = value
    return applied


def _find_dotenv() -> Path | None:
    """Search for a `.env` file by walking up from CWD."""

    start = Path.cwd().resolve()
    for candidate in (start, *start.parents):
        env_path = candidate / ".env"
        if env_path.exists():
            return env_path
    return None


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_path(name: str, default: Path | None) -> Path | None:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return Path(raw).expanduser()


@dataclass(frozen=True)
class Settings:
    """Runtime configuration sourced from environment variables."""

    device: str
    cache_dir: Path
    log_dir: Path
    log_level: str
    seed: int
    deterministic: bool
    ffmpeg_path: Path | None
    ffprobe_path: Path | None

    def backend(self, model: str, *, fallback_key: str = "HANDPOSE_BACKEND") -> str:
        """Resolve the backend implementation for a model slot.

        Reads ``MMPIPE_<MODEL>_BACKEND`` (e.g. ``MMPIPE_HAWOR_S1_BACKEND``).
        If unset, falls back to ``MMPIPE_<fallback_key>`` (an umbrella default
        for a group — e.g. ``HANDPOSE_BACKEND`` for the 5 handpose models, or
        ``ANNOTATOR_BACKEND`` for the 3 Layer-1.5 annotators). If both are
        unset, returns ``"mock"``.
        """

        key = f"{_ENV_PREFIX}{model.upper()}_BACKEND"
        value = os.environ.get(key)
        if value is None or not value.strip():
            value = os.environ.get(f"{_ENV_PREFIX}{fallback_key}", "mock")
        value = value.strip().lower()
        if value not in _KNOWN_BACKENDS:
            raise ValueError(
                f"{key}={value!r} is not a recognised backend "
                f"(expected one of: {', '.join(_KNOWN_BACKENDS)})"
            )
        return value

    def weights(self, model: str) -> Path | None:
        """Resolve a weights path for a model slot, or ``None`` if unset."""

        key = f"{_ENV_PREFIX}{model.upper()}_WEIGHTS"
        raw = os.environ.get(key)
        if raw is None or not raw.strip():
            return None
        return Path(raw).expanduser()

    def device_for(self, model: str) -> str:
        """Per-model device override, falling back to the global ``device``."""

        key = f"{_ENV_PREFIX}{model.upper()}_DEVICE"
        raw = os.environ.get(key)
        if raw and raw.strip():
            return raw.strip()
        return self.device


def _build_settings() -> Settings:
    return Settings(
        device=os.environ.get(f"{_ENV_PREFIX}DEVICE", "cpu").strip() or "cpu",
        cache_dir=_env_path(
            f"{_ENV_PREFIX}CACHE_DIR", Path("~/.cache/mmpipe").expanduser()
        ) or Path("~/.cache/mmpipe").expanduser(),
        log_dir=_env_path(
            f"{_ENV_PREFIX}LOG_DIR", Path("./logs").resolve()
        ) or Path("./logs").resolve(),
        log_level=os.environ.get(f"{_ENV_PREFIX}LOG_LEVEL", "INFO").strip() or "INFO",
        seed=_env_int(f"{_ENV_PREFIX}SEED", 0),
        deterministic=_env_bool(f"{_ENV_PREFIX}DETERMINISTIC", True),
        ffmpeg_path=_env_path(f"{_ENV_PREFIX}FFMPEG_PATH", None),
        ffprobe_path=_env_path(f"{_ENV_PREFIX}FFPROBE_PATH", None),
    )


# Load .env once at import time, then materialise the singleton.
load_dotenv()
settings: Settings = _build_settings()


def reload() -> Settings:
    """Re-read ``.env`` + environment variables. Returns the new ``Settings``."""

    global settings
    load_dotenv(override=True)
    settings = _build_settings()
    return settings


__all__ = [
    "Settings",
    "settings",
    "reload",
    "load_dotenv",
]
