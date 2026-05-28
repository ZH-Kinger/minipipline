"""Shared ``from_file`` / ``from_mapping`` mixin for model hyperparam dataclasses."""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from typing import Any, TypeVar


T = TypeVar("T", bound="ModelHyperMixin")


class ModelHyperMixin:
    """Mixin providing JSON loading + unknown-key rejection."""

    @classmethod
    def from_file(cls: type[T], path: str | Path | None) -> T:
        if path is None:
            return cls()  # type: ignore[call-arg]
        with Path(path).open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
        return cls.from_mapping(raw)

    @classmethod
    def from_mapping(cls: type[T], raw: dict[str, Any]) -> T:
        allowed = {f.name for f in fields(cls)}  # type: ignore[arg-type]
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ValueError(
                f"Unknown {cls.__name__} key(s): {', '.join(unknown)}"
            )
        return cls(**raw)  # type: ignore[call-arg]


__all__ = ["ModelHyperMixin"]
