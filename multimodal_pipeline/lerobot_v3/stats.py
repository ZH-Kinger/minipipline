from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class StatsAccumulator:
    """Per-feature streaming Welford accumulator with min/max tracking.

    Stays disabled by default (LeRobotConfig.enable_real_stats=False); when
    disabled, finalize() returns a placeholder dict matching the spec
    (mean=0/std=1/min=-1/max=1 per dim).
    """

    dim: int
    name: str
    enabled: bool = True
    _count: int = 0
    _mean: np.ndarray = None  # type: ignore[assignment]
    _m2: np.ndarray = None  # type: ignore[assignment]
    _min: np.ndarray = None  # type: ignore[assignment]
    _max: np.ndarray = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self._mean = np.zeros(self.dim, dtype=np.float64)
        self._m2 = np.zeros(self.dim, dtype=np.float64)
        self._min = np.full(self.dim, np.inf, dtype=np.float64)
        self._max = np.full(self.dim, -np.inf, dtype=np.float64)

    def update(self, rows: np.ndarray) -> None:
        if not self.enabled or rows.size == 0:
            return
        if rows.ndim == 1:
            rows = rows[np.newaxis, :]
        if rows.shape[1] != self.dim:
            raise ValueError(f"{self.name}: expected dim {self.dim}, got {rows.shape[1]}")
        for row in rows:
            self._count += 1
            delta = row - self._mean
            self._mean += delta / self._count
            delta2 = row - self._mean
            self._m2 += delta * delta2
            np.minimum(self._min, row, out=self._min)
            np.maximum(self._max, row, out=self._max)

    def finalize(self) -> dict[str, list[float]]:
        if not self.enabled or self._count == 0:
            ones = np.ones(self.dim, dtype=np.float64)
            return {
                "mean": np.zeros(self.dim, dtype=np.float64).tolist(),
                "std": ones.tolist(),
                "min": (-ones).tolist(),
                "max": ones.tolist(),
                "count": int(self._count),
            }
        variance = self._m2 / max(self._count, 1)
        std = np.sqrt(variance)
        return {
            "mean": self._mean.tolist(),
            "std": std.tolist(),
            "min": self._min.tolist(),
            "max": self._max.tolist(),
            "count": int(self._count),
        }


__all__ = ["StatsAccumulator"]
