from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .align import align
from .annotate import annotate
from .calibration import load_calibration
from ..config import ITWConfig
from .discover import discover
from .normalize import normalize
from .pack_adapter import PackResult, pack_session_to_tar
from .schemas import IngestReport, NIRSession
from .validate import validate


@dataclass(frozen=True)
class IngestRunResult:
    session: NIRSession
    report: IngestReport
    pack: PackResult | None
    stage_timings: dict[str, float]


def run_itw_ingest(
    session_dir: str | Path,
    output_root: str | Path,
    cfg: ITWConfig | None = None,
    *,
    pack_tar: bool = True,
) -> IngestRunResult:
    """Run the full Layer 1 (ITW ingest) pipeline on one session.

    Output layout under `output_root`:
        nir/<session_id>/...
        shards/shard-000000.tar
        shards/package_index.jsonl
    """
    cfg = cfg or ITWConfig()
    session_path = Path(session_dir)
    output_root = Path(output_root)
    nir_root = output_root / "nir"
    shards_root = output_root / "shards"

    timings: dict[str, float] = {}

    t = time.perf_counter()
    discovery = discover(session_path, cfg)
    timings["discover"] = time.perf_counter() - t

    t = time.perf_counter()
    calibration = load_calibration(discovery, cfg)
    timings["calibration"] = time.perf_counter() - t

    t = time.perf_counter()
    validation = validate(discovery, calibration, cfg)
    timings["validate"] = time.perf_counter() - t

    t = time.perf_counter()
    aligned = align(discovery, cfg)
    timings["align"] = time.perf_counter() - t

    t = time.perf_counter()
    annotated = annotate(discovery, aligned, cfg)
    timings["annotate"] = time.perf_counter() - t

    t = time.perf_counter()
    nir_session, report = normalize(
        discovery, aligned, annotated, calibration, validation, nir_root, cfg
    )
    timings["normalize"] = time.perf_counter() - t

    pack_result: PackResult | None = None
    if pack_tar:
        t = time.perf_counter()
        pack_result = pack_session_to_tar(nir_session, shards_root, cfg)
        timings["pack"] = time.perf_counter() - t

    return IngestRunResult(
        session=nir_session,
        report=report,
        pack=pack_result,
        stage_timings=timings,
    )


__all__ = ["IngestRunResult", "run_itw_ingest"]
