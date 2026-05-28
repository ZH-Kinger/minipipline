from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import PipelineConfig
from .stages import (
    StageResult,
    stage_align,
    stage_annotate,
    stage_discover,
    stage_pack,
    stage_validate,
)


def run_pipeline(input_dir: Path, output_dir: Path, config: PipelineConfig) -> dict[str, Any]:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[StageResult] = [
        stage_discover(input_dir, output_dir, config),
        stage_validate(input_dir, output_dir),
        stage_align(output_dir, config),
        stage_annotate(output_dir, config),
        stage_pack(input_dir, output_dir, config),
    ]

    report = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "stages": [
            {
                "name": result.name,
                "output": str(result.output),
                "metrics": result.metrics,
            }
            for result in results
        ],
    }
    report_path = output_dir / "pipeline_report.json"
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")
    report["report_path"] = str(report_path)
    return report
