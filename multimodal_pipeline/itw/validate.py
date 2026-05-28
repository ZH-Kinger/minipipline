from __future__ import annotations

import csv
from pathlib import Path

from ..config import ITWConfig
from .schemas import (
    Calibration,
    DiscoveryResult,
    ValidationIssue,
    ValidationResult,
)


def _read_timestamps_csv(path: Path) -> list[tuple[int, float]]:
    rows: list[tuple[int, float]] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if header is None:
            return rows
        # Header is `frame_index, timestamp_s` but we don't enforce names — just
        # accept 2-column rows.
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            if len(row) < 2:
                continue
            rows.append((int(row[0]), float(row[1])))
    return rows


def _is_monotonic_increasing(values: list[float]) -> tuple[bool, int]:
    for i in range(1, len(values)):
        if values[i] <= values[i - 1]:
            return False, i
    return True, -1


def _contiguous_index_check(rows: list[tuple[int, float]]) -> tuple[bool, int]:
    for expected, (actual, _) in enumerate(rows):
        if expected != actual:
            return False, expected
    return True, -1


def validate(
    discovery: DiscoveryResult,
    calibration: Calibration,
    cfg: ITWConfig,
) -> ValidationResult:
    issues: list[ValidationIssue] = []

    # 1. Re-surface discovery mismatches as validation issues.
    for mm in discovery.mismatches:
        sev = "error" if mm.actual_file is None else "warn"
        issues.append(
            ValidationIssue(
                code="DISCOVERY_MISMATCH",
                severity=sev,
                message=f"{mm.manifest_name}: {mm.note}",
                details={
                    "manifest_name": mm.manifest_name,
                    "declared_file": mm.declared_file,
                    "actual_file": mm.actual_file,
                    "candidates": list(mm.candidates),
                },
            )
        )

    # 2. RGB + Depth timestamp CSVs.
    rgb_csv = discovery.found_files.get("rgb_head_timestamps")
    depth_csv = discovery.found_files.get("depth_head_timestamps")
    rgb_rows: list[tuple[int, float]] = []
    depth_rows: list[tuple[int, float]] = []

    if rgb_csv is None:
        issues.append(
            ValidationIssue(
                code="MISSING_RGB_TIMESTAMPS",
                severity="error",
                message=f"RGB timestamps file {cfg.rgb_timestamps_filename} missing; cannot align.",
            )
        )
    else:
        rgb_rows = _read_timestamps_csv(rgb_csv)
        ok, bad_idx = _contiguous_index_check(rgb_rows)
        if not ok:
            issues.append(
                ValidationIssue(
                    code="RGB_FRAME_INDEX_NON_CONTIGUOUS",
                    severity="error",
                    message=f"RGB frame indices are not contiguous; first gap at row {bad_idx}.",
                )
            )
        ok2, bad_idx2 = _is_monotonic_increasing([t for _, t in rgb_rows])
        if not ok2:
            issues.append(
                ValidationIssue(
                    code="RGB_TIMESTAMPS_NON_MONOTONIC",
                    severity="error",
                    message=f"RGB timestamps are not monotonically increasing; first violation at row {bad_idx2}.",
                )
            )

    if depth_csv is None:
        issues.append(
            ValidationIssue(
                code="MISSING_DEPTH_TIMESTAMPS",
                severity="warn",
                message=f"Depth timestamps file {cfg.depth_timestamps_filename} missing; depth alignment will use 1:1 frame mapping.",
            )
        )
    else:
        depth_rows = _read_timestamps_csv(depth_csv)
        ok, bad_idx = _contiguous_index_check(depth_rows)
        if not ok:
            issues.append(
                ValidationIssue(
                    code="DEPTH_FRAME_INDEX_NON_CONTIGUOUS",
                    severity="error",
                    message=f"Depth frame indices are not contiguous; first gap at row {bad_idx}.",
                )
            )
        ok2, bad_idx2 = _is_monotonic_increasing([t for _, t in depth_rows])
        if not ok2:
            issues.append(
                ValidationIssue(
                    code="DEPTH_TIMESTAMPS_NON_MONOTONIC",
                    severity="error",
                    message=f"Depth timestamps are not monotonically increasing; first violation at row {bad_idx2}.",
                )
            )

    # 3. Cross-check frame counts against config.json.
    rgb_item = discovery.manifest.item("rgb_head")
    depth_item = discovery.manifest.item("depth_head")
    if rgb_item and rgb_item.count is not None and rgb_rows:
        if rgb_item.count != len(rgb_rows):
            issues.append(
                ValidationIssue(
                    code="RGB_FRAME_COUNT_MISMATCH",
                    severity="warn",
                    message=(
                        f"Manifest declares rgb_head.count={rgb_item.count} but rgb_head.csv has "
                        f"{len(rgb_rows)} rows."
                    ),
                )
            )
    if depth_item and depth_item.count is not None and depth_rows:
        if depth_item.count != len(depth_rows):
            issues.append(
                ValidationIssue(
                    code="DEPTH_FRAME_COUNT_MISMATCH",
                    severity="warn",
                    message=(
                        f"Manifest declares depth_head.count={depth_item.count} but depth_head.csv has "
                        f"{len(depth_rows)} rows."
                    ),
                )
            )
    if rgb_rows and depth_rows and len(rgb_rows) != len(depth_rows):
        issues.append(
            ValidationIssue(
                code="RGB_DEPTH_FRAME_COUNT_MISMATCH",
                severity="error",
                message=(
                    f"RGB ({len(rgb_rows)} frames) and depth ({len(depth_rows)} frames) counts differ; "
                    "1:1 alignment is not possible."
                ),
            )
        )

    # 4. Sanity check duration vs task_info.
    if rgb_rows:
        observed_duration = rgb_rows[-1][1] - rgb_rows[0][1] + (1.0 / cfg.nominal_fps)
        declared = discovery.task_info.duration_s
        if declared > 0 and abs(observed_duration - declared) > 0.5:
            issues.append(
                ValidationIssue(
                    code="DURATION_MISMATCH",
                    severity="warn",
                    message=(
                        f"task_info.duration_s={declared:.3f} disagrees with rgb-csv "
                        f"observed duration={observed_duration:.3f} (>0.5s)."
                    ),
                )
            )

    # 5. Calibration sanity.
    if calibration.rgb.fx <= 0 or calibration.rgb.fy <= 0:
        issues.append(
            ValidationIssue(
                code="CALIBRATION_INVALID_INTRINSICS",
                severity="error",
                message=f"RGB intrinsics non-positive: fx={calibration.rgb.fx}, fy={calibration.rgb.fy}.",
            )
        )
    if not calibration.cross_check_passed:
        for note in calibration.cross_check_notes:
            issues.append(
                ValidationIssue(
                    code="CALIBRATION_CROSS_CHECK",
                    severity="warn",
                    message=note,
                )
            )
    if calibration.imu is None:
        issues.append(
            ValidationIssue(
                code="CALIBRATION_NO_IMU",
                severity="warn",
                message="IMU calibration block missing from kalibr file; IMU alignment will use defaults.",
            )
        )

    return ValidationResult(issues=tuple(issues))


__all__ = ["validate"]
