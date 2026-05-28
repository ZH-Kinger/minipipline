from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import yaml

from ..config import ITWConfig
from .schemas import (
    Calibration,
    CameraIntrinsics,
    DiscoveryResult,
    IMUParams,
)


def _parse_kalibr_yaml(path: Path) -> dict[str, Any]:
    # The file may start with an OpenCV-style `%YAML: 1.0` directive (note the
    # colon — non-standard, PyYAML rejects it). Strip any leading `%YAML*` line
    # before parsing so we accept both kalibr and OpenCV-flavored exports.
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if lines and lines[0].lstrip().startswith("%YAML"):
        lines = lines[1:]
    cleaned = "\n".join(lines)
    return yaml.safe_load(cleaned)


def _kalibr_camera(raw: dict[str, Any]) -> CameraIntrinsics:
    intr = raw["intrinsics"]
    res = raw.get("resolution", [0, 0])
    dist = tuple(float(x) for x in raw.get("distortion_coeffs", (0.0,) * 5))
    return CameraIntrinsics(
        fx=float(intr[0]),
        fy=float(intr[1]),
        cx=float(intr[2]),
        cy=float(intr[3]),
        width=int(res[0]),
        height=int(res[1]),
        distortion_model=str(raw.get("distortion_model", "radtan")),
        distortion_coeffs=dist,
    )


def _kalibr_imu(raw: dict[str, Any]) -> IMUParams:
    return IMUParams(
        rate_hz=float(raw.get("rate_hz", 0.0)),
        gyroscope_noise_density=float(raw.get("gyroscope_noise_density", 0.0)),
        gyroscope_random_walk=float(raw.get("gyroscope_random_walk", 0.0)),
        accelerometer_noise_density=float(raw.get("accelerometer_noise_density", 0.0)),
        accelerometer_random_walk=float(raw.get("accelerometer_random_walk", 0.0)),
        T_cam_imu=tuple(float(x) for x in raw.get("T_cam_imu", (0.0,) * 16)),
        time_offset_s=float(raw.get("time_offset_s", 0.0)),
    )


def _head_param_camera(raw: dict[str, Any], width: int, height: int) -> CameraIntrinsics:
    intr = raw["intrinsic"]
    return CameraIntrinsics(
        fx=float(intr["fx"]),
        fy=float(intr["fy"]),
        cx=float(intr["ppx"]),
        cy=float(intr["ppy"]),
        width=width,
        height=height,
        distortion_model=str(intr.get("distortion_model", "radtan")),
        distortion_coeffs=(
            float(intr.get("k1", 0.0)),
            float(intr.get("k2", 0.0)),
            float(intr.get("p1", 0.0)),
            float(intr.get("p2", 0.0)),
            float(intr.get("k3", 0.0)),
        ),
    )


def _close(a: float, b: float, atol: float) -> bool:
    return math.isclose(a, b, abs_tol=atol)


def _cross_check_intrinsics(
    primary: CameraIntrinsics,
    other: CameraIntrinsics,
    label: str,
    atol: float,
) -> list[str]:
    notes: list[str] = []
    for field in ("fx", "fy", "cx", "cy"):
        pv = getattr(primary, field)
        ov = getattr(other, field)
        if not _close(pv, ov, atol):
            notes.append(f"{label}: {field}={pv} vs {ov} (diff={abs(pv - ov):.6f})")
    return notes


def load_calibration(discovery: DiscoveryResult, cfg: ITWConfig) -> Calibration:
    """Assemble a `Calibration` from kalibr yaml and/or head_param json.

    Strategy:
    - Prefer `kalibr_parameters.yaml` for intrinsics + extrinsics + IMU.
    - Use `camera_params/head_param.json` as cross-check; report differences.
    - If kalibr is missing, fall back to head_param with manifest resolution.
    """
    kalibr_path = discovery.found_files.get("kalibr_parameters")
    head_param_path = discovery.found_files.get("head_param")
    source_files: list[str] = []
    notes: list[str] = []

    if kalibr_path is not None:
        kalibr = _parse_kalibr_yaml(kalibr_path)
        source_files.append(kalibr_path.name)
        rgb = _kalibr_camera(kalibr["cam0"])
        depth = _kalibr_camera(kalibr["cam1"])
        imu = _kalibr_imu(kalibr["imu0"]) if "imu0" in kalibr else None
        T_depth_cam = tuple(float(x) for x in kalibr["cam1"].get("T_depth_cam", (0.0,) * 16))
        device_sn = kalibr.get("sn")

        if head_param_path is not None:
            with head_param_path.open("r", encoding="utf-8") as fh:
                head_raw = json.load(fh)
            source_files.append(str(head_param_path.relative_to(discovery.session_dir)))
            # head_param doesn't carry resolution; use kalibr's.
            hp_rgb = _head_param_camera(head_raw.get("rgb_camera", {}), rgb.width, rgb.height)
            hp_depth = _head_param_camera(head_raw.get("depth_camera", {}), depth.width, depth.height)
            notes.extend(_cross_check_intrinsics(rgb, hp_rgb, "rgb_intrinsics", cfg.intrinsics_match_atol))
            notes.extend(
                _cross_check_intrinsics(depth, hp_depth, "depth_intrinsics", cfg.intrinsics_match_atol)
            )

    elif head_param_path is not None:
        with head_param_path.open("r", encoding="utf-8") as fh:
            head_raw = json.load(fh)
        source_files.append(str(head_param_path.relative_to(discovery.session_dir)))
        # Need resolution from manifest items.
        rgb_item = discovery.manifest.item("rgb_head")
        depth_item = discovery.manifest.item("depth_head")
        rgb_w = rgb_item.width if rgb_item and rgb_item.width else 0
        rgb_h = rgb_item.height if rgb_item and rgb_item.height else 0
        depth_w = depth_item.width if depth_item and depth_item.width else rgb_w
        depth_h = depth_item.height if depth_item and depth_item.height else rgb_h
        rgb = _head_param_camera(head_raw.get("rgb_camera", {}), rgb_w, rgb_h)
        depth = _head_param_camera(head_raw.get("depth_camera", {}), depth_w, depth_h)
        imu = None
        T_depth_cam = tuple([1.0, 0.0, 0.0, 0.0,
                             0.0, 1.0, 0.0, 0.0,
                             0.0, 0.0, 1.0, 0.0,
                             0.0, 0.0, 0.0, 1.0])
        device_sn = None
        notes.append("kalibr_parameters.yaml missing; calibration assembled from head_param.json only")
    else:
        raise FileNotFoundError(
            "Calibration missing: neither kalibr_parameters.yaml nor "
            "camera_params/head_param.json was found in the session."
        )

    return Calibration(
        rgb=rgb,
        depth=depth,
        imu=imu,
        T_depth_cam=T_depth_cam,
        device_sn=device_sn,
        source_files=tuple(source_files),
        cross_check_passed=len(notes) == 0,
        cross_check_notes=tuple(notes),
    )


__all__ = ["load_calibration"]
