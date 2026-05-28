from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


# ---------------------------------------------------------------------------
# Manifest (config.json) + task_info.json
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ITWManifestItem:
    name: str
    file: str
    description: str = ""
    fps: float | None = None
    count: int | None = None
    width: int | None = None
    height: int | None = None
    sample_rate: int | None = None


@dataclass(frozen=True)
class ITWManifest:
    version: str
    type: str
    duration_s: float
    items: tuple[ITWManifestItem, ...]
    raw: dict[str, Any] = field(default_factory=dict)

    def item(self, name: str) -> ITWManifestItem | None:
        for it in self.items:
            if it.name == name:
                return it
        return None


@dataclass(frozen=True)
class ActorInfo:
    gender: str | None = None
    height_cm: float | None = None
    weight_kg: float | None = None
    age: int | None = None


@dataclass(frozen=True)
class TaskInfo:
    task_id: str
    name: str
    scene: str
    task_scene: str
    success: int
    duration_s: float
    modality: tuple[str, ...]
    steps: tuple[str, ...]
    tags: tuple[str, ...]
    items: tuple[str, ...]
    hints: tuple[str, ...]
    actor: ActorInfo
    version: str
    request_id: str


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    distortion_model: str = "radtan"
    distortion_coeffs: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True)
class IMUParams:
    rate_hz: float
    gyroscope_noise_density: float
    gyroscope_random_walk: float
    accelerometer_noise_density: float
    accelerometer_random_walk: float
    T_cam_imu: tuple[float, ...]
    time_offset_s: float = 0.0


@dataclass(frozen=True)
class Calibration:
    rgb: CameraIntrinsics
    depth: CameraIntrinsics
    imu: IMUParams | None
    T_depth_cam: tuple[float, ...]
    device_sn: str | None = None
    source_files: tuple[str, ...] = ()
    cross_check_passed: bool = True
    cross_check_notes: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Discovery / Validation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FileMismatch:
    manifest_name: str
    declared_file: str
    actual_file: str | None
    candidates: tuple[str, ...]
    note: str


@dataclass(frozen=True)
class DiscoveryResult:
    session_dir: Path
    session_id: str
    manifest: ITWManifest
    task_info: TaskInfo
    found_files: dict[str, Path]
    extra_files: tuple[str, ...]
    mismatches: tuple[FileMismatch, ...]
    has_camera_params_subdir: bool


Severity = Literal["info", "warn", "error"]


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    severity: Severity
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ValidationResult:
    issues: tuple[ValidationIssue, ...]

    @property
    def ok(self) -> bool:
        return not any(i.severity == "error" for i in self.issues)

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "warn")

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "error")


# ---------------------------------------------------------------------------
# Alignment / Annotation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GapAnomaly:
    frame_index: int
    interval_ms: float
    nominal_ms: float


@dataclass(frozen=True)
class AlignmentResult:
    reference_stream: str
    fps_nominal: float
    fps_actual: float
    t0_s: float
    t1_s: float
    duration_s: float
    rgb_frame_count: int
    depth_frame_count: int
    depth_offset_mean_ms: float
    depth_offset_max_ms: float
    imu_kept_samples: int
    imu_dropped_samples: int
    audio_kept_samples: int
    audio_dropped_samples: int
    gap_anomalies: tuple[GapAnomaly, ...]


@dataclass(frozen=True)
class HandKeypointFrameSummary:
    frame_index: int
    left_present: bool
    right_present: bool
    excluded: bool
    exclude_reason: str | None


@dataclass(frozen=True)
class AnnotationSummary:
    left_present_frames: int
    right_present_frames: int
    both_present_frames: int
    tail_excluded_frames: int
    any_excluded_frames: int
    first_left_frame: int | None
    first_right_frame: int | None
    last_valid_frame: int | None


# ---------------------------------------------------------------------------
# NIR output handle + Report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NIRSession:
    root: Path
    session_id: str
    frame_count: int
    fps: float
    duration_s: float
    has_depth: bool
    has_audio: bool
    has_imu: bool
    has_hand_keypoints: bool


@dataclass(frozen=True)
class IngestReport:
    session_id: str
    session_dir: Path
    nir_dir: Path
    discovery: DiscoveryResult
    validation: ValidationResult
    alignment: AlignmentResult
    annotation: AnnotationSummary
    output_files: tuple[str, ...]
    calibration_ok: bool
