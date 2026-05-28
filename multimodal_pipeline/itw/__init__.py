"""Layer 1 — ITW human-demonstration session ingest and normalization."""

from ..config import ITWConfig
from .discover import is_session_dir
from .pipeline import IngestRunResult, run_itw_ingest
from .schemas import (
    AlignmentResult,
    AnnotationSummary,
    Calibration,
    DiscoveryResult,
    GapAnomaly,
    IngestReport,
    ITWManifest,
    ITWManifestItem,
    NIRSession,
    TaskInfo,
    ValidationIssue,
    ValidationResult,
)

__all__ = [
    "ITWConfig",
    "ITWManifest",
    "ITWManifestItem",
    "TaskInfo",
    "Calibration",
    "DiscoveryResult",
    "ValidationIssue",
    "ValidationResult",
    "AlignmentResult",
    "AnnotationSummary",
    "GapAnomaly",
    "NIRSession",
    "IngestReport",
    "IngestRunResult",
    "run_itw_ingest",
    "is_session_dir",
]
