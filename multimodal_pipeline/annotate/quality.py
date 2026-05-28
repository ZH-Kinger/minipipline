"""Per-frame quality scorer (Layer 1.5 — frames → blur / exposure / kept flag)."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .._system import check_ffmpeg
from ..config import AnnotateConfig, settings
from ..config.models import QualityHyper
from ._mock import mock_rng
from .schemas import FrameQualityRow


@dataclass
class MockQualityScorer:
    """Deterministic placeholder that produces plausible quality signals.

    A real backend would decode frames with ffmpeg + run OpenCV Laplacian
    variance for blur and luminance histogram for exposure. The mock skips
    decoding entirely.
    """

    cfg: AnnotateConfig
    hyper: QualityHyper
    name: str = "mock"

    def score_frames(
        self,
        *,
        video_id: str,
        n_frames: int,
        video_path: Path | None = None,  # unused by mock
        hand_visible: np.ndarray | None = None,  # shape (n_frames,) bool, optional
    ) -> list[FrameQualityRow]:
        rows: list[FrameQualityRow] = []
        if hand_visible is None:
            hand_visible = np.ones(n_frames, dtype=bool)
        blur_lo, blur_hi = self.hyper.mock_blur_range
        exp_lo, exp_hi = self.hyper.mock_exposure_range
        for i in range(n_frames):
            rng = mock_rng(
                scheme_version=self.cfg.mock_scheme_version,
                salt=self.cfg.mock_seed_salt,
                video_id=video_id,
                stage="quality",
                frame_idx=i,
            )
            blur = float(rng.uniform(blur_lo, blur_hi))
            exposure = float(rng.uniform(exp_lo, exp_hi))
            # Normalise blur into [0,1] using the configured threshold.
            blur_norm = min(1.0, blur / max(self.hyper.blur_min_laplacian_var, 1e-6))
            overall = max(
                self.hyper.mock_overall_quality_floor,
                0.5 * blur_norm + 0.5 * exposure,
            )
            kept = overall >= self.cfg.quality_overall_threshold
            if self.cfg.quality_drop_on_invalid_hand:
                kept = kept and bool(hand_visible[i])
            rows.append(
                FrameQualityRow(
                    frame_idx=i,
                    blur_score=blur,
                    exposure_score=exposure,
                    hand_visible=bool(hand_visible[i]),
                    overall_quality=overall,
                    kept=kept,
                )
            )
        return rows


def _decode_grayscale_frames(
    video_path: Path, target_height: int = 240, target_width: int = 320
) -> np.ndarray:
    """Decode the entire video to a (T, H, W) uint8 grayscale ndarray via ffmpeg.

    We downscale to keep memory bounded (240x320 ~= 75KB/frame). For an 8.5 s
    30 fps clip this is ~19 MB.
    """
    bundle = check_ffmpeg()
    cmd = [
        str(bundle.ffmpeg),
        "-v", "error",
        "-i", str(video_path),
        "-vf", f"scale={target_width}:{target_height}:flags=fast_bilinear,format=gray",
        "-f", "rawvideo",
        "-pix_fmt", "gray",
        "-",
    ]
    proc = subprocess.run(cmd, check=True, capture_output=True)
    buf = np.frombuffer(proc.stdout, dtype=np.uint8)
    if buf.size % (target_height * target_width) != 0:
        raise RuntimeError(
            f"ffmpeg returned {buf.size} bytes which is not a multiple of "
            f"{target_height}x{target_width}"
        )
    return buf.reshape(-1, target_height, target_width)


def _laplacian_variance(frame: np.ndarray) -> float:
    """Approximate OpenCV's Laplacian variance using a 4-neighbour stencil.

    Avoids the opencv-python dependency for a one-line blur metric.
    """
    f = frame.astype(np.float32)
    lap = (
        -4.0 * f
        + np.roll(f, 1, axis=0)
        + np.roll(f, -1, axis=0)
        + np.roll(f, 1, axis=1)
        + np.roll(f, -1, axis=1)
    )
    # Strip the wrap-around 1-pixel border to match OpenCV's BORDER_DEFAULT.
    inner = lap[1:-1, 1:-1]
    return float(inner.var())


@dataclass
class RuleBasedQualityScorer:
    """OpenCV-style rule-based quality.

    - Blur:   variance of a 4-neighbour Laplacian (no cv2 dep, same idea).
    - Exposure: how close the frame mean is to the target midtone.
    - Overall: weighted blend of normalised blur + exposure.
    """

    cfg: AnnotateConfig
    hyper: QualityHyper
    name: str = "rule_based"

    def score_frames(
        self,
        *,
        video_id: str,
        n_frames: int,
        video_path: Path | None = None,
        hand_visible: np.ndarray | None = None,
    ) -> list[FrameQualityRow]:
        if video_path is None or not Path(video_path).exists():
            raise RuntimeError(
                f"rule_based quality backend needs a readable source video; got {video_path!r}"
            )
        if hand_visible is None:
            hand_visible = np.ones(n_frames, dtype=bool)

        frames = _decode_grayscale_frames(Path(video_path))
        # ffmpeg may decode slightly more or fewer frames than NIR's count when
        # the source is VFR. Trim or pad to match n_frames.
        actual = frames.shape[0]
        if actual > n_frames:
            frames = frames[:n_frames]
        elif actual < n_frames:
            pad = np.repeat(frames[-1:], n_frames - actual, axis=0)
            frames = np.concatenate([frames, pad], axis=0)

        rows: list[FrameQualityRow] = []
        stride = max(1, self.hyper.sample_every_n_frames)
        last_blur = 0.0
        last_exposure = 0.0
        last_overall = 0.0
        for i in range(n_frames):
            if i % stride == 0:
                blur = _laplacian_variance(frames[i])
                mean_lum = float(frames[i].mean())
                exposure_dev = abs(mean_lum - self.hyper.exposure_target_mean)
                exposure_norm = max(
                    0.0, 1.0 - exposure_dev / max(self.hyper.exposure_max_dev, 1e-6)
                )
                blur_norm = min(1.0, blur / max(self.hyper.blur_min_laplacian_var, 1e-6))
                overall = 0.5 * blur_norm + 0.5 * exposure_norm
                last_blur, last_exposure, last_overall = blur, exposure_norm, overall
            kept = last_overall >= self.cfg.quality_overall_threshold
            if self.cfg.quality_drop_on_invalid_hand:
                kept = kept and bool(hand_visible[i])
            rows.append(
                FrameQualityRow(
                    frame_idx=i,
                    blur_score=last_blur,
                    exposure_score=last_exposure,
                    hand_visible=bool(hand_visible[i]),
                    overall_quality=last_overall,
                    kept=kept,
                )
            )
        return rows


def _real_unwired(backend: str) -> NotImplementedError:
    return NotImplementedError(
        f"Quality backend '{backend}' is not wired. "
        f"Available: mock | rule_based. Set MMPIPE_QUALITY_BACKEND accordingly."
    )


def build_quality_scorer(
    cfg: AnnotateConfig, hyper: QualityHyper | None = None
):
    backend = settings.backend("quality", fallback_key="ANNOTATOR_BACKEND")
    hyper = hyper or QualityHyper()
    if backend == "mock":
        return MockQualityScorer(cfg=cfg, hyper=hyper)
    if backend == "rule_based":
        return RuleBasedQualityScorer(cfg=cfg, hyper=hyper)
    raise _real_unwired(backend)


__all__ = ["MockQualityScorer", "RuleBasedQualityScorer", "build_quality_scorer"]
