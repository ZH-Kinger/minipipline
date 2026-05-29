"""Per-frame derived features from the auxiliary modalities (IMU + audio).

These turn the high-rate sidecar streams Layer 1 already cropped into NIR
(`imu_cropped.npz`, `audio_cropped.wav`) into per-RGB-frame vectors aligned to
the camera timeline, so Layer 3 can store them alongside each frame.

Pure numpy + stdlib ``wave`` — no librosa dependency.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

IMU_DIM = 6  # acc(3) + gyro(3)

# Contact-phase labels (int8).
PHASE_PRE = 0      # before the first detected contact transient
PHASE_CONTACT = 1  # within a contact-spike window
PHASE_POST = 2     # after the last detected contact transient


def imu_per_frame(nir_dir: Path, frame_ts: np.ndarray) -> np.ndarray:
    """Aggregate IMU samples to per-frame (N, 6) = [acc xyz, gyro xyz].

    For each frame timestamp, averages all IMU samples falling in the interval
    centred on that frame (half a frame-period either side). Frames with no
    samples in range fall back to nearest-sample. NaN-free output.
    """
    n = int(frame_ts.shape[0])
    out = np.zeros((n, IMU_DIM), dtype=np.float32)
    p = nir_dir / "imu_cropped.npz"
    if not p.exists() or n == 0:
        return out
    z = np.load(p)
    ts = np.asarray(z["timestamps_s"], dtype=np.float64)
    acc = np.asarray(z["acc"], dtype=np.float32)
    gyr = np.asarray(z["gyr"], dtype=np.float32)
    if ts.size == 0:
        return out
    # Frame period (median spacing) → half-window for binning.
    if n >= 2:
        dt = float(np.median(np.diff(frame_ts)))
    else:
        dt = 1.0 / 30.0
    half = max(dt * 0.5, 1e-4)
    for i in range(n):
        t = float(frame_ts[i])
        mask = (ts >= t - half) & (ts < t + half)
        if np.any(mask):
            out[i, :3] = acc[mask].mean(axis=0)
            out[i, 3:] = gyr[mask].mean(axis=0)
        else:
            j = int(np.argmin(np.abs(ts - t)))
            out[i, :3] = acc[j]
            out[i, 3:] = gyr[j]
    return out


def _read_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    """Read a WAV as mono float32 in [-1, 1]; returns (samples, sample_rate)."""
    with wave.open(str(path), "rb") as w:
        n_ch = w.getnchannels()
        rate = w.getframerate()
        sampwidth = w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if sampwidth == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
    if n_ch > 1:
        data = data.reshape(-1, n_ch).mean(axis=1)
    return data, rate


def contact_phase_per_frame(
    nir_dir: Path, frame_ts: np.ndarray, *, spike_z: float = 3.0,
) -> np.ndarray:
    """Per-frame contact phase (N,) int8 from audio energy transients.

    Computes a short-time energy envelope over the cropped audio, flags frames
    whose energy exceeds ``mean + spike_z*std`` as contact transients, then
    labels frames PRE (before first spike), CONTACT (at/around a spike), POST
    (after last spike). Silent/absent audio → all PRE (0), which is harmless
    for non-contact tasks.
    """
    n = int(frame_ts.shape[0])
    phase = np.zeros(n, dtype=np.int8)
    p = nir_dir / "audio_cropped.wav"
    if not p.exists() or n == 0:
        return phase
    try:
        sig, rate = _read_wav_mono(p)
    except Exception:
        return phase
    if sig.size == 0 or rate <= 0:
        return phase

    # Per-frame energy: sum of squares of audio samples in each frame window.
    # Audio is assumed to start at the first frame timestamp (cropped to align).
    t0 = float(frame_ts[0])
    energy = np.zeros(n, dtype=np.float64)
    for i in range(n):
        t = float(frame_ts[i]) - t0
        t_next = (float(frame_ts[i + 1]) - t0) if i + 1 < n else (t + 1.0 / 30.0)
        a = int(max(0, t * rate))
        b = int(min(sig.size, max(a + 1, t_next * rate)))
        if b > a:
            seg = sig[a:b]
            energy[i] = float(np.dot(seg, seg) / seg.size)

    mu, sd = float(energy.mean()), float(energy.std())
    thr = mu + spike_z * sd
    spikes = np.where(energy > thr)[0]
    if spikes.size == 0:
        return phase  # no detectable contact → all PRE

    first, last = int(spikes[0]), int(spikes[-1])
    phase[:first] = PHASE_PRE
    phase[first:last + 1] = PHASE_CONTACT
    phase[last + 1:] = PHASE_POST
    # Mark exact spike frames as CONTACT even in the POST region tail.
    phase[spikes] = PHASE_CONTACT
    return phase


__all__ = ["imu_per_frame", "contact_phase_per_frame", "IMU_DIM"]
