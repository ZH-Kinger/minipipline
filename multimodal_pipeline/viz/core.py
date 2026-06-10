"""Human-check visualization for packed LeRobot v3 episodes.

Renders, per episode, an overlay MP4 that draws the 21-joint hand skeleton
(projected from the stored camera-frame ``observation.hand_keypoints`` using the
session's real intrinsics) on top of the RGB video — so you can eyeball whether
the keypoints actually track the hands. Optionally renders the 16-bit depth
stream as a viewable colormap alongside.

Depends only on numpy + PIL + matplotlib (+ ffmpeg) — all already available.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw

from .._system import check_ffmpeg

# 21-keypoint MANO-style skeleton bones (index pairs into the per-hand block).
# 0 wrist; 1-4 thumb; 5-8 index; 9-12 middle; 13-16 ring; 17-20 pinky.
_BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (0, 9), (9, 10), (10, 11), (11, 12),   # middle
    (0, 13), (13, 14), (14, 15), (15, 16), # ring
    (0, 17), (17, 18), (18, 19), (19, 20), # pinky
]
_LEFT_COLOR = (0, 200, 255)    # cyan
_RIGHT_COLOR = (255, 120, 0)   # orange


def _find_nir_calibration(dataset_root: Path) -> dict | None:
    """Locate the session's calibration.json (sibling nir/<sess>/)."""
    # dataset_root = <session>/lerobot_dataset ; nir lives at <session>/nir/<sess>/
    session = dataset_root.parent
    for cand in (session / "nir").glob("*/calibration.json"):
        try:
            return json.loads(cand.read_text(encoding="utf-8"))
        except Exception:
            continue
    return None


def _intrinsics(dataset_root: Path, width: int, height: int) -> tuple[float, float, float, float]:
    """Real (fx, fy, cx, cy). Prefers NIR calibration; falls back to centred K."""
    calib = _find_nir_calibration(dataset_root)
    if calib and "rgb" in calib:
        r = calib["rgb"]
        cw = int(r.get("width", width)) or width
        ch = int(r.get("height", height)) or height
        sx, sy = width / cw, height / ch
        return (
            float(r["fx"]) * sx, float(r["fy"]) * sy,
            float(r["cx"]) * sx, float(r["cy"]) * sy,
        )
    # Fallback: assume a 60° horizontal FOV, centred principal point.
    import math
    fx = (width / 2.0) / math.tan(math.radians(60.0) / 2.0)
    return fx, fx, width / 2.0, height / 2.0


def _decode_rgb(ffmpeg: Path, mp4: Path, w: int, h: int) -> np.ndarray:
    """Decode an MP4 to (T, H, W, 3) uint8 via ffmpeg rawvideo."""
    res = subprocess.run(
        [str(ffmpeg), "-v", "error", "-i", str(mp4), "-f", "rawvideo",
         "-pix_fmt", "rgb24", "-"],
        check=True, capture_output=True,
    )
    buf = res.stdout
    fb = w * h * 3
    n = len(buf) // fb
    if n == 0:
        return np.zeros((0, h, w, 3), dtype=np.uint8)
    return np.frombuffer(buf, np.uint8, count=n * fb).reshape(n, h, w, 3).copy()


def _decode_depth_gray16(ffmpeg: Path, mkv: Path, w: int, h: int) -> np.ndarray:
    """Decode a gray16le depth MKV to (T, H, W) uint16."""
    res = subprocess.run(
        [str(ffmpeg), "-v", "error", "-i", str(mkv), "-f", "rawvideo",
         "-pix_fmt", "gray16le", "-"],
        check=True, capture_output=True,
    )
    buf = res.stdout
    fb = w * h * 2
    n = len(buf) // fb
    if n == 0:
        return np.zeros((0, h, w), dtype=np.uint16)
    return np.frombuffer(buf, "<u2", count=n * fb // 2).reshape(n, h, w).copy()


def _aa_to_rotmat(aa: np.ndarray) -> np.ndarray:
    """(3,) axis-angle → 3×3 rotation matrix (Rodrigues)."""
    theta = float(np.linalg.norm(aa))
    if theta < 1e-8:
        return np.eye(3)
    k = aa / theta
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]], dtype=np.float64)
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def _project_pts(pts_cam: np.ndarray, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """(M,3) camera-frame metres → (M,2) pixels. NaN where behind camera."""
    out = np.full((pts_cam.shape[0], 2), np.nan, dtype=np.float64)
    z = pts_cam[:, 2]
    ok = np.isfinite(z) & (z > 1e-3) & np.all(np.isfinite(pts_cam), axis=1)
    out[ok, 0] = fx * pts_cam[ok, 0] / z[ok] + cx
    out[ok, 1] = fy * pts_cam[ok, 1] / z[ok] + cy
    return out


# Wrist-frame axis colors (X=red, Y=green, Z=blue) — standard RGB convention.
_AXIS_COLORS = ((255, 60, 60), (60, 255, 60), (80, 140, 255))


def _draw_wrist_axes(
    draw: ImageDraw.ImageDraw, transl_cam: np.ndarray, orient_aa_cam: np.ndarray,
    fx: float, fy: float, cx: float, cy: float, axis_len_m: float = 0.05,
) -> None:
    """Draw the wrist coordinate frame (3 axes) from camera-frame state.

    `transl_cam` (3,) is the wrist position and `orient_aa_cam` (3,) its
    orientation, both in the camera frame (sourced from observation.state). A
    correctly-fitted wrist frame should sit at the wrist with axes following
    the hand's anatomy — this is the direct visual check for the real MANO
    `global_orient` wiring (keypoints alone don't show orientation).
    """
    if not (np.all(np.isfinite(transl_cam)) and np.all(np.isfinite(orient_aa_cam))):
        return
    if np.linalg.norm(orient_aa_cam) < 1e-9 and np.linalg.norm(transl_cam) < 1e-9:
        return
    R = _aa_to_rotmat(orient_aa_cam)
    ends = transl_cam[None, :] + axis_len_m * R.T  # rows: X,Y,Z axis endpoints
    pts = _project_pts(np.vstack([transl_cam[None, :], ends]), fx, fy, cx, cy)
    origin = pts[0]
    if not np.all(np.isfinite(origin)):
        return
    for a in range(3):
        end = pts[1 + a]
        if np.all(np.isfinite(end)):
            draw.line([tuple(origin), tuple(end)], fill=_AXIS_COLORS[a], width=3)


def _project(kp_cam: np.ndarray, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """(21,3) camera-frame metres → (21,2) pixels. NaN where invalid/behind."""
    out = np.full((kp_cam.shape[0], 2), np.nan, dtype=np.float64)
    z = kp_cam[:, 2]
    ok = np.isfinite(z) & (z > 1e-3) & np.all(np.isfinite(kp_cam), axis=1)
    out[ok, 0] = fx * kp_cam[ok, 0] / z[ok] + cx
    out[ok, 1] = fy * kp_cam[ok, 1] / z[ok] + cy
    return out


def _draw_hand(draw: ImageDraw.ImageDraw, pts: np.ndarray, color: tuple[int, int, int]) -> None:
    for a, b in _BONES:
        pa, pb = pts[a], pts[b]
        if np.all(np.isfinite(pa)) and np.all(np.isfinite(pb)):
            draw.line([tuple(pa), tuple(pb)], fill=color, width=2)
    for p in pts:
        if np.all(np.isfinite(p)):
            x, y = float(p[0]), float(p[1])
            draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=color)


def _depth_to_rgb(depth: np.ndarray) -> np.ndarray:
    """uint16 depth (H,W) → (H,W,3) uint8 via a perceptual colormap."""
    import matplotlib.cm as cm
    valid = depth[depth > 0]
    if valid.size:
        lo, hi = np.percentile(valid, 2), np.percentile(valid, 98)
    else:
        lo, hi = 0, 1
    norm = np.clip((depth.astype(np.float32) - lo) / max(hi - lo, 1.0), 0, 1)
    rgba = cm.get_cmap("turbo")(norm)
    rgb = (rgba[..., :3] * 255).astype(np.uint8)
    rgb[depth == 0] = 0  # holes black
    return rgb


def render_episode(
    dataset_root: Path, ep_idx: int, out_dir: Path, *, with_depth: bool = False,
    with_raw: bool = False, fps: float = 30.0,
) -> Path:
    """Render one episode's keypoint-overlay MP4.

    Panels, left→right: [raw RGB if with_raw] | overlay | [depth if with_depth].
    The ``with_raw`` side-by-side lets you compare the unannotated frame against
    the keypoint/axis overlay to judge fit.
    """
    ffb = check_ffmpeg()
    info = json.loads((dataset_root / "meta" / "info.json").read_text(encoding="utf-8"))
    feat = info["features"]
    h, w, _ = feat["observation.images.ego"]["shape"]
    fx, fy, cx, cy = _intrinsics(dataset_root, w, h)

    # Per-episode keypoint rows from the data parquet.
    data = pq.read_table(dataset_root / "data" / "chunk-000" / "file-000.parquet").to_pydict()
    ep_col = data["episode_index"]
    rows = [i for i, e in enumerate(ep_col) if int(e) == ep_idx]
    if not rows:
        raise ValueError(f"episode {ep_idx} has no rows in data parquet")
    kp_flat = [data["observation.hand_keypoints"][i] for i in rows]
    # Camera-frame wrist state (transl + orient) for the orientation-axis overlay.
    state_rows = [data["observation.state"][i] for i in rows]

    rgb_mp4 = dataset_root / "videos" / "observation.images.ego" / "chunk-000" / f"episode_{ep_idx:06d}.mp4"
    frames = _decode_rgb(ffb.ffmpeg, rgb_mp4, w, h)
    T = min(len(frames), len(kp_flat))

    depth_rgb = None
    if with_depth:
        depth_mkv = dataset_root / "videos" / "observation.images.depth" / "chunk-000" / f"episode_{ep_idx:06d}.mkv"
        if depth_mkv.exists():
            d = _decode_depth_gray16(ffb.ffmpeg, depth_mkv, w, h)
            depth_rgb = d

    out_dir.mkdir(parents=True, exist_ok=True)
    out_mp4 = out_dir / f"episode_{ep_idx:06d}_overlay.mp4"

    # Pipe composed RGB frames into ffmpeg.
    n_panels = 1 + (1 if with_raw else 0) + (1 if (with_depth and depth_rgb is not None) else 0)
    out_w = w * n_panels
    proc = subprocess.Popen(
        [str(ffb.ffmpeg), "-y", "-loglevel", "error", "-f", "rawvideo",
         "-pix_fmt", "rgb24", "-s", f"{out_w}x{h}", "-r", f"{fps}",
         "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
         str(out_mp4)],
        stdin=subprocess.PIPE,
    )
    assert proc.stdin is not None
    for t in range(T):
        raw_frame = frames[t].copy() if with_raw else None  # clean snapshot pre-draw
        img = Image.fromarray(frames[t])
        draw = ImageDraw.Draw(img)
        kp = np.asarray(kp_flat[t], dtype=np.float64).reshape(2, 21, 3)
        _draw_hand(draw, _project(kp[0], fx, fy, cx, cy), _LEFT_COLOR)
        _draw_hand(draw, _project(kp[1], fx, fy, cx, cy), _RIGHT_COLOR)
        # Wrist orientation axes from observation.state (camera frame). Layout:
        # left transl (0:3) + orient (3:6); right transl (61:64) + orient (64:67).
        st = np.asarray(state_rows[t], dtype=np.float64)
        if st.shape[0] >= 67:
            _draw_wrist_axes(draw, st[0:3], st[3:6], fx, fy, cx, cy)
            _draw_wrist_axes(draw, st[61:64], st[64:67], fx, fy, cx, cy)
        panels = []
        if raw_frame is not None:
            panels.append(raw_frame)           # raw (left)
        panels.append(np.asarray(img))         # overlay
        if with_depth and depth_rgb is not None and t < len(depth_rgb):
            panels.append(_depth_to_rgb(depth_rgb[t]))
        composed = np.concatenate(panels, axis=1)
        proc.stdin.write(composed.astype(np.uint8).tobytes())
    proc.stdin.close()
    proc.wait()
    return out_mp4


def render_pointcloud(
    dataset_root: Path, ep_idx: int, out_dir: Path, *,
    frame: int | None = None, depth_scale: float = 0.001, stride: int = 5,
) -> Path:
    """Render a 3D point-cloud 'world' view for one episode frame.

    Back-projects the metric depth map into camera-frame 3D points (coloured by
    RGB) and overlays the 21-joint hand skeleton in the SAME camera frame, so
    you can see the hands sitting on the reconstructed scene geometry. Depth is
    gray16le millimetres → metres via ``depth_scale`` (matches the keypoints'
    metric frame). Saved as a PNG (two viewing angles).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ffb = check_ffmpeg()
    info = json.loads((dataset_root / "meta" / "info.json").read_text(encoding="utf-8"))
    h, w, _ = info["features"]["observation.images.ego"]["shape"]
    fx, fy, cx, cy = _intrinsics(dataset_root, w, h)

    data = pq.read_table(dataset_root / "data" / "chunk-000" / "file-000.parquet").to_pydict()
    rows = [i for i, e in enumerate(data["episode_index"]) if int(e) == ep_idx]
    if not rows:
        raise ValueError(f"episode {ep_idx} has no rows")
    kp_flat = [data["observation.hand_keypoints"][i] for i in rows]

    rgb_mp4 = dataset_root / "videos" / "observation.images.ego" / "chunk-000" / f"episode_{ep_idx:06d}.mp4"
    depth_mkv = dataset_root / "videos" / "observation.images.depth" / "chunk-000" / f"episode_{ep_idx:06d}.mkv"
    if not depth_mkv.exists():
        raise FileNotFoundError(f"no depth stream for episode {ep_idx}: {depth_mkv}")
    rgb = _decode_rgb(ffb.ffmpeg, rgb_mp4, w, h)
    depth = _decode_depth_gray16(ffb.ffmpeg, depth_mkv, w, h)
    T = min(len(rgb), len(depth), len(kp_flat))

    # Pick the frame with the most finite keypoints (prefer both hands) if unset.
    if frame is None:
        best, best_n = 0, -1
        for t in range(T):
            kp = np.asarray(kp_flat[t], dtype=np.float64).reshape(2, 21, 3)
            nfin = int(np.all(np.isfinite(kp), axis=2).sum())
            if nfin > best_n:
                best, best_n = t, nfin
        frame = best
    frame = min(frame, T - 1)

    kp = np.asarray(kp_flat[frame], dtype=np.float64).reshape(2, 21, 3)

    # Back-project depth → camera-frame metric points.
    d = depth[frame].astype(np.float64) * depth_scale  # metres
    ys, xs = np.mgrid[0:h:stride, 0:w:stride]
    dz = d[::stride, ::stride]
    valid = dz > 1e-3
    X = (xs[valid] - cx) * dz[valid] / fx
    Y = (ys[valid] - cy) * dz[valid] / fy
    Z = dz[valid]
    cols = rgb[frame][::stride, ::stride][valid].astype(np.float64) / 255.0

    # Frame the scene to the robust extent of the cloud (drop far-background
    # clutter so the foreground + hands read clearly).
    def _lim(a):
        return float(np.percentile(a, 1)), float(np.percentile(a, 99))
    xlo, xhi = _lim(X); zlo, zhi = _lim(Z); ylo, yhi = _lim(Y)
    rng = max(xhi - xlo, yhi - ylo, zhi - zlo, 0.1)

    fig = plt.figure(figsize=(16, 8))
    for sp, (elev, azim) in enumerate(((-75, -90), (-60, -55)), start=1):
        ax = fig.add_subplot(1, 2, sp, projection="3d")
        # Camera frame: X right, Y down, Z forward. Plot (X, Z, Y); invert Y so up
        # is up. Equal real proportions via box_aspect from data extents.
        ax.scatter(X, Z, Y, c=cols, s=2, marker=".", linewidths=0)
        for hand, col in ((0, "cyan"), (1, "orange")):
            pts = kp[hand]
            if np.all(np.isfinite(pts)):
                ax.scatter(pts[:, 0], pts[:, 2], pts[:, 1], c=col, s=45, depthshade=False)
                for a, b in _BONES:
                    ax.plot([pts[a, 0], pts[b, 0]], [pts[a, 2], pts[b, 2]],
                            [pts[a, 1], pts[b, 1]], c=col, linewidth=2.0)
        ax.set_xlim(xlo, xhi); ax.set_ylim(zlo, zhi); ax.set_zlim(ylo, yhi)
        ax.set_box_aspect((xhi - xlo, zhi - zlo, yhi - ylo))
        ax.set_xlabel("X (m)"); ax.set_ylabel("Z depth (m)"); ax.set_zlabel("Y (m)")
        ax.invert_zaxis()  # camera +Y is down → flip so up is up
        ax.view_init(elev=elev, azim=azim)
    fig.suptitle(f"episode {ep_idx} frame {frame}: depth point-cloud + hand keypoints (camera frame)")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / f"episode_{ep_idx:06d}_pointcloud.png"
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    return out_png


def visualize_dataset(
    dataset_root: str | Path, *, episodes: list[int] | None = None,
    with_depth: bool = False, with_raw: bool = False, pointcloud: bool = False,
    out_dir: str | Path | None = None,
) -> list[Path]:
    """Render overlays (or 3D point clouds) for the given episodes (default: first 3)."""
    dataset_root = Path(dataset_root)
    info = json.loads((dataset_root / "meta" / "info.json").read_text(encoding="utf-8"))
    fps = float(info.get("fps", 30.0))
    total = int(info.get("total_episodes", 0))
    if episodes is None:
        episodes = list(range(min(3, total)))
    out = Path(out_dir) if out_dir else dataset_root / "viz"
    written: list[Path] = []
    for ep in episodes:
        if pointcloud:
            written.append(render_pointcloud(dataset_root, ep, out))
        else:
            written.append(
                render_episode(dataset_root, ep, out, with_depth=with_depth,
                               with_raw=with_raw, fps=fps)
            )
    return written


__all__ = ["visualize_dataset", "render_episode"]
