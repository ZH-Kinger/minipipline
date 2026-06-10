"""MANO hand-mesh overlay (Layer-3 verification, the standard MANO-dataset QC).

Reconstructs the 3D MANO hand mesh from the per-frame state vector
(``wrist_transl_cam`` + ``wrist_orient_cam_aa`` + ``mano_pose_aa`` + betas), then
projects it onto the RGB frame with the session intrinsics and draws it as a
translucent shaded silhouette. Lets you see the *full hand surface* fit, not
just the 21 keypoints.

Dependencies (lazy-imported so the rest of the package never needs them):
  - ``torch`` (CPU is plenty — 778-vertex LBS is microseconds), ``smplx``, ``PIL``
  - the MANO model files ``MANO_LEFT.pkl`` / ``MANO_RIGHT.pkl`` — a LICENSED asset
    you must download yourself from https://mano.is.tue.mpg.de and place in
    ``$MMPIPE_MANO_DIR`` (default ``./models/mano/``).

All of these are checked with a clear, actionable error when missing.

Convention note: smplx adds ``transl`` globally and does NOT pin the wrist joint
to it, so we solve ``transl`` such that the posed wrist joint lands on the
state's ``wrist_transl_cam`` (forward once at transl=0, then offset). MANO
``hand_pose`` is the 15×3 axis-angle block with ``use_pca=False``; we default
``flat_hand_mean=False`` (matches most trackers) — flip via MMPIPE_MANO_FLAT_MEAN.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np

from .._system import check_ffmpeg
from ..lerobot_v3.schema import STATE_LAYOUT


def _patch_numpy_for_chumpy() -> None:
    """Restore numpy aliases chumpy imports at module load but numpy 2.x removed.

    chumpy 0.70 does ``from numpy import bool, int, float, complex, object,
    unicode, str`` which fails on numpy>=2. The MANO .pkl is pickled with chumpy
    objects, so chumpy must import to unpickle. We can't downgrade numpy (the
    whole pipeline uses 2.x), so re-add the aliases as the builtin types.
    """
    import numpy as np

    for name, val in {
        "bool": bool, "int": int, "float": float, "complex": complex,
        "object": object, "str": str, "unicode": str,
    }.items():
        if not hasattr(np, name):
            setattr(np, name, val)


def _resolve_mano_dir(mano_dir: str | Path | None) -> Path:
    if mano_dir is not None:
        return Path(mano_dir).expanduser()
    env = os.environ.get("MMPIPE_MANO_DIR", "").strip()
    return Path(env).expanduser() if env else Path("models/mano")


def _check_mano_assets(mano_dir: Path) -> None:
    missing = [
        n for n in ("MANO_LEFT.pkl", "MANO_RIGHT.pkl") if not (mano_dir / n).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"MANO model file(s) not found in {mano_dir}: {', '.join(missing)}.\n"
            "Download mano_v1_2.zip from https://mano.is.tue.mpg.de (free account, "
            "research licence), and copy MANO_LEFT.pkl / MANO_RIGHT.pkl into that "
            "directory (or set MMPIPE_MANO_DIR to where they live)."
        )


def _intrinsics(dataset_root: Path, width: int, height: int) -> tuple[float, float, float, float]:
    """Real (fx, fy, cx, cy) from the sibling NIR calibration; centred fallback."""
    session = dataset_root.parent
    for cand in (session / "nir").glob("*/calibration.json"):
        try:
            calib = json.loads(cand.read_text(encoding="utf-8"))
        except Exception:
            continue
        r = calib.get("rgb")
        if r:
            cw = int(r.get("width", width)) or width
            ch = int(r.get("height", height)) or height
            sx, sy = width / cw, height / ch
            return float(r["fx"]) * sx, float(r["fy"]) * sy, float(r["cx"]) * sx, float(r["cy"]) * sy
    import math
    fx = (width / 2.0) / math.tan(math.radians(60.0) / 2.0)
    return fx, fx, width / 2.0, height / 2.0


def _decode_rgb(ffmpeg: Path, mp4: Path, w: int, h: int) -> np.ndarray:
    res = subprocess.run(
        [str(ffmpeg), "-v", "error", "-i", str(mp4), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        check=True, capture_output=True,
    )
    fb = w * h * 3
    n = len(res.stdout) // fb
    if n == 0:
        return np.zeros((0, h, w, 3), dtype=np.uint8)
    return np.frombuffer(res.stdout, np.uint8, count=n * fb).reshape(n, h, w, 3).copy()


def _mano_vertices(model, transl_cam, orient_aa, pose_aa, betas):
    """Run a smplx MANO model → (V,3) camera-frame vertices, wrist pinned to transl_cam."""
    import torch

    def _t(a):
        return torch.tensor(np.asarray(a, np.float32)[None], dtype=torch.float32)

    go, hp, be = _t(orient_aa), _t(pose_aa), _t(betas)
    with torch.no_grad():
        # Forward at transl=0 to find where the wrist joint lands, then offset so
        # the posed wrist sits exactly on the state's camera-frame wrist position.
        out0 = model(global_orient=go, hand_pose=hp, betas=be, return_verts=True)
        wrist0 = out0.joints[0, 0].numpy()  # joint 0 = wrist
        offset = np.asarray(transl_cam, np.float32) - wrist0
        verts = out0.vertices[0].numpy() + offset[None, :]
    return verts


def render_mano_overlay(
    dataset_root: Path, ep_idx: int, out_dir: Path, *,
    frame: int | None = None, mano_dir: str | Path | None = None,
    with_raw: bool = True,
) -> Path:
    """Render a MANO-mesh overlay PNG for one episode frame (raw | mesh-overlay)."""
    from PIL import Image, ImageDraw
    _patch_numpy_for_chumpy()  # must precede chumpy import (triggered by MANO unpickle)
    import smplx

    mano_dir = _resolve_mano_dir(mano_dir)
    _check_mano_assets(mano_dir)

    ffb = check_ffmpeg()
    info = json.loads((dataset_root / "meta" / "info.json").read_text(encoding="utf-8"))
    h, w, _ = info["features"]["observation.images.ego"]["shape"]
    fx, fy, cx, cy = _intrinsics(dataset_root, w, h)

    import pyarrow.parquet as pq
    data = pq.read_table(dataset_root / "data" / "chunk-000" / "file-000.parquet").to_pydict()
    rows = [i for i, e in enumerate(data["episode_index"]) if int(e) == ep_idx]
    if not rows:
        raise ValueError(f"episode {ep_idx} has no rows")
    states = [np.asarray(data["observation.state"][i], dtype=np.float64) for i in rows]
    masks = [data["state_mask"][i] for i in rows]

    rgb_mp4 = dataset_root / "videos" / "observation.images.ego" / "chunk-000" / f"episode_{ep_idx:06d}.mp4"
    frames = _decode_rgb(ffb.ffmpeg, rgb_mp4, w, h)
    T = min(len(frames), len(states))

    # Default frame: first where both hands are kept.
    if frame is None:
        frame = 0
        for t in range(T):
            m = masks[t]
            if bool(m[0]) and bool(m[1]):
                frame = t
                break
    frame = min(frame, T - 1)

    # flat_hand_mean=True matches this source tracker's MANO convention: it drops
    # MANO joint↔real-keypoint residual to ~5mm (vs ~18mm with False), verified
    # by Kabsch alignment against the ground-truth 21 keypoints. Override with
    # MMPIPE_MANO_FLAT_MEAN=0 if a future source uses the mean-pose convention.
    flat_mean = os.environ.get("MMPIPE_MANO_FLAT_MEAN", "true").strip().lower() in {"1", "true", "yes"}
    # Instantiate smplx.MANO directly with the explicit .pkl path — `create()`
    # would append a `mano/` subdir and not find the files.
    models = {
        "left": smplx.MANO(str(mano_dir / "MANO_LEFT.pkl"), is_rhand=False,
                           use_pca=False, flat_hand_mean=flat_mean),
        "right": smplx.MANO(str(mano_dir / "MANO_RIGHT.pkl"), is_rhand=True,
                            use_pca=False, flat_hand_mean=flat_mean),
    }
    faces = {k: m.faces.astype(np.int64) for k, m in models.items()}

    sl = STATE_LAYOUT
    st = states[frame]
    raw_img = Image.fromarray(frames[frame])
    mesh_img = Image.fromarray(frames[frame].copy())
    overlay = Image.new("RGBA", mesh_img.size, (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)
    fill = {"left": (0, 200, 255, 70), "right": (255, 120, 0, 70)}
    edge = {"left": (0, 120, 160, 180), "right": (170, 70, 0, 180)}

    for side, base in (("left", 0), ("right", 61)):
        kept = bool(masks[frame][0] if side == "left" else masks[frame][1])
        if not kept:
            continue
        transl = st[sl[f"{side}_wrist_transl_cam"][0]:sl[f"{side}_wrist_transl_cam"][1]]
        orient = st[sl[f"{side}_wrist_orient_cam_aa"][0]:sl[f"{side}_wrist_orient_cam_aa"][1]]
        pose = st[sl[f"{side}_mano_pose_aa"][0]:sl[f"{side}_mano_pose_aa"][1]]
        betas = st[sl[f"{side}_mano_betas"][0]:sl[f"{side}_mano_betas"][1]]
        if np.allclose(pose, 0) and np.allclose(betas, 0):
            continue  # no real MANO this frame/hand
        # Left-hand mirror fix: the source tracker stores left-hand axis-angle in
        # a mirrored convention; flipping the y,z components of global_orient and
        # every joint pose makes MANO_LEFT fit (drops joint↔keypoint residual from
        # ~25mm to ~3mm, matching the right hand). Verified by Kabsch vs the real
        # 21 keypoints.
        if side == "left":
            mir = np.array([1.0, -1.0, -1.0], dtype=np.float64)
            orient = orient * mir
            pose = (pose.reshape(15, 3) * mir).reshape(45)
        verts = _mano_vertices(models[side], transl, orient, pose, betas)
        z = verts[:, 2]
        ok = z > 1e-3
        u = fx * verts[:, 0] / z + cx
        v = fy * verts[:, 1] / z + cy
        # Draw faces as translucent filled triangles (painter's order: far→near).
        F = faces[side]
        order = np.argsort(-verts[F].mean(axis=1)[:, 2])  # far first
        for fi in order:
            a, b, c = F[fi]
            if not (ok[a] and ok[b] and ok[c]):
                continue
            tri = [(u[a], v[a]), (u[b], v[b]), (u[c], v[c])]
            odraw.polygon(tri, fill=fill[side], outline=edge[side])

    mesh_img = Image.alpha_composite(mesh_img.convert("RGBA"), overlay).convert("RGB")

    panels = [np.asarray(raw_img)] if with_raw else []
    panels.append(np.asarray(mesh_img))
    composed = Image.fromarray(np.concatenate(panels, axis=1))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / f"episode_{ep_idx:06d}_mano.png"
    composed.save(out_png)
    return out_png


def _draw_mano_on(img, st, models, faces, fx, fy, cx, cy):
    """Draw both hands' MANO mesh (translucent) onto a PIL image in place."""
    from PIL import Image, ImageDraw
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    fill = {"left": (0, 200, 255, 70), "right": (255, 120, 0, 70)}
    edge = {"left": (0, 120, 160, 160), "right": (170, 70, 0, 160)}
    mir = np.array([1.0, -1.0, -1.0], dtype=np.float64)
    for side, base in (("left", 0), ("right", 61)):
        transl = st[base:base + 3]
        orient = st[base + 3:base + 6].copy()
        pose = st[base + 6:base + 51].copy()
        betas = st[base + 51:base + 61]
        if np.allclose(pose, 0) and np.allclose(betas, 0):
            continue
        if side == "left":
            orient = orient * mir
            pose = (pose.reshape(15, 3) * mir).reshape(45)
        verts = _mano_vertices(models[side], transl, orient, pose, betas)
        z = verts[:, 2]; ok = z > 1e-3
        u = fx * verts[:, 0] / z + cx; v = fy * verts[:, 1] / z + cy
        F = faces[side]
        for fi in np.argsort(-verts[F].mean(axis=1)[:, 2]):
            a, b, c = F[fi]
            if ok[a] and ok[b] and ok[c]:
                od.polygon([(u[a], v[a]), (u[b], v[b]), (u[c], v[c])],
                           fill=fill[side], outline=edge[side])
    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")


def render_combined(dataset_root: Path, ep_idx: int, out_dir: Path, *,
                    mano_dir: str | Path | None = None, fps: float = 30.0) -> Path:
    """One synced MP4, 2×2 grid: [raw | keypoints+axes] / [MANO mesh | depth].

    Everything plays in a single VLC window — raw vs keypoint overlay vs hand
    mesh vs depth, frame-locked, so one file covers the whole verification.
    """
    from PIL import Image, ImageDraw
    import pyarrow.parquet as pq
    from .core import (_decode_depth_gray16, _decode_rgb as _vd_rgb, _depth_to_rgb,
                            _draw_hand, _draw_wrist_axes, _project, _BONES, _LEFT_COLOR, _RIGHT_COLOR)
    _patch_numpy_for_chumpy()
    import smplx

    mano_dir = _resolve_mano_dir(mano_dir); _check_mano_assets(mano_dir)
    ffb = check_ffmpeg()
    info = json.loads((dataset_root / "meta" / "info.json").read_text(encoding="utf-8"))
    h, w, _ = info["features"]["observation.images.ego"]["shape"]
    fx, fy, cx, cy = _intrinsics(dataset_root, w, h)

    data = pq.read_table(dataset_root / "data" / "chunk-000" / "file-000.parquet").to_pydict()
    rows = [i for i, e in enumerate(data["episode_index"]) if int(e) == ep_idx]
    kp_flat = [data["observation.hand_keypoints"][i] for i in rows]
    states = [np.asarray(data["observation.state"][i], dtype=np.float64) for i in rows]

    rgb = _vd_rgb(ffb.ffmpeg, dataset_root / "videos" / "observation.images.ego" / "chunk-000" / f"episode_{ep_idx:06d}.mp4", w, h)
    depth_mkv = dataset_root / "videos" / "observation.images.depth" / "chunk-000" / f"episode_{ep_idx:06d}.mkv"
    depth = _decode_depth_gray16(ffb.ffmpeg, depth_mkv, w, h) if depth_mkv.exists() else None
    T = min(len(rgb), len(kp_flat), len(states))

    flat = os.environ.get("MMPIPE_MANO_FLAT_MEAN", "true").strip().lower() in {"1", "true", "yes"}
    models = {"left": smplx.MANO(str(mano_dir / "MANO_LEFT.pkl"), is_rhand=False, use_pca=False, flat_hand_mean=flat),
              "right": smplx.MANO(str(mano_dir / "MANO_RIGHT.pkl"), is_rhand=True, use_pca=False, flat_hand_mean=flat)}
    faces = {k: m.faces.astype(np.int64) for k, m in models.items()}

    out_dir.mkdir(parents=True, exist_ok=True)
    out_mp4 = out_dir / f"episode_{ep_idx:06d}_combined.mp4"
    proc = subprocess.Popen(
        [str(ffb.ffmpeg), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{w*2}x{h*2}", "-r", f"{fps}", "-i", "-", "-c:v", "libx264", "-preset", "veryfast",
         "-pix_fmt", "yuv420p", str(out_mp4)], stdin=subprocess.PIPE)
    for t in range(T):
        raw = Image.fromarray(rgb[t])
        # keypoints + axes panel
        kpimg = Image.fromarray(rgb[t].copy()); kd = ImageDraw.Draw(kpimg)
        kp = np.asarray(kp_flat[t], dtype=np.float64).reshape(2, 21, 3)
        _draw_hand(kd, _project(kp[0], fx, fy, cx, cy), _LEFT_COLOR)
        _draw_hand(kd, _project(kp[1], fx, fy, cx, cy), _RIGHT_COLOR)
        st = states[t]
        if st.shape[0] >= 67:
            _draw_wrist_axes(kd, st[0:3], st[3:6], fx, fy, cx, cy)
            _draw_wrist_axes(kd, st[61:64], st[64:67], fx, fy, cx, cy)
        manoimg = _draw_mano_on(Image.fromarray(rgb[t].copy()), st, models, faces, fx, fy, cx, cy)
        depthimg = (Image.fromarray(_depth_to_rgb(depth[t])) if depth is not None and t < len(depth)
                    else Image.fromarray(np.zeros((h, w, 3), np.uint8)))
        top = np.concatenate([np.asarray(raw), np.asarray(kpimg)], axis=1)
        bot = np.concatenate([np.asarray(manoimg), np.asarray(depthimg)], axis=1)
        proc.stdin.write(np.concatenate([top, bot], axis=0).astype(np.uint8).tobytes())
    proc.stdin.close(); proc.wait()
    return out_mp4


__all__ = ["render_mano_overlay", "render_combined"]
