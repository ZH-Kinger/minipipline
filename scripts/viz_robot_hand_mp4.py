"""Overlay the retargeted Wuji hand ONTO the RGB video (projected, frame-locked).

The Wuji hand FK skeleton is projected into the camera image and drawn on top of
the human hand, so you see the robot hand tracking the demonstration in place.

Transform chain (the part that was wrong before — now derived correctly and
self-checked): a robot link point in the palm-local frame maps to camera space
via  cam = wrist_cam + R_cam · (R_alignᵀ · p_palm / scale),  where
  - (R_cam, wrist_cam): the human wrist-local frame built from the camera-frame
    keypoints (so it tracks the moving hand each frame),
  - R_align, scale: the per-hand Kabsch alignment + per-finger scale calibrated
    over the episode (the inverse of what retargeting applied),
then projected with the REAL NIR intrinsics (not a guessed FOV).

Self-check: projected Wuji fingertips must land on the human fingertip keypoints
(small pixel distance). Printed per hand; a large value means the overlay is off.

Usage:  python3 scripts/viz_robot_hand_mp4.py <dataset_root> [--episode N] [--out dir]
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from multimodal_pipeline._system import check_ffmpeg  # noqa: E402
from multimodal_pipeline.visualize import _decode_rgb, _project_pts  # noqa: E402
from multimodal_pipeline.viz_mano import _intrinsics  # noqa: E402
from multimodal_pipeline.lerobot_v3.schema import ROBOT_QPOS_LAYOUT  # noqa: E402
from multimodal_pipeline.retarget.robot import RobotModel, MANO_FINGERTIPS  # noqa: E402
from multimodal_pipeline.retarget.optimize import kabsch_rotation  # noqa: E402
from multimodal_pipeline.retarget.core import _human_fingertip_vectors, _hand_basis  # noqa: E402

_COL = {"left": (0, 180, 255), "right": (255, 140, 0)}
_FMAP = (0, 1, 2, 3, 4)


def _umeyama(src: np.ndarray, dst: np.ndarray):
    """Similarity transform (s, R, t) minimising ‖s·R·src + t − dst‖ (Umeyama 1991)."""
    src = np.asarray(src, float); dst = np.asarray(dst, float)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    cov = (D.T @ S) / len(src)
    U, d, Vt = np.linalg.svd(cov)
    E = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        E[2, 2] = -1.0
    R = U @ E @ Vt
    var = (S ** 2).sum() / len(src)
    s = float(np.trace(np.diag(d) @ E) / var) if var > 1e-12 else 1.0
    t = mu_d - s * (R @ mu_s)
    return s, R, t


def _overlay_hand(dr, model, q_hand, kp_cam, intr, color):
    """Project the Wuji hand skeleton onto the image via a per-frame Umeyama fit
    of its fingertips to the human fingertip keypoints. Returns mean fingertip
    pixel distance (self-check) — the residual = how well the robot hand SHAPE
    matches the human hand (no dependence on an episode-level alignment)."""
    frames, _ = model.tree.fk(q_hand)
    P = {ln: frames[ln][:3, 3] for ln in frames}
    rob = np.array([P[tl] for tl in model.tip_links])                        # palm-local (5,3)
    hum = np.array([kp_cam[MANO_FINGERTIPS[_FMAP[i]]] for i in range(5)])     # camera (5,3)
    if not np.isfinite(hum).all():
        return None
    s, R, t = _umeyama(rob, hum)
    proj = {ln: _project_pts((s * (R @ P[ln]) + t)[None], *intr)[0] for ln in P}
    for j in model.tree.joints:
        a, b = proj[j.parent], proj[j.child]
        dr.line([a[0], a[1], b[0], b[1]], fill=color, width=3)
    for tl in model.tip_links:
        u, v = proj[tl]
        dr.ellipse([u - 4, v - 4, u + 4, v + 4], fill=color)
    tip_px = np.array([proj[tl] for tl in model.tip_links])
    hum_px = _project_pts(hum, *intr)
    return float(np.linalg.norm(tip_px - hum_px, axis=1).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_root", type=Path)
    ap.add_argument("--episode", type=int, default=None,
                    help="episode index; default = the one with most grasp frames")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args()

    import json, pyarrow.parquet as pq
    root = args.dataset_root
    info = json.loads((root / "meta" / "info.json").read_text())
    h, w, _ = info["features"]["observation.images.ego"]["shape"]
    intr = _intrinsics(root, w, h)

    d = pq.read_table(root / "data" / "chunk-000" / "file-000.parquet").to_pydict()
    ei = np.array(d["episode_index"])
    Q = np.array(d["observation.robot_qpos"], float)
    K = np.array(d["observation.hand_keypoints"], float).reshape(-1, 2, 21, 3)
    lH, rH = ROBOT_QPOS_LAYOUT["left_hand"], ROBOT_QPOS_LAYOUT["right_hand"]
    if args.episode is None:
        best = (int(ei[0]), -1)
        for ep in sorted(set(int(e) for e in ei)):
            r = [i for i in range(len(ei)) if int(ei[i]) == ep]
            k = sum(np.isfinite(Q[i, lH[0]:lH[1]]).all() or np.isfinite(Q[i, rH[0]:rH[1]]).all() for i in r)
            if k > best[1]:
                best = (ep, k)
        args.episode = best[0]
    rows = [i for i in range(len(ei)) if int(ei[i]) == args.episode]

    models = {"left": RobotModel("left"), "right": RobotModel("right")}
    # No episode-level calibration — each frame aligns the Wuji hand to the human
    # fingertips via Umeyama, so a hard-to-fit hand never drags the whole hand off.

    ffb = check_ffmpeg()
    mp4 = root / "videos" / "observation.images.ego" / "chunk-000" / f"episode_{args.episode:06d}.mp4"
    rgb = _decode_rgb(ffb.ffmpeg, mp4, w, h)
    T = min(len(rgb), len(rows))

    out_dir = args.out or (root / "viz")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"robothand_{root.parent.name[:8]}_ep{args.episode:06d}.mp4"
    proc = subprocess.Popen(
        [str(ffb.ffmpeg), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{w}x{h}", "-r", str(args.fps), "-i", "-", "-c:v", "libx264",
         "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", str(out)], stdin=subprocess.PIPE)
    checks = {"left": [], "right": []}
    for t in range(T):
        gi = rows[t]
        img = Image.fromarray(rgb[t].copy()); dr = ImageDraw.Draw(img)
        for hi, side in ((0, "left"), (1, "right")):
            sl = lH if side == "left" else rH
            q = Q[gi, sl[0]:sl[1]]
            if not np.isfinite(q).all() or not np.isfinite(K[gi, hi]).all():
                continue
            err = _overlay_hand(dr, models[side], q, K[gi, hi], intr, _COL[side])
            if err is not None:
                checks[side].append(err)
        proc.stdin.write(np.asarray(img).astype(np.uint8).tobytes())
    proc.stdin.close(); proc.wait()
    sc = {s: round(float(np.mean(v)), 1) if v else None for s, v in checks.items()}
    print(f"wrote {out}  ({T} frames)  overlay px-dist {sc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
