"""Frame-locked MP4: human hand (RGB + keypoints) | Wuji robot hand FK skeleton.

Renders one VLC-friendly video per episode showing, side by side and frame-
locked: the human demonstration (RGB with the 21-keypoint overlay) and the
retargeted Wuji hand — its 20-DOF FK skeleton driven by observation.robot_qpos,
moving in lock-step. Lets you watch the robot hand mimic the human grasp.

Usage:  python3 scripts/viz_robot_hand_mp4.py <dataset_root> [--episode N] [--out dir]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from multimodal_pipeline._system import check_ffmpeg  # noqa: E402
from multimodal_pipeline.visualize import _decode_rgb, _draw_hand, _project, _LEFT_COLOR, _RIGHT_COLOR  # noqa: E402
from multimodal_pipeline.lerobot_v3.schema import ROBOT_QPOS_LAYOUT  # noqa: E402
from multimodal_pipeline.retarget.robot import RobotModel  # noqa: E402

_COL = {"left": "#0096ff", "right": "#ff7800"}


def _hand_skeleton(ax, model, qpos20, color):
    """Draw a Wuji hand FK skeleton (palm + 5 finger chains) in palm-local frame."""
    frames, _ = model.tree.fk(qpos20)
    P = {ln: frames[ln][:3, 3] for ln in frames}
    for j in model.tree.joints:
        a, b = P[j.parent], P[j.child]
        ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], color=color, lw=2.0)
    tips = np.array([P[t] for t in model.tip_links])
    ax.scatter(tips[:, 0], tips[:, 1], tips[:, 2], c=color, s=22, depthshade=False)
    ax.scatter([0], [0], [0], c="w", s=30, marker="s")  # palm origin
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.set_pane_color((0.12, 0.12, 0.12, 1.0))
    ax.grid(False)
    ax.set_xlim(-0.05, 0.08); ax.set_ylim(-0.07, 0.07); ax.set_zlim(-0.02, 0.16)
    ax.set_box_aspect((0.13, 0.14, 0.18))
    ax.view_init(elev=18, azim=-72)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])


def _wuji_panel(models, qL, qR, w, h, dpi=100):
    """Render both Wuji hands' FK skeletons to an (h, w, 3) uint8 array."""
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi, facecolor="#181818")
    for i, (side, q) in enumerate((("left", qL), ("right", qR))):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        ax.set_facecolor("#181818")
        if q is not None and np.isfinite(q).all():
            _hand_skeleton(ax, models[side], q, _COL[side])
            ax.set_title(f"Wuji {side}", color=_COL[side], fontsize=10)
        else:
            ax.set_title(f"Wuji {side}: (no grasp)", color="#888888", fontsize=9)
            ax.set_axis_off()
    fig.tight_layout()
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return np.asarray(Image.fromarray(buf).resize((w, h)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_root", type=Path)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args()

    import json, pyarrow.parquet as pq
    root = args.dataset_root
    info = json.loads((root / "meta" / "info.json").read_text())
    h, w, _ = info["features"]["observation.images.ego"]["shape"]
    fx = (w / 2.0) / np.tan(np.radians(60.0) / 2.0)
    fxiy = (fx, fx, w / 2.0, h / 2.0)

    d = pq.read_table(root / "data" / "chunk-000" / "file-000.parquet").to_pydict()
    rows = [i for i, e in enumerate(d["episode_index"]) if int(e) == args.episode]
    if not rows:
        print(f"episode {args.episode} not found"); return 1
    qpos = np.array([d["observation.robot_qpos"][i] for i in rows], float)   # (T,46)
    kp = np.array([d["observation.hand_keypoints"][i] for i in rows], float)  # (T,126)

    ffb = check_ffmpeg()
    mp4 = root / "videos" / "observation.images.ego" / "chunk-000" / f"episode_{args.episode:06d}.mp4"
    rgb = _decode_rgb(ffb.ffmpeg, mp4, w, h)
    T = min(len(rgb), len(qpos))

    models = {"left": RobotModel("left"), "right": RobotModel("right")}
    lH, rH = ROBOT_QPOS_LAYOUT["left_hand"], ROBOT_QPOS_LAYOUT["right_hand"]

    out_dir = args.out or (root / "viz")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"episode_{args.episode:06d}_robothand.mp4"
    proc = subprocess.Popen(
        [str(ffb.ffmpeg), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{w*2}x{h}", "-r", str(args.fps), "-i", "-", "-c:v", "libx264",
         "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", str(out)], stdin=subprocess.PIPE)
    for t in range(T):
        img = Image.fromarray(rgb[t].copy())
        dr = ImageDraw.Draw(img)
        k = kp[t].reshape(2, 21, 3)
        if np.isfinite(k[0]).all():
            _draw_hand(dr, _project(k[0], *fxiy), _LEFT_COLOR)
        if np.isfinite(k[1]).all():
            _draw_hand(dr, _project(k[1], *fxiy), _RIGHT_COLOR)
        qL = qpos[t, lH[0]:lH[1]]; qR = qpos[t, rH[0]:rH[1]]
        qL = qL if np.isfinite(qL).all() else None
        qR = qR if np.isfinite(qR).all() else None
        panel = _wuji_panel(models, qL, qR, w, h)
        frame = np.concatenate([np.asarray(img), panel], axis=1)
        proc.stdin.write(frame.astype(np.uint8).tobytes())
    proc.stdin.close(); proc.wait()
    print(f"wrote {out}  ({T} frames)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
