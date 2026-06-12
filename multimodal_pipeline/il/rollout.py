"""Open-loop ACT rollout → render predicted Wuji hand vs ground-truth.

Picks a single-hand kept run from a session, feeds the real ego image + current
hand config each frame, lets the ACT policy predict the next hand config (chunked),
un-normalizes it, and renders the predicted hand skeleton (red) against the GT
hand skeleton (green) next to the RGB frame. The most direct "does the policy do
something sensible" check (no sim/robot available). Also prints val-style MSE.

Run:  python3 -m multimodal_pipeline.il.rollout --session 0 --hand right
"""

from __future__ import annotations

import argparse
import glob
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from .._system import check_ffmpeg
from ..lerobot_v3.schema import ROBOT_QPOS_LAYOUT
from ..retarget.robot import RobotModel
from ..viz.core import _decode_rgb
from .dataset import hand_vec
from .train import _IMEAN, _ISTD, build_policy

W, H = 640, 480


def _skel(ax, model, q20, color, label=None):
    frames, _ = model.tree.fk(np.asarray(q20, float))
    P = {ln: frames[ln][:3, 3] for ln in frames}
    first = True
    for j in model.tree.joints:
        a, b = P[j.parent], P[j.child]
        ax.plot(*zip(a, b), color=color, lw=2.2, label=label if first else None)
        first = False
    tips = np.array([P[t] for t in model.tip_links])
    ax.scatter(*tips.T, color=color, s=18)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", type=Path, default=Path("artifacts/il"))
    ap.add_argument("--session", type=int, default=0)
    ap.add_argument("--hand", choices=["left", "right"], default="right")
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--out", type=Path, default=Path("artifacts/gallery"))
    args = ap.parse_args()
    dev = "cuda"

    st = np.load(args.policy / "stats.npz")
    sm = torch.tensor(st["state_mean"]); ss = torch.tensor(st["state_std"])
    am, asd = st["action_mean"], st["action_std"]

    policy = build_policy(args.chunk, (192, 256))
    policy.load_state_dict(torch.load(args.policy / "act_policy.pt", map_location=dev))
    policy.to(dev).eval(); policy.reset()

    import pyarrow.parquet as pq
    sd = Path(sorted(glob.glob("output/*/lerobot_dataset"))[args.session])
    info = json.loads((sd / "meta" / "info.json").read_text())
    h, w, _ = info["features"]["observation.images.ego"]["shape"]
    d = pq.read_table(glob.glob(str(sd / "data/chunk-000/*.parquet"))[0]).to_pydict()
    ei = np.array(d["episode_index"])
    Q = np.array(d["observation.robot_qpos"], float); E = np.array(d["observation.robot_ee_pose"], float)

    ff = check_ffmpeg().ffmpeg
    model = RobotModel(args.hand)

    def _longest_run(valid):
        run, cur = [], []
        for k, v in enumerate(valid):
            if v:
                cur.append(k)
            else:
                if len(cur) > len(run): run = cur
                cur = []
        return cur if len(cur) > len(run) else run

    # rollout the longest kept run of EVERY episode, concatenated → a long video
    preds, gts, rgb_frames = [], [], []
    for ep in sorted(set(int(e) for e in ei)):
        rows = [i for i in range(len(ei)) if int(ei[i]) == ep]
        vecs = np.array([hand_vec(Q[i], E[i], args.hand) for i in rows])
        run = _longest_run(np.isfinite(vecs).all(1))
        if len(run) < 2:
            continue
        mp4 = sd / "videos/observation.images.ego/chunk-000" / f"episode_{ep:06d}.mp4"
        rgb = _decode_rgb(ff, mp4, w, h)
        policy.reset()
        for n in range(len(run) - 1):
            li = run[n]
            if li >= len(rgb):
                break
            img = np.asarray(Image.fromarray(rgb[li]).resize((256, 192)), np.float32) / 255.0
            img = (torch.from_numpy(img.transpose(2, 0, 1))[None] - _IMEAN) / _ISTD
            state = (torch.tensor(vecs[li], dtype=torch.float32)[None] - sm) / ss
            batch = {"observation.images.ego": img.to(dev), "observation.state": state.to(dev)}
            a = policy.select_action(batch).cpu().numpy()[0] * asd + am
            preds.append(a[:20]); gts.append(vecs[run[n + 1]][:20]); rgb_frames.append(rgb[li])
    print(f"session {sd.parent.name[:8]} hand {args.hand}: {len(preds)} rollout frames across episodes")

    preds = np.array(preds); gts = np.array(gts)
    mse = float(np.mean((preds - gts) ** 2))
    print(f"rollout frames {len(preds)}  next-hand_qpos MSE {mse:.4f}")

    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / f"il_rollout_{sd.parent.name[:8]}_{args.hand}.mp4"
    ff = check_ffmpeg().ffmpeg
    proc = subprocess.Popen(
        [str(ff), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W*2}x{H}", "-r", "10", "-i", "-", "-c:v", "libx264", "-preset", "medium",
         "-crf", "18", "-pix_fmt", "yuv420p", str(out)], stdin=subprocess.PIPE)
    dpi = 100
    for n in range(len(preds)):
        rgb_panel = np.asarray(Image.fromarray(rgb_frames[n]).resize((W, H)))
        fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi, facecolor="#101014")
        ax = fig.add_subplot(111, projection="3d"); ax.set_facecolor("#101014")
        _skel(ax, model, gts[n], "#3cdc6e", "GT")
        _skel(ax, model, preds[n], "#ff5050", "pred")
        ax.set_xlim(-0.05, 0.08); ax.set_ylim(-0.07, 0.07); ax.set_zlim(-0.02, 0.16)
        ax.set_box_aspect((0.13, 0.14, 0.18)); ax.view_init(18, -72)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.legend(loc="upper left", fontsize=8)
        ax.set_title(f"Wuji {args.hand}: pred(red) vs GT(green)  [{n+1}/{len(preds)}]", color="w", fontsize=9)
        fig.tight_layout()
        fig.canvas.draw()
        skel = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        skel = np.asarray(Image.fromarray(skel).resize((W, H)))
        plt.close(fig)
        proc.stdin.write(np.concatenate([rgb_panel, skel], axis=1).astype(np.uint8).tobytes())
    proc.stdin.close(); proc.wait()
    print(f"wrote {out}  ({len(preds)} frames, MSE {mse:.4f})")


if __name__ == "__main__":
    main()
