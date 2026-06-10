"""Visualize retarget fit: human target fingertips vs Wuji FK fingertips.

In the palm-local aligned space (where the solver works), overlays the human
fingertip targets (scaled+aligned) against the robot's FK fingertips, per hand,
coloured per finger. Makes the fit quality — and where the robot hand cannot
reach a human pose — visible at a glance.

Usage:  python3 scripts/viz_retarget.py <nir_dir> [out.png]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..handpose import HandPoseConfig, run_handpose_pipeline  # noqa: E402
from ..config.retarget import RetargetConfig  # noqa: E402
from ..retarget.robot import RobotModel  # noqa: E402
from ..retarget.optimize import kabsch_rotation, solve_hand_qpos  # noqa: E402
from ..retarget.core import _human_fingertip_vectors  # noqa: E402

FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
COLORS = ["#e6194B", "#3cb44b", "#4363d8", "#f58231", "#911eb4"]


def retarget_hand_debug(kp_seq, kept, model, cfg, max_frames=80):
    """Calibrate (per-finger) and solve; return (targets, fks, errs) (N,5,3)/(5,)."""
    fmap = cfg.finger_map
    valid = [t for t in range(kp_seq.shape[0])
             if kept[t] and _human_fingertip_vectors(kp_seq[t], fmap) is not None]
    if not valid:
        return None
    if len(valid) > max_frames:
        valid = valid[:: len(valid) // max_frames][:max_frames]
    V = np.stack([_human_fingertip_vectors(kp_seq[t], fmap) for t in valid])
    Vbar = V.mean(0)
    V_robot = model.fingertips_local(model.tree.midpoint())
    rlen = np.linalg.norm(V_robot, axis=1); hlen = np.linalg.norm(Vbar, axis=1)
    scale_vec = np.where(hlen > 1e-6, rlen / hlen, 1.0)
    R_align = kabsch_rotation(scale_vec[:, None] * Vbar, V_robot)

    targets, fks = [], []
    prev = None
    for vt in valid:
        v = _human_fingertip_vectors(kp_seq[vt], fmap)
        tgt = scale_vec[:, None] * (v @ R_align.T)
        q = solve_hand_qpos(model, tgt, q_init=prev, smooth_ref=prev,
                            smooth_weight=cfg.smooth_weight, n_iters=cfg.n_iters)
        prev = q
        full = model.tree.midpoint().copy(); full[model.hand_idx] = q
        targets.append(tgt); fks.append(model.fingertips_local(full))
    targets = np.stack(targets); fks = np.stack(fks)
    errs = np.linalg.norm(targets - fks, axis=2).mean(0) * 1000  # (5,) per-finger mm
    return targets, fks, errs


def main():
    nir = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else (
        Path(__file__).resolve().parents[2] / "artifacts" / "gallery" / f"retarget_{nir.name[:8]}.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    res = run_handpose_pipeline(nir, HandPoseConfig.from_file(None))
    m = res.merged
    if m.hand_keypoints_world is None:
        print("no real keypoints"); return 1
    cfg = RetargetConfig.from_env()

    fig = plt.figure(figsize=(13, 6))
    for h, side in enumerate(("left", "right")):
        model = RobotModel(cfg.urdf_for(side), cfg.finger_map)
        dbg = retarget_hand_debug(m.hand_keypoints_world[h], m.pred_kept[h], model, cfg)
        ax = fig.add_subplot(1, 2, h + 1, projection="3d")
        if dbg is None:
            ax.set_title(f"{side}: no valid frames"); continue
        targets, fks, errs = dbg
        for fi in range(5):
            ax.scatter(*fks[:, fi].T * 100, c=COLORS[fi], s=12, marker="o",
                       label=f"{FINGERS[fi]} {errs[fi]:.1f}mm")
            ax.scatter(*targets[:, fi].T * 100, c=COLORS[fi], s=18, marker="x", alpha=0.5)
        ax.scatter([0], [0], [0], c="k", s=40, marker="s")  # palm origin
        ax.set_title(f"{side} hand  (mean {errs.mean():.1f}mm)\n● Wuji FK   ✕ human target  [cm]")
        ax.legend(fontsize=7, loc="upper left")
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    fig.suptitle(f"Retarget fit — {nir.name[:8]}  (palm-local aligned space, per-finger scale)")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
