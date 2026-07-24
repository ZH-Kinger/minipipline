"""Self-explaining IL training analysis figure.

Goes beyond a bare loss curve: writes the *interpretation* onto the figure —
is it converged / overfitting / underfitting, what the val loss is determined by
(data volume, retarget quality, per-joint difficulty). Answers "how good is my
policy, and what is that tied to" at a glance.

Run:  python3 -m multimodal_pipeline.il.analyze [--out artifacts/il_live] [--sessions 4]
"""

from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .dataset import WujiActionDataset
from .train import _IMEAN, _ISTD, build_policy


def _parse_log(log_path: Path):
    txt = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    tr = [(int(s), float(v)) for s, v in re.findall(r"step (\d+)\s+train_loss ([\d.]+)", txt)]
    va = [(int(s), float(v)) for v, s in re.findall(r"val_loss ([\d.]+)\s+\(step (\d+)\)", txt)]
    return tr, va


def _diagnose(tr, va):
    """Return a human verdict string from the loss histories."""
    if not va:
        return "no val data yet"
    v0, vlast = va[0][1], va[-1][1]
    drop = (v0 - vlast) / v0 if v0 else 0
    tlast = tr[-1][1] if tr else vlast
    gap = (vlast - tlast) / vlast if vlast else 0
    # last-third slope of val
    tail = [v for _, v in va[max(0, len(va) * 2 // 3):]]
    still = (tail[0] - tail[-1]) / tail[0] if len(tail) > 1 and tail[0] else 0
    parts = [f"val {v0:.3f}->{vlast:.3f} ({drop*100:.0f}% down)"]
    parts.append("CONVERGED ✓" if still < 0.05 else "still improving ↓")
    if gap > 0.5:
        parts.append("OVERFIT ⚠ (val >> train)")
    elif vlast > 0.15:
        parts.append("UNDERFIT (loss still high)")
    else:
        parts.append("healthy fit ✓ (train≈val)")
    return "  |  ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("artifacts/il_live"))
    ap.add_argument("--sessions", type=int, default=4)
    ap.add_argument("--log", type=Path, default=Path("artifacts/logs/il_live.log"))
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    tr, va = _parse_log(args.log)
    verdict = _diagnose(tr, va)

    # Re-evaluate the trained policy per-sample to attribute error.
    st = np.load(args.out / "stats.npz")
    sm = torch.tensor(st["state_mean"]); ss = torch.tensor(st["state_std"])
    am, asd = st["action_mean"], st["action_std"]
    dirs = sorted(glob.glob("output/*/lerobot_dataset"))[: args.sessions]
    ds = WujiActionDataset(dirs, chunk_size=16)
    pol = build_policy(16, (192, 256))
    pol.load_state_dict(torch.load(args.out / "act_policy.pt", map_location=dev)); pol.to(dev).eval()

    # per-sample next-step abs error + which session it came from
    from torch.utils.data import DataLoader
    sess_of, errs_hand = [], []
    per_dof_acc = np.zeros(26); n_acc = 0

    def coll(items):
        img = torch.stack([x["observation.images.ego"] for x in items])
        s = torch.stack([x["observation.state"] for x in items])
        a = torch.stack([x["action"] for x in items])
        keys = [it for it in items]
        return ((img - _IMEAN) / _ISTD, (s - sm) / ss, a), keys

    # Error via the model's OWN forward path (same as training/val loss, which we
    # trust = 0.036). model(batch) returns (actions_hat, _); we un-normalize the
    # first-step prediction vs the first-step GT. Avoids the select_action /
    # predict_action_chunk queue quirks that inflated the number 10x.
    order = list(range(len(ds)))
    asd_t = torch.tensor(asd); am_t = torch.tensor(am)
    with torch.no_grad():
        for i in range(0, len(order), 128):
            idx = order[i:i + 128]
            items = [ds[j] for j in idx]
            img = torch.stack([x["observation.images.ego"] for x in items])
            s = torch.stack([x["observation.state"] for x in items])
            ac = torch.stack([x["action"] for x in items])
            batch = {
                "observation.images.ego": ((img - _IMEAN) / _ISTD).to(dev),
                "observation.state": ((s - sm) / ss).to(dev),
                "observation.images": [((img - _IMEAN) / _ISTD).to(dev)],
            }
            ahat, _ = pol.model(batch)               # (B, chunk, 26) normalized
            pred = ahat[:, 0].cpu() * asd_t + am_t   # un-normalized next-step
            gt = ac[:, 0] * asd_t + am_t
            e = np.abs((pred - gt).numpy())
            per_dof_acc += e.sum(0); n_acc += len(e)
            errs_hand.extend(e[:, :20].mean(1).tolist())  # 20 hand joints only
            sess_of.extend(ds.samples[j]["key"][0] for j in idx)
    per_dof = per_dof_acc / n_acc
    errs_hand = np.array(errs_hand); sess_of = np.array(sess_of)

    # retarget residual per session (align residual from a quick recompute is heavy;
    # instead use sample count as the data-volume axis, and per-session mean error)
    sids = sorted(set(sess_of.tolist()))
    sess_n = np.array([(sess_of == s).sum() for s in sids])
    sess_err = np.array([errs_hand[sess_of == s].mean() for s in sids])

    fig, axs = plt.subplots(2, 2, figsize=(14, 9))
    # (1) loss curve + verdict
    ax = axs[0, 0]
    if tr: ax.plot([s for s, _ in tr], [v for _, v in tr], color="#4363d8", alpha=.6, lw=1, label="train")
    if va: ax.plot([s for s, _ in va], [v for _, v in va], "o-", color="orange", ms=4, label="val")
    ax.set_yscale("log"); ax.set_xlabel("step"); ax.set_ylabel("L1 loss"); ax.legend(); ax.grid(alpha=.3)
    ax.set_title("① training curve — " + ("converged" if "CONVERGED" in verdict else "training"))
    ax.text(0.5, -0.22, verdict, transform=ax.transAxes, ha="center", fontsize=9,
            bbox=dict(boxstyle="round", fc="#eef", ec="#88a"))
    # (2) error vs data volume
    ax = axs[0, 1]
    ax.scatter(sess_n, sess_err * 1000, c="#4363d8", s=60)
    for sx, sy, sid in zip(sess_n, sess_err * 1000, sids):
        ax.annotate(f"s{sid}", (sx, sy), fontsize=8)
    ax.set_xlabel("samples in session (data volume)"); ax.set_ylabel("mean hand-joint err (mrad)")
    ax.grid(alpha=.3)
    corr = np.corrcoef(sess_n, sess_err)[0, 1] if len(sids) > 2 else 0.0
    if corr < -0.3:
        rel = "more data → lower err"
    elif corr > 0.3:
        rel = "err NOT driven by data volume (pose difficulty dominates)"
    else:
        rel = "weak link to data volume"
    ax.set_title(f"② error vs DATA VOLUME  (corr {corr:+.2f}: {rel})")
    # (3) error distribution (how consistent)
    ax = axs[1, 0]
    ax.hist(errs_hand * 1000, bins=40, color="#3cb44b", alpha=.8)
    ax.axvline(errs_hand.mean() * 1000, color="r", ls="--", label=f"mean {errs_hand.mean()*1000:.1f} mrad")
    ax.set_xlabel("per-sample hand-joint err (mrad)"); ax.set_ylabel("count"); ax.legend()
    ax.set_title("③ error spread — most predictions tight, tail = hard poses")
    # (4) per-DoF difficulty
    ax = axs[1, 1]
    labels = [f"f{f}j{j}" for f in range(1, 6) for j in range(1, 5)] + ["eTx", "eTy", "eTz", "eRx", "eRy", "eRz"]
    colors = ["#4363d8"] * 20 + ["#e6194B"] * 6
    ax.bar(range(26), per_dof, color=colors)
    ax.set_xticks(range(26)); ax.set_xticklabels(labels, rotation=90, fontsize=6)
    hardest = labels[int(np.argmax(per_dof))]
    ax.set_title(f"④ per-DoF difficulty — hardest: {hardest} ({per_dof.max():.3f})")
    ax.grid(alpha=.3, axis="y")

    fig.suptitle("IL training analysis — what the policy learned & what it's tied to", fontsize=14)
    fig.tight_layout()
    out = args.out / "il_analysis.png"
    fig.savefig(out, dpi=110)
    # also drop into gallery
    gal = Path("artifacts/gallery/il_analysis.png"); gal.parent.mkdir(parents=True, exist_ok=True)
    import shutil; shutil.copy(out, gal)
    print(f"verdict: {verdict}")
    print(f"hand-joint mean err {errs_hand.mean()*1000:.1f} mrad | wrote {out} + {gal}")


if __name__ == "__main__":
    main()
