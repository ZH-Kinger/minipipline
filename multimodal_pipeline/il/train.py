"""Train a lerobot ACT policy to predict single-hand Wuji actions.

Normalization is done here in the collate fn (state/action standardized by
dataset mean-std, image by ImageNet stats) and the policy's normalization_mapping
is IDENTITY — this lerobot build's ACTPolicy doesn't normalize internally and
takes no stats, so doing it ourselves is the controllable path. Saves the policy
state_dict + the norm stats (needed for rollout un-normalization).

Run:  python3 -m multimodal_pipeline.il.train --sessions 8 --epochs 8
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy

from .dataset import HAND_DIM, WujiActionDataset
from .viz import render_training_curve

_IMEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_ISTD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _compute_stats(ds):
    S = np.stack([s["state"] for s in ds.samples])
    A = np.concatenate([s["action"][~s["pad"]] for s in ds.samples if (~s["pad"]).any()])
    return (S.mean(0).astype(np.float32), (S.std(0) + 1e-6).astype(np.float32),
            A.mean(0).astype(np.float32), (A.std(0) + 1e-6).astype(np.float32))


def _make_collate(sm, ss, am, asd):
    sm, ss = torch.tensor(sm), torch.tensor(ss)
    am, asd = torch.tensor(am), torch.tensor(asd)

    def collate(items):
        img = torch.stack([x["observation.images.ego"] for x in items])
        st = torch.stack([x["observation.state"] for x in items])
        ac = torch.stack([x["action"] for x in items])
        pad = torch.stack([x["action_is_pad"] for x in items])
        img = (img - _IMEAN) / _ISTD
        st = (st - sm) / ss
        ac = (ac - am) / asd
        return {"observation.images.ego": img, "observation.state": st,
                "action": ac, "action_is_pad": pad}

    return collate


def build_policy(chunk: int, image_hw):
    ident = {"VISUAL": NormalizationMode.IDENTITY, "STATE": NormalizationMode.IDENTITY,
             "ACTION": NormalizationMode.IDENTITY}
    cfg = ACTConfig(
        input_features={
            "observation.images.ego": PolicyFeature(type=FeatureType.VISUAL, shape=(3, *image_hw)),
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(HAND_DIM,)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(HAND_DIM,))},
        normalization_mapping=ident,
        chunk_size=chunk, n_action_steps=chunk, n_obs_steps=1,
        vision_backbone="resnet18", device="cuda",
        use_vae=False,  # avoid this build's VAE-in-eval KL crash; plain transformer BC
    )
    return ACTPolicy(cfg)


def _eval_per_dof(policy, vdl, am, asd, dev):
    """Per-DoF mean-abs next-step error via clean single-step prediction (no queue)."""
    policy.eval()
    E = []
    with torch.no_grad():
        for batch in vdl:
            pred = policy.predict_action_chunk({
                "observation.images.ego": batch["observation.images.ego"].to(dev),
                "observation.state": batch["observation.state"].to(dev),
            })[:, 0].cpu().numpy() * asd + am
            gt = batch["action"][:, 0].numpy() * asd + am
            E.append(np.abs(pred - gt))
    policy.train()
    return np.concatenate(E).mean(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-steps", type=int, default=0, help=">0 to cap steps (quick self-check)")
    ap.add_argument("--out", type=Path, default=Path("artifacts/il"))
    ap.add_argument("--tb", action=argparse.BooleanOptionalAction, default=True,
                    help="write TensorBoard events to <out>/tb (default on)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    dev = "cuda"

    dirs = sorted(glob.glob("output/*/lerobot_dataset"))[: args.sessions]
    print(f"building dataset from {len(dirs)} sessions ...")
    ds = WujiActionDataset(dirs, chunk_size=args.chunk)
    sm, ss, am, asd = _compute_stats(ds)
    print(f"samples={len(ds)}  state_std~{ss.mean():.3f}  action_std~{asd.mean():.3f}")
    n_val = max(1, int(len(ds) * 0.1))
    tr, va = random_split(ds, [len(ds) - n_val, n_val],
                          generator=torch.Generator().manual_seed(0))
    collate = _make_collate(sm, ss, am, asd)
    dl = DataLoader(tr, batch_size=args.bs, shuffle=True, collate_fn=collate, num_workers=2, drop_last=True)
    vdl = DataLoader(va, batch_size=args.bs, collate_fn=collate, num_workers=2)

    policy = build_policy(args.chunk, (192, 256)).to(dev)
    policy.train()
    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=1e-4)

    writer = None
    if args.tb:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(str(args.out / "tb"))
        print(f"tensorboard logdir: {args.out / 'tb'}  "
              f"(VS Code: Ctrl+Shift+P → Python: Launch TensorBoard)")

    step = 0
    train_hist, val_hist, last_val = [], [], None
    for ep in range(args.epochs):
        for batch in dl:
            gpu_batch = {k: v.to(dev) for k, v in batch.items()}
            loss, _ = policy.forward(gpu_batch)
            opt.zero_grad(); loss.backward(); opt.step(); step += 1
            if writer:
                writer.add_scalar("loss/train", loss.item(), step)
            if step % 50 == 0:
                print(f"  step {step}  train_loss {loss.item():.4f}")
                train_hist.append((step, loss.item()))
                # live loss curve — overwrite each time; open it in VS Code and it
                # auto-refreshes as training progresses.
                render_training_curve(train_hist, val_hist, None, args.out / "il_train_curve.png")
            if args.max_steps and step >= args.max_steps:
                break
        policy.eval(); vl = []
        with torch.no_grad():
            for batch in vdl:
                gpu_batch = {k: v.to(dev) for k, v in batch.items()}
                vl.append(policy.forward(gpu_batch)[0].item())
        policy.train()
        last_val = float(np.mean(vl)); val_hist.append((step, last_val))
        if writer:
            writer.add_scalar("loss/val", last_val, step)
        print(f"epoch {ep}  val_loss {last_val:.4f}  (step {step})")
        if args.max_steps and step >= args.max_steps:
            break

    torch.save(policy.state_dict(), args.out / "act_policy.pt")
    np.savez(args.out / "stats.npz", state_mean=sm, state_std=ss, action_mean=am, action_std=asd)
    print(f"saved policy + stats to {args.out}")

    # Final visualization: loss curve + per-DoF next-step error.
    per_dof = _eval_per_dof(policy, vdl, am, asd, dev)
    curve = render_training_curve(train_hist, val_hist, per_dof,
                                  args.out / "il_train_curve.png", val_final=last_val)
    if writer:
        from .viz import _DOF_LABELS
        for lbl, e in zip(_DOF_LABELS, per_dof):
            writer.add_scalar(f"per_dof/{lbl}", float(e), 0)
        writer.close()
    print(f"hand-joint mean abs err {per_dof[:20].mean():.4f} rad | wrote {curve}")


if __name__ == "__main__":
    main()
