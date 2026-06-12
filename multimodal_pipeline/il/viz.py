"""IL training visualization: loss curve + per-DoF next-step error chart."""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_DOF_LABELS = (
    [f"f{f}j{j}" for f in range(1, 6) for j in range(1, 5)]
    + ["eeTx", "eeTy", "eeTz", "eeRx", "eeRy", "eeRz"]
)


def render_training_curve(train_hist, val_hist, per_dof, out_path, val_final=None):
    """train_hist/val_hist: list[(step, loss)]; per_dof: (26,) mean-abs next-step err."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    if train_hist:
        ax1.plot([s for s, _ in train_hist], [l for _, l in train_hist],
                 label="train", alpha=0.5, lw=1)
    if val_hist:
        ax1.plot([s for s, _ in val_hist], [l for _, l in val_hist],
                 "o-", color="orange", label="val", ms=4)
    ax1.set_xlabel("step"); ax1.set_ylabel("L1 loss (norm)"); ax1.set_yscale("log")
    ax1.legend(); ax1.grid(alpha=0.3)
    ax1.set_title("ACT training" + (f"  (val {val_final:.3f})" if val_final is not None else ""))

    per = np.asarray(per_dof)
    ax2.bar(range(20), per[:20], color="#4363d8", label="hand joints (rad)")
    ax2.bar(range(20, 26), per[20:], color="#e6194B", label="EE pose")
    ax2.set_xticks(range(26)); ax2.set_xticklabels(_DOF_LABELS, rotation=90, fontsize=6)
    ax2.set_ylabel("mean abs err"); ax2.legend(fontsize=8); ax2.grid(alpha=0.3, axis="y")
    ax2.set_title("per-DoF next-step error (single-step)")

    fig.tight_layout(); fig.savefig(out_path, dpi=110); plt.close(fig)
    return out_path


__all__ = ["render_training_curve"]
