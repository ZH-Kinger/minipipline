"""Live animated loss plot — a real matplotlib window (plt.ion), not a static PNG.

Tails the training log and redraws train/val loss curves in real time, so you
watch them grow as the model trains. This needs a display to pop the window, so
**run it in YOUR OWN terminal** (same as the mujoco viewer — my detached shell
can't pop GUI windows):

    python3 -m multimodal_pipeline.il.live_plot [logfile]

Default logfile: artifacts/logs/il_live.log. Close the window to stop.
If it errors with "no display"/"no GUI backend", install a backend:
    sudo apt install python3-tk        # Tk backend, or
    pip install PyQt5                  # Qt backend
"""

from __future__ import annotations

import os
import re
import sys

import matplotlib
for _bk in ("Qt5Agg", "QtAgg", "TkAgg", "GTK4Agg"):  # pick whatever GUI backend exists
    try:
        matplotlib.use(_bk); break
    except Exception:
        continue
import matplotlib.pyplot as plt  # GUI backend on purpose — NOT Agg

_TRAIN = re.compile(r"step (\d+)\s+train_loss ([\d.]+)")
_VAL = re.compile(r"val_loss ([\d.]+)\s+\(step (\d+)\)")


def main():
    log = sys.argv[1] if len(sys.argv) > 1 else "artifacts/logs/il_live.log"
    print(f"live-plotting {log} — close the window to stop")
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 5))
    l_tr, = ax.plot([], [], color="#4363d8", alpha=0.6, lw=1, label="train")
    l_va, = ax.plot([], [], "o-", color="orange", ms=4, label="val")
    ax.set_xlabel("step"); ax.set_ylabel("L1 loss"); ax.set_yscale("log")
    ax.legend(); ax.grid(alpha=0.3)

    while plt.fignum_exists(fig.number):
        trx, trv, vax, vav = [], [], [], []
        if os.path.exists(log):
            txt = open(log, encoding="utf-8", errors="replace").read()
            for s, v in _TRAIN.findall(txt):
                trx.append(int(s)); trv.append(float(v))
            for v, s in _VAL.findall(txt):
                vax.append(int(s)); vav.append(float(v))
        l_tr.set_data(trx, trv); l_va.set_data(vax, vav)
        if trx:
            ax.relim(); ax.autoscale_view()
            ax.set_title(f"IL training — live   step {trx[-1]}   train {trv[-1]:.3f}"
                         + (f"   val {vav[-1]:.3f}" if vav else ""))
        plt.pause(1.0)   # redraw + process window events every 1s
    print("window closed")


if __name__ == "__main__":
    main()
