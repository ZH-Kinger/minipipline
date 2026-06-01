"""Build a human-verification gallery: N representative sessions.

Picks N sessions evenly spread across the output root, and for each renders:
  - <name>_world.mp4  one frame-locked VLC window, 3x2 grid (all views aligned):
        raw | 2D keypoints+axes | depth / 2D MANO mesh | 3D world | 3D keypoint fit
  - <name>_mano.png   raw | MANO mesh overlay (high-res static cross-check)
all into one flat folder for easy arrow-key browsing. <name> = NN_<english task>__<uuid8>.
"""
from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from multimodal_pipeline.viz_mano import render_mano_overlay
from multimodal_pipeline.viz3d import render_world_synced

_REPO = Path(__file__).resolve().parents[1]
OUT_ROOT = _REPO / "output"
GALLERY = _REPO / "artifacts" / "gallery"
N = 10


def _both_hands_frame(ds: Path) -> tuple[int, int]:
    """Return (episode_index, has_both) picking the episode with the most both-hands frames."""
    d = pq.read_table(ds / "data" / "chunk-000" / "file-000.parquet").to_pydict()
    ep = np.array(d["episode_index"])
    m = np.array([list(x) for x in d["state_mask"]])
    both = m[:, 0] & m[:, 1]
    import collections
    c = collections.Counter(int(ep[i]) for i in range(len(ep)) if both[i])
    if c:
        return max(c, key=c.get), 1
    return int(ep[0]) if len(ep) else 0, 0


def _task_name(ds: Path) -> str:
    try:
        t = pq.read_table(ds / "meta" / "tasks.parquet").to_pydict()
        txt = t["task"][0] if t.get("task") else ""
    except Exception:
        txt = ""
    s = re.sub(r"[^A-Za-z0-9 ]", "", txt).strip().replace(" ", "-")[:40]
    return s or "task"


def main() -> None:
    GALLERY.mkdir(parents=True, exist_ok=True)
    sessions = sorted(p for p in OUT_ROOT.iterdir() if p.is_dir() and (p / "lerobot_dataset" / "meta" / "info.json").exists())
    if not sessions:
        print("no sessions"); return
    step = max(1, len(sessions) // N)
    picked = sessions[::step][:N]
    print(f"[gallery] {len(picked)} sessions -> {GALLERY}")
    for i, sess in enumerate(picked, 1):
        ds = sess / "lerobot_dataset"
        uuid8 = sess.name[:8]
        ep, _ = _both_hands_frame(ds)
        name = f"{i:02d}_{_task_name(ds)}__{uuid8}"
        tmp = GALLERY / f".tmp_{uuid8}"
        try:
            world = render_world_synced(ds, ep, tmp)
            shutil.move(str(world), GALLERY / f"{name}_world.mp4")
            mano = render_mano_overlay(ds, ep, tmp)
            shutil.move(str(mano), GALLERY / f"{name}_mano.png")
            print(f"  [{i}/{len(picked)}] {name} ok (ep{ep})")
        except Exception as exc:
            import traceback; traceback.print_exc()
            print(f"  [{i}/{len(picked)}] {sess.name}: ERROR {type(exc).__name__}: {exc}")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    print(f"[gallery] done -> {GALLERY}")


if __name__ == "__main__":
    main()
