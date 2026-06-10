"""Build a human-verification gallery: N representative sessions.

Picks N sessions evenly spread across the output root, and for each renders:
  - <name>_world.mp4  one frame-locked VLC window, 3x2 grid (all views aligned):
        raw | 2D keypoints+axes | depth / 2D MANO mesh | 3D world | 3D keypoint fit
  - <name>_mano.png   raw | MANO mesh overlay (high-res static cross-check)
all into one flat folder for easy arrow-key browsing. <name> = NN_<english task>__<uuid8>.
"""
from __future__ import annotations

import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from .mano import render_mano_overlay
from .scene3d import render_world_synced, prefuse_world_cache

_REPO = Path(__file__).resolve().parents[2]
OUT_ROOT = _REPO / "output"
GALLERY = _REPO / "artifacts" / "gallery"
# Number of sessions to sample (evenly spread). Pass "all" or a count as argv[1];
# default 10. "all" covers every processed session.
N = (10 if len(sys.argv) < 2
     else (10 ** 9 if sys.argv[1].lower() == "all" else int(sys.argv[1])))
# Parallel pre-fusion workers: argv[2] or $MMPIPE_GALLERY_JOBS, default 3. Only
# the CPU TSDF fusion is parallelised (across processes, EGL-free → safe); the
# Open3D *render* then runs SERIALLY because concurrent offscreen renderers
# deadlock on a single shared GPU.
JOBS = int(sys.argv[2]) if len(sys.argv) > 2 else int(os.environ.get("MMPIPE_GALLERY_JOBS", "3"))


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


def _render_session(i: int, total: int, sess_str: str) -> str:
    """Render one session's world.mp4 + mano.png. Top-level so it is picklable
    for the process pool. Returns a status line."""
    sess = Path(sess_str)
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
        msg = f"  [{i}/{total}] {name} ok (ep{ep})"
    except Exception as exc:
        import traceback; traceback.print_exc()
        msg = f"  [{i}/{total}] {sess.name}: ERROR {type(exc).__name__}: {exc}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(msg, flush=True)
    return msg


def _prefuse(sess_str: str):
    """Pre-build one session's world-mesh cache (CPU only, parallel-safe)."""
    try:
        return prefuse_world_cache(str(Path(sess_str) / "lerobot_dataset"))
    except Exception as exc:
        return f"FUSE-ERR {Path(sess_str).name[:8]}: {type(exc).__name__}: {exc}"


def main() -> None:
    GALLERY.mkdir(parents=True, exist_ok=True)
    sessions = sorted(p for p in OUT_ROOT.iterdir() if p.is_dir() and (p / "lerobot_dataset" / "meta" / "info.json").exists())
    if not sessions:
        print("no sessions"); return
    step = max(1, len(sessions) // N)
    picked = sessions[::step][:N]
    jobs = max(1, min(JOBS, len(picked)))
    print(f"[gallery] {len(picked)} sessions, prefuse jobs={jobs} -> {GALLERY}", flush=True)

    # Phase 1: parallel CPU pre-fusion (builds world_session.ply caches). Skipped
    # for sessions already cached.
    if jobs > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        todo = [s for s in picked if not (s / "world_session.ply").exists()]
        if todo:
            print(f"[gallery] pre-fusing {len(todo)} world meshes ({jobs} parallel)...", flush=True)
            with ProcessPoolExecutor(max_workers=jobs) as ex:
                done = 0
                for f in as_completed([ex.submit(_prefuse, str(s)) for s in todo]):
                    done += 1
                    print(f"  [fuse {done}/{len(todo)}] {f.result()}", flush=True)

    # Phase 2: serial render (Open3D EGL — must not run concurrently).
    for i, sess in enumerate(picked, 1):
        _render_session(i, len(picked), str(sess))
    print(f"[gallery] done -> {GALLERY}", flush=True)


if __name__ == "__main__":
    main()
