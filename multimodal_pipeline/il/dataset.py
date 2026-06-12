"""PyTorch Dataset over our LeRobot-v3 output for single-hand Wuji IL.

One sample per (kept hand × frame): obs = ego image + that hand's current
26-D config [hand_qpos 20 + ee_pose 6]; target = the ACT-style future chunk of
that same 26-D config. Hands/frames with NaN (not-kept / placeholder arm is
excluded — we only take hand+EE) are dropped. Reads parquet + decodes the ego
video once per episode (frames cached in RAM, deduped across both hands).
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .._system import check_ffmpeg
from ..lerobot_v3.schema import ROBOT_EE_LAYOUT, ROBOT_QPOS_LAYOUT
from ..viz.core import _decode_rgb

HAND_DIM = 26  # hand_qpos 20 + ee transl3 + ee orient3


def hand_vec(qpos46: np.ndarray, ee12: np.ndarray, hand: str) -> np.ndarray:
    """Single-hand 26-D config [hand_qpos 20 + ee 6] (excludes placeholder arm)."""
    h0, h1 = ROBOT_QPOS_LAYOUT[f"{hand}_hand"]
    et = ROBOT_EE_LAYOUT[f"{hand}_ee_transl"]
    eo = ROBOT_EE_LAYOUT[f"{hand}_ee_orient_aa"]
    return np.concatenate([qpos46[h0:h1], ee12[et[0]:et[1]], ee12[eo[0]:eo[1]]])


class WujiActionDataset(Dataset):
    def __init__(self, session_dirs, chunk_size: int = 16, image_size=(192, 256)):
        self.chunk = int(chunk_size)
        self.ih, self.iw = image_size
        ff = check_ffmpeg().ffmpeg
        self._frames: dict[tuple, np.ndarray] = {}   # (si,ep,t) -> CHW float32 [0,1]
        self.samples: list[dict] = []

        for si, sd in enumerate(session_dirs):
            sd = Path(sd)
            info = json.loads((sd / "meta" / "info.json").read_text())
            h, w, _ = info["features"]["observation.images.ego"]["shape"]
            import pyarrow.parquet as pq
            d = pq.read_table(glob.glob(str(sd / "data/chunk-000/*.parquet"))[0]).to_pydict()
            ei = np.array(d["episode_index"])
            Q = np.array(d["observation.robot_qpos"], float)
            E = np.array(d["observation.robot_ee_pose"], float)

            for ep in sorted(set(int(e) for e in ei)):
                rows = [i for i in range(len(ei)) if int(ei[i]) == ep]
                ep_samples = []
                for hand in ("left", "right"):
                    vecs = np.array([hand_vec(Q[i], E[i], hand) for i in rows])  # (T,26)
                    valid = np.isfinite(vecs).all(1)
                    vi = np.where(valid)[0]
                    if len(vi) < 2:
                        continue
                    last_valid = vi[-1]
                    for li in vi:
                        if li >= last_valid:          # need a future → skip the last kept frame
                            continue
                        chunk = np.zeros((self.chunk, HAND_DIM), np.float32)
                        pad = np.ones(self.chunk, bool)
                        prev = vecs[li]
                        for k in range(self.chunk):
                            fi = li + 1 + k
                            if fi < len(rows) and valid[fi]:
                                chunk[k] = vecs[fi]; pad[k] = False; prev = vecs[fi]
                            else:
                                chunk[k] = prev      # repeat last real (masked by pad)
                        ep_samples.append((si, ep, li, hand, vecs[li].astype(np.float32), chunk, pad))
                if not ep_samples:
                    continue
                # decode this episode's video once; cache frames used by any sample
                mp4 = sd / "videos/observation.images.ego/chunk-000" / f"episode_{ep:06d}.mp4"
                rgb = _decode_rgb(ff, mp4, w, h)  # (T,H,W,3) uint8
                used_t = sorted({s[2] for s in ep_samples})
                for t in used_t:
                    if t < len(rgb):
                        img = Image.fromarray(rgb[t]).resize((self.iw, self.ih))
                        self._frames[(si, ep, t)] = (np.asarray(img, np.float32) / 255.0).transpose(2, 0, 1)
                for (si_, ep_, t, hand, state, chunk, pad) in ep_samples:
                    if (si_, ep_, t) in self._frames:
                        self.samples.append({"key": (si_, ep_, t), "hand": hand,
                                             "state": state, "action": chunk, "pad": pad})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        return {
            "observation.images.ego": torch.from_numpy(self._frames[s["key"]]),
            "observation.state": torch.from_numpy(s["state"]),
            "action": torch.from_numpy(s["action"]),
            "action_is_pad": torch.from_numpy(s["pad"]),
        }


__all__ = ["WujiActionDataset", "hand_vec", "HAND_DIM"]
