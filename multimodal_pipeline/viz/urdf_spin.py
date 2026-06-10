"""Headless MuJoCo render of a hand MJCF: orbit + finger open/close → MP4.

No GUI needed — renders offscreen (EGL) so you just VLC the output. Cycles the
joints from open to closed while the camera orbits once.

Usage:  python3 scripts/viz_urdf_mp4.py <mjcf> [out.mp4]
"""
from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import mujoco

from .._system import check_ffmpeg  # noqa: E402

W, H, N = 640, 480, 150


def main():
    mjcf = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(f"/tmp/{mjcf.stem}_spin.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)

    m = mujoco.MjModel.from_xml_path(str(mjcf))
    d = mujoco.MjData(m)
    r = mujoco.Renderer(m, H, W)
    lo, hi = m.jnt_range[:, 0].copy(), m.jnt_range[:, 1].copy()
    cam = mujoco.MjvCamera()
    cam.lookat = [0, 0, 0.08]; cam.distance = 0.34; cam.elevation = -18

    ff = check_ffmpeg().ffmpeg
    proc = subprocess.Popen(
        [str(ff), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", "30", "-i", "-", "-c:v", "libx264", "-preset", "medium",
         "-crf", "18", "-pix_fmt", "yuv420p", str(out)], stdin=subprocess.PIPE)
    for i in range(N):
        p = i / N
        curl = 0.5 * (1 - np.cos(2 * np.pi * p))      # open → fist → open
        d.qpos[: len(lo)] = lo + curl * (hi - lo)
        mujoco.mj_forward(m, d)
        cam.azimuth = 130 + 360 * p                    # one full orbit
        r.update_scene(d, cam)
        proc.stdin.write(r.render().tobytes())
    proc.stdin.close(); proc.wait()
    print(f"wrote {out}  ({N} frames)")
    sys.stdout.flush()
    os._exit(0)  # skip EGL context teardown (noisy but harmless)


if __name__ == "__main__":
    main()
