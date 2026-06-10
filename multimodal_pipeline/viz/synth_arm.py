"""Spin MP4 of the synthetic arm+hand URDF: hand STL meshes + arm link segments.

The synthetic arm links (base/arm_link1/arm_link2) are placeholder kinematics
with NO visual mesh, so they're drawn as thick segments between FK joint
positions; the hand links keep their real STL meshes. Rest pose, camera orbits
once. Pure trimesh + matplotlib (headless); the mesh is built once and only the
view rotates, so it's fast.

Usage:  python3 scripts/viz_synth_arm_mp4.py <synthetic_urdf> <hand_mesh_dir> [out.mp4]
"""
from __future__ import annotations

import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .._system import check_ffmpeg  # noqa: E402
from ..retarget.urdf_fk import KinematicTree, _rpy_to_matrix, _homogeneous  # noqa: E402

W, H, N = 640, 480, 90


def _origin(e):
    o = e.find("origin")
    if o is None:
        return np.eye(4)
    return _homogeneous(_rpy_to_matrix(np.fromstring(o.get("rpy", "0 0 0"), sep=" ")),
                        np.fromstring(o.get("xyz", "0 0 0"), sep=" "))


def main():
    urdf = Path(sys.argv[1]); mesh_dir = Path(sys.argv[2])
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else Path(f"/tmp/{urdf.stem}_spin.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    tree = KinematicTree(urdf); root = ET.parse(urdf).getroot()

    vis = {}
    for link in root.findall("link"):
        for v in link.findall("visual"):
            g = v.find("geometry/mesh")
            if g is None:
                continue
            sc = g.get("scale"); sc = np.fromstring(sc, sep=" ") if sc else np.ones(3)
            vis.setdefault(link.get("name"), []).append((g.get("filename").split("/")[-1], _origin(v), sc))

    frames, _ = tree.fk(tree.midpoint())
    tris = []
    for ln, items in vis.items():
        if ln not in frames:
            continue
        for fn, oT, sc in items:
            m = trimesh.load(mesh_dir / fn, force="mesh")
            T = frames[ln] @ oT
            Vt = (T[:3, :3] @ (np.asarray(m.vertices) * sc).T).T + T[:3, 3]
            tris.append(Vt[np.asarray(m.faces)])
    tris = np.concatenate(tris, 0)
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n /= (np.linalg.norm(n, axis=1, keepdims=True) + 1e-9)
    shade = 0.45 + 0.55 * np.clip(n @ np.array([0.3, 0.3, 0.9]), 0, 1)
    fc = np.clip(shade[:, None] * np.array([0.78, 0.80, 0.85]), 0, 1)

    # arm segments: joints whose child link has no mesh (base/arm links) → the arm chain
    arm_segs = [(frames[j.parent][:3, 3], frames[j.child][:3, 3])
                for j in tree.joints if j.child not in vis or j.parent not in vis]
    allpts = np.vstack([tris.reshape(-1, 3)] + [np.array([a, b]) for a, b in arm_segs])
    ctr = allpts.mean(0); rng = np.ptp(allpts, 0).max() * 0.6

    ff = check_ffmpeg().ffmpeg
    proc = subprocess.Popen(
        [str(ff), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", "30", "-i", "-", "-c:v", "libx264", "-preset", "medium",
         "-crf", "18", "-pix_fmt", "yuv420p", str(out)], stdin=subprocess.PIPE)
    dpi = 100
    for i in range(N):
        fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi, facecolor="#101014")
        ax = fig.add_subplot(111, projection="3d"); ax.set_facecolor("#101014")
        ax.add_collection3d(Poly3DCollection(tris, facecolors=fc, edgecolor="none"))
        for a, b in arm_segs:
            ax.plot(*zip(a, b), color="#ff9030", lw=9, solid_capstyle="round")
            ax.scatter(*b, color="#ffd060", s=40)
        ax.set_xlim(ctr[0] - rng, ctr[0] + rng); ax.set_ylim(ctr[1] - rng, ctr[1] + rng); ax.set_zlim(ctr[2] - rng, ctr[2] + rng)
        ax.set_box_aspect((1, 1, 1)); ax.set_axis_off()
        ax.view_init(elev=12, azim=130 + 360 * i / N)
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        proc.stdin.write(np.ascontiguousarray(buf).tobytes())
        plt.close(fig)
    proc.stdin.close(); proc.wait()
    print(f"wrote {out}  ({N} frames, {len(arm_segs)} arm segments)")


if __name__ == "__main__":
    main()
