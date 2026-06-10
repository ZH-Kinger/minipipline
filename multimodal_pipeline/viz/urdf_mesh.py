"""Render a URDF's visual meshes for inspection (default: vendored Wuji hand).

Loads each link's <visual> STL, places it with forward kinematics (rest pose =
joint-limit midpoints), and renders a few viewpoints to a PNG. Pure trimesh +
matplotlib (no GL), so it works headless.

Usage:  python3 scripts/viz_urdf.py <urdf> <mesh_dir> [out.png]
"""
from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from ..retarget.urdf_fk import KinematicTree, _rpy_to_matrix, _homogeneous  # noqa: E402


def _origin(e):
    o = e.find("origin")
    if o is None:
        return np.eye(4)
    xyz = np.fromstring(o.get("xyz", "0 0 0"), sep=" ")
    rpy = np.fromstring(o.get("rpy", "0 0 0"), sep=" ")
    return _homogeneous(_rpy_to_matrix(rpy), xyz)


def main():
    urdf = Path(sys.argv[1]); mesh_dir = Path(sys.argv[2])
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("/tmp/urdf_view.png")
    tree = KinematicTree(urdf); root = ET.parse(urdf).getroot()

    vis = {}
    for link in root.findall("link"):
        for v in link.findall("visual"):
            g = v.find("geometry/mesh")
            if g is None:
                continue
            fn = g.get("filename").split("/")[-1]
            sc = g.get("scale")
            sc = np.fromstring(sc, sep=" ") if sc else np.ones(3)
            vis.setdefault(link.get("name"), []).append((fn, _origin(v), sc))

    frames, _ = tree.fk(tree.midpoint())
    tris = []
    for ln, items in vis.items():
        if ln not in frames:
            continue
        for fn, oT, sc in items:
            m = trimesh.load(mesh_dir / fn, force="mesh")
            V = np.asarray(m.vertices) * sc
            T = frames[ln] @ oT
            Vt = (T[:3, :3] @ V.T).T + T[:3, 3]
            tris.append(Vt[np.asarray(m.faces)])
    tris = np.concatenate(tris, 0)
    P = tris.reshape(-1, 3)
    print("triangles", len(tris), "bbox_min", P.min(0).round(3), "bbox_max", P.max(0).round(3))

    # shade faces by normal for some depth cue
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n /= (np.linalg.norm(n, axis=1, keepdims=True) + 1e-9)
    shade = 0.45 + 0.55 * np.clip(n @ np.array([0.3, 0.3, 0.9]), 0, 1)
    base = np.array([0.80, 0.64, 0.38])
    fc = np.clip(shade[:, None] * base, 0, 1)

    views = [(18, -70), (18, 20), (80, -90), (-5, -90)]
    titles = ["3/4 view", "other side", "top (palm)", "front"]
    fig = plt.figure(figsize=(13, 3.4))
    for i, (el, az) in enumerate(views):
        ax = fig.add_subplot(1, 4, i + 1, projection="3d")
        pc = Poly3DCollection(tris, facecolors=fc, edgecolor="none")
        ax.add_collection3d(pc)
        for setlim, a in ((ax.set_xlim, 0), (ax.set_ylim, 1), (ax.set_zlim, 2)):
            setlim(P[:, a].min(), P[:, a].max())
        ax.set_box_aspect([np.ptp(P[:, a]) for a in range(3)])
        ax.view_init(el, az); ax.set_axis_off(); ax.set_title(titles[i], fontsize=9)
    fig.suptitle(f"{urdf.name}  ({len(vis)} visual links, rest pose)")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print("wrote", out)


if __name__ == "__main__":
    main()
