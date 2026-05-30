"""Clear GPU-rendered 3D video: depth point-cloud + MANO hand mesh per frame.

Uses Open3D's offscreen renderer (EGL headless on the GPU) to render, for each
frame, the back-projected coloured scene point cloud plus the fitted MANO hand
meshes (both hands, with the left-hand mirror fix), from a gently orbiting
virtual camera — a crisp 3D "world" view, unlike the flat matplotlib scatter.

Needs ``open3d`` + the MANO assets / torch / smplx (see viz_mano).
"""

from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path

import numpy as np

from ._system import check_ffmpeg
from .viz_mano import (
    _check_mano_assets, _mano_vertices, _patch_numpy_for_chumpy, _resolve_mano_dir,
)
from .visualize import _decode_depth_gray16, _decode_rgb, _intrinsics

_MIR = np.array([1.0, -1.0, -1.0], dtype=np.float64)


def _backproject(depth_m, rgb, fx, fy, cx, cy, stride):
    h, w = depth_m.shape
    ys, xs = np.mgrid[0:h:stride, 0:w:stride]
    z = depth_m[::stride, ::stride]
    ok = z > 1e-3
    X = (xs[ok] - cx) * z[ok] / fx
    Y = (ys[ok] - cy) * z[ok] / fy
    pts = np.stack([X, Y, z[ok]], axis=1)
    cols = rgb[::stride, ::stride][ok].astype(np.float64) / 255.0
    return pts, cols


def render_3d_video(dataset_root: Path, ep_idx: int, out_dir: Path, *,
                    mano_dir=None, fps: float = 30.0, stride: int = 4,
                    orbit_deg: float = 35.0) -> Path:
    import open3d as o3d
    from open3d.visualization import rendering
    _patch_numpy_for_chumpy()
    import smplx

    mano_dir = _resolve_mano_dir(mano_dir); _check_mano_assets(mano_dir)
    ffb = check_ffmpeg()
    info = json.loads((dataset_root / "meta" / "info.json").read_text(encoding="utf-8"))
    h, w, _ = info["features"]["observation.images.ego"]["shape"]
    fx, fy, cx, cy = _intrinsics(dataset_root, w, h)

    import pyarrow.parquet as pq
    data = pq.read_table(dataset_root / "data" / "chunk-000" / "file-000.parquet").to_pydict()
    rows = [i for i, e in enumerate(data["episode_index"]) if int(e) == ep_idx]
    states = [np.asarray(data["observation.state"][i], dtype=np.float64) for i in rows]

    rgb = _decode_rgb(ffb.ffmpeg, dataset_root / "videos" / "observation.images.ego" / "chunk-000" / f"episode_{ep_idx:06d}.mp4", w, h)
    depth_mkv = dataset_root / "videos" / "observation.images.depth" / "chunk-000" / f"episode_{ep_idx:06d}.mkv"
    if not depth_mkv.exists():
        raise FileNotFoundError(f"no depth stream for episode {ep_idx}")
    depth = _decode_depth_gray16(ffb.ffmpeg, depth_mkv, w, h)
    T = min(len(rgb), len(depth), len(states))

    flat = os.environ.get("MMPIPE_MANO_FLAT_MEAN", "true").strip().lower() in {"1", "true", "yes"}
    models = {"left": smplx.MANO(str(mano_dir / "MANO_LEFT.pkl"), is_rhand=False, use_pca=False, flat_hand_mean=flat),
              "right": smplx.MANO(str(mano_dir / "MANO_RIGHT.pkl"), is_rhand=True, use_pca=False, flat_hand_mean=flat)}
    faces = {k: m.faces.astype(np.int32) for k, m in models.items()}

    # Render at the source resolution; one offscreen renderer reused per frame.
    rw, rh = w, h
    ren = rendering.OffscreenRenderer(rw, rh)
    ren.scene.set_background([0.05, 0.05, 0.05, 1.0])
    ren.scene.scene.set_sun_light([0.3, 0.3, -1.0], [1, 1, 1], 60000)
    ren.scene.scene.enable_sun_light(True)
    pc_mat = rendering.MaterialRecord(); pc_mat.shader = "defaultUnlit"; pc_mat.point_size = 3.0
    mesh_mat = {}
    for side, col in (("left", (0.0, 0.78, 1.0, 1.0)), ("right", (1.0, 0.47, 0.0, 1.0))):
        m = rendering.MaterialRecord(); m.shader = "defaultLit"; m.base_color = col
        mesh_mat[side] = m

    out_dir.mkdir(parents=True, exist_ok=True)
    out_mp4 = out_dir / f"episode_{ep_idx:06d}_3d.mp4"
    proc = subprocess.Popen(
        [str(ffb.ffmpeg), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{rw}x{rh}", "-r", f"{fps}", "-i", "-", "-c:v", "libx264", "-preset", "veryfast",
         "-pix_fmt", "yuv420p", str(out_mp4)], stdin=subprocess.PIPE)

    # Stable target: median wrist position over the clip (the action region), so
    # the camera doesn't jitter and always frames the hands.
    wrists = []
    for st in states:
        for base in (0, 61):
            wp = st[base:base + 3]
            if np.any(wp):
                wrists.append(wp)
    target = np.median(np.array(wrists), axis=0) if wrists else np.array([0.0, 0.0, 0.6])

    for t in range(T):
        ren.scene.clear_geometry()
        d_m = depth[t].astype(np.float64) * 0.001
        pts, cols = _backproject(d_m, rgb[t], fx, fy, cx, cy, stride)
        # Keep the foreground around the hands; drop far background clutter.
        if len(pts):
            keep = (pts[:, 2] < target[2] + 0.7) & (np.linalg.norm(pts - target, axis=1) < 1.2)
            pts, cols = pts[keep], cols[keep]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(cols)
        ren.scene.add_geometry("pc", pcd, pc_mat)

        st = states[t]
        for side, base in (("left", 0), ("right", 61)):
            transl = st[base:base + 3]; orient = st[base + 3:base + 6].copy()
            pose = st[base + 6:base + 51].copy(); betas = st[base + 51:base + 61]
            if np.allclose(pose, 0) and np.allclose(betas, 0):
                continue
            if side == "left":
                orient = orient * _MIR; pose = (pose.reshape(15, 3) * _MIR).reshape(45)
            verts = _mano_vertices(models[side], transl, orient, pose, betas)
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(verts.astype(np.float64))
            mesh.triangles = o3d.utility.Vector3iVector(faces[side])
            mesh.compute_vertex_normals()
            ren.scene.add_geometry(f"hand_{side}", mesh, mesh_mat[side])

        # Orbiting virtual camera around the hand region: sweep azimuth ±orbit_deg
        # and a slight downward elevation, pulled back so the hands + foreground
        # fill the frame. Camera +Y is down in this frame, so "above" is -Y.
        ang = math.radians(orbit_deg * math.sin(2 * math.pi * t / max(T, 1)))
        radius = 0.75
        eye = target + np.array([
            radius * math.sin(ang),
            -0.35,                          # above the hands (Y is down)
            -radius * math.cos(ang),        # in front, toward the real camera
        ])
        ren.setup_camera(55.0, target.tolist(), eye.tolist(), [0.0, -1.0, 0.0])
        img = np.asarray(ren.render_to_image())
        proc.stdin.write(img[:, :, :3].astype(np.uint8).tobytes())

    proc.stdin.close(); proc.wait()
    return out_mp4


# ---------------------------------------------------------------------------
# One synced window: all 2D + 3D views, frame-locked into a single MP4.
# ---------------------------------------------------------------------------


def _label(img_arr, text):
    """Stamp a small caption in the top-left corner of an (H,W,3) uint8 panel."""
    from PIL import Image, ImageDraw
    im = Image.fromarray(img_arr); d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 8 + 7 * len(text), 16], fill=(0, 0, 0))
    d.text((4, 3), text, fill=(255, 255, 255))
    return np.asarray(im)


def _kp_geometries(o3d, kp, bones):
    """Build (joints PointCloud, bones LineSet) for the 2×21×3 camera-frame keypoints."""
    pts, cols, lines, lcols = [], [], [], []
    palette = ((0.0, 0.85, 1.0), (1.0, 0.5, 0.0))  # left cyan, right orange
    for hand in range(2):
        h = kp[hand]
        if not np.all(np.isfinite(h)):
            continue
        base = len(pts)
        for j in range(21):
            pts.append(h[j]); cols.append(palette[hand])
        for a, b in bones:
            lines.append([base + a, base + b]); lcols.append(palette[hand])
    geoms = []
    if pts:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.asarray(pts, np.float64))
        pcd.colors = o3d.utility.Vector3dVector(np.asarray(cols, np.float64))
        geoms.append(("kpts", pcd))
    if lines:
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(np.asarray(pts, np.float64))
        ls.lines = o3d.utility.Vector2iVector(np.asarray(lines, np.int32))
        ls.colors = o3d.utility.Vector3dVector(np.asarray(lcols, np.float64))
        geoms.append(("bones", ls))
    return geoms


_PALETTE = ((0.0, 0.78, 1.0), (1.0, 0.47, 0.0))  # left cyan, right orange


def _median_depth(depth_u16):
    """Despeckle a uint16 depth map (remove flying pixels) with a 5×5 median."""
    from scipy.ndimage import median_filter
    return median_filter(depth_u16, size=5)


def _hand_bbox_mask(depth, kp, valid, fx, fy, cx, cy, pad=100):
    """Zero out depth around each valid hand, extended down toward the image
    bottom to also cut the forearm (egocentric arms enter from the bottom).

    Keeps the moving hands AND arms OUT of the fused static world (they'd
    otherwise smear across the work area). Multi-frame fusion refills the table
    from frames where the hand is elsewhere. Returns a copy.
    """
    out = depth.copy()
    h, w = depth.shape
    for hand in range(2):
        if not valid(hand):
            continue
        p = kp[hand]
        z = p[:, 2]
        ok = np.isfinite(z) & (z > 1e-3)
        if not ok.any():
            continue
        u = fx * p[ok, 0] / z[ok] + cx
        v = fy * p[ok, 1] / z[ok] + cy
        x0 = max(0, int(u.min()) - pad); x1 = min(w, int(u.max()) + pad)
        y0 = max(0, int(v.min()) - pad); yhb = min(h, int(v.max()) + pad)
        out[y0:yhb, x0:x1] = 0                       # hand
        # forearm: a wider band from the hand down to the image bottom (the arm
        # fans out wider than the hand and enters from below in egocentric view).
        fx0 = max(0, x0 - 2 * pad); fx1 = min(w, x1 + 2 * pad)
        out[yhb:h, fx0:fx1] = 0
    return out


def _fuse_world(o3d, rgb, depth, w2c, fx, fy, cx, cy, kps, valids, T, ref, *,
                voxel=0.006, trunc=1.6, max_tris=120000, roi=0.5):
    """TSDF-fuse the whole clip (moving egocentric camera, hands masked) into one
    clean world mesh, then express it in the reference frame's camera coordinates.

    Uses the real per-frame ``extrinsics_w2c`` so the moving head camera fuses
    coherently; median-filtered depth removes flying pixels. Returns an Open3D
    TriangleMesh in ref-camera coordinates (static, hole-filled, denoised)."""
    vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel, sdf_trunc=4 * voxel,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    h, w = depth.shape[1], depth.shape[2]
    intr = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)
    c2w = [np.linalg.inv(m) for m in w2c]
    # Only integrate frames where BOTH hands have valid keypoints — those are the
    # frames we can fully mask. Frames with dropped keypoints would fuse the
    # unmaskable hand/arm straight into the static world (the flesh-coloured
    # smear). Fall back to any-valid frames if too few both-valid exist.
    integ = [t for t in range(T) if valids(t, 0) and valids(t, 1)]
    if len(integ) < 5:
        integ = [t for t in range(T) if valids(t, 0) or valids(t, 1)] or list(range(T))
    centers = []
    for t in integ:
        dep = _median_depth(depth[t])
        dep = _hand_bbox_mask(dep, kps[t], lambda hd, _t=t: valids(_t, hd), fx, fy, cx, cy)
        color = o3d.geometry.Image(np.ascontiguousarray(rgb[t]))
        depth_img = o3d.geometry.Image(np.ascontiguousarray(dep.astype(np.uint16)))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth_img, depth_scale=1000.0, depth_trunc=trunc,
            convert_rgb_to_intensity=False)
        vol.integrate(rgbd, intr, w2c[t])
        for hd in (0, 1):
            if valids(t, hd):
                wc = kps[t][hd, 0]
                centers.append(c2w[t][:3, :3] @ wc + c2w[t][:3, 3])  # wrist in world
    mesh = vol.extract_triangle_mesh()

    # --- clean the noisy single-view reconstruction ---
    # 1) crop to the workspace ROI around the hands (drop torn far/edge shards)
    if centers:
        ctr = np.median(np.array(centers), axis=0)
        bb = o3d.geometry.AxisAlignedBoundingBox(ctr - roi, ctr + roi)
        mesh = mesh.crop(bb)
    # 2) drop small disconnected triangle clusters (flying fragments)
    if len(mesh.triangles):
        idx, ntri, _ = mesh.cluster_connected_triangles()
        idx = np.asarray(idx); ntri = np.asarray(ntri)
        if len(ntri):
            keep = ntri[idx] >= max(int(0.02 * ntri.max()), 80)
            mesh.remove_triangles_by_mask(~keep)
            mesh.remove_unreferenced_vertices()
    if len(mesh.triangles) > max_tris:
        mesh = mesh.simplify_quadric_decimation(max_tris)
    mesh.remove_degenerate_triangles()
    # 3) Taubin smoothing: denoise the bumpy surface without shrinking it
    if len(mesh.triangles):
        mesh = mesh.filter_smooth_taubin(number_of_iterations=12)
    mesh.transform(w2c[ref])  # world -> ref-camera frame
    mesh.compute_vertex_normals()
    return mesh


def _cyl(o3d, p, q, r):
    """A cylinder TriangleMesh spanning points p→q with radius r (+Z aligned then rotated)."""
    d = q - p
    L = float(np.linalg.norm(d))
    if L < 1e-6:
        return None
    cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=r, height=L, resolution=12, split=1)
    z = np.array([0.0, 0.0, 1.0])
    axis = d / L
    v = np.cross(z, axis); s = np.linalg.norm(v); c = float(np.dot(z, axis))
    if s < 1e-8:
        R = np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))
    cyl.rotate(R, center=(0, 0, 0))
    cyl.translate((p + q) / 2.0)
    return cyl


def _kp_solid(o3d, kp, bones, *, joint_r=0.008, bone_r=0.004):
    """Build solid (sphere joints + cylinder bones) meshes per valid hand.

    Returns list of (name, TriangleMesh, hand_index) for lit rendering — a crisp
    3D skeleton instead of flat point splats."""
    out = []
    for hand in range(2):
        h = kp[hand]
        if not np.all(np.isfinite(h)):
            continue
        merged = o3d.geometry.TriangleMesh()
        for j in range(21):
            s = o3d.geometry.TriangleMesh.create_sphere(radius=joint_r, resolution=10)
            s.translate(h[j]); merged += s
        for a, b in bones:
            cyl = _cyl(o3d, h[a], h[b], bone_r)
            if cyl is not None:
                merged += cyl
        merged.compute_vertex_normals()
        out.append((f"hand{hand}", merged, hand))
    return out


def _setup_lit(ren, rendering):
    """Sun + a softer fill light + ambient, so meshes read with shape not flatness."""
    sc = ren.scene.scene
    sc.set_sun_light([0.3, 0.4, -1.0], [1.0, 1.0, 1.0], 75000)
    sc.enable_sun_light(True)
    for fn, args in (
        ("add_directional_light", ("fill", [1.0, 1.0, 1.0], [-0.5, -0.2, 1.0], 28000, False)),
    ):
        try:
            getattr(sc, fn)(*args)
        except Exception:
            pass
    try:
        ren.scene.set_lighting(ren.scene.LightingProfile.NO_SHADOWS, (0.3, 0.4, -1.0))
    except Exception:
        pass


def _downscale(arr, w, h):
    """Lanczos-downscale a supersampled (H*ss, W*ss, 3) render to (h, w, 3)."""
    if arr.shape[1] == w and arr.shape[0] == h:
        return np.ascontiguousarray(arr)
    from PIL import Image as _Im
    return np.asarray(_Im.fromarray(arr).resize((w, h), _Im.LANCZOS))


def _traj_panel(traj_l, traj_r, t, w, h, fig_ctx):
    """Render the wrist-trajectory curve plot for frame t at exactly (h,w,3) px.

    Plots left & right wrist position (x,y,z, camera frame) over the whole clip
    as curves, with a vertical cursor at the current frame — a frame-locked
    "fitted curve" view. NaN gaps (no detection) break the lines naturally.
    """
    import numpy as _np
    fig, ax = fig_ctx
    ax.clear()
    n = traj_l.shape[0]
    frames = _np.arange(n)
    styles = ("-", "--", ":")  # x, y, z
    for traj, base_col, tag in ((traj_l, (0.0, 0.78, 1.0), "L"), (traj_r, (1.0, 0.47, 0.0), "R")):
        for c, comp in enumerate("xyz"):
            ax.plot(frames, traj[:, c], styles[c], color=base_col, linewidth=1.4,
                    label=f"{tag}-{comp}")
    ax.axvline(t, color="white", linewidth=1.5, alpha=0.9)
    # mark current value of each finite curve
    for traj, base_col in ((traj_l, (0.0, 0.78, 1.0)), (traj_r, (1.0, 0.47, 0.0))):
        for c in range(3):
            v = traj[t, c]
            if _np.isfinite(v):
                ax.plot([t], [v], "o", color=base_col, markersize=4)
    ax.set_title("wrist trajectory (camera frame, metres)", color="white", fontsize=9)
    ax.set_xlabel("frame", color="white", fontsize=8)
    ax.set_xlim(0, max(n - 1, 1))
    ax.tick_params(colors="white", labelsize=7)
    for s in ax.spines.values():
        s.set_color("white")
    ax.legend(ncol=6, fontsize=6, loc="upper center", framealpha=0.2,
              labelcolor="white", columnspacing=0.8, handlelength=1.2)
    fig.canvas.draw()
    arr = _np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    if arr.shape[:2] != (h, w):
        from PIL import Image as _Im
        arr = _np.asarray(_Im.fromarray(arr).resize((w, h)))
    return _np.ascontiguousarray(arr)


def render_world_synced(dataset_root: Path, ep_idx: int, out_dir: Path, *,
                        mano_dir=None, fps: float = 30.0, ss: int = 2,
                        orbit_deg: float = 35.0) -> Path:
    """One frame-locked MP4 for a single VLC window, 2×2 grid (aligned timestamps):

        [ raw RGB           | 3D keypoint skeleton fit       ]
        [ wrist trajectory  | 3D world (fused mesh + MANO)   ]

    Quality build: the world tile is a **TSDF-fused** scene mesh — the whole
    clip's median-filtered depth is integrated with the real per-frame
    ``extrinsics_w2c`` (hands masked out) into one clean, hole-filled surface,
    expressed in a reference frame's camera coordinates. The per-frame MANO hands
    (camera-frame fit, wrist pinned to the real keypoint) are rigidly carried
    into that reference frame, so the hands move within a static world model.
    3D keypoints are solid spheres+cylinders. Both 3D tiles are supersampled
    (``ss``×) and downscaled (Lanczos) for crisp, anti-aliased edges; the MP4 is
    H.264 ``-preset slow -crf 16``. Needs open3d + MANO + scipy + matplotlib + ffmpeg.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import open3d as o3d
    from open3d.visualization import rendering
    import pyarrow.parquet as pq
    from PIL import Image, ImageDraw
    from .visualize import (_decode_rgb as _vd_rgb, _decode_depth_gray16, _BONES,
                            _draw_hand, _draw_wrist_axes, _project,
                            _LEFT_COLOR, _RIGHT_COLOR)
    _patch_numpy_for_chumpy()
    import smplx

    mano_dir = _resolve_mano_dir(mano_dir); _check_mano_assets(mano_dir)
    ffb = check_ffmpeg()
    info = json.loads((dataset_root / "meta" / "info.json").read_text(encoding="utf-8"))
    h, w, _ = info["features"]["observation.images.ego"]["shape"]
    fx, fy, cx, cy = _intrinsics(dataset_root, w, h)

    data = pq.read_table(dataset_root / "data" / "chunk-000" / "file-000.parquet").to_pydict()
    rows = [i for i, e in enumerate(data["episode_index"]) if int(e) == ep_idx]
    states = [np.asarray(data["observation.state"][i], dtype=np.float64) for i in rows]
    kp_flat = [data["observation.hand_keypoints"][i] for i in rows]
    masks = [list(data["state_mask"][i]) for i in rows]
    have_ext = "extrinsics_w2c" in data
    w2c = ([np.asarray(data["extrinsics_w2c"][i], dtype=np.float64).reshape(4, 4) for i in rows]
           if have_ext else None)

    rgb = _vd_rgb(ffb.ffmpeg, dataset_root / "videos" / "observation.images.ego" / "chunk-000" / f"episode_{ep_idx:06d}.mp4", w, h)
    depth_mkv = dataset_root / "videos" / "observation.images.depth" / "chunk-000" / f"episode_{ep_idx:06d}.mkv"
    depth = _decode_depth_gray16(ffb.ffmpeg, depth_mkv, w, h) if depth_mkv.exists() else None
    T = min(len(rgb), len(states), len(kp_flat))

    kps = [np.asarray(kp_flat[t], dtype=np.float64).reshape(2, 21, 3) for t in range(T)]
    def _valid(t, hand):
        return bool(masks[t][hand]) and np.all(np.isfinite(kps[t][hand]))

    traj_l = np.full((T, 3), np.nan); traj_r = np.full((T, 3), np.nan)
    for t in range(T):
        if _valid(t, 0): traj_l[t] = kps[t][0, 0]
        if _valid(t, 1): traj_r[t] = kps[t][1, 0]

    flat = os.environ.get("MMPIPE_MANO_FLAT_MEAN", "true").strip().lower() in {"1", "true", "yes"}
    models = {"left": smplx.MANO(str(mano_dir / "MANO_LEFT.pkl"), is_rhand=False, use_pca=False, flat_hand_mean=flat),
              "right": smplx.MANO(str(mano_dir / "MANO_RIGHT.pkl"), is_rhand=True, use_pca=False, flat_hand_mean=flat)}
    faces = {k: m.faces.astype(np.int32) for k, m in models.items()}

    # Reference frame for the static world = the middle frame that has any valid
    # hand (its camera coords anchor the fused mesh + the hand transforms).
    valid_frames = [t for t in range(T) if _valid(t, 0) or _valid(t, 1)]
    ref = valid_frames[len(valid_frames) // 2] if valid_frames else T // 2
    c2w = [np.linalg.inv(m) for m in w2c] if w2c is not None else None
    def _T_ref(t):  # camera(t) -> reference-camera frame
        if w2c is None:
            return np.eye(4)
        return w2c[ref] @ c2w[t]

    # --- Fuse the static world mesh (ref-camera frame) ---
    world_mesh = None
    if depth is not None and w2c is not None:
        depth_stack = depth[:T]
        world_mesh = _fuse_world(o3d, rgb[:T], depth_stack, w2c, fx, fy, cx, cy,
                                 kps, _valid, T, ref)

    # --- Two supersampled offscreen renderers (world + keypoints) ---
    RW, RH = w * ss, h * ss
    fxs, fys, cxs, cys = fx * ss, fy * ss, cx * ss, cy * ss

    # A single offscreen renderer reused for both 3D tiles (clear + re-add per
    # pass). Two simultaneous renderers trip a Filament double-free at teardown,
    # which would abort a multi-episode gallery run — one renderer is safe.
    ren = rendering.OffscreenRenderer(RW, RH)
    ren.scene.set_background([0.04, 0.04, 0.05, 1.0])
    _setup_lit(ren, rendering)

    world_mat = rendering.MaterialRecord(); world_mat.shader = "defaultLit"
    world_mat.base_color = (1.0, 1.0, 1.0, 1.0)
    hand_mat = {}
    for hand, col in ((0, (0.0, 0.78, 1.0, 1.0)), (1, (1.0, 0.47, 0.0, 1.0))):
        m = rendering.MaterialRecord(); m.shader = "defaultLit"; m.base_color = col
        hand_mat[hand] = m

    # Orbit targets: ref-frame wrist median (world tile) and camera-frame wrist
    # median (keypoint tile).
    wlist_ref, wlist_cam = [], []
    for t in range(T):
        for hd in (0, 1):
            if _valid(t, hd):
                wc = kps[t][hd, 0]; wlist_cam.append(wc)
                Tr = _T_ref(t); wlist_ref.append(Tr[:3, :3] @ wc + Tr[:3, 3])
    tgt_w = np.median(np.array(wlist_ref), axis=0) if wlist_ref else np.array([0.0, 0.0, 0.6])
    tgt_k = np.median(np.array(wlist_cam), axis=0) if wlist_cam else np.array([0.0, 0.0, 0.6])

    def _aim(ren, t, target):
        ang = math.radians(orbit_deg * math.sin(2 * math.pi * t / max(T, 1)))
        eye = target + np.array([0.75 * math.sin(ang), -0.35, -0.75 * math.cos(ang)])
        ren.setup_camera(55.0, target.tolist(), eye.tolist(), [0.0, -1.0, 0.0])

    def _hand_verts_cam(t, side, base):
        st = states[t]
        orient = st[base + 3:base + 6].copy()
        pose = st[base + 6:base + 51].copy(); betas = st[base + 51:base + 61]
        if np.allclose(pose, 0) and np.allclose(betas, 0):
            return None
        if side == "left":
            orient = orient * _MIR; pose = (pose.reshape(15, 3) * _MIR).reshape(45)
        wrist = kps[t][0 if side == "left" else 1, 0]
        return _mano_vertices(models[side], wrist, orient, pose, betas)

    dpi = 100
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi, facecolor="#0d0d0d")
    ax = fig.add_axes([0.10, 0.12, 0.87, 0.78]); ax.set_facecolor("#0d0d0d")
    fig_ctx = (fig, ax)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_mp4 = out_dir / f"episode_{ep_idx:06d}_world.mp4"
    proc = subprocess.Popen(
        [str(ffb.ffmpeg), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{w*2}x{h*2}", "-r", f"{fps}", "-i", "-", "-c:v", "libx264", "-preset", "slow",
         "-crf", "16", "-pix_fmt", "yuv420p", str(out_mp4)], stdin=subprocess.PIPE)

    for t in range(T):
        kp_masked = kps[t].copy()
        for hd in (0, 1):
            if not _valid(t, hd):
                kp_masked[hd] = np.nan

        # --- RGB with the 2D skeleton + wrist axes drawn on it (the fit check) ---
        ov = Image.fromarray(rgb[t].copy()); od = ImageDraw.Draw(ov)
        _draw_hand(od, _project(kp_masked[0], fx, fy, cx, cy), _LEFT_COLOR)
        _draw_hand(od, _project(kp_masked[1], fx, fy, cx, cy), _RIGHT_COLOR)
        st = states[t]
        if st.shape[0] >= 67:
            if _valid(t, 0):
                _draw_wrist_axes(od, st[0:3], st[3:6], fx, fy, cx, cy)
            if _valid(t, 1):
                _draw_wrist_axes(od, st[61:64], st[64:67], fx, fy, cx, cy)
        raw = _label(np.asarray(ov), "RGB + skeleton fit")

        # --- 3D keypoint fit: solid spheres + cylinders ---
        ren.scene.clear_geometry()
        for name, mesh, hand in _kp_solid(o3d, kp_masked, _BONES):
            ren.scene.add_geometry(name, mesh, hand_mat[hand])
        _aim(ren, t, tgt_k)
        kp3d = _downscale(np.asarray(ren.render_to_image())[:, :, :3].astype(np.uint8), w, h)
        kp3d = _label(kp3d, "3D keypoint fit")

        # --- wrist trajectory curve ---
        traj = _label(_traj_panel(traj_l, traj_r, t, w, h, fig_ctx), "wrist trajectory")

        # --- 3D world: static fused mesh + per-frame MANO hands (ref frame) ---
        ren.scene.clear_geometry()
        if world_mesh is not None:
            ren.scene.add_geometry("world", world_mesh, world_mat)
        Tr = _T_ref(t)
        for side, base in (("left", 0), ("right", 61)):
            hand = 0 if side == "left" else 1
            if not _valid(t, hand):
                continue
            verts = _hand_verts_cam(t, side, base)
            if verts is None:
                continue
            verts_ref = (Tr[:3, :3] @ verts.T).T + Tr[:3, 3]
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(verts_ref.astype(np.float64))
            mesh.triangles = o3d.utility.Vector3iVector(faces[side])
            mesh.compute_vertex_normals()
            ren.scene.add_geometry(f"hand_{side}", mesh, hand_mat[hand])
        _aim(ren, t, tgt_w)
        world = _downscale(np.asarray(ren.render_to_image())[:, :, :3].astype(np.uint8), w, h)
        world = _label(world, "3D world (TSDF + MANO)")

        top = np.concatenate([raw, kp3d], axis=1)
        bot = np.concatenate([traj, world], axis=1)
        proc.stdin.write(np.concatenate([top, bot], axis=0).astype(np.uint8).tobytes())

    plt.close(fig)
    proc.stdin.close(); proc.wait()
    return out_mp4


__all__ = ["render_3d_video", "render_world_synced"]
