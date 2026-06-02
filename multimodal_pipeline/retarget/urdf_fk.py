"""Minimal URDF forward kinematics for rigid kinematic trees (numpy-only).

Handles exactly the URDF subset the Wuji models use — ``fixed`` + ``revolute``
joints with ``origin`` (xyz + rpy), ``axis`` and ``limit`` — and ignores
visual / collision / inertial / mesh entirely (FK never touches geometry). The
tree is general: it works for the hand-only URDF (root ``*_palm_link``, five
finger chains) and for the arm+hand URDF (root = arm base, ``*_palm_link`` mid-
chain, fingers branching off it), so the same code serves both without changes.

No external robotics dependency (no pinocchio / urdfpy); just ``xml.etree`` +
numpy, matching the package's minimal-deps, lazy-import ethos.

Conventions follow the URDF spec: a joint's ``origin`` is the fixed parent→child
transform applied *before* the joint's own motion, and ``rpy`` is a fixed-axis
roll-pitch-yaw, i.e. ``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)``.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    """URDF fixed-axis roll-pitch-yaw → 3×3 rotation (R = Rz @ Ry @ Rx)."""
    r, p, y = float(rpy[0]), float(rpy[1]), float(rpy[2])
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return Rz @ Ry @ Rx


def _axis_angle_matrix(axis: np.ndarray, theta: float) -> np.ndarray:
    """Rotation about a unit ``axis`` by ``theta`` (Rodrigues)."""
    a = axis / (np.linalg.norm(axis) + 1e-12)
    x, y, z = a
    c, s, C = np.cos(theta), np.sin(theta), 1.0 - np.cos(theta)
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ], dtype=np.float64)


def _homogeneous(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = R
    M[:3, 3] = t
    return M


@dataclass
class Joint:
    name: str
    jtype: str            # "revolute" | "fixed" (others treated as fixed)
    parent: str
    child: str
    origin_T: np.ndarray  # 4×4 fixed parent→child transform (before joint motion)
    axis: np.ndarray      # (3,) unit local rotation axis (revolute only)
    lower: float
    upper: float


class KinematicTree:
    """Forward kinematics + position Jacobians over a URDF rigid tree.

    ``qpos`` is ordered by ``joint_names`` (revolute joints in URDF file order);
    ``limits`` is the matching ``(N, 2)`` lower/upper array.
    """

    def __init__(self, urdf_path: str | Path):
        self.path = Path(urdf_path)
        root_el = ET.parse(self.path).getroot()
        self.joints: list[Joint] = [self._parse_joint(j) for j in root_el.findall("joint")]
        self.link_names: list[str] = [l.get("name") for l in root_el.findall("link")]

        self._children: dict[str, list[Joint]] = {}
        self._joint_by_child: dict[str, Joint] = {}
        for j in self.joints:
            self._children.setdefault(j.parent, []).append(j)
            self._joint_by_child[j.child] = j

        children_links = set(self._joint_by_child)
        roots = [l for l in self.link_names if l not in children_links]
        if len(roots) != 1:
            raise ValueError(f"{self.path.name}: expected one root link, got {roots}")
        self.root_link = roots[0]

        self.joint_names: list[str] = [j.name for j in self.joints if j.jtype == "revolute"]
        self.qpos_index: dict[str, int] = {n: i for i, n in enumerate(self.joint_names)}
        self.dof = len(self.joint_names)
        limits = [(j.lower, j.upper) for j in self.joints if j.jtype == "revolute"]
        self.limits = np.array(limits, dtype=np.float64) if limits else np.zeros((0, 2))

        # For each link, the ordered revolute joint names on the root→link path
        # (these are the only joints whose motion moves that link).
        self._path_revolute: dict[str, list[str]] = {}
        for link in self.link_names:
            chain: list[str] = []
            cur = link
            while cur in self._joint_by_child:
                j = self._joint_by_child[cur]
                if j.jtype == "revolute":
                    chain.append(j.name)
                cur = j.parent
            self._path_revolute[link] = list(reversed(chain))

    @staticmethod
    def _parse_joint(j: ET.Element) -> Joint:
        o = j.find("origin")
        xyz = np.fromstring(o.get("xyz", "0 0 0"), sep=" ") if o is not None else np.zeros(3)
        rpy = np.fromstring(o.get("rpy", "0 0 0"), sep=" ") if o is not None else np.zeros(3)
        origin_T = _homogeneous(_rpy_to_matrix(rpy), xyz)
        ax = j.find("axis")
        axis = np.fromstring(ax.get("xyz", "0 0 1"), sep=" ") if ax is not None else np.array([0, 0, 1.0])
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        lim = j.find("limit")
        lo = float(lim.get("lower", "0")) if lim is not None else 0.0
        hi = float(lim.get("upper", "0")) if lim is not None else 0.0
        return Joint(
            name=j.get("name", ""), jtype=j.get("type", "fixed"),
            parent=j.find("parent").get("link"), child=j.find("child").get("link"),
            origin_T=origin_T, axis=axis, lower=lo, upper=hi,
        )

    def fk(self, qpos: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, tuple[np.ndarray, np.ndarray]]]:
        """Compute every link's 4×4 pose in the root frame.

        Returns ``(frames, joint_world)`` where ``frames[link]`` is a 4×4 pose
        and ``joint_world[name] = (axis_world, origin_world)`` is the revolute
        joint's world axis and application point (used to build Jacobians).
        """
        qpos = np.asarray(qpos, np.float64)
        frames: dict[str, np.ndarray] = {self.root_link: np.eye(4, dtype=np.float64)}
        joint_world: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        stack = [self.root_link]
        while stack:
            parent = stack.pop()
            T_parent = frames[parent]
            for j in self._children.get(parent, []):
                T_pre = T_parent @ j.origin_T
                if j.jtype == "revolute":
                    origin_w = T_pre[:3, 3].copy()
                    axis_w = T_pre[:3, :3] @ j.axis
                    joint_world[j.name] = (axis_w, origin_w)
                    theta = float(qpos[self.qpos_index[j.name]])
                    T_child = T_pre @ _homogeneous(_axis_angle_matrix(j.axis, theta), np.zeros(3))
                else:
                    T_child = T_pre
                frames[j.child] = T_child
                stack.append(j.child)
        return frames, joint_world

    def link_pose(self, qpos: np.ndarray, link: str) -> np.ndarray:
        return self.fk(qpos)[0][link]

    def position_jacobian(
        self, link: str, frames: dict[str, np.ndarray],
        joint_world: dict[str, tuple[np.ndarray, np.ndarray]],
    ) -> np.ndarray:
        """Analytic ∂position/∂qpos for ``link`` → (3, dof).

        Revolute column ``k``: ``axis_world × (p_link − origin_world)`` for joints
        on the root→link path; zero elsewhere.
        """
        J = np.zeros((3, self.dof), dtype=np.float64)
        p = frames[link][:3, 3]
        for name in self._path_revolute[link]:
            axis_w, origin_w = joint_world[name]
            J[:, self.qpos_index[name]] = np.cross(axis_w, p - origin_w)
        return J

    def clamp(self, qpos: np.ndarray) -> np.ndarray:
        if self.limits.shape[0] == 0:
            return qpos
        return np.clip(qpos, self.limits[:, 0], self.limits[:, 1])

    def midpoint(self) -> np.ndarray:
        """Joint-limit midpoints — a neutral default / warm-start pose."""
        if self.limits.shape[0] == 0:
            return np.zeros(self.dof)
        return 0.5 * (self.limits[:, 0] + self.limits[:, 1])


__all__ = ["KinematicTree", "Joint"]
