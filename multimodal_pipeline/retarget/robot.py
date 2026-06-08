"""Wuji robot model wrapper: kinematics + MANO↔robot finger correspondence.

Wraps a :class:`KinematicTree` and adds the semantics retargeting needs:

* identifies the ``*_palm_link`` and the five ``*_fingerN_tip_link`` end frames,
* splits the DOF into **arm** joints (root→palm path) and **hand** joints
  (the five finger chains hanging off the palm),
* exposes fingertip positions/Jacobians **in the palm-local frame** (so hand
  retargeting is invariant to where the arm has placed the palm), and the palm
  pose/position-Jacobian in the root frame (for arm IK),
* carries the MANO↔Wuji finger map and a human↔robot size ``scale``.

Hand-only URDFs (root = palm, zero arm DOF) and arm+hand URDFs both work: the
arm-joint set is simply empty in the hand-only case.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .urdf_fk import KinematicTree

# MANO 21-keypoint layout (wrist + 5 fingers × 4, tip last): tip indices and wrist.
MANO_WRIST = 0
MANO_FINGERTIPS = (4, 8, 12, 16, 20)  # thumb, index, middle, ring, pinky tips
# Default Wuji finger N (1..5) → MANO finger order (thumb..pinky), i.e. tip k.
DEFAULT_FINGER_MAP = (0, 1, 2, 3, 4)


def _assets_dir() -> Path:
    return Path(__file__).resolve().parent / "assets" / "wuji"


def resolve_urdf(side_or_path: str) -> Path:
    """Resolve a URDF path. Accepts an explicit path, or ``"left"``/``"right"``
    (vendored hand URDF), honouring ``$MMPIPE_WUJI_URDF_DIR`` as an override dir."""
    p = Path(side_or_path).expanduser()
    if p.suffix == ".urdf" and p.is_file():
        return p
    base = os.environ.get("MMPIPE_WUJI_URDF_DIR", "").strip()
    base_dir = Path(base).expanduser() if base else _assets_dir()
    cand = base_dir / f"{side_or_path}.urdf"
    if not cand.is_file():
        raise FileNotFoundError(
            f"Wuji URDF not found: {cand}. Pass an explicit .urdf path, or set "
            f"MMPIPE_WUJI_URDF_DIR. (Arm URDFs like dual_arm.urdf must be supplied "
            f"from the wh120_arm_mujoco branch — see the plan.)"
        )
    return cand


class RobotModel:
    """A loaded Wuji hand (or arm+hand) with retargeting-oriented kinematics."""

    def __init__(self, urdf: str, finger_map: tuple[int, ...] = DEFAULT_FINGER_MAP):
        self.urdf_path = resolve_urdf(urdf)
        self.tree = KinematicTree(self.urdf_path)
        self.finger_map = tuple(finger_map)

        # Palm link: the single link named "*_palm_link" (root for hand-only).
        palms = [l for l in self.tree.link_names if l.endswith("palm_link")]
        if len(palms) != 1:
            raise ValueError(f"{self.urdf_path.name}: expected one *_palm_link, got {palms}")
        self.palm_link = palms[0]

        # Five fingertip links, ordered finger1..finger5 by name.
        self.tip_links = sorted(l for l in self.tree.link_names if l.endswith("_tip_link"))
        if len(self.tip_links) != 5:
            raise ValueError(f"{self.urdf_path.name}: expected 5 *_tip_link, got {self.tip_links}")

        # Arm joints = revolute joints on the root→palm path; hand joints = rest.
        arm = self.tree._path_revolute[self.palm_link]
        self.arm_joint_names = list(arm)
        arm_set = set(arm)
        self.hand_joint_names = [n for n in self.tree.joint_names if n not in arm_set]
        self.arm_idx = np.array([self.tree.qpos_index[n] for n in self.arm_joint_names], dtype=int)
        self.hand_idx = np.array([self.tree.qpos_index[n] for n in self.hand_joint_names], dtype=int)
        # Which hand-joint columns drive each fingertip (relative to hand_idx order).
        self._tip_hand_cols = []
        hand_pos = {n: i for i, n in enumerate(self.hand_joint_names)}
        for tip in self.tip_links:
            cols = [hand_pos[n] for n in self.tree._path_revolute[tip] if n in hand_pos]
            self._tip_hand_cols.append(cols)

        # Robot canonical fingertip span (palm-local, neutral pose) → size scale.
        self._robot_span = self._fingertip_span(self.tree.midpoint())

    @property
    def dof(self) -> int:
        return self.tree.dof

    @property
    def n_arm(self) -> int:
        return len(self.arm_joint_names)

    @property
    def n_hand(self) -> int:
        return len(self.hand_joint_names)

    @property
    def hand_limits(self) -> np.ndarray:
        return self.tree.limits[self.hand_idx]

    @property
    def arm_limits(self) -> np.ndarray:
        return self.tree.limits[self.arm_idx] if self.n_arm else np.zeros((0, 2))

    def fingertips_local(self, qpos: np.ndarray) -> np.ndarray:
        """Fingertip positions (5,3) in the palm-local frame."""
        frames, _ = self.tree.fk(qpos)
        return self._tips_from_frames(frames)

    def _tips_from_frames(self, frames: dict[str, np.ndarray]) -> np.ndarray:
        palm = frames[self.palm_link]
        R, t = palm[:3, :3], palm[:3, 3]
        out = np.empty((5, 3), dtype=np.float64)
        for i, tip in enumerate(self.tip_links):
            out[i] = R.T @ (frames[tip][:3, 3] - t)
        return out

    def fingertips_local_with_jac(self, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Palm-local fingertips (5,3) and Jacobian wrt **hand** joints (5,3,n_hand).

        The arm is held fixed while solving the hand, so the palm frame is
        constant: ``∂local/∂q_hand = R_palmᵀ · ∂p_root/∂q_hand``.
        """
        frames, jw = self.tree.fk(qpos)
        palm = frames[self.palm_link]
        Rt = palm[:3, :3].T
        tips = self._tips_from_frames(frames)
        jac = np.zeros((5, 3, self.n_hand), dtype=np.float64)
        for i, tip in enumerate(self.tip_links):
            J_root = self.tree.position_jacobian(tip, frames, jw)  # (3, dof)
            jac[i] = Rt @ J_root[:, self.hand_idx]
        return tips, jac

    def palm_pose(self, qpos: np.ndarray) -> np.ndarray:
        """4×4 palm pose in the root (arm-base) frame."""
        return self.tree.fk(qpos)[0][self.palm_link]

    def palm_position_jac(self, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Palm 4×4 pose and palm-origin position Jacobian wrt **arm** joints (3,n_arm)."""
        frames, jw = self.tree.fk(qpos)
        palm = frames[self.palm_link]
        if self.n_arm == 0:
            return palm, np.zeros((3, 0))
        J_root = self.tree.position_jacobian(self.palm_link, frames, jw)
        return palm, J_root[:, self.arm_idx]

    def _fingertip_span(self, qpos: np.ndarray) -> float:
        """Mean palm-local fingertip distance from the palm origin (a size proxy)."""
        tips = self.fingertips_local(qpos)
        return float(np.mean(np.linalg.norm(tips, axis=1)))

    def size_scale(self, human_fingertip_vectors: np.ndarray) -> float:
        """robot_span / human_span, from human tip-relative-to-wrist vectors (...,5,3)."""
        hv = np.asarray(human_fingertip_vectors, np.float64).reshape(-1, 5, 3)
        human_span = float(np.mean(np.linalg.norm(hv, axis=2)))
        if human_span < 1e-6:
            return 1.0
        return self._robot_span / human_span


__all__ = ["RobotModel", "resolve_urdf", "MANO_WRIST", "MANO_FINGERTIPS", "DEFAULT_FINGER_MAP"]
