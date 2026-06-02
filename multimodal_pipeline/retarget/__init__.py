"""Layer 2.5 — retarget human MANO hands onto the Wuji robot (arms + dexterous hands).

Pure-numpy URDF forward kinematics + a small Levenberg–Marquardt solver map each
captured human hand (MANO 21 keypoints + wrist 6-DOF) to a robot-executable
action: per-joint angles (`robot_qpos`) and an embodiment-agnostic end-effector
pose (`robot_ee_pose`). No pinocchio / nlopt / scipy — matches the package's
minimal-deps, lazy-import, self-check ethos.

Submodules are imported lazily (numpy is the only hard dep) so importing this
package never pulls heavy optional machinery.
"""

from __future__ import annotations

__all__ = ["KinematicTree", "RobotModel", "retarget_episode", "self_check"]


def __getattr__(name: str):  # lazy re-export
    if name == "KinematicTree":
        from .urdf_fk import KinematicTree
        return KinematicTree
    if name == "RobotModel":
        from .robot import RobotModel
        return RobotModel
    if name in ("retarget_episode", "self_check"):
        from . import core
        return getattr(core, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
