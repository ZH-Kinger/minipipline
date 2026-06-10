"""Visualization subpackage.

Library helpers:
  - ``core``     dataset keypoint/depth overlays + shared decode/project helpers
  - ``mano``     MANO hand-mesh overlay (needs torch+smplx)
  - ``scene3d``  Open3D 3D scene / world renders

Render entry-points (runnable as modules, e.g.
``python -m multimodal_pipeline.viz.urdf_spin <mjcf>``):
  - ``retarget_fit``   human vs Wuji fingertip fit (palm-local point cloud)
  - ``robot_overlay``  Wuji hand FK projected onto the RGB video
  - ``urdf_mesh``      static multi-view of a URDF's visual meshes
  - ``urdf_spin``      headless MuJoCo orbit + open/close MP4
  - ``synth_arm``      synthetic arm+hand spin (hand mesh + arm segments)
  - ``gallery``        build the verification gallery
"""

from .core import visualize_dataset  # noqa: F401
from .mano import render_combined, render_mano_overlay  # noqa: F401

__all__ = ["visualize_dataset", "render_mano_overlay", "render_combined"]
