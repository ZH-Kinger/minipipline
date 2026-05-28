from __future__ import annotations

import numpy as np

from .schemas import CameraTrajectory, ClipPlan, PredResult, VideoMeta


def merge_clip_predictions(
    video: VideoMeta,
    clips: list[ClipPlan],
    predictions: list[PredResult],
    trajectories: list[CameraTrajectory],
) -> tuple[
    np.ndarray,  # pred_trans (2, N, 3)
    np.ndarray,  # pred_rot_aa (2, N, 3)
    np.ndarray,  # pred_hand_pose (2, N, 45)
    np.ndarray,  # pred_betas (2, N, 10)
    np.ndarray,  # pred_valid (2, N)
    CameraTrajectory,  # video-level trajectory
]:
    """Linear-blend per-clip predictions into a video-level sequence.

    Single-clip videos pass through directly. Multi-clip blends apply a linear
    ramp in the overlap region (cosine ramp would be smoother but is overkill
    for mock data; switch later if visible artefacts appear).
    """
    if len(clips) != len(predictions):
        raise ValueError("clips and predictions must have matching lengths")
    if len(clips) != len(trajectories):
        raise ValueError("clips and trajectories must have matching lengths")

    N = video.n_frames
    trans = np.zeros((2, N, 3), dtype=np.float32)
    rot = np.zeros((2, N, 3), dtype=np.float32)
    pose = np.zeros((2, N, 45), dtype=np.float32)
    betas = np.zeros((2, N, 10), dtype=np.float32)
    valid = np.zeros((2, N), dtype=bool)
    weights = np.zeros((1, N, 1), dtype=np.float32)

    cam_c2w = np.broadcast_to(np.eye(4, dtype=np.float32), (N, 4, 4)).copy()
    cam_weight = np.zeros(N, dtype=np.float32)
    K = trajectories[0].K
    slam_hw = trajectories[0].slam_hw

    for clip, pred, traj in zip(clips, predictions, trajectories, strict=True):
        f0 = clip.frame_start
        f1 = clip.frame_end
        T = f1 - f0
        # Linear ramp: rises over overlap_left at start, drops over overlap_right at end.
        ramp = np.ones(T, dtype=np.float32)
        if clip.overlap_left > 0:
            ramp_len = min(clip.overlap_left, T)
            ramp[:ramp_len] = np.linspace(0.0, 1.0, ramp_len + 1)[1:]
        if clip.overlap_right > 0:
            ramp_len = min(clip.overlap_right, T)
            ramp[T - ramp_len:] = np.linspace(1.0, 0.0, ramp_len + 1)[:-1]

        for slot in range(2):
            trans[slot, f0:f1] += pred.pred_trans[slot] * ramp[:, None]
            rot[slot, f0:f1] += pred.pred_rot[slot] * ramp[:, None]
            pose[slot, f0:f1] += pred.pred_hand_pose[slot] * ramp[:, None]
            betas[slot, f0:f1] += pred.pred_betas[slot] * ramp[:, None]
            valid[slot, f0:f1] |= pred.pred_valid[slot]
        weights[0, f0:f1, 0] += ramp

        cam_c2w[f0:f1] = traj.cam_c2w * ramp[:, None, None] + cam_c2w[f0:f1] * (1 - ramp[:, None, None])
        cam_weight[f0:f1] += ramp

    safe = np.where(weights > 1e-6, weights, 1.0)
    trans /= safe
    rot /= safe
    pose /= safe
    betas /= safe

    merged_traj = CameraTrajectory(
        video_id=video.video_id,
        clip_idx=-1,
        frame_start=0,
        cam_c2w=cam_c2w.astype(np.float32),
        K=K.astype(np.float32),
        slam_hw=slam_hw,
        valid=cam_weight > 0.0,
    )
    return trans, rot, pose, betas, valid, merged_traj


__all__ = ["merge_clip_predictions"]
