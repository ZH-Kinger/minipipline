"""IL training (Layer 4-ish): train a lerobot ACT policy to predict Wuji robot
hand actions from the captured dataset.

- ``dataset``  : PyTorch Dataset reading our LeRobot-v3 parquet/video directly
                 (one sample per kept hand×frame; action = single-hand
                 [hand_qpos 20 + ee 6] = 26-D, future chunk).
- ``train``    : configure + train a lerobot ACTPolicy on GPU.
- ``rollout``  : open-loop rollout + render predicted vs GT Wuji hand.
"""
