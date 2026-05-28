# 架构总览：通用多模态人类示范数据管线

## 设计目标

将 ITW（In-The-Wild）人类示范会话原始数据，端到端转换为可直接用于 VLA/模仿学习训练的 LeRobot v3 标准数据集。

## 三层叠加架构

```
原始 ITW 会话目录
       │
       ▼  Layer 1 (multimodal_pipeline.itw)
   NIR + WebDataset tar
       │
       ▼  Layer 2 (multimodal_pipeline.handpose, 当前 mock)
   MergedPrediction + AtomicActions
       │
       ▼  Layer 3 (multimodal_pipeline.lerobot_v3)
   LeRobot v3 数据集
```

每层有**独立的输入输出契约**，因此可以独立替换实现：
- Layer 1 未来可加 USD/Omniverse 导出
- Layer 2 接入真实的 GeoCalib / MoGe-2 / HaWoR / MegaSAM 模型权重时不影响上下游
- Layer 3 输出格式可以从 LeRobot v3 切换到 RLDS / WebDataset / 任意其他格式

## Layer 1 — ITW Ingest (`multimodal_pipeline.itw`)

**输入**：会话目录，自描述清单 `config.json` + 任务元数据 `task_info.json` + 多模态原始文件。

**输出（NIR — Normalized Intermediate Representation）**：

| 文件 | 内容 |
|------|------|
| `calibration.json` | 相机内参、T_depth_cam、T_cam_imu、IMU 噪声参数 |
| `task.json` | 任务文本 + 演员信息 + 标注摘要 |
| `frame_index.parquet` | 257 行；每帧的 RGB/depth 时间戳、有效性掩码、异常标记 |
| `head_pose.parquet` | 每帧 7-DoF 头部位姿 (tx,ty,tz,qx,qy,qz,qw) |
| `hand_keypoints.parquet` | 每帧 × 2 手 × 21 关键点 (3D, 相机系) |
| `imu_cropped.npz` | 按 RGB 时间窗裁剪后的 IMU 序列 |
| `audio_cropped.wav` | 按 RGB 时间窗裁剪后的双声道 48kHz 音频 |
| `media_paths.json` | 指向源 RGB/Depth 视频的引用 |
| `ingest_report.json` | 全流程摘要（异常、对齐、校准） |

**6 阶段流水**：discover → validate → calibration → align → annotate → normalize → pack。

**关键处理**：
1. **自描述清单交叉校验**：识别 `config.json` 声明 `head_hands_sixdof.csv` 但磁盘实际是 `head_hands_sixdof2.csv` 等错配
2. **时间对齐**：以 `rgb_head.csv` 为参考时钟，裁剪 IMU/audio 到视频时长
3. **异常标记**：自动检测 frame gap（默认 > nominal 1.6 倍 → anomaly）
4. **手部有效性**：处理 tail-excluded 帧、前段无关键点段
5. **校准跨校验**：kalibr ↔ head_param.json 的 K 矩阵一致性

## Layer 2 — Hand Pose Features (`multimodal_pipeline.handpose`)

**当前为 mock**。所有 5 个模型后端都通过 BLAKE2b 派生的种子产生**确定性**的随机张量，shape 与真实模型一致。

**输入**：NIR 会话目录。
**输出**：`MergedPrediction`（视频级 MANO 参数 + 相机轨迹）+ `list[AtomicAction]`（基于手腕 3D 速度极小值切分）。

**7 阶段流水**：

```
4.0 video probe (ffprobe)
4.1 GeoCalib         ─┐
                      │
4.2 MoGe-2 ──────────┼─► 4.4 MegaSAM ─► 4.5 HaWoR S2 ─► merge → clean → action_seg
                      │           ▲
4.3 HaWoR S1 ────────┴───────────┘
```

调度：`ThreadPoolExecutor`，4.2 / 4.3 自动并发，4.4 在两者完成后启动，4.5 在 4.4 完成后启动。可通过 `executor_mode="sequential"` 关闭并发。

**接入真实模型时**：保留每个 `*Backend.infer(...)` 的输入输出契约，仅替换内部实现。`schemas.py` 中的 dataclass 是层间数据契约。

## Layer 3 — LeRobot v3 Pack (`multimodal_pipeline.lerobot_v3`)

**输入**：Layer 2 的 `HandPoseRunResult` + 源视频路径。
**输出**：完整的 LeRobot v3 标准数据集。

**数据集结构**：

```
<dataset_root>/
├── meta/
│   ├── info.json                  # codebase_version, fps, features, splits
│   ├── stats.json                 # observation.state + action 统计（默认占位）
│   ├── tasks.parquet              # 去重后的任务文本
│   └── episodes/chunk-000/file-000.parquet
├── data/
│   └── chunk-000/file-000.parquet # 逐帧字段表
└── videos/observation.images.ego/chunk-000/
    ├── episode_000000.mp4
    └── ...
```

**state 向量**（每帧 122 维）= 左手 61 + 右手 61，每手 = 3 transl + 3 axis-angle orient + 45 MANO pose + 10 betas（相机系）。布局常量见 `lerobot_v3/schema.py` 的 `STATE_LAYOUT`。

**action 向量**（每帧 102 维）= 左手 51 + 右手 51，由 dataloader **即时计算**，不落盘。布局常量见 `ACTION_LAYOUT`。

## 数据流契约速查

| 阶段间 | 数据形态 |
|--------|---------|
| ITW → NIR | parquet + npz + wav + json |
| NIR → Layer 2 | 通过 `media_paths.json` 找回源视频 |
| Layer 2 → Layer 3 | `MergedPrediction (numpy)` + `list[AtomicAction]` |
| Layer 3 → 训练 | 标准 LeRobot v3 datasets API |

## Mock 确定性

所有 Layer 2 mock 输出由 `BLAKE2b((scheme_version, salt, video_sha1, stage, frame_idx, slot))` 派生种子，保证：
- **线程无关**：并发执行不改变结果
- **重跑一致**：同一输入连续两次 `run-all` 产出 byte-identical 的 state 张量
- **版本可控**：升级 mock 时改 `mock_scheme_version`（默认 `"v1"`）即可使所有下游 cache 失效

## 真实数据契约（基于 `00010a33-...` 验证）

| 项 | 实际值 |
|----|--------|
| RGB / Depth 帧数 | 257 / 257 |
| FPS | 30 (实测 29.88，frame 30→31 有 66.67ms gap) |
| 时长 | 8.567s（RGB），10.65s（IMU/audio） |
| Depth 偏移 | mean 2.83ms / max 5.23ms |
| 手部检测 | 164 帧含手（任一），137 帧双手 |
| Tail-excluded | 61 帧 |
| 文件错配 | `head_hands_sixdof.csv` 声明 vs `_sixdof2.csv` 实际 |

## 扩展路径

| 扩展 | 修改点 | 触发条件 |
|------|--------|---------|
| 接入真实 HaWoR 等模型 | 替换 `handpose/models/*.py` 中的 `infer` 实现 | 拿到模型权重 + GPU 环境 |
| 多会话批处理 | CLI 加 `--batch-root`，循环遍历 | 数据量 > 1 个会话 |
| USD/Omniverse 导出 | 新增 `multimodal_pipeline.usd` 子包，消费 NIR | 需求落地 |
| 真实 stats.json | `LeRobotConfig.enable_real_stats=True` | 真实数据接入 |
| Ray / K8s 编排 | 见 ADR-0001 退出准则 | 数据量 > 100GB 或团队 > 3 人 |

## 验证清单

`run-all` 在 `00010a33-...` 上执行后应满足：

- [x] `ingest_report.json` 含 ≥ 1 条 anomaly（文件错配 + frame gap）
- [x] NIR `frame_index.parquet` 行数 = 257
- [x] IMU 裁剪后样本数 ≈ 200 × 8.57 = 1714（实测 1722）
- [x] Audio 裁剪后时长 ≈ 8.567s（实测 412798 / 48000 = 8.60s）
- [x] LeRobot v3 `info.json` 字段齐全（codebase_version=v3.0, robot_type, fps, splits, features）
- [x] 每个 episode 的 parquet 行数 = 视频帧数（ffprobe 校验，±1 帧容忍）
- [x] `state` 向量 float32[122]，符合 STATE_LAYOUT
- [x] `lerobot-validate` 0 errors / 0 warnings

## CLI 参考

```powershell
# 三层连跑（推荐）
python -m multimodal_pipeline run-all <session_dir> <output_root>

# 单层独跑
python -m multimodal_pipeline ingest <session_dir> <output_dir>
python -m multimodal_pipeline handpose <nir_dir>
python -m multimodal_pipeline lerobot <nir_dir> <dataset_root>
python -m multimodal_pipeline lerobot-validate <dataset_root>
```

## 配置分层

```
multimodal_pipeline/config/                # 代码即配置（算法层）
├── settings.py                            # Settings 单例：读 .env，暴露 device/backend/weights/...
├── itw.py / handpose.py / lerobot.py      # 每层算法参数 dataclass（默认值入 git）
├── legacy.py                              # 老版 PipelineConfig（sensor-ETL）
└── models/                                # 单模型算法超参 dataclass
    ├── geocalib.py / moge2.py / hawor_s1.py / megasam.py / hawor_s2.py
    └── _base.py                           # 共享 from_file/from_mapping mixin

examples/configs/                          # JSON 模板（可选 --*-config override）
├── itw_default.json / handpose_default.json / lerobot_default.json
└── models/<name>.json

.env.example                               # 环境层模板（拷贝为 .env 后修改）
.env                                       # 真实环境值（.gitignore）
```

**职责切分**

| 类型 | 落点 | 例子 |
|------|------|------|
| 算法/编排不可变参数 | `config/<layer>.py` dataclass 默认值 | `STATE_DIM=122`, `clip_len_s=30`, `gap_anomaly_factor=1.6` |
| 模型算法私有超参 | `config/models/<name>.py` dataclass 默认值 | `HaWoRStage1Hyper.max_seq_len=1024`, `MoGe2Hyper.fp16=True` |
| 设备 / 权重路径 / backend 切换 / 凭据 | `.env`（`MMPIPE_*`） | `MMPIPE_DEVICE=cuda:0`, `MMPIPE_HAWOR_S1_BACKEND=real`, `MMPIPE_HAWOR_S1_WEIGHTS=/models/hawor_s1.pt` |
| 输入输出路径（每次跑变） | CLI 参数 | `session_dir`, `output_root` |
| 上述算法层的 ad-hoc 覆盖 | `examples/configs/*.json` + `--*-config` 可选传入 | 比如想试改 `clip_len_s` 而不动代码 |

**Backend dispatch**：`multimodal_pipeline.handpose.models.build_<name>_backend(cfg)` 读 `settings.backend("<name>")`：
- `mock`（默认）→ 返回 mock 实现
- `real` → 抛 `NotImplementedError`（含 env var 提示），接入真实模型时把分支补上即可

`MMPIPE_HANDPOSE_BACKEND` 是 5 个 slot 的全局默认；`MMPIPE_<SLOT>_BACKEND` 覆盖之。
