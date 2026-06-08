# 架构总览：通用多模态人类示范数据管线

## 设计目标

将 ITW（In-The-Wild）人类示范会话原始数据，端到端转换为可直接用于 VLA/模仿学习训练的 LeRobot v3 标准数据集。

## 4 层叠加架构

```
原始 ITW 会话目录
       │
       ▼  Layer 1 (multimodal_pipeline.itw)
   NIR + WebDataset tar
       │
       ├──────────────────────────────┐
       ▼                              ▼
  Layer 2 (handpose)            Layer 1.5 (annotate)
  MergedPrediction +            ClipAnnotation +
  AtomicActions                 FrameQualityRow
  （real_ingest | mock）         （mock | dashscope | rule_based）
       │                              │
       └──────────────┬───────────────┘
                      ▼
            Layer 3 (lerobot_v3)
            LeRobot v3 数据集
            （tasks 来自 clip 语言，
              data 含 action_label/score 列）
```

每层有**独立的输入输出契约**，可以独立替换实现：
- Layer 1 未来可加 USD/Omniverse 导出
- Layer 2 接入真实 GeoCalib / MoGe-2 / HaWoR / MegaSAM 模型权重时不影响上下游
- Layer 1.5 可换 VLM（Qwen-VL / GPT-4o / 本地模型），可替换 quality（OpenCV / 训练分类器）
- Layer 3 输出格式可以从 LeRobot v3 切换到 RLDS / WebDataset / 任意其他格式

**Layer 2 与 Layer 1.5 都直接消费 NIR，互相独立**。当 atomic_actions 来自 Layer 2 时，Layer 1.5 把它们当作 clip 边界打标（语义对齐）；若 Layer 2 没跑，Layer 1.5 退化为按固定窗口切 clip。

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

## Layer 1.5 — Annotation (`multimodal_pipeline.annotate`)

**输入**：NIR 会话目录 + 可选 `list[AtomicAction]`（来自 Layer 2）。

**输出**（写回 NIR 目录）：

| 文件 | 内容 |
|------|------|
| `clip_annotations.parquet` | 每个 atomic clip 一行：`clip_idx`, `frame_start/end`, `t_start/end_s`, `language_text`, `action_label`, `action_score`, `language_source`, `actions_source` |
| `frame_quality.parquet` | 每帧一行：`frame_idx`, `blur_score`, `exposure_score`, `hand_visible`, `overall_quality`, `kept` |

**3 个独立 component 与 backend**：

| component | 可选 backend | 说明 |
|-----------|------------|------|
| `language`（clip 描述） | `mock` / `dashscope` | mock 返回模板字符串；dashscope 走阿里云 Qwen-VL REST，stdlib `urllib` 实现，无 SDK 依赖 |
| `actions`（clip 类别） | `mock` / `dashscope` | mock 随机从 7 类（reach/grasp/lift/move/rotate/place/release）选；dashscope 用同样 VLM 0-shot 分类，结果归一化到该 vocab |
| `quality`（每帧质量） | `mock` / `rule_based` | mock 出确定性随机值；rule_based 用 ffmpeg 解码灰度帧 + 4-邻域 Laplacian 方差（无 cv2 依赖）算模糊，灰度均值算曝光 |

**模型与成本**（DashScope，参考阿里云公开价）：

| 模型 | 单价（每图） | 19 clip × 16 帧 单 session 全开 language+actions 成本 |
|------|------------|------------------------------------------------|
| `qwen-vl-plus` | ~¥0.008 | ~¥4.8 |
| `qwen-vl-max` | ~¥0.02  | ~¥12 |

**关键约定**：
- mock 与 dashscope 用同一 BLAKE2b seed scheme，所以连续两次跑 mock 的 `clip_annotations` 字段 byte-identical
- DashScope backend 在 `__post_init__` 里 eager-check `MMPIPE_DASHSCOPE_API_KEY`，fail-fast，不会做完一半才发现没 key
- action_score 在 dashscope backend 是固定 0.8（VLM 不返回真实概率），mock 是 `uniform(0.6, 0.95)`；下游使用方应避免把它当作精确置信度

## Layer 2 — Hand Pose Features (`multimodal_pipeline.handpose`)

**默认 backend 为 `real_ingest`**（`MMPIPE_HANDPOSE_BACKEND=real_ingest`，当前生产配置）。它**不跑也不 mock** 5 阶段模型链，而是直接从 Layer 1 已抽好的 NIR 真值组装 `MergedPrediction`：

- 真实 3D 手部关键点（相机系，米）→ 经每帧头部位姿 c2w 变换到世界系（`p_world = R(q) @ p_cam + t`）
- 真实头部 6DOF 轨迹 → 每帧 camera-to-world
- 真实 MANO 参数（pose / betas / global_orient，来自源 tracker）→ 直接用；缺失帧才退化为由关键点几何估计腕部朝向
- 真实相机内参（Kalibr）→ K

world→camera 往返应复现原始相机系坐标，作为内置正确性自检。实现见 `handpose/real_ingest.py`，编排器在 `MMPIPE_HANDPOSE_BACKEND=real_ingest` 时走 fast path 跳过模型链。

**输入**：NIR 会话目录。
**输出**：`MergedPrediction`（视频级 MANO 参数 + 相机轨迹）+ `list[AtomicAction]`（基于手腕 3D 速度极小值切分）。

> **可选 mock backend（默认不启用）**：设 `MMPIPE_HANDPOSE_BACKEND=mock` 时，5 个模型后端通过 BLAKE2b 派生种子产生**确定性**随机张量（shape 与真实模型一致），仅用于在没有真实模型权重时调通链路。生产数据**不使用** mock。下方 7 阶段流水描述的是 mock / 未来真实模型链的拓扑；`real_ingest` 直接绕过它。

**7 阶段流水**（mock / 未来真实模型链）：

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

## Layer 2.5 — Retarget (`multimodal_pipeline.retarget`)

把每帧**人手 MANO** 重定向到 **Wuji 机器人**，给数据集补上机器人可执行的动作目标，是接进 IL 训练闭环的关键一环。纯 numpy 实现（URDF 正向运动学 + Levenberg–Marquardt），无 pinocchio/nlopt/scipy，惰性导入，遵循仓库自检风格。

- **手**：人手 21 关键点 → Wuji 灵巧手 **20 关节角**。先用关键点构造手腕局部系（去除全局位姿），再用 Kabsch 从整段平均手形**自标定** MANO↔Wuji 约定差与尺度（数据驱动，非硬编码），逐帧 vector 重定向匹配指尖（限位内、帧间平滑）。
- **臂**：每臂 3 关节（`*_arm_joint1/2`、`*_palm_joint`）的欠驱动 IK 拟合手腕 6-DOF — 等 `dual_arm.urdf`（`wh120_arm_mujoco` 分支）到位后落地；在此之前臂列为 **NaN（诚实“未计算”）**。
- **写入两列**（`schema.py`）：`observation.robot_qpos[46]`（左臂3+左手20+右臂3+右手20，全关节，可直接执行）+ `observation.robot_ee_pose[12]`（每手 transl3+orient_aa3，跨本体通用的末端位姿）。
- **No-fake-data**：某手无真实关键点（mock 链）或该帧未 `kept` → 对应列 **NaN 填充**，与 `observation.hand_keypoints` 一致。
- **MIT URDF** 随仓库分发于 `retarget/assets/wuji/`（FK 只需 URDF，不需 mesh）。`MMPIPE_WUJI_URDF_DIR` 可覆盖（指向含 dual-arm URDF 的目录即可重定向臂）。
- **自检**：`mmpipe retarget-check <nir_dir>` 报每手指尖误差（mm）、对齐残差（mm）、限位合规率；`python3 -m multimodal_pipeline.retarget.selftest` 跑 FK/雅可比/往返断言。已验证（`00010a33`）：右手指尖 ~10mm、左手 ~16mm、限位 100%。

## 数据流契约速查

| 阶段间 | 数据形态 |
|--------|---------|
| ITW → NIR | parquet + npz + wav + json |
| NIR → Layer 2 | 通过 `media_paths.json` 找回源视频 |
| NIR → Layer 1.5 | 同上，外加可选 `list[AtomicAction]`（Layer 2 输出）作 clip 边界 |
| Layer 2 → Layer 3 | `MergedPrediction (numpy)` + `list[AtomicAction]` |
| Layer 2 → Layer 2.5 → Layer 3 | `merged.hand_keypoints_world` → `robot_qpos[46]` / `robot_ee_pose[12]`（NaN 当无真值） |
| Layer 1.5 → Layer 3 | `list[ClipAnnotation]` → 写入 `task` 字段（per-clip 语言）+ data parquet 的 `action_label` / `action_score` 列 |
| Layer 3 → 训练 | 标准 LeRobot v3 datasets API |

## Mock 确定性

Layer 2 和 Layer 1.5 的 mock backends 都用**同一套** BLAKE2b 派生 seed 方案：

- Layer 2 key: `(scheme_version, salt, video_sha1, stage, frame_idx, slot)`
- Layer 1.5 key: `(scheme_version, salt, video_sha1, stage, clip_idx | frame_idx)`

两者都保证：
- **线程无关**：并发执行不改变结果
- **重跑一致**：同一输入连续两次 `run-all` 产出 byte-identical 的 state 张量 + clip_annotations
- **版本可控**：升级 mock 时改 `mock_scheme_version`（默认 `"v1"`）即可使所有下游 cache 失效

DashScope 等真实 backend 当然**不确定**（受温度参数 + 服务端随机性影响）。

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
| 接入真实 HaWoR 等 5 个模型 | 替换 `handpose/models/*.py` 中的 `infer` 实现 + `build_*_backend` factory 加分支 | 拿到模型权重 + GPU 环境 |
| 接入本地 Qwen-VL（不走 API） | `annotate/language.py` + `annotate/actions.py` 加 `LocalQwenVlAnnotator`，pyproject 加 `transformers` extra | 想离线跑 / 数据敏感不发外网 |
| 加身体 3D 关键点 | 新增 `annotate/body_pose.py`（NLF / 4DHumans），orchestrator 串入，schema 加 `body_keypoints.parquet` | 需求落地 |
| 加物体 6DoF | 新增 `annotate/object_pose.py`（FoundationPose），同上 | 需求落地 |
| 多会话批处理 | 已支持：`mmpipe <parent_dir>` 自动检测 + 断点续跑 | — |
| USD/Omniverse 导出 | 新增 `multimodal_pipeline.usd` 子包，消费 NIR | 需求落地 |
| 真实 stats.json | `LeRobotConfig.enable_real_stats=True` | 真实数据接入 |
| Ray / K8s 编排 | 见 ADR-0001 退出准则 | 数据量 > 100GB 或团队 > 3 人 |

## 验证清单

`run-all` 在 `00010a33-...` 上执行后应满足：

- [x] `ingest_report.json` 含 ≥ 1 条 anomaly（文件错配 + frame gap）
- [x] NIR `frame_index.parquet` 行数 = 257
- [x] IMU 裁剪后样本数 ≈ 200 × 8.57 = 1714（实测 1722）
- [x] Audio 裁剪后时长 ≈ 8.567s（实测 412798 / 48000 = 8.60s）
- [x] NIR `clip_annotations.parquet` 行数 = `len(atomic_actions)`（实测 19）
- [x] NIR `frame_quality.parquet` 行数 = NIR 帧数（实测 257）
- [x] `clip_annotations` 中 `language_source` / `actions_source` 与当前 `.env` backend 设置一致
- [x] LeRobot v3 `info.json` 字段齐全（codebase_version=v3.0, robot_type, fps, splits, features）
- [x] LeRobot v3 data parquet 含 `action_label`（string）+ `action_score`（float32）列
- [x] LeRobot v3 `tasks.parquet` 是去重后的 per-clip 语言（不再只有 session-level）；实测 19 个 atomic clip → 6 ~ 17 个去重任务（取决于 backend 是 mock 还是真 VLM）
- [x] 每个 episode 的 parquet 行数 = 视频帧数（ffprobe 校验，±1 帧容忍）
- [x] `state` 向量 float32[122]，符合 STATE_LAYOUT
- [x] `lerobot-validate` 0 errors / 0 warnings

## CLI 参考

云端 Linux 推荐用 `mmpipe`（pip 安装后的 console_script）；本地 Windows + Python 3.14 上 console_script 有已知问题，改用 `python -m multimodal_pipeline ...` 即可。

```bash
# 端到端（4 层连跑，推荐）
mmpipe <session_dir>                          # 默认输出到 ./output/<basename>/
mmpipe <session_dir> <output_root>            # 显式输出根
mmpipe <parent_dir> <output_root>             # 批量：自动扫一级子 session，断点续跑

# 单层独跑
mmpipe ingest <session_dir>                   # 仅 Layer 1
mmpipe handpose <nir_dir>                     # 仅 Layer 2
mmpipe annotate <nir_dir>                     # 仅 Layer 1.5
mmpipe lerobot <nir_dir>                      # Layer 2 + 3

# 工具
mmpipe doctor                                 # 自检 ffmpeg / 依赖 / backend / 凭据
mmpipe info <dataset_root>                    # 摘要数据集（episodes/frames/sizes）
mmpipe validate <dataset_root>                # 读回校验数据集合规
mmpipe retarget-check <nir_dir>               # Layer 2.5：重定向到 Wuji 并报拟合质量（mm/%）
```

## 配置分层

```
multimodal_pipeline/config/                # 代码即配置（算法层）
├── settings.py                            # Settings 单例：读 .env，暴露 device/backend/weights/...
├── itw.py / handpose.py / annotate.py / lerobot.py    # 每层算法参数 dataclass（默认值入 git）
├── retarget.py                            # Layer 2.5 重定向参数（URDF/平滑/迭代/尺度）
├── legacy.py                              # 老版 PipelineConfig（sensor-ETL）
└── models/                                # 单模型算法超参 dataclass
    ├── geocalib.py / moge2.py / hawor_s1.py / megasam.py / hawor_s2.py
    ├── qwen_vl.py                         # Layer 1.5 语言+动作模型超参（prompt/温度/帧数）
    ├── quality.py                         # Layer 1.5 质量阈值（blur/exposure）
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

**Backend dispatch**（Layer 2）：`multimodal_pipeline.handpose.models.build_<name>_backend(cfg)` 读 `settings.backend("<name>", fallback_key="HANDPOSE_BACKEND")`：
- `mock`（默认）→ 返回 mock 实现
- `real` → 抛 `NotImplementedError`（含 env var 提示），接入真实模型时把分支补上即可

`MMPIPE_HANDPOSE_BACKEND` 是 5 个 slot 的全局默认；`MMPIPE_<SLOT>_BACKEND` 覆盖之。

**Backend dispatch**（Layer 1.5）：`multimodal_pipeline.annotate.{language,actions,quality}.build_*_backend(cfg)` 读 `settings.backend("<component>", fallback_key="ANNOTATOR_BACKEND")`：
- `language` 接受：`mock` | `dashscope`
- `actions`  接受：`mock` | `dashscope`
- `quality`  接受：`mock` | `rule_based`

`MMPIPE_ANNOTATOR_BACKEND` 是 3 个 component 的全局默认；`MMPIPE_<COMPONENT>_BACKEND` 覆盖之。语言/动作走 `dashscope` 时还需要 `MMPIPE_DASHSCOPE_API_KEY`，未填会在 backend `__post_init__` 抛 `DashScopeError`，不会做完 ffmpeg 抽帧才发现。
