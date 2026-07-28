# 架构总览：人手示范 → 机器人策略 数据引擎

## 这个项目本质是做什么的

一句话：**把人在真实世界里徒手做操作的视频，自动加工成机器人（Wuji 五指灵巧手）可以直接拿去做模仿学习训练的数据集，并训出一个策略验证这条链路真的走通。**

为什么值得做：机器人操作数据极贵（要真机遥操作采集），而人手视频采集成本几乎为零。本项目就是把「人怎么做」自动翻译成「机器人该输出什么动作」的那根管子——一个廉价的机器人数据水龙头。

## 设计目标

将 ITW（In-The-Wild）人类示范会话原始数据，端到端转换为可直接用于 VLA/模仿学习训练的 LeRobot v3 标准数据集，并闭环到一个可训练、可 rollout 的策略。

## 分层叠加架构

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
       ▼  Layer 2.5 (retarget)        │
  人手 MANO → Wuji 机器人            │
  robot_qpos[46] / ee_pose[12]      │
       │                              │
       └──────────────┬───────────────┘
                      ▼
            Layer 3 (lerobot_v3)
            LeRobot v3 数据集
      （state[122] + robot_qpos/ee，
        tasks 来自 clip 语言，
        data 含 action_label/score 列）
                      │
                      ▼  Layer 4 (il)
            lerobot ACT policy
      （从 ego 图像预测单手 Wuji 动作 26-D，
        val≈0.04；rollout / analyze 可视化）

    （贯穿全线：multimodal_pipeline.viz —— 叠加/3D/看板可视化）
```

每层有**独立的输入输出契约**，可以独立替换实现：
- Layer 1 未来可加 USD/Omniverse 导出
- Layer 2 接入真实 GeoCalib / MoGe-2 / HaWoR / MegaSAM 模型权重时不影响上下游
- Layer 1.5 可换 VLM（Qwen-VL / GPT-4o / 本地模型），可替换 quality（OpenCV / 训练分类器）
- Layer 2.5 换机器人只需换 URDF（`MMPIPE_WUJI_URDF_DIR`），FK/IK 与上下游解耦
- Layer 3 输出格式可以从 LeRobot v3 切换到 RLDS / WebDataset / 任意其他格式
- Layer 4 策略可从 ACT 换成 Diffusion Policy / 任意 lerobot policy

**Layer 2 与 Layer 1.5 都直接消费 NIR，互相独立**。当 atomic_actions 来自 Layer 2 时，Layer 1.5 把它们当作 clip 边界打标（语义对齐）；若 Layer 2 没跑，Layer 1.5 退化为按固定窗口切 clip。

## 仓库目录结构

```
minipipline/
├── main.py                        # 入口薄壳 → multimodal_pipeline.cli:main
├── pyproject.toml                 # 包元数据 + mmpipe console_script
├── multimodal_pipeline/
│   ├── cli.py                     # argparse CLI（run-all / ingest / ... / visualize）
│   ├── __main__.py                # python -m multimodal_pipeline 入口
│   ├── config/                    # 代码即配置（settings 单例 + 每层 dataclass + models/）
│   ├── itw/                       # Layer 1：ITW → NIR（discover/validate/calib/align/normalize/pack）
│   ├── handpose/                  # Layer 2：NIR → MergedPrediction（real_ingest / mock 模型链）
│   ├── annotate/                  # Layer 1.5：NIR → 语言/动作标签 + 帧质量
│   ├── retarget/                  # Layer 2.5：人手 MANO → Wuji（纯 numpy FK+LM；assets/wuji/ 存 URDF）
│   ├── lerobot_v3/                # Layer 3：打 LeRobot v3 数据集（schema/writer/validate/actions）
│   ├── il/                        # Layer 4：lerobot ACT 训练（dataset/train/rollout/analyze/live_plot）
│   ├── viz/                       # 可视化子包（overlay/scene3d/mano/robot_overlay/dashboard/gallery）
│   ├── quality.py                 # 4 维数据质量报告（run-all 末尾自动跑）
│   └── pipeline.py/stages.py/records.py + config/legacy.py   # 遗留 sensor-ETL（隐藏命令 run/init-config）
├── docs/                          # 本文档 + MIGRATION + adr/
├── examples/configs/              # JSON 配置模板（--*-config 可选覆盖）
└── scripts/create_demo_data.py    # 造演示数据
```

> `.venv/`、`.idea/`、`output/`、`artifacts/`、`models/`（MANO 授权资产）、`.env`（真实凭据）、
> `*_synthetic.urdf`（占位臂）均已在 `.gitignore` 中，不入库。

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

**时序去抖（One-Euro，`handpose/smoothing.py`，默认开）**：源 tracker 逐帧独立出 MANO、无时序滤波(jerk/速度比 ~0.8，高频抖)。`smooth_merged` 用 1€ 滤波在**连续 kept 段内**去噪 `hand_keypoints_world` / `pred_trans` / `pred_rot` / `pred_hand_pose`——绝不跨 gap、绝不给非 kept 帧造值(守 no-fake-data)。向量场逐分量标量 1€；旋转场(手腕朝向 + 15 手指关节 axis-angle)用**四元数 1€ + SLERP** 球面低通(避开 axis-angle 的 2π wrap)。`pred_trans` 由滤后腕关键点派生保持一致，`pred_betas`(手形)不动。默认 `2.0/0.7`(00010a33 实测：pose jerk ×0.42、真实运动保留 ~2/3、Wuji 指尖拟合 6.59→6.00mm、限位 100%)。旋钮见 `config/handpose.py`；`smoothing_enabled=False` 关闭。自检：`python3 -m multimodal_pipeline.handpose.smoothing`。

**输入**：NIR 会话目录。
**输出**：`MergedPrediction`（视频级 MANO 参数 + 相机轨迹，已去抖）+ `list[AtomicAction]`（基于手腕 3D 速度极小值切分，跑在去抖后的 `pred_trans` 上）。

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

**说明**：Layer 3 落盘的是**人手** state/action（MANO 系）；机器人可执行的动作目标由 Layer 2.5 追加为 `robot_qpos` / `robot_ee_pose` 两列（见下）。Layer 4 训练消费的正是这两列。

## Layer 2.5 — Retarget (`multimodal_pipeline.retarget`)

（按数据流位于 Layer 2 与 Layer 3 之间：Layer 2 出人手 MANO → 本层重定向到机器人 → Layer 3 打包时把机器人列一并落盘。）

把每帧**人手 MANO** 重定向到 **Wuji 机器人**，给数据集补上机器人可执行的动作目标，是接进 IL 训练闭环的关键一环。纯 numpy 实现（URDF 正向运动学 + Levenberg–Marquardt），无 pinocchio/nlopt/scipy，惰性导入，遵循仓库自检风格。

- **手**：人手 21 关键点 → Wuji 灵巧手 **20 关节角**。先用关键点构造手腕局部系（去除全局位姿），再用 Kabsch 从整段平均手形**自标定** MANO↔Wuji 约定差与尺度（数据驱动，非硬编码），逐帧 vector 重定向匹配指尖（限位内、帧间平滑）。
- **臂**：每臂 3 关节（`*_arm_joint1/2`、`*_palm_joint`）的欠驱动 IK 拟合手腕 6-DOF — 等 `dual_arm.urdf`（`wh120_arm_mujoco` 分支）到位后落地；在此之前臂列为 **NaN（诚实“未计算”）**。
- **写入两列**（`schema.py`）：`observation.robot_qpos[46]`（左臂3+左手20+右臂3+右手20，全关节，可直接执行）+ `observation.robot_ee_pose[12]`（每手 transl3+orient_aa3，跨本体通用的末端位姿）。
- **No-fake-data**：某手无真实关键点（mock 链）或该帧未 `kept` → 对应列 **NaN 填充**，与 `observation.hand_keypoints` 一致。
- **MIT URDF** 随仓库分发于 `retarget/assets/wuji/`（FK 只需 URDF，不需 mesh）。`MMPIPE_WUJI_URDF_DIR` 可覆盖（指向含 dual-arm URDF 的目录即可重定向臂）。
- **自检**：`mmpipe retarget-check <nir_dir>` 报每手指尖误差（mm）、对齐残差（mm）、限位合规率；`python3 -m multimodal_pipeline.retarget.selftest` 跑 FK/雅可比/往返断言。已验证（`00010a33`）：右手指尖 ~10mm、左手 ~16mm、限位 100%。

## Layer 4 — IL 训练 (`multimodal_pipeline.il`)

闭环的最后一环：直接读 Layer 3 产出的 LeRobot v3 数据集，训一个 **lerobot ACT 策略**，用 ego 图像 + 当前手部状态预测未来的 Wuji 手动作，验证「人手数据 → 可训练策略」这条链真的通。

**样本定义**（`dataset.py`）：一个样本 = 一个 (kept 手 × 帧)。观测 = ego 图像 + 该手当前 **26-D** 配置 `[hand_qpos 20 + ee transl3 + ee orient_aa3]`（只取手 + 末端，排除占位臂）；目标 = 同一 26-D 配置的 ACT 式**未来 chunk**（默认 16 步，pad 掩码补齐）。含 NaN（未 kept / 占位臂）的手/帧直接丢弃——延续 no-fake-data。每 episode 解码一次 ego 视频，帧在 RAM 缓存、双手去重。

**训练**（`train.py`）：

| 组件 | 选择 | 说明 |
|------|------|------|
| policy | lerobot 0.4.4 `ACTPolicy` | resnet18 视觉 backbone，chunk=16 |
| VAE | **关闭**（`use_vae=False`） | 规避此 build 的 VAE-in-eval KL 崩溃，退化为纯 transformer BC |
| 归一化 | 自己在 collate 里做 | state/action 用数据集 mean-std，图像用 ImageNet；policy 的 `normalization_mapping` 设 IDENTITY（此 build 的 ACT 不内部归一、不吃 stats）。保存 policy state_dict + norm stats（rollout 反归一化要用）|
| 评估 | `_eval_per_dof` | 用 `predict_action_chunk[:,0]` 干净单步预测算逐关节 mean-abs 误差（避开 select_action 的 chunk 队列在乱序 batch 上把误差放大 10× 的坑）|

**监控可视化**：
- `il_train_curve.png` —— 训练中每 50 步覆盖重画的 loss 曲线（VS Code 图片预览自动刷新）+ 结束时逐 DoF 误差柱状图。
- **TensorBoard**（`--tb`，默认开）—— events 写到 `<out>/tb/`；VS Code `Python: Launch TensorBoard` 可在编辑器标签页内实时看 `loss/train`、`loss/val`（可缩放/平滑/悬停）。
- `analyze.py` —— 自解释分析图（`il_analysis.png`）：训练曲线 + 判词（收敛/过拟合/欠拟合）、误差 vs 数据量、误差分布、逐 DoF 难度，一眼回答「策略多好、瓶颈系于什么」。
- `live_plot.py` —— `plt.ion` 真窗口实时曲线（**需自己在有显示的终端跑**，detached shell 弹不出 GUI）。
- `rollout.py` —— 开环 rollout，渲染预测 vs GT 的 Wuji 手骨架 MP4。

**实测**：val loss ≈ 0.042，rollout MSE ≈ 0.0006，逐关节 mean-abs 误差 ≈ 0.018 rad。

**运行**（无 CLI 子命令，走模块）：
```bash
python3 -m multimodal_pipeline.il.train    --sessions 8 --epochs 8   # 训练（默认 artifacts/il）
python3 -m multimodal_pipeline.il.rollout                            # 预测 vs GT 骨架 MP4
python3 -m multimodal_pipeline.il.analyze  --sessions 4              # 自解释分析图
python3 -m multimodal_pipeline.il.live_plot                          # 自己终端里的实时曲线窗口
```

## 可视化 (`multimodal_pipeline.viz`)

横切所有层的可视化子包（`mmpipe visualize` 与 `il`/`retarget` 复用），供人工核查拟合质量：

| 模块 | 作用 |
|------|------|
| `core.py` | 关键点叠加 MP4（可选 depth / raw 对比 / 点云）——`mmpipe visualize` 默认路径 |
| `mano.py` | MANO 手网格叠加 PNG / 2×2 合成 MP4（`--mano` / `--all`） |
| `scene3d.py` | Open3D/EGL 3D 视频（`--3d`）+ 帧锁 3×2 全景 world MP4（`--world`） |
| `robot_overlay.py` | 每帧 Umeyama 相似变换把 Wuji 手投影**叠加到 RGB 真手上**（自检 = 指尖像素距离）|
| `retarget_fit.py` | 人手↔机器人指尖 3D 拟合对照 |
| `urdf_mesh.py` / `urdf_spin.py` / `synth_arm.py` | URDF 网格加载 / 旋转展示 / 占位臂可视 |
| `gallery.py` / `dashboard.py` | 扫 `artifacts/gallery/` 生成 `index.html` 看板，VS Code Live Preview 内打开看全部 mp4/png |

> **GUI 限制**：detached shell 弹不出持久 GUI 窗口。要交互看的（mujoco viewer、live_plot、TensorBoard 面板）都在**用户自己的终端/VS Code** 里开；我这边产出的是文件（MP4/PNG/HTML）。

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
mmpipe visualize <dataset_root> [--all|--world|--mano|--3d]   # 可视化核查

# Layer 4 IL 训练（无子命令，走模块）
python3 -m multimodal_pipeline.il.train --sessions 8 --epochs 8
python3 -m multimodal_pipeline.il.rollout
python3 -m multimodal_pipeline.il.analyze
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
