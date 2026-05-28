# Multimodal Embodied-AI Data Pipeline

通用多模态人类示范数据管线。三层叠加架构：

1. **Layer 1 (ITW Ingest)** — 从原始会话目录摄取多模态文件，时间对齐、坐标统一、有效性标记，输出 NIR + WebDataset tar
2. **Layer 2 (Hand Pose Features, 当前 mock)** — 7 阶段模型链（GeoCalib + MoGe-2 + HaWoR + MegaSAM）输出手部 MANO 参数与相机轨迹
3. **Layer 3 (LeRobot v3 Pack)** — 装配为 LeRobot v3 标准数据集，附 dataloader 读回校验

依赖：`numpy>=1.26`、`pyarrow>=15.0`、`pyyaml>=6.0` + 系统 ffmpeg。

完整架构与数据契约见 [docs/architecture.md](docs/architecture.md)。

## 配置约定

参数按"会不会随环境而变"分两层：

| 位置 | 内容 | 是否入 git |
|------|------|----------|
| `multimodal_pipeline/config/{itw,handpose,lerobot}.py` | 算法/编排参数的 dataclass 默认值（如 `nominal_fps`、`clip_len_s`、`STATE_DIM`） | 是 |
| `multimodal_pipeline/config/models/<name>.py` | 单模型算法私有超参（variant / batch_size / mock 行为） | 是 |
| `examples/configs/*.json` + `examples/configs/models/*.json` | 上述 dataclass 的 JSON 模板，**仅作示例**，可选传给 `--*-config <path>` 覆盖 | 是 |
| `.env`（拷贝自 `.env.example`） | 真实环境参数：`MMPIPE_DEVICE` / `MMPIPE_<MODEL>_BACKEND` / `MMPIPE_<MODEL>_WEIGHTS` / `MMPIPE_FFMPEG_PATH` / 凭据等 | **否**（`.gitignore`） |
| CLI 参数 | 每次跑变的输入输出路径 | — |

切换 mock → real：在 `.env` 中设 `MMPIPE_<MODEL>_BACKEND=real` + `MMPIPE_<MODEL>_WEIGHTS=/path/to/weights`，real 实现未接入时会抛 `NotImplementedError` 含清晰提示。

## 快速开始

```bash
cp .env.example .env       # 按需修改 device / 权重路径 / MMPIPE_OUTPUT_ROOT
pip install -e .
mmpipe doctor              # 自检 ffmpeg / 依赖 / backend / 权重
mmpipe <session_dir>       # 跑全流程，输出到 ./output/<session_basename>/
```

> 本地 Windows + Python 3.14 上 `mmpipe.exe` console_script 有已知问题；改用 `python -m multimodal_pipeline ...` 即可。云端 Linux 部署不受影响。

### 三种调用方式

```bash
# 1) 单 session（无输出参数，落到 ./output/<basename>/）
mmpipe /data/00010a33-...

# 2) 单 session（显式输出根目录，落到 <out>/<basename>/）
mmpipe /data/00010a33-... /mnt/results

# 3) 批量目录（自动检测所有子 session，逐个处理，跳过已完成）
mmpipe /data /mnt/results
```

判别规则：输入目录里同时含 `config.json` + `rgb_head.mp4` → 单 session；否则扫描一级子目录里满足条件的全部当 session。

### 断点续跑

批量模式下，如果 `<output>/<session_basename>/lerobot_dataset/meta/info.json` 已存在，自动 SKIP 该 session。需要重新处理就删除对应输出目录。

### 调试 / 单层调用

```bash
mmpipe ingest <session_dir>            # 仅 Layer 1
mmpipe handpose <nir_dir>              # 仅 Layer 2
mmpipe lerobot <nir_dir>               # Layer 2 + 3
mmpipe validate <dataset_root>         # 读回校验
mmpipe info <dataset_root>             # 数据集摘要
```

所有子命令都接受 `--*-config <path>` 传入 JSON 覆盖算法参数。

## 云端部署典型用法（阿里云 ECS/DSW + OSS）

OSS 挂载到本地路径后，把固定的输出根目录写进 `.env`：

```bash
# .env
MMPIPE_OUTPUT_ROOT=/mnt/oss-output/processed/
MMPIPE_DEVICE=cuda:0
```

之后输入路径直接传 OSS 挂载点，无须每次写输出：

```bash
# 跑单个 session
mmpipe /mnt/oss-input/session_xyz
# → /mnt/oss-output/processed/session_xyz/lerobot_dataset/

# 批量整个 OSS 桶
mmpipe /mnt/oss-input/
# → /mnt/oss-output/processed/<session_basename>/ 每个 session 一份
```

中断后重跑等价于断点续跑 —— 已完成的 session 跳过，未完成的继续。

---

## 旧版 sensor-ETL 管线（保留）

下方为初版基于 stdlib 的传感器文件 ETL，与上述三层管线并存。

这个项目把已经采样好的 RGB、Depth、IMU、Pose、音频、点云等散装文件，流程化处理成可训练的数据包。

当前实现只依赖 Python 标准库，适合先把本地数据规范、时间对齐、低置信度分流和 WebDataset 风格 tar 分片跑通。后续可以把 `annotation` 阶段替换成 VLM/VLA API，把 `pack` 阶段接到对象存储。

## 处理流程

```text
raw files / manifest
  -> discover      生成 raw_manifest.jsonl
  -> validate      校验文件存在、时间戳、stream 字段
  -> align         按参考流做最近邻时间对齐
  -> annotate      规则标注 + 低置信度 review 队列
  -> pack          打成 shard-000000.tar + package_index.jsonl
```

## 快速开始

生成默认配置：

```powershell
python main.py init-config --output configs/default.json
```

运行完整管线：

```powershell
python main.py run --input .\data\raw --output .\data\processed --config .\configs\default.json
```

如果你没有把 Python 加入 PATH，在 Windows 上也可以用：

```powershell
py main.py run --input .\data\raw --output .\data\processed --config .\configs\default.json
```

## 输入方式

### 方式一：目录自动发现

文件名或路径中需要包含 10 位以上时间戳，例如：

```text
data/raw/
  cam_left/1716810000000000000.jpg
  depth/1716810000000005000.npy
  imu/1716810000000010000.json
  pose/1716810000000020000.json
```

管线会根据扩展名和路径关键词推断 modality：

| modality | 常见扩展名或路径关键词 |
| --- | --- |
| `rgb` | `.jpg`, `.png`, `.webp` |
| `depth` | `depth`, `.npy`, `.exr`, `.tiff` |
| `audio` | `.wav`, `.flac`, `.mp3` |
| `imu` | `imu`, `.json`, `.csv` |
| `pose` | `pose`, `6dof`, `tf` |
| `pointcloud` | `pointcloud`, `lidar`, `.pcd`, `.ply`, `.bin` |

### 方式二：显式 manifest

在配置里设置：

```json
{
  "manifest_file": "manifest.jsonl"
}
```

`manifest.jsonl` 每行一条记录，推荐字段：

```json
{"session_id":"run_001","stream":"rgb:cam_left","modality":"rgb","sensor_id":"cam_left","timestamp_ns":1716810000000000000,"path":"cam_left/1716810000000000000.jpg","metadata":{"label":"pick"}}
```

CSV 也支持，字段名保持一致即可。

## 输出结构

```text
data/processed/
  manifests/
    raw_manifest.jsonl
    validated_manifest.jsonl
    invalid_manifest.jsonl
    aligned_samples.jsonl
  annotation/
    accepted_samples.jsonl
    review_queue.jsonl
  shards/
    shard-000000.tar
    package_index.jsonl
  pipeline_report.json
```

`review_queue.jsonl` 是人工复核入口。默认逻辑会把缺流、时间偏移过大、标签置信度不足的样本拦下来，避免污染训练集。

## 关键配置

见 [configs/default.json](configs/default.json)。

| 配置项 | 作用 |
| --- | --- |
| `reference_stream` | 对齐锚点。可以是 `rgb`，也可以是完整 stream，如 `rgb:cam_left` |
| `expected_streams` | 期望每个样本必须具备的流；空数组表示使用发现到的全部流 |
| `alignment_tolerance_ms` | 时间对齐最大容忍偏移 |
| `annotation_confidence_threshold` | 低于该置信度的样本进入人工队列 |
| `shard_max_size_mb` | 单个 tar 分片最大体积 |
| `manifest_file` | 显式 manifest 文件路径，留空则扫描目录 |

## 下一步接入点

- 把 `stage_annotate` 替换为 Qwen/VLA 自动标注服务调用。
- 把 `stage_pack` 输出目录替换为对象存储上传。
- 把 `run_pipeline` 包成 Argo Workflow 或 Ray Task。
- 在 `pipeline_report.json` 基础上接 Prometheus 指标。

## 架构决策

关键架构选型记录在 `docs/adr/` 下，采用 MADR 模板：

| ADR | 主题 | 状态 |
| --- | --- | --- |
| [ADR-0001](docs/adr/0001-orchestration-selection.md) | 编排技术选型（Ray / KubeRay / Argo 分阶段路径） | Proposed |

引入新依赖或改变管线拓扑前，请先 review 相关 ADR 的退出准则。
