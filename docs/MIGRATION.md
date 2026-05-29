# 迁移到新机器运行

把本管线迁移到性能更强的机器（GPU + 更好散热）时的清单。

> 动机：原笔记本 CPU 长期 100%、高温触发关机。诊断结论——**高温来自软件视频编码（x264 + FFV1），不是 ML 计算**（`real_ingest` 后端是近乎免费的 numpy）。换机 + GPU 硬件编码直接解决根因。

## 一、git 带不走、必须手动准备的东西

以下都在 `.gitignore` 里，`git clone` 之后不会出现，需要单独准备：

### 1. `.env`（最关键）

含 DashScope API key，**绝不进 git**。在新机参照 `.env.example` 重建，注意这些键：

| 键 | 说明 |
|---|---|
| `MMPIPE_DASHSCOPE_API_KEY` | DashScope key，机密，单独安全传输 |
| `MMPIPE_DASHSCOPE_MODEL` | `qwen-vl-plus` |
| `MMPIPE_LANGUAGE_BACKEND` / `MMPIPE_ACTIONS_BACKEND` / `MMPIPE_QUALITY_BACKEND` | Layer 1.5 后端（`dashscope` / `rule_based` / `mock`） |
| `MMPIPE_HANDPOSE_BACKEND` | **务必设为 `real_ingest`**，否则退回 mock 假数据 |
| `MMPIPE_DEVICE` | `cpu` 即可（重活在 ffmpeg 编码，非 ML） |
| `MMPIPE_OUTPUT_ROOT` | 输出根目录（云端可指向挂载的 OSS） |
| `MMPIPE_LOG_LEVEL` | 日志级别 |

### 2. 标注缓存 `.annotation_cache.db`

位于 `<MMPIPE_OUTPUT_ROOT>/.annotation_cache.db`（已 gitignore）。

- Layer 1.5 的 DashScope 响应持久化缓存。
- **把它拷到新机的 output_root 可保留缓存命中**，否则全量重跑会重新消耗 DashScope 调用。
- （管线的重活在 Layer 1 摄取 + Layer 3 编码；标注命中缓存后应近乎零成本。）

### 3. 源数据

40 个 ITW session，共 ~3.4GB，不在 git 里，需单独传输。

### 4. 产物目录

`output/`、`output_test_*/`、`*.log`、`quality_*.json` 都不传，新机重新生成。

## 二、GPU 硬件编码（自动生效）

- 视频编码器配置 `video_encoder=auto`（默认），启动时探测 `h264_nvenc`。
- 原笔记本 GPU 驱动版本过低（566 < 所需 570），探测失败 → 回退 x264（软件编码，吃 CPU、发热）。
- **新机驱动够新时，`auto` 会自动启用 `h264_nvenc`**（GPU ASIC 编码，近零 CPU、低热）。无需改任何配置。
- 16-bit 米制深度走 FFV1 gray16le 无损流（H264/yuv420p 会破坏深度值），这部分仍是 CPU 编码。

## 三、迁移后：全量重跑

当前 T0–T4 多模态接入 + 可视化 + Layer 1.5 提速均已提交，但**仅在单 session 验证过**。新机上做一次全量 40 session 重跑：

```bash
mmpipe <sessions_parent_dir> <output_root> --force --parallel-sessions 4
```

- `--parallel-sessions N`：按核数调整，4 起步。
- 跑完默认自动生成 `<output_root>/quality_report.json`。

**验证项**：
- 关键点（`observation.hand_keypoints`）非 NaN 率高
- 深度流（`observation.images.depth`）存在
- IMU（`observation.imu`）/ 接触相位（`observation.contact_phase`）特征齐全
- `validate_dataset` 全过

**人工抽检可视化**：

```bash
mmpipe visualize <output_root>/<session>/lerobot_dataset --episode 0 --depth
```

## 四、环境约定

部署目标为 Linux + Python 3.10/3.11，OSS 挂载到本地文件系统作输入输出，通过 `MMPIPE_OUTPUT_ROOT` 指定输出根目录。详见 `docs/adr/0001-orchestration-selection.md`。
