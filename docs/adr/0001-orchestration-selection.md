# ADR-0001: 多模态数据管线编排技术选型

- **状态**: Proposed
- **日期**: 2026-05-27
- **决策人**: 项目负责人 / ML 平台负责人（待补）
- **相关文件**: `multimodal_pipeline/pipeline.py`, `multimodal_pipeline/stages.py`, `README.md`

---

## 1. 上下文与问题陈述

本项目当前是一个 stdlib-only 的多模态数据管线骨架，6 阶段 ETL（discover / validate / align / annotate / pack / upload）已实现为 `stages.py` 中的纯函数，`pipeline.py` 以同步串行方式串起它们。`pyproject.toml` 中**没有任何编排工具依赖**，README 仅含糊提到"未来可能用 Argo Workflow 或 Ray Task"。

业务目标是支撑具身智能 / VLA 模型训练，需要：

- 数据规模从 GB 级原型成长到 TB → PB 级生产
- VLM 自动标注 + HITL（人机协同）人工复核闭环
- WebDataset 风格 tar 分片打包后入对象存储
- 与 GPU 训练任务深度耦合的数据预处理

硬约束：

- **目标场景**：公司生产级项目，需支撑多团队协作与 SLA
- **部署环境**：国内云（火山引擎 / 阿里云 / 腾讯云）
- **团队画像**：以 Python ML 工程师为主，K8s 经验中等

**待决策问题**：用什么工具来编排这条管线，并且在 6 个月、1 年、2 年三个时间窗内都不被迫推倒重选？

---

## 2. 决策驱动因素（Decision Drivers）

定义 7 个评分维度及其权重（权重越高对最终选型影响越大）：

| ID | 维度 | 权重 | 说明 |
|----|------|------|------|
| D1 | Python 原生 / ML 工程师友好度 | 5 | 团队画像决定，能否让 ML 工程师无障碍接入 |
| D2 | GPU 资源调度与异构算力支持 | 5 | 数据管线必然耦合训练，GPU 编排是核心能力 |
| D3 | 国内云生态兼容性 | 4 | 火山 / 阿里 ACK / 腾讯 TKE 的支持度与镜像可获取性 |
| D4 | 数据血缘与可观测性 | 4 | 生产级 SLA 必需，决定能否定位线上数据问题 |
| D5 | 多租户与团队隔离 | 3 | 公司共享集群必需，决定能否横向扩团队 |
| D6 | 学习曲线 / Time-to-Production | 3 | 决定 MVP 多快上线、新人多快上手 |
| D7 | 社区活跃度与中文资料 | 2 | 影响 onboarding 与故障排查速度 |

评分采用 1–5 分制（5 分最优）。总分 = Σ(分数 × 权重)，理论上限 130 分。

---

## 3. 候选方案对比

### 3.1 Ray + KubeRay

**定位**：Python 原生分布式计算框架，KubeRay 是它在 K8s 上的 Operator。

**架构一句话**：把 Python 函数加 `@ray.remote` 装饰器就能跑在集群上，原生支持 GPU 资源声明（`num_gpus=1`），KubeRay 让 Ray 集群在 K8s namespace 内动态弹性伸缩。

**国内云落地关键点**：
- 阿里云 ACK、火山 VKE 均已有用户验证过 KubeRay 0.6+；ACK 1.24+ 与 KubeRay CRD 兼容性良好
- 需使用阿里 ACR / 火山 CR 做 `rayproject/ray:*` 镜像缓存，否则国内拉取速度感人
- 字节跳动内部大规模使用 Ray，中文实践案例最丰富

**评分**：

| D1 | D2 | D3 | D4 | D5 | D6 | D7 | 加权总分 |
|----|----|----|----|----|----|----|----------|
| 5 | 5 | 4 | 2 | 3 | 4 | 4 | **103** |

### 3.2 Dagster

**定位**：现代化数据编排框架，以"数据资产"为核心一等公民，血缘和质量是核心卖点。

**架构一句话**：用 `@asset` / `@op` 装饰器声明数据资产与转换，框架自动维护血缘图、版本化、自动 materialize 缺失资产。

**国内云落地关键点**：
- 镜像 `dagster/dagster:*` 在国内 PyPI / Docker Hub 拉取速度差，需自建镜像仓库
- 国内生产案例较少，中文文档稀缺
- 对 GPU 编排无原生支持，需要配合 Ray 或 K8s 资源请求

**评分**：

| D1 | D2 | D3 | D4 | D5 | D6 | D7 | 加权总分 |
|----|----|----|----|----|----|----|----------|
| 5 | 2 | 3 | 5 | 4 | 3 | 3 | **94** |

### 3.3 Argo Workflows

**定位**：K8s 原生的容器化 DAG 引擎，CNCF 顶级项目。

**架构一句话**：用 YAML 声明 DAG，每个节点是一个 Pod，原生支持回填、重试、定时调度（CronWorkflow）、Artifact 传递。

**国内云落地关键点**：
- 阿里云 ACK、火山 VKE、腾讯 TKE 全部官方推荐，部署成熟
- 镜像 `quay.io/argoproj/*` 需要镜像加速
- 中文资料因 KubeCon China 推广已较丰富

**评分**：

| D1 | D2 | D3 | D4 | D5 | D6 | D7 | 加权总分 |
|----|----|----|----|----|----|----|----------|
| 2 | 4 | 4 | 3 | 5 | 2 | 4 | **87** |

### 3.4 Apache Airflow

**定位**：传统数据调度系统，社区最大，阿里云 DataWorks 等托管服务的底层。

**架构一句话**：用 Python 写 DAG，每个 Operator 是一个任务，调度中心化（Scheduler + Worker）。

**国内云落地关键点**：
- 阿里云有 DataWorks（托管 Airflow 变体），生态最成熟
- 中文资料最丰富，但模型多偏向数仓 ETL 而非 ML

**评分**：

| D1 | D2 | D3 | D4 | D5 | D6 | D7 | 加权总分 |
|----|----|----|----|----|----|----|----------|
| 4 | 1 | 5 | 3 | 3 | 3 | 5 | **85** |

### 3.5 Prefect

**定位**：现代化的 Airflow 替代品，Python 优先。

**架构一句话**：用 `@flow` / `@task` 装饰器写 Python 函数，Prefect Cloud / Prefect Server 提供编排控制平面。

**国内云落地关键点**：
- 国内生产案例极少，中文资料最少
- 对 GPU/ML 工作负载支持弱

**评分**：

| D1 | D2 | D3 | D4 | D5 | D6 | D7 | 加权总分 |
|----|----|----|----|----|----|----|----------|
| 5 | 2 | 2 | 4 | 4 | 4 | 2 | **87** |

### 3.6 Flyte

**定位**：Lyft 出品的 K8s 原生 ML 管线框架，强血缘、强类型、强多租户。

**架构一句话**：用 Python 写有类型签名的任务，编译后产生 K8s 资源对象，Project/Domain 提供原生多租户隔离。

**国内云落地关键点**：
- 在 ACK/VKE 上部署需自行处理 Helm Chart 与镜像
- 国内案例少，但 Union.ai 在推国际化文档
- 学习曲线最陡，对 ML 工程师不直观

**评分**：

| D1 | D2 | D3 | D4 | D5 | D6 | D7 | 加权总分 |
|----|----|----|----|----|----|----|----------|
| 4 | 4 | 3 | 4 | 5 | 2 | 2 | **93** |

### 3.7 云厂商托管（火山 veMLP / 阿里云 PAI）

**定位**：云厂商提供的 ML 平台，包含数据处理 + 训练 + 部署的全套托管能力。

**架构一句话**：通过控制台或 SDK 提交任务，云厂商负责底层 K8s/GPU 调度，用户只负责业务逻辑。

**国内云落地关键点**：
- 接入快，运维成本最低
- **强厂商锁定**：迁移成本极高，API 不兼容
- 部分高级能力（如自定义 Operator）受限

**评分**（不含锁定风险的纯能力分）：

| D1 | D2 | D3 | D4 | D5 | D6 | D7 | 加权总分 |
|----|----|----|----|----|----|----|----------|
| 3 | 5 | 5 | 3 | 5 | 5 | 5 | **112** |

> **重要说明**：托管平台的纯能力分最高，但**锁定风险**未计入打分模型。详见第 8 节"被否决的方案"。

### 3.8 总分排序

| 方案 | 总分 | 主要短板 |
|------|------|---------|
| 云厂商托管 | 112 | 锁定风险极高 |
| Ray + KubeRay | 103 | 血缘弱 |
| Dagster | 94 | GPU 调度弱 |
| Flyte | 93 | 学习曲线陡 |
| Argo Workflows | 87 | Python 不友好 |
| Prefect | 87 | 国内生态薄弱 |
| Airflow | 85 | GPU 几乎不支持 |

**关键洞察**：没有任何单一方案在所有维度上都赢。这直接导出第 5 节的"分阶段组合"决策。

---

## 4. 决策：分阶段组合架构

**核心论点**：不要一次性选定终态工具栈。按数据规模与团队规模分四阶段演进，每阶段只引入"刚好够用"的复杂度。

```text
Phase 0           Phase 1            Phase 2                  Phase 3
stdlib only  →   Ray Core      →    K8s + KubeRay + Argo  →   + Dagster/Flyte
(now)           (1-3 月)            (3-6 月)                  (6 月+)
```

### Phase 0（当前 → 1 个月）：stdlib + `ProcessPoolExecutor`

**不引入任何编排框架**。当前数据量 < 100 GB，单机就能处理。

具体动作：

- `pipeline.py` 内用 `concurrent.futures.ProcessPoolExecutor` 把 stages 之间能并行的部分（如 validate 内部按文件分片）并行化
- 给 `stages.py` 的每个函数加结构化日志（`logging` + JSON formatter），输出阶段耗时、处理样本数、失败率
- 在 `pipeline_report.json` 基础上加 stage-level metrics，为后续 Prometheus 接入留接口

**为什么这一步**：现在引入 Ray 都是 over-engineering。先让管线跑通、把可观测打底。

### Phase 1（1–3 个月）：引入 Ray Core

数据量到了 100 GB+ 或需要远程调用 VLM API 时启动。

具体动作：

- `pyproject.toml` 添加 `ray[default]`
- `stages.py` 的高耗时函数加 `@ray.remote`（特别是 `stage_annotate` 调用远程 VLM 时）
- 使用 Ray Data 并行化 `stage_pack` 的 tar 分片写入
- 部署形态：本地 Ray 单机 / 火山 ECS 上手工起 3 节点 Ray 集群

**为什么 Ray 而不是 Dagster**：D1（Python 原生 5/5）+ D2（GPU 5/5）压倒性优势；当前需求是"算得快"，不是"血缘可视化"。

### Phase 2（3–6 个月）：上 K8s，引入 KubeRay + Argo Workflows

团队 > 3 人或数据 > 1 TB 时启动。

具体动作：

- 部署到阿里云 ACK / 火山 VKE / 腾讯 TKE（任一）
- KubeRay 跑训练侧的弹性 Ray 集群（GPU autoscale）
- Argo Workflows 跑批处理 ETL DAG（定时拉取、按 session 分片、失败回填）
- 引入 Prometheus + Grafana 监控，对象存储用火山 TOS / 阿里 OSS 作为 shard 落盘
- 镜像策略：所有 `rayproject/*`、`quay.io/argoproj/*` 通过阿里 ACR / 火山 CR 做镜像同步

**为什么 Ray + Argo 组合而不是单选**：Ray 擅长 Python 原生计算与 GPU 调度（管线内部），Argo 擅长容器级 DAG 与定时/回填（管线之间）。两者职责互补，是国内大厂的常见组合。

### Phase 3（6 个月+）：补齐数据治理 —— Dagster 或 Flyte

数据资产 > 50 个、跨团队消费时启动。

具体动作：

- 评估 Dagster vs Flyte：
  - Dagster：更轻、Python 原生、血缘可视化最强 → 推荐为默认选择
  - Flyte：K8s 原生、多租户最强 → 仅当团队 > 20 人且对类型安全有强需求时选
- Ray + Argo 留在底层执行，Dagster / Flyte 作为上层 control plane 与数据资产目录

---

## 5. 退出准则（Exit Criteria）—— 本 ADR 的核心价值

每个 Phase 必须满足**至少 2 条退出准则**才能进入下一阶段。这避免拍脑袋升级，也避免"看着别人用 K8s 我们也上"的盲目跟进。

### Phase 0 → Phase 1（引入 Ray）

- [ ] 单批次端到端处理时间超过 30 分钟，且 `top` 显示 CPU 平均利用率 < 60%（说明 IO/调度成为瓶颈，而非算力）
- [ ] 数据总量 > 100 GB 或单 session 文件数 > 10 万
- [ ] 出现需要跨进程并行的远程调用（典型：`stage_annotate` 调远程 VLM API 时，本机大部分 CPU 空闲）

### Phase 1 → Phase 2（上 K8s + KubeRay + Argo）

- [ ] 团队规模 > 3 人，且开始抢同一台机器的 GPU
- [ ] 需要定时或事件触发的 ETL（典型：每天凌晨自动拉取前一天数据并打包）
- [ ] 出现需要回填的失败任务，但 Ray 缺少原生 DAG 回填能力
- [ ] 数据总量 > 1 TB 或单次训练任务跨节点

### Phase 2 → Phase 3（引入 Dagster / Flyte）

- [ ] 数据资产（独立可消费的数据产物）数量 > 50 个
- [ ] 出现"上游改了 schema 下游不知道"导致的训练失败事故
- [ ] 合规或安全审计要求数据血缘可追溯（如等保 / GDPR 类需求）
- [ ] 数据团队与 ML 团队成为独立的组织单位

---

## 6. 风险与缓解

| 风险 | 影响 | 缓解 |
|------|------|------|
| 国内云 K8s 版本与 KubeRay 兼容性 | 高 | Phase 2 启动前在测试集群验证 KubeRay 与 ACK/VKE/TKE 当前 GA 版本组合 |
| Argo / KubeRay 镜像在国内拉取慢 | 中 | 使用云厂商 CR 服务（火山 CR / 阿里 ACR / 腾讯 TCR）建立镜像同步管道 |
| Ray 在 6 阶段管线里的故障传播 | 中 | 每个 stage 落盘 checkpoint；`@ray.remote(max_retries=3)`；使用 Ray 的 Task Lineage Reconstruction |
| 团队 K8s 经验不足导致 Phase 2 卡壳 | 高 | Phase 1 期间安排专人系统学习 K8s；或评估火山 veMLP / 阿里 PAI 作为兜底退路 |
| 提前引入 Dagster 导致复杂度爆炸 | 中 | 严格遵守 Phase 3 退出准则，不在数据资产数量 < 50 时引入 |
| 跳过 Phase 0 直接上 Ray | 中 | 用本文档作为 code review checklist，PR 引入 `ray` 依赖必须 link 到 Phase 0→1 退出准则证据 |

---

## 7. 决策影响（Consequences）

**正面**：

- 团队在每个阶段只学一种新工具，认知负担可控
- 工具栈始终匹配数据规模，不会出现"P0 阶段 K8s 集群空跑烧钱"
- 每次升级都有可观测、可验证的退出准则，减少决策摩擦
- Ray + Argo 是国内大厂验证过的成熟组合，招聘和 onboarding 友好

**负面 / 待权衡**：

- Phase 1 → 2 跨度大，引入 K8s 需要专门的迁移窗口（建议预留 2 周）
- Ray 与 Argo 之间需要约定接口（Argo 的 Pod 启动 Ray Job 还是反过来），这部分需要单独 ADR
- 没有上来就引入 Dagster 意味着 Phase 0/1 的血缘只能靠日志，初期数据问题排查会偏粗放

---

## 8. 被否决的方案

### Airflow 作为主用编排
- Python 子进程模型对长任务、GPU 任务、动态资源场景不友好
- 即便用 KubernetesExecutor 也比 Argo 多一层抽象，本质是 K8s 原生不足
- 阿里云 DataWorks 是托管 Airflow，但偏数仓场景，对 ML 数据管线不是最优

### Prefect 作为主用编排
- 国内案例极少，故障排查时缺中文社区支持
- 对 GPU 工作负载的原生支持弱于 Ray
- 数据血缘弱于 Dagster

### 一上来就用火山 veMLP / 阿里云 PAI
- 锁定风险极高，跨云迁移几乎要重写
- Phase 0 / 1 阶段杀鸡用牛刀，付费曲线陡
- 高级定制（自定义 CRD / 调度策略）受平台限制
- **保留作为 Phase 2 卡壳时的兜底退路**，但不作为主路径

---

## 9. 参考资料

- [Ray on Kubernetes 官方文档](https://docs.ray.io/en/latest/cluster/kubernetes/index.html)
- [KubeRay GitHub](https://github.com/ray-project/kuberay)
- [Argo Workflows](https://argoproj.github.io/argo-workflows/)
- [Dagster Concepts](https://docs.dagster.io/concepts)
- [Flyte](https://flyte.org/)
- [阿里云 ACK 文档](https://help.aliyun.com/product/85222.html)
- [火山引擎 VKE 文档](https://www.volcengine.com/docs/6460)
- [MADR 模板](https://adr.github.io/madr/)
- [字节跳动 Ray 实践分享（KubeCon China）](https://www.cncf.io/kubecon-cloudnativecon-events/)
- 本仓库 `README.md` 中的"下一步接入点"章节

---

## 10. 后续行动

本 ADR 通过后，立即开启 Phase 0 执行单：

1. 在 `pipeline.py` 引入 `ProcessPoolExecutor` 并行化（独立 PR）
2. 给 `stages.py` 添加结构化日志（独立 PR）
3. 扩展 `pipeline_report.json` 输出 stage-level metrics（独立 PR）

Phase 1 启动条件成立时，开新 ADR（ADR-0002）专门讨论 Ray 集成的具体方式（local mode vs cluster mode、Ray Data vs ray.remote、是否使用 Ray Serve 等）。
