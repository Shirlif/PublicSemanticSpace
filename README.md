# Public Semantic Space (PSS) — Phase 0

依据 `公共语义空间推理架构-推演记录-11.19.md` 与 `公共语义空间-落地执行计划-v1.md` 建立的工程起点。

> 目录结构与核心代码导读：`notes/PROJECT_MAP.md`。

## 核心入口：三步

1. **冻结专家**：视觉 SmallCNN/预训练骨干、触觉 WT-MM CNN、几何手工特征。
2. **MLP 投影头**：256 / 64 / 16 → 128 维公共空间，L2 归一化。
3. **原型库 + 识别**：128 维向量与类别原型做余弦检索，输出 top-k。

```bash
# 重建原型库（首次或换专家后）
python -m pss.demos.public_semantic_space_demo --build-bank

# 输入一张图 + 一块触觉图，输出 top-3
python -m pss.demos.public_semantic_space_demo \
  --image data/demo_input.png --tactile data/demo_tactile.npy
```

当前测试集 top-1 **0.990**（6 类）。示例输出：

```
Top-k:
  1. cup          +0.952
  2. block        +0.522
  3. bottle       +0.509
各模态 top-1: {'vision': 'cup', 'tactile': 'plate'}
冲突标记: True
```

> 触觉单独判错、视觉/几何判对时，融合仍是 cup，并打冲突标记。这是架构的预期行为。

---

## 类别识别预训练（v2，当前主线）

预训练阶段只做一件事：**识别物体类别**。实例 ID、追踪、绑定、具身部署往后放。

- 20 类（16 已见 + 4 未见），每类 24 变化 × 4 视角，1,920 帧
- 视觉专家 test **0.943**；几何 **0.911**；触觉 0.245（材质代理，弱）
- 公共空间融合 test **0.969**；开放集 AUROC **0.960**，未见拒绝 0.818
- 融合权重由训练集 3 折 CV 选出：`vision 1.0 / tactile 0.0 / geometry 1.0`

> 2026-11-22 更新：Phase 4/5 材质感知投影后，融合 test **0.979**，开放集 AUROC **0.993**，未见拒绝 **0.958**，权重 `vision 1.0 / tactile 0.5 / geometry 1.0`。详见 `PHASE45_REPORT.md`。

```bash
make category-pretrain
```

详见 `CATEGORY_PRETRAIN_REPORT.md`。


下面是工程化与下游记录（Phase 0–10、追踪、规划、安全、公共数据），**不是核心三步的前置条件**，可按需阅读。


Phase 0 只做三件事：**冻结接口契约、重建三个基线 Demo、量化绑定算子**。感知学习、原型树、3D 场景图、世界模型、权限与安全层都在后续阶段。

## 快速开始

```bash
python -m pytest
python -m pss.demos.semantic_space_demo
python -m pss.demos.token_vs_vector_reasoning
python -m pss.demos.laundry_task_demo
python bench/binding_stability.py            # Phase 0 基准
python bench/phase1_binding.py               # Phase 1 Hadamard vs HRR
python bench/binding_stability.py --quick    # CI 快速版
python -m pss.sim.scripts.generate_dataset --out data/v1 --scale small --seed 0
python -m pss.sim.scripts.generate_dataset --out data/expert_v1 --scale expert --seed 0
python -m pss.encoders.scripts.train_experts --data data/expert_v1 --out checkpoints/expert_v1 --epochs 30 --size 64 --seed 0
python -m pss.projection.scripts.train_projections --data data/expert_v1 --experts checkpoints/expert_v1 --out checkpoints/projection_v1 --epochs 60 --batch-size 64 --temp 0.07 --seed 0
python -m pss.prototypes.scripts.build_prototypes --data data/expert_v1 --experts checkpoints/expert_v1 --projections checkpoints/projection_v1 --out checkpoints/prototypes_v1 --seed 0
python -m pss.memory.scripts.run_tracking --data data/v1 --experts checkpoints/expert_v1 --projections checkpoints/projection_v1 --out checkpoints/memory_v1 --occlusion 4 --episodes 2 --seed 0
python -m pss.reasoning.scripts.run_phase7 --data data/v1 --out checkpoints/phase7 --episodes 8 --epochs 100 --seed 0
python -m pss.heads.scripts.run_phase8 --data data/v1 --experts checkpoints/expert_v1 --projections checkpoints/projection_v1 --prototypes checkpoints/prototypes_v1 --out checkpoints/phase8 --trials 10 --seed 0
python -m pss.learning.scripts.run_phase9 --data data/expert_v1 --experts checkpoints/expert_v1 --projections checkpoints/projection_v1 --prototypes checkpoints/prototypes_v1 --out checkpoints/phase9
python -m pss.eval.scripts.run_phase10 --data data/expert_v1 --occlusion-data data/hard_occlusion --experts checkpoints/expert_v1 --projections checkpoints/projection_v1 --prototypes checkpoints/prototypes_v1 --out eval/results --seed 0
python -m pss.sim.scripts.generate_hard_occlusion --out data/hard_occlusion --sequences 6 --frames 12 --seed 0
python -m pss.rd2.scripts.make_mock --out data/rd2_mock --data data/expert_v1 --episodes 8 --seed 0
python -m pss.rd2.scripts.validate --path data/rd2_mock
python -m pss.encoders.scripts.train_pretrained_expert --data data/expert_v1 --out checkpoints/expert_pretrained_v1 --backbone squeezenet1_1 --pretrained true --epochs 30 --size 64 --seed 0
```

Windows PowerShell 一键：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_all.ps1
```

## Phase 0 验收清单

- [x] 三个 Demo 重建（客体识别 / Token vs 向量 / 洗衣任务）
- [x] 接口契约 v0（Entity / Relation / PrototypeNode）
- [x] 修复 `unbind` 近零负分量符号翻转缺陷
- [x] 槽方案对比：Hadamard / Rademacher / Gaussian
- [x] 绑定稳定性基准（噪声 × 轮次 × 容量 × 范数漂移）
- [x] 单测覆盖数学边界（单分量可逆、同槽相加不可分离、正交槽）
- [ ] 真实数据 schema 合并（RD0，见执行计划）

结果汇总见 `PHASE0_REPORT.md`。

## Phase 1（已完成）

- [x] 分模态实体装配（方案 A，禁止跨模态相加）
- [x] 关系三元组三份独立分量 + `fact` 仅检索
- [x] HRR 对照实现与基准
- [x] 槽池：槽 = 工作记忆位置，不是实体身份
- [x] rebind 策略与漂移监控
- [x] 契约 v0.1.1（`Entity.slot`、`bound` 临时性、`BINDING_SCHEME`）
- [x] 38 个测试通过

**结论：保留 Hadamard。** HRR 在迭代下误差指数发散（10 轮 ~1e6，100 轮 ~1e80），
容量也无一致优势。详见 `PHASE1_REPORT.md` 与 `bench/results/phase1_binding.md`。

## Phase 2（已完成，v1）

- [x] MuJoCo 桌面场景（8 类原始几何体 + 2 容器 + 遮挡物）
- [x] 传感器：RGB / 深度 / 分割 / 位姿 / 接触 / 关系
- [x] 动作代理：push / lift / place / wait
- [x] 三类样本：category / episode / occlusion
- [x] Dataset v1：46 条，5.3 MB；`index.json` + `DATA_CARD.md`
- [x] 44 个测试通过

**边界：原始几何体、合成触觉、运动学抓取代理。** 只够打通管线，不作为能力结论。
详见 `PHASE2_REPORT.md`。

## Phase 3（已完成，冒烟版）

- [x] 数据加载与按分割裁剪
- [x] 视觉专家（小 CNN，256-d）
- [x] 触觉专家（MLP，64-d）
- [x] 几何专家（16-d 手工特征 + 逻辑回归探针）
- [x] 训练 CLI 与冻结导出（`checkpoints/expert_v1/`）
- [x] 51 个测试通过

闭集 top-1（6 类，随机基线 0.167）：

| 专家 | train | val | test |
|---|---:|---:|---:|
| 视觉 | 1.000 | 1.000 | 0.958 |
| 触觉 | 0.479 | 0.500 | 0.365 |
| 几何 | 1.000 | 0.948 | 0.802 |

**这是管线冒烟结果，不是能力结论。** 视觉任务过于简单（原始几何体 + 强区分颜色）；
触觉是合成材质代理，接近随机；未见类别（`test_category`）留待 Phase 5 原型匹配。
详见 `PHASE3_REPORT.md`。

## Phase 4（已完成，类别级对齐）

- [x] 每模态独立投影头（256/64/16 → 128，L2 归一化）
- [x] 多正样本监督对比（V-T / V-G / T-G 六向）
- [x] 属性辅助头（材质 + 尺寸）
- [x] 冻结专家，只训投影头
- [x] 投影权重与指标（`checkpoints/projection_v1/`）

类别 R@1（验证/测试）：vision↔geometry **0.82 / 0.85**；vision↔tactile **0.51 / 0.51**。
**pair R@1 接近随机（~0.01–0.06）**：类别级损失不约束实例级配对，触觉也是视角无关的合成代理。
实例级绑定必须等 RD2 同步数据。详见 `PHASE4_REPORT.md`。

## Phase 5（已完成，冒烟版）

- [x] 分模态原型库（6 类 × 3 模态，128-d）
- [x] 数据驱动的分层原型树
- [x] 四级检索 + 置信度加权融合
- [x] 开放集阈值与冲突标记
- [x] 原型准入预审 + 阈值校准诊断
- [x] `checkpoints/prototypes_v1/`（含 manifest + sha256）
- [x] 60 个测试通过

| 查询 | val | test |
|---|---:|---:|
| 融合 + 树 | 0.969 | 0.938 |
| 融合 flat | 0.990 | 0.969 |
| 仅视觉 | 1.000 | 0.969 |
| 仅触觉 | 0.500 | 0.521 |
| 仅几何 | 0.854 | 0.854 |

开放集：阈值 0.808，已见误拒 11.5%，未见拒绝 77.1%。
**关键发现**：固定融合权重不如最强单模态；树贪心裁剪损失 2–3 点；准入阈值 0.25 会误杀现有合法类别（tactile 最近邻仅 0.063）。阈值只是占位值。详见 `PHASE5_REPORT.md`。

## Phase 6（已完成）

- [x] 3D 场景图 + 网格空间索引 + 持久化
- [x] 位姿/外观/时序关联 + 生命周期 + re-ID
- [x] 4 槽工作记忆 + 注意力调度 + raw 重建
- [x] 碎片 API（目睹事件分支）
- [x] 追踪/工作记忆/规模基准（`checkpoints/memory_v1/`）
- [x] 69 个测试通过

| 指标 | 值 |
|---|---:|
| 遮挡序列 ID 保持 | **1.000** |
| Episode ID 保持 | 0.917（瞬移动作导致） |
| 工作记忆焦点切换召回 | **1.000** |
| 焦点切换延迟 p50 | 0.018 ms |
| 200 实体下活跃/全量关系数 | **6 / 19,900** |

**关键发现**：遮挡追踪达标；episode 的 ID 切换来自 Phase 2 的运动学瞬移动作，不是关联算法。工作记忆"槽 ≠ 身份"在运行时成立。详见 `PHASE6_REPORT.md`。

## Phase 7（已完成）

- [x] 关系记录 + 向量编码（三份 part + fact）
- [x] 动作算子库（软前置条件 + 效果）
- [x] A* 规划器 + 异常重规划
- [x] 观测-预测冲突检测
- [x] 解析世界模型 + 学习 MLP（target-only）
- [x] 77 个测试通过

| 指标 | 值 |
|---|---:|
| 关系总体 F1 | **0.878**（`in` 1.000，`contact` 0.000） |
| A* 洗衣规划 | 12 步 / 50 节点 / 成功 |
| 无洗衣液 | 正确返回 no path |
| 洒出后重规划 | 9 步 / 目标达成 |
| 世界模型 1 步误差 | zero 0.046 / 解析 0.040 / **学习 0.025** |
| 世界模型 rollout 误差 | zero 0.154 / 解析 0.145 / **学习 0.091** |

**关键发现**：规划与异常恢复跑通；`contact` 用半径近似会漏掉真实接触，必须改用碰撞几何。详见 `PHASE7_REPORT.md`。

## Phase 8（已完成）

- [x] 动作头（目标 → 推力原语），闭环 10/10 试验
- [x] 文本头（零参数只读投影 + 可选冻结 LLM 适配器）
- [x] 价值/目标头（注意力先验）
- [x] 文本头消融：核心准确率 delta = 0，原型库 hash 不变
- [x] 82 个测试通过

| 指标 | 值 |
|---|---:|
| 动作头闭环成功率 | **0.900** |
| 平均最终距离 / 推力次数 | 0.039 m / 7.6 |
| 文本头参数量 | **0** |
| 消融准确率 | 1.0000 → 1.0000（delta 0） |
| 原型库 hash 未变 | ✅ |

**关键发现**：语言读出与核心表征在结构上隔离；动作头是规则控制器，验证的是读出链路而非学习策略。详见 `PHASE8_REPORT.md`。

## Phase 9（已完成）

- [x] L1 实例记忆 + 脱敏提案
- [x] L2 原型包：hash、增量追加、原子激活、回滚、损坏拒绝
- [x] L3/L4 版本注册表
- [x] 审核流水线：聚合 / 预审 / 回归 / 人工占位
- [x] 三层护栏：硬黑名单 / 价值向量 / 场景原型
- [x] 四级降级 FSM + debounce + 离线模式
- [x] 审计快照 + 回放校验
- [x] 93 个测试通过

| 指标 | 值 |
|---|---:|
| L1 提案样本 | 28（已脱敏） |
| 预审 / 回归 | reject / **1.000** |
| 审核结论 | **accept**，激活 v1.1 |
| 回滚 | v1.0；损坏包拒绝 |
| 硬安全否决 | ✅（含 LLM 来源动作） |
| 降级序列 | normal → warning → degraded → emergency → normal |
| 审计回放 | ✅ 通过 |

**关键发现**：预审距离阈值会误杀可用类别，回归准确率才是当前决策门；L2 完整性与回滚可用。详见 `PHASE9_REPORT.md`。

## Phase 10（已完成，预注册评测）

- [x] 预注册假设、指标、杀死标准（`eval/preregistration.md`）
- [x] 指标库、基线/消融、评测 CLI
- [x] E1–E6 全部运行并报告
- [x] 119 个测试通过

| 项 | 值 |
|---|---:|
| 无槽：逐实体 / 全局向量 | **0.766 / 0.250** |
| V-T R@1：训练 / 随机投影 | **0.510 / 0.115** |
| V-G R@1 | 0.854 |
| 遮挡 ID retention：main / no-SG | **1.000 / 0.128** |
| 主方法 vs B1 文本代理 | **+0.000** |
| 杀死标准 K1–K4 | **全部触发** |

**结论：结构可行性成立，能力优势未成立。** 槽位与训练投影的价值可见；但遮挡数据太弱（B1 打平）、触觉对齐不达标（V-T 0.510）、D=128 容量不足、E0 未运行。下一步应补 RD2 真机数据、更难的遮挡基准、真实 VLM 基线。详见 `PHASE10_REPORT.md`。

## Phase 10 加固（1–4，已完成）

- [x] 完全遮挡基准（幕布覆盖，4 帧零观测）
- [x] RD2 schema / 校验器 / 适配器 / 采集协议 / mock
- [x] VLM 基线适配器 + mock（真实 API 未调用）
- [x] 真实预训练骨干（SqueezeNet1.1，test 0.979）

| 项 | 结果 |
|---|---:|
| 硬遮挡 ID retention：main / B1 / no-SG | **1.000 / 0.500 / 0.125** |
| K3 | **未触发**（+0.500） |
| RD2 mock 校验 | 8/8 通过 |
| Mock VLM retention | 0.500 |
| SqueezeNet test vs SmallCNN | **0.979 / 0.958** |
| K1 / K2 / K4 | 仍触发 |

详见 `PHASE10_HARDENING_REPORT.md`。

## 公共数据适配（LeRobot v3.0）

- 注册表：`pss/rd2/public/registry.py`（WT-MM / OpenGraphLabs / Daimon-Infinity / VisuoTactile-GS）
- 适配器：`pss/rd2/public/lerobot.py`（Parquet + 视频，tactile/state 读取）
- 冒烟：`tactile_stats`、`temporal_consistency`、`train_autoencoder`、`video_tactile_alignment`
- CLI：`python -m pss.rd2.public.scripts.download_smoke --dataset wt_mm --limit 1500`

WT-MM（Apache-2.0）最小对验证（episode 50，1027 帧，data file-004 + video file-005）：

| 指标 | 值 |
|---|---:|
| 接触帧比例 | 0.367 |
| 时序余弦：相邻 / 随机 | **0.963 / 0.409** |
| AE 验证 MSE / latent std | 0.000117 / 0.080 |
| **视频-触觉 R@1** | **0.211** |
| 打乱基线 / 随机 | 0.0019 / 0.00097 |

R@1 约为随机的 217 倍，说明 WT-MM 的视-触是真正逐帧同步且可学的；但它是 B 类（无物体 6D 位姿/实例 ID），不能替代 C 类/E0。

```bash
python -m pss.rd2.public.scripts.download_smoke --dataset wt_mm \
  --root data/public/wt_mm_pair --limit 1027 \
  --video data/public/wt_mm_pair/videos/observation.images.cam/chunk-000/file-005.mp4
```

OpenGraphLabs（CC-BY-4.0）作为备选，尚未跑完；Daimon-Infinity 需 Gated 申请。

### WT-MM 触觉编码器

- 预训练：`pss/encoders/tactile/wtmm.py`（12×32，时序对比 + 重建 + 接触头）
- 结果：接触探针随机平衡 **0.978**；跨传感器迁移到 8×8 合成触觉 test **0.583**（原 0.365）
- Phase 4 重训：**V-T R@1 0.510 → 0.583**；tactile→geometry 0.438 → 0.552
- K2 仍未过（0.583 < 0.60）；pair R@1 仍接近随机（B 类数据无实例级同步）

```bash
make tactile-wtmm
make projection-wtmm
```

详见 `PHASE10_WTMM_TACTILE_REPORT.md`。

### T1 实例级对照（合成数据）

- 视角切分：每个实例视角 0–5 训练、6–7 测试（模拟 T1 异步采集）
- 类别级 vs 实例级 SupCon：instance R@1 **0.198 vs 0.177**，category R@1 都是 0.844
- 结论：实例信号存在（约 9.5× 随机），但当前合成触觉的实例差异太弱，实例级训练没有提升
- 报告：`PHASE10_T1_INSTANCE_REPORT.md`；复现：`make t1-instance`

### T1 数据诊断

先判断原始特征有没有实例信号，再决定要不要训实例级投影：

```bash
python -m pss.rd2.scripts.t1_diagnostics --source expert_v1 --data data/expert_v1
# 或自己的 T1 npz：vision/tactile/geometry + object_id/category/session_id
python -m pss.rd2.scripts.t1_diagnostics --source npz --path data/t1/my_t1.npz
```

`data/expert_v1` 诊断：视觉 instance NN 0.771、触觉 0.188（gap 0.017）、几何 0.294；融合 0.703。决策 **partial**：融合能区分实例，但触觉几乎没有实例信号——这正是实例级 SupCon 没提升的原因。复现：`make t1-diagnostics`。

国内镜像：`download_smoke.py` 默认 `--endpoint https://hf-mirror.com`，并直接流式 HTTP 下载（绕过 `hf_hub_download` 的 xet 元数据校验）。实测镜像会重定向到海外 CDN，速度仍受出口带宽限制；大文件支持断点续传（`.part`）。

## 目录

```
pss/
  binding/       绑定算子（bind/unbind、槽生成）
  contracts/     v0 数据结构与版本常量
  demos/         三个基线 Demo
  encoders/      感官专家（视觉/触觉/几何，Phase 3 已实现）
  projection/    跨模态投影（Phase 4，占位）
  memory/        工作记忆 + 3D 场景图（Phase 6，占位）
  prototypes/    分层原型树（Phase 5，占位）
  reasoning/     关系 / 规划 / 冲突检测（Phase 7，占位）
  world_model/   实体状态世界模型（Phase 7，占位）
  heads/         文本 / 动作 / 价值输出头（Phase 8，占位）
  safety/        护栏与降级（Phase 9，占位）
  learning/      L1–L4 权限（Phase 9，占位）
  sim/           仿真场景、传感器、数据生成（Phase 2，已实现）
  eval/          指标、基准、基线（Phase 10，占位）
bench/           绑定稳定性基准与结果
contracts/       契约说明（人读）
tests/           单元测试
scripts/         一键运行脚本
```

## 依赖

- 核心：`numpy`
- 仿真：`mujoco`、`pillow` → `pip install -e ".[sim]"`
- 训练：`torch`、`scikit-learn` → `pip install -e ".[train]"`
- 测试：`pytest`
- 说明：当前未安装 torchvision/timm，视觉骨干为从零小 CNN；生产路径应换真实预训练骨干

## 契约要点（v0）

| 项 | 决定 | 原因 |
|---|---|---|
| 公共维度 | `D = 128`（可升 256/512） | 原文 4.8：32 维太危险，128/256 更稳 |
| 归一化 | L2，单位范数 | 余弦相似度才等价于向量距离 |
| 槽基向量 | Hadamard 行（D 为 2 的幂） | 两两正交、分量恒为 `1/√D`，解绑无近零放大 |
| 绑定 | `bind(f,s)=f⊙s`；`unbind=b/s` | 逐元素乘除，精确可逆 |
| 近零保护 | **保号** clamp：`sign(s)·max(|s|,eps)` | 原 Demo 会把负的近零分量翻正 |
| 跨模态存储 | **分模态独立存放，禁止相加** | 9.2.3：加法后无法拆回单模态 |
| 相加的用途 | 仅单模态相似度打分 | 对应 9.2.3 方案 B |
| 槽数 | 4 | 对应 4±1 工作记忆组块 |
| 绑定方案 | **Hadamard** | HRR 迭代发散，容量无优势（Phase 1） |
| 槽语义 | 工作记忆位置，非实体身份 | 换槽必须从 `raw` 重建 `bound` |
| 关系三元组 | `subj/rel/obj` 三个独立分量 + 一个仅供检索的 `fact` | 加法不可分离 |

完整说明见 `contracts/README.md`。

## 当前结果

运行 `python bench/binding_stability.py` 后查看 `bench/results/binding_stability.md`。核心预期：

- Hadamard / Rademacher 槽在噪声下误差稳定，Hadamard 槽间正交；
- Gaussian 槽因近零分量在除法时放大噪声，误差显著更大；
- 叠加容量随 K 增长而下降，D 越大越能撑住更多事实。

## 边界

Phase 0 的所有特征都是**人工定义**的，投影是**随机正交矩阵**，没有真实感官输入、没有学习到的对齐、没有动态追踪。它验证的是骨架链路与数学契约，不是完整能力（原文 6.5）。
