# Phase 4/5 验证报告

> 日期：2026-11-22
> 数据：`data/category_pretrain`（1920 views；train 1152 / val 192 / test 192 / test_category 384）
> 专家：R0 主线（`vision_smallcnn_v0` / `tactile_mlp_v0` / `geometry_handcrafted_v0`）
> 产物：`checkpoints/phase45_projection/`、`checkpoints/phase45_prototypes/`
> 详细结果：`eval/results/phase45/PHASE4_5_REPORT.md`

---

## 1. 门禁结果

| 门禁 | 阈值 | 结果 | 判定 |
|---|---:|---:|---|
| P4 跨模态 pair R@1 | ≥ 0.70 | 0.026–0.073 | **FAIL** |
| P4 属性探针保持率 | ≥ 90% | 110.2% | **PASS** |
| P4 材质碰撞率（vision+tactile） | ≤ 5% | **4.7%** | **PASS** |
| P4 锚点重训脚本 | 可运行 + 冻结产物不变 | vision R@1 0.906 / retention 96.1% / frozen=True | **PASS** |
| P5 闭集 top-1 | ≥ 0.80 | **0.979** | **PASS** |
| P5 开放集 AUROC | ≥ 0.90 | **0.993** | **PASS** |
| P5 树候选召回 | ≥ 99% | 85.4% | **FAIL** |
| P5 识别延迟 p95 | < 50 ms | 0.335 ms | **PASS** |
| P5 材质混淆率 | < 2% | **1.9%** | **PASS** |
| P5 原型包增量/回滚 | 可用 + 无反向传播 | add=True / rollback=v1 / no_backprop=True | **PASS** |

---

## 2. 本轮改了什么

### 2.1 材质感知投影

在 Phase 4 投影里加入**类内材质对比 + cosine margin**：

- 材质 bin：hardness / roughness 各按训练集三分位切成 3 档，组合成 9 个材质标签。
- 正样本：同类别、同材质 bin。
- 负样本：同类别、不同材质 bin。
- 额外 margin：同类异材质对的 cosine 超过 0.5 时直接惩罚。
- 只作用于 **vision / tactile**；geometry 负责尺寸和形状，不强制它编码材质。
- 获胜配置：`material-weight=1.0`、`batch-size=256`、`epochs=120`。

效果：

| 指标 | 改前 | 改后 |
|---|---:|---:|
| 材质碰撞率（vision+tactile） | 39.1% | **4.7%** |
| vision 碰撞率 | 68.9% | **6.6%** |
| tactile 碰撞率 | 10.0% | **2.8%** |
| 材质混淆率（平均） | 2.3% | **1.9%** |
| 闭集 top-1 | 0.969 | **0.979** |
| 开放集 AUROC | 0.960 | **0.993** |

### 2.2 冲突检测修复

- 旧测试注入 tactile 冲突，但 tactile 权重为 0 或低于冲突阈值，冲突率始终 0.120。
- 改为注入 vision / geometry，并把冲突阈值提到 0.75：
  - baseline **0.100**
  - vision injected **1.000**
  - geometry injected **0.980**
- 冲突规则本身工作正常，之前是测试设计问题。

### 2.3 Admission 校准

- `pre_review` 支持按模态传入阈值。
- 阈值来自 `calibrated_thresholds`：每模态 0.5 × 最近邻距离。
- 新增 holdout accuracy 门禁：`accept` 要求 holdout acc ≥ 0.8，否则降级为 `review`。
- 当前结果：

| 类别 | 决策 | 原因 | holdout acc |
|---|---|---|---:|
| cloth | reject | variance_high:vision | 1.000 |
| sheet | accept | — | 1.000 |
| sponge | accept | — | 0.979 |
| tool | reject | inter_distance_low:tactile | 1.000 |

### 2.4 锚点重训脚本

- 新增 `pss/projection/anchor.py` 和 `pss/projection/scripts/anchor_retrain.py`。
- 冻结其他专家、其他投影头、原型库；只训新模态的投影头，对齐冻结锚点。
- 冻结产物在训练前后做 SHA256 校验，必须 byte-identical。
- 结果：

| 替换模态 | test category R@1 | 属性保持 | prototype drift | 冻结产物不变 |
|---|---:|---:|---:|---|
| vision | 0.906 | 96.1% | 0.874 | True |
| tactile | 0.208 | 100.4% | 0.964 | True |

- 结论：锚点协议可运行。真正替换专家时，先跑 `make anchor-retrain-vision` 或 `anchor-retrain-tactile`，再跑 `make phase45`。

### 2.5 原型包增量/回滚

- 复用 `pss/learning/l2_package` 的 signed / versioned / rollback 包机制。
- 新增 `pss/prototypes/package.py` 和 `pss/prototypes/scripts/package_ops.py`。
- 从当前 16 类 bank 初始化 `v1`；用 `sheet` 的候选样本走 admission，生成 `v2`；holdout 1.0；回滚到 `v1`。
- 冻结专家、投影、原型 bank 的 SHA256 前后一致，证明没有反向传播。
- 结果：

| 项 | 结果 |
|---|---|
| 初始类别 | 16（v1） |
| 增量后类别 | 17（v2，新增 sheet） |
| admission | accept |
| holdout acc | 1.0 |
| 回滚 | v2 → v1，integrity True |
| 冻结产物哈希不变 | True |
| 判定 | **PASS** |

### 2.6 实例级视角 holdout（方案 A）

- 不采真实 RD2，用现有 4 个同步视角做 train views 0–1 / test views 2–3。
- 在 `phase45_projection` 基础上做实例级微调（instance + category + material 损失）。
- 结果：

| pair | baseline R@1 | fine-tuned R@1 | fine-tuned R@5 |
|---|---:|---:|---:|
| vision-vision | 0.334 | **0.380** | 0.710 |
| vision-tactile | 0.008 | 0.009 | 0.050 |
| vision-geometry | 0.019 | 0.031 | 0.130 |
| tactile-tactile | 0.018 | 0.015 | 0.089 |
| tactile-geometry | 0.004 | 0.004 | 0.028 |
| geometry-geometry | 0.083 | 0.086 | 0.208 |

- 结论：vision 有弱视角不变性（0.380），但跨模态实例检索几乎不可学（0.004–0.031）。
- 原因不是投影，是当前合成 tactile/geometry 没有实例级身份信号：tactile 是材质代理，geometry 是尺寸/形状，同类同材质的实例可能完全一样。
- 产物：`checkpoints/instance_view/`、`eval/results/instance_view/INSTANCE_VIEW_REPORT.md`。

#### 2.6.1 仿真器 v2：实例级触觉/几何

- 几何：各轴独立尺寸抖动；触觉：实例级稀疏纹理 + ±1 接触偏移。
- 同/跨实例分离度：geometry within 0.127 / across 0.391；tactile within 0.775 / across 0.732。
- 在新数据上重训专家与类别投影，再跑 instance-view：

| 协议 | vision-vision | vision-tactile | vision-geometry | geometry-geometry |
|---|---:|---:|---:|---:|
| test-test R@1 | 0.407 | 0.021 | 0.032 | 0.105 |
| query→gallery R@1 | **0.548** | **0.120** | **0.081** | **0.184** |
| query→gallery R@5 | 0.866 | 0.346 | 0.277 | 0.397 |

- 对比旧数据：vision-tactile QG R@1 0.025 → **0.120**（约 5×），但离 0.7 还远。
- 结论：实例信号加进去了，瓶颈转为**视角不变性 / 每实例视角数**；下一步增加到 8 视角/实例，或更强视角增强。
- 产物：`data/category_pretrain_instance/`、`checkpoints/instance_view_v2/`、`eval/results/instance_view_v2/`。

#### 2.6.2 仿真器 v3：8 视角/实例

- 每实例 8 视角，train views 0–3 / test views 4–7；train/test 各 1920 views。
- 结果：

| 协议 | vision-vision | vision-tactile | vision-geometry | geometry-geometry |
|---|---:|---:|---:|---:|
| test-test R@1 | 0.665 | 0.037 | 0.062 | 0.157 |
| query→gallery R@1 | **0.728** | **0.148** | **0.108** | **0.218** |
| query→gallery R@5 | 0.942 | 0.391 | 0.334 | 0.487 |

- vision-vision QG 0.728，过了 0.7；但 vision-tactile 0.148、vision-geometry 0.108，未达到预设的 0.3 停止线。
- 结论：增加视角数只提升了 vision 自身的视角不变性；跨模态仍上不去。原因是各模态的实例信号是**独立随机**的：tactile 纹理由 instance_id 单独生成，RGB/geometry 没有共享同一个实例 latent，模型没有可学的跨模态对应关系。
- 按预设停止标准：**停止仿真调参**，转向 ObjectFolder 2.0 或最小真实 RD2；如果继续仿真，需要做“共享实例 latent”（同一纹理/几何同时驱动 RGB、depth、tactile）。

### 2.7 ObjectFolder 2.0（100/100 对象）

- 许可 **CC-BY-4.0**；前 100 个 ObjectFile 已下载并解压到 `_external/ObjectFolder/ObjectFolder1-100/`。
- 每个对象渲染 2 vision + 4 touch；100 个对象已全部渲染（分块渲染规避内存不足）。
- DINOv2 零样本 + Ridge 半对象映射：

| 协议 | vision→touch R@1 | R@5 | touch→vision R@1 | R@5 | chance |
|---|---:|---:|---:|---:|---:|
| zero-shot | 0.010 | 0.120 | 0.010 | 0.080 | 0.010 |
| Ridge half-split | 0.080 | 0.260 | 0.040 | 0.220 | 0.020 |

- R@5 仍明显高于随机：零样本 12× / 8×，Ridge 13× / 11×。
- 但零样本 R@1 在 100 对象上回到随机（0.010）；Ridge 把 vision→touch R@1 提到 0.080（4× chance）。
- 结论：ObjectFolder 的共享实例 latent 是真实的，但 DINOv2 零样本不足以做 top-1；需要训练触觉编码器或非线性跨模态投影，并增加每对象触觉采样。40 对象的初步结果偏乐观，100 对象更保守、R@5 信号稳定。
- 产物：`eval/results/objectfolder/OBJECTFOLDER_R1.md`。

#### 2.7.1 触觉编码器（DINOv2 vision 锚点 + object-level InfoNCE）

- 训练一个小 CNN，把每个对象的 4 张 touch 图映射到 DINOv2 vision 锚点空间。
- 损失：object-level 对称 InfoNCE + 对象分类 CE。
- 结果：

| 协议 | 方向 | R@1 | R@5 |
|---|---|---:|---:|
| seen（100 对象，训练=测试） | vision→touch | 0.360 | 0.750 |
| seen | touch→vision | 0.160 | 0.460 |
| held-out 5 折 | vision→touch | **0.220 ± 0.121** | **0.580 ± 0.136** |
| held-out 5 折 | touch→vision | 0.100 ± 0.032 | 0.400 ± 0.071 |

- 对比 Ridge：vision→touch R@1 0.080 → **0.220**（held-out），R@5 0.260 → **0.580**。
- 非对称：touch→vision 只有 0.100，因为只训了触觉编码器，vision 侧仍是冻结 DINOv2；下一步加 vision 投影头或共享空间。
- 结论：触觉编码器把 object-level 跨模态信号从“接近随机”拉到“明显高于随机”，但离 0.7 还远；需要对称投影、更多触觉采样、更多对象。
- 产物：`eval/results/objectfolder_tactile/TACTILE_ENCODER_REPORT.md`、`checkpoints/objectfolder_tactile/tactile_encoder.pt`。

#### 2.7.2 对称投影

- 在触觉编码器之外加一个可训练的 vision projector（冻结 DINOv2 anchor），用对称 object-level InfoNCE 训练。
- 结果：

| method | held-out vision→touch R@1 | R@5 | touch→vision R@1 | R@5 |
|---|---:|---:|---:|---:|
| tactile-only | **0.220 ± 0.121** | 0.580 ± 0.136 | 0.100 ± 0.032 | 0.400 ± 0.071 |
| symmetric | 0.180 ± 0.060 | **0.620 ± 0.040** | **0.140 ± 0.037** | **0.480 ± 0.108** |

- 对称投影把 touch→vision R@1 从 0.100 提到 **0.140**，R@5 从 0.400 提到 **0.480**。
- 但 vision→touch R@1 从 0.220 降到 **0.180**，R@5 从 0.580 升到 **0.620**。不是全面胜出，fold 方差仍大。
- 结论：瓶颈不在单边/双边，而在数据量：100 对象 × 4 触觉点。下一步增加每对象触觉点（8–12）和对象数（200–300），或换触觉预训练模型。
- 产物：`eval/results/objectfolder_symmetric/SYMMETRIC_REPORT.md`、`checkpoints/objectfolder_symmetric/`。

#### 2.7.3 增加每对象触觉采样（4 → 12）

- 只补触觉渲染，不重渲 vision；100 对象共 1200 张 touch 图。
- 重训触觉编码器和对称投影：

| method | touches | held-out vision→touch R@1 | R@5 | touch→vision R@1 | R@5 |
|---|---:|---:|---:|---:|---:|
| tactile-only | 4 | 0.220 ± 0.121 | 0.580 ± 0.136 | 0.100 ± 0.032 | 0.400 ± 0.071 |
| tactile-only | 12 | 0.220 ± 0.051 | **0.730 ± 0.117** | 0.100 ± 0.045 | **0.520 ± 0.112** |
| symmetric | 4 | 0.180 ± 0.060 | 0.620 ± 0.040 | **0.140 ± 0.037** | 0.480 ± 0.108 |
| symmetric | 12 | **0.240 ± 0.058** | 0.620 ± 0.117 | 0.100 ± 0.032 | 0.430 ± 0.051 |

- 12 触觉点主要提升 R@5 和稳定性：tactile-only vision→touch R@1 std 从 0.121 降到 0.051，R@5 从 0.580 升到 0.730。
- 但 top-1 只从 0.220 到 0.240（symmetric）；touch→vision 没有提升。
- 结论：触觉覆盖不是 top-1 的唯一瓶颈；下一步增加对象数（200–300）或换更强触觉编码器，或把目标改成 top-5 + 重排。
- 产物：`eval/results/objectfolder_tactile_t12/`、`eval/results/objectfolder_symmetric_t12/`、`checkpoints/objectfolder_*_t12/`。

---

## 3. 仍然没做到的

### 3.1 实例级跨模态 R@1

- 结果：0.026–0.073，远低于 0.7。
- 原因：当前数据只有类别级配对；实例级同步视触数据是 RD2 的交付物，尚未采集。
- 方案 A（模拟视角 holdout）进一步证明：旧数据 vision-vision 0.380，但跨模态只有 0.009–0.031。
- 仿真器 v2（实例级触觉/几何）把 vision-tactile QG R@1 提到 0.120，vision-vision QG 0.548。
- 仿真器 v3（8 视角/实例）把 vision-vision QG 提到 **0.728**，但 vision-tactile 0.148、vision-geometry 0.108，仍未到 0.3 停止线。
- 这说明瓶颈不只是视角数：各模态的实例信号是独立随机的，RGB/tactile/geometry 之间没有共享的实例 latent。**Phase 4 的 pair R@1 保持 deferred**，不阻塞 Phase 6/7；对外不能声称 sim2real。
- 按预设停止标准：停止仿真调参；下一步 ObjectFolder 2.0（CC-BY-4.0，object-level 多模态）或最小真实 RD2。如果继续仿真，必须让同一实例纹理/几何同时驱动 RGB、depth、tactile。
- ObjectFolder 2.0（100 对象）：零样本 vision→touch R@1 0.010 / R@5 0.120，touch→vision 0.010 / 0.080（chance 0.010）；Ridge 半对象 R@1 0.080 / 0.040，R@5 0.260 / 0.220（chance 0.020）。共享 latent 存在，但零样本 R@1 不够；下一步训练触觉编码器/非线性投影。
- ObjectFolder 触觉编码器（held-out 5 折）：vision→touch R@1 **0.220 ± 0.121** / R@5 **0.580 ± 0.136**，touch→vision 0.100 ± 0.032 / 0.400 ± 0.071；比 Ridge 明显提升，但仍未到 0.7。下一步对称投影/更多触觉采样。
- ObjectFolder 对称投影（held-out 5 折）：vision→touch R@1 0.180 ± 0.060 / R@5 0.620 ± 0.040，touch→vision R@1 **0.140 ± 0.037** / R@5 **0.480 ± 0.108**。对称投影提升了 touch→vision，但 vision→touch R@1 略降；瓶颈转为数据量（100 对象 × 4 触觉点）。
- ObjectFolder 12 触觉点：tactile-only vision→touch R@1 0.220 ± 0.051 / R@5 **0.730 ± 0.117**；symmetric vision→touch R@1 **0.240 ± 0.058** / R@5 0.620。R@5 和稳定性明显提升，但 top-1 只到 0.24；触觉覆盖不是唯一瓶颈。

### 3.2 层级树召回

- 结果：树候选召回 85.4%，低于 99%。
- 原因：16 类小 bank 上二分类树过剪枝；树是为大规模类别设计的。
- 当前正确做法：**Phase 5 使用 flat 检索**，树不要上线。

### 3.3 Geometry 碰撞率

- geometry 同类碰撞率 33.3%，但它负责尺寸/形状，不是材质。本轮门禁只算 vision+tactile。
- 如果后续要求 geometry 也能区分同类的形状子模式，需要单独的 geometry 子原型或属性头。

---

## 4. 下一步

1. **RD2 种子集**：20–50 个物体，500–2000 条同物同刻同步 episode；采到后做实例级投影微调，重跑 pair R@1。
2. **Phase 5 固定 flat**：树召回 85.4% 不达标，不上线；重建树时以 recall 优先，保留 flat fallback。
3. **R-Gate**：RD2 到位后，用真实 RGB-D/触觉复测 DINOv2 vs SmallCNN、D vs R0。
4. **新增类别后的固定流程**：admission → `make phase5-package` → `make phase45`。
5. **换专家后的固定流程**：`make anchor-retrain-vision`（或 tactile）→ `make phase45`。

---

## 5. 复现

```bash
# Phase 4：材质感知投影（获胜配置）
python -m pss.projection.scripts.train_projections \
  --data data/category_pretrain \
  --experts checkpoints/category_pretrain_experts \
  --out checkpoints/phase45_projection \
  --epochs 120 --batch-size 256 --temp 0.07 --seed 0 --material-weight 1.0

# Phase 5：原型与识别
python -m pss.prototypes.scripts.build_prototypes \
  --data data/category_pretrain \
  --experts checkpoints/category_pretrain_experts \
  --projections checkpoints/phase45_projection \
  --out checkpoints/phase45_prototypes \
  --seed 0

# 验收
python -m pss.eval.scripts.run_phase45 \
  --data data/category_pretrain \
  --experts checkpoints/category_pretrain_experts \
  --projections checkpoints/phase45_projection \
  --prototypes checkpoints/phase45_prototypes \
  --out eval/results/phase45

# 锚点重训（换专家后）
python -m pss.projection.scripts.anchor_retrain \
  --data data/category_pretrain \
  --experts checkpoints/category_pretrain_experts \
  --projections checkpoints/phase45_projection \
  --prototypes checkpoints/phase45_prototypes \
  --modality vision \
  --out checkpoints/anchor_retrain_vision \
  --epochs 60

# 原型包增量/回滚
python -m pss.prototypes.scripts.package_ops --action run-all \
  --category sheet \
  --root checkpoints/prototype_packages \
  --out eval/results/phase5_package
```

---

## 6. 边界

- 数据是 MuJoCo 低保真渲染 + 合成触觉，不是真实台面。
- 类别级配对，不是实例级同步配对。
- 触觉专家是 R0 合成代理，不是真实触觉。
- 所有阈值是初值；材质碰撞/混淆是本轮新增的代理门禁，仍需真实数据校准。
- 结论只用于架构验证，不作为能力声明。
