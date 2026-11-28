# PSS 问题重构：从“哪个模块到顶”到“几何边界在噪声下是否可靠”

> 日期：2026-11-28
> 上游：`docs/experiments/PSS-架构上界复核报告.md`、`PSS-架构上界复核-第二阶段.md`、`PSS-架构上界复核-第三阶段.md`
> 目的：把三轮实验之后真正剩下的问题写成**可证伪、有停止规则**的一页契约，停止继续加 U 系列 arm。
> 状态：D1–D5 已执行；Task 1–4（canonical / KDE / 多帧 / runtime 接入）已完成；canonical 类平衡确认已完成。结论：自然先验 Bayes 为 NO-GO（0.63–0.69）；**类平衡 Bayes（strength 0.5，训练 cal）在像素契约 0.848/0.962、canonical 3°+1% 0.871/0.957，GO**；压力档 σ=0.04 / 5°+2% 需多帧或 full-balance 的 overall 回退。最终决策见 `docs/experiments/PSS-Boundary-Decision-Memo.md`。

---

## 0. 一句话

**三轮实验不是“三次都没解决”，而是把问题从“哪个模块到顶”逐步缩小到了“几何边界在测量噪声下是否可靠”。** 覆盖问题已经有解（U2：组合补 1.000），clean 边界已经有解（U3d：几何骨架 α=0，absence 0.045），开集读出已经有解（U1b：双头 0.991/0.962）；**唯一还没有解的是 U6：clean-trained 硬 dead-zone 在传感器噪声下崩塌（worst-cell 0.252 / absence 0.743）**。所以下一阶段不是再跑一批 U，而是只回答一个问题：**一个噪声感知的边界层，能不能把 U6 的 worst-cell 从 0.70 抬到 0.80 以上，同时不伤害已经解决的三项？**

---

## 1. 诊断：为什么会有“打转”感

三轮实验的方法本身有硬的部分（预注册、负对照、同基质比较、失败保留），但程序层面有四个问题：

| 问题 | 表现 | 后果 |
|---|---|---|
| **分解模型不可证伪** | `f × g` 是解释性乘积；A/B 两线在 U3 之前不在同一基质上 | “瓶颈从 f 跑到 g”部分是任务切换的 artifact |
| **没有固定部署契约** | 合成网格、CLEVR、instance8、replay、噪声注入轮换 | “上界”随任务移动，永远追不上 |
| **程序级没有停止规则** | 每个 arm 有 bar，但整个 program 没有“何时停/换假设” | 看到异常就加新 arm，像打地鼠 |
| **把边界问题当模块问题** | 开集、absence、噪声分别修 | 同一个机制被拆成三次打补丁 |

关键不是“我们不够努力”，而是**没有先定义什么叫解决**。

---

## 2. 已经解决的（不要再回头打）

| 子问题 | 证据 | 状态 |
|---|---|---|
| **覆盖**：缺 pair type 导致 absence 未监督 | U2：single 0.419 vs combination 1.000，no-op 0.000 | **有解**：按 pair type 补数据 |
| **clean 边界**：embedding 丢 near-aligned 信息 | U3d：几何骨架 + α=0，`L_gap −0.002`、absence 0.045；U3e：位置编码通过，content gate 0.024 | **有解**：显式几何/位置通道做骨架 |
| **开集读出**：DINOv2 单头开集退化 | U1b：dual-head 0.991/0.962；CLIP 单头 0.997/0.995 | **有解**：闭集 LDA + 开集原型 |
| **长期增量**：删除/合并后的遗忘 | U4：refit_10 forgetting 0.013、rollback exact | **有解** |
| **审计**：rollback 截断日志 | U5b-audit：0.800 → 1.000 | **已修**（version 单调化未做） |
| **文本/VLM 读出** | U1c：文本零样本 0.292、dual 0.958/0.744 | **已排除**，不作为路线 |

这些是资产，不是失败。下一阶段不应再重新验证它们；只把它们当**守卫指标**（不能回退）。

---

## 3. 唯一未解决的问题

**U6：clean-trained 硬 dead-zone 骨架在传感器噪声下崩塌。**

| 噪声档 | worst-cell | absence error |
|---|---:|---:|
| clean | 0.933 | 0.043 |
| σ=0.005 | 0.830 | 0.162 |
| σ=0.01 | 0.692 | 0.297 |
| σ=0.02 | 0.418 | 0.582 |
| σ=0.04 | 0.252 | 0.743 |
| pose5°+depth2% | 0.516 | 0.484 |

机制已经定位：**dead-zone 边界穿越**；absence error 近似由 `σ_rel / eps` 控制（0.35→2.83）；overall 被 diagonal 托住，掩盖 near-aligned cell 的崩塌。深度噪声主导，位姿噪声几乎无害。

这就是本轮唯一要打的靶子。

---

## 4. 问题契约（一页）

### 4.1 任务

CLEVR-CoGenT 关系 grounding：给定两个对象，输出 `left/right/above/below` 四谓词，并正确处理 **absence**（两对象几乎对齐时，该轴谓词全为假）。

### 4.2 传感器契约

| 项 | 契约值 | 对应 U6 档 |
|---|---|---|
| 深度噪声 | ≤1% @1m | `3°+1%` |
| 位姿噪声 | ≤3° | `3°+1%` |
| 像素/坐标噪声 | ≤2 px（归一化后 σ≈0.01） | σ=0.01 |
| 压力档（不参与主 bar） | 5°+2%、σ=0.04 | 只报，不判 |

**主 bar 只在契约档（3°+1% / σ=0.01）上读。** 压力档用于看退化曲线，不作为通过条件。

### 4.3 主指标与守卫

| 类型 | 指标 | 定义 |
|---|---|---|
| **主指标** | `noise_boundary_worst_cell` | 契约噪声下，near-aligned cell 的 worst exact-4 |
| 守卫 1 | `clean_boundary_worst_cell` | clean 下同上；不得比 U3d 的 0.933 低 >2 pt |
| 守卫 2 | `coverage_worst_cell` | 未见 pair type；不得低于 U2 的 1.000 −2 pt |
| 守卫 3 | `open_set_auroc / reject` | 沿用 U1b dual-head；不得低于 0.99 / 0.95 |
| 守卫 4 | `overall exact-4` | 不得比 baseline 低 >2 pt |
| 守卫 5 | `p95 latency` | ≤10 ms（单对，CPU/GPU 一致口径） |

### 4.4 Baseline 与 target

| 项 | 值 |
|---|---|
| Baseline | U6 的 clean-trained 硬 dead-zone 骨架，在契约档 worst-cell **0.701** |
| Target | 契约档 worst-cell **≥0.80**（+9.9 pt），overall ≥0.95，abstention precision ≥0.90 |
| 负对照 | capacity-matched 硬层 + noise augmentation；若它追平，增益不是“不确定性”带来的 |

### 4.5 失败代价

在 absence 上**假阳性**（把对齐对判成 above/below）比**弃权**更糟。因此允许弃权；弃权不算错，但要有校准的 selective accuracy / coverage 曲线。

### 4.6 预算与停止

- 5 个工作日、1 张 GPU、3 seeds。
- 停止规则见 §7；到点必须做 go/no-go，不允许“再跑一个 arm 看看”。

---

## 5. 统一假设 H

> **H：U6 的失败不是“模型不够大”，而是“边界决策没有把测量不确定性带进来”。一个 σ-aware 的软/概率化 dead zone + 弃权，应当在契约噪声下把 near-aligned worst-cell 抬到 ≥0.80，同时不伤害 clean / coverage / open-set。**

拆成可证伪的子假设：

| 子假设 | 内容 | 证伪预测 | 若成立 |
|---|---|---|---|
| **H1 单调性** | 失败由 `σ_rel/eps` 控制 | absence error 随 σ_rel/eps 单调升（U6 已初步支持） | 把 σ/eps 作为控制变量 |
| **H2 共同修复** | 同一个 σ-aware 边界层同时改善 clean / noise 两个 boundary cell | 只改善一个、另一个不动或变差 | H2 假，退回轴特定修复 |
| **H3 非容量** | 增益不是容量/数据增强 | capacity-matched 硬层 + noise augmentation 追平 | 增益归因于数据/容量，不宣称不确定性 |
| **H4 可校准** | 一个 τ(σ) 能在不同噪声档间迁移 | 每个噪声档都要单独调 τ 才有效 | 需要 σ 估计 + 条件校准 |

**注意**：H 只覆盖 boundary + noise。**coverage 不在这条假设里**——U2 已经证明它靠补数据，不靠不确定性层。不要把 coverage 的守卫失败误读成 H 失败。

---

## 6. 一个端到端基准（PSS-Boundary）

### 6.1 固定数据与划分

- 训练：CLEVR `trainA` 30k pairs。
- clean 测试：`valA` 10k pairs。
- coverage 测试：held-out pair type（horizontal-only 训练，测 vertical/diagonal），复用 U2 协议。
- open-set 测试：held-out 类/组合（复用 U1b 的 held split）。
- 噪声测试：`valA` 同一批 pairs，注入契约档噪声（3°+1% / σ=0.01）与压力档（5°+2% / σ=0.04）。

### 6.2 一个主指标，四个守卫

```
primary = noise_boundary_worst_cell @ contract noise
guards  = clean_boundary, coverage, open_set, overall, latency
```

报告必须同时给：primary、guards、per-cell 表、`σ_rel/eps` 曲线、abstention 曲线。**不允许只报 overall。**

### 6.3 臂

| arm | 作用 |
|---|---|
| `hard_clean` | U6 baseline（clean-trained 硬 dead zone） |
| `hard_noise_aug` | 负对照：同容量硬层 + 噪声增强训练 |
| `soft_deadzone` | 主候选：3-way softmax over {left, right, neither}，τ 随 σ 校准 |
| `soft_deadzone_abstain` | 主候选 + 弃权；报 selective accuracy/coverage |
| `oracle_sigma` | 诊断：已知真实 σ 时的上界 |
| `estimated_sigma` | 诊断：从深度/位姿估计 σ，再喂给 τ(σ) |

---

## 7. 干预：最小实现

### 7.1 软 dead zone

把硬谓词 `1[dx > eps]` 换成对 `{left, right, neither}` 的 3-way softmax：

```
logit_left  = (dx - eps) / τ
logit_right = (-dx - eps) / τ
logit_neither = 0
p = softmax([logit_left, logit_right, logit_neither])
```

- `τ` 由噪声水平决定：`τ = max(τ0, k·σ_rel)`。
- `σ_rel` 从深度/位姿误差传播估计；没有估计时用契约档常数。
- 弃权：`max(p) < threshold` 时 abstain；threshold 在 calibration split 上按 selective accuracy 选。

### 7.2 校准纪律

- calibration split 与 test 分离；test 只报一次。
- τ 和 abstain threshold 都在 calibration 上选；不允许 test 调参。
- 每个噪声档单独报“共享 τ”和“按 σ 条件化 τ”两种结果。

### 7.3 不做的事

- 不换编码器、不加维度、不加读出容量。
- 不做 per-cell 补丁；只做一个边界层。
- 不用 test 选 τ；不因为结果不好看调 bar。

---

## 8. 停止规则（程序级，先定）

| 观测 | 结论 | 动作 |
|---|---|---|
| 契约档 worst-cell 提升 ≥5 pt 且 guards 不回退 | H2 成立 | 把 soft dead zone 设为默认；进入真实 RGB-D 回放 |
| 提升 <5 pt，但 clean/coverage 回退 ≤2 pt | H2 部分成立 | 只保留 σ 条件化，查 H4 |
| capacity-matched 硬层追平 | H3 假 | 增益归因于数据/容量；不宣称不确定性 |
| 只改善 clean、不改善 noise | H2 假 | 停止加层；做信息论 ceiling |
| 信息论 ceiling 在契约档低于 0.80 | 传感器契约不可行 | 建议升级传感器/多帧融合，不再调模型 |
| 任何 guard 回退 >2 pt | 干预有代价 | 回滚；记录 trade-off，不掩盖 |

**到第 5 个工作日必须做 go/no-go，不允许新增 U 系列 arm。**

---

## 9. 交付物

```
docs/plans/PSS-问题重构.md                     # 本文件
pss/reasoning/relations/soft_deadzone.py        # 软/概率谓词 + 弃权
pss/eval/scripts/run_boundary_bench.py          # 一个基准，五个臂
tests/test_soft_deadzone.py
tests/test_boundary_bench.py
eval/results/boundary_bench/                    # 主表、σ/eps 曲线、abstention 曲线
docs/experiments/PSS-Boundary-Report.md         # go/no-go 报告
```

复现入口（计划）：

```bash
python -m pss.eval.scripts.run_boundary_bench \
  --contract-noise 0.01 --stress-noise 0.04 --seeds 0 1 2 --device cuda
```

---

## 10. 时间盒

| 天 | 内容 | 出口 |
|---|---|---|
| D1 | 固定契约、划分、baseline 复现（hard_clean = U6） | 主指标与 guards 可计算 |
| D2 | soft dead zone + τ(σ) + 弃权 | smoke 通过 |
| D3 | 5 臂 × 3 seeds × 契约/压力档 | 主表 |
| D4 | 消融：capacity-matched、oracle/estimated σ | H2/H3 判定 |
| D5 | go/no-go 报告 | 决定：默认化 / 转信息 ceiling / 换传感器 |

---

## 11. 一句话

**三轮实验已经把问题从“哪个模块到顶”缩小到“几何边界在测量噪声下是否可靠”；覆盖、clean 边界、开集、增量、审计都已有解，唯一未解的是 U6 的 noise boundary。下一步只做一个 σ-aware 软 dead zone，在一个固定基准上同时看 primary 和四个守卫；5 天内必须给出 go/no-go，不再加新 arm。**
