# PSS-Boundary：D1–D5 报告（噪声边界）

> 日期：2026-11-28
> 上游：`docs/plans/PSS-问题重构.md`
> 结果目录：`eval/results/boundary_bench/`
> 状态：D1–D5 已执行。**全局温度 soft dead zone 是 NO-GO；修正后的 empirical-Bayes 边界在契约噪声下 GO。**
>
> **更正（2026-11-28，多帧 Task 复核）**：§3.3 的 0.948 用的是**干净测试坐标**查表，口径错误。给观测按对象加噪后，同一张 grid Bayes 表的契约档 near-worst 是 **0.634**，不是 0.948；stress 档 0.000。因此本报告的 **GO 暂标为“待复核”**，最终结论以 `PSS-Boundary-Phase4.md` 为准：自然先验 0.634；**类平衡 Bayes（strength 0.5，训练 cal 选出）0.848 / overall 0.962 → GO**。0.948 只能作为“干净输入 + 噪声表”的诊断，不能作为契约结果。

---

## 0. 一句话

**问题不是“模型不够大”，而是“边界决策没有用上局部先验”。** 全量基准里，clean-trained 硬骨架在契约噪声下 near-worst 只有 **0.685**，全局温度 soft 是 **0.682**（NO-GO）；信息论 ceiling 显示同一份噪声下 Bayes 可达 **0.971**，说明信息没丢、是决策规则太粗。改成“经验先验 + 高斯噪声似然”的 grid Bayes 边界后，契约档 near-worst 到 **0.948**（clean 1.000，H3 负对照 0.645），预注册通过。压力档 σ=0.04 仍崩（near 0.109），这是下一步的边界，不是契约要求。

---

## 1. D1：契约与基准

| 项 | 固定值 |
|---|---|
| 任务 | CLEVR 关系 grounding：left/right/above/below + absence |
| 训练/测试 | trainA 30k pairs / valA 10k pairs |
| 契约噪声 | 坐标 σ=0.01（代理 3°+1% depth/pose） |
| 压力噪声 | σ=0.04（只报不判） |
| dead zone | eps=0.02 |
| 主指标 | 契约档 near-aligned worst-cell = min(horizontal, vertical) exact-4 |
| 守卫 | clean 边界 ≥0.913；coverage=1.000；open-set 0.9915/0.9618；overall ≥ baseline−0.02；latency ≤10 ms/1000 |
| baseline | U6 clean-trained 硬骨架 |
| target | 契约档 near-worst ≥0.80 |

---

## 2. D2–D3：全局温度 soft dead zone（失败）

### 2.1 主表（3 seeds，mean）

| arm | clean near | contract near | stress near |
|---|---:|---:|---:|
| `hard_clean` | 0.959 | **0.685** | 0.236 |
| `hard_noise_aug` | 0.906 | 0.645 | 0.212 |
| `soft_fixed` | 1.000 | 0.682 | 0.238 |
| `soft_oracle` | 1.000 | 0.682 | 0.288 |
| `soft_estimated` | 1.000 | **0.682** | 0.238 |

### 2.2 失败原因

- 校准在 noisy cal 上大多选了 `k=0`（seed 0/2），即 τ 没有随 σ 放宽；seed 1 选 `k=1`，结果仍是 0.682。
- 一个全局 τ 只能把边界整体“模糊化”，不能表达**局部先验**：在坐标分布密集处和稀疏处，同一个 τ 给出的后验完全不同。
- **smoke 的 0.853 是假象**：smoke 只训练 20 epochs，hard baseline 只有 0.375；全量 hard baseline 是 0.685，soft 的优势消失。这是一个方法教训：**go/no-go 不能用 smoke 的欠训练 baseline**。

### 2.3 预注册

`u_boundary_soft`（soft_estimated）0.682 < 0.80 → **NO-GO**。bar 未调。

---

## 3. D4：信息论 ceiling 与修正干预

### 3.1 Ceiling

用训练对的经验先验 + 高斯噪声似然，计算 Bayes exact-4：

| σ | σ/eps | Bayes exact-4 |
|---:|---:|---:|
| 0.005 | 0.25 | 0.975 |
| **0.01** | **0.50** | **0.971** |
| 0.02 | 1.00 | 0.950 |
| 0.04 | 2.00 | 0.918 |

**结论**：契约档的信息上限是 0.971，target 0.80 完全可达；0.682 是决策规则的问题，不是合同不可行。

### 3.2 修正干预：grid Bayes 边界

- 先验：训练对 `(dx,dy)` 的 257×257 直方图。
- 对每个观测格，计算后验均值/方差（先验 × 高斯似然的高斯近似），再用 probit 得到 `{positive, negative, neither}` 三态概率。
- 无学习参数；表按 σ 预计算并缓存，查表 O(1)。

### 3.3 结果

| level | overall | horizontal | vertical | diagonal | near worst |
|---|---:|---:|---:|---:|---:|
| clean | 1.000 | 1.000 | 1.000 | 1.000 | **1.000** |
| **contract σ=0.01** | **0.997** | 0.970 | 0.948 | 0.999 | **0.948** |
| stress σ=0.04 | 0.955 | 0.143 | 0.109 | 1.000 | 0.109 |

- `u_boundary_soft`（Bayes）0.948 ≥ 0.80 → **通过**。
- H3 负对照：`hard_noise_aug` 0.645；Bayes − control = **+0.303** ≥ 0.05 → H3 通过（增益不是容量/数据增强）。
- Guards：clean 1.000 ≥0.913；overall 0.997 ≥ baseline 0.968−0.02；coverage/open-set 常数通过；latency：查表 **0.05–0.2 ms/1000**，建表约 18 ms/σ（启动时预计算）。

---

## 4. D5：Go / No-Go

### 4.1 结论

| 干预 | 契约档 near-worst | 判定 |
|---|---:|---|
| 全局温度 soft dead zone | 0.682 | **NO-GO** |
| **empirical-Bayes 边界** | **0.948** | **GO** |

### 4.2 停止规则检查

| 规则 | 结果 |
|---|---|
| primary ≥0.80 | ✅ 0.948 |
| guards 不回退 >2 pt | ✅ |
| H3（soft − hard_noise_aug ≥0.05） | ✅ +0.303 |
| ceiling ≥ target | ✅ 0.971 |
| 压力档 | ❌ 0.109（契约不要求；记为 residual） |

### 4.3 结论

**契约档 GO：正确的架构形态不是“学一个更强的头”，而是“显式几何 + 局部先验 + 噪声似然”的边界层。** 全局温度不够；经验 Bayes 够。

> **更正**：上面的 GO 基于干净输入口径（0.948）。按对象加噪后自然先验 grid Bayes 是 0.634；KDE 0.632；canonical 3°+1% 0.693。最终 GO 来自**类平衡 Bayes**（类平衡 strength 0.5，训练 cal 选出）：near **0.848**、overall **0.962**。详见 `PSS-Boundary-Phase4.md`。不要把 0.948 当作契约结果引用。

---

## 5. 失败与残余

1. **全局温度 soft dead zone 失败**：一个标量不能表达局部先验；H2 假。
2. **压力档 σ=0.04 失败**：Bayes 高斯近似在 σ/eps=2 时不够；near-worst 0.109，overall 被 diagonal 托到 0.955。需要更细的后验（KDE/mixture）或多帧滤波。
3. **契约是像素坐标代理**：σ=0.01 不是真实 RGB-D 标定；U6 的 canonical 3°+1% 档 baseline 0.701，需要在同一 Bayes 层上重跑。
4. **先验漂移**：Bayes 层依赖训练对分布；部署换场景需要重估先验。
5. **smoke 误导**：20-epoch baseline 造成 0.853 的假 GO；全量纠正为 0.682。以后 go/no-go 只认全量 baseline。

---

## 6. 产物与复现

```
pss/reasoning/relations/soft_deadzone.py       # 全局温度 soft（失败臂）
pss/reasoning/relations/bayes_boundary.py      # 修正干预：grid Bayes
pss/eval/scripts/run_boundary_bench.py         # D1-D3 主基准
pss/eval/scripts/run_boundary_ceiling.py       # D4 ceiling
pss/eval/scripts/run_boundary_bayes.py         # D4 修正干预
tests/test_soft_deadzone.py
tests/test_bayes_boundary.py
eval/results/boundary_bench/                   # 主表、ceiling、bayes
```

复现：

```bash
python -u -m pss.eval.scripts.run_boundary_bench \
  --out eval/results/boundary_bench --device cuda \
  --train-pairs 30000 --test-pairs 10000 --epochs 300 --seeds 0 1 2

python -u -m pss.eval.scripts.run_boundary_ceiling \
  --out eval/results/boundary_bench/ceiling.json

python -u -m pss.eval.scripts.run_boundary_bayes \
  --out eval/results/boundary_bench/bayes
```

---

## 7. 下一步

1. **把 Bayes 边界接到 U6 canonical 3D 噪声**（3°+1% / 5°+2%），确认不是像素代理的特例。
2. **压力档用 KDE/mixture 后验**，替代高斯近似；目标 stress near ≥0.60。
3. **多帧滤波/深度融合**：σ 降到 0.005 以下，Bayes 上限 0.975。
4. **接入 runtime**：把 Bayes 边界作为 `RelationFactLayer` 的默认几何谓词，保留 hard 作为可审计 fallback。
5. **真实 RGB-D 回放**：有数据后替换像素代理，重跑同一契约。

---

## 8. 一句话

**全局温度是 NO-GO，经验 Bayes 边界是 GO：契约档 near-worst 从 0.685 抬到 0.948，clean 1.000，H3 负对照 0.645。** 这不是“模型更大”，而是“边界决策终于用上了局部先验和噪声似然”。
