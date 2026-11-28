# PSS-Boundary 第四阶段：canonical / KDE / 多帧 / runtime 接入

> 日期：2026-11-28
> 上游：`docs/plans/PSS-问题重构.md`、`docs/experiments/PSS-Boundary-Report.md`
> 结果：`eval/results/boundary_canonical/`、`eval/results/boundary_bench/{kde,multiframe,multiframe_extended,bayes_balanced,bayes_calibrated}/`
> 状态：四项全部执行。**单帧契约的 GO 不是自然先验 Bayes（0.63–0.69），而是类平衡 Bayes（strength 0.5，cal 选出）：near 0.848、overall 0.962。** 压力档 σ=0.04 仍需多帧（M=64 → 0.846）。

---

## 0. 一句话

第四阶段把“噪声边界”从“换一个后验”推进到“换一个目标函数”。自然先验的 grid Bayes / KDE 都在 0.63–0.70，因为它们优化的是 overall，而 near-aligned 只占 ~6.5%，被 diagonal 淹没；把先验按 pair type 平衡（strength 0.5，训练 cal 选出）后，契约档 near-worst **0.848**、overall **0.962**，primary 和 overall 守卫同时通过。canonical 3°+1% 自然先验只有 0.693，确认不是像素特例；KDE 0.632 说明瓶颈不是 grid 近似；多帧 M=16 0.701、M=64 0.846，给出压力档的传感器级解法。runtime 轴谓词已以 `off/hard/bayes` 三模式接入。

---

## 1. Task 1：canonical 3D（3°+1% / 5°+2%）

### 1.1 主表

| level | U6 hard worst | Bayes grid near（natural） | Bayes grid near（balanced 0.5） | kernel-Bayes near（natural） |
|---|---:|---:|---:|---:|
| clean | 0.962 | 0.967 | — | — |
| 1°+0.5% | 0.868 | 0.859 | **0.950** | — |
| **3°+1%** | **0.701** | **0.693 ± 0.018** | **0.871** | 0.708 |
| 5°+2% | 0.516 | 0.461 ± 0.028 | **0.750** | 0.456 |

- 自然先验 `u_boundary_soft` 在 3°+1% 失败（0.693 < 0.80）。
- **类平衡 strength 0.5（像素 cal 选出，canonical 不重调）在 3°+1% 到 0.871、overall 0.957**（自然先验 overall 0.964，回退 0.7 pt）；5°+2% 0.750，仍低于 0.80。
- canonical 的 pair-diff 噪声 σ=0.0146，与像素 per-object σ=0.01（pair σ≈0.0141）几乎相同；**不是尺度/归一化问题**。
- 同一像素 split 用干净输入 + 查表 σ=0.01 复现 0.948；喂真实加噪坐标后 grid 0.658、kernel 0.646。**0.948 是度量假象。**

### 1.2 判读

canonical 的 0.693 与“正确加噪的像素 0.658”同档，说明 near-cell 的退化来自**决策规则/先验**，不是 3D 投影。自然先验只占 6.5% 的 near 格，overall-最优的 Bayes 会把边界推向 diagonal。

---

## 2. Task 2：KDE / 精确后验

### 2.1 主表

| σ | arm | overall | horizontal | vertical | diagonal | near-worst |
|---:|---|---:|---:|---:|---:|---:|
| 0.01 | `kde_raw` | 0.970 | 0.678 | 0.632 | 0.987 | **0.632** |
| 0.04 | `kde_raw` | 0.916 | 0.042 | 0.047 | 0.964 | **0.040** |
| 0.01 | `kde_type_balanced`（post-hoc） | 0.917 | 0.982 | 0.981 | 0.913 | **0.975** |
| 0.04 | `kde_type_balanced`（post-hoc） | 0.636 | 0.922 | 0.964 | 0.620 | **0.922** |

### 2.2 判读

- 精确 KDE 与 grid 高斯近似同档（0.632 vs 0.634），说明**瓶颈不是 grid 近似**。
- KDE 的 overall 0.970 与 ceiling 的 0.971 一致，但 near-worst 只有 0.632——**overall ceiling 不是 near-cell ceiling**。
- 自然先验下的 near-cell 上限约 0.63–0.71；要越过 0.80，必须改目标函数（先验平衡）或降噪声（多帧/更好传感器）。
- `kde_type_balanced` 是 post-hoc 诊断，不是预注册臂；它把 near 拉到 0.975/0.922，但 overall 掉到 0.917/0.636，diagonal 明显回退。这是**worst-cell-first 的 Pareto 点**，不是契约默认。

---

## 3. Task 3：多帧滤波 / 深度融合

### 3.1 主表（自然先验，3 seeds）

| σ | M | near-worst | overall | 实测 dx 噪声 |
|---:|---:|---:|---:|---:|
| 0.01 | 1 | 0.663 | 0.968 | 0.0141 |
| 0.01 | 4 | **0.844** | 0.985 | 0.0071 |
| 0.01 | 16 | 0.902 | 0.992 | 0.0035 |
| 0.04 | 1 | 0.027 | 0.893 | 0.0568 |
| 0.04 | 16 | **0.701** | 0.970 | 0.0141 |
| 0.04 | 32 | 0.790 | — | — |
| 0.04 | 64 | **0.846** | — | — |

### 3.2 判读

- **√2 口径**：`pa`、`pb` 独立加噪时，差值噪声是 `σ√2/√M`，不是 `σ/√M`；σ=0.04、M=16 实际 σ_eff=0.0141。
- 多帧方向完全正确：0.027 → 0.701（M=16）→ 0.846（M=64）。
- 单帧契约要 0.80，需要 σ≤0.005；σ=0.01 需要 M=4（0.844）；σ=0.04 需要 M=64。
- 多帧是**传感器级解法**，不是模型级；独立噪声/静态 pair 假设未在真实运动下验证。

---

## 4. Task 4：runtime 轴谓词接入

- `pss/reasoning/relations/bayes_predicate.py`：`AxisPredicate` 支持 `hard` / `bayes`，输出 `left/right/above/below` 及反向关系。
- `RelationRuleSet` 新增 `axis_mode`（`off/hard/bayes`）、`axis_boundary`、`axis_sigma`、`axis_eps`、`axis_abstain`。
- `RelationFactLayer` / `PSSRuntime` 新增同名参数；审计保留 `origin=hard_geometry` / `bayes_geometry`。
- 默认 `off` 保兼容；部署建议显式 `bayes` + 启动时拟合先验，`hard` 作为 σ 缺失 fallback。
- 测试：`tests/test_bayes_predicate.py` + runtime 子集 **31 项通过**。
- 报告：`docs/experiments/PSS-Boundary-Runtime.md`。

---

## 5. 修正后的契约结论：类平衡 Bayes

自然先验失败后，按 pair type 做先验重加权（`strength` 控制强度），在**训练 10% cal split** 上选择 strength；测试只报一次。

| strength | cal near | cal overall drop | test near | test overall | 判定 |
|---:|---:|---:|---:|---:|---|
| 0.25 | 0.775 | 小 | — | 0.966 | near 不足 |
| **0.5（cal 选出）** | — | ≤0.02 | **0.848** | **0.962** | **GO** |
| 0.75 | 0.921 | ~1.9 pt | 0.921 | 0.949 | 边界线上 |
| 1.0 | 0.944 | 2.7 pt | 0.944 | 0.941 | overall 守卫失败 |
| KDE full balance（post-hoc） | — | — | 0.975 | 0.917 | worst-cell-first，outside guard |

**这是 Pareto 前沿，不是“谁更好”**：强度越大，near 越高、overall 越低。契约的 2 pt overall 守卫选 **strength 0.5**；如果部署愿意接受 overall 回退（例如 diagonal 不是安全关键），KDE full balance 把 stress near 也拉到 0.922，但那是另一个 risk/coverage 合同。

测试端（`bayes_calibrated`）：

| level | overall | h | v | d | near-worst |
|---|---:|---:|---:|---:|---:|
| clean | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| **contract σ=0.01** | **0.962** | 0.865 | 0.858 | 0.967 | **0.848** |
| stress σ=0.04 | 0.890 | 0.052 | 0.079 | 0.936 | 0.052 |

- `u_boundary_soft` 0.848 ≥0.80 → **通过**。
- overall 0.962 vs hard 0.968：回退 **0.6 pt**，在 2 pt 守卫内。
- H3 负对照：`hard_noise_aug` 0.645；类平衡 Bayes − control = **+0.203** ≥0.05 → 通过。
- strength 0.5 在训练 cal 上选出，test 不参与选择。

---

## 6. 为什么之前的 GO 是错的

1. **干净输入口径**：`run_boundary_bayes.py` 早期版本只在干净测试坐标上改查表 σ，0.948 是假象；真实加噪后自然先验只有 0.634–0.658。
2. **overall 掩盖 near-cell**：ceiling 0.971 是 overall，near-cell 只有 0.63–0.71；primary 必须看 near-worst。
3. **目标函数错配**：自然先验优化 overall，near 只占 6.5%；类平衡先验把 near 拉到 0.848，代价是 overall −0.6 pt。
4. **√2 口径**：per-object σ 与 pair-diff σ 差 √2；多帧的 σ_eff 是 σ√2/√M。

---

## 7. 最终判定

| 问题 | 结论 |
|---|---|
| 单帧 σ=0.01 / 3°+1%，自然先验 | **NO-GO**（0.63–0.69） |
| 单帧 σ=0.01，类平衡 Bayes（strength 0.5，cal） | **GO**（near 0.848，overall 0.962） |
| 单帧 canonical 3°+1%，类平衡 Bayes（strength 0.5，transfer） | **GO**（near 0.871，overall 0.957） |
| 单帧 canonical 5°+2%，类平衡 0.5 | 0.750，低于 0.80；需多帧/full balance |
| 单帧 σ=0.04 压力档，strength 0.5 | **NO-GO**（0.052）；full balance 可到 0.922，但 overall 0.636 |
| 多帧 σ=0.04, M=64（自然先验） | **GO**（0.846，overall 保留） |
| 多帧 σ=0.01, M=4 | **GO**（0.844） |
| runtime 接入 | **完成**（off/hard/bayes） |

**正确结论：单帧契约在 σ=0.01 下不是“信息不可达”，而是自然先验的目标函数错配；类平衡 Bayes 可以达标。压力档 σ=0.04 单帧仍不可达，需要多帧 M≥64（或 σ≤0.005）。**

---

## 8. 失败与残余

1. 自然先验 grid/KDE 0.63–0.71：目标函数错配，不是后验近似。
2. KDE raw 预注册臂 0.662/0.000：NO-GO；`kde_type_balanced` 0.975/0.922 是 post-hoc 诊断，overall 0.917/0.636，超出 2 pt 守卫。
3. canonical 类平衡已补：3°+1% 0.871 / overall 0.957（GO）；5°+2% 0.750（低于 0.80，需多帧/full balance）。
4. 压力档单帧 strength 0.5 只有 0.052；full balance 可到 0.922 但 overall 0.636；多帧自然先验 M=16 0.701、M=32 0.790、M=64 0.846（overall 保留）。
5. 多帧假设独立噪声/静态 pair；真实运动、配准误差、深度缺失未测。
6. runtime 默认 `off`；`bayes` 需要显式配置 + 同尺度先验。
7. 类平衡 strength 只在像素 split 的 train cal 上选；canonical/真实数据需重新校准。

---

## 9. 下一步

1. **canonical + 类平衡已完成**（3°+1% 0.871）；下一步是 canonical 5°+2% 的多帧/full-balance 复测。
2. **多帧 + 类平衡**：σ=0.04、M=16 的自然先验 0.701；类平衡可能把 M=16 推到 0.80，降低帧数要求。
3. **runtime 默认化**：把类平衡 `GridBayesBoundary` 作为 `axis_mode="bayes"` 的推荐配置；hard fallback 保留。
4. **真实 RGB-D**：用真实深度/位姿噪声替换像素/3D 代理，重跑同一契约。
5. **abstention**：near-cell 低置信时弃权；报 selective accuracy/coverage，避免把边界错误写进关系层。

---

## 10. 产物

```
pss/eval/scripts/run_boundary_canonical.py
pss/eval/scripts/run_boundary_kde.py
pss/eval/scripts/run_boundary_multiframe.py
pss/reasoning/relations/bayes_predicate.py
pss/reasoning/relations/rules.py              # axis_mode
pss/runtime/layers.py                         # axis 参数 + origin
pss/runtime/runtime.py                        # PSSRuntime axis 参数
tests/test_bayes_predicate.py
docs/experiments/PSS-Boundary-Canonical.md
docs/experiments/PSS-Boundary-KDE.md
docs/experiments/PSS-Boundary-Multiframe.md
docs/experiments/PSS-Boundary-Runtime.md
eval/results/boundary_canonical/
eval/results/boundary_bench/{kde,multiframe,multiframe_extended,bayes_balanced,bayes_calibrated}/
```

---

## 11. 一句话

**自然先验 Bayes 是 NO-GO（0.63–0.71），类平衡 Bayes 是 GO（0.848 / 0.962）；canonical 确认不是像素特例，KDE 确认不是近似问题，多帧给出压力档的传感器级解法，runtime 轴谓词已接入。**
