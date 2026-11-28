# PSS-Boundary-KDE：用 KDE/mixture 后验替代 grid 高斯近似

> 日期：2026-11-28
> 上游：`docs/experiments/PSS-Boundary-Report.md` §3–§5、`docs/plans/PSS-问题重构.md`
> 结果：`eval/results/boundary_bench/kde/boundary_kde.json`（bench 口径）与 `eval/results/boundary_bench/kde_difference/boundary_kde.json`（ceiling 口径）
> 状态：KDE 已实现，并与 ceiling 脚本的 `_bayes_exact4` 逐位对齐。**raw 经验先验的 KDE 在两种口径下都没有修复 stress near；根因是先验/代价失衡，不是核近似。type-balanced 诊断在两种口径下都到 ≥0.92，但它是 post-hoc 诊断，不是预注册臂。**

---

## 0. 一句话

**KDE 证明了 grid 高斯近似不是 stress 失败的原因：在真正含噪的输入上，KDE 与 grid 的 overall/near 几乎一样。真正的原因是先验里 near cell 只占 3.2%/1.6%，overall-optimal 的 Bayes 投票在 σ=0.04 时从不投 "neither"，于是 near cell 被结构性牺牲。** 同一套 KDE 核，只把先验按 pair type 重新配平，stress near 就从 0.000 抬到 0.922（bench 口径）或从 0.006 抬到 0.948（ceiling 口径）；oracle 知道 cell 类型时上限是 0.994（bench）到 1.000（ceiling）。也就是说：信息在坐标里，丢的是先验/代价平衡。

**另一个必须先说清楚的问题：bench 和 ceiling 的 σ 不是同一个量。** `run_boundary_bench` 给每个物体坐标加 N(0, σ)，所以 pair 差值的噪声是 σ√2；`run_boundary_ceiling` 直接把 σ 加在差值上。本文两个口径都报，主表用 bench 口径（与 hard/soft 可比），敏感性表用 ceiling 口径（与 `_bayes_exact4` 可比）。

---

## 1. 实现与口径

| 项 | 固定值 |
|---|---|
| 数据 | CLEVR CoGenT：train `traina` 30k pairs / test `vala` 10k pairs |
| 坐标 | 与 boundary bench 相同：train min-max 归一化，test 用同一 lo/hi |
| 标签 | `pair_labels(eps=0.02)`，四 bit：left/right/above/below |
| 单元 | `pair_types(eps=0.02)`：horizontal / vertical / diagonal |
| 契约噪声 | σ=0.01（预注册判定档） |
| 压力噪声 | σ=0.04（只报不判） |
| 噪声 seed | 3 个（repeat 0/1/2），同一 seed 下三个 arm 共用同一份噪声 |
| 先验粒子 | 30k train pairs 全量（主结果）；3000 粒子作为敏感性检查 |
| chunk | 500（10k×30k 不整块落内存） |

### 1.1 噪声口径

| 脚本 / arm | 噪声怎么加 | 差值噪声 σ_d |
|---|---|---:|
| `run_boundary_bench`（hard/soft） | 每个物体坐标加 N(0, σ) | σ√2 |
| `run_boundary_ceiling`（`_bayes_exact4`） | 差值加 N(0, σ) | σ |
| `run_boundary_bayes`（已发布 grid） | 不加噪声；σ 只当先验平滑宽度 | 0 |
| `run_boundary_kde --noise-space object` | 差值加 N(0, σ√2)，核带宽 σ√2 | σ√2 |
| `run_boundary_kde --noise-space difference` | 差值加 N(0, σ)，核带宽 σ | σ |

**主表用 `object`（与 bench 的 hard/soft 同一口径）；敏感性表用 `difference`（与 ceiling 同一口径）。两者不能混比。** 例如 σ=0.01 在 object 口径下是差值 0.01414，在 difference 口径下是差值 0.01000，后者明显更简单。

### 1.2 KDE 规则

对每个观测对 `obs=(dx,dy)` 和每个先验粒子 `k`：

```
w_k ∝ exp(-||obs - prior_k||^2 / (2 σ_d^2))
```

对每轴三态（positive / negative / neither）加权投票，取 argmax，得到 bits 与 confidence：

- bits 顺序与 `pair_labels` 一致：`[left, right, above, below]`；
- confidence = 两轴获胜态后验权重的较小值；
- 先验粒子来自 train pairs，粒子状态用同一 `eps=0.02` 计算。

实现上用 log-sum-exp 稳定化（先减每行最大 log 权重再 `exp`），避免小 σ 下整行下溢。观测按 chunk 处理。

### 1.3 三个 arm

| arm | 说明 |
|---|---|
| `kde_raw` | 预注册臂：raw train 先验（diagonal 95.2%），无重加权 |
| `kde_type_balanced` | 诊断臂：同一 KDE 核，先验按 pair type 反频率加权，使 horizontal/vertical/diagonal 先验质量相等 |
| `grid_noisy` | 诊断臂：现有 `GridBayesBoundary` 在同一份含噪输入上复评（不改 `bayes_boundary.py` / `run_boundary_bayes.py`） |

### 1.4 正确性校验

在 3000 test pairs、同一先验、同一差值噪声上，`KDEBayesBoundary` 与 `run_boundary_ceiling._bayes_exact4` 的 bits **逐位相同**：

| σ_d | `_bayes_exact4` exact-4 | KDE exact-4 | 逐位不同的行 |
|---:|---:|---:|---:|
| 0.01 | 0.974333 | 0.974333 | 0 |
| 0.04 | 0.924000 | 0.924000 | 0 |

因此下面的差异不是实现误差。

---

## 2. 主表（bench 口径，`--noise-space object`，3 seeds mean ± std）

σ 列是每物体坐标噪声；差值噪声在括号里。near-worst = 每个 seed 的 `min(horizontal, vertical)` exact-4，再取 mean；absence = 每个 seed 的 `max(1-h, 1-v)`，再取 mean。因此 near-worst 不必等于表内 h/v 均值的较小者（per-seed 最小值的均值 ≤ 均值的较小者）。

| arm | σ（差值） | overall | horizontal | vertical | diagonal | near-worst | absence |
|---|---:|---:|---:|---:|---:|---:|---:|
| `kde_raw` | 0.01 (0.0141) | 0.9699 ± 0.0016 | 0.6758 ± 0.0217 | 0.6718 ± 0.0255 | 0.9861 | **0.6623 ± 0.0254** | 0.3377 |
| `kde_raw` | 0.04 (0.0566) | 0.8932 ± 0.0009 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.9423 | **0.0000 ± 0.0000** | 1.0000 |
| `kde_type_balanced` | 0.01 (0.0141) | 0.9167 ± 0.0005 | 0.9817 ± 0.0066 | 0.9810 ± 0.0065 | 0.9131 | **0.9753 ± 0.0029** | 0.0247 |
| `kde_type_balanced` | 0.04 (0.0566) | 0.6362 ± 0.0066 | 0.9217 ± 0.0029 | 0.9637 ± 0.0042 | 0.6196 | **0.9217 ± 0.0029** | 0.0783 |
| `grid_noisy` | 0.01 (0.0141) | 0.9701 ± 0.0010 | 0.7317 ± 0.0260 | 0.6684 ± 0.0254 | 0.9845 | **0.6684 ± 0.0254** | 0.3316 |
| `grid_noisy` | 0.04 (0.0566) | 0.8931 ± 0.0014 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.9422 | **0.0000 ± 0.0000** | 1.0000 |

判读：

- `kde_raw` 与 `grid_noisy` 的 overall 几乎相同（0.970 vs 0.970；0.893 vs 0.893），contract near 也几乎相同（0.662 vs 0.668），stress near 都是 0.000。**在真正的 bench 口径下，精确 KDE 和高斯 grid 没有本质差别；stress 失败不是近似造成的。**
- `kde_type_balanced` 不是免费午餐：它把 near 拉高的同时，diagonal 从 0.986 掉到 0.913（contract）、从 0.942 掉到 0.620（stress），overall 也降到 0.917/0.636。它是“按 worst-cell 目标重配平代价”的诊断，不是部署建议。

---

## 3. 敏感性：ceiling 口径（`--noise-space difference`）

这一节把 σ 直接加在差值上，与 `run_boundary_ceiling._bayes_exact4` 同口径。它的噪声比 bench 口径低（σ=0.01 时 0.0100 vs 0.0141；σ=0.04 时 0.0400 vs 0.0566），所以数字普遍更好，**不能拿来和 hard/soft 比**。near/absence 的定义同 §2（per-seed 再 mean）。

| arm | σ（差值） | overall | horizontal | vertical | diagonal | near-worst | absence |
|---|---:|---:|---:|---:|---:|---:|---:|
| `kde_raw` | 0.01 | 0.9783 ± 0.0011 | 0.7642 ± 0.0166 | 0.7703 ± 0.0233 | 0.9899 | **0.7616 ± 0.0177** | 0.2384 |
| `kde_raw` | 0.04 | 0.9182 ± 0.0003 | 0.0061 ± 0.0025 | 0.0328 ± 0.0049 | 0.9678 | **0.0061 ± 0.0025** | 0.9939 |
| `kde_type_balanced` | 0.01 | 0.9451 ± 0.0008 | 0.9837 ± 0.0038 | 0.9827 ± 0.0065 | 0.9430 | **0.9791 ± 0.0042** | 0.0209 |
| `kde_type_balanced` | 0.04 | 0.7363 ± 0.0024 | 0.9482 ± 0.0066 | 0.9793 ± 0.0042 | 0.7241 | **0.9482 ± 0.0066** | 0.0518 |
| `grid_noisy` | 0.01 | 0.9789 ± 0.0008 | 0.8130 ± 0.0076 | 0.7668 ± 0.0153 | 0.9890 | **0.7668 ± 0.0153** | 0.2332 |
| `grid_noisy` | 0.04 | 0.9183 ± 0.0009 | 0.0742 ± 0.0112 | 0.0466 ± 0.0220 | 0.9653 | **0.0462 ± 0.0217** | 0.9538 |

结论与主表一致：raw KDE 与 grid 几乎重合；raw KDE 的 stress near 只有 0.006，仍然远低于 0.60；type-balanced 到 0.948。

---

## 4. 与 hard / grid Bayes 并排比较

| arm | 噪声口径 | clean near | contract near | stress near | contract overall | stress overall |
|---|---|---:|---:|---:|---:|---:|
| `hard_clean` | 每物体 σ | 0.959 | 0.685 | 0.236 | 0.968 | 0.872 |
| `hard_noise_aug` | 每物体 σ | 0.906 | 0.645 | 0.212 | 0.968 | 0.876 |
| `soft_estimated` | 每物体 σ | 1.000 | 0.682 | 0.238 | 0.968 | 0.872 |
| `bayes_grid_published` | clean 输入 | 1.000 | 0.948 | 0.109 | 0.997 | 0.955 |
| `grid_noisy` | 差值 σ√2（bench） | — | 0.668 | 0.000 | 0.970 | 0.893 |
| `kde_raw` | 差值 σ√2（bench） | — | **0.662** | **0.000** | 0.970 | 0.893 |
| `kde_type_balanced` | 差值 σ√2（bench） | — | 0.975 | 0.922 | 0.917 | 0.636 |

关键读数：

- **在 bench 口径下，raw Bayes 的 contract near（0.662）略低于 hard_clean（0.685），stress near（0.000）远低于 hard_clean（0.236）。** 它的 overall 略高（0.970 vs 0.968；0.893 vs 0.872），但主指标 near-worst 更差。也就是说：raw 经验 Bayes 并没有解决 bench 的 near 问题，它只是把误差从 near 换到了别处。
- `kde_type_balanced` 是唯一在 bench 口径下把 near 拉到 0.60 以上的 arm（contract 0.975，stress 0.922），但 overall 掉到 0.917/0.636，diagonal 明显回退。这是“worst-cell 优先”的代价。
- `bayes_grid_published` 的 0.948/0.109 不能与上面两行直接比，见 §5。

---

## 5. 口径核查：已发布的 grid Bayes 用的是 clean 输入

`run_boundary_bayes.py` 的调用是：

```python
bits, conf = boundary.predict(dx_te, dy_te, sigma=level)
```

其中 `dx_te, dy_te` 直接来自 clean test pairs；`sigma` 只被 `GridBayesBoundary` 当作**先验平滑宽度**，并没有加到观测坐标上。也就是说，已发布的 `bayes_grid_published` 在 σ=0.01 的 0.948 是“clean 输入 + 按 σ 平滑的先验”，而不是“含噪输入 + 噪声似然”。这与 `run_boundary_bench.py`（`_noisy(pa_te, pb_te, level, ...)`，所有 arm 都吃噪声）和 `run_boundary_ceiling.py`（`observed + rng.normal(0, sigma)`）的口径都不一致。

三个独立证据：

1. 同一个 `GridBayesBoundary`，在真正含噪的输入上复评：ceiling 口径（差值 σ）contract near 从 0.948 掉到 0.767，stress 从 0.109 掉到 0.046；bench 口径（差值 σ√2）contract near 进一步掉到 0.668，stress 掉到 0.000。
2. `bayes_grid_published` 的 contract overall 是 0.997，高于同先验的精确 Bayes 的 0.978（difference 口径）和 0.970（object 口径）；如果消费的是同一份含噪输入，这是不可能的。
3. `kde_raw`（精确经验 Bayes）与 `grid_noisy` 在两种口径下的 overall 和 near 都几乎重合，说明 0.948 的优势不是“grid 近似更好”，而是输入更干净。

**结论：上一份报告的 “Bayes 契约档 0.948 GO” 是在 clean 输入上得到的，不能与 hard/soft 的含噪数字直接比较。** 按 bench 口径，contract near 的真实水平是 ~0.66，低于 0.80 的预注册 bar。本任务不修改 `run_boundary_bayes.py` / `bayes_boundary.py`，只把这一口径问题记录下来，交给上游报告处理。

---

## 6. 预注册判定

`u_boundary_soft` 读 contract σ=0.01 的 near-worst（映射到 `worst_cell_acc`）。

| arm | 噪声口径 | contract near-worst | bar | 判定 |
|---|---|---:|---:|---|
| `kde_raw`（预注册臂） | bench（差值 σ√2） | **0.6623** | ≥ 0.800 | **未通过** |
| `kde_raw`（预注册臂） | ceiling（差值 σ） | **0.7616** | ≥ 0.800 | **未通过** |
| `kde_type_balanced`（post-hoc） | bench | 0.9753 | ≥ 0.800 | 通过，但不是预注册臂 |
| `kde_type_balanced`（post-hoc） | ceiling | 0.9791 | ≥ 0.800 | 通过，但不是预注册臂 |

判定以 `kde_raw` 为准：**NO-GO**。两种口径下都未通过，结论稳健。`kde_type_balanced` 的通过不能当作预注册胜利，因为它是在看到 raw 失败之后加的诊断臂，且它改变了先验/代价，不是原任务指定的 raw 经验先验 KDE。bar 未调整。

---

## 7. 回答 stress 问题：是否 ≥0.60，以及 per-cell ceiling

### 7.1 直接回答

| 问题 | bench 口径（差值 σ√2） | ceiling 口径（差值 σ） |
|---|---:|---:|
| `kde_raw` stress near ≥ 0.60？ | **否，0.0000** | **否，0.0061** |
| `kde_type_balanced` stress near ≥ 0.60？ | **是，0.9217** | **是，0.9482** |
| oracle 知道 cell 类型时的 stress near 上限？ | **0.994** | **1.000** |

所以：**“KDE 换掉高斯近似”本身没有把 stress near 抬到 0.60；把先验/代价配平可以。** 这两件事必须分开说。

### 7.2 per-cell ceiling

| rule / prior | 噪声口径 | σ | horizontal | vertical | near-worst | overall |
|---|---|---:|---:|---:|---:|---:|
| raw empirical Bayes（`kde_raw`） | bench | 0.01 | 0.676 | 0.672 | 0.662 | 0.970 |
| raw empirical Bayes（`kde_raw`） | bench | 0.04 | 0.000 | 0.000 | 0.000 | 0.893 |
| type-balanced（`kde_type_balanced`） | bench | 0.01 | 0.982 | 0.981 | 0.975 | 0.917 |
| type-balanced（`kde_type_balanced`） | bench | 0.04 | 0.922 | 0.964 | 0.922 | 0.636 |
| raw empirical Bayes（`kde_raw`） | ceiling | 0.01 | 0.764 | 0.770 | 0.762 | 0.978 |
| raw empirical Bayes（`kde_raw`） | ceiling | 0.04 | 0.006 | 0.033 | 0.006 | 0.918 |
| type-balanced（`kde_type_balanced`） | ceiling | 0.01 | 0.984 | 0.983 | 0.979 | 0.945 |
| type-balanced（`kde_type_balanced`） | ceiling | 0.04 | 0.948 | 0.979 | 0.948 | 0.736 |
| oracle cell-conditional | bench | 0.01 | 1.000 | 1.000 | 1.000 | — |
| oracle cell-conditional | bench | 0.04 | 0.994 | 1.000 | 0.994 | — |
| oracle cell-conditional | ceiling | 0.01/0.04 | 1.000 | 1.000 | 1.000 | — |

- raw KDE 的 per-cell 数字就是 overall-optimal 经验 Bayes 在 near cell 上的表现。它不是“核不够细”：换成精确 KDE 后仍是 0.000/0.006；它也不是“表示里没有信号”：oracle 和 type-balanced 都能恢复。
- oracle cell-conditional 的做法是：对每个 test cell，只用同类型的 train particles 做 KDE（即假设知道 cell 类型）。它给出 0.994（bench，stress horizontal）到 1.000，说明**只要 cell 类型已知，缺失轴的状态基本是可分的**；难点在 cell 类型本身，不在轴信号。
- type-balanced 是可实现的近似：它不使用 test 标签，只把 train 先验按类型反频率加权（horizontal 3.21%、vertical 1.56%、diagonal 95.23% 变成等质量）。

### 7.3 机制：为什么 raw Bayes 在 σ=0.04 从不投 “neither”

先验类型频率：

| type | frequency |
|---|---:|
| diagonal | 0.9523 |
| horizontal | 0.0321 |
| vertical | 0.0156 |

在后验里，缺失轴（near cell 的 y 或 x）要投 “neither”，才能把 near cell 的四 bit 全部答对。但在 σ=0.04 时：

| 口径 | 差值 σ_d | 观测桶 | n | mean p(pos) | mean p(neg) | mean p(neither) | argmax=neither |
|---|---:|---|---:|---:|---:|---:|---:|
| bench | 0.0566 | `abs(dy)<eps` | 468 | 0.404 | 0.399 | **0.197** | **0.000** |
| ceiling | 0.0400 | `abs(dy)<eps` | 429 | 0.350 | 0.365 | **0.285** | **0.000** |

对真正的 horizontal test rows（n=328），两种口径下 argmax 也从不取 neither。原因是 near cell 在 raw 先验里只占 3.2%/1.6%；σ=0.04 = 2×eps 时，正/负两类的先验质量各约 48%，即使在 `obs≈0` 处也压过 3.2% 的 neither。overall-optimal Bayes 为了 95% 的 diagonal 行，理性地放弃 near cell。**这是先验/代价问题，不是 KDE 近似问题。**

bench 口径的差值噪声更大（0.0566），所以 `p_neither` 进一步降到 0.197，raw KDE 的 stress near 直接归零；ceiling 口径噪声小一些，`p_neither` 是 0.285，但由于 pos/neg 各自仍 ~0.35，argmax 仍不取 neither，near 只有 0.006。两个口径讲的是同一个机制。

---

## 8. Latency 诊断

在 1000 test pairs、30k 先验粒子、chunk=500 下，warm 后 5 次取中位数：

| arm | bench σ=0.01（差值 0.0141） | bench σ=0.04（差值 0.0566） | ceiling σ=0.01 | ceiling σ=0.04 |
|---|---:|---:|---:|---:|
| KDE（raw / balanced 同复杂度） | 941 ms / 1000 pairs | 747 ms / 1000 pairs | 1082 ms / 1000 pairs | 859 ms / 1000 pairs |
| grid 查表 | 0.05 ms / 1000 pairs | 0.05 ms / 1000 pairs | 0.05 ms / 1000 pairs | 0.05 ms / 1000 pairs |

- KDE 比 grid 查表慢约 **1.5×10^4–2.2×10^4 倍**。KDE 是 O(N·K) 每对；30k 粒子时不可能作为部署路径。
- JSON 里的 in-run 计时受系统负载影响更大：bench 口径 raw 为 894/687 ms/1000，balanced 为 890/839，grid 为 0.52/0.74；ceiling 口径 raw 为 1167/718，balanced 为 1319/743，grid 为 1.12/0.63。上表是 warm 后的中位数，用于排除冷启动和负载噪声。
- 建表/拟合：KDE 1.27 ms（raw）、1.61 ms（balanced）；grid 3.02 ms。建表成本不是问题，查询成本才是。
- 小 σ 反而比大 σ 慢约 25%，原因是小 σ 下大量 `exp` 结果落到 subnormal/underflow 区，不是正确性问题；与 `_bayes_exact4` 的逐位一致已经排除了数值错误。
- 结论：**grid 查表仍是部署形态；KDE 只用于诊断/上限。**

---

## 9. 敏感性：先验粒子数

主结果用 30k 全量先验。为了与 `run_boundary_ceiling.py` 的 3000 粒子对齐，另跑了一次 `--prior-samples 3000 --noise-space difference`（3 seeds）：

| arm | σ=0.01 near | σ=0.04 near | σ=0.01 overall | σ=0.04 overall |
|---|---:|---:|---:|---:|
| `kde_raw`（3000） | 0.6321 | 0.0402 | 0.9704 | 0.9157 |
| `kde_type_balanced`（3000） | 0.8946 | 0.9437 | 0.9536 | 0.7470 |
| `grid_noisy`（与先验粒子数无关） | 0.7668 | 0.0462 | 0.9789 | 0.9183 |

- 3000 粒子的 raw KDE 在 σ=0.01 时，每个观测的核邻域有效粒子数（ESS）平均约 4（中位 3.6，p10 1.4），全量 30k 时约 34（中位 32，p10 11）；near cell 的 Monte Carlo 噪声明显，contract near 从全量的 0.762 掉到 0.632。
- 但 ceiling 脚本报的 **overall** 0.970 与全量 KDE 的 0.978 接近，说明 **overall 会掩盖 near cell 的先验稀疏问题**。如果只看 overall，会误以为 3000 粒子足够。
- type-balanced 对先验粒子数更稳（stress 0.944 vs 0.948），因为它在 near 区域有效加权了少数粒子。

---

## 10. 边界与结论

1. **KDE 是诊断/上限，不是部署路径。** O(N·K) 查询在 30k 粒子时是秒级/千对；grid 查表仍是部署形态。
2. **grid 高斯近似不是 stress 失败的原因。** 在 bench 口径和 ceiling 口径下，KDE 与 grid 的 overall/near 都几乎重合；精确 KDE 也复现了 ceiling 的 overall。
3. **stress near 低的根因是先验/代价失衡。** raw 先验 95.2% 是 diagonal，near cell 只占 3.2%/1.6%；σ=0.04 时 near 桶的后验 `p_neither` 只有 0.197（bench）/0.285（ceiling），低于 pos/neg 的 ~0.35–0.40，argmax 从不投 neither。overall-optimal Bayes 结构性放弃 near cell。
4. **信息没有丢。** oracle 知道 cell 类型时 stress near = 0.994（bench）/1.000（ceiling）；type-balanced 先验（可实现、不用 test 标签）stress near = 0.922（bench）/0.948（ceiling）。因此 0.60 的目标不是信息上限问题，而是先验/代价/决策问题。
5. **预注册的 raw KDE 在两种口径下都未通过 contract bar（0.662/0.762 < 0.80）**；type-balanced 的通过是 post-hoc 诊断，不能替代预注册臂。
6. **已发布 grid Bayes 的 0.948/0.109 是 clean 输入口径**，与 bench/ceiling 的含噪口径不一致；按 bench 口径复评是 0.668/0.000，按 ceiling 口径是 0.767/0.046。上一份报告的契约 GO 需要按含噪口径重新审视。
7. **数据边界**：CLEVR 渲染几何；σ 是归一化像素坐标代理，不是真实 RGB-D 标定。先验是 train-pair 经验分布，部署换场景需要重估。
8. **type-balanced 的代价**：它把 near 拉高的同时牺牲 diagonal/overall（bench 口径 stress overall 0.636 vs raw 0.893），不是免费的；是否可接受取决于任务是否真的以 near-worst 为主指标。

---

## 11. 产物与复现

新建文件：

```
pss/reasoning/relations/kde_boundary.py       # KDE/mixture 经验 Bayes 边界
pss/eval/scripts/run_boundary_kde.py          # 三 arm 评估（raw / type-balanced / grid-noisy）
docs/experiments/PSS-Boundary-KDE.md          # 本报告
eval/results/boundary_bench/kde/boundary_kde.json            # bench 口径（主）
eval/results/boundary_bench/kde_difference/boundary_kde.json # ceiling 口径（敏感性）
```

未修改：`bayes_boundary.py`、`soft_deadzone.py`、`run_boundary_bayes.py`、`run_boundary_bench.py`、`run_boundary_ceiling.py`、U1–U6 脚本、preregistration、runtime 代码。

复现（bench 口径，主表）：

```bash
python -u -m pss.eval.scripts.run_boundary_kde \
  --noise-space object --out eval/results/boundary_bench/kde
```

敏感性（ceiling 口径，与 `_bayes_exact4` 同口径）：

```bash
python -u -m pss.eval.scripts.run_boundary_kde \
  --noise-space difference --out eval/results/boundary_bench/kde_difference
```

先验粒子数敏感性：

```bash
python -u -m pss.eval.scripts.run_boundary_kde \
  --noise-space difference --prior-samples 3000 \
  --out eval/results/boundary_bench/kde_prior3000
```

默认：30k train pairs / 10k test pairs / 30k 先验粒子 / chunk 500 / 3 个噪声 seed / eps 0.02。

---

## 12. 一句话

**精确 KDE 证明了两件事：grid 高斯近似不是 stress 问题的原因；raw 经验先验下的 overall-Bayes 会理性地放弃只占 3.2% 的 near cell。把先验按 cell 配平，同一个核就把 stress near 从 0.000/0.006 抬到 0.922/0.948；因此 0.60 的瓶颈是先验/代价，不是核或轴信号。**
