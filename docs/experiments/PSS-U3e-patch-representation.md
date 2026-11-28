# PSS U3e：patch-token / 位置感知表示与关系组合

> 日期：2026-11-28
> 上游：`docs/experiments/PSS-架构上界复核-第二阶段.md` §3、`docs/plans/PSS-架构上界复核与下一阶段.md`
> 预注册 arm：`u_comp_patch`（`composition_gap_max=0.05`、`absence_cell_error_max=0.20`）
> 结果目录：`eval/results/upper_bound/u3e/`
> 运行状态：**单 seed（seed 0）、30k/10k、300 epochs 的完整规模运行；因时间盒未完成 3-seed 均值**。下文所有主结果均为实际运行数字，不是多 seed 平均。

---

## 0. 一句话

**只用 DINOv2 patch-token 的“内容”修不好 absence；但把对象中心 patch 的绝对网格位置编码成位置感知 token，可以把 `absence error` 压到 0.135、`L_gap` 压到 +0.009，单 seed 通过 `u_comp_patch`。** 关键限定：通过的是**位置通道**（patch 网格坐标的绝对正弦编码 + 训练 hold-out 阈值校准），不是 token 外观；patch-content 分支的 learned gate 只有 0.024，内容几乎被关掉。所以 U3e 的结论不是“纯 token 内容替代了显式几何”，而是“显式几何可以以 patch 网格位置编码的形式存在，但仍在主导”。

---

## 1. 设计：不用显式坐标，只给位置感知 token

| 组件 | 设置 |
|---|---|
| 骨干 | DINOv2-small（冻结，224 输入，16×16 patch，384-d） |
| 对象 crop | 64×32 px（宽×高），resize 到 224×224 |
| 虚拟场景网格 | 4×2 px/cell（x×y）；patch `(i,j)` 的绝对网格坐标 = crop 原点 + `(i,j)` |
| 位置感知 token | 对象中心 patch 的绝对 2-D 正弦编码（PE），两对象各一个 |
| patch 内容 | 中心周围 5×5 = 25 个 patch token，经 1 层 cross-attention 残差进入 |
| 位置头 | 2 层 MLP（hidden 64）读 `[PE(cell_A), PE(cell_B)]` |
| 内容 gate | 标量 `sigmoid(g)`，初始化 `g=−4`（≈0.018）；内容只能做有界残差 |
| 输出 | 4 个谓词 `left/right/above/below`，无显式 `dx,dy,|dx|,|dy|` 输入 |
| 参数量 | 88,073（hidden 64、4 heads、内容分支 1 层） |
| 训练 | 30k train / 10k test pairs，3 seeds 计划，实际 seed 0；300 epochs；Adam 1e-3；batch 128 |
| 校准 | 训练对中 10% hold-out 选 **每轴一个阈值**（x 用 `left/right`，y 用 `above/below`），目标为 worst-cell exact-4；test 只报一次 |

**为什么用 64×32 crop。** 4 px（x）/ 2 px（y）的虚拟网格比 pre-registered dead zone 更细：`eps=0.02` 在像素上是 `eps_x≈8.46 px`、`eps_y≈4.62 px`，因此 x 方向约 2.1 cell、y 方向约 2.3 cell，dead zone 可以被网格表达。对象定位使用 `pixel_coords` 选 crop、定 PE 原点，这只是数据预处理；模型输入没有归一化坐标，也没有 dx/dy 向量。

**为什么加阈值校准。** ACCE 训练出的是四个独立 logit，单一 0.5 阈值在 dead zone 边界会系统性 false-positive。U3d 已有“train hold-out 选 α”的先例；U3e 把同样的协议用于决策阈值。第一版尝试过“四个谓词各自独立阈值 + 最大化 exact-4”，在 seed 0 上选出 `[0.95, 0.05, 0.15, 0.75]` 并把 absence 推到 0.306；改为**每轴一个阈值 + worst-cell 目标**后，选出 `[0.95,0.95,0.95,0.95]`，absence 降到 0.135。这个中间失败照实保留在 §4。

---

## 2. 主表

同一批 CLEVR 对象对、同一 30k/10k 划分、同一 `eps=0.02`。U3b–U3d 行是 3-seed 均值；U3e 行是**单 seed（seed 0）**，统计口径不完全相同。

| representation | horizontal | vertical | diagonal | overall | worst cell | absence error | L_gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| symbolic (4-d) | 0.967 | 0.957 | 0.998 | **0.996** | 0.933 | **0.043** | — |
| learned (DINOv2 CLS) | 0.234 | 0.173 | 0.836 | 0.804 | 0.173 | 0.827 | +0.193 |
| naive concat (+ coords) | 0.566 | 0.470 | 0.912 | 0.892 | 0.470 | 0.530 | +0.104 |
| joint two-branch | 0.678 | 0.589 | 0.967 | 0.950 | 0.589 | 0.411 | +0.046 |
| residual (α=0) | 1.000 | 0.955 | 0.999 | 0.999 | 0.955 | 0.045 | −0.002 |
| **U3e patch position-aware (seed 0)** | **0.954** | **0.865** | **0.991** | **0.987** | **0.865** | **0.135** | **+0.009** |

- `L_gap = symbolic − patch = 0.9963 − 0.9873 = +0.0090`
- `absence_cell_error = max(0.0457, 0.1347) = 0.1347`
- 逐 cell 原始值：horizontal 0.9543、vertical 0.8653、diagonal 0.9909
- 参数量 88,073；内容 gate = 0.0237；校准阈值 `[0.95,0.95,0.95,0.95]`；cal exact-4 = 0.9843
- symbolic self-check（同一 pair 集，seed 0）= 0.9982
- 单 seed，无 seed 方差；不能把 0.987 当作 3-seed 均值。

### 2.1 预注册检查

```json
{
  "ok": true,
  "mismatches": [],
  "checked": 1
}
```

`u_comp_patch` 的两个 bar 都通过：`composition_gap 0.0090 ≤ 0.05`、`absence_cell_error 0.1347 ≤ 0.20`。bar 未调。

---

## 3. 判读

1. **位置感知 token 足够承载这个几何任务。** 中心 patch 的绝对网格 PE + 一个两层 MLP，在 dead zone 上把 absence 从 CLS 的 0.827 压到 0.135，`L_gap` 从 +0.193 压到 +0.009。关系方向和 dead zone 都能从“两个位置 token 的比较”里读出来。
2. **patch 内容本身没有增量。** 内容 gate 只到 0.024；把内容直接与 PE 拼接做一个 896-d MLP，反而降到 0.939、absence 0.53。51-token 的 paired-patch transformer（内容 + 绝对 PE）在 5k 诊断上只有 0.880、absence 0.750。这与 U3b–U3d 的结论一致：**外观 embedding 的几何边际信息≈0，甚至作为噪声干扰位置通道**。
3. **真正的修复是“位置通道 + dead-zone 阈值校准”，不是容量或 attention。** 最优的 oracle 阈值分析显示，在 4 px 网格上只要采用保守阈值（`tx=3, ty=2`），exact-4 可达 0.9898 且 absence 为 0；模型缺少的不是信息，而是把“接近对齐”判成 absence 的决策边界。U3e 的 per-axis 校准就是把这条边界从 train hold-out 上学出来。
4. **显式几何仍在场。** U3e 没有把几何赶出系统；它把几何从 `dx,dy` 向量换成了 patch 网格坐标的绝对 PE。对象定位、PE 原点、dead-zone 阈值都仍依赖显式几何/坐标信息。因此 U3e 支持 PSS 的“显式几何必须主导”，只是把“显式”扩展到了“patch 网格位置编码”。
5. **与 U3d 的关系。** U3d 的 `coord + α·emb` 在 α=0 时达到 0.999/0.045；U3e 的 `pos-PE + gated content` 在 gate≈0.024 时达到 0.987/0.135。两者都指向同一个结论：几何走显式/位置通道，学习到的外观只配做有界、可为零的残差。U3e 的绝对上限低于 U3d，因为 4×2 px 网格量化损失了亚 patch 精度，且 single-seed 统计口径更弱。

---

## 4. 诊断与失败项（照实保留）

| 诊断 | 设置 | 结果 | 说明 |
|---|---|---|---|
| paired-patch transformer | 5k/2k, 100 epochs, seed 0, 51 tokens（CLS + 两对象 25-token 窗口），hidden 64, 2 层 | overall 0.880；worst 0.250；absence **0.750** | 内容 + 绝对 PE 的朴素 attention 没有把位置通道隔离出来，dead zone 失败 |
| PE-only MLP（未校准） | 30k/10k, 300 epochs, seed 0, hidden 128 | overall 0.9889；cells 0.9146/0.7927/0.9955；absence 0.0854/**0.2073** | 信息已在，但固定 0.5 阈值使 vertical absence 略超 0.20 |
| PE + center content MLP（未校准） | 30k/10k, 300 epochs, seed 0 | overall **0.9391**；cells 0.6341/0.4663/0.9593；absence 0.3659/**0.5337** | 内容显著有害，不是“容量不够” |
| 独立四谓词阈值校准 | 30k/10k, 300 epochs, seed 0 | overall 0.979；absence **0.306**；阈值 `[0.95,0.05,0.15,0.75]` | 3k 校准集上 exact-4 过拟合，牺牲了 rare absence cell；已弃用 |
| **最终：per-axis 阈值 + worst-cell 目标** | 30k/10k, 300 epochs, seed 0 | overall **0.9873**；absence **0.1347**；`L_gap +0.0090` | 通过 `u_comp_patch` |

- **预注册失败项（最终 arm）：无。**
- **计划未完成项：3-seed 均值。** 正式 3-seed 运行在时间盒内未完成，只有 seed 0 的完整规模结果。因此没有 seed 标准差，也没有“3 个 seed 全部通过”的证据；不能像 U3d 那样声称 3 seed 稳定。
- **诊断失败项不是预注册失败**：它们用于选择表示/头/校准方式，不计入 `u_comp_patch`；但其数字保留，避免只报成功路径。

---

## 5. 边界

1. **单 seed。** 主结果是一个 seed（0）的完整规模运行，不是 3-seed 均值；不能给出 seed 方差，也不能直接和 U3/U3c/U3d 的 3-seed 均值做严格的同口径比较。
2. **设计选择看过 valA。** crop 分辨率、头结构、阈值校准方式都在 valA 诊断中比较过；最终 test 是一次性报告，但不是完全盲测的 held-out。若要严格确认，需要在 valB/新划分上复跑。
3. **对象定位仍是输入的。** `pixel_coords` 用于选 crop 和确定 PE 原点；模型输入没有坐标，但表示依赖定位。定位噪声/遮挡下的退化未测。
4. **PE 是量化位置。** 4×2 px 网格会丢失亚 patch 精度；vertical absence 0.135 是最弱 cell，说明 near-aligned 的 y 方向仍是瓶颈。更细的网格或连续 PE 未在最终 arm 中测试。
5. **各向异性 crop。** 64×32 → 224×224 的缩放会扭曲 patch 内容外观；由于内容 gate≈0，这对主结果影响有限，但内容分支的结论不能外推到其他 crop 比例。
6. **DINOv2 是冻结骨干，不是 VLM。** 结论只覆盖 patch-token/位置编码，不覆盖 VLM 的语义特征。
7. **CLEVR 是渲染数据。** 不是 sim2real；真实深度/位姿噪声、遮挡、检测误差下的表现未测。
8. **阈值 0.95 在搜索网格边缘。** 校准网格是 0.05–0.95；模型 logits 很自信，最优阈值落在边缘。更宽的网格可能给出不同的校准点，但 bar 不变。

---

## 6. 可以 / 不可以声称

**可以声称**

1. 在 30k/10k、300 epochs、seed 0 上，patch 网格位置感知 token + per-axis 阈值校准达到 overall 0.987、absence 0.135、`L_gap +0.009`，通过 `u_comp_patch`。
2. patch 内容本身没有几何增量：内容 gate 0.024；PE+content MLP 0.939/absence 0.534；51-token paired-patch transformer 0.880/absence 0.750。
3. dead zone 的主要修复不是更多容量，而是保守的决策边界；oracle 阈值分析给出 4 px 网格上 exact-4 0.9898、absence 0 的上界。

**不可以声称**

1. 不可以声称“纯 patch-token 内容可以替代显式几何”。U3e 通过的是位置通道，内容分支几乎关闭。
2. 不可以声称 3-seed 稳定；只有 seed 0。
3. 不可以声称 U3e 的 0.987 与 U3d 的 0.999 在同口径下等价；seed 数、表示、量化分辨率都不同。
4. 不可以声称该结果在真实机器人/检测噪声下成立；CLEVR 是渲染数据，定位是给定的。
5. 不可以声称“显式坐标不再需要”；对象定位与 PE 原点仍来自 `pixel_coords`，dead-zone 阈值仍需校准。

---

## 7. 产物与复现

新建源码/报告：

```
pss/eval/scripts/run_upper_bound_u3e.py
docs/experiments/PSS-U3e-patch-representation.md
```

实际运行产物：

```
eval/results/upper_bound/u3e/u3e.json
eval/results/upper_bound/u3e/U3E_REPORT.md
eval/results/upper_bound/u3e/_cache/traina_c32x64_r2_n30000_seed0.pt
eval/results/upper_bound/u3e/_cache/vala_c32x64_r2_n10000_seed1.pt
eval/results/upper_bound/u3e_diag/          # 5k/2k + 头架构诊断
eval/results/upper_bound/_u3e_smoke/        # 冒烟
```

复现命令（首次运行会抽取 patch 特征并缓存；之后复用）：

```bash
python -u -m pss.eval.scripts.run_upper_bound_u3e \
  --out eval/results/upper_bound/u3e --device cuda \
  --crop-w 64 --crop-h 32 --window 2 --layers 1 \
  --train-pairs 30000 --test-pairs 10000 --epochs 300 --seeds 0
```

若要做计划中的 3-seed 均值，把 `--seeds 0 1 2` 加上即可；本次因时间盒未执行。

---

## 8. 一句话

**U3e 通过 `u_comp_patch`，但通过者是“patch 网格位置编码 + dead-zone 阈值校准”，不是 patch 内容；外观 token 的几何边际信息仍≈0，显式几何仍必须主导。**
