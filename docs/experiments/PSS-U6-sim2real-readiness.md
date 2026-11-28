# PSS U6：显式几何骨架在传感器噪声下的鲁棒性（sim2real readiness，**不是 sim2real**）

> 日期：2026-11-28
> 上游：`docs/experiments/PSS-架构上界复核-第二阶段.md` §3、`docs/plans/PSS-架构上界复核与下一阶段.md`
> 结果目录：`eval/results/upper_bound/u6/`
> 状态：已实际执行（3 seeds × 6 clean heads，30k train / 10k test pairs，300 epochs）；本报告所有数字来自本轮 `u6.json` 与终端日志。
> **标题与结论边界：这是传感器噪声压力测试，不是 sim2real。仓库没有真实 RGB-D / 真机数据，没有真实传感器标定，CLEVR 是渲染几何。**

---

## 0. 一句话结论

U6 在 CLEVR 关系对上训练 clean 的显式几何骨架，然后在测试坐标上注入两类噪声。**overall exact-4 很稳，但 absence / near-aligned cell 很脆**：坐标噪声 σ=0.04 时 overall 仍有 **0.875 ± 0.002**、diagonal **0.908 ± 0.002**，但 worst-cell 掉到 **0.252 ± 0.006**、absence error **0.743**；3D→2D 投影族最高档（yaw/pitch 5° + depth 2%）worst-cell **0.516 ± 0.010**、absence error **0.484**、overall 仍有 **0.937 ± 0.003**。预注册 arm `u_geom_noise` 在最高噪声档**两项都失败**（`worst_cell_acc 0.252 < 0.85`；`delta_vs_full −0.681 < −0.10`），失败照实保留。归因显示：位姿噪声几乎无害，**深度噪声主导**；机制是 dead-zone 边界穿越，absence error 近似由 `σ_rel / eps` 控制。因此，clean-trained 硬阈值几何骨架**尚未达到可部署的传感器噪声鲁棒性**；这不是 sim2real 通过，而是一个明确的 readiness gap。

---

## 1. 设置与噪声模型

### 1.1 数据与任务

- CLEVR CoGenT v1.0：train `traina`（30,000 pairs）、test `vala`（10,000 pairs）。
- 关系对与 U3/U3d 完全一致：同一批场景内有序对象对、同一 min-max 归一化（pixel 坐标）、同一 dead zone `eps=0.02`、同一 pair type 定义（`|dy|≤eps → horizontal`；`|dx|≤eps → vertical`；其余 `diagonal`）。
- 标签与 pair type 一律由 **clean 坐标**计算；只有送入 head 的输入坐标带噪。这对应部署问题：传感器有噪声，物理关系本身没有。
- 每个 seed 只训练一次 clean head；所有噪声档复用同一个 head。没有做 noise augmentation，因此本实验测的是 **clean-trained 骨架的脆弱性**，不是噪声感知训练后的最好水平。

### 1.2 主族：归一化像素坐标噪声

对每个对象的归一化坐标独立注入各向同性高斯噪声：

```
u' = u + N(0, σ²),  v' = v + N(0, σ²)
```

- σ ∈ {0.005, 0.01, 0.02, 0.04}，单位为归一化图像 span（0.5%–4%）。
- 两个对象独立加噪，因此 pair 的**相对坐标噪声**为 `σ_rel = σ·√2`（0.7%–5.7%）。
- dead zone `eps=0.02` 是这套坐标下的 near-aligned 判据；`σ_rel / eps` 从 0.35 升到 2.83，正好覆盖“几乎无损”到“absence 饱和失效”的区间。

### 1.3 副族：3D→2D 投影 + 位姿/深度噪声

用 CLEVR 场景自带的 `3d_coords` 构造真正的 3D→2D 投影：

1. 在 **train cache** 上用 DLT 标定一个 pinhole 相机 `pixel ≈ P·[X,1]`（4,000 个点；R 的 `det=+1`；camera center `C=[6.25, −5.33, 4.41]`；全 cache 重投影 RMS **4.84 px**，这是标注中心与物体原点的残差诊断，不是传感器标定）。
2. 将 3D 点投影到世界 XY 桌面平面，作为 2D 观测。
3. **位姿噪声**：每个场景共享一次 yaw/pitch 误差，`R' = R·Ry(yaw)·Rx(pitch)`；世界点绕相机中心做刚体旋转 `X' = C + M(X−C)`，`M = R'ᵀR`。yaw/pitch ~ N(0, 1°/3°/5°)。
4. **深度噪声**：每个对象独立 `X' = C + (1+ε)(X−C)`，`ε ~ N(0, σ_d²)`，σ_d ∈ {0.5%, 1%, 2%}。
5. 组合档：`(1°, 0.5%)`、`(3°, 1%)`、`(5°, 2%)`；另报 pose-only 与 depth-only 做归因。

> 说明：`data/category_pretrain_instance8` 有 `depth` 与单物体 `pose`，但没有对象对、没有相机内参、没有共享场景几何，无法构造 U3 所需的关系对。CLEVR scene 的 `3d_coords` 是更强的选择；category 数据只在 §7 边界中说明。

### 1.4 读数规则（运行前固定）

`u_geom_noise` 在两个族的**最高档**各读一次，取二者中**更差**的一个：

```
worst_cell_acc = min(coord σ=0.04, canonical pose5+depth2)
delta_vs_full  = min(coord σ=0.04 − clean pixel, canonical P3 − clean canonical)
```

`delta_vs_full` 的 “full” 是与该族**匹配的 clean head**（同 seed、同 test pairs），因为这里的 full sensor fidelity 就是 clean 几何。bar：`worst_cell_acc_min=0.85`、`delta_vs_full_min=−0.10`。不允许看到结果后放宽。

---

## 2. 主表

### 2.1 归一化像素坐标噪声

3 seeds 的 mean ± std；worst cell 是每个 seed 先取三格最小值、再跨 seed 平均；absence error 取 `1 − cell_acc` 的较大者。

| level | horizontal | vertical | diagonal | overall | worst cell | absence error |
|---|---:|---:|---:|---:|---:|---:|
| clean | 0.967 ± 0.046 | 0.957 ± 0.015 | 0.998 ± 0.001 | **0.996 ± 0.002** | **0.933 ± 0.023** | 0.043 |
| σ = 0.005 | 0.852 ± 0.007 | 0.838 ± 0.032 | 0.992 ± 0.001 | 0.985 ± 0.001 | 0.830 ± 0.022 | 0.162 |
| σ = 0.01 | 0.716 ± 0.019 | 0.703 ± 0.042 | 0.982 ± 0.001 | 0.968 ± 0.002 | 0.692 ± 0.026 | 0.297 |
| σ = 0.02 | 0.464 ± 0.018 | 0.418 ± 0.014 | 0.958 ± 0.002 | 0.931 ± 0.002 | 0.418 ± 0.014 | 0.582 |
| σ = 0.04 | 0.257 ± 0.013 | 0.280 ± 0.029 | 0.908 ± 0.002 | 0.875 ± 0.002 | **0.252 ± 0.006** | **0.743** |

- clean 行与 U3 symbolic **逐位一致**（0.9963 / 0.9329 / 0.9675 / 0.9568 / 0.9981），说明 pair、归一化、dead zone 与 U3 完全同源；不是另一套任务。
- `delta_vs_full` 逐档：σ=0.005 **−0.103**、σ=0.01 **−0.241**、σ=0.02 **−0.515**、σ=0.04 **−0.681**。
- **最低档 σ=0.005 已经跌破 0.85 的 worst-cell bar**（0.830）；σ=0.005 的总体仍有 0.985，说明失败完全藏在 near-aligned/absence cell。

### 2.2 3D→2D 投影：位姿 + 深度

| level | horizontal | vertical | diagonal | overall | worst cell | absence error |
|---|---:|---:|---:|---:|---:|---:|
| clean | 0.984 ± 0.019 | 0.969 ± 0.007 | 0.998 ± 0.000 | **0.997 ± 0.000** | **0.962 ± 0.003** | 0.031 |
| 1° + 0.5% | 0.878 ± 0.020 | 0.876 ± 0.016 | 0.991 ± 0.000 | 0.983 ± 0.001 | 0.868 ± 0.008 | 0.124 |
| 3° + 1% | 0.736 ± 0.012 | 0.701 ± 0.025 | 0.981 ± 0.002 | 0.964 ± 0.001 | 0.701 ± 0.025 | 0.299 |
| 5° + 2% | 0.569 ± 0.019 | 0.516 ± 0.010 | 0.964 ± 0.003 | 0.937 ± 0.003 | **0.516 ± 0.010** | **0.484** |

- `delta_vs_full` 逐档：1°+0.5% **−0.094**、3°+1% **−0.261**、5°+2% **−0.446**。
- 最高档 overall 仍有 0.937，但 worst-cell 0.516；与像素族同样的模式：**均值掩盖 absence 崩塌**。

### 2.3 归因：位姿 vs 深度

| level | pose-only worst cell | pose-only overall | depth-only worst cell | depth-only overall |
|---|---:|---:|---:|---:|
| 1° / 0.5% | 0.958 ± 0.000 | 0.996 ± 0.000 | 0.845 ± 0.009 | 0.982 ± 0.001 |
| 3° / 1% | 0.950 ± 0.010 | 0.994 ± 0.001 | 0.695 ± 0.002 | 0.965 ± 0.001 |
| 5° / 2% | 0.932 ± 0.002 | 0.991 ± 0.000 | 0.472 ± 0.011 | 0.934 ± 0.001 |

- **位姿噪声几乎无害**：5° 的 worst-cell 仍有 0.932，与 clean 0.962 只差 0.030。原因是场景内所有对象共享同一次刚体旋转：轴对齐关系只在靠近坐标轴时才会翻转，绝大多数 diagonal pair 保持象限。
- **深度噪声主导**：2% 的 depth-only 把 worst-cell 打到 0.472；组合档 0.516 基本由深度贡献。深度误差是绕相机中心的径向缩放，会改变相对向量本身，因此直接冲击 near-aligned 判据。
- 这一点对部署有直接含义：如果真实误差主要来自相机位姿标定，显式几何骨架比预想稳；如果主要来自深度，则 absence cell 会先坏。

---

## 3. 预注册 `u_geom_noise` 判定

| 项 | bar | 最高档实测 | 判定 |
|---|---:|---:|---|
| `worst_cell_acc` | ≥ 0.850 | **0.252**（coord σ=0.04；canonical 档为 0.516） | **失败** |
| `delta_vs_full` | ≥ −0.100 | **−0.681**（coord；canonical 为 −0.446） | **失败** |

- `preregistration.ok = false`，`checked = 1`，mismatches 两条均照实保留。
- 读数规则是运行前固定的“两族最高档取更差”；没有在结果出来后切换档位或改成单族读数。
- 补充：即使只看 canonical 族的最高档，worst-cell 0.516 < 0.85、delta −0.446 < −0.10，同样失败；只看 coord 族更高档也失败。**不是读数规则导致的失败。**
- 最低档的边界情况：canonical 1°+0.5% 的 worst-cell 0.868 ≥ 0.85 且 delta −0.094 ≥ −0.10，**单档可通过**；但 coord σ=0.005 的 0.830 已经失败，所以“最低噪声档完全鲁棒”也不成立。

---

## 4. 判读

1. **overall 不是合适的 readiness 指标。** 噪声升高时 diagonal（占 test pairs 94.8%）保持在 0.908–0.964，把 overall 托在 0.875–0.937；真正的失效集中在 horizontal/vertical 两个 near-aligned cell。U3d 的“absence 0.045”是 clean 几何下的结论，**不能外推到任何有坐标噪声的传感器**。
2. **absence error 由 `σ_rel / eps` 控制，机制是 dead-zone 边界穿越。** 把 σ=0.005/0.01/0.02/0.04 换算成 `σ_rel/eps = 0.35/0.71/1.41/2.83`，实测 absence error 为 0.162/0.297/0.582/0.743，与“真值在 dead zone 内、噪声把它推出去”的一阶模型一致。`eps=0.02` 就是这套几何骨架的**灵敏度尺度**：坐标噪声只要达到 eps 的 ~35%，absence cell 就开始明显掉。
3. **clean-trained 硬阈值骨架没有不确定性通道。** 它学到的近似是 `|Δ| > eps → 谓词为真`；当输入噪声与 eps 同量级时，near-aligned pair 会被系统性地判成“有方向”。若要部署，需要至少一种噪声感知机制：按标定不确定度放宽 dead zone、输出 per-predicate 不确定度、或做噪声增强训练。**本实验没有测这些修复，不能声称它们有效。**
4. **位姿与深度的鲁棒性不对称。** 刚体位姿误差保持场景内相对结构，主要风险是轴附近的翻转；深度误差改变相对向量，直接破坏 near-aligned 判据。真实 RGB-D 的深度误差通常是系统性的（随距离增长、边缘失效），比这里的各向同性高斯更不利于 absence cell，而不是更有利。
5. **clean head 的复现是强校验。** clean pixel 行与 U3 symbolic 逐位一致；clean canonical 行 overall 0.997、worst 0.962，说明 3D→2D 支路本身没有引入任务偏差。噪声后的下降不能归因于数据/标签/划分。
6. **U3d 的 α=0 结论在 clean 条件下仍然成立，但不解决噪声。** α=0 说明 embedding 在 clean 几何上没有可加校准价值；U6 说明真正需要校准的是**坐标不确定性**，不是语义 embedding。这是两个不同的问题。

---

## 5. 可以 / 不可以声称

### 可以声称

1. 在 CLEVR 的 10,000 个测试关系对上，clean 显式几何骨架的 overall exact-4 为 **0.996 ± 0.002**、worst-cell **0.933 ± 0.023**、absence error **0.043**，与 U3 symbolic 逐位一致。
2. 在归一化像素坐标注入 σ=0.005–0.04 的高斯噪声后，overall 从 0.996 降到 **0.875 ± 0.002**，但 worst-cell 从 0.933 降到 **0.252 ± 0.006**、absence error 升到 **0.743**；diagonal 仍为 **0.908 ± 0.002**。
3. 在 3D→2D 桌面投影 + yaw/pitch 5° + depth 2% 下，overall 为 **0.937 ± 0.003**，worst-cell **0.516 ± 0.010**，absence error **0.484**。
4. 位姿噪声单独几乎无害（5° 时 worst-cell **0.932 ± 0.002**）；深度噪声单独主导退化（2% 时 worst-cell **0.472 ± 0.011**）。
5. 预注册 arm `u_geom_noise` **失败**：最高档 worst-cell 0.252、delta −0.681；最低坐标档 σ=0.005 也已跌破 0.85（0.830）。
6. 失效机制是 dead-zone 边界穿越，absence error 随 `σ_rel/eps` 单调上升；clean-trained 硬阈值骨架没有不确定性通道。

### 不可以声称

1. **不可以声称 sim2real 已通过或显式几何骨架已具备真机鲁棒性。** 本轮没有真实 RGB-D、没有真机、没有真实传感器标定；CLEVR 是渲染几何。
2. **不可以声称 CLEVR 渲染 + 高斯噪声等价于真实传感器误差。** 真实深度噪声随距离增长、在边缘/反光/透明物体上失效，通常比各向同性高斯更不利。
3. **不可以声称显式几何骨架在真实部署中会失败/成功。** 这里只测了 clean-trained head；噪声感知训练、dead-zone 放宽、不确定性输出都没有测。
4. **不可以把 overall 0.875/0.937 当作通过依据。** 预注册指标是 worst-cell / absence；均值被 diagonal 主导，掩盖了 near-aligned cell 的崩塌。
5. **不可以把 canonical 族与像素族的绝对数字直接横向比较。** canonical 的 dead zone 在归一化世界 XY 上，物理容差与像素族不同；只有族内趋势可读。
6. **不可以声称位姿噪声在真实系统无害。** 这里位姿误差建模为场景共享的刚体旋转；真实标定误差可能带平移、尺度、非刚体畸变和逐帧抖动，未覆盖。

---

## 6. 产物与复现

### 6.1 产物

| 产物 | 路径 |
|---|---|
| U6 脚本 | `pss/eval/scripts/run_upper_bound_u6.py` |
| 结果 JSON | `eval/results/upper_bound/u6/u6.json` |
| 机器报告 | `eval/results/upper_bound/u6/U6_RESULTS.md` |
| 运行日志 | `eval/results/upper_bound/u6.log` |
| 本报告 | `docs/experiments/PSS-U6-sim2real-readiness.md` |

### 6.2 复现命令

```bash
python -u -m pss.eval.scripts.run_upper_bound_u6 \
  --out eval/results/upper_bound/u6 --device cuda
```

默认：`--train-pairs 30000 --test-pairs 10000 --type-eps 0.02 --epochs 300 --seeds 0 1 2 --camera-points 4000`。运行环境：RTX 4070 Laptop GPU、PyTorch 2.10.0+cu126；6 个 clean head（2 族 × 3 seeds）在单卡上约数分钟完成。

---

## 7. 边界

1. **不是 sim2real，也不是真机。** 没有真实 RGB-D、没有真实机器人、没有接触/遮挡/运动模糊；CLEVR 是渲染数据。标题和结论都只写 “sensor-noise stress test / readiness”。
2. **不是真实传感器标定。** 像素噪声是各向同性高斯；位姿噪声是各向同性高斯 yaw/pitch；深度噪声是各向同性乘性高斯。没有从真实相机/深度传感器拟合误差分布，也没有建模偏置、漂移、相关性和失效模式。
3. **位姿误差按场景共享、深度误差按物体独立。** 这是有意的建模选择：位姿是全局量，深度是逐点量。真实系统中两者可能相关，且位姿误差可能包含平移分量（本实验只测了旋转）。
4. **3D→2D 使用世界 XY 桌面平面。** 相机由 train cache 的 DLT 标定（RMS 4.84 px，主要来自 CLEVR 标注中心与物体原点的偏移），只使用 `R` 和 `C`；不是对真实相机的标定声明。
5. **没有使用 `data/category_pretrain_instance8` 做投影。** 该数据有 `depth` 和单物体 `pose`，但没有对象对、没有相机内参、没有共享场景几何，无法构造关系对；其 DATA_CARD 也明确写着 “No real sensor noise model: add noise/domain randomization before sim2real claims”。CLEVR `3d_coords` 是更合适的配对 3D→2D 来源。
6. **只测了 clean-trained head。** 没有做 noise augmentation、没有训练不确定性头、没有校准 dead zone。因此本实验回答的是“当前 clean 骨架在噪声下有多脆”，不是“噪声感知骨架的最好水平”。
7. **near-aligned cell 样本少。** test 10k pairs 中 horizontal 只有 328、vertical 193（canonical 为 307/331），所以 worst-cell 的 seed std 为 0.006–0.03；绝对值的尾部不确定性比 overall 大。clean 与各噪声档的差距远大于 seed std，但具体数值会随 dead zone 定义和 pair 分布移动。
8. **单一 dead zone。** 全程 `eps=0.02`（归一化像素 span）。没有扫描 eps，因此不能给出“把 dead zone 放宽到多少可以恢复”的结论；这属于下一步。
9. **没有测 embedding 残差分支。** U3d 的 α=0 使 residual 在 clean 条件下等价于坐标骨架；U6 直接用坐标 head。embedding 在噪声下是否能提供有界校准，未测。
10. **3 seeds，不是置信区间。** 表中 std 是 3 个 (head seed, noise draw) 组合的样本标准差；不是 95% CI。

---

## 8. 下一步（不改变本轮结论）

1. **噪声感知训练**：在训练时注入与标定不确定度匹配的坐标噪声，观察 absence error 是否随 `σ_rel/eps` 曲线下移。
2. **dead-zone 扫描**：固定噪声，扫 `eps`，给出“恢复 0.85 worst-cell 需要多宽的 dead zone”的可操作曲线。
3. **不确定性输出**：让 head 输出 per-predicate 置信度，并在 dead-zone 附近 abstain / 交由几何校验，而不是硬判。
4. **真实误差模型**：如果后续拿到真实 RGB-D，用其深度/位姿误差分布替换本轮的各向同性高斯，再重复 U6；在此之前不能把本轮称为 sim2real。
