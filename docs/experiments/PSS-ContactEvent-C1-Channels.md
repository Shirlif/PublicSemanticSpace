# PSS ContactEvent C1：单通道 baseline（force / tactile / geometry / vision / audio）

> 日期：2026-11-29
> 阶段：C1（`docs/plans/PSS-ContactEvent-计划.md` §3、§7、§8）
> 范围：**仿真 proxy**。force/tactile 在 MuJoCo 里接近 oracle；audio 由 force 合成；vision 是渲染 RGB-D/分割，不是真实相机。本文不声称真实传感器、真实音频或 sim2real 性能。

---

## 0. 实际运行

完整配置运行，**没有缩减**（render 未触发超时）：

```bash
python -u -m pss.eval.scripts.run_contact_event_bench \
  --scenarios drop collide near_miss --seeds 0 1 2 --render \
  --arms force force_tuned tactile tactile_tuned geometry vision_motion audio_proxy \
  --out eval/results/contact_event/c1
```

| 项 | 值 |
|---|---|
| 场景 | `drop`、`collide`、`near_miss` |
| seeds | 0, 1, 2（共 9 个场景实例） |
| 帧 | `max_steps=400`，`capture_every=2`，dt=0.002 → 0.004 s/帧 |
| 分辨率 | 96×72，`render=True` |
| GT | `normal_force >= 1.0 N`，`min_duration=0.02`，`max_gap=0.05` |
| 事件匹配 | 同 pair 且（时间 IoU ≥ 0.3 **或** onset 差 ≤ 50 ms） |
| GT 事件数 | 6（每个 seed 的 drop + collide；near_miss 为 0） |

产物：

- `eval/results/contact_event/c1/contact_event.json`
- `eval/results/contact_event/c1/CONTACT_EVENT_REPORT.md`

**口径提醒**：JSON/下表里的 P/R/F1 是“先按 seed 对场景取均值、再对 seed 取均值”。`near_miss` 没有 GT 也没有预测，按 `evaluate_events` 约定 precision=recall=F1=1.0，因此会抬高各臂的平均值。事件计数口径（tp/fp/fn）单独列出；例如 base `tactile` 的事件级 precision 是 6/(6+3)=0.667，而不是 0.833。

---

## 1. 主表（按 seed 聚合，实际运行结果）

| arm | precision | recall | F1 | onset latency (s) | intensity MAE | tp/fp/fn |
|---|---:|---:|---:|---:|---:|---|
| `force` | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 6/0/0 |
| `force_tuned` | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 6/0/0 |
| `tactile` | 0.8333 ± 0.0000 | 1.0000 ± 0.0000 | 0.8889 ± 0.0000 | 0.0000 ± 0.0000 | 0.9356 ± 0.0021 | 6/3/0 |
| `tactile_tuned` | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.9356 ± 0.0021 | 6/0/0 |
| `geometry` | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.0127 ± 0.0025 | 0.9356 ± 0.0021 | 6/0/0 |
| `vision_motion` | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.0100 ± 0.0000 | 0.9612 ± 0.0065 | 6/0/0 |
| `audio_proxy` | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 6/0/0 |

**intensity MAE 的单位警告**：MAE 是相对 GT 峰值法向力的相对误差。`tactile`/`tactile_tuned`/`geometry` 输出的是归一化证据（约 [0,1]），`vision_motion` 输出运动学代理（m/s），它们与牛顿不可通约；0.94/0.96 是尺度错配，**不能**解释为“这些通道力估计差 94%”。`audio_proxy` 的 MAE=0 也有循环论证问题（见 §4）。

---

## 2. 每场景预测区间（实际运行）

### 2.1 drop（三个 seed 的 GT 相同：0.244–0.280 s，峰值 20.88 N）

| arm | 预测区间 (s) | 峰值强度 | 说明 |
|---|---|---|---|
| `force` / `force_tuned` | 0.244–0.280 | 20.88 | 与 GT 完全一致 |
| `tactile` / `tactile_tuned` | 0.244–1.600 | 1.0（归一化） | 落地后球一直压在盒子上，tactile 一直有读数；只靠 onset 匹配为 TP |
| `geometry` | 0.228–1.600 | 1.0（归一化） | 提前 16 ms；同样把静止接触延续到帧尾 |
| `vision_motion` | 0.232–1.600 | 0.46–0.74 m/s | 提前 12 ms；深度点云持续邻近 → 事件过长 |
| `audio_proxy` | 0.244–0.268 | 20.88 | 冲击瞬态 + ring-down，长度刚好过 `min_duration` |

### 2.2 collide（GT 随 seed 变化）

| seed | GT (s) / 峰值 | `force` | `tactile` | `tactile_tuned` | `geometry` | `vision_motion` | `audio_proxy` |
|---|---|---|---|---|---|---|---|
| 0 | 0.084–0.180 / 12.03 | 0.084–0.180 | 0.084–0.200 **+ 0.612–1.600 (FP)** | 0.084–0.200 | 0.080–0.212 | 0.076–0.248 | 0.084–0.196 |
| 1 | 0.096–0.228 / 11.83 | 0.096–0.228 | 0.096–0.240 **+ 0.844–1.600 (FP)** | 0.096–0.240 | 0.080–0.260 | 0.088–0.264 | 0.096–0.244 |
| 2 | 0.088–0.240 / 13.32 | 0.088–0.240 | 0.088–0.244 **+ 0.848–1.600 (FP)** | 0.088–0.244 | 0.080–0.300 | 0.080–0.476 | 0.088–0.256 |

### 2.3 near_miss

三个 seed 的所有 7 个臂都输出 **0 个事件**，没有误报。该场景 objects 全程静止且相距较远（视觉点云最小 3D 距离约 0.26 m），是各通道的共同负样本。

---

## 3. 每通道强弱与盲区

### 3.1 force（`force` / `force_tuned`）

- **强**：本基准的最强通道。事件 F1=1、onset 延迟 0、intensity 精确到 GT 峰值。
- **调参**：`f0` × threshold 的网格上存在很宽的 F1=1 平台（见 §5）。最终取 `f0=1.0 N`、threshold=0.5，使 score 过阈值点正好落在 GT 的 1 N 上；`force_tuned` 与未调 `force` 指标完全相同，**本基准没有可提升空间**。
- **盲区/风险**：MuJoCo 接触力近 oracle，没有真实力传感器的噪声、带宽、漂移和安装矩阵；这个满分是仿真上限，不是真实性能。
- **结论**：C2 融合的主通道，但必须用真实/加噪数据重测后才能宣称鲁棒性。

### 3.2 tactile（`tactile` / `tactile_tuned`）

- **强**：只要两个物体发生非桌面接触，tactile 立刻响应；onset 延迟 0，near_miss 无误报。
- **base 的失败项**：每个 seed 的 collide 都多报一个事件（0.61–0.85 s 起，持续到帧尾）。这是碰撞后球滚落/轻擦产生的弱接触；GT 的 1 N 阈值不把它算作事件，但 tactile map 被**逐物体归一化**成峰值 1.0，幅度信息已在上游丢失，无法用 t0/threshold 区分强弱。
- **tuned 的修正**：`tactile_tuned` 改为 `min(peak_a, peak_b)`（两端都必须有触觉证据）。上述 FP 中只有一个物体有读数，因此被剔除；3 个 seed 的 precision 从 0.833 升到 1.0。
- **盲区**：
  - 归一化后 intensity 只有 0/1，intensity MAE 0.9356 是尺度错配，不是力估计能力；
  - 事件 offset 过长（落地后持续压住 → 到帧尾），只靠 onset 容差匹配；
  - `min` 聚合会漏掉“单侧触觉”的真实接触（例如只有指尖有触觉阵列、物体一侧无传感器），这是用 precision 换 recall 的取舍；
  - 真实触觉阵列的覆盖范围、串扰、漂移未建模。

### 3.3 geometry（`geometry`）

- **强**：F1=1，near_miss 无误报；onset 平均绝对延迟 12.7 ms（seed 间 4–16 ms），多数情况下略**提前**于力阈值（几何先贴近）。
- **盲区**：
  - intensity 是 `clip(1-gap/eps)` 的归一化量，不是力；MAE 0.9356 同样是尺度错配；
  - 依赖 3D 位姿/半径；真实位姿误差、标定误差、遮挡未测；
  - 静止接触会持续预测（drop 到帧尾），offset 无区分力。

### 3.4 vision（`vision_motion`，真实视觉通道）

实现：`pss/events/extractors_vision.py`。只用 `--render` 提供的 `segmentation[...,0]`（geom id）、`depth`、`camera`：

1. 每个分割 label 的 mask 像素用 depth + camera 反投影成 3D 点云；
2. **掩码/点云邻近**：两个物体点云的最小 3D 距离，过 sigmoid；
3. **运动突变**：两端质心速度变化的较小值 `min(|Δv_a|, |Δv_b|)`，用邻近做门控，再过一个短衰减包络；
4. 分数取两者较大值，并加 0.02 的运动 tie-break，使事件的峰值帧落在冲击帧（否则 C0 的 `argmax(score)` 会在静止接触段取到零强度）。

label→instance 的关联只在场景开始时用 `frame.poses` 做一次最近邻匹配（仅用于身份，不参与接触打分）；打分本身只用渲染深度/分割。

- **强**：F1=1，平均 onset 延迟 10 ms（drop 12 ms、collide 8 ms），near_miss 零误报；对遮挡/2D 轮廓重叠不敏感，因为用的是 3D 点云距离。
- **盲区/失败项**：
  - **offset 严重过长**：drop 中球落地后一直贴着盒子，视觉只能看到“邻近”，无法像力一样判断静载是否仍超过 1 N；预测区间到 1.6 s，IoU 很低，靠 onset ≤50 ms 才匹配为 TP；
  - **无法估计力**：intensity 是 m/s 的运动学代理（本次峰值 0.23–0.85 m/s），MAE 0.9612 是单位错配；
  - seed 2 collide 的事件延伸到 0.476 s，把碰撞后的再次邻近/运动也并进来，offset 不可信；
  - 需要渲染；无 `--render` 时返回全零（该臂不适用）；
  - 96×72 下球只有约 40–60 像素，点云很稀；参数 `eps=0.05 m` 是在本基准上选的（见 §5），不是标定过的真实视觉阈值；
  - 真实光照、运动模糊、深度噪声、相机标定误差均未建模。

### 3.5 audio（`audio_proxy`）

- **实现**：`pss/events/arms_channels.py` 的 `AudioProxyExtractor`。由 force 的正向差分（冲击）过 sigmoid，再乘指数 ring-down 包络，使单帧冲击也能跨过 `min_duration=0.02`。
- **强**：F1=1，onset 延迟 0，near_miss 无误报。
- **失败项/边界**：
  - **不是真实音频**；它由 force 合成，只验证“音频作为证据通道”的接口；
  - intensity 直接复用窗口内的 force 峰值，因此 MAE=0 是**循环论证**，不能当作音频幅度估计能力；
  - 真实音频的混响、背景噪声、频带选择、麦克风延迟未测；
  - C0 的占位 `AudioExtractor`（纯 force 差分，无 ring-down）在本基准上产生 **0 个事件**：冲击只持续 1–4 帧，短于 `min_duration`。这是 C0→C1 的一个具体失败项，已由 ring-down 修正。

---

## 4. 失败项汇总

| # | 失败项 | 观察 | 处理/结论 |
|---|---|---|---|
| 1 | base `tactile` 在 collide 每 seed 1 个 FP | 弱接触被逐物体归一化放大到 1.0 | `tactile_tuned` 用两端 `min` 聚合；代价是单侧触觉盲区 |
| 2 | `tactile`/`geometry` intensity MAE ≈0.94 | 输出是归一化证据，不是力 | 只报告，不解读为力误差 |
| 3 | `vision_motion` offset 过长（drop 到 1.6 s） | 视觉邻近无法区分“撞击”与“静载接触” | 记录为视觉盲区；C2/C3 需要形变/力/主动感知 |
| 4 | `vision_motion` intensity 是 m/s | 相机不能测力 | 标为运动学代理，MAE 不作力误差解读 |
| 5 | C0 `vision` 占位返回全零 | 未实现真实视觉 | 由 `vision_motion` 取代（本报告只评估后者） |
| 6 | C0 `audio` 纯差分 0 事件 | 瞬态短于 `min_duration` | `audio_proxy` 加 ring-down |
| 7 | `force_tuned` 与 `force` 完全相同 | F1=1 平台无提升空间 | 保留为 sanity control；不是无效实验 |
| 8 | 所有通道在 near_miss 上“满分” | 空 GT + 零预测时约定为 1.0 | 主表已注明；不要把它当作正向证据 |

---

## 5. 调参过程（实际 sweep，非事后编造）

所有 sweep 都在同一批 `drop/collide/near_miss`、seeds 0–2 上做，**没有 train/cal/test 划分**。这是 C1 的方法学缺口：参数是在评测集上选的，C2 必须改成 train/cal 选参、test 一次。

- **force**：`f0 ∈ {0.5, 0.75, 1.0, 1.5, 2.0}` × `threshold ∈ {0.3, 0.35, 0.4, 0.5, 0.6, 0.667}`。大面积 F1=1；部分组合 F1=0.907（边界抖动导致一个事件被切掉）。最终取 `f0=1.0, threshold=0.5`（与 GT 的 1 N 对齐，延迟 0）。
- **tactile**：base `max` 聚合在 `t0 ∈ {0.25,0.5,1.0,2.0}`、threshold 扫描下最高只有 F1=0.907，FP 来自单侧弱接触，调阈值无法消除。改用 mutual `min` 后，`t0 ∈ {0.25,0.5,1.0}`、threshold ≤0.5 均为 F1=1；取 `t0=0.5, threshold=0.5`。
- **audio**：`d0 ∈ {0.1,0.25,0.5,1.0}` × `decay ∈ {0.85,0.9,0.93,0.95}` × threshold。多个组合 F1=1；取 `d0=0.5, decay=0.9, threshold=0.5`（drop 事件长度约 24 ms，刚好 ≥20 ms）。
- **vision**：粗扫 `(eps, tau) ∈ {(0.03,0.004), (0.04,0.006), (0.05,0.008)}`。`eps=0.03` 在 collide 上漏检（稀疏点云的最小距离约 0.03–0.05 m，阈值太紧）；`eps=0.04` 在 seed 2 collide 产生一个额外事件（F1=0.963）；`eps=0.05, tau=0.008` 在 9 个场景上 F1=1。取 `eps=0.05, tau=0.008`，并明确标注这是本基准上的选择。

---

## 6. 结论（C1 出口：知道每通道强弱与盲区）

1. **force 仍是最强单通道**：F1=1、0 延迟、精确强度；但这是 MuJoCo 近 oracle 的上限，真实传感器性能未知。
2. **tactile 的价值在“有无接触”，不在幅度**：逐物体归一化使幅度不可用，base 会因弱接触误报；两端 `min` 聚合可修掉本基准的 FP，但会带来单侧触觉盲区。
3. **geometry 是可靠的预测/确认通道**：F1=1、约 13 ms 的 onset 误差；但强度是归一化量，且依赖位姿/半径。
4. **vision 可用，但只是仿真 RGB-D 代理**：`vision_motion` 实现了真实的深度+分割通道，F1=1、onset 延迟 10 ms、near_miss 零误报；**但事件 offset 不可信**（静载接触被当作持续事件），且无法估计力。不是真实相机/真实视觉性能。
5. **audio 只是接口代理**：`audio_proxy` F1=1 是由 force 派生出来的必然结果；intensity MAE=0 是循环论证，不能声称真实声学能力。
6. **对 C2 的含义**：单通道已经全部 F1=1（除 base tactile），因此按 §9 的停止规则需要警惕“融合无增量”。C2 的融合收益不应看本基准的 F1 上限，而应看：加噪/遮挡/near-miss 更难的场景、冲突处理、校准（ECE），以及 vision/tactile 在 force 退化时能否兜底。

---

## 7. 边界

- 只有 MuJoCo 图元与 3 个场景（drop/collide/near_miss）；遮挡、主动感知、真实冲突由后续阶段补。
- force/tactile 是仿真近 oracle；audio 是合成；vision 是渲染 proxy。
- 不声称真实机器人、真实传感器、真实音频或 sim2real。
- 参数在评测集上选取；C1 只建立“通道强弱/盲区”的基线，不做统计显著性声明。
