# PSS ContactEvent Coverage：force 退化 × geometry 错配

> 日期：2026-11-29
> 上游：`docs/plans/PSS-ContactEvent-计划.md`、`docs/experiments/PSS-ContactEvent-Report.md`、`docs/experiments/PSS-ContactEvent-C3-Conflict.md`
> 结果：`eval/results/contact_coverage/coverage.json`、`eval/results/contact_coverage/COVERAGE_REPORT.md`
> 状态：48 cells（coarse 24 × 2 seeds + refine 24 × 3 seeds，共 120 cell-seed runs）；结论：**C2 force+geometry 融合相对 `force_noisy` 没有普遍增益**。

---

## 0. 一句话

**在 48-cell 覆盖网格上，`fusion` 只在 1/48 个 cell 超过 `force_noisy` +0.05（drop、σ=0.20、mismatch=1×eps、dropout=0.20、delay=0、4 objects、occlusion）；其余 47/48 个 cell 是打平或下降（4 个 drop cell 下降，最差 −0.300）。相对 `geometry`，fusion 有系统性增益（+0.324 F1，压掉 40/40 near-miss 误报），但 `force_noisy` 本身已经 0 误报且 F1 更高。结论：force 单通道是默认；fusion 只在“force 已知退化、geometry 仍可信”的窄 corner 有证据，而且该 corner 只有 2 seeds、不可外推。**

---

## 1. 设计与加速

### 1.1 因子与水平

`EPS = 0.02 m`；`geometry_mismatch = k × EPS`，即 k=0/0.5/1/2 对应 0/1/2/4 cm。

| 因子 | coarse | refine |
|---|---|---|
| `force sigma` | 0, 0.05, 0.10, 0.20, 0.40 | 0.10, 0.15, 0.20, 0.25, 0.30 |
| `dropout` | 0, 0.05, 0.20 | 同 coarse |
| `delay`（帧） | 0, 1, 2, 4 | 同 coarse |
| `geometry mismatch` | 0, 0.5, 1, 2 × EPS | 0.5, 0.75, 1, 1.25, 1.5 × EPS |
| `occlusion` | False / True | False / True |
| `n_objects` | 2 / 4 | 2 / 4 |
| `kind` | drop / collide / near_miss | drop / collide / near_miss |

`force sigma` / `dropout` / `delay` 只作用于 `NoisyForceExtractor`；场景、接触、GT 与它们无关。

### 1.2 cell 设计

- 每阶段 12 个 physical config（kind × mismatch），每个 physical config 配 2 个 noise setting，得到 24 cells/阶段；共 **48 cells**。
- seeds：coarse 用 0/1（2 seeds），refine 用 0/1/2（3 seeds）；共 **120 cell-seed runs**。
- 物理场景去重后 **60 个 group**；每个 group 只建一次 MuJoCo 场景，然后在缓存 frames 上 sweep 噪声。
- 各因子在 48 cells 上均衡：sigma 4–10 cells/level，dropout 16/level，delay 12/level，occlusion 24/24，n_objects 24/24，kind 16/16。
- cal cells（仅用于选 fusion 阈值）：`[5, 6, 10, 15, 16, 18, 20, 22]`，8 个 coarse cell，覆盖 3 种 kind、mismatch 0 与 2、以及高 sigma；test 为其余 40 cells。

**cell 全表（`cell_id`, stage, kind, n_objects, occlusion, mismatch×EPS, sigma, dropout, delay, seeds）**

| cell | stage | kind | n | occ | mismatch ×EPS | sigma | dropout | delay | seeds |
|---:|---|---|---:|---|---:|---:|---:|---:|---|
| 0 | coarse | near_miss | 2 | False | 1.00 | 0.05 | 0.00 | 1 | 0,1 |
| 1 | coarse | near_miss | 2 | False | 1.00 | 0.00 | 0.20 | 4 | 0,1 |
| 2 | coarse | near_miss | 2 | True | 0.50 | 0.10 | 0.00 | 4 | 0,1 |
| 3 | coarse | near_miss | 2 | True | 0.50 | 0.05 | 0.00 | 2 | 0,1 |
| 4 | coarse | collide | 4 | True | 0.50 | 0.00 | 0.05 | 2 | 0,1 |
| 5 | coarse | collide | 4 | True | 0.50 | 0.40 | 0.20 | 2 | 0,1 |
| 6 | coarse | collide | 4 | False | 0.00 | 0.05 | 0.05 | 2 | 0,1 |
| 7 | coarse | collide | 4 | False | 0.00 | 0.00 | 0.20 | 0 | 0,1 |
| 8 | coarse | drop | 4 | True | 1.00 | 0.00 | 0.00 | 4 | 0,1 |
| 9 | coarse | drop | 4 | True | 1.00 | 0.20 | 0.20 | 0 | 0,1 |
| 10 | coarse | collide | 4 | True | 2.00 | 0.05 | 0.05 | 1 | 0,1 |
| 11 | coarse | collide | 4 | True | 2.00 | 0.20 | 0.05 | 2 | 0,1 |
| 12 | coarse | collide | 2 | False | 1.00 | 0.10 | 0.20 | 4 | 0,1 |
| 13 | coarse | collide | 2 | False | 1.00 | 0.00 | 0.20 | 1 | 0,1 |
| 14 | coarse | drop | 2 | True | 0.50 | 0.20 | 0.20 | 2 | 0,1 |
| 15 | coarse | drop | 2 | True | 0.50 | 0.40 | 0.05 | 0 | 0,1 |
| 16 | coarse | drop | 2 | False | 2.00 | 0.10 | 0.05 | 4 | 0,1 |
| 17 | coarse | drop | 2 | False | 2.00 | 0.20 | 0.00 | 4 | 0,1 |
| 18 | coarse | near_miss | 4 | False | 2.00 | 0.20 | 0.05 | 1 | 0,1 |
| 19 | coarse | near_miss | 4 | False | 2.00 | 0.10 | 0.05 | 0 | 0,1 |
| 20 | coarse | near_miss | 2 | True | 0.00 | 0.05 | 0.00 | 0 | 0,1 |
| 21 | coarse | near_miss | 2 | True | 0.00 | 0.40 | 0.00 | 0 | 0,1 |
| 22 | coarse | drop | 4 | False | 0.00 | 0.40 | 0.00 | 1 | 0,1 |
| 23 | coarse | drop | 4 | False | 0.00 | 0.10 | 0.20 | 1 | 0,1 |
| 24 | refine | drop | 4 | True | 0.75 | 0.10 | 0.00 | 4 | 0,1,2 |
| 25 | refine | drop | 4 | True | 0.75 | 0.20 | 0.00 | 4 | 0,1,2 |
| 26 | refine | near_miss | 2 | True | 1.25 | 0.20 | 0.00 | 2 | 0,1,2 |
| 27 | refine | near_miss | 2 | True | 1.25 | 0.25 | 0.05 | 2 | 0,1,2 |
| 28 | refine | collide | 2 | False | 0.75 | 0.10 | 0.20 | 1 | 0,1,2 |
| 29 | refine | collide | 2 | False | 0.75 | 0.15 | 0.05 | 0 | 0,1,2 |
| 30 | refine | near_miss | 4 | False | 0.50 | 0.10 | 0.05 | 2 | 0,1,2 |
| 31 | refine | near_miss | 4 | False | 0.50 | 0.15 | 0.20 | 1 | 0,1,2 |
| 32 | refine | collide | 2 | True | 1.50 | 0.20 | 0.05 | 2 | 0,1,2 |
| 33 | refine | collide | 2 | True | 1.50 | 0.30 | 0.00 | 2 | 0,1,2 |
| 34 | refine | drop | 2 | False | 1.00 | 0.25 | 0.05 | 4 | 0,1,2 |
| 35 | refine | drop | 2 | False | 1.00 | 0.15 | 0.20 | 4 | 0,1,2 |
| 36 | refine | collide | 4 | False | 1.00 | 0.15 | 0.05 | 0 | 0,1,2 |
| 37 | refine | collide | 4 | False | 1.00 | 0.25 | 0.20 | 2 | 0,1,2 |
| 38 | refine | collide | 2 | True | 1.25 | 0.25 | 0.00 | 1 | 0,1,2 |
| 39 | refine | collide | 2 | True | 1.25 | 0.30 | 0.05 | 0 | 0,1,2 |
| 40 | refine | drop | 2 | False | 1.25 | 0.25 | 0.00 | 4 | 0,1,2 |
| 41 | refine | drop | 2 | False | 1.25 | 0.10 | 0.20 | 0 | 0,1,2 |
| 42 | refine | drop | 4 | True | 0.50 | 0.20 | 0.20 | 1 | 0,1,2 |
| 43 | refine | drop | 4 | True | 0.50 | 0.30 | 0.05 | 0 | 0,1,2 |
| 44 | refine | near_miss | 4 | True | 1.00 | 0.15 | 0.00 | 0 | 0,1,2 |
| 45 | refine | near_miss | 4 | True | 1.00 | 0.20 | 0.20 | 4 | 0,1,2 |
| 46 | refine | near_miss | 4 | False | 1.50 | 0.10 | 0.20 | 1 | 0,1,2 |
| 47 | refine | near_miss | 4 | False | 1.50 | 0.30 | 0.00 | 1 | 0,1,2 |

### 1.3 物理/噪声解耦与并行

- **物理缓存**：按 `(kind, n_objects, occlusion, geometry_mismatch, seed)` 分组；一个 worker 只 build/capture/GT 一次，然后在该 frames 上 sweep 所有分配到的 `(sigma, dropout, delay)`。
- **score 缓存**：clean force、geometry、noisy-force series、fusion score 都按 `(id(frames), pair)` memoize；fusion 只做标准化 logit 的加权。
- **多进程**：`multiprocessing` spawn；本次运行 `--jobs 12`（本机 24 核；脚本默认 `cpu_count-1`）；worker 内 `OMP_NUM_THREADS=1`。
- **配对随机数**：同一 scene/seed 的不同 noise 档共用 `NoisyForceExtractor(seed=seed)` 的随机序列；sigma 只缩放同一组标准正态，dropout 的随机 draw 位置不变。
- **粗后细**：先 24-cell coarse（2 seeds）覆盖全因子，再在 σ 10–30%、mismatch 0.5–1.5×eps 的预期 crossover 区加 24-cell refine（3 seeds）。

### 1.4 采样与早停

- `width=96, height=72`，`max_steps=200`，`capture_every=4`；仿真 `dt=0.002 s`，即每帧 **0.008 s**。
- GT：`normal_force ≥ 1 N`，`min_duration=0.02 s`，`max_gap=0.05 s`。
- 早停：只对出现过 object-object 接触的场景，在接触消失 **25 帧（0.20 s）** 后停止；near-miss 无接触，跑满 200 帧。
- 实际帧数：drop 59–60，collide 47–82（4-object collide 的事件更长），near-miss 200。无 render、无 vision arm。

---

## 2. 臂与口径

### 2.1 三个臂

| arm | 定义 | 事件阈值 |
|---|---|---|
| `force_noisy` | `NoisyForceExtractor(seed=seed, sigma=σ, delay=d, dropout=p)`；quantization 固定 0.1 N（默认，不是因子） | 0.5（即 1 N，与 GT 同阈值） |
| `geometry` | `GeometryExtractor()`，在**被偏移的 `frame.scene_model`** 上计算表面间隙；contacts/GT 保持真实 | 0.5 |
| `fusion` | `FusionExtractor(ChannelExtractor)`：`pair_score = sigmoid(b + w_f·z_f + w_g·z_g)`，`z = (logit(score)-mean)/std`；输入是同一个 `force_noisy` 与 `geometry` | cal 上选得 0.10 |

`FusionExtractor` 的权重来自 **C2 noisy calibration 的 force+geometry 两通道版本**，在本网格之前冻结：

```text
mean   = [0.142047, 0.311929]     # force, geometry
std    = [0.225643, 0.452989]
coef   = [+0.187377, -0.106040]   # w_f, w_g
b      = -1.099747
```

- 来源：`pss.events.fusion.calibrate`，channels=("force","geometry")，drop/collide/near_miss，train seed 0 / cal seed 1，noisy force σ=0.10、delay=1+seed%2、quantization=0.1 N、dropout=0.05（C2 协议）。**没有在本覆盖网格上拟合。**
- **geometry 系数为负**（−0.106）是 C2 的已知 artifact：训练集里 geometry 的“接触”区间远长于 GT（drop 的静置段），logistic 把它学成反证据。因此本 fusion 的行为是“force 减去 geometry”，而不是正证据融合。这一点直接决定了下面的结果。
- 只选事件阈值，不选权重；阈值只在 8 个 cal cells 上选，test cells 只报告。

### 2.2 strict 口径

- **strict**：同 pair + 时间 IoU ≥ 0.5 + **onset ≤ 50 ms** + **offset ≤ 100 ms** 同时成立。
- 这比 C3 的 “pure IoU≥0.5”（`run_contact_conflict` 里用 `onset_tolerance=-1` 关掉 onset 容差）更严：本报告同时要求 onset/offset。因此不能把这里的 geometry 0.252 与 C3 的 0.400 直接对比。
- loose 口径不在本报告使用。

### 2.3 no-GT cell 的约定

near-miss cell 没有 GT，标准 event F1 无定义。为给出每 cell 的 Δ，使用显式约定：

- GT cell：`strict_score = strict F1`；
- no-GT cell：`strict_score = 1.0` 若 FP=0，`0.0` 若 FP>0。

报告里凡称 “每 cell ΔF1” 都指这个 `strict_score`；标准 F1 在 no-GT cell 上记为 `—`。σ×mismatch 表只用 GT cells（drop/collide），near-miss 放在 false-alarm 表。

---

## 3. 结果

### 3.1 总体（strict, micro over 120 runs）

| arm | P | R | strict F1 | tp/fp/fn | onset | offset |
|---|---:|---:|---:|---|---:|---:|
| `force_noisy` | 0.645 | 0.613 | **0.628** | 49/27/31 | 9.1 ms | 15.3 ms |
| `geometry` | 0.206 | 0.325 | 0.252 | 26/100/54 | 6.2 ms | 24.0 ms |
| `fusion` | 0.526 | 0.637 | 0.576 | 51/46/29 | 8.9 ms | 17.6 ms |

- `fusion` 相对 `force_noisy`：**−0.052 F1**（0.576 vs 0.628），不是增益。
- `fusion` 相对 `geometry`：**+0.324 F1**（0.576 vs 0.252），主要来自压掉 geometry 的 near-miss 误报和过长区间。
- 三个臂的 onset/offset 都在 bar 内（≤50/≤100 ms）；但这是“匹配上的事件”的均值，geometry 匹配数只有 26，参考价值有限。

cal/test 拆分（同一 selected threshold）：

| subset | arm | P | R | strict F1 | tp/fp/fn |
|---|---|---:|---:|---:|---|
| cal（8 cells） | `force_noisy` | 0.692 | 0.750 | 0.720 | 9/4/3 |
| cal | `geometry` | 0.176 | 0.250 | 0.207 | 3/14/9 |
| cal | `fusion` | 0.643 | 0.750 | 0.692 | 9/5/3 |
| test（40 cells） | `force_noisy` | 0.635 | 0.588 | **0.611** | 40/23/28 |
| test | `geometry` | 0.211 | 0.338 | 0.260 | 23/86/45 |
| test | `fusion` | 0.506 | 0.618 | 0.556 | 42/41/26 |

test 上 fusion 仍低于 force（−0.055）。cal 上 fusion 与 force 几乎打平（0.692 vs 0.720），这是 cal 选阈值后的乐观值。

### 3.2 按 kind

| kind | force F1 | geometry F1 | fusion F1 | Δ(fusion−force) | Δ(fusion−geometry) |
|---|---:|---:|---:|---:|---:|
| drop | 0.263 | 0.000 | 0.247 | −0.016 | +0.247 |
| collide | 0.975 | 0.605 | 0.975 | 0.000 | +0.370 |
| near_miss | — | — | — | — | — |

- **drop 是硬场景**：事件只有 ~4 帧（0.032 s），在 σ/dropout/delay 下 strict IoU 很容易掉。force 0.263；fusion 0.247，仍略低。
- **collide 是容易场景**：事件 11–45 帧，force/fusion 都 0.975；geometry 0.605。
- near-miss 无 GT，F1 记为 `—`；误报见 3.4。

### 3.3 coarse：sigma × mismatch（只用 GT cells）

表中 n 是该 (σ, mismatch) 组合下的 GT cell 数；Δ 为 micro F1 差。

| σ | mismatch ×EPS | n | force F1 | geometry F1 | fusion F1 | Δ(fusion−force) | Δ(fusion−geometry) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.00 | 1 | 1.000 | 0.500 | 1.000 | 0.000 | +0.500 |
| 0.00 | 0.50 | 1 | 1.000 | 0.800 | 1.000 | 0.000 | +0.200 |
| 0.00 | 1.00 | 2 | 0.500 | 0.500 | 0.500 | 0.000 | 0.000 |
| 0.05 | 0.00 | 1 | 1.000 | 0.500 | 1.000 | 0.000 | +0.500 |
| 0.05 | 2.00 | 1 | 1.000 | 0.000 | 1.000 | 0.000 | +1.000 |
| 0.10 | 0.00 | 1 | 0.667 | 0.000 | 0.667 | 0.000 | +0.667 |
| 0.10 | 1.00 | 1 | 0.500 | 1.000 | 0.500 | 0.000 | −0.500 |
| 0.10 | 2.00 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 0.20 | 0.50 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 0.20 | 1.00 | 1 | 0.000 | 0.000 | **0.667** | **+0.667** | +0.667 |
| 0.20 | 2.00 | 2 | 0.500 | 0.000 | 0.500 | 0.000 | +0.500 |
| 0.40 | 0.00 | 1 | 0.800 | 0.000 | 0.800 | 0.000 | +0.800 |
| 0.40 | 0.50 | 2 | 0.750 | 0.444 | 0.667 | −0.083 | +0.222 |

未列出的 (σ, mismatch) 组合只有 near-miss cells（无 GT），或该组合没有 GT cell。**coarse 里唯一的正增益是 σ=0.20、mismatch=1.00 的 1 个 drop cell（+0.667）。**

### 3.4 refine：预期 crossover 区（σ 0.10–0.30，mismatch 0.5–1.5）

| σ | mismatch ×EPS | n | force F1 | geometry F1 | fusion F1 | Δ(fusion−force) | Δ(fusion−geometry) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.10 | 0.75 | 2 | 0.500 | 0.333 | 0.400 | −0.100 | +0.067 |
| 0.10 | 1.25 | 1 | 0.800 | 0.000 | 0.500 | **−0.300** | +0.500 |
| 0.15 | 0.75 | 1 | 1.000 | 0.667 | 1.000 | 0.000 | +0.333 |
| 0.15 | 1.00 | 2 | 0.545 | 0.308 | 0.462 | −0.084 | +0.154 |
| 0.20 | 0.50 | 1 | 0.667 | 0.000 | 0.500 | −0.167 | +0.500 |
| 0.20 | 0.75 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 0.20 | 1.50 | 1 | 1.000 | 0.667 | 1.000 | 0.000 | +0.333 |
| 0.25 | 1.00 | 2 | 0.500 | 0.308 | 0.462 | −0.038 | +0.154 |
| 0.25 | 1.25 | 2 | 0.500 | 0.308 | 0.500 | 0.000 | +0.192 |
| 0.30 | 0.50 | 1 | 0.571 | 0.000 | 0.500 | −0.071 | +0.500 |
| 0.30 | 1.25 | 1 | 1.000 | 0.571 | 1.000 | 0.000 | +0.429 |
| 0.30 | 1.50 | 1 | 1.000 | 0.667 | 1.000 | 0.000 | +0.333 |

**refine 区没有任何 Δ(fusion−force) ≥ +0.05；最大是 0.000，stage 总体 −0.081。** 预期 crossover 区没有出现 fusion 的系统性增益。

### 3.5 边际因子（GT cells；注意非全因子，边际会被其他因子混杂）

| σ | force F1 | geometry F1 | fusion F1 | Δ(fusion−force) |
|---:|---:|---:|---:|---:|
| 0.00 | 0.750 | 0.588 | 0.750 | 0.000 |
| 0.05 | 1.000 | 0.250 | 1.000 | 0.000 |
| 0.10 | 0.500 | 0.267 | 0.412 | −0.088 |
| 0.15 | 0.706 | 0.421 | 0.632 | −0.074 |
| 0.20 | 0.452 | 0.118 | 0.439 | −0.013 |
| 0.25 | 0.500 | 0.308 | 0.480 | −0.020 |
| 0.30 | 0.842 | 0.421 | 0.800 | −0.042 |
| 0.40 | 0.769 | 0.308 | 0.714 | −0.055 |

| mismatch ×EPS | force F1 | geometry F1 | fusion F1 | Δ(fusion−force) |
|---:|---:|---:|---:|---:|
| 0.00 | 0.875 | 0.250 | 0.875 | 0.000 |
| 0.50 | 0.643 | 0.267 | 0.529 | −0.113 |
| 0.75 | 0.500 | 0.333 | 0.414 | −0.086 |
| 1.00 | 0.486 | 0.381 | 0.500 | +0.014 |
| 1.25 | 0.696 | 0.308 | 0.615 | −0.080 |
| 1.50 | 1.000 | 0.667 | 1.000 | 0.000 |
| 2.00 | 0.500 | 0.000 | 0.500 | 0.000 |

| 因子 | level | force F1 | geometry F1 | fusion F1 | Δ(fusion−force) |
|---|---|---:|---:|---:|---:|
| occlusion | False | 0.615 | 0.341 | 0.571 | −0.044 |
| occlusion | True | 0.641 | 0.286 | 0.581 | −0.060 |
| n_objects | 2 | 0.593 | 0.372 | 0.533 | −0.059 |
| n_objects | 4 | 0.667 | 0.250 | 0.621 | −0.046 |
| dropout | 0.00 | 0.372 | 0.186 | 0.333 | −0.039 |
| dropout | 0.05 | 0.754 | 0.349 | 0.719 | −0.035 |
| dropout | 0.20 | 0.692 | 0.367 | 0.615 | −0.077 |
| delay | 0 | 0.800 | 0.318 | 0.735 | −0.065 |
| delay | 1 | 0.882 | 0.343 | 0.833 | −0.049 |
| delay | 2 | 0.919 | 0.537 | 0.872 | −0.047 |
| delay | 4 | 0.044 | 0.087 | 0.038 | −0.007 |
| stage | coarse | 0.623 | 0.303 | 0.618 | −0.005 |
| stage | refine | 0.632 | 0.320 | 0.550 | −0.081 |

- 所有 σ 边际的 Δ 都是 ≤0；mismatch 1.00 的 +0.014 也是被那一个 gain cell 拉起来的（<0.05）。
- **occlusion 是 null factor**：True/False 的差异来自 cell 组成，不是遮挡本身；本实验没有 vision arm、没有 render，occluder 只在物理场景里被钉住，force/geometry 不受影响。
- **delay=4 是死区**：force F1 0.044、fusion 0.038。4 帧 × 0.008 s = 32 ms 的延迟本身在 onset 容差内，但 drop 事件只有 ~4 帧，IoU≥0.5 直接失败。这是 strict IoU 与短事件的交互，不是融合问题。
- `n_objects=4` 的 force 更高，部分因为 4-object collide 的 GT 事件更长（0.096–0.456 s vs 2-object 的 0.088–0.176 s）；n_objects 与事件时长/动力学混杂，不是干净的因果因子。

### 3.6 False alarm（near-miss，40 runs）

| arm | runs | FP events | runs with FP | run FA rate | event FA rate |
|---|---:|---:|---:|---:|---:|
| `force_noisy` | 40 | 0 | 0 | 0.0% | — |
| `geometry` | 40 | 40 | 40 | **100.0%** | 100.0% |
| `fusion` | 40 | 0 | 0 | 0.0% | — |

- 硬 near-miss 的中心距 0.115 m，真实表面间隙约 0.005 m < EPS=0.02 m；mismatch 0 时 geometry 已经 >0.5，mismatch 1–2 时进一步升到 0.91–0.99。
- fusion 的负 geometry 权重把 geometry 的“接触”压下去，40/40 全部静默；但 `force_noisy` 本来就没有 object-object contact，也是 0 FP。
- 因此 **fusion 的 false-alarm 价值只存在于“相对 geometry”**；相对 force_noisy 没有增量。

### 3.7 Onset / offset（匹配事件，按 tp 加权）

| arm | onset | offset | 匹配数 tp |
|---|---:|---:|---:|
| `force_noisy` | 9.1 ms | 15.3 ms | 49 |
| `geometry` | 6.2 ms | 24.0 ms | 26 |
| `fusion` | 8.9 ms | 17.6 ms | 51 |

三者的 onset/offset 都远低于计划 §7 的 50/100 ms bar。geometry 的 onset 小但 offset 大，符合它“区间过长”的已知问题；fusion 保留了 force 的时序，offset 比 force 多 2.3 ms。

### 3.8 阈值选择与敏感性

cal cells 上的阈值选择（micro strict F1；FP 已计入分母）：

| threshold | cal F1 | cal P | cal R | tp/fp/fn |
|---:|---:|---:|---:|---|
| 0.05 | 0.500 | 0.500 | 0.500 | 6/6/6 |
| **0.10** | **0.692** | 0.643 | 0.750 | 9/5/3 |
| 0.15 | 0.636 | 0.700 | 0.583 | 7/3/5 |
| 0.20 | 0.545 | 0.600 | 0.500 | 6/4/6 |
| 0.25 | 0.364 | 0.400 | 0.333 | 4/6/8 |
| 0.30 | 0.364 | 0.400 | 0.333 | 4/6/8 |
| 0.40 | 0.118 | 0.200 | 0.083 | 1/4/11 |
| 0.50 | 0.000 | 0.000 | 0.000 | 0/4/12 |
| 0.60 | 0.000 | 0.000 | 0.000 | 0/2/12 |
| 0.75 | 0.000 | — | 0.000 | 0/0/12 |
| 0.90 | 0.000 | — | 0.000 | 0/0/12 |

cal 选 **0.10**。作为事后诊断（**不用于选择**），test cells 上各阈值的 fusion F1：

| threshold | 0.10（选中） | 0.15 | 0.20 | 0.25 | 0.30 |
|---|---:|---:|---:|---:|---:|
| test fusion F1 | 0.556 | 0.561 | **0.579** | 0.554 | 0.505 |

即使取事后 test 最优的 0.20（0.579），仍低于同一 test 子集的 `force_noisy` 0.611。所以阈值选择不是结论的反例；结论对阈值不敏感。

---

## 4. Crossover / operating envelope

### 4.1 每 cell Δ

按 `strict_score`（no-GT cell 用 1/0 约定）统计 48 cells：

| 比较 | Δ ≥ +0.05 | 打平（|Δ|<0.05） | 下降 | max Δ | min Δ |
|---|---:|---:|---:|---:|---:|
| `fusion − force_noisy` | **1** | 43 | **4** | **+0.667** | **−0.300** |
| `fusion − geometry` | 37 | 10 | 1 | **+1.000** | **−0.500** |

- `fusion − force_noisy` 的 1 个正增益就是 cell 9；4 个下降全部是 drop（cell 15/41/42/43）。
- `fusion − geometry` 的 37 个正增益主要是 near-miss（geometry 100% FP、fusion 0 FP）和 drop（geometry 长区间/漏检）；唯一的下降是 cell 12（collide，σ=0.10，mismatch=1.00，dropout=0.20，delay=4），geometry F1 1.000 vs fusion 0.500——delay 4 伤了 force/fusion，而没有延迟的 geometry 反而对。

### 4.2 唯一正增益 cell

| cell | kind | σ | mismatch | dropout | delay | n_obj | occ | force tp/fp/fn → F1 | fusion tp/fp/fn → F1 | Δ |
|---:|---|---:|---:|---:|---:|---:|---|---|---|---:|
| 9 | drop | 0.20 | 1.00 ×EPS | 0.20 | 0 | 4 | True | 0/0/2 → 0.000 | 2/2/0 → 0.667 | **+0.667** |

- 2 个 seeds 都一致：force 完全漏掉唯一的 GT 事件；fusion 两个都检出，但各多 1 个 FP。
- 机制：force 在 20% dropout + σ=0.20 下漏检；mismatch 1×eps 时 geometry 仍有部分证据（近场），负 geometry 系数没有把 fusion 压死，于是 fusion 把事件捞回来。
- 这不是“σ 越高越好”的规律：同是 σ=0.20，mismatch 0.50 的 cell 42 Δ=−0.167，mismatch 1.25 的 cell 41 Δ=−0.300；σ=0.10–0.30 的 refine 区总体 −0.081。**所以这是一个点，不是区域。**

### 4.3 4 个下降 cell（全部 drop）

| cell | σ | mismatch | dropout | delay | n_obj | occ | force F1 | fusion F1 | Δ |
|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 41 | 0.10 | 1.25 | 0.20 | 0 | 2 | False | 0.800 | 0.500 | **−0.300** |
| 42 | 0.20 | 0.50 | 0.20 | 1 | 4 | True | 0.667 | 0.500 | −0.167 |
| 15 | 0.40 | 0.50 | 0.05 | 0 | 2 | True | 0.500 | 0.400 | −0.100 |
| 43 | 0.30 | 0.50 | 0.05 | 0 | 4 | True | 0.571 | 0.500 | −0.071 |

共同点：force 还能检出事件，fusion 的 TP 不变（或几乎不变），但 FP 增加 1–2 个。负 geometry 系数在 geometry 说“无接触”时反而抬高融合概率，drop 的短事件/静置段附近就会多出事件。

### 4.4 “force 单通道够用” vs “需要 fusion”

- **force 单通道够用**：47/48 cells。这里的“够用”指 **fusion 没有超过 force**，不是指 force 的绝对 F1 高；例如 delay=4 的 drop cells，force 也只有 0.044，fusion 0.038，二者都不可用。具体：
  - collide 汇总：force = fusion = 0.975（micro）；16 个 collide cells 逐 cell 的 Δ 都是 0（force 与 fusion 的 strict_score 相同）。
  - 16 个 near-miss cells：force/fusion 都是 0 FP，strict_score 都是 1，逐 cell Δ=0。
  - 16 个 drop cells 中 11 个打平；4 个下降；1 个上升。
- **需要 fusion**：只有 cell 9 一个点（drop、σ=0.20、mismatch=1×EPS、dropout=0.20、delay=0、4 objects、occlusion）。它不能推广为“σ≥20% 或 mismatch≥1×EPS 就需要 fusion”，因为：
  - 同一 σ=0.20 下，mismatch 0.50/1.25 的 drop cells fusion 更差；
  - refine 的 σ=0.10–0.30、mismatch=0.5–1.5 区总体 Δ=−0.081；
  - 47/48 cells 的 Δ≤0。
- **相对 geometry 的“需要 fusion”**：如果只用 geometry 一个通道，near-miss 40/40 误报、drop F1=0、总体 0.252；fusion 把这些压到 0.576。但 `force_noisy` 单独就 0.628 且 0 误报，所以这不是“需要 fusion”，而是“不要只用 geometry”。
- **operating envelope 的准确写法**：本网格没有找到稳健的 fusion operating envelope；唯一正增益是一个高噪声 + 高 dropout + mismatch≈1×eps + 零延迟 + 4-object + occlusion 的 drop corner，n=2 seeds，且相邻 cell 反号。按停止规则，这只能写成“无普遍增益；一个孤立 cell 有 +0.667”。

---

## 5. 失败项、边界与未测

1. **force 近 oracle**：GT 来自 MuJoCo contact force；clean force 与 GT 同源。`force_noisy` 是模拟退化（相对高斯、延迟、量化、dropout），不是真实传感器。
2. **C2 的负 geometry 系数**：fusion 不是正证据融合，而是“force 减去 geometry”。这既解释了它在 near-miss 上压掉 geometry 误报，也解释了它在 drop 上多出 FP。若改用正权重/等权，结果可能不同；本次按停止规则没有加新臂。
3. **drop 事件太短**：0.008 s/帧下 drop 只有 ~4 帧；delay=4（32 ms）就足以让 strict IoU 失败。delay=4 的 F1（force 0.044、fusion 0.038）不能读成“融合退化”，应读成“该采样/延迟组合超出 strict 短事件包络”。
4. **geometry mismatch 是单向、误差放大**：只偏移 `ball_00` 的 `scene_model`，drop 用 +z（远离）、collide/near-miss 用 −x（远离/靠近）。这是故意制造 geometry 失败的方向，不是随机跟踪误差；换方向/换物体/换半径膨胀会得到不同结果。C3 的 pose offset/radius inflation 是另一类错配。
5. **occlusion 是 null factor**：没有 vision arm、没有 render；occluder 只被钉在物理场景里，force/geometry 不受影响。本实验不能验证视觉遮挡下的融合；C3 的遮挡可见性结论仍然独立成立。
6. **非全因子**：48 cells 是 space-filling，不是全因子；边际表（by sigma/dropout/delay/kind）会被其他因子混杂，不能当因果主效应。σ×mismatch 表更可控，但很多组合只有 1 个 GT cell。
7. **小样本**：coarse 2 seeds、refine 3 seeds；cell 级 Δ 的方差没有单独估计。唯一正增益 cell 只有 2 seeds，且 2 个 seed 都是 force 漏检、fusion 检出 + 1 FP。
8. **strict 口径更严**：本报告同时要求 onset≤50 ms 与 offset≤100 ms；C3 的 “strict” 是 pure IoU≥0.5。数字不能直接横向比较。
9. **采样不同**：C2/C3 用 400 steps × 2（0.004 s/帧），本覆盖用 200 × 4（0.008 s/帧）+ 早停。事件边界更粗，短事件更容易掉 IoU；绝对 F1 与 C2/C3 不可直接比。
10. **阈值只在 8 个 cal cells 上选**：cal 选 0.10；事后 test 最优 0.20 也只有 0.579，仍低于 force 0.611。阈值不是结论的反例，但 cal 集小，阈值本身不稳定。
11. **不是真实机器人/真实传感器**：MuJoCo 图元是 proxy；不声称 sim2real。
12. **早停可能截断延迟的第二事件**：接触消失 25 帧后停止；本网格的 drop/collide 只有一个主事件，但若真实场景有 >0.2 s 后的第二次接触，会被漏掉。near-miss 无接触，跑满 200 帧。

---

## 6. 复现

```bash
python -u -m pss.eval.scripts.run_contact_coverage \
  --out eval/results/contact_coverage --jobs 12
```

- 默认 `--max-steps 200 --capture-every 4 --width 96 --height 72`，与报告一致。
- 产物：
  - `eval/results/contact_coverage/coverage.json`（design、rows、per-cell、聚合、crossover）
  - `eval/results/contact_coverage/COVERAGE_REPORT.md`（脚本自动生成的简表）

---

## 7. 新建 / 修改文件

- `pss/events/scenarios_grid.py`（新建）：参数化 drop/collide/near-miss 网格场景，物理 key 不含噪声因子，含 occluder 钉住、geometry mismatch 只改 `scene_model`、早停。
- `pss/eval/scripts/run_contact_coverage.py`（新建）：48-cell coarse/refine 设计、物理分组缓存、噪声 sweep、score 缓存、多进程、strict 指标、cal 阈值选择、聚合与 crossover。
- `docs/experiments/PSS-ContactEvent-Coverage.md`（本报告）。
- 未修改 C0–C5 任何文件；未新增通道或 arm。

---

## 8. 一句话

**覆盖网格的结论是负面的：`force_noisy` 单通道在 47/48 个 cell 上不输 `fusion`（1 个 +0.667 的孤立 drop cell，4 个下降，最差 −0.300）；`fusion` 相对 `geometry` 有 +0.324 和 40/40 误报抑制，但那是因为 geometry 单独不可用，而 force 已经能做到同样的事。所以 ContactEvent 的默认仍应是 force/opt-in fusion，而不是“fusion 普遍更好”。**
