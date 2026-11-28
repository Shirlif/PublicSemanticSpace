# PSS 架构上界复核（第三阶段）：审计修复 / U3e / U1c / U5b / U6

> 日期：2026-11-28
> 上游：`docs/plans/PSS-架构上界复核与下一阶段.md`、`docs/experiments/PSS-架构上界复核-第二阶段.md`
> 结果目录：`eval/results/upper_bound/{u3e,u1c,u5b,u6}/`、`eval/results/upper_bound/u5_fixed/`
> 状态：五项均已执行；U3e 为单 seed 300 epochs，未完成 3-seed 均值；U1c 为 CLIP 双塔代理，不是真实 VLM；U5b 为仿真回放流，不是真实 fact store；U6 为传感器噪声压力测试，不是 sim2real。

---

## 0. 一句话

**第三阶段把“显式几何”的结论推到了边界：它是必要的（U3d/U3e），但不是充分的（U6）。** 审计修复把 rollback 后的 primary completeness 从 0.800 抬到 1.000；U3e 用位置编码 + 阈值校准通过 `u_comp_patch`（`L_gap +0.009`、absence 0.135），但 patch 内容分支 gate 只有 0.024——通过的是几何，不是学习内容；U1c 的 CLIP 文本双塔代理失败（文本零样本 0.292、dual 开集 0.958/0.744）；U5b 在仿真回放流上确认 query conditioning 已饱和（k≥Q 时 1.000），coverage loss 只在欠预算 k<Q 时有 +0.125~+0.271；U6 显示 clean-trained 硬 dead-zone 骨架在深度噪声下崩塌（worst-cell 0.252 / absence 0.743）。**下一阶段的 f 因子应从“表示容量”转向“几何精度与噪声鲁棒性”。**

---

## 1. 总表

| 项 | 问题 | 主结果 | 预注册 | 判定 |
|---|---|---|---:|---|
| **审计修复** | rollback 是否截断 audit？ | primary completeness **0.800 → 1.000**；rollback exact True；violations 0；`truncated=false` | 回归 12 项通过 | **通过**；残余 version 非单调 |
| **U3e** | patch/位置感知能否替代显式坐标？ | overall **0.987**；`L_gap +0.009`；absence **0.135**；content gate **0.024** | `u_comp_patch` 通过（单 seed） | **几何通过，内容不通过** |
| **U1c** | CLIP 文本双塔能否替代学习 LDA？ | 16 类：LDA 0.973、文本零样本 0.292；dual AUROC 0.958 / reject 0.744 | `u_real_vlm_proxy` 失败 | **不能替代** |
| **U5b** | 回放长历史 + coverage loss | k≥Q 时 query_recency/coverage 均 1.000；k<Q 时 coverage +0.125~+0.271；`as_of` 必需 | 无 checker | **饱和，预算不足时有增量** |
| **U6** | 显式几何在传感器噪声下是否可用？ | clean worst 0.933 → σ=0.04 **0.252**；pose5°+depth2% **0.516**；depth 主导 | `u_geom_noise` 失败 | **clean 上界不可部署** |

---

## 2. 第 1 项：审计 append-only

### 2.1 修复

`pss/runtime/state.py` 的 `CognitiveState.rollback` 不再执行 `self.audit = payload["audit"]`，只恢复状态字段；audit 只在末尾追加一条 `rollback`。完整说明见 `docs/experiments/PSS-U5b-audit-append-only.md`。

### 2.2 复测

| arm | rollback target | completeness | cumulative | rollback exact | violations | truncated |
|---|---|---:|---:|---|---:|---|
| 修复前 primary | v1（pre-denial） | 0.800（4/5） | 1.000 | True | 0 | true |
| **修复后 primary** | v1（pre-denial） | **1.000（5/5）** | 1.000 | True（6/6） | 0 | **false** |
| 修复后 control | v2（latest） | 1.000（5/5） | 1.000 | True（6/6） | 0 | false |

- `permission_denied` 在回滚后保留（`denied_entries_retained_after_rollback=1`）。
- 回归：`tests/test_runtime_audit_append_only.py` + runtime/permissions/semantics 子集 **27 项通过**。

### 2.3 残余

- `AuditEntry.version` 不是单调日志序号：primary 的 `final_audit_versions = [1,2,1,2,3,4]`。若下游需要单调序号，应加独立 `audit_seq`，本轮未改 schema。
- 只修了“rollback 不截断”，没有做哈希链/签名，审计仍不是 tamper-evident。

---

## 3. U3e：patch-token / 位置感知表示

### 3.1 设置

- CLEVR 30k train / 10k test pairs；300 epochs；**单 seed（seed 0）**；crop 64×32 px @224；patch grid 16×16；object 中心 5×5 window；位置编码是 scene-grid 绝对正弦 PE。
- head：positions 2-layer MLP + gated 2-layer cross-attention content，hidden 64，4 heads，88,073 params。
- 阈值校准在训练 10% hold-out 上进行；test 报一次。

### 3.2 主表

| representation | horizontal | vertical | diagonal | overall | worst cell | absence error |
|---|---:|---:|---:|---:|---:|---:|
| symbolic (reference) | 0.967 | 0.957 | 0.998 | 0.996 | 0.933 | 0.043 |
| **patch position-aware** | **0.954** | **0.865** | **0.991** | **0.987** | **0.865** | **0.135** |

- `L_gap = +0.0090`（bar ≤0.05，通过）。
- `absence_cell_error = 0.1347`（bar ≤0.20，通过）。
- `u_comp_patch`：**ok=True**，无 mismatch。

### 3.3 关键判读：通过的是几何，不是内容

- 通过 arm 的信息主要来自 **对象中心 patch 的绝对网格正弦 PE + per-axis dead-zone 阈值校准**；这本质是一个**量化后的几何通道**。
- content gate 仅 **0.0237**，patch-content 分支几乎关闭。
- 消融失败项：
  - PE + content MLP：0.939 / absence **0.534**；
  - 51-token paired-patch transformer：0.880 / absence **0.750**；
  - 独立四谓词阈值校准：absence **0.306**；
  - PE-only 未校准：absence V **0.207**。
- 结论：**纯 patch 内容不能替代显式几何；patch 网格位置编码可以承载 dead zone，但前提是位置量化和 per-axis 阈值校准都在场。**

### 3.4 边界

- **单 seed**，没有 seed 方差；不能与 U3/U3c/U3d 的 3-seed 均值作严格同口径比较。
- 设计选择看过 valA 诊断，最终 test 不是完全盲测。
- `pixel_coords` 仍用于选 crop / PE 原点；不是端到端从图像学到位置。
- CLEVR 渲染数据，不是 sim2real。

---

## 4. U1c：CLIP 双塔 VLM 代理

### 4.1 设置

工作区没有 Qwen-VL/LLaVA/InternVL 权重，U1c 用冻结 CLIP ViT-B/16 视觉塔 + CLIP 文本塔做 two-tower 代理；任务为 16 类 category（不是 384 类 view-instance）。文本模板 4 个，取归一化均值；3 个投影 seed。

### 4.2 主表

| arm | closed-set acc | AUROC | unseen reject @95 |
|---|---:|---:|---:|
| `proto` | 0.973 ± 0.002 | 0.958 ± 0.009 | 0.744 ± 0.031 |
| `lda` | **0.973 ± 0.001** | **0.966 ± 0.002** | **0.873 ± 0.004** |
| `text_zero_shot` | 0.292 ± 0.000 | 0.492 ± 0.000 | 0.023 ± 0.000 |
| `dual_head` | 0.973 ± 0.001 | 0.958 ± 0.009 | 0.744 ± 0.031 |

### 4.3 判读

1. **16 类闭集饱和**：proto 与 LDA 都是 0.973，U1b 在 384 类上的 LDA 增益（+0.110）消失。读出的价值是任务相对的。
2. **CLIP 文本零样本不可用**：闭集 0.292、开集 AUROC 0.492 / reject 0.023；合成 primitive 渲染与 CLIP 文本语义不对齐。
3. **双头无收益**：cal 上选 proto 开集，test 上 0.958/0.744，低于 LDA 单头 0.966/0.873；即便让 LDA 参与开集选择也仍低于 bar。
4. **`u_real_vlm_proxy` 失败**：AUROC 0.9575 < 0.99、reject 0.7439 < 0.95；唯一通过项是 `accuracy_delta_vs_proto=0.0000`（压线）。

### 4.4 边界

- **不是真实 VLM**；只是 CLIP 双塔代理。
- 任务是 16 类 category，不是 384 类 view-instance；文本原型只覆盖 16 个 seen 类名。
- held-out 4 类（cloth/tool/sponge/sheet）与 seen 类语义接近，开集比 U1b 更难；这是任务属性。

---

## 5. U5b：回放长历史流上的 fact selection

### 5.1 数据来源

- 仓库没有真实 fact 流：`data/open_stream` 是渲染帧，`tracking_bench_long` 是 80 帧跟踪基准，`bundles/repro` 排除了派生轨迹。
- 使用 `bundles/dev/s1_occlusion_20/.../seed0/gt.parquet`（MuJoCo/Isaac occlusion **replay**，20 实体 × 1200 步），抽出 **21,168** 条 visibility fact；`real_fact_stream_available=false`。

### 5.2 主表（accuracy，3 seeds）

| regime | Q | k | query_recency | coverage_aware | hot |
|---|---:|---:|---:|---:|---:|
| current | 8 | 4 | 0.542 ± 0.059 | **0.667 ± 0.118** | 0.167 |
| current | 8 | 8/16 | 1.000 | 1.000 | — |
| current | 16 | 4 | 0.292 ± 0.029 | **0.562 ± 0.051** | 0.188 |
| current | 16 | 8 | 0.583 ± 0.078 | **0.854 ± 0.059** | — |
| current | 16 | 16 | 1.000 | 1.000 | — |
| historical | 16 | 4 | 0.250 ± 0.000 | **0.458 ± 0.078** | 0.000 |
| historical | 16 | 8 | 0.521 ± 0.029 | **0.708 ± 0.078** | 0.000 |
| historical | 16 | 16 | 1.000 | 1.000 | 0.000 |

### 5.3 判读

1. **query conditioning 已饱和**：k≥Q 的 6/12 个单元格，query_recency 与 coverage_aware 都是 1.000，**增量 0.000，没有可测空间**。
2. **coverage loss 只在欠预算时有增量**：k<Q 的 6 个单元格 +0.125~+0.271；机制是共享预算分配 / query multiplicity，不是 value resolution。
3. **`learned_query` 与 `query_recency` 在全部 12 个单元格逐位相等**：学习模型没有额外增益。
4. **`as_of` 是必要条件**：historical 下忽略 `as_of` 的 recency 直接掉到 0.000。
5. **coverage 1.000 ≠ accuracy 1.000**：historical 的 coverage-aware + learned 在 coverage=1.000 时 accuracy 最低 0.667；tie-break 可能选到更早的同值 fact，被同 key 更晚的错误 value 覆盖。

### 5.4 边界

- **replay，不是 real**；只有 visibility 一个动态属性，pose 几乎静止，无矛盾写，support 是 oracle 信号。
- 不能外推到真实 fact store、更大 Q/key 集合或生产更新分布。

---

## 6. U6：显式几何骨架在传感器噪声下

### 6.1 设置

- CLEVR 30k/10k pairs；clean 训练一次，所有噪声档复用同一 head（没有 noise augmentation）。
- 主族：归一化像素坐标高斯噪声 σ ∈ {0.005, 0.01, 0.02, 0.04}；相对噪声 σ_rel=σ√2。
- 副族：3D→2D 针孔投影 + yaw/pitch {1°,3°,5°} + depth {0.5%,1%,2%}。
- 读数规则运行前固定：两族最高档取更差；bar `worst_cell_acc ≥0.85`、`delta_vs_full ≥−0.10`。

### 6.2 主表（坐标噪声，3 seeds）

| level | overall | worst cell | absence error |
|---|---:|---:|---:|
| clean | 0.996 ± 0.002 | 0.933 ± 0.023 | 0.043 |
| σ=0.005 | 0.985 ± 0.001 | 0.830 ± 0.022 | 0.162 |
| σ=0.01 | 0.968 ± 0.002 | 0.692 ± 0.026 | 0.297 |
| σ=0.02 | 0.931 ± 0.002 | 0.418 ± 0.014 | 0.582 |
| σ=0.04 | 0.875 ± 0.002 | **0.252 ± 0.006** | **0.743** |

### 6.3 3D→2D 投影族

| level | overall | worst cell | absence error |
|---|---:|---:|---:|
| clean | 0.997 | 0.962 | 0.031 |
| 1°+0.5% | 0.983 | 0.868 | 0.124 |
| 3°+1% | 0.964 | 0.701 | 0.299 |
| 5°+2% | 0.937 | **0.516** | **0.484** |

- 归因：位姿噪声几乎无害（5° 单独 worst 0.932），**深度噪声主导**（2% 单独 worst 0.472）。
- 机制：dead-zone 边界穿越；absence error 近似由 `σ_rel / eps` 控制（0.35→2.83）。
- overall 被占 94.8% 的 diagonal 托住，掩盖 near-aligned cell 的崩塌。

### 6.4 `u_geom_noise` 判定

**失败，两项都未通过**：

- `worst_cell_acc = 0.252`（bar ≥0.85）；
- `delta_vs_full = −0.681`（bar ≥−0.10）；
- 参考：最低坐标档 σ=0.005 的 worst-cell 0.830 也已低于 0.85。

### 6.5 边界

- **不是 sim2real**；没有真实 RGB-D、没有真实传感器标定、CLEVR 是渲染几何。
- 只测 clean-trained 硬阈值骨架；noise-aware 训练 / 软 dead zone / 多帧滤波未测。
- 3D→2D 投影是 DLT 标定的针孔相机（RMS 4.84 px），不是真实内参。

---

## 7. 交叉结论

### 7.1 B1–B3 更新

| 级别 | 第二阶段 | 第三阶段 | 结论 |
|---|---|---|---|
| **B1 局部上界** | 表示相对 | 同 | **成立** |
| **B2 迁移上界** | CLIP/DINOv2 读出成立 | U1c 文本代理失败；U3e 位置通道通过；U6 噪声迁移失败 | **读出 B2 成立；几何 B2 只在 clean 成立** |
| **B3 联合上界** | 显式骨架 α=0 成立 | U3e 位置编码也成立（单 seed）；U6 噪声下失败 | **B3 仅在 clean 几何成立；部署需噪声鲁棒性** |

### 7.2 三条主结论

1. **显式几何 / 位置感知是必要的，但 patch 内容不是替代品。** U3e 的通过来自位置编码 + 阈值校准（content gate 0.024）；纯内容头 absence 0.53–0.75。几何信息必须以某种显式/位置形式存在。
2. **clean 几何的上界不能外推到传感器噪声。** U6 的 worst-cell 在 σ=0.005 已跌破 0.85，σ=0.04 到 0.252；深度噪声主导。硬 dead zone 是 clean 数据的产物，不是部署语义。
3. **文本/VLM 读出不能替代学习判别读出。** U1c 的文本零样本 0.292、dual 0.958/0.744；U1b 的学习 LDA 才是有效读出。16 类任务上 LDA 增益也消失，说明读出价值是任务相对的。

### 7.3 对架构的含义

- 下一阶段的 f 因子应从“表示容量/维度”转向 **几何精度与噪声鲁棒性**：软 dead zone、概率化谓词、`σ/eps` 感知的阈值、多帧滤波/深度融合。
- U6 的失败不是“显式几何错了”，而是“显式几何还停在 clean 数据上”；这正好是 PSS 从合成走向部署的下一道门。
- 审计修复是运行时正确性的一小步；`audit_seq` 单调化、tamper-evidence 仍未做。

---

## 8. 失败项（照实保留，不调 bar）

| arm / 项 | 指标 | bar | 实测 |
|---|---|---|---:|
| `u_real_vlm_proxy` | AUROC | ≥0.990 | 0.9575 |
| `u_real_vlm_proxy` | unseen reject @95 | ≥0.950 | 0.7439 |
| `u_geom_noise` | worst-cell（最高档） | ≥0.850 | 0.252 |
| `u_geom_noise` | delta vs full（最高档） | ≥−0.100 | −0.681 |
| U3e 诊断 | PE+content absence | — | 0.534 |
| U3e 诊断 | paired-patch transformer absence | — | 0.750 |
| U3e 诊断 | 独立阈值校准 absence | — | 0.306 |
| U5b | 无真实 fact 流 | — | replay only |
| 审计 | audit version 单调 | — | 否（[1,2,1,2,3,4]） |

通过项：审计修复复测、U3e `u_comp_patch`（单 seed）、U1b CLIP/DINOv2 dual-head、U2、U3d、U4。

---

## 9. 可以 / 不可以声称

**可以声称**

1. `rollback` 不再截断 audit；primary completeness 0.800→1.000，rollback exact / violations 0 不变。
2. 位置感知 patch 表示（绝对网格 PE + per-axis 阈值校准）在单 seed 上达到 `L_gap +0.009`、absence 0.135，通过 `u_comp_patch`。
3. 通过来自位置编码而非 patch 内容：content gate 0.024，纯内容头 absence 0.53–0.75。
4. 在仿真回放 visibility 流上，query conditioning 在 k≥Q 时已饱和（1.000）；coverage loss 只在 k<Q 时有 +0.125~+0.271，且 `as_of` 是必要条件。
5. 在 clean-trained 硬 dead-zone 骨架上，坐标噪声 σ=0.04 时 worst-cell 0.252、absence 0.743；3D 投影 5°+2% 时 worst-cell 0.516、absence 0.484；深度噪声主导。
6. CLIP 双塔文本代理在 16 类合成 primitive 上不可用（文本零样本 0.292，dual 开集 0.744）；它不是真实 VLM。

**不可以声称**

1. 不可以声称 patch 内容能替代显式几何；数据相反。
2. 不可以声称 U3e 已 3-seed 复现；只有 seed 0，且设计看过 valA 诊断。
3. 不可以声称显式几何骨架已具备部署级噪声鲁棒性；U6 失败。
4. 不可以声称 U6 是 sim2real；没有真实传感器/真机。
5. 不可以声称 U5b 是真实 fact store；是仿真 replay，只有 visibility 一个属性。
6. 不可以声称文本/VLM 读出可以替代学习 LDA；U1c 失败。
7. 不可以声称审计已 tamper-evident 或 version 单调；只修了 rollback 截断。

---

## 10. 下一步

1. **噪声感知几何（最高优先）**：软 dead zone、概率化谓词、`σ/eps` 自适应阈值、多帧滤波/深度融合；用 U6 的噪声档复测，目标 worst-cell ≥0.85。
2. **U3e 3-seed + 盲测**：固定设计后跑 seed 0/1/2，test 完全不参与设计；给出 patch 网格分辨率与 eps 的量化关系。
3. **U1c 真实 VLM**：如果有 Qwen-VL/LLaVA/InternVL 权重，重跑；否则保留 CLIP 代理的负结果，不外推。
4. **U5b 真实 fact store**：接入真实长历史流或多属性更新；coverage-aware 只在 k<Q 时有空间，重点测预算压力场景。
5. **审计 `audit_seq`**：给 `AuditEntry` 加单调序号（兼容性变更需评估），再做哈希链/签名。

---

## 11. 产物与复现

```
pss/runtime/state.py                          # rollback append-only
tests/test_runtime_audit_append_only.py
pss/eval/scripts/run_upper_bound_u3e.py
pss/eval/scripts/run_upper_bound_u1c.py
pss/eval/scripts/run_upper_bound_u5b.py
pss/eval/scripts/run_upper_bound_u6.py
docs/experiments/PSS-U5b-audit-append-only.md
docs/experiments/PSS-U3e-patch-representation.md
docs/experiments/PSS-U1c-vlm-proxy.md
docs/experiments/PSS-U5b-fact-real-stream.md
docs/experiments/PSS-U6-sim2real-readiness.md
eval/results/upper_bound/{u3e,u1c,u5b,u6,u5_fixed}/
```

复现（关键命令）：

```bash
python -m pytest tests/test_runtime_audit_append_only.py -q
python -u -m pss.eval.scripts.run_upper_bound_u5 \
  --out eval/results/upper_bound/u5_fixed/audit_completeness.json

python -u -m pss.eval.scripts.run_upper_bound_u3e \
  --out eval/results/upper_bound/u3e --device cuda \
  --train-pairs 30000 --test-pairs 10000 --epochs 300 --seeds 0

python -u -m pss.eval.scripts.run_upper_bound_u1c \
  --out eval/results/upper_bound/u1c --device cuda \
  --projection-seeds 0 1 2

python -u -m pss.eval.scripts.run_upper_bound_u5b \
  --out eval/results/upper_bound/u5b

python -u -m pss.eval.scripts.run_upper_bound_u6 \
  --out eval/results/upper_bound/u6 --device cuda
```

---

## 12. 一句话

**显式几何是必要的（U3d/U3e），但不是充分的（U6）：clean 几何上的 B3 上界在传感器噪声下崩塌，下一阶段的架构优化是“噪声感知的几何层”，不是继续加表示容量。**
