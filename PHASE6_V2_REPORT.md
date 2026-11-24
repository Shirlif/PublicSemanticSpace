# Phase 6 第一版验证报告（canonical 栈）

> 日期：2026-11-22
> 数据：`data/v1`（16 episode + 6 occlusion）
> 专家：`checkpoints/category_pretrain_experts`（R0 主线）
> 投影：`checkpoints/phase45_projection`（材质感知 canonical）
> 输出：`checkpoints/memory_v2/`

---

## 1. 运行

```bash
python -m pss.memory.scripts.run_tracking \
  --data data/v1 \
  --experts checkpoints/category_pretrain_experts \
  --projections checkpoints/phase45_projection \
  --out checkpoints/memory_v2 \
  --occlusion 4 --episodes 2 --seed 0
```

跑的是 4 条 occlusion + 2 条 episode，姿势噪声为模拟 SLAM 噪声。

## 2. 门禁结果

| 门禁 | 阈值 | 结果 | 判定 |
|---|---:|---:|---|
| 5 秒遮挡后 ID retention | ≥ 95% | occlusion **1.000**；aggregate **0.972** | **PASS** |
| 焦点切换召回 | ≥ 95% | **1.000** | **PASS** |
| 唤醒延迟 | 满足预算 | 0.3 frames | **PASS** |
| 200 休眠 + 4 激活 单步延迟 | 与 4 实体同量级 | 0.122 ms/frame | **PASS** |
| episode reappearance | 参考项 | 0.667 aggregate | 需关注 |

## 3. 明细

### Tracking

| 序列 | objects | frames | ID retention | switches | reappear | wake latency | ID inflation |
|---|---:|---:|---:|---:|---:|---:|---:|
| occ_0000–0003（4 条） | 3 | 8 | 1.000 | 0.00 | 1.000 | 0.0 | 1.00 |
| ep_0000 | 4 | 5 | 0.917 | 0.25 | 0.000 | 1.0 | 1.25 |
| ep_0001 | 4 | 5 | 0.917 | 0.25 | 0.000 | 1.0 | 1.25 |
| **aggregate** | — | — | **0.972** | 0.08 | 0.667 | 0.3 | 1.08 |

### Working memory

- capacity 4；tracks 6
- focus switch recall **1.000**
- switch latency p50 0.010 ms / p95 0.043 ms

### Scaling

- entities 200，active 4
- pair counts：active 6 vs all 19,900
- spatial query 0.122 ms/frame

## 4. 结论

- **Phase 6 第一版通过**：遮挡 ID retention 1.000，aggregate 0.972；focus recall 1.000；延迟和 scaling 都达标。
- **episode reappearance 0.0 是已知压力项**：episode 里有 teleport-like lift/place，不是真实操作；occlusion 序列才是 5 秒遮挡门禁。
- **第一版 tracking 用 GT category**；端到端原型识别见第 6 节。

## 5. 下一步

1. **把 `phase45_prototypes` 和 `recognize()` 接进 `pss/memory/perception.py`**，用识别结果替代 GT category；重跑 tracking，看识别误差对 ID retention 的影响。已完成，见第 6 节。
2. 增加 episode/occlusion 数量。**fragment/decomposition 规则已完成**，见 `PHASE6_DECOMPOSITION_REPORT.md`。
3. 进 Phase 7：关系、动作算子、世界模型、规划。

---

## 6. 端到端：接入 Phase 5 原型

### 6.1 改动

- `pss/memory/perception.py`：加载 `prototypes.npz` 和 `metrics.json`；每帧调用 `recognize()`，写入 `category` / `category_score` / `unknown` / `conflict`；零信号通道视为传感器缺失，不参与融合。
- **可靠性加权**：融合权重改用原型库 `single_modality.*.test_acc`，而不是 Phase 5 在预训练域调出的 `weights`。场景域里 tactile 单模态准确率只有 0.056，固定 0.5 权重会把融合分拉低。可用 `--recognition-weights tuned` 复现 Phase 5 权重。
- **裁剪一致**：tracking 的 object crop 打开 `preserve_aspect=True`，与 `category_pretrain` 的训练/建库流程一致。修复前 vision 单模态在场景里明显偏低。
- **类别进入跟踪**：`Tracker` 在双方类别都已知且冲突时加 0.05 代价，re-id 加 0.1 代价；匹配后更新 track category。
- **数据**：`data/tracking_v2`，用当前仿真器重新生成，`--tactile-mode material`。`data/v1` 的 tactile 是 contact-derived，无接触时全 0，且类别名过期（`toy_fruit`），不适合端到端识别。
- **评测切分**：`run_tracking --split test`，只用 held-out 序列（occ_0004–0005，ep_0012–0015）。

```bash
python -m pss.sim.scripts.generate_dataset \
  --out data/tracking_v2 --scale small --seed 0 --tactile-mode material

python -m pss.memory.scripts.run_tracking \
  --data data/tracking_v2 \
  --experts checkpoints/category_pretrain_experts \
  --projections checkpoints/phase45_projection \
  --prototypes checkpoints/phase45_prototypes \
  --split test --occlusion 100 --episodes 100 \
  --out checkpoints/memory_v2_proto
```

### 6.2 Tracking：GT category vs 原型识别

| 配置 | ID retention | occlusion | episode | switches | reappear | wake latency | ID inflation |
|---|---:|---:|---:|---:|---:|---:|---:|
| GT category | 0.944 | 1.000 | 0.917 | 0.17 | 0.333 | 0.7 | 1.17 |
| Phase 5 原型 | 0.944 | 1.000 | 0.917 | 0.17 | 0.333 | 0.7 | 1.17 |

两行完全一致。原因：原型识别只在双方类别都已知时参与代价；本次 128 个观测里被接受的 28 个全部正确，没有产生类别冲突，所以关联决策不变。tracking 的主信号仍是 pose + vision appearance。

### 6.3 Recognition（test split，128 观测）

| 指标 | 值 |
|---|---:|
| top-1（unknown 记为错） | 0.219 |
| accepted precision（known-only） | **1.000**（28/28） |
| unknown rate | 0.781 |
| conflict rate | 0.664 |

拆开看：

- seen 类别 87 个观测：ranking top-1 **0.851**；accepted 28 个（32.2%），accepted 全对；其余 59 个被拒。
- unseen 类别 41 个观测：reject **1.000**。
- 所以 0.781 的 unknown 里，41 个是应该拒的 open-set，59 个是域偏移导致的 seen 误拒。

### 6.4 诊断

1. **场景域偏移是主因**。`category_pretrain` 是单物体、居中、占满画面；场景 crop 小、带背景、偶尔被遮挡。vision 单模态在预训练 test 是 0.969，在场景 seen 观测上约 0.7，全场 0.522。
2. **tactile proxy 不承载类别**。8×8 材质纹理的 tactile 单模态 test acc 0.208（预训练）和 0.056（场景）。可靠性加权把 seen ranking top-1 从 0.709 提到 **0.789**，但 tactile 仍不能提供独立类别证据。
3. **阈值偏保守，不是坏了**。0.732 是 Phase 5 在预训练 val 上定的。场景分数整体下移，导致 seen coverage 只有 32%，但 unseen reject 100%。这是 precision/coverage 取舍，不是 open-set 失效。
4. `conflict 0.664` 主要来自 vision 与 geometry 在场景 crop 上不一致；当前 conflict 只记录不决策，不影响 tracking。

### 6.5 下一步（按优先级）

1. **场景上下文专家**：用多物体场景 crop 做 R1.5 adapter 或重训 vision/geometry expert。这是提升端到端识别覆盖率的根本解，不动 prototype 机制。
2. **域内阈值校准**：在 val 上按目标 seen false-reject 选阈值，画出 precision/coverage 曲线，再冻结到 test。当前 0.732 不适合场景域。
3. **tactile 定位**：把它当材质/接触信号，而不是类别证据；类别融合用 vision+geometry，tactile 单独进材质槽位。

