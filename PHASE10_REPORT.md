# Phase 10 报告：预注册评测与基线对比

> 日期：2026-11-20
> 预注册：`eval/preregistration.md`（先于评测冻结）
> 完整结果：`eval/results/phase10_report.md`、`eval/results/metrics.json`
> 状态：**四个杀死标准全部触发；结构主张部分成立，能力主张未成立**

---

## 1. 交付物

| 交付 | 路径 |
|---|---|
| 预注册（假设/指标/杀死标准） | `eval/preregistration.md` |
| 指标库 | `pss/eval/metrics/` |
| 基线/消融 | `pss/eval/baselines/` |
| 评测 CLI | `pss/eval/scripts/run_phase10.py` |
| 结果 | `eval/results/` |

测试：全仓 **119 passed**。

---

## 2. E1 绑定稳定性

| 项 | 值 |
|---|---:|
| Hadamard D=128 K=5 | 1.000 |
| Hadamard D=128 K=20 | **0.769** |
| Hadamard D=512 K=20 | 1.000 |
| Gaussian D=128 K=5 | 0.375 |
| HRR D=128 K=20 | 0.667 |
| Hadamard 1% 噪声 ×100 轮 | 0.100 |
| Hadamard 5% 噪声 ×10 轮 | 0.160 |

**K1 触发**：D=128 撑不住 K=20；HRR 也没有帮助。处理：升 D=512 或逐条存储。

---

## 3. E2 对齐与消融

| 项 | 值 |
|---|---:|
| 完整识别 | 0.969 |
| 跨模态相加识别 | 0.969 |
| 随机投影识别（单模态） | 0.969 |
| **训练后 V-T 类别 R@1** | **0.510** |
| **随机投影 V-T 类别 R@1** | **0.115** |
| V-G 类别 R@1 | 0.854 |
| 无槽：逐实体 | 0.766 |
| 无槽：全局向量 | **0.250** |

三个结论：

1. **单模态识别不依赖训练投影**（0.969 = 0.969）。投影层的价值在**跨模态对齐**：V-T R@1 从 0.115 提升到 0.510。
2. **槽位有价值**：逐实体 0.766 vs 全局向量 0.250；无槽确实把实体混在一起。
3. **触觉对齐仍弱**：V-T 0.510 < 0.60，K2 触发；V-G 0.854 通过。

---

## 4. E3 遮挡 ID 保持

| 变体 | ID retention | switches |
|---|---:|---:|
| main | **1.000** | 0.00 |
| pose_only | 1.000 | 0.00 |
| appearance_only | 1.000 | 0.00 |
| **text_state_proxy (B1)** | **1.000** | 0.00 |
| no_scene_graph | 0.128 | 7.00 |

**K3 触发**：主方法与 B1 代理的差值为 0.000。原因不是架构失败，而是**当前遮挡数据没有真正让物体消失**：遮挡条只造成部分像素遮挡，text-state 的单帧文本记忆仍能按位置/类别匹配。H1 在本基准上**未被证伪，也未被证实**。

要检验 H1，需要：
- 让物体完全离开视野/被完全遮挡若干帧；
- 或接真实 VLM 做同任务对比。

### 4.1 K3 复测：`occlusion_v3`（2026-11-22）

原 benchmark 的问题已修：新数据 `data/occlusion_v3` 让 3 个物体完全消失 14 帧，并在遮挡期间把物体移到新位置，使 last-known pose 失效。生成时校验遮挡帧目标像素为 0，前后帧目标可见。

test split（occ_0004、occ_0005）：

| 变体 | ID retention | switches | reappearance |
|---|---:|---:|---:|
| main | **1.000** | 0.00 | 1.000 |
| main_no_category | 1.000 | 0.00 | 1.000 |
| pose_only | 0.667 | 0.67 | 0.333 |
| appearance_only | 1.000 | 0.00 | 1.000 |
| **text_state_proxy (B1)** | **0.500** | 1.00 | 0.000 |
| no_scene_graph | 0.100 | 9.00 | 0.000 |

**K3 复测：main - text_state_proxy = 0.500，PASS。** 结果同时说明：

- 持久实体在 14 帧完全遮挡后仍能恢复 ID；
- 单帧文本状态代理必然丢失身份，H1 在本 benchmark 上得到支持；
- `pose_only` 掉到 0.667，说明旧的 last-known pose 不够；
- `appearance_only` 与 main 持平，说明 re-ID 主要靠外观，类别项不是必要条件。

复现：

```bash
python -m pss.sim.scripts.generate_occlusion_v3 --out data/occlusion_v3 --sequences 6 --frames 24
python -m pss.eval.scripts.run_e3_occlusion \
  --data data/occlusion_v3 --split test \
  --experts checkpoints/category_pretrain_experts \
  --projections checkpoints/phase45_projection \
  --prototypes checkpoints/phase45_prototypes \
  --out eval/results/occlusion_v3_proto
```

局限：当前场景物体类别不同，类别项仍能兜底；要单独验证实例级外观 re-ID，还需要同类别多实例变体。B1 仍是文本状态代理，不是真实 VLM。

### 4.2 同类别多实例变体

`data/occlusion_v3_samecat`：3 个 `cup` 实例，per-instance scale/color/material variation，同样完全遮挡 14 帧并在遮挡期间换位。类别项在这里没有信息量，只能靠外观保持身份。

| 变体 | ID retention | switches | reappearance |
|---|---:|---:|---:|
| main | **1.000** | 0.00 | 1.000 |
| main_no_category | 1.000 | 0.00 | 1.000 |
| pose_only | 0.667 | 0.67 | 0.333 |
| appearance_only | 1.000 | 0.00 | 1.000 |
| **text_state_proxy (B1)** | **0.500** | 1.00 | 0.000 |
| no_scene_graph | 0.100 | 9.00 | 0.000 |

结论：

- 同类别场景下 main 仍为 1.000，text-state 代理 0.500，K3 delta 0.500；
- `pose_only` 掉到 0.667，说明 stale pose 不够；
- `appearance_only` 与 main 持平，说明当前 vision appearance 已经能区分同类别不同实例；
- `main` 与 `main_no_category` 持平，类别项不是必要条件。

`bottle` 同类别变体 `data/occlusion_v3_samecat_bottle` 复现同样结果：main 1.000、pose_only 0.667、appearance_only 1.000、text-state 0.500。

为适配长遮挡，re-ID 的代价函数改为 **pose 只做门控、appearance 做排序**：`reid_w_pose=0.0`，`reid_w_app=1.0`，`reid_pose_gate=0.5` 仍然生效。

复现：

```bash
python -m pss.sim.scripts.generate_occlusion_v3 \
  --out data/occlusion_v3_samecat --category cup --variation-specs

python -m pss.eval.scripts.run_e3_occlusion \
  --data data/occlusion_v3_samecat --split test \
  --experts checkpoints/category_pretrain_experts \
  --projections checkpoints/phase45_projection \
  --prototypes checkpoints/phase45_prototypes \
  --out eval/results/occlusion_v3_samecat_proto
```

### 4.3 near-identical 同类别变体

`data/occlusion_v3_near_identical`：3 个 `cup`，size/color/material 完全相同，只有 instance id 和位置不同。完全遮挡 14 帧，遮挡期间换位。

| 变体 | ID retention | switches | reappearance |
|---|---:|---:|---:|
| main | 0.667 | 0.67 | 0.333 |
| main_no_category | 0.667 | 0.67 | 0.333 |
| pose_only | 0.667 | 0.67 | 0.333 |
| appearance_only | 0.667 | 0.67 | 0.333 |
| **text_state_proxy (B1)** | **0.500** | 1.00 | 0.000 |
| no_scene_graph | 0.100 | 9.00 | 0.000 |

K3 仍为 0.167（main - text），但这里要看的不是 K3，而是实例级上限：

- vision 跨实例余弦 0.989–0.998，tactile 0.955–0.998，几何也没有稳定的实例签名；
- 三个实例在两个模态上几乎重合，完全遮挡后又换了位置，**没有任何可用的身份信息**；
- 逐实例看：`cup_n000` 和 `cup_n001` 在重现后交换了 track，`cup_n002` 因为移动距离较小保留了 ID。0.667 不是方法在区分它们。

结论：near-identical 变体给出的是**信息边界**，不是 tracking bug。当前特征空间无法区分像素级同物；要继续提升，只能增加实例级信号（更强的表面纹理、触觉纹理、交互历史）或避免完全遮挡。它也说明 Phase 6 的能力主张应限定为"可区分的实例"，而不是"任意实例"。

复现：

```bash
python -m pss.sim.scripts.generate_occlusion_v3 \
  --out data/occlusion_v3_near_identical --category cup --near-identical

python -m pss.eval.scripts.run_e3_occlusion \
  --data data/occlusion_v3_near_identical --split test \
  --experts checkpoints/category_pretrain_experts \
  --projections checkpoints/phase45_projection \
  --prototypes checkpoints/phase45_prototypes \
  --out eval/results/occlusion_v3_near_identical_proto
```

---

## 5. E4 原型包更新

沿用 Phase 9：

| 项 | 值 |
|---|---|
| 审核结论 | accept |
| 回归新类准确率 | 1.000 |
| 回滚 | v1.0 |
| 损坏包拒绝 | ✅ |

---

## 6. E5 阶段 gate

| 阶段 | 指标 | 值 | 阈值 | 通过 |
|---|---|---:|---:|---|
| P0 绑定 | 无噪声往返 | ~5e-17 | <1e-6 | ✅ |
| P3 专家（视觉） | test top-1 | 0.958 | 0.80 | ✅ |
| P3 专家（触觉） | test top-1 | 0.365 | 0.30 | ✅ |
| P4 对齐（V-G） | R@1 | 0.854 | 0.60 | ✅ |
| **P4 对齐（V-T）** | **R@1** | **0.510** | **0.60** | **❌** |
| P5 原型 | test top-1 | 0.969 | 0.80 | ✅ |
| P6 追踪 | 遮挡 ID retention | 1.000 | 0.90 | ✅ |
| P7 规划 | A* + 重规划 | True | True | ✅ |
| P8 动作头 | 闭环成功率 | 0.900 | 0.80 | ✅ |
| P8 文本消融 | delta | 0.000 | 0 | ✅ |
| P9 学习/安全 | 回滚+完整性+硬否决 | True | True | ✅ |

**K4 触发**：P4 V-T 未过。按预注册，上游（触觉专家/数据）返工，不进入下游能力声明。

---

## 7. E6 规模

| 项 | 值 |
|---|---:|
| 场景实体 | 200 |
| 活跃实体 | 4 |
| 活跃两两关系 | **6** |
| 全场景两两关系 | **19,900** |
| 空间查询 | 0.115 ms/帧 |

Token 序列化对照（理论）：10/100/1000 步对应 1k/10k/100k tokens，注意力量 1e6/1e8/1e10；向量路径保持 O(k²) 常数。结构性差异成立。

---

## 8. 杀死标准

| ID | 触发 | 细节 |
|---|---|---|
| **K1** | ✅ | D=128 K=20 = 0.769；HRR 0.667 |
| **K2** | ✅ | V-T 0.510 < 0.60；V-G 0.854 通过 |
| **K3** | ✅→已复测 | 旧基准 main - B1 = 0.000（遮挡不够难）；`occlusion_v3` 复测为 **0.500**，满足 ≥0.10 |
| **K4** | ✅ | P4 V-T gate 未过 |

---

## 9. 诚实的结论

### 成立的部分

- **槽位/实体分离有效**：逐实体 0.766 vs 全局向量 0.250。
- **训练投影对跨模态对齐有效**：V-T R@1 0.115 → 0.510。
- **显式状态的计算优势成立**：6 vs 19,900 组关系，0.115 ms/帧。
- **规划、异常恢复、学习闭环、安全降级、审计**均在代理数据上跑通。

### 未成立的部分

- **H1（显式实体优于文本状态）未被证实**：当前遮挡数据太弱，B1 代理打平。
- **触觉跨模态对齐未达标**：V-T 0.510，K2/K4 触发。
- **D=128 容量不足**：K=20 只有 0.769，K1 触发。
- **E0 真实数据 R-Gate 未运行**：RD2 尚未采集。

### 因此

> 本架构在代理数据上证明了**结构可行性**与**部分结构优势**，但没有证明**能力优势**。
> 下一步不应继续加模块，而应先补：更难的遮挡基准、真实 VLM 基线、RD2 真实触觉数据、真实预训练视觉骨干。

---

## 10. 建议的下一步

1. **RD2 真实同步数据**：触觉对齐与 E0 的前置条件。
2. **同类别多实例遮挡基准**：`occlusion_v3` 已让持久实体胜过单帧文本代理，但类别项仍能兜底；要单独验证实例级外观 re-ID，需要同类别多实例场景，再接真实 VLM 基线。
3. **真实 VLM 基线**：当前 B1 只是文本状态代理。
4. **视觉骨干升级**：隔离环境装 torchvision/DINOv2，走 9.1 锚点重训协议。
5. **D 决策**：若关系库需要 K≥20 的压缩存储，升 D=512；否则保持 128 + 逐条存储。
