# PSS 架构上界复核（第二阶段）：U1b / U3b–U3d / U5

> 日期：2026-11-28
> 上游：`docs/plans/PSS-架构上界复核与下一阶段.md`、`docs/experiments/PSS-架构上界复核报告.md`
> 结果目录：`eval/results/upper_bound/{u1b_clip,u1b_dinov2,u3b,u3c,u3d,u5}/`
> 状态：U1b、U3b、U3c、U3d、U5 均已执行；U5 由独立子任务运行并单独成文。
> 第三阶段（审计修复 / U3e / U1c / U5b / U6）见 `docs/experiments/PSS-架构上界复核-第三阶段.md`。

---

## 0. 一句话

**第二阶段把三个悬而未决的问题都闭合了。** U1b：A4 的闭集增益在 CLIP 和 DINOv2 上都迁移（+11.0 / +13.1 pt），CLIP 单头即通过开集，DINOv2 需要双头（LDA 闭集 + 原型开集）才能恢复开集；U3b–U3d：纯 embedding、naive concat、联合双分支都修不好 absence cell，只有**显式几何骨架 + α=0 的 embedding 残差**同时通过 `L_gap` 和 absence（`L_gap −0.002`、absence 0.045），三个 seed 全部把 α 选成 0；U5：query conditioning 把 fact active-set 从 hot 的 0.237/0.421/0.473 抬到 1.000，但 query_recency 与 learned_query 打平，Qwen2.5-1.5B 为 0.857/0 引用，rollback 会截断 audit 历史（primary completeness 0.800）。

---

## 1. 总表

| 阶段 | 问题 | 主结果 | 预注册 | 判定 |
|---|---|---|---:|---|
| **U1b CLIP** | 真实骨干 + 3 投影 seed + 开集重校准 | proto 0.812 / LDA **0.922** / dual 0.922；LDA AUROC **0.997**、reject **0.995** | 3 arms 全通过 | **成立**（单头即可） |
| **U1b DINOv2** | 同上 | proto 0.731 / LDA **0.862** / dual 0.862；LDA AUROC 0.987、reject 0.916（失败）；dual **0.991 / 0.962** | LDA 开集 2 项失败；dual 通过 | **闭集成立；开集需双头** |
| **U3b naive concat** | 坐标拼进 embedding 能否修复无损度？ | 0.892；`L_gap +0.104`；absence 0.530 | 失败 | **拼接不够** |
| **U3c joint branch** | 双分支联合训练呢？ | 0.950；`L_gap +0.046`；absence 0.411 | L_gap 通过，absence 失败 | **联合训练仍被 embedding 干扰** |
| **U3d 残差骨架** | 显式骨架 + α·embedding？ | 0.999；`L_gap −0.002`；absence 0.045；**α=0 三个 seed** | 通过 | **成立：几何骨架必须主导** |
| **U5** | fact / LLM / 审计 | query-conditioning 1.000；LLM 0.857/5/0 引用；audit primary 0.800 | 见 U5 报告 | **部分成立，审计有语义缺陷** |

---

## 2. U1b：真实骨干 + 3 投影种子 + 开集重校准

### 2.1 CLIP ViT-B/16

| arm | closed-set acc | AUROC | unseen reject @95 |
|---|---:|---:|---:|
| `proto` | 0.812 ± 0.010 | 0.996 ± 0.001 | 0.992 ± 0.004 |
| `lda`（shrinkage 0.001） | **0.922 ± 0.004** | **0.997 ± 0.000** | **0.995 ± 0.004** |
| `dual_head`（LDA + proto 开集） | 0.922 ± 0.004 | 0.996 ± 0.001 | 0.992 ± 0.004 |

- `lda − proto = +0.110`，AUROC/reject 全部通过预注册 bar。
- 三个 seed 的 shrinkage 都选到 **0.001**（默认值），开集重校准没有额外作用——CLIP 的 LDA 开集本来就高。
- 双头与单头在 CLIP 上几乎相同；没有必须切换的理由。

### 2.2 DINOv2-small

| arm | closed-set acc | AUROC | unseen reject @95 |
|---|---:|---:|---:|
| `proto` | 0.731 ± 0.016 | 0.991 ± 0.000 | 0.962 ± 0.002 |
| `lda`（shrinkage 0.001） | **0.862 ± 0.004** | 0.987 ± 0.001 | 0.916 ± 0.010 |
| `dual_head` | **0.862 ± 0.004** | **0.991 ± 0.000** | **0.962 ± 0.002** |

- `lda − proto = +0.131`，闭集增益稳定；但 LDA 的开集 AUROC 0.987、reject 0.916 仍失败。
- **双头把开集恢复到原型头的 0.991/0.962，全部通过**；闭集保持 LDA 的 0.862。
- shrinkage 仍是 0.001；这说明 DINOv2 的问题不是校准阈值，而是 LDA 的判别方向在开集上不如原型 cosine。

### 2.3 判读

1. **闭集读出增益在两个真实骨干上都成立**：CLIP +0.110、DINOv2 +0.131。A4 的“先混合再判别”不是合成代理的特例。
2. **开集是 backbone-specific**：CLIP 单头全过；DINOv2 单头退化，双头修复。不能把“A4 开集 0.998/0.987”外推到所有真实骨干。
3. **shrinkage 不是修复手段**：三个 seed 都选最小 shrinkage，重校准没有改变排序。真正的修复是**分维度读出**——闭集与开集用不同头。
4. **绝对上界随表示移动**：合成 0.948 > CLIP 0.922 > DINOv2 0.862；更强的编码器不等于更高的闭集，但 CLIP 的开集明显更好。

---

## 3. U3b–U3d：组合保真度的表示消融

### 3.1 主表

同一批 CLEVR 对象对、同一 head 深度/宽度、同一训练预算（30k train / 10k test pairs，3 seeds × 300 epochs）。

| representation | horizontal | vertical | diagonal | overall | L_gap | absence error |
|---|---:|---:|---:|---:|---:|---:|
| symbolic (4-d) | 0.967 | 0.957 | 0.998 | **0.996** | — | **0.043** |
| learned (DINOv2 CLS) | 0.234 | 0.173 | 0.836 | 0.804 | +0.193 | 0.827 |
| naive concat (+ coords) | 0.566 | 0.470 | 0.912 | 0.892 | +0.104 | 0.530 |
| joint two-branch | 0.678 | 0.589 | 0.967 | 0.950 | +0.046 | 0.411 |
| **residual (α=0)** | **1.000** | **0.955** | **0.999** | **0.999** | **−0.002** | **0.045** |

### 3.2 判读

1. **纯 embedding 不是组合无损的**：`L_gap +0.193`、absence 0.827；diagonal 还有 0.836，但两个 near-aligned cell 崩到 0.234/0.173。DINOv2 CLS 能区分“明显分开”，不能可靠判断“几乎对齐”。
2. **拼接不够**：把 4 维坐标拼到 1,536 维输入，`L_gap` 只从 0.193 降到 0.104，absence 仍 0.530。4 维被高维 embedding 淹没。
3. **联合双分支仍不够**：`L_gap` 降到 0.046（通过），但 absence 0.411（失败）；1,536 维分支在联合训练中仍能压过 4 维分支。
4. **显式骨架 + α=0 才干净**：两个分支分开训练，`coord + α·emb`，α 在训练 hold-out 上选；**三个 seed 全部选 α=0**，test overall 0.999、absence 0.045、`L_gap −0.002`。
5. **架构含义**：几何谓词必须由**显式几何通道主导**；embedding 在这个任务上没有可加的校准价值（α=0），更不应该让它与坐标联合训练去“学” absence。PSS 的“精确几何做骨架、学习关系做校准”在本轮得到直接支持，且校准权重应该是**有界、可为零**的。

### 3.3 边界

- 只测了 DINOv2 CLS；patch tokens、位置感知 embedding、CLIP embedding 未测。
- 任务只有几何关系；语义/触觉关系可能从 embedding 获益，不能外推。
- CLEVR 是渲染数据，不是 sim2real。
- U3d 的 α=0 是校准结果，不是“embedding 永远无用”；它说明在这个几何任务上，embedding 的边际信息为 0。

---

## 4. U5：分解外运行时完整性（摘要）

完整报告：`docs/experiments/PSS-U5-runtime-completeness.md`。

### 4.1 fact active-set

| method | k=4 | k=8 | k=16 |
|---|---:|---:|---:|
| full | 1.000 | 1.000 | 1.000 |
| hot | 0.237 | 0.421 | 0.473 |
| learned_old（query-independent） | 0.246 | 0.402 | 0.605 |
| query_recency | **1.000** | **1.000** | **1.000** |
| learned_query | **1.000** | **1.000** | **1.000** |

- query conditioning 把 gap 闭合到 0；但 `query_recency` 与 `learned_query` 完全打平，增益应归因于**按查询选集合**，不是学习模型。
- 旧负结果 `learned 0.372 < hot 0.480 < full 1.000` 只在 k=8 复现排序；k=16 反转（learned_old 0.605 > hot 0.473），绝对值 0.372 未复现。

### 4.2 真实 LLM（Qwen2.5-1.5B-Instruct）

| agent | accuracy | tool calls | citations | citation coverage | mean latency |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-1.5B | 0.857（6/7） | 5 | 0 | 0.000 | 5792.7 ms |
| ToolAgent | 1.000（7/7） | 8 | 2 | 1.000 | 0.3 ms |

- 失败查询是“列出所有对象”：LLM 漏掉 `plate_01`；无 unsupported 实体 id。
- LLM 倾向直接回答，不调用带 citations 的工具，因此引用覆盖为 0。
- 不能把 1.5B 当作 ToolAgent 的等价替代；它能跑通、能调工具、能保持实体不幻觉，但准确率/引用/延迟都落后。

### 4.3 审计完整性

| arm | rollback target | audit completeness | cumulative | rollback exact | violations |
|---|---|---:|---:|---|---:|
| primary | v1（pre-denial） | **0.800（4/5）** | 1.000 | True（6/6） | 0 |
| control | v2（latest） | 1.000（5/5） | 1.000 | True（6/6） | 0 |

- 四档权限矩阵正确：read_only/service/operator/admin 的 denied audit entries 为 5/4/1/0。
- **关键缺陷**：`CognitiveState.rollback` 把 `audit` 列表恢复到目标快照；回滚点之后的 `permission_denied` 从最终 audit 中消失。审计日志不是 append-only，拒绝动作可以被后续 rollback 从审计中抹掉。
- 这是部署前必须修的语义问题，不是埋点缺失。

---

## 5. B1–B3 分级更新

| 级别 | 第一阶段 | 第二阶段 | 结论 |
|---|---|---|---|
| **B1 局部上界** | 合成 0.948；DINOv2 0.867 | CLIP 0.922；DINOv2 0.862；rerank 无增量 | **成立，但表示相对** |
| **B2 迁移上界** | 闭集成立、开集失败 | CLIP 全过；DINOv2 双头全过；U2 覆盖成立 | **成立（按骨干选择单头/双头）** |
| **B3 联合上界** | 纯 embedding 不成立 | 纯 embedding / 拼接 / 联合分支都不成立；**显式骨架 + α=0 成立** | **仅显式几何形态成立；纯学习空间不成立** |

一句话：**B3 不是“学习表示达到上界”，而是“显式几何骨架 + 可为零的学习校准”达到上界。** 这正好是 PSS 架构的核心主张，而不是它的软肋。

---

## 6. 失败项（照实保留，不调 bar）

| arm | 项 | bar | 实测 |
|---|---|---|---:|
| `u_real_lda_whiten`（DINOv2） | AUROC | ≥ 0.990 | 0.9865 |
| `u_real_lda_whiten`（DINOv2） | unseen reject @95 | ≥ 0.950 | 0.9158 |
| `u_comp_learned` | composition gap | ≤ 0.050 | 0.1926 |
| `u_comp_learned` | absence cell error | ≤ 0.200 | 0.8273 |
| `u_comp_hybrid` | composition gap | ≤ 0.050 | 0.1040 |
| `u_comp_hybrid` | absence cell error | ≤ 0.200 | 0.5302 |
| `u_comp_branch` | absence cell error | ≤ 0.200 | 0.4111 |

通过项：U1b CLIP 全部、U1b DINOv2 dual-head、U2 全部、U3d 全部、U4 全部。

---

## 7. 可以 / 不可以声称

**可以声称**

1. A4 的闭集读出增益在 CLIP（+11.0 pt）和 DINOv2（+13.1 pt）上都成立，且 3 个投影 seed 稳定。
2. CLIP 上单头 LDA 即可同时满足闭集与开集；DINOv2 上需要双头（LDA 闭集 + 原型开集）才能恢复开集。
3. 纯 DINOv2 CLS 不是组合无损的（`L_gap +0.193`、absence 0.827）；显式几何骨架 + α=0 的 embedding 残差可以达到 `L_gap −0.002`、absence 0.045。
4. 三个 seed 都把 α 选成 0：在这个几何关系任务上，embedding 的边际信息为 0，显式几何必须主导。
5. query conditioning 把合成长历史的 fact active-set 从 hot 的 0.237/0.421/0.473 抬到 1.000；Qwen2.5-1.5B 可跑通 7 查询（0.857、5 tools、0 citations、5.79 s）。
6. 权限矩阵与 rollback exactness 通过；审计埋点在实时路径完整（cumulative 1.000）。

**不可以声称**

1. 不可以声称 A4 的开集质量可以整体外推到所有真实骨干；DINOv2 单头失败，必须双头。
2. 不可以声称“换更强编码器提升闭集”；CLIP 0.922 < 合成 0.948，DINOv2 0.862 更低。
3. 不可以声称纯学习表示达到 B3；只有显式几何形态达到。
4. 不可以声称所有学习表示都不行；只测了 DINOv2 CLS，patch tokens、位置感知表示、CLIP embedding 未测。
5. 不可以声称 learned fact selector 优于 query_recency；两者打平，增益来自 query conditioning。
6. 不可以声称 LLM 达到 ToolAgent 水平（0.857 vs 1.000、0 引用、5.8 s vs 0.3 ms）。
7. 不可以声称审计日志在 rollback 下是 append-only；primary completeness 0.800。
8. 不可以声称 CLEVR 结论等于 sim2real；它是渲染数据。

---

## 8. 架构结论与下一步

### 8.1 三个结论

1. **上界是表示的函数，且不是单调的。** 合成 0.948、CLIP 0.922、DINOv2 0.862；开集 CLIP 0.997、DINOv2 0.987。换骨干会移动上界，方向不一定是向上。
2. **关系组合的约束是“显式几何是否在场”，不是容量/维度/搜索。** 纯 embedding、拼接、联合分支都修不好 absence；只有显式骨架 + α=0 同时通过。
3. **闭集与开集要分开看。** DINOv2 上单头 LDA 的闭集增益与开集退化同时存在；双头是部署形态，不是可选项。

### 8.2 下一步

1. **修审计语义**：`CognitiveState.rollback` 不应截断 audit；应维护独立的 append-only 审计日志，或在快照中保留 audit 的单调前缀。
2. **U3e（可选）**：测 patch tokens / 位置感知 embedding 是否能在不依赖显式坐标的情况下降低 `L_gap`；如果仍然 α=0，说明几何谓词应永远走显式通道。
3. **U1c**：把 CLIP/DINOv2 的 U1b 结论搬到真实 VLM 特征（如果可用），并保留双头作为默认部署形态。
4. **U5 后续**：query-conditioned fact selection 已在合成流闭合；下一步是真实长历史流 + coverage loss，确认 query_recency 的 1.000 不是合成流的 artifact。
5. **sim2real**：CLEVR/合成几何不能替代真实机器人；显式几何骨架在真实深度/位姿噪声下的表现仍未测。

---

## 9. 产物与复现

```
pss/eval/scripts/run_upper_bound_u1b.py       # CLIP/DINOv2 + 3 seeds + 双头
# U3b 复用 run_upper_bound_u3.py（symbolic / learned / hybrid 三条表示）
pss/eval/scripts/run_upper_bound_u3c.py       # 联合双分支
pss/eval/scripts/run_upper_bound_u3d.py       # 几何骨架 + α·embedding 残差
pss/eval/scripts/run_upper_bound_u5.py        # U5 审计完整性（子任务新建）
docs/experiments/PSS-U5-runtime-completeness.md
eval/results/upper_bound/u1b_clip/
eval/results/upper_bound/u1b_dinov2/
eval/results/upper_bound/u3b/
eval/results/upper_bound/u3c/
eval/results/upper_bound/u3d/
eval/results/upper_bound/u5/
```

复现命令：

```bash
python -u -m pss.eval.scripts.run_upper_bound_u1b \
  --backbone clip --out eval/results/upper_bound/u1b_clip \
  --projection-seeds 0 1 2 --device cuda

python -u -m pss.eval.scripts.run_upper_bound_u1b \
  --backbone dinov2 --out eval/results/upper_bound/u1b_dinov2 \
  --projection-seeds 0 1 2 --device cuda

python -u -m pss.eval.scripts.run_upper_bound_u3 \
  --out eval/results/upper_bound/u3b --device cuda \
  --train-pairs 30000 --test-pairs 10000 --epochs 300 --seeds 0 1 2

python -u -m pss.eval.scripts.run_upper_bound_u3c \
  --out eval/results/upper_bound/u3c --device cuda \
  --train-pairs 30000 --test-pairs 10000 --epochs 300 --seeds 0 1 2

python -u -m pss.eval.scripts.run_upper_bound_u3d \
  --out eval/results/upper_bound/u3d --device cuda \
  --train-pairs 30000 --test-pairs 10000 --epochs 300 --seeds 0 1 2

python -u -m pss.eval.scripts.run_upper_bound_u5 \
  --out eval/results/upper_bound/u5
```

---

## 10. 一句话

**B1 成立但表示相对；B2 按骨干选择单头/双头后成立；B3 只在“显式几何骨架 + 可为零的学习校准”形态下成立——纯 embedding 在关系组合上不是无损的。** 这不是 PSS 的失败，而是它“显式状态 + typed 关系”主张的直接证据。
