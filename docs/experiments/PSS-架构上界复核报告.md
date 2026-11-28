# PSS 架构上界复核报告：U1–U4

> 日期：2026-11-28
> 上游：`docs/plans/PSS-架构上界复核与下一阶段.md`、`PSS-读出与推理优化总报告.md` §4.4、`PSS-g因子优化报告.md`
> 结果目录：`eval/results/upper_bound/{u1,u2,u3,u4}/`
> 状态：已执行 U1–U4；U5（分解外完整性）未执行。
> 第二阶段（U1b / U3b–U3d / U5）见 `docs/experiments/PSS-架构上界复核-第二阶段.md`。

---

## 0. 一句话

**“上界”不是单一数字，而是表示的函数。** U1 证明 A4 的闭集增益可以迁移到真实骨干（+12.5 pt），但开集质量不能（AUROC 0.9866 / reject 0.918 未达标）；U2 证明覆盖规则可以迁移到真实渲染几何（单轴 0.419 vs 组合 1.000）；U3 给出本轮最重要的负结果——**DINOv2 公共空间的关系组合保真度只有 0.804，符号坐标是 0.996，`L_gap = +0.193`，absence error 0.827**，说明“无损度”不是定义空洞，而是关系任务上的真实瓶颈；U4 证明 1.3 pt 地板在删除/合并下可维护（forgetting 0.013、rollback exact）。因此 B1 局部上界成立，B2 部分成立（闭集/覆盖），**B3 联合上界不成立**。

---

## 1. 判定口径

| 级别 | 定义 | 本轮状态 |
|---|---|---|
| **B1 局部上界** | 固定表示、固定任务，换更强的同族读出/容量没有增量 | **成立，但只在各自表示上**：合成 0.948、DINOv2 0.867；rerank 相对闸门 Δ=0 |
| **B2 迁移上界** | 同一排序与增益在真实骨干/真实几何上保持 | **部分成立**：U1 闭集通过、开集失败；U2 全通过 |
| **B3 联合上界** | f 与 g 在同一基质上测量，`A_current ≈ A_full` 且 `L_gap ≤ 0.05` | **不成立**：U3 `L_gap = +0.193`，absence error 0.827 |

---

## 2. 总表

| 阶段 | 问题 | 主结果 | 预注册 | 判定 |
|---|---|---|---:|---|
| **U1** | A4 读出增益能否迁移到真实视觉骨干？ | proto 0.742 → lda_whiten **0.867（+12.5 pt）**；rerank 相对闸门 **Δ=0**；AUROC 0.9866、reject@95 0.918 | 4 arms，2 失败 | 闭集迁移 **成立**；开集迁移 **不成立** |
| **U2** | “补缺失 pair type”能否迁移到真实几何？ | baseline 0.000、single **0.419**、combination **1.000**、no-op 0.000、full 1.000 | 5 arms，全通过 | **成立**（限 CLEVR 渲染几何） |
| **U3** | 学习公共空间是否保留关系组合所需信息？ | symbolic **0.996** vs learned **0.804**；`L_gap +0.193`；absence error **0.827** | 2 arms，2 失败 | **不成立**：无损度是真实瓶颈 |
| **U4** | 1.3 pt 地板在删除/合并下是否可维护？ | `incr_refit_10` forgetting **0.013**、added **0.948**、rollback **True** | 1 arm，通过 | **成立** |

---

## 3. U1：真实骨干（DINOv2）上的 A 线读出

### 3.1 主表

| arm | closed-set acc | Δ vs proto | AUROC | unseen reject @95 |
|---|---:|---:|---:|---:|
| `u_real_proto` | 0.742 | — | — | — |
| `u_real_lda_whiten` | **0.867** | **+0.125** | 0.9866 | 0.918 |
| `u_real_rerank`（Δ vs 同一闸门） | 0.867 | **+0.000** | — | — |
| `u_real_mlp` | 0.729 | — | — | — |

同一读出、仅换骨干的对照：

| backbone | proto | lda_whiten | Δ vs proto |
|---|---:|---:|---:|
| 合成 CNN | 0.833 | 0.948 | +0.115 |
| DINOv2-small | 0.742 | 0.867 | +0.125 |

### 3.2 判读

1. **闭集增益迁移成立**：`lda_concat_whiten − proto = +0.125`，超过预注册的 +0.08；A4 的“先混合再判别”不是合成代理的特例。
2. **重排仍无增量**：rerank 相对同一闸门 Δ=0.000，A2 的“线性度量已是该表示的读出上限”在 DINOv2 上仍成立。
3. **开集不迁移**：AUROC 0.9866（bar ≥0.99）、reject@95 0.918（bar ≥0.95）失败。A4 的开集数字 0.998/0.987 是合成表示特有的，不能随闭集结论一起外推。
4. **换编码器不是免费提升**：DINOv2 的绝对闭集 0.867 低于合成 0.948。这直接支持“上界是表示的函数”——更强的编码器改变了表示，也移动了读出上界，方向不一定是向上。
5. 边界：只换 vision；tactile/geometry 仍是合成专家；投影头按同一配方重训，单投影 seed（0）；CLIP 权重未缓存，不能声称 VLM。

---

## 4. U2：真实几何（CLEVR）上的覆盖审计

### 4.1 主表

2000 train scenes / 500 test scenes；测试每个 pair type 5000 行；3 seeds × 300 epochs。

| arm | train rows | horizontal | vertical | diagonal | worst cell | overall | Δ vs full |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 5000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.333 | −1.000 |
| single | 6000 | 1.000 | 1.000 | 0.419 | 0.419 | 0.806 | −0.581 |
| **combination** | 6000 | 1.000 | 1.000 | 1.000 | **1.000** | 1.000 | **0.000** |
| no-op | 6000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.333 | −1.000 |
| full | 6000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |

### 4.2 判读

1. **单轴不够**：`single` worst-cell 0.419，仍落后 `combination` 0.581，远超预注册的 −0.10；G2 的“补缺失 pair type，不是只补缺失轴”在真实 CLEVR 布局上复现。
2. **组合补到满**：`combination` 与 `full` 都是 1.000，且所有 arm 的 row budget 对齐（5000 base + 1000 augment），所以 `Δ vs full` 归因于覆盖，不是样本量。
3. **no-op 无效**：多给已覆盖的 horizontal 数据，worst-cell 仍是 0.000，与 baseline 相同。
4. **边界**：CLEVR 是渲染数据，不是机器人；horizontal/vertical 由 snapping 一条轴构造，另一轴保留真实布局；结论写“真实几何/渲染”，不写 sim2real。

---

## 5. U3：公共空间的组合保真度（本轮关键负结果）

### 5.1 主表

同一批 CLEVR 对象对；同一 head 深度/宽度；同一训练预算；符号头输入 4-d 相对坐标，学习头输入 4×384-d DINOv2（`[z_a, z_b, z_b−z_a, |z_b−z_a|]`）。

| representation | horizontal | vertical | diagonal | overall | worst cell | absence error |
|---|---:|---:|---:|---:|---:|---:|
| symbolic (4-d) | 0.967 | 0.957 | 0.998 | **0.996** | 0.933 | 0.043 |
| learned (DINOv2) | 0.234 | 0.173 | 0.836 | **0.804** | 0.173 | **0.827** |

- `L_gap = symbolic − learned = **+0.193**`（bar ≤0.05）
- `absence_cell_error (learned) = **0.827**`（bar ≤0.20）
- params：symbolic 4,740；learned 102,788

### 5.2 判读

1. **方向明确**：学习表示在 diagonal 上还有 0.836，但在两个 absence cell 上崩到 0.234 / 0.173。也就是说，DINOv2 CLS 能区分“明显分开”的对象，却不能可靠判断“几乎对齐”的对象——正好是关系谓词 dead zone 所需的信息。
2. **不是容量问题**：学习头的参数量是符号头的 21.7 倍，仍然更差；差距来自表示里的信息，不是拟合能力。
3. **不是任务太难**：符号头在同一批 pair 上是 0.996，说明标签和划分没有不可约噪声；`L_gap` 可归因于表示。
4. **边界的正确写法**：这只证明 **DINOv2 CLS 公共空间** 不是组合无损的；没有证明所有学习表示都不行。patch tokens、位置感知 embedding、或显式拼接几何通道可能缩小 `L_gap`，这是下一步。
5. **架构含义**：PSS 的“显式状态 + typed 关系”不是便利层，而是必要层——如果只用学习 embedding 做关系，absence case 会系统性失效。这与 Phase 12 的“精确几何做骨架、学习关系做校准”一致。

---

## 6. U4：长期增量维护（删除 / 合并）

### 6.1 主表

384 类；base 192 + add 192；之后删除 10%、合并 5%；评估 = val+test。

| policy | base before | after add | forgetting | added acc | after delete | after merge | rollback | refits |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| `incr_frozen` | 0.974 | 0.943 | +0.031 | 0.927 | 0.949 | 0.904 | True | 1 |
| **`incr_refit_10`** | 0.974 | 0.961 | **+0.013** | **0.948** | 0.963 | 0.937 | True | 20 |
| `incr_refit_all` | 0.974 | 0.961 | +0.013 | 0.948 | 0.963 | 0.946 | True | 193 |

协方差漂移（`incr_refit_10`）：add 后 0.319；delete+merge 后 0.456；条件数 1663.7 → 848.0。

### 6.2 判读

1. **主 bar 通过**：`incr_refit_10` forgetting 0.013 ≤ 0.03、added 0.948 ≥ 0.93；A5 的 1.3 pt 地板在删除/合并下仍成立。
2. **冻结精度不够**：`incr_frozen` 合并后 base 0.904，比 refit_10 低 3.3 pt；删除/合并改变了类间散度，必须刷新。
3. **刷新频率**：refit_10 与 refit_all 的 add 后精度相同（0.961/0.948），refit_all 只在 merge 后多 +0.9 pt，却多 173 次 refit；refit_10 仍是合理默认。
4. **可回滚**：三个策略 `rollback_exact=True`，删除/合并没有破坏快照语义。

---

## 7. 交叉结论：B1–B3 分级

### 7.1 B1 局部上界：成立，但是“表示相对”的

- 解码器：合成 0.948、DINOv2 0.867；重排相对闸门 Δ=0；完整 MLP 更差（0.729）。在每个表示上，线性判别读出已到局部上限。
- 维度：128→512 无增量（旧证据）；本轮没有新的维度实验。
- 容量：rerank 147k 参数无增量（旧证据）；U3 学习头 21.7× 参数仍更差，进一步支持“容量不是瓶颈”。

### 7.2 B2 迁移上界：闭集/覆盖成立，开集不成立

- **U1 闭集**：+12.5 pt 迁移成立。
- **U1 开集**：AUROC/reject 失败；A4 的开集质量是表示特有的，不能外推。
- **U2 覆盖**：单轴 0.419 vs 组合 1.000，在 CLEVR 渲染几何上成立。

### 7.3 B3 联合上界：不成立

- `L_gap = +0.193 ≥ 0.10`，absence error 0.827；学习公共空间在关系组合上不是无损的。
- 因此不能把 `PSS 上界 ≈ f × g` 写成实测陈述；目前只能写“B1 局部上界 + B2 部分迁移”。

---

## 8. 失败项（照实保留，不调 bar）

| arm | 项 | bar | 实测 |
|---|---|---|---:|
| `u_real_lda_whiten` | AUROC | ≥ 0.990 | 0.9866 |
| `u_real_lda_whiten` | unseen reject @95 | ≥ 0.950 | 0.918 |
| `u_comp_learned` | composition gap | ≤ 0.050 | 0.193 |
| `u_comp_learned` | absence cell error | ≤ 0.200 | 0.827 |

口径说明：

- U1 的失败是**开集**，不是闭集；闭集 bar 和 rerank 负对照都通过。
- U3 的失败是**表示**，不是容量或标签：符号头在同一 pair 上 0.996。
- U2 和 U4 没有失败项。

---

## 9. 不能声称

- 不能声称 A4 的 0.948/0.998 可以整体外推到真实骨干；闭集可以，开集不行。
- 不能声称“换更强编码器会提升读出”；DINOv2 的绝对闭集低于合成 CNN。
- 不能声称所有学习表示都不支持关系组合；只测了 DINOv2 CLS，patch tokens/位置感知表示未测。
- 不能声称 CLEVR 结论等于 sim2real；它是渲染几何，不是机器人。
- 不能声称 B3 联合上界已经达到；U3 明确不成立。
- 不能把 U2 的 1.000 外推到任意几何；它仍是受控的 pair-type 覆盖实验。

---

## 10. 对架构的含义

1. **绑定约束从“g 覆盖”转到了“f 无损度”**。U2 把 g_cov 在渲染几何上推到 1.000；U3 随即显示：即使覆盖满了，学习表示本身也丢掉了 absence case 所需的信息。继续补数据不会修 `L_gap`。
2. **显式几何/状态是必要的，不是便利层**。PSS 的“精确几何做骨架、学习关系做校准”在本轮得到直接支持：纯 embedding 关系头在 near-aligned cell 上崩，符号坐标不崩。
3. **开集是真实骨干上的独立问题**。U1 的闭集增益与开集退化同时出现，说明“换表示”不是单调改进；部署需要按表示重新校准 open-set。
4. **下一步优先级**：
   - U3b：位置感知表示（patch tokens、坐标嵌入、或 embedding+geometry 拼接）能否把 `L_gap` 压到 ≤0.05；
   - U1b：CLIP/VLM 特征 + 3 个投影 seed + 开集重校准；
   - U5：分解外的 runtime 完整性（长历史 fact active-set、audit completeness、真实 LLM），不改变 f×g 但决定可部署性。

---

## 11. 产物与复现

```
pss/eval/upper_bound.py                          # 纯函数：pair type / exact-4 / worst cell
pss/eval/scripts/run_upper_bound_u1.py           # 真实骨干 A 线
pss/eval/scripts/run_upper_bound_u2.py           # 真实几何覆盖
pss/eval/scripts/run_upper_bound_u3.py           # 组合保真度
pss/eval/scripts/run_upper_bound_u4.py           # 删除/合并增量
pss/prototypes/incremental.py                    # 新增 remove_class / merge_classes / snapshot
pss/eval/readout_reasoning_preregistration.py    # 新增 u_* 字段与 arms
tests/test_upper_bound.py
eval/results/upper_bound/u1/                     # fusion_ablation.json / rerank_eval.json / U1_REPORT.md
eval/results/upper_bound/u2/                     # u2.json / U2_REPORT.md
eval/results/upper_bound/u3/                     # u3.json / U3_REPORT.md
eval/results/upper_bound/u4/                     # u4.json / U4_REPORT.md
```

复现命令：

```bash
python -u -m pss.eval.scripts.run_upper_bound_u1 \
  --out eval/results/upper_bound/u1 --device cuda \
  --projection-epochs 120 --seeds 0 1 2

python -u -m pss.eval.scripts.run_upper_bound_u2 \
  --out eval/results/upper_bound/u2 --device cuda \
  --train-scenes 2000 --test-scenes 500 --per-type 5000 \
  --baseline-rows 5000 --augment 1000 --full-rows 6000 \
  --epochs 300 --seeds 0 1 2

python -u -m pss.eval.scripts.run_upper_bound_u3 \
  --out eval/results/upper_bound/u3 --device cuda \
  --train-pairs 30000 --test-pairs 10000 --epochs 300 --seeds 0 1 2

python -u -m pss.eval.scripts.run_upper_bound_u4 \
  --out eval/results/upper_bound/u4 --device cuda --n-added 192
```

---

## 12. 一句话

**B1 局部上界成立，B2 只在闭集和覆盖上成立，B3 联合上界不成立：学习公共空间在关系组合上不是无损的（`L_gap +0.193`）。下一阶段的架构优化不是继续加读出/容量/维度，而是让表示保留关系所需的信息——位置感知表示或显式几何通道。**
