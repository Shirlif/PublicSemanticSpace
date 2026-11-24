# PSS 五优先级最终结论

> 日期：2026-11-24
> 范围：`PSS-认知层架构定位.md` §7 的五个优化优先级
> 依据：`PHASE11`–`PHASE15` 报告、`eval/results/{encoder_priority,typed_relations,active_policy,language_head,bilevel}`
> 数据：全部为合成数据（`category_pretrain`、`tracking_bench_v2`、`phase7_v2`）

---

## 1. 一句话结论

> **五个优先级证明了 PSS 的价值在“显式、可审计的状态 + typed 关系 + active-set 记忆 + 工具接口”，而不在“换更强的编码器”或“用一个联合模型替代所有模块”。**
>
> 感知层不是 PSS 的杠杆；关系层、记忆选择和输出审计才是。联合训练能跑通，但还没有超过成熟的分模块 pipeline。

---

## 2. 总表

| 优先级 | 目标 | 最好结果 | 判定 |
|---|---|---|---|
| 第一：换编码器 | 冻结 CLIP/DINOv2、触觉更强编码器、投影对齐/校准 | DINOv2 经校准 re-ID 0.963→0.993；base 0.997；residual-only 开放集 0.998 | 部分达标 |
| 第二：关系层可学习 | typed heads、relation logits 进 scene graph、不手写边 | 关系 F1 0.957/0.996/0.970；hybrid 缺 tactile 0.791→0.819；P5b 分布内 1.000/1.000 | 部分达标 |
| 第三：active-set / write policy | 学“看哪些”、学“写什么”、不确定回退 | k=1+fallback 0.972→1.000（10% fallback）；write 0.84→0.88 @匹配写入，review 0.92 | 部分达标 |
| 第四：输出头 | 语言接 LLM decoder，PSS 给状态/工具/审计 | 7 查询：flat 0.714 / tools 1.000；citation 1.000；虚构实体可检测 | 达标（有边界） |
| 第五：端到端/bilevel | 冻结编码器，联合投影+绑定+关系；任务+一致性约束 | joint re-ID 0.889 vs sequential 0.867；joint_instance 0.922 | 部分达标 |

---

## 3. 逐项结论

### 3.1 第一优先：换编码器

**结论：换编码器本身不是免费提升；投影头才是冻干特征能不能用的关键。**

| 变体 | 闭集 | 开放集 | re-ID | near |
|---|---:|---:|---:|---:|
| base（SmallCNN+MLP） | 0.979 | 0.993 | **0.997** | 0.987 |
| residual-only | 0.984 | **0.998** | 0.989 | 0.956 |
| whiten-only | 0.990 | 0.969 | 0.948 | 0.791 |
| calibrated（whiten+res） | **0.995** | 0.977 | 0.989 | 0.956 |
| dinov2_plain | 0.974 | 0.976 | 0.963 | 0.880 |
| dinov2（校准） | 0.969 | 0.897 | 0.993 | 0.982 |
| dinov2_vlad | 0.964 | 0.841 | **1.000** | **1.000** |
| conv_tactile | **0.995** | 0.973 | 0.991 | **1.000** |

- 冻结 DINOv2/CLIP 直接替换 **没有** 提升闭集/开放集；DINOv2 接近但低于 base。
- **投影头是此前 DINOv2 re-ID 失败的主因**：同一 DINOv2，plain head re-ID 0.963，校准 head 0.993；但校准会伤开放集（0.976→0.897）。
- **没有单一投影头同时最优**：base 头最好 re-ID，residual-only 最好开放集，whiten 最好闭集。识别和 re-ID 应分开用头。
- **P5b 对编码器构造性不变**：它用 GT 类别、`w_category=0`、只用位置；编码器不进入 P5b 状态，不能把 P5b 当编码器证据。
- CLIP 已实现并接入 registry，但本地无权重，本轮跳过。

### 3.2 第二优先：关系层可学习

**结论：typed 关系可学，但“纯学习边替代手写边”不成立；正确形态是 hand + learned 的混合，以及“精确几何谓词 + 学习关系”的并存。**

- 关系 F1（clean）：spatial 0.957、semantic 0.996、tactile 0.970；缺 tactile 时 tactile 掉到 0.483（说明没有走捷径）。
- re-ID：hand 1.000 / 缺 tactile 0.791；pure learned 0.985 / 0.696；**hybrid 0.996 / 0.819**。纯学习边不如手写边，hybrid 在缺 tactile 时更好。
- relation logits 已写入 `SceneGraph`，3 物体测试帧 18 条边，save/load 完整保留。
- P5b：手写 `_rel` 1.000/1.000；学习关系分布内 **1.000/1.000**；held-out 区域 0.842/**0.688**；transformer 1.000/0.462。
- **学习关系比端到端策略泛化好，但不如精确几何**；held-out 区域布局仍需要几何先验。
- temporal 未训练；task 通过 P5b 评估；对齐区域需要 dead zone（`|delta|<0.1` 四向都不触发）。

### 3.3 第三优先：active-set / write policy

**结论：observation-conditioned 的 active-set 和 write policy 可用；稀疏、无 query 条件的 fact 覆盖仍未解决。**

- re-ID 候选 active-set：k=1 clean 0.972 → **1.000**（10% fallback，平均 1.2 个候选）；pose noise 0.961→0.978。
- **listwise 是正确目标**：同特征 pairwise BCE full-score 0.756，listwise 0.972。
- **fallback 不能补信息缺失**：缺 tactile 时 full 和 active 都 0.689，说明是信息没了，不是没选中。
- write policy：write-all 0.70；规则 0.88/716；learned 0.88/1083；learned+review **0.92/1051**；匹配写入量下 learned 0.88 vs rule 0.84。
- 长历史 fact active-set：full 1.000、hot 0.480、learned 0.372、diversity+fallback 0.680。**负结果**：当前 fact 打分没有 query 条件、缺少覆盖率训练，不能超过 hot heuristic。

### 3.4 第四优先：输出头

**结论：语言必须是指向结构化状态/工具/审计的适配器，不是张量→文本的 MLP。**

| agent | accuracy | tool calls | citation coverage |
|---|---:|---:|---:|
| no context | 0.000 | 0 | 0.000 |
| flat JSON dump，无工具 | 0.714 | 0 | 0.000 |
| structured state + tools | **1.000** | 8 | **1.000** |

- 7 条查询：flat 只能答静态 5 条；**update material / rollback + audit 只有工具路径能完成**。
- 写工具默认拒绝，显式授权后才写，返回 version；写操作进入 audit。
- 虚构实体 `obj_9999` 被 `verify_answer_entities` 抓到；citation 可验证。
- 上下文：小 episode 564 vs 511 tokens（无可压缩）；60 物体/200 关系 **3439 vs 12666 tokens（0.27x）**。
- **边界**：本地无 LLM，`ToolAgent`/脚本化 decoder 是 harness 替身；本轮证明的是 state/tool/audit 表面，不是语言质量。动作头未改。

### 3.5 第五优先：端到端 / bilevel

**结论：联合训练能跑通并略优于分期训练，但还没有超过成熟 pipeline；一致性约束有效但有精度代价；联合关系头远弱于专门关系头。**

| mode | rec | re-ID | near | spatial F1 | semantic F1 | tactile F1 |
|---|---:|---:|---:|---:|---:|---:|
| joint | 0.979 | 0.889 | 0.600 | 0.371 | 0.779 | 0.779 |
| joint_instance | 0.896 | **0.922** | 0.733 | 0.439 | 0.789 | 0.782 |
| sequential | 0.984 | 0.867 | 0.467 | 0.401 | 0.667 | 0.782 |
| no_consistency | 0.969 | 0.889 | 0.600 | **0.663** | **0.856** | 0.779 |
| linear_head | **0.995** | 0.900 | 0.733 | 0.285 | 0.695 | 0.782 |
| concat | 0.979 | 0.861 | 0.578 | 0.329 | 0.922 | 0.740 |

- joint re-ID 0.889 vs sequential 0.867、concat 0.861：**联合训练 +2 点左右**，rec 基本持平。
- joint_instance re-ID 最高 **0.922**，但 rec 掉到 0.896；linear_head rec 最高 **0.995**。
- 一致性约束：antisymmetry 0.054→0.005、exclusivity 0.277→0.000，但 spatial F1 0.663→0.371。**audit-vs-accuracy trade-off**。
- **联合关系 F1 远低于单独关系头**（0.957/0.996/0.970）：联合 binder 被类别/re-ID 目标主导，不能替代专门关系 token + 监督。
- **绝对 re-ID 仍低于成熟 pipeline**（0.989–0.997）：joint 用简单 pairwise MLP，不是 graph matcher 边消息；差距是架构 + 多任务组成。
- 方法学修正：识别视图必须屏蔽 category one-hot（否则 1.000 是泄漏）；置换一致性最初目标写反，报过虚假 3.3 MSE，修正后为 0。

---

## 4. 跨优先级的三条主结论

### 4.1 感知不是 PSS 的杠杆，结构与选择才是

- 换编码器、加 DINOv2/VLAD、加 conv tactile，都没有稳定提升闭集/开放集。
- 真正有效的是：投影校准（第一）、typed 关系（第二）、active-set/write（第三）、结构化输出（第四）。
- 这与架构文档的判断一致：PSS 应站在感知之上做认知，不应和 transformer 抢感知。

### 4.2 “学习”必须和“精确先验”并存，不能简单替代

- 学习边不如手写边（第二）；学习关系在 held-out 区域不如精确几何（第二）；联合关系头不如专门关系头（第五）。
- 但学习层在缺失模态、噪声、分布内替代、审计一致性上提供了手写规则没有的鲁棒性和可校准性。
- 正确形态：**精确几何/规则做骨架，学习 typed 关系做校准和鲁棒性通道**；不是二选一。

### 4.3 PSS 真正可辩护的价值是“可审计的状态 + 工具 + 选择”，不是“更强的模型”

- 语言头：flat context 能答静态问题，但 update/rollback/audit 只有工具路径能完成。
- active-set：k=1+fallback 用 1.2 个候选追平 full；fallback 的边界是信息缺失。
- write policy：review band 用弃权换准确率；长历史 fact 覆盖是明确的未解问题。
- 联合训练：能跑通，但没有改变“分模块 + 成熟 matcher 更强”的事实。

---

## 5. 明确不能声称的

- ❌ “换更强编码器直接提升识别/开放集” —— 不成立。
- ❌ “学习边/学习关系已替代手写边/精确几何” —— 不成立。
- ❌ “P5b 全变体不变说明关系层优于感知” —— 是构造性不变，不是新证据。
- ❌ “联合训练已达到或超过成熟 pipeline” —— re-ID 0.889–0.922 vs 0.989–0.997。
- ❌ “长历史 fact active-set 已解决” —— learned 0.372 < hot 0.480 < full 1.000。
- ❌ “语言头具备语言能力” —— 本轮是 harness，不是 LLM 生成质量。
- ❌ “合成数据结论可外推到真实机器人” —— 全部为合成数据。

---

## 6. 仍未完成 / 下一步

1. **真实数据 / sim2real / R-Gate**：所有结论仍是合成数据。
2. **CLIP**：代码已接入，权重未缓存；需要时再跑。
3. **temporal / task 关系头**：schema 和写回路径已有，未训练。
4. **query-conditioned fact active-set + coverage loss**：第三优先的明确负结果。
5. **联合训练用同一 matcher 架构**：把 graph matcher 的边消息放进 joint，才能公平比较 joint vs separate。
6. **多任务权重 / 5 seeds / 统计检验**：当前多为 1–3 seeds。
7. **真实 LLM 接入**：`LLMLanguageHead` 接口已实现，需接本地或 API decoder。

---

## 7. 复现

```bash
make encoder-priority      # 第一优先
make typed-relations       # 第二优先
make active-policy         # 第三优先
make language-head         # 第四优先
make bilevel               # 第五优先
```

报告：`PHASE11_ENCODER_PRIORITY_REPORT.md`、`PHASE12_TYPED_RELATIONS_REPORT.md`、`PHASE13_ACTIVE_WRITE_POLICY_REPORT.md`、`PHASE14_LANGUAGE_HEAD_REPORT.md`、`PHASE15_BILEVEL_REPORT.md`。

---

## 8. 后续：Phase 16 认知运行时

五个优先级之后，下一阶段不再加新模型，而是按架构分层把模块收敛成一个 **PSS 认知运行时**：

- `CognitiveState`：entities / relations / facts / goals / plans / uncertainty / audit，带版本、快照、回滚；
- `EntityLayer`：persistent ID、object slots、属性、置信度、provenance、生命周期；
- `RelationFactLayer`：symbolic + learned relations、多源 facts、冲突、规则派生；
- `UncertaintyService`：对象置信度、fact/relation 冲突、缺失模态、accept/review/abstain；
- `RuntimePlanner`：goal + A* + plan validation；
- `OutputRouter`：语言工具/citations、safety veto；
- `PSSRuntime`：ingest / assert_fact / set_goal / plan / act / explain / rollback。

闭环任务（遮挡目标 + 冲突事实 + 坏写入回滚 + 语言解释）结果：

| variant | entity | plan | task | conflict | rollback | citations |
|---|---:|---:|---:|---:|---:|---:|
| full | yes | yes | yes | yes | yes | 1 |
| no_entity | no | no | no | no | yes | 0 |
| no_uncertainty | yes | yes | yes | no | yes | 1 |
| no_audit | yes | yes | yes | yes | no | 1 |
| no_language | yes | yes | yes | yes | yes | 0 |

逐层消融正好打在各自层上；`no_relations` 仍能完成位置目标（facts 足够），说明当前任务的边界，而不是关系层无效。

详见 `PHASE16_RUNTIME_REPORT.md`，复现：`python -m pss.eval.scripts.run_runtime`。

### 8.1 Phase 17：学习模块进入运行时

Phase 17 把 Phase 11–15 的学习模块接进 runtime 默认路径：

- learned spatial relation 驱动 `RuntimePlanner.plan_relational`（4 个谓词符号/学习计划一致，区域头 0.988）；
- learned `WritePolicy` 在 `assert_fact` 边界做 write/review/skip（准确率 0.75→0.85，77 条被拒并审计）；
- `VersionedOperatorLibrary`：移除直连算子后计划从 1 步变 2 步，rollback 恢复；plan 记录 operator version；
- 70 帧遮挡（tracking_bench_long，12 条）：ID retention 1.000、entity persistence 1.000；
- 真实 decoder 接口已接，但本地无文本 LLM，明确记录 `decoder_available=false`。

详见 `PHASE17_RUNTIME_INTEGRATION_REPORT.md`，复现：`python -m pss.eval.scripts.run_runtime_phase17 --sequences 12`。

### 8.2 Phase 18：学习模块设为 runtime 默认

Phase 18 把 learned relations、active-set、write policy 设为默认路径，手写规则保留为可切换 fallback：

- learned relations：27 vs 12 total relations，任务契约不变；
- write policy：低可靠 noisy fact 被拒（1 条），关掉后写入；
- active policy：排除高置信但无观测支持的 distractor；confidence top-k fallback 会选中它；
- `components()` 审计当前路径，三个开关可单独关闭。

详见 `PHASE18_LEARNED_DEFAULTS_REPORT.md`，复现：`python -m pss.eval.scripts.run_runtime_defaults`。

### 8.3 Gap closure 1–2：编码器 / 投影默认

按架构表逐条收口的前两项：

- **编码器**：runtime 默认切到 frozen DINOv2（SmallCNN 为 fallback）；短序列 retention 两者 1.000，DINOv2 observation p50 16.7 ms vs SmallCNN 4.7 ms。精度 trade-off 仍是 Phase 11（DINOv2 re-ID 0.993 vs 0.997，闭集/开放集更低）。CLIP/VLM 无本地权重。
- **投影**：DINOv2 投影默认 `whiten_dim=96`（vision+geometry）+ residual；`PerceptionService.describe()` 可审计 calibration；modality reliability 进入 `UncertaintyService`，tactile 低可靠触发 review。
- **状态：部分**——DINOv2 默认路径已校准；但识别/re-ID 共用头、开放集未超 bank、SmallCNN fallback 仍是 plain、CLIP/VLM 投影未验证。

详见 `PHASE19_GAP_CLOSURE_ENCODER_PROJECTION.md`，复现：`python -m pss.eval.scripts.run_runtime_encoder --sequences 5`。

### 8.4 Gap closure 3：object attention 绑定选项

- `ObjectAttentionReID` 接入 `EntityLayer(associator=...)` / `PSSRuntime(associator=...)`，接口与 GraphMatcherReID 对齐；
- pair re-ID 与 tracker retention 在该 benchmark 上 graph/object attention 都 1.000；P1-15 五 seed 是 0.987 vs 0.989（打平）；
- 关闭的是“object attention 可作 runtime 绑定选项”，不是“object attention 更强”；runtime 默认仍是 hand，opt-in。

详见 `PHASE20_GAP_CLOSURE_BINDING.md`，复现：`python -m pss.eval.scripts.run_runtime_binding --sequences 8`。

### 8.5 Gap closure 4：可学习、可版本化关系规则

- `RelationRuleSet` 学习 `near_dist/contact_pad/bin_half/bin_z`，支持 update/rollback；
- `RelationFactLayer.add_geometry` 消费 learned rule set，`components.relation_rules` 可审计；
- contact F1 0→0.267，in 保持 1.000；near 在该 split 无 GT（n/a）；
- 边界：contact GT 来自 collision geometry，规则用 bounding sphere，特征限制上限；typed heads 仍是更强的学习关系通道。

详见 `PHASE21_GAP_CLOSURE_RULES.md`，复现：`python -m pss.eval.scripts.run_runtime_rules`。

### 8.6 Gap closure 5：learned heuristic + A* 搜索

- A* 保留，heuristic 从 `len(goal-state)` 换成可学习 cost-to-go MLP；
- expansions 6.29→3.28（weighted 2.48），cost/success 不变；
- 小域 wall-clock 0.13→3.3 ms（MLP 评估主导），不能声称整体加速；
- runtime 可选用 learned heuristic，`planning.heuristic=learned` 可审计。

详见 `PHASE22_GAP_CLOSURE_PLANNING.md`，复现：`python -m pss.eval.scripts.run_runtime_planning --train 4000 --test 400`。

### 8.7 Gap closure 6：真实本地 decoder

- 下载并接入 Qwen2.5-1.5B-Instruct；`load_local_decoder` 优先使用它；
- 7 查询：LLM 0.857（5 tool calls, citations 0, 4.0s/query）vs ToolAgent 1.000（8 tools, 2 citations）；
- 能正确调用 `update_fact` / `rollback_fact`；对象枚举漏 1 个；citation 覆盖不足；
- 增加 tool-loop convergence guard：重复调用时收敛到工具结果，不再 budget 耗尽；
- 边界：单 episode、7 查询、1.5B 模型，不是 frontier LLM 结果。

详见 `PHASE23_GAP_CLOSURE_LANGUAGE.md`，复现：`python -m pss.eval.scripts.run_runtime_language --device cuda`。

### 8.8 Gap closure 7：runtime 权限

- `read_only` / `service` / `operator` / `admin` 四档 profile；
- `ingest / assert_fact / set_goal / plan / act / rollback / explain / summary` 全部强制检查；
- 拒绝动作先写 `permission_denied` audit，再抛异常；
- operator 可写/规划/执行，rollback 仅 admin；默认 admin 保持兼容。

详见 `PHASE24_GAP_CLOSURE_AUDIT.md`，复现：`python -m pss.eval.scripts.run_runtime_audit`。

### 8.9 Phase A 评测套件骨架

- 新增 `pss/eval/suite/`：`EvalEpisode` schema、`BenchmarkAdapter`、metrics、runner、`bfcl/openood/zsre` 三个 adapter；
- 不合并原始数据：`data/public/<bench>/` gitignored，adapter 只做格式转换；
- smoke 验证了 schema→adapter→predictor→metrics→report 的端到端链路；
- 未接官方数据；smoke predictor 是确定性 baseline，不是 PSS 能力结论。

详见 `PHASE25_EVAL_SUITE.md`，复现：`python -m pss.eval.suite.runner --smoke`。

### 8.10 官方数据接入

- TAO 官方 annotations 已下载并接入，50 val videos 的 oracle-box association baseline：IDF1 0.478 / retention 0.482 / switches 7.94；
- CLEVR-CoGenT 官方数据已下载/解压，valA/valB 各 30,000 episodes：relation accuracy **0.994 / 0.994**（GT-box + pixel-depth baseline）；
- 两者当前 predictor 分别是 oracle boxes / pixel coordinates，不是 PSS 感知结果；relation-only 也不检验 CoGenT 属性组合 shift；下一步换图像编码器 + relation head，并加属性组合 task。

详见 `PHASE26_OFFICIAL_DATA.md`。

### 8.11 CLEVR 图像关系 + 属性组合

- image crops → 冻结 DINOv2 → relation head + position：valA/valB **0.994/0.996**（geometry rule 0.986/0.991）；
- CoGenT 属性组合：joint 0.810→**0.268**；纯 DINOv2 factorized 0.747→0.316；
- **color 修复**：颜色直方图 color valB 0.336→**0.841**，factorized exact valB 0.316→**0.795**；shape/material/size 仍用 DINOv2；
- 仍是 GT boxes（detection-oracle），relation head 是 MLP，不是 PSS typed head。

详见 `PHASE27_CLEVR_IMAGE_RELATION.md`，复现：`python -m pss.eval.scripts.run_clevr_image --train 8000 --val 3000`。

### 8.12 TAO 官方帧 + PSS tracker

- Charades 官方帧（5 视频/176 帧）+ GT boxes → PSS tracker；
- PSS tracker IDF1 **0.979** / retention 0.973 / switches 0.44 vs IoU **0.653** / 0.611 / 5.12；
- 关键修复：TAO boxes 1280×720 → 480×270 缩放；`reid_pose_gate` 需从 0.5（米）改成 ~300（像素）；
- GT boxes 是 oracle detections，不是检测器；小样本，配置在该子集上调过。

详见 `PHASE28_TAO_PSS.md`，复现：`python -m pss.eval.scripts.prepare_tao_frames --videos 5` + `python -m pss.eval.scripts.run_tao_pss --videos 5`。

### 8.13 把 GT boxes 换成检测器

- CLEVR：trainA 训练 CenterNet-lite（中心 heatmap + depth head），测试只用检测中心；
  valA/valB detection recall **0.997/0.996**，center error **4.96/5.10 px**，
  relation **0.995/0.994**，property exact **0.792/0.790**；
- 与 GT-box baseline 相比，关系基本持平（0.994/0.996），属性只掉 0.5–0.9 点
  （0.801/0.795）；检测器不再是主要瓶颈；
- TAO：COCO person detector + PSS，person-only 子集 IDF1 **0.984** / retention
  1.000 / switches 0.00，IoU 0.653，GT-box oracle 1.000；detection recall 0.970 /
  precision 0.809；
- **边界**：CLEVR 检测器 in-domain；TAO 只测 person、zero-shot COCO、5 个视频；
  未跑官方 HOTA，CLEVR relation head 仍是 MLP。

详见 `PHASE29_DETECTOR_REPLACEMENT.md`，复现：`python -m pss.eval.scripts.train_clevr_detector --device cuda --train 8000 --val 3000 --epochs 25` + `python -m pss.eval.scripts.run_tao_detector --device cuda --threshold 0.3`。
