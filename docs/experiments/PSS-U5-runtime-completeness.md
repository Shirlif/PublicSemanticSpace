# PSS U5：分解外运行时完整性

> 日期：2026-11-28
> 上游：`docs/plans/PSS-架构上界复核与下一阶段.md`、`docs/experiments/PSS-架构上界复核报告.md`
> 结果目录：`eval/results/upper_bound/u5/`
> 状态：三个子项均已实际执行；本报告所有数字来自本轮运行的 JSON / 终端输出。

---

## 0. 一句话结论

U5 三子项完成：query-conditioned fact selection 在合成长历史上把 k=4/8/16 的准确率从 hot 的 **0.237/0.421/0.473** 提升到 **1.000**，但 query_recency 同样为 **1.000**，增益应归因于“按查询选集合”而不是“学习模型”；Qwen2.5-1.5B 真实 LLM 达到 **0.857（6/7）、5 次工具调用、0 条引用、5792.7 ms/查询**，低于 ToolAgent 的 **1.000 / 8 / 引用覆盖率 1.000**；权限矩阵与实时审计通过（cumulative completeness **1.000**、permission violations **0**、rollback exact **True**），但 `rollback` 会把 audit 列表恢复到快照，回滚点之后的 `permission_denied` 从最终 audit 中消失，primary audit completeness 为 **0.800**，因此不能声称审计日志在 rollback 下是 append-only。

---

## 1. 三个子项的总表

| 子项 | 指标 | 结果 | 判定 |
|---|---|---|---|
| **1. 长历史 fact active-set** | k=4/8/16 accuracy（3 seeds） | full **1.000**；learned_query **1.000 / 1.000 / 1.000**；query_recency 同为 **1.000**；hot **0.237 / 0.421 / 0.473**；learned_old **0.246 / 0.402 / 0.605** | query conditioning 有效；学习模型无额外增益；旧负结果排序只在 k=8 出现（绝对值未精确复现） |
| **2. 真实 LLM 输出路径** | Qwen2.5-1.5B accuracy / tools / citations / latency | **0.857（6/7）/ 5 / 0 条（coverage 0.000）/ 5792.7 ms**；ToolAgent **1.000 / 8 / 2 条（coverage 1.000）/ 0.3 ms** | 可用，但未达 ToolAgent 参考；LLM 不产出引用 |
| **3. 审计完整性** | completeness / rollback exactness / violations | primary **0.800**（cumulative **1.000**）；control **1.000**；rollback exact **True（6/6 字段）**；violations **0** | 有条件通过：实时记录完整，rollback 会截断审计历史 |

权限矩阵作为子项 3 的一部分单独记录在 §4.1：`read_only / service / operator / admin` 的 denied audit entries 分别为 **5 / 4 / 1 / 0**，与文档定义一致。

---

## 2. fact active-set 明细

### 2.1 设置

- 流长度 `T=1000`，`N_KEYS=50`，`N_VALUES=16`，训练 seeds=3，测试查询采样 seed=99。
- 每个 seed 用独立训练流训练两个策略：query-independent `learned_old`，以及 query-conditioned `learned_query`。
- 测试流固定；`full` 使用全部 1000 条 fact，`recency` 取最近 k 条，`hot` 取 hot-key 的 top-k，`random` 为随机 k 条，`query_recency` 为每个查询取该 key 最近的 k 条。
- 每个 seed 在每种 k 下评估 500 个查询；表中为 3 个 seed 的 mean ± std。

### 2.2 主表（accuracy，越高越好）

| method | k=4 | k=8 | k=16 |
|---|---:|---:|---:|
| full | **1.000 ± 0.000** | **1.000 ± 0.000** | **1.000 ± 0.000** |
| recency | 0.074 ± 0.015 | 0.259 ± 0.026 | 0.313 ± 0.028 |
| hot | 0.237 ± 0.020 | 0.421 ± 0.024 | 0.473 ± 0.015 |
| random | 0.001 ± 0.001 | 0.004 ± 0.003 | 0.008 ± 0.005 |
| learned_old（query-independent） | 0.246 ± 0.016 | 0.402 ± 0.059 | 0.605 ± 0.167 |
| query_recency | **1.000 ± 0.000** | **1.000 ± 0.000** | **1.000 ± 0.000** |
| learned_query（query-conditioned） | **1.000 ± 0.000** | **1.000 ± 0.000** | **1.000 ± 0.000** |

### 2.3 Gap 表

`full − method` 为正表示该方法离 full 上界还有差距；`learned_query − hot` 为 query-conditioned 相对 hot 的增量。

| k | full − hot | full − learned_old | full − learned_query | learned_query − hot |
|---:|---:|---:|---:|---:|
| 4 | 0.763 | 0.754 | **0.000** | **+0.763** |
| 8 | 0.579 | 0.598 | **0.000** | **+0.579** |
| 16 | 0.527 | 0.395 | **0.000** | **+0.527** |

`query_recency − hot` 与 `learned_query − hot` 完全相同（0.763 / 0.579 / 0.527），因为两者在每个 k 上都达到 1.000。

### 2.4 旧负结果是否复现

上游引用的已知负结果为 **learned 0.372 < hot 0.480 < full 1.000**。本次 3-seed 重跑的结果是：

- **k=8 复现排序**：learned_old **0.402 ± 0.059** < hot **0.421 ± 0.024** < full **1.000**。这是三个 k 中唯一复现“learned < hot < full”的档位。
- **k=16 排序反转**：learned_old **0.605 ± 0.167** > hot **0.473 ± 0.015**。逐 seed 为 0.666 / 0.772 / 0.376，方差很大；seed 2 低于 hot，但均值高于 hot。
- **k=4 轻微反转**：learned_old **0.246 ± 0.016** > hot **0.237 ± 0.020**。
- **绝对值 0.372 未复现**。k=16 seed 0 的 hot 恰好为 **0.480**，与引用值一致，但同格 learned_old 为 **0.666**，不是 0.372。

因此，本次不能声称“0.372 < 0.480”被精确复现；能复现的是 k=8 上的定性排序。k=16 的 learned_old 对 seed 敏感，旧负结果不应被当作稳定点估计。

### 2.5 query-conditioned 是否超过 hot

是，且三个 k 全部通过预注册检查：

| k | learned_query > hot | learned_query > learned_old | query_recency > hot |
|---:|---|---|---|
| 4 | True | True | True |
| 8 | True | True | True |
| 16 | True | True | True |

但必须同时记录：**learned_query 与 query_recency 在每个 k 上完全相等（均为 1.000）**。这说明“查询条件化”本身已经足够闭合 active-set gap，学习模型没有带来额外增益。诚实结论是“缺的是 query conditioning”，不是“ learned model 更强”。

---

## 3. 真实 LLM 明细

### 3.1 运行环境

- 设备：CUDA，NVIDIA GeForce RTX 4070 Laptop GPU；未发生 OOM，未回退 CPU。
- 模型：`Qwen/Qwen2.5-1.5B-Instruct`，`local_files_only=True`，从本地 HF cache 加载；1.5B 指令模型，不是 frontier LLM。
- 解码：greedy（`do_sample=False`），`max_new_tokens=192`；工具循环上限 6 轮。
- 同一脚本化 episode、同样 7 个查询，与确定性 `ToolAgent` 对照。

### 3.2 主表

| agent | accuracy | 正确数 | tool calls | citations | citation coverage | unsupported entities | mean latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| `llm_structured_tools`（Qwen2.5-1.5B） | **0.857** | 6/7 | **5** | **0** | **0.000** | 0 | **5792.7 ms** |
| `tool_agent_reference` | **1.000** | 7/7 | **8** | **2** | **1.000** | 0 | **0.3 ms** |

说明：citation coverage 用 `verify_citations` 检查每条引用的 `(entity, attribute)` 是否存在于结构化 state；ToolAgent 的 2 条引用全部通过（2/2，coverage 1.000）。LLM 没有产出引用，helper 在 `n_citations=0` 时按 `max(1, 0)` 返回 **0.000**。LLM 的 5.8 s vs ToolAgent 的 0.3 ms 是单查询平均延迟，比例为约 1.9 万倍；这不是吞吐基准。

### 3.3 逐查询结果

| query | LLM ok | LLM mode | LLM tools | LLM latency | ToolAgent ok |
|---|---:|---|---:|---:|---:|
| 列出所有对象 | **no** | decoder | 0 | 9779.5 ms | yes |
| 有哪些关系？ | yes | decoder | 0 | 4482.6 ms | yes |
| bowl_02 的材质是什么？ | yes | decoder | 0 | 3297.1 ms | yes |
| 最近的审计记录 | yes | converged_on_repeated_call | 1 | 2904.6 ms | yes |
| 把 bowl_02 的材质改成 wood，并告诉我新版本号 | yes | converged_on_repeated_call | 2 | 7537.0 ms | yes |
| 回滚到 v1 并列出审计 | yes | converged_on_repeated_call | 2 | 6177.3 ms | yes |
| 哪些对象不确定？ | yes | decoder | 0 | 5511.5 ms | yes |

逐查询延迟来自同一脚本、同一 episode 的补充运行（共享 GPU 条件下），该次运行的 accuracy 仍为 **0.857（6/7）**、citations 仍为 **0**，平均延迟 **5669.9 ms**；主表中的 **5792.7 ms** 来自原始 `run_runtime_language` 输出。两次均值差异约 2%，在共享 GPU 的波动范围内。

失败查询是“列出所有对象”。该 episode 的结构化 state 有 4 个对象：`bowl_02`、`sponge_01`、`cup_02`、`plate_01`；LLM 列出了前三个，漏掉 `plate_01`，而检查函数要求所有对象 id 都出现，因此该查询判否。其余 6 个查询通过；`unsupported_entities=0`，说明答案中没有出现 state 里不存在的实体 id。

模式分布：4/7 直接由 decoder 从上下文回答（0 次工具调用）；3/7 触发了工具循环（审计 1 次、更新 2 次、回滚 2 次），均以 `converged_on_repeated_call` 结束，说明重复调用收敛保护生效，没有耗尽 6 轮预算。LLM 的引用数为 0，主要原因是它倾向从上下文直接回答，而不是调用 `get_facts` 等带 citations 的工具。

### 3.4 与 ToolAgent 1.000 / 8 / 1.000 对照

- accuracy：**0.857 vs 1.000**，差 0.143（1 个查询）。
- tool calls：**5 vs 8**，LLM 更少调用工具。
- citation coverage：**0.000 vs 1.000**，LLM 没有可验证引用。
- latency：**5792.7 ms vs 0.3 ms**，LLM 路径慢约 1.9 万倍。

结论：真实 LLM 路径可以跑通并保持实体不幻觉，但在脚本化查询上的准确率、引用覆盖和延迟均不如确定性 ToolAgent；不能把它当作 ToolAgent 的等价替代。

---

## 4. 审计完整性明细

### 4.1 权限矩阵（`run_runtime_audit.py`）

| profile | read | ingest | write_facts | plan | act | rollback | denied audit entries |
|---|---:|---:|---:|---:|---:|---:|---:|
| `read_only` | yes | no | no | no | no | no | **5** |
| `service` | yes | yes | no | no | no | no | **4** |
| `operator` | yes | yes | yes | yes | yes | no | **1** |
| `admin` | yes | yes | yes | yes | yes | yes | **0** |

矩阵与 `pss/runtime/permissions.py` 定义一致：`read_only` 只能读；`service` 增加 ingest；`operator` 增加写 fact、plan、act；`rollback` 只有 `admin` 可以执行。每个被拒绝的动作在矩阵运行中都产生了 `permission_denied` audit entry（denied 计数等于被拒动作数）。

### 4.2 U5 审计完整性场景

场景按顺序执行：**写 fact → 拒绝写 → rollback → set_goal + plan → act**。期望事件为 5 类：

| 期望事件 | 实际 audit action | 期望数 |
|---|---|---:|
| `fact_write` | `assert_fact` | 1 |
| `permission_denied` | `permission_denied` | 1 |
| `rollback` | `rollback` | 1 |
| `plan` | `plan` | 1 |
| `act` | `act` | 1 |

`set_goal` 是 `runtime.set_goal` 产生的辅助 audit entry，不属于五类期望事件，不计入分子和分母。

| arm | rollback target | audit completeness | cumulative completeness | missing events | rollback exactness | permission violations |
|---|---|---:|---:|---|---|---:|
| **primary** | v1（pre-denial） | **0.800（4/5）** | **1.000（5/5）** | `permission_denied` × 1 | **True（6/6 字段）** | **0** |
| **control** | v2（latest） | **1.000（5/5）** | **1.000（5/5）** | 无 | **True（6/6 字段）** | **0** |

Primary 的最终 audit 为：

- actions：`["assert_fact", "rollback", "set_goal", "plan", "act"]`
- versions：`[1, 1, 2, 3, 4]`
- 五类事件保留数：`fact_write=1, permission_denied=0, rollback=1, plan=1, act=1`

Control（rollback 到 latest）的最终 audit 为：

- actions：`["assert_fact", "permission_denied", "rollback", "set_goal", "plan", "act"]`
- versions：`[1, 2, 2, 3, 4, 5]`
- 五类事件保留数：`fact_write=1, permission_denied=1, rollback=1, plan=1, act=1`

`rollback_exactness` 在 two arms 中均为 True，逐字段检查 `entities / relations / facts / goals / plans / uncertainty` 全部相等。`permission_violations` 定义为“被拒绝时没有追加 `permission_denied` 条目的动作数”：本场景拒绝 1 次、当时审计 1 次，因此 violations = **0**。

### 4.3 关键发现：rollback 会截断 audit 历史

`CognitiveState.rollback(version)` 的实现是：

```python
payload = copy.deepcopy(self.snapshots[version])
self.audit = payload["audit"]
self.version = payload["version"]
self.audit.append(AuditEntry(self.version, "rollback", ...))
```

也就是说，rollback 不仅恢复 facts/entities/plans 等状态字段，也把 **audit 列表恢复成目标快照的 audit**。任何比 rollback target 更新的 audit entry 都会从最终 `state.audit` 中消失。

在 primary 中，`permission_denied` 发生在 v2，rollback 目标为 v1（denial 之前），因此：

- 拒绝动作在发生时确实被审计（`denials_audited_at_time=1`）；
- 但回滚后该条目不再保留（`denied_entries_retained_after_rollback=0`）；
- 最终 audit completeness 从累计的 1.000 降到 **0.800**。

Control 把 rollback target 设为 latest（v2），所以所有五类事件都保留，completeness 为 **1.000**。这说明缺失不是“没有埋点”，而是 **rollback 的目标选择会删除历史审计**。对部署而言，这等价于“拒绝动作可以被后续 rollback 从审计中抹掉”：审计日志不是 append-only；tamper-evidence 不在本场景的测试范围内。

> **2026-11-28 修复**：`CognitiveState.rollback` 已改为只恢复状态字段、不截断 audit；复测 primary completeness **1.000（5/5）**、`rollback_truncated_audit=false`、`permission_violations=0`。详见 `docs/experiments/PSS-U5b-audit-append-only.md`。本节保留修复前的原始观测，作为问题记录。

---

## 5. 可以 / 不可以声称

### 可以声称

1. 在 `T=1000, N_KEYS=50, N_VALUES=16` 的合成长历史上，query-conditioned selection（learned 或 recency）在 k=4/8/16 都把 accuracy 提升到 **1.000**，而 hot 只有 **0.237/0.421/0.473**，query-independent learned_old 为 **0.246/0.402/0.605**。
2. 增益来自 **query conditioning**，不是学习模型：`query_recency` 与 `learned_query` 在每个 k 上完全相等（均 1.000）。
3. `Qwen/Qwen2.5-1.5B-Instruct` 可以从本地 cache 在 CUDA 上加载并跑完 7 个脚本化查询：accuracy **0.857（6/7）**，tool calls **5**，unsupported entities **0**，平均延迟 **5792.7 ms**，无 OOM、无 CPU 回退。
4. `run_runtime_audit.py` 的四档权限矩阵与文档一致；矩阵运行中每个 denied action 都有 `permission_denied` audit entry。
5. 在 control arm（rollback 到 latest）中，五类期望事件全部保留（completeness **1.000**），rollback 后六个状态字段全部等于快照（exactness **True**），permission violations 为 **0**。

### 不可以声称

1. **不可以**声称 learned fact selector 优于 hot / recency heuristic。`learned_query` 与 `query_recency` 打平，无法把增益归因于学习。
2. **不可以**声称旧负结果 `learned 0.372 < hot 0.480 < full 1.000` 被精确复现。本次只在 k=8 复现排序；k=16 反转（learned_old 0.605 > hot 0.473）且 seed 方差大（std 0.167）。
3. **不可以**声称 LLM 达到 ToolAgent 水平：accuracy **0.857 vs 1.000**，citation coverage **0.000 vs 1.000**，延迟 **5.8 s vs 0.3 ms**。
4. **不可以**声称审计日志在 rollback 下是 append-only 或完整：primary completeness **0.800**，回滚点之后的 `permission_denied` 被丢弃。
5. **不可以**把本轮结论外推到 frontier LLM、真实长历史流、多 episode 生产负载或对抗性安全场景；本报告的 LLM 只有 1.5B，审计场景只有单 episode、单次拒绝、单次 rollback。

---

## 6. 产物与复现命令

### 6.1 产物

| 产物 | 路径 |
|---|---|
| U5 审计完整性脚本 | `pss/eval/scripts/run_upper_bound_u5.py` |
| fact active-set 指标 | `eval/results/upper_bound/u5/fact_active/metrics.json` |
| fact active-set 报告 | `eval/results/upper_bound/u5/fact_active/REPORT.md` |
| 真实 LLM 结果（JSON / Markdown） | `eval/results/upper_bound/u5/runtime_language.json`、`runtime_language.md` |
| 逐查询延迟与引用覆盖 | `eval/results/upper_bound/u5/runtime_language_per_query.json` |
| 权限矩阵结果（JSON / Markdown） | `eval/results/upper_bound/u5/runtime_audit.json`、`runtime_audit.md` |
| 审计完整性结果（JSON / Markdown） | `eval/results/upper_bound/u5/audit_completeness.json`、`audit_completeness.md` |
| 运行日志 | `eval/results/upper_bound/u5_fact_active.log`、`u5_runtime_language.log`、`u5_runtime_audit.log` |

`runtime_language.json` 与 `runtime_audit.json` 的原始脚本输出分别在 `eval/results/runtime_language/` 和 `eval/results/runtime_audit/`；U5 目录中的副本是本轮快照。

### 6.2 复现命令

```bash
# 1. 长历史 fact active-set（3 seeds, CPU）
python -m pss.eval.scripts.run_gap_fact_active \
  --seeds 3 --device cpu \
  --out eval/results/upper_bound/u5/fact_active

# 2. 真实 LLM 输出路径（CUDA；失败时改 --device cpu）
python -m pss.eval.scripts.run_runtime_language --device cuda

# 3. 权限矩阵
python -m pss.eval.scripts.run_runtime_audit

# 4. 审计完整性量化
python -m pss.eval.scripts.run_upper_bound_u5 \
  --out eval/results/upper_bound/u5/audit_completeness.json
```

运行环境：Python 3.x、PyTorch 2.10.0+cu126、NVIDIA RTX 4070 Laptop GPU；Qwen2.5-1.5B-Instruct 来自本地 HF cache。`run_runtime_language.py` 的原始输出目录固定为 `eval/results/runtime_language`，`run_runtime_audit.py` 固定为 `eval/results/runtime_audit`；如需 U5 目录中的副本，复制对应 JSON/Markdown 即可。

补充：逐查询延迟与 citation coverage 在 `run_runtime_language` 的同一 episode 上通过 `time.perf_counter` + `pss.heads.text.verify_citations` 采集，输出 `eval/results/upper_bound/u5/runtime_language_per_query.json`；主 accuracy / tools 仍以 `run_runtime_language` 的原始 JSON 为准。

---

## 7. 边界

1. **合成流，不是真实长历史。** fact active-set 使用 `T=1000`、50 keys、16 values 的合成 fact 流；查询采样有 70% 概率偏向 hot keys，答案定义为同 key 的最新高可靠 fact。真实部署的 key 分布、更新频率、查询分布和冲突语义都可能不同。
2. **单 episode，不是多 episode 部署。** 审计完整性只构造了一个脚本化 episode（1 次写、1 次拒绝、1 次 rollback、1 次 plan、1 次 act）；没有覆盖并发、多用户、多 rollback 嵌套或跨进程审计。
3. **1.5B 不是 frontier LLM。** LLM 结果来自 Qwen2.5-1.5B-Instruct、greedy decoding、7 个脚本化查询、单轮工具循环；不能外推到 7B/70B/frontier 模型、长上下文、多轮对话或开放查询分布。
4. **fact active-set 的绝对数字有 run-to-run 波动。** 脚本固定 numpy seed，但策略网络的 torch 初始化未显式固定；本轮 learned_old 为 0.246/0.402/0.605，早先报告为 0.241/0.379/0.604，差异在此波动范围内。旧负结果的精确 0.372 未复现。
5. **citation coverage 的定义依赖脚本。** 用 `verify_citations` 检查引用的 `(entity, attribute)` 是否存在；LLM 0 条引用时 helper 返回 0.000，这不等于“引用了错误事实”，而是“没有可验证引用”。
6. **审计条目是内存态。** 没有做持久化、签名、哈希链或 tamper-evidence 测试；“rollback 截断 audit”是当前实现的结构性行为，生产部署需要单独修复和复测。
7. **权限矩阵是静态 profile 测试，不是对抗性安全审计。** 它验证四档 profile 的允许/拒绝与 denied audit，不覆盖权限提升、并发竞争、工具参数注入或越权数据访问。
8. **延迟不是吞吐基准，且测量时 GPU 被共享。** 5792.7 ms 是 laptop GPU 上 7 个查询的平均单查询延迟；测量期间同一 GPU 上还有 U1b/U3 长任务在运行，因此这是共享 GPU 条件下的值，不是独占延迟基准。ToolAgent 的 0.3 ms 是确定性内存工具调用，两者不是同一类计算负载。
