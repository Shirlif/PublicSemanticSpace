# PSS P0–P2 结论文档与论文修正建议

> **日期**：2026-11-26
> **范围**：F1–F10 全部执行完毕后，这批工作能支撑什么结论、必须收回什么、论文需要改哪些句子。
> **依据**：`paper/main.tex`、`reports/claim_gaps/{F1..F10}/`、`pss_eval/`、`reports/claim_gaps/INDEX.md`。
> **一句话结论**：这批工作的产出不是能力扩展，而是 **4 条干净的正面结果 + 6 条收缩或负结果**；其中三条先前报告的结果分别被证明是调参 artifact、实现成本和未达成的上界。论文的贡献因此从"我们做到了"变成"我们的主张可被证伪，且经得起证伪"。

---

## 0. 结论摘要

1. **F1–F10 让论文的四条 Limitations 有两条可以销账、两条必须收窄。** 感知级 swap（F2）把 identity 路径从 oracle pose 中解放出来；异构宽度（F4）证明宽度不是瓶颈；TMS 迁移（F7）闭合了"未全量迁移"；尾延迟（F3）从"有个 bug"升级为"可达边界的量化"。
2. **F8 是整批唯一改变已发表结论的发现。** E3 的 `hybrid_diag` 优势（0.857 vs 0.828）在调好固定死区后消失（0.972 vs 0.859）。论文 §Results 与 §Discussion 里"distribution-aware hybrid 在高噪声下最好"这句必须撤回。
3. **F7 要求重新解释 E8 的 `1000 facts 52.9 ms`。** 那不是规模墙，是 full 快照模式的实现成本；同一 10k 负载在 delta 模式下是 0.014 ms。
4. **F3 要求把"10 ms 上界"改成"API 边界的有界工作 + anytime 降级"。** 硬墙钟上界在非 RT OS 上未达成，且可以证明为什么未达成。
5. **F2 的 naive 对照给出了一条方法论结论**：能力指标检测不到契约违规——4/4 bundle 上回滚栈已损坏，但 `INT_post` 只在 2/4 上下降。这是"为什么需要 contract-level protocol"的最直接实验证据。
6. **仍然不能说的**：真机、sim2real、闭环具身、竞争性能力 benchmark。F5 的外部基线只有一个，且按构造无法被 contract 属性评分。

---

## 1. 批次构成

| 类别 | 任务 | 性质 | 最强数字 |
|---|---|---|---|
| 干净正面 | F1 复现包 | 把"不可复现"变成"一条命令" | 干净 clone 复现 valid 19/33：1.000 / 0.827 / 0.270 |
| 干净正面 | F2 感知级 swap | 把 Limitations 变成 Results | 正例 SSR/RBE/INT_post/RCR 全 1.000；naive RCR 0/4 |
| 干净正面 | F5 协议包 | 协议独立可插拔 | pss-runtime 全 pass；e5-flat contract/swap = 0.000 |
| 干净正面 | F7 TMS 迁移 | runtime 默认 + 代价解耦 | 10k facts update 0.014 ms vs 580.764 ms |
| 收缩 | F3 硬上界 | 有界性 ≠ 硬实时 | BAR 0.944→0.999，但 guarded max 仍 64.91 ms |
| 收缩 | F4 触觉宽度 | 宽度不是瓶颈，校准才是 | k=0 全线 0.000–0.200；k=1 DINOv2 0.983 / CLIP 1.000 |
| 收缩 | F6 细微投毒 | 投毒检测有盲区 | N=8 恢复 ≥97%；σ=0.05 与 5% flip 漏检 |
| 负结果 | F8 学习门控 | 不迁移，安全收益来自死区 | 0.969 vs 固定死区 0.972；fallback 率 0.943 |
| 收缩 | F9 多 query | 合成流，coverage ≠ 正确性 | conflict Q=4,k=4：1.000 vs 0.167 |
| 负结果 | F10 真实回放 | 强主张 NO-GO | trained R@1 0.22 / R@5 0.73；无 L2/sim2real |

**4:6 的比例本身是可信度背书。** 一个只会产出正结果的评测套件没有说服力；能推翻自己的两组数字才有。

---

## 2. 四条论文原来没有的新知识

### 2.1 准入契约与专家宽度无关，只与校准数据有关

DINOv2 384 / CLIP 768 / WT-MM 64 跨越 12 倍宽度，数据驱动的准入阈值分别是 0.329 / 0.360 / 0.010（差 36 倍），但 reject / accept / review 三种裁决完全一致。同时 k=0 全线 0.000–0.200，k=1 跳到 0.983 / 1.000。

**结论**：公共空间是**可校准接口**，不是**零样本通用空间**；工程宽度从来不是瓶颈，校准样本才是。这第一次给论文里 hedged 的 "calibratable-interface view" 提供了因果证据——跃迁发生在 k=1，与宽度无关。

### 2.2 学习式门控在分布外不迁移，安全收益来自更宽的几何死区

噪声 0.15、A/B/C → D/G：

| 方法 | 值 |
|---|---:|
| geometry(0.25)（E3 strict） | 0.829 |
| hybrid_diag（E3） | 0.859 |
| 无 fallback 的 binary gate | 0.614 |
| 无 fallback 的 multi gate | 0.644 |
| binary gate + OOD fallback | 0.830 |
| **multi gate + OOD fallback** | **0.969** |
| **验证集选出的固定死区** | **0.972** |

fallback 在 D/G 上触发率 0.943——学习头在 A/B/C 拟合良好，到 D/G 基本全程弃权。

**结论**：唯一安全的收益是更宽 / 自适应的几何死区；学习路径在调好的死区面前没有增量。E3 的 `hybrid_diag` 优势是死区未调好的 artifact。

### 2.3 状态代价可以从更新/传播延迟里完全解耦，但回滚仍是 O(N)

10k facts：

| 指标 | full | delta | 倍率 |
|---|---:|---:|---:|
| update 中位数 | 580.764 ms | 0.014 ms | ~41,000× |
| 增量传播 vs 全量 | 78.547 ms | 0.013 ms | 5,906× |
| 每 update 常驻内存 | 19,581.9 KB | 5.0 KB | ~3,900× |
| build | 263.87 ms | 36.56 ms | 7.2× |
| rollback | 161.82 ms | 6.226 ms | 26× |
| 回滚后首次查询 relabel | 10.751 ms | 9.944 ms | ~1× |

**结论**：truth maintenance 与快照成本可以解耦（update + propagate 不再随事实数增长）；但 rollback 与回滚后首次查询仍是 O(N)，且 `full` 模式在一串 update 上是二次的。E8 的 `1000 facts 52.9 ms` 必须改述为快照模式的实现成本。

### 2.4 有界性 ≠ 硬实时，且边界能量化

注入 30 ms 慢阶段（每 20 帧）：

| 指标 | guard off | guard on |
|---|---:|---:|
| BAR mean | 0.9439 | 0.9986 |
| p99 | 31.31 ms | 8.88 ms |
| max | 59.94 ms | **64.91 ms** |
| timeouts / degraded（57 runs） | 0 | 241 |
| relation rows | 16,520.9 | 14,254.7（−13.7%） |
| ID retention | 1.000 | 1.000 |

**结论**：调用方等待可以限界，代价是有界降级（丢 13.7% 关系行）；底层 worker 没变快，guarded max 仍在 65–80 ms，残余是 OS 抢占 guard 线程。硬墙钟上界需要 OS 级实时调度。这把"我们有尾延迟问题"升级为"anytime 契约在非 RT OS 上的可达边界"。

### 2.5 方法论：能力指标检测不到契约违规

naive swap（直接改 `registry._modules['vision']`，绕过校验，回滚栈为空）：

| 指标 | 正例（4 bundle） | naive（4 bundle） |
|---|---|---|
| SSR / RBE | 1.000 | **0.000 / 0.000** |
| RCR | 1.000 | **0.000（4/4）** |
| INT_post | 1.000 | 0.968（**仅 2/4 下降**） |

**结论**：有 2 个 bundle 上，一个回滚已经坏掉的系统在能力指标上看起来正常。这是 contract-level protocol 不可被 capability benchmark 替代的最直接实验证据。

---

## 3. 论文修正清单（diff 级别）

### 3.1 最重要：撤回 `hybrid_diag` 的优势主张（E3 → F8）

**位置**：`paper/main.tex:607–618`（§Results）、`:600–603`（图注）、`:767–771`（§Discussion）

**原文（607–618）**：

```latex
Figure~\ref{fig:relations} shows the relation result. In-distribution F1 is
high for all three types. Removing tactile drops tactile F1 to 0.483, so the
head is not relying on a shortcut. On held-out regions under coordinate noise,
learned relations generalize better than an end-to-end policy but worse than
exact geometry. The distribution-aware hybrid is the best at high noise; the
naive hybrid, which applies learned relations to unseen aligned axes, is the
worst.

The conclusion is narrow. Learned relations are a calibration channel, not a
replacement for geometry. The correct deployment is a hybrid: exact geometry
owns the aligned and held-out axes; learned relations calibrate the clear
regime they were trained on.
```

**建议改为**：

```latex
Figure~\ref{fig:relations} shows the relation result. In-distribution F1 is
high for all three types. Removing tactile drops tactile F1 to 0.483, so the
head is not relying on a shortcut. On held-out regions under coordinate noise,
learned relations generalize better than an end-to-end policy but worse than
exact geometry.

We do not claim a learned-relation gain. In a three-seed replication on held-out
regions, a learned gate paired with an out-of-distribution fallback reaches
0.969 at noise 0.15, against 0.829 for geometry at $\epsilon=0.25$ and 0.859
for the distribution-aware hybrid; but a fixed geometric dead zone selected on
validation reaches 0.972, and the fallback fires on 94.3\% of held-out pairs,
so the learned path contributes no accuracy once the dead zone is tuned. The
earlier hybrid advantage does not survive a tuned dead zone.

The conclusion is narrower still. Learned relations are a calibration channel
inside the regime they were trained on. The deployment rule is a
geometry-owned dead zone whose width is selected on validation; the naive
hybrid, which applies learned relations to unseen aligned axes, is the worst
of the candidates.
```

**图注（600–603）**：

```latex
% before
(b) Hybrid relations under coordinate noise on held-out regions. Learned
relations help only inside their training regime; the naive hybrid is worse
than geometry.

% after
(b) Hybrid relations under coordinate noise on held-out regions. Learned
relations help only inside their training regime; the naive hybrid is worse
than geometry, and a learned gate with an OOD fallback does not beat a
validation-tuned geometric dead zone.
```

**Discussion（767–771）**：

```latex
% before
The hybrid result is conditional. Learned relations help in the clear-diagonal
regime they were trained on, and hurt on unseen aligned axes. Exact geometry
remains the backbone. This is a general lesson for neurosymbolic systems: the
learned component should be scoped to the distribution where it has evidence,
and the symbolic component should own the cases where it is exact.

% after
The hybrid result is conditional, and the condition is met by the symbolic
component alone. Learned relations help in the clear-diagonal regime they were
trained on and hurt on unseen aligned axes; a learned gate with an
out-of-distribution fallback does not beat a validation-tuned geometric dead
zone (0.969 vs 0.972 at noise 0.15). Exact geometry remains the backbone. This
is a general lesson for neurosymbolic systems: when the symbolic component has
a tunable tolerance, the learned component must be shown to beat the tuned
symbolic baseline, not the untuned one.
```

> **不要动** `paper/main.tex:326–331`（§Intro）。那里只说 "learned relations generalize better than an end-to-end policy but worse than exact geometry" 和 "naive hybrid 更差"，这两句 F8 支持保留。要撤的只有 "distribution-aware hybrid 在高噪声下最好"。

### 3.2 尾延迟：从"未满足的上界"改为"有界工作的边界"

**位置**：`paper/main.tex:652–657`

**原文**：

```latex
Bounded inference is measured as a contract. The quality--budget curve is in
the supplementary material; the node count stays within the cap, and the
degradation FSM transitions correctly under injected faults. The tail latency of
the L2 bundle is not fully covered: two of fourteen traces contain single-frame
spikes above 10 ms. We report this as a partial result and a need for a latency
guard, not as a satisfied bound.
```

**建议改为**：

```latex
Bounded inference is measured as a contract. The quality--budget curve is in
the supplementary material; the node count stays within the cap, and the
degradation FSM transitions correctly under injected faults.

The tail latency of the L2 bundle is bounded, not hard. Unguarded replay
exceeds 10 ms on 4 of 57 repeats, with a worst frame of 67.3 ms. A deadline
guard that bounds the caller wait restores BAR from 0.944 to 0.999 and p99 from
31.3 ms to 8.9 ms under an injected 30 ms slow stage, at the cost of 13.7\%
fewer relation rows and 241 shed frames per 57 repeats. The guard does not make
the worker fast: the guarded maximum remains 64.9 ms, and the residual overrun
is guard-thread preemption. We therefore report a bounded-work contract with
anytime degradation at the API boundary, not a hard wall-clock bound; a hard
bound requires OS-level real-time scheduling.
```

**Limitation（`paper/main.tex:821–823`）**：

```latex
% before
\item \textbf{Tail latency.} The L2 bundle violates the 10 ms bound on two of
      fourteen traces. A latency guard is needed.

% after
\item \textbf{Tail latency is bounded, not hard.} Unguarded L2 replay exceeds
      10 ms on 4 of 57 repeats (worst 67.3 ms). A deadline guard bounds the
      caller wait (BAR 0.944$\to$0.999, p99 31.3$\to$8.9 ms under an injected
      30 ms slowdown) but the guarded maximum stays at 64.9 ms because a
      non-real-time OS can preempt the guard thread. We claim bounded work with
      anytime degradation, not a hard wall-clock bound.
```

### 3.3 TMS：从"包了一层"改为"runtime 默认 + 代价分解"

**位置**：`paper/main.tex:661–665`

**原文**：

```latex
The production \texttt{MultiSourceFactStore} is wrapped by \texttt{TMSFactStore};
K3/K4/K5, audit completeness, and source retraction are all $1.00$.
```

**建议改为**：

```latex
The production \texttt{MultiSourceFactStore} is wrapped by \texttt{TMSFactStore},
which is now the runtime's default fact store with delta snapshots; K3/K4/K5,
audit completeness, and source retraction are all $1.00$. At 10{,}000 facts a
single-fact update costs 0.014 ms in median against 580.8 ms for whole-graph
snapshots, and incremental propagation costs 0.013 ms against 78.5 ms for full
re-evaluation, because truth maintenance no longer scales with the graph.
Rollback and the first query after rollback remain $O(N)$ at 6.2 ms and
9.9 ms, so the decoupling covers update and propagation, not restore.
```

### 3.4 Limitations：两条销账、两条收窄

**位置**：`paper/main.tex:801–825`

```latex
\begin{itemize}[leftmargin=*]
  \item \textbf{No real robot.} All evidence is from MuJoCo, Isaac, and offline
        bundle replay. The report does not claim real-world reliability. The
        nearest neighbours are object-level replay on public data
        (ObjectFolder 2.0; a retrained encoder reaches cross-modal R@1 0.22 /
        R@5 0.73) and perception-level swap on detector embeddings. Neither is
        sim2real.
  \item \textbf{L2 is not fully perception-driven.} The identity path is now
        ground-truth free: the perception-level swap consumes only
        \texttt{obs\_detections.parquet} and frame images, and GT is opened only
        by the post-hoc scorer. The geometry relation path still uses a fixed
        geometry stub, so a fully perception-driven L2 harness is still open.
  \item \textbf{Expert width is no longer the barrier.} The admission contract
        is width-agnostic: DINOv2 384, CLIP 768, and WT-MM 64 all satisfy
        reject / accept / review, with k$\ge$1 vision accuracy 0.983 / 1.000.
        What remains is calibration, not width: zero-shot $k=0$ is
        0.000--0.200 for all three, and the tactile bridge stays at or below
        0.617.
  \item \textbf{Reduced design.} The factorial design is reduced; L1 has five
        seeds. The interactions between all factors are not measured.
  \item \textbf{Partial production integration.} The runtime's fact layer
        defaults to \texttt{TMSFactStore} with delta snapshots; remaining
        unmigrated call sites are outside \texttt{RelationFactLayer}.
  \item \textbf{Protocol scope.} The protocol separates the factors it
        manipulates. It does not discover unknown confounders, and it is not a
        competitive benchmark. It currently ships with two external adapters
        (a flat runtime and a transformer-state baseline); the transformer
        baseline cannot be scored on contract properties by construction, so
        external validity remains thin.
  \item \textbf{Tail latency is bounded, not hard.} (见 3.2)
\end{itemize}
```

### 3.5 Conclusion：下一步的清单已经变了

**位置**：`paper/main.tex:835–839`

**原文**：

```latex
\pss is a cognitive runtime above perception, not a foundation model. It makes
sensory experts pluggable under a contract, makes the cognitive state explicit
and auditable, and measures architectural claims with a property-level protocol
that separates expert quality from architecture.

The evidence supports contract-level modularity, a calibratable-interface view
of shared semantic spaces, and honest negative results. It does not support
capability SOTA or zero-shot public-space claims. The next steps are a
perception-driven L2 harness, more expert families, a real-robot demonstration,
and a competitive non-\pss baseline for the evaluation protocol.
```

**建议改为**：

```latex
The evidence supports contract-level modularity, a calibratable-interface view
of shared semantic spaces, and honest negative results. It does not support
capability SOTA or zero-shot public-space claims.

Three of the four next steps in the earlier version of this report are now
closed: a perception-level L2 harness exists (\S\ref{sec:perception-swap}), the
expert-width barrier is gone (\S\ref{sec:widths}), and the protocol ships as a
standalone package with a system-under-test adapter (\S\ref{sec:protocol}). What
remains is a real-robot demonstration, a competitive external baseline that can
be scored on contract properties, and a hard real-time scheduling substrate for
the latency claim.
```

> 上面三个 `\ref` 指向第 4 节建议新增的小节，label 需要随新增小节一起创建；若暂不新增小节，则把三处 `\S\ref{...}` 换成对附录表格的引用。

### 3.6 仓库内部一致性（不改论文，但必须改）

| 位置 | 问题 | 处理 |
|---|---|---|
| `reports/claim_gaps/INDEX.md:39` | "Strongest new evidence for C2: E3 (distribution-aware hybrid)" 与同行文件里的 F8 行直接矛盾 | 改为 "F8（唯一安全收益是更宽/自适应几何死区；E3 的 hybrid_diag 优势通不过调好的死区）" |
| `reports/claim_gaps/INDEX.md:10` | E3 行仍只写 "hybrid_diag 0.857 vs geo_tau 0.828"，未标注已被 F8 收窄 | 补充 "(superseded by F8; see 0.972 fixed dead zone)" |
| `reports/claim_gaps/INDEX.md:17` / `FOLLOWUPS.md:22` | "1000 facts update 52.9 ms" 未说明是 full 快照模式 | 补 "full snapshot mode; delta mode is 0.014 ms at 10k facts (F7)" |
| `docs/plans/PSS-后续研究方向评估.md:240` | "P0/P1/P2 既定任务已闭合" 没有给出结论层收口 | 加一句指向本文档 |

---

## 4. 建议在论文中新增的内容

这批工作有几项是**新证据**而不是**修正**，值得单独成节。建议按下面的最小形态加入，不要扩写。

| 新增内容 | 建议位置 | 内容上限 | 证据 |
|---|---|---|---|
| §Perception-level swap | §Results，图注 `fig:detection-l2` 之后 | 1 段 + 1 张 4×4 表（bundle × SSR/RBE/INT/RCR）+ naive 对照列 | F2 |
| §Width-agnostic admission | §Results，`fig:families` 之后 | 1 段 + 1 张 3×4 表（宽度 × k=0/1/5/20）+ 准入裁决列 | F4 |
| §Cost decomposition | §Results，TMS 段之后 | 1 段 + 1 张 3×3 表（facts × full/cow/delta 的 update/rollback/传播） | F7 |
| §Protocol as artifact | §Evaluation 或附录 | 1 段 + adapter 接口签名 + 预注册状态 | F5 |
| §Anytime vs hard bound | §Discussion，尾延迟段 | 1 段 + 8 行表（off/on × 无注入/注入） | F3 |
| 复现指令 | 附录 `app:repro` | `make reproduce-l2` + release URL + SHA256 | F1 |

**不建议**为 F9（合成流）、F10（object-level）单独开节——放在 Limitations 和一段 "external validity" 里即可。它们的价值是界定边界，不是提供主张。

---

## 5. 边界：新证据不能支持的话

| 不能声称 | 原因 | 证据 |
|---|---|---|
| 真机 / sim2real / 闭环具身 | 无硬件；ObjectFolder 只有 object-level | F10 SPIKE |
| 竞争性能力 benchmark | 外部 adapter 只有一个，且按构造 skip 掉 contract 属性 | F5 |
| 感知级 swap 的统计稳健性 | 单 swap 点、单 seed、4 个 bundle；校准是 transductive | F2 |
| 检测位姿等价于 GT | match rate 0.870–0.998，pos mean 1.7–31.8 mm，`s1_occlusion_real` 只匹配 87% | F2 |
| 多 query 选择提升答案正确性 | 合成流，coverage 不等于正确性 | F9 |
| 投毒检测普遍有效 | σ=0.05 噪声与 5% label flip 会漏检 | F6 |
| 硬 10 ms 上界 | guarded max 仍 64.9–80.2 ms | F3 |
| 学习关系门控优于几何 | 0.969 < 0.972 | F8 |

---

## 6. v0.2 的立意

如果要把这批工作写进论文或作为 v0.2 的差异化，建议只立一句话：

> 我们把论文剩余的模拟器与 oracle 脚手架拆掉后，契约仍然成立（F1/F2/F4/F5/F7），而三条先前报告的正面结果分别被证明是调参 artifact（E3→F8）、实现成本（E8→F7）和未达成的上界（10 ms→F3）。因此 runtime 与 protocol 的贡献不是"能力更强"，而是**主张可被证伪且经得起证伪**。

对应的三件可交付物：

1. **复现工件**（F1）：公开 release + `make reproduce-l2`，干净 clone 10 分钟内复现。
2. **独立协议包**（F5）：`pss_eval` + `SystemUnderTest` adapter + 可执行预注册。
3. **契约级负结果**（F3/F7/F8）：三个"边界量化"，每个都带机制解释。

这个立意的优点：不需要新增 capability claim，不需要真机，不与"not a foundation model"的定位冲突；6/10 的负结果比例反而是可信度背书。

---

## 7. 执行顺序与工作量

| 顺序 | 动作 | 工作量 | 依赖 |
|---:|---|---|---|
| 1 | 改 `INDEX.md` 三处过期指引 | 0.5 h | 无 |
| 2 | 论文 3.1（撤回 hybrid 主张）——风险最高，先定稿 | 0.5 天 | 无 |
| 3 | 论文 3.2 + 3.3（尾延迟、TMS 改述） | 0.5 天 | 无 |
| 4 | 论文 3.4 + 3.5（Limitations、Conclusion） | 0.5 天 | 2、3 |
| 5 | 新增 §Perception-level swap + §Width-agnostic（4 的两节） | 1 天 | 2 |
| 6 | 新增 §Cost decomposition + §Protocol artifact | 1 天 | 5 |
| 7 | 附录复现指令 + release 链接 | 0.5 天 | 无 |
| 8 | 图 `fig:relations` 重绘（加 tuned dead zone 曲线） | 0.5 天 | 2 |

合计约 4–5 天。第 2 项是唯一会改变论文结论的改动，应该先做并单独审阅。

---

## 8. 证据索引

| 主题 | 证据 |
|---|---|
| 感知级 swap | `reports/claim_gaps/perception_swap/REPORT.md` |
| 尾延迟 / guard | `reports/claim_gaps/latency_guard/REPORT.md`（E21） |
| 异构宽度准入 | `reports/claim_gaps/families_width/REPORT.md` |
| 协议包 | `pss_eval/`、`reports/claim_gaps/protocol_pss/REPORT.md`、`docs/plans/PSS-F5-协议包与预注册.md` |
| 在线校准 / 投毒 | `reports/claim_gaps/online_calibration/REPORT.md` |
| TMS 规模 | `reports/claim_gaps/tms_scale/REPORT.md` + `REGRESSION.md` |
| 关系门控 | `reports/claim_gaps/relation_gating/REPORT.md` |
| 多 query | `reports/claim_gaps/fact_multi_query/REPORT.md` |
| 真实回放 | `reports/claim_gaps/real_replay/SPIKE.md` |
| L2 记忆消融 | `reports/claim_gaps/l2_memory_all/REPORT.md` |
| 论文待改位置 | `paper/main.tex:600–603, 607–618, 652–657, 661–665, 767–771, 801–825, 835–839` |
| 仓库过期指引 | `reports/claim_gaps/INDEX.md:10, 17, 39` |
