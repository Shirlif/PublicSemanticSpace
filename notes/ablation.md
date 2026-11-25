# 9 变体 preliminary 消融（dev）

> 日期：2026-11-25
> 数据：S1 `s1_occlusion_20`（20 cubes, 5 seeds, oracle）、S6 `s6_memory`（5 子 episode, 1 seed, oracle）
> 性质：**dev preliminary**，不是 official 证据。

## S1 实体层消融

| variant | IDF1 | retention | switches | duplicate | reappearance |
|---|---|---|---|---|---|
| full | 1.0000 | 1.0000 | 0.000 | 0.0000 | 1.000 |
| no_entity | 0.0113 | 0.0113 | 45.0 | 0.989 | 0.000 |
| no_relations | 1.0000 | 1.0000 | 0.000 | 0.0000 | 1.000 |
| no_memory | 1.0000 | 1.0000 | 0.000 | 0.0000 | 1.000 |
| no_active_set | 1.0000 | 1.0000 | 0.000 | 0.0000 | 1.000 |
| no_uncertainty | 1.0000 | 1.0000 | 0.000 | 0.0000 | 1.000 |
| no_audit | 1.0000 | 1.0000 | 0.000 | 0.0000 | 1.000 |
| flat_context | 0.0113 | 0.0113 | 45.0 | 0.989 | 0.000 |
| no_tools | 1.0000 | 1.0000 | 0.000 | 0.0000 | 1.000 |

**解读**：
- S1 的指标只对“持久实体槽”敏感；`no_entity/flat_context` 每帧新建 ID，IDF1 直接崩到 0.011。
- `no_relations/no_uncertainty/no_audit/no_tools/no_active_set` 在 S1 上不改变实体指标——
  这是**预期**，不是它们无效；它们的作用应在 S3/S2/S6 上测，S1 不覆盖。
- 所以 S1 消融只能支持一个结论：**实体槽是 S1 的必要组件**；不能据此推断其他组件无用。

## S6 记忆消融

| variant | edited | old_forgetting | retention | rollback |
|---|---|---|---|---|
| full | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| no_entity | 1.0000 | 1.0000 | 0.0067 | 1.0000 |
| no_memory | 1.0000 | 1.0000 | 0.2000 | 1.0000 |
| no_relations | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| no_audit | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| flat_context | 1.0000 | 1.0000 | 0.0067 | 1.0000 |

**解读**：
- `no_memory` 让跨 episode 的实体 ID 断开，retention 从 1.0 掉到 0.2；
  `no_entity`/`flat_context` 掉到 0.0067。
- `no_audit` 让 rollback 不可用（0.0），符合预期。
- **关键限制**：`edited_accuracy` / `old_forgetting` 在当前实现里基于“当前 PSS 位姿”，
  每个 episode 都会重新观测，所以记忆消融不会改变它们。它们测的是“当前事实是否
  更新”，不是“旧事实是否被记住/覆盖”。真正的事实级记忆需要把 `state.facts` 接进
  S6 指标，这是下一步。

## 结论边界

- 可以说的：在 dev/oracle 下，**实体槽是 S1 的必要条件**；**跨 episode 记忆对 retention
  是必要条件**；**audit 对 rollback 是必要条件**。
- 不能说的：其他组件“无用”；不能把 S1/S6 的结论外推到 S3/S2；不能当 official。
- 下一步：
  1. 把 `state.facts` 写进 S6 指标，让 edited/old 对记忆消融敏感；
  2. 在 S2/S3 上跑 no_uncertainty/no_relations 的对应消融；
  3. 用 `frozen_real` 检测重复 S1 消融，确认结论不依赖 oracle；
  4. official 在 3090 复跑同一矩阵。
