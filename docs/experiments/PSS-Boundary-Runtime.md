# PSS-Boundary：runtime 轴谓词接入（Task 4）

> 日期：2026-11-28
> 上游：`docs/experiments/PSS-Boundary-Report.md`
> 变更：`pss/reasoning/relations/bayes_predicate.py`、`pss/reasoning/relations/rules.py`、`pss/runtime/layers.py`、`pss/runtime/runtime.py`
> 回归：`tests/test_bayes_predicate.py` + runtime 子集（31 项通过）

---

## 0. 一句话

`RelationRuleSet` 原来只产出 `in/contact/near`；现在新增 `left/right/above/below` 轴谓词，支持三种模式：**off（默认，行为不变）/ hard（可审计 fallback）/ bayes（经验 Bayes 边界）**。Bayes 模式用 `GridBayesBoundary`，hard 模式用固定 `eps`；两者都写回 `RelationFactLayer`，审计里能看到 `origin=hard_geometry` 或 `bayes_geometry`。

---

## 1. 改动

| 文件 | 改动 |
|---|---|
| `pss/reasoning/relations/bayes_predicate.py` | 新增 `AxisPredicate`：`hard` / `bayes` 两种模式，输出 `left/right/above/below`，同时写反向关系 |
| `pss/reasoning/relations/rules.py` | `RelationRuleSet` 新增 `axis_mode` / `axis_boundary` / `axis_sigma` / `axis_eps` / `axis_abstain`；`relations()` 在 `axis_mode != "off"` 时追加轴记录 |
| `pss/runtime/layers.py` | `RelationFactLayer` 新增同名参数并传给 `rule_set`；`add_symbolic` 保留 `hard_geometry` / `bayes_geometry` 的 origin |
| `pss/runtime/runtime.py` | `PSSRuntime` 新增同名参数；默认构造 `RelationFactLayer` 时传入；自定义层也会同步配置 |

---

## 2. 三种模式

| 模式 | 行为 | 用途 |
|---|---|---|
| `off`（默认） | 只产出 `in/contact/near`，与历史行为完全一致 | 兼容旧测试/旧部署 |
| `hard` | 固定 dead zone `eps`；`dx > eps → left`、`dy > eps → below` 等 | 可审计 fallback；无先验/无 σ 时 |
| `bayes` | `GridBayesBoundary.predict_proba` 的局部后验；记录 confidence | 有 clean 先验 + σ 时的推荐模式 |

反向关系：`left ↔ right`、`above ↔ below` 成对写入，confidence 相同。

---

## 3. 审计

- `RelationRecord.source` 为 `hard_geometry` / `bayes_geometry`。
- `Relation.origin` 保留该 source；`relation_type` 仍是 `symbolic`。
- `axis_mode` / `axis_sigma` / `axis_eps` 是**配置**，不进 `RelationRuleSet.snapshot()` 的四个学习阈值；回滚语义不受影响。
- Bayes 的 confidence 来自后验最大概率；`axis_abstain > 0` 时低于阈值的谓词不写。

---

## 4. 用法

```python
from pss.reasoning.relations.bayes_boundary import GridBayesBoundary
from pss.runtime import PSSRuntime

# Fit the prior on clean coordinate pairs at the runtime's scale.
boundary = GridBayesBoundary(eps=0.02, bins=257).fit(dx_clean, dy_clean)

runtime = PSSRuntime(
    axis_mode="bayes",
    axis_boundary=boundary,
    axis_sigma=0.01,
    axis_eps=0.02,
    axis_abstain=0.0,
)
# add_geometry() now also writes left/right/above/below with origin="bayes_geometry".
```

Hard fallback:

```python
runtime = PSSRuntime(axis_mode="hard", axis_eps=0.02)
```

---

## 5. 为什么默认是 `off`，不是 `bayes`

- `off` 保证旧行为和旧测试不变；轴谓词是新增记录，默认开启会改变所有 runtime 场景的关系数量。
- `bayes` 需要一份**在 runtime 坐标尺度上拟合的 clean 先验**；没有先验时不能静默启用。
- 推荐部署配置：**`axis_mode="bayes"` + 启动时拟合先验 + `hard` 作为 `sigma` 缺失时的 fallback**。代码已经支持；是否设为默认应由部署配置决定，而不是库默认。

---

## 6. 测试

`tests/test_bayes_predicate.py`：

- hard 模式写出 `a right b` / `b left a`、`c above b` / `b below c`；
- `RelationRuleSet()` 默认不写轴谓词；`axis_mode="hard"` 才写；
- Bayes 模式在合成先验上给出 `a right b`，confidence ≥0.5；
- `RelationFactLayer(axis_mode="hard").add_geometry` 写入 `origin="hard_geometry"`。

回归：`test_runtime.py`、`test_runtime_phase17.py`、`test_runtime_permissions.py`、`test_runtime_semantics_lock.py`、`test_runtime_rules.py` + 新测试，共 **31 项通过**。

---

## 7. 边界

- 先验必须与 runtime 坐标同尺度；换场景/换单位要重新 fit。
- Bayes 模式假设噪声 σ 已知或可估；σ 缺失时用 hard fallback，不要用 0 冒充。
- 轴谓词只覆盖桌面 XY 的 `left/right/above/below`；不覆盖 `front/behind` 或 3D 谓词。
- 本轮没有把 `axis_mode="bayes"` 设为库默认；这是兼容性选择，不是能力限制。
- 没有做在线先验漂移检测；部署前需要监控 `bayes_geometry` 的 confidence 分布。

---

## 8. 一句话

**轴谓词已接入 runtime：`off/hard/bayes` 三模式，Bayes 用局部先验 + 噪声似然，hard 做可审计 fallback；默认 off 保兼容，部署建议显式启用 bayes。**
