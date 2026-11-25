# Phase 6 解体规则报告（计划 7.4）

> 日期：2026-11-22
> 实现：`pss/memory/decomposition.py`
> 演示：`pss/memory/scripts/run_decomposition_demo.py`
> 输出：`eval/results/decomposition_demo/`

---

## 1. 交付

| 分支 | 行为 | 状态 |
|---|---|---|
| 目睹解体 | 调用 `SceneGraph.fragment`，父实体标 `fragmented`，子碎片写 `part-of-fracture` 溯源 | ✅ |
| 仅结果观测 | 几何贴合 + 原型相似度 → 置信度假说；过阈值才写场景图 | ✅ |
| 证据不足 | 输出 `uncertain`，不改场景图 | ✅ |
| 假说推翻 | `retract()` 恢复父实体、解除碎片归属、保留撤销溯源 | ✅ |
| 假说重评 | `revise()` 用新证据重算，可升级或撤销 | ✅ |

核心 API：

```python
rule = DecompositionRule(graph, bank=None)
rule.witness(parent_id, shards, t)          # 确定性分支
hyp = rule.infer_result_only(parent, shards, t)  # 概率分支
rule.retract(hyp.id, reason="parent intact", t=9)
rule.revise(hyp.id, parent, new_shards, t=10)
```

## 2. 演示结果

| 分支 | 证据 | 置信度 | 状态 | 是否写图 | 父实体 |
|---|---|---:|---|---|---|
| 目睹解体 | 直接观测 | 1.000 | deterministic | 是 | fragmented |
| 仅结果，强证据 | 碎片距父 3–4 cm，特征余弦 1.0 | 0.958 | probabilistic | 是 | fragmented |
| 仅结果，弱证据 | 碎片距父 50 cm，特征正交 | 0.326 | uncertain | 否 | active |
| 强证据后推翻 | 父实体再次完整出现 | 0.000 | retracted | 撤销 | active |

弱证据不会写场景图；强证据被推翻后，碎片 `parent` 清空、`decomposition_status=retracted`，父实体恢复 `active`。

## 3. 置信度构成

```text
confidence = 0.55 * geometry + 0.45 * prototype
geometry   = 0.45 * within + 0.35 * compactness + 0.20 * count
prototype  = 0.6 * feature_cosine + 0.4 * category_match   # 有 prototype bank 时
```

- `within`：碎片到父实体 last-known pose 的距离衰减。
- `compactness`：碎片两两最大间距相对预期 extent 的衰减。
- `count`：≥2 个碎片才给满分。
- 状态阈值：`≥0.72` probabilistic 并写图；`0.45–0.72` candidate，只记录；`<0.45` uncertain。

## 4. 验收

- [x] 目睹事件 → 确定性结果 + `part-of-fracture` 溯源。
- [x] 仅结果观测 → 带置信度的概率假说。
- [x] 证据不足 → 明确输出 `uncertain`，不强行确定。
- [x] 假说可被推翻，父实体和碎片归属可恢复。
- [x] 单元测试覆盖四个分支，全量 pytest 通过。

## 5. 局限

- **没有真实断裂数据**：演示用合成 Track，仿真器还没有"物体碎裂"场景。规则接口已就绪，等动作层/仿真器产生碎片观测后可直接接。
- **几何贴合是包围盒代理**：用距离和 spread 近似，没有网格拼接或接触面匹配。
- **阈值是手定的**：`0.72/0.45` 还没有在真实碎片分布上标定。
- **原型相似度是特征余弦 + top-1 类别匹配**：没有做碎片级属性一致性检查。

## 6. 复现

```bash
python -m pss.memory.scripts.run_decomposition_demo
python -m pytest -q tests/test_memory.py
```

输出：

- `eval/results/decomposition_demo/metrics.json`
- `eval/results/decomposition_demo/REPORT.md`
