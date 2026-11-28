# PSS 三维场景模型接入 Runtime（D5 / S4）

- 日期：2026-11-28
- 阶段：S4（PSS 集成，见 `docs/plans/PSS-三维场景建模实验计划.md` §5、§8-S4、§10）
- 范围：`SceneModel` + `solve_scene` → `Entity` / `Relation` / `CognitiveState` 审计
- 结论：链路打通、可审计、opt-in；**不**把三维场景模型设为 runtime 默认几何层，1D `bayes_predicate` 仍是 fallback。

---

## 1. 接入链路

```
CLEVR SceneModel (pss/scene/model.py)
  + solve_scene(example, config)                  # 3D 位置 + 线性化 3x3 协方差
        │
        ├─ scene_to_entities(scene, positions, covariances)
        │        └─ Entity(pos, quat, attributes["covariance"], provenance, confidence)
        │
        ├─ scene_relation_records(scene, positions, covariances, eps_*, table_z)
        │        └─ RelationRecord(aligned_x / aligned_y / same_level / contact / collision / support)
        │
        └─ apply_scene_to_runtime(runtime, scene, positions, covariances, ...)
                 ├─ CognitiveState.add_entity(...)      # 不新增 state 字段
                 ├─ CognitiveState.add_relation(...)    # origin="scene_model"
                 └─ CognitiveState.commit("scene_ingest", scene_id, ...)  # audit
```

实现文件：`pss/scene/pss_bridge.py`。

- `scene_to_entities(scene, positions, covariances=None)` → `list[Entity]`
- `scene_relation_records(scene, positions, covariances, eps_xy, eps_z, eps_c, table_z, *, eps_s=0.05, probabilities=None)` → `list[RelationRecord]`
- `apply_scene_to_runtime(runtime, scene, positions, covariances=None, *, eps_xy, eps_z, eps_c, table_z, eps_s, probabilities, actor="scene_model", commit=True)` → 摘要 dict

`apply_scene_to_runtime` 只调用 `CognitiveState.add_entity` / `add_relation` / `commit`，**没有修改** `pss/runtime/state.py`、`layers.py` 或 runtime 默认值。协方差仅承载在 `Entity.attributes["covariance"]` 中。

---

## 2. 写入字段

### 2.1 Entity

| 字段 | 内容 |
|---|---|
| `id` | 场景内对象 id（如 `obj00`） |
| `category` | `ObjectState.shape`（`cube` / `sphere` / `cylinder`） |
| `pos` | `solve_scene` 估计的 3D 位置 |
| `quat` | 由 yaw 构造的 z 轴四元数 `[cos(yaw/2), 0, 0, sin(yaw/2)]` |
| `attributes["covariance"]` | 3x3 list；无协方差时为零矩阵 |
| `attributes["covariance_source"]` | `"linearised"` 或 `"missing"` |
| `attributes["radius" / "size" / "shape" / "scene_id"]` | 场景对象元数据 |
| `confidence` | 由协方差迹映射：`1 / (1 + trace / 3e-3)`，裁到 `[0,1]`；无协方差为 1.0 |
| `provenance["source"]` | `"scene_model"` |
| `provenance["scene_id"]` | 场景 id |
| `provenance["object_id"]` | 对象 id |
| `status` | `"active"` |

**confidence 不是校准概率。** 它只保证对协方差迹单调；S3 的校准结果才是概率口径。关系 confidence 同理。

### 2.2 Relation

| 字段 | 内容 |
|---|---|
| `source` / `target` | 对象 id；`support` 的 target 是 `"table"` |
| `relation_type` | `"scene_geometry"`（与 1D 的 `"symbolic"` / 学习边区分） |
| `origin` | `"scene_model"` |
| `frame` | `"world"` |
| `predicate` | 见 §3 |
| `probability` | 启发式 confidence，裁到 `[0,1]` |

**为什么不用 `RelationFactLayer.add_symbolic`：** 该方法把 `RelationRecord.source` 只识别为 `bayes_geometry` / `hard_geometry`，其余一律写成 `origin="symbolic"`，无法标记三维场景来源。为保留可审计的来源，`apply_scene_to_runtime` 直接调用 `CognitiveState.add_relation` 写 `origin="scene_model"`；`add_geometry` / `add_symbolic` 仅作为接口参考。

### 2.3 Audit

每次 `apply_scene_to_runtime(commit=True)` 追加一条：

| 字段 | 内容 |
|---|---|
| `action` | `"scene_ingest"` |
| `target` | `scene_id` |
| `actor` | `"scene_model"`（可改） |
| `reason` | `"3D scene model -> runtime"` |
| `after` | `{scene_id, source, entities, relations, predicates, eps_xy, eps_z, eps_c, eps_s, table_z}` |

`commit=False` 时不写 audit、不改 version，便于纯查询/测试。

---

## 3. 三维关系谓词与 confidence

定义与 `pss/scene/metrics.py` 保持一致（估计位姿上查询）：

| predicate | 条件 | confidence 启发式 |
|---|---|---|
| `aligned_x` | `|dx| <= eps_xy` | `1 - |dx|/eps_xy` |
| `aligned_y` | `|dy| <= eps_xy` | `1 - |dy|/eps_xy` |
| `same_level` | `|(z_a-r_a) - (z_b-r_b)| <= eps_z` | `1 - |Δbottom_z|/eps_z` |
| `contact` | `d <= r_a+r_b+eps_c` | 表面间隙 `d-(r_a+r_b)`；重叠时 1.0 |
| `collision` | `d < r_a+r_b` | 归一化穿透 `((r_a+r_b)-d)/(r_a+r_b)`，下限 0.05 |
| `support` | `|(z-r) - table_z| <= eps_s` | `1 - |gap|/eps_s` |

- `axis_aligned` 拆成 `aligned_x` / `aligned_y`；没有额外发 `left/right/above/below`，因为"对齐"时符号方向本身退化，方向谓词仍由 1D `bayes_predicate` 负责。
- 对称谓词双向各写一条（`a→b`、`b→a`），`support` 每个对象一条。
- 若传入 `covariances`，记录 confidence 再乘以两端 `positional_confidence` 的上限；传入 `probabilities={(subject,predicate,object): p}` 可外部覆盖，且最后生效。
- 上述全部是启发式，不是校准概率。

---

## 4. 测试

```
python -m pytest tests/test_scene_runtime.py -q
......                                                                   [100%]
```

6 个测试全部通过：

1. 合成场景 + `solve_scene` 后 `scene_to_entities`：位姿、3x3 协方差、`provenance`、confidence 范围。
2. 无协方差路径：零矩阵 + `covariance_source="missing"` + confidence 1.0。
3. `scene_relation_records`：`aligned_x` / `same_level` / `contact` / `collision` / `support` 均出现，双向记录；外部 `probabilities` 覆盖生效。
4. `apply_scene_to_runtime`：entities / relations / audit / version / `origin="scene_model"` / `relation_type="scene_geometry"`。
5. `commit=False`：不写 audit、version 不变。
6. 1D fallback：`AxisPredicate(mode="hard")` 仍产出 `left/right`；`RelationFactLayer().rule_set.axis_mode == "off"`，接入后也不变。

---

## 5. Demo 实跑

```
python -u -m pss.eval.scripts.run_scene_runtime
```

数据：`data/public/clevr_cogent/CLEVR_CoGenT_v1.0`，split `vala`，5 个场景，3 视角，σ_px 2.0，σ_depth 0.01，无 outlier；`eps_xy=0.12`、`eps_z=0.05`、`eps_c=0.02`、`eps_s=0.05`、`table_z=0.0`。

| 场景 | entities | relations | predicates | pose median |
|---|---:|---:|---|---:|
| scene00000 | 8 | 64 | same_level 56, support 8 | 0.0622 |
| scene00001 | 7 | 49 | same_level 42, support 7 | 0.0431 |
| scene00002 | 5 | 20 | same_level 16, support 4 | 0.0430 |
| scene00003 | 10 | 91 | aligned_x 2, same_level 80, support 9 | 0.0520 |
| scene00004 | 10 | 100 | aligned_x 4, aligned_y 2, same_level 84, support 10 | 0.0453 |
| **合计** | **40** | **324** | aligned_x 6, aligned_y 2, same_level 278, support 38 | — |

校验输出：

```
provenance=ok covariance=ok relation_origin=ok audit=ok
1D fallback: records=2 predicates=['left', 'right'] runtime_axis_mode='off' ok
```

说明：

- 5 个 CLEVR 场景中没有接触/碰撞对，因此 `contact` / `collision` 计数为 0；测试里用合成重叠对象覆盖了这两个谓词。
- 位置误差中位数 4.3–6.2 cm，高于计划 §7.2 的 1 cm bar。这不是本次集成引入的；本阶段直接复用 S1 的 `solve_scene`，未调估计器/噪声模型。数字照实记录，不做调参或改 bar。
- 每个场景使用**独立的** `PSSRuntime`：对象 id（`obj00`…）是场景内局部的，共享 runtime 会互相覆盖实体并混合关系。跨场景共享需要给 id 加 scene 前缀，列为开放项，不在 S4 范围内。

---

## 6. 边界

- **不是真实 RGB-D。** 观测是已知内参相机对 CLEVR `3d_coords` 的虚拟多视角投影 + 高斯噪声；没有检测器、分割器、标定误差、遮挡或错误关联。结论只覆盖"三维状态 + 三维查询能否接入 runtime"。
- 物体用球形半径近似（small 0.35 / large 0.70），`contact` / `collision` 是包围球口径，不是网格碰撞。
- 关联默认已知；本阶段不测试错误关联。
- `contact` / `collision` 的 confidence 下限（0.05）和 penetration 归一化是启发式，未校准。
- **1D `bayes_predicate` 仍是 fallback。** `RelationFactLayer` / `PSSRuntime` 的 `axis_mode` 默认仍是 `"off"`；三维场景接入是显式调用，不改变任何既有默认路径，也没有修改 runtime 语义或 state schema。
- 计划 §10 提到的 `scene_version` / `pose_covariance` 字段属于兼容性变更，本次**未**新增；协方差放在 `Entity.attributes`，来源放在 `provenance`。
- `UncertaintyService` 尚未消费场景关系 belief；本阶段只完成实体、关系、审计写入。

---

## 7. 新建/修改文件

- 新建 `pss/scene/pss_bridge.py`
- 新建 `pss/eval/scripts/run_scene_runtime.py`
- 新建 `tests/test_scene_runtime.py`
- 新建 `docs/experiments/PSS-Scene-Runtime.md`（本文）
- 未修改任何 runtime / scene 既有模块。
