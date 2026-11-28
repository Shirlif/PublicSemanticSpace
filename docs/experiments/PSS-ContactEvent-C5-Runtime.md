# PSS ContactEvent 接入 Runtime（C5）

- 日期：2026-11-29
- 阶段：C5（PSS 集成，见 `docs/plans/PSS-ContactEvent-计划.md` §2、§4、§8-C5）
- 范围：`ContactEvent` → `Relation(predicate="contact", origin="contact_event")` → `CognitiveState` / `RelationFactLayer` / `UncertaintyService` / audit
- 结论：链路打通、可审计、可回滚；**opt-in**，不把 ContactEvent 设为 runtime 默认接触层，未修改 `pss/runtime/*` 的任何语义或 state schema。

---

## 1. 接入链路

```
MuJoCo 场景（drop / collide / near_miss）
  → run_channel(..., channel="force")            # pss/events/bench_api.py
  → ContactEvent(status / confidence / evidence / provenance)
  → apply_events(runtime, events)                # pss/events/layer.py（本次）
        ├─ event_to_relation(event)
        │     └─ Relation(predicate="contact", origin="contact_event",
        │                 probability=confidence, relation_type="contact_event")
        │         + relation.attributes / relation.metadata（evidence、provenance、时序）
        ├─ CognitiveState.add_relation(...)      # 关系写入
        ├─ RelationFactLayer._remember(...)      # 关系层记录缓存（镜像）
        ├─ UncertaintyService.assess(...)        # 低置信度 → review / abstain
        └─ contact_event_ingest audit            # 事件数、状态、来源、决策
```

实现文件：`pss/events/layer.py`。

- `event_to_relation(event)` → `Relation`
- `apply_events(runtime, events, commit=True, ...)` → 摘要 dict
- `contact_relations(state)` / `relation_metadata(relation)` → 查询辅助

`apply_events` 只调用 runtime 已有的 `state.add_relation`、`_commit`（存在时）、`uncertainty.assess`、`permissions`，**没有修改** `pss/runtime/state.py`、`layers.py`、`services.py` 或 `runtime.py`。

---

## 2. 接入字段

### 2.1 Relation

| 字段 | 内容 |
|---|---|
| `source` / `target` | `ContactEvent.subject` / `ContactEvent.object` |
| `predicate` | 固定 `"contact"` |
| `relation_type` | `"contact_event"`（与符号几何的 `"symbolic"`、场景模型的 `"scene_geometry"` 区分） |
| `origin` | `"contact_event"` |
| `probability` | `event.confidence`，裁到 `[0, 1]`；**不是校准概率**（C2 校准由调用方负责） |
| `frame` | `"world"` |
| `attributes` / `metadata` | 事件级证据（见下）；实例属性，不是 state schema 字段 |

`attributes` / `metadata` 是同一个 dict：

| key | 内容 |
|---|---|
| `event_id` | `ContactEvent.event_id` |
| `status` | `predicted` / `confirmed` / `contradicted` / `expired` / `resolved` |
| `onset` / `offset` / `duration` | 事件时序（秒） |
| `location` | 接触点，数组转 list |
| `intensity` | 峰值力 / 强度 |
| `event_version` | `ContactEvent.version` |
| `evidence` | `[{source, timestamp, score, intensity, location, reliability, metadata}, ...]` |
| `provenance` | `ContactEvent.provenance`（demo 中为 `{"channel": "force"}`） |

**为什么不新增 schema：** `Relation` 只有 7 个字段。为保留证据来源，`event_to_relation` 在返回的 `Relation` 实例上附加 `attributes` / `metadata`，`Relation.to_dict()` 仍只输出原来的 7 个字段，runtime 的序列化视图不变。读取用 `relation_metadata(relation)`。

**为什么不走 `RelationFactLayer.add_symbolic`：** 该方法只把 `RelationRecord.source in {"bayes_geometry","hard_geometry"}` 识别为几何来源，其余一律写成 `origin="symbolic"`，无法保留 `contact_event` 来源。因此关系直接写入 `CognitiveState.add_relation`，同时把边镜像到 `RelationFactLayer` 的记录缓存，供 `records()` / 后续推理查询。

### 2.2 置信度分档与 UncertaintyService

| 条件 | decision | 是否写入 state | 处理 |
|---|---|---|---|
| `status ∈ {contradicted, expired}` | `rejected` | 否 | 只进 audit |
| `confidence < abstain_threshold`（默认 0.25） | `abstain` | 否 | 只进 audit；`recommendation="abstain"` |
| `confidence < review_threshold`（默认取 runtime 的 `UncertaintyService.relation_threshold`，即 0.5） | `review` | 是，概率为原 confidence | 可查询但标记待审；`recommendation="review"` |
| 其余 | `accept` | 是 | 正常接触边 |

`apply_events` 在写入后调用 `UncertaintyService.assess(state, fact_conflicts=runtime.relations.conflicts())`，保留 runtime 既有的事实冲突 / 缺失模态判断，然后叠加接触事件信号：

- 有 review：追加 flag `contact_event_review`；若服务原本返回 `accept`，提升为 `review`；
- 有 abstain：追加 flag `contact_event_abstain`；`recommendation` 直接置为 `abstain`；
- `uncertainty_enabled=False`：不调用服务，摘要中 `recommendation="disabled"`。

### 2.3 权限

`apply_events` 在任何写入前调用 runtime 的 `_require("ingest", ...)`（存在时）。`read_only` profile 会抛 `PermissionDenied`，runtime 自身会先写一条 `permission_denied` audit；接触关系不会被写入。

---

## 3. 审计

每次非空 `apply_events(commit=True)` 追加一条：

| 字段 | 内容 |
|---|---|
| `action` | `"contact_event_ingest"` |
| `target` | `"contact_events:<n>"` |
| `actor` | `"contact_event"`（可改） |
| `reason` | `"contact events -> runtime"`（可改） |
| `before` | `{version, relations, contact_relations}`（写入前计数） |
| `after.events` | 事件数 |
| `after.written` | 实际写入的关系数 |
| `after.decisions` | `{accept, review, abstain, rejected}` 直方图 |
| `after.statuses` | 事件状态直方图 |
| `after.sources` | 事件来源直方图（provenance.source / channel，缺省时用 evidence 通道并集） |
| `after.evidence_sources` | 所有 evidence 通道直方图 |
| `after.recommendation` | 叠加后的 UncertaintyService 建议 |
| `after.before_version` | 写入前版本 |
| `after.entries` | 逐事件 `{event_id, subject, object, confidence, status, decision, source, evidence_sources}` |

`commit=False` 只写关系、不写 audit、不改 version（与 S4 `apply_scene_to_runtime` 的 `commit` 语义一致）；该模式**不具备回滚保护**，仅用于预览/测试。

---

## 4. 回滚

- `commit=True` 时通过 runtime 的 `_commit`（存在时）提交；它同时维护 `_relation_snapshots`，因此 `runtime.rollback(before_version)` 会同时恢复 `CognitiveState.relations` 和关系层的 fact snapshot。
- audit 是 append-only：`state.rollback` 恢复状态字段但**不截断** audit，`contact_event_ingest` 记录在回滚后仍存在，末尾追加 `rollback` 记录。
- `RelationFactLayer._records` 不在 layer snapshot 中（这是 runtime 既有行为）；回滚恢复的是 `CognitiveState.relations`，记录缓存可能保留已回滚的 contact 记录。查询以 state 为准。

实测（demo）：`before_v=1`（demo 种子实体已提交），写入后 `v=2`，`rollback(1)` 后 `contact_relations=0`、实体保留、audit 从 2 条变 3 条（新增 rollback 条目）。

---

## 5. 测试

```
python -m pytest tests/test_contact_event_runtime.py -q
.............                                                            [100%]
```

13 个测试全部通过：

1. `event_to_relation`：predicate / origin / probability / metadata / evidence / `to_dict` 字段不变。
2. 合成事件写入：relation、relation layer records、audit 字段、version。
3. 低置信度（0.3）→ `review`：写入低概率关系，UncertaintyService flag/recommendation。
4. 极低置信度（0.1）→ `abstain`：不写关系，audit 仍记录事件。
5. `contradicted` / `expired` → `rejected`，不写关系。
6. 混合 4 类事件：计数、写入数、recommendation 为 abstain。
7. rollback：保留已有 fact、移除 contact 关系、audit 不截断。
8. `commit=False`：写关系、version 0、无 audit。
9. 空事件列表：完全 no-op。
10. 非法阈值（abstain > review）：写入前抛 `ValueError`。
11. `read_only` 权限：抛 `PermissionDenied`，不写 contact 关系，记录 `permission_denied`。
12. `audit=False`：写关系、version 0、无 audit、`audit_action=None`。
13. runtime 默认不变：无 contact 关系、无记录、`assert_fact` 仍工作、`axis_mode="off"`。

旧测试回归：

```
python -m pytest tests/test_contact_event.py tests/test_runtime.py \
  tests/test_runtime_semantics_lock.py tests/test_scene_runtime.py -q
.............................                                            [100%]   # 29 passed

python -m pytest tests/test_runtime_audit_append_only.py \
  tests/test_runtime_permissions.py tests/test_runtime_phase17.py \
  tests/test_runtime_rules.py -q
..............                                                           [100%]   # 14 passed
```

合计 56 passed，无回归。

---

## 6. Demo 实跑

```
python -u -m pss.eval.scripts.run_contact_event_runtime
```

配置：`drop`，seed 0，`max_steps=400`、`capture_every=2`、force 阈值 0.5。输出摘要：

| 项 | 值 |
|---|---:|
| 场景帧数 | 400 |
| GT 事件 | 1 |
| force 通道事件 | 1 |
| 写入关系 | 1 |
| 决策 | `{accept: 1}` |
| 状态 | `{predicted: 1}` |
| 来源 | `{force: 1}` |
| recommendation | `accept` |
| 版本 | `1 → 2` |
| 实体（demo 种子） | 2 |
| contact 关系 | 1 |
| relation layer records | 1 |
| audit 条目 | 2（seed + ingest） |
| rollback 后 contact 关系 | 0 |

事件：

```
drop0-force-ball_00-box_00-0.2440-0  ball_00 -> box_00
  onset=0.2440  offset=0.28  confidence=0.954  intensity=20.877
  status=predicted  source=force
```

写入的关系：

```
ball_00 -contact-> box_00  p=0.954  origin=contact_event
type=contact_event  event=drop0-force-ball_00-box_00-0.2440-0
```

校验输出：

```
[apply] events=1 written=1 decisions={'accept': 1} ... before_v=1 after_v=2
[relations] contact=1 total=1
  relation_layer_records(predicate=contact)=1
[audit]
  - v1 seed_contact_participants target=drop0 actor=contact_event_demo ...
  - v2 contact_event_ingest target=contact_events:1 actor=contact_event
       events=1 statuses={'predicted': 1} sources={'force': 1} recommendation=accept
[rollback] to v1: contact_relations=0 entities=2 audit=3 (audit stays append-only)
  checks: provenance=ok origin=ok audit=ok rollback=ok
```

说明：

- demo 用场景第 0 帧位姿为参与者写入 **demo-only** 实体，并先 `commit("seed_contact_participants", ...)`；这不是 `apply_events` 的行为，只是为了展示 entities 摘要和可回滚版本。
- force 通道由 `run_channel` 产生的事件只有 `provenance={"channel": "force"}`，**没有逐通道 `ContactEvidence`**，所以 demo 中 `evidence=[]`；`event_to_relation` 的 evidence 映射在合成事件测试中已覆盖。GT 事件（`pss/events/gt.py`）会带 `source="mujoco_contact"` provenance。
- `status="predicted"` 是通道事件的原始状态；C5 不负责把它推进为 confirmed（那是融合/状态机的职责）。

---

## 7. 边界

- **不是 runtime 默认接触层。** 必须显式调用 `apply_events`；`PSSRuntime` 的默认关系路径、`RelationFactLayer` 规则、`axis_mode="off"` 均未改变。
- **confidence 不是校准概率。** 它来自 force 通道的归一化分数（`force/(force+f0)`）；ECE / 校准属于 C2，不在 C5。
- **关系是边级，不是事件级。** `CognitiveState.add_relation` 按 `(source, target, relation_type, predicate)` 去重；同一对物体的后续 contact 事件会替换当前 contact 边。事件历史保留在 `ContactEvent`、relation metadata（最新事件）和 audit `entries` 中。若需要同时保留同一 pair 的多个事件边，需要扩展 runtime schema 或把 `event_id` 编进 `relation_type`，本阶段未做。
- **`RelationFactLayer._records` 不随回滚恢复**（既有 runtime snapshot 行为）；回滚以 `CognitiveState.relations` 为准。
- **`commit=False` 不可回滚**：写关系但不产生 version/snapshot。
- **证据覆盖有限：** demo 只跑 force 通道；tactile/vision/audio 的事件融合属于 C2–C4，不在 C5。合成 `ContactEvidence` 的映射已测试，但不代表多模态性能。
- **模拟口径：** MuJoCo 图元是 proxy，force 近 oracle；不声称真实机器人/真实传感器结果。
- **权限：** 通过 `ingest` 权限门；`read_only` / 无 ingest 的 profile 被拒绝并审计。
- **state schema：** 未新增字段；evidence/provenance 只放在 `Relation` 实例属性和 audit payload 中。

---

## 8. 新建文件

- 新建 `pss/events/layer.py`
- 新建 `pss/eval/scripts/run_contact_event_runtime.py`
- 新建 `tests/test_contact_event_runtime.py`
- 新建 `docs/experiments/PSS-ContactEvent-C5-Runtime.md`（本文）
- 未修改 `pss/runtime/*`、C0–C4 文件或任何既有默认路径。
