# PSS U5b：审计日志 append-only 修复

> 日期：2026-11-28
> 上游：`docs/experiments/PSS-U5-runtime-completeness.md` §4.3
> 变更：`pss/runtime/state.py` 的 `CognitiveState.rollback`
> 回归：`tests/test_runtime_audit_append_only.py`
> 复测结果：`eval/results/upper_bound/u5_fixed/audit_completeness.json`

---

## 0. 一句话

**修复前**：`rollback` 把 `audit` 列表恢复到目标快照，回滚点之后的 `permission_denied` 会从最终审计中消失，primary completeness **0.800（4/5）**。**修复后**：`rollback` 只恢复状态字段，不再截断 audit；primary completeness **1.000（5/5）**，`rollback_exactness=True`，`permission_violations=0`，`rollback_truncated_audit=false`。

---

## 1. 问题

`CognitiveState.rollback(version)` 原实现：

```python
payload = copy.deepcopy(self.snapshots[version])
self.entities = payload["entities"]
...
self.audit = payload["audit"]      # ← 这里把审计也回滚了
self.version = payload["version"]
self.audit.append(AuditEntry(... "rollback" ...))
```

U5 的 primary 场景：写 fact（v1）→ 被拒绝（v2，产生 `permission_denied`）→ rollback 到 v1。回滚后 `audit` 被替换成 v1 的快照，`permission_denied` 条目消失。结果是：

- 拒绝动作在发生时**确实**被审计（`denials_audited_at_time=1`）；
- 但最终审计只保留 4/5 个期望事件（`audit_completeness=0.800`）；
- 对部署而言，这等价于“拒绝动作可以被后续 rollback 从审计中抹掉”，审计不是 tamper-evident。

---

## 2. 修复

`pss/runtime/state.py`：

```python
# Audit is append-only: rollback restores state fields but deliberately
# does not truncate the audit log.  Restoring payload["audit"] would let a
# later rollback erase a permission_denied entry.
self.version = payload["version"]
self.audit.append(
    AuditEntry(self.version, "rollback", f"v{version}", None, None, "runtime", f"to v{version}")
)
self.snapshots.append(self._snapshot())
```

语义：

- 状态字段（entities / relations / facts / goals / plans / uncertainty）仍精确回滚到目标快照；
- audit 只在末尾追加一条 `rollback`，历史条目全部保留；
- 后续 commit 仍正常工作，不会复活被回滚的 fact。

---

## 3. 回归结果

`python -m pytest tests/test_runtime_audit_append_only.py tests/test_runtime.py tests/test_runtime_permissions.py -q` → **12 passed**。

复测 U5 场景（`run_upper_bound_u5 --out eval/results/upper_bound/u5_fixed/audit_completeness.json`）：

| arm | rollback target | completeness | cumulative | rollback exact | violations | truncated |
|---|---|---:|---:|---|---:|---|
| primary | v1（pre-denial） | **1.000（5/5）** | 1.000 | True（6/6） | 0 | **false** |
| control | v2（latest） | 1.000（5/5） | 1.000 | True（6/6） | 0 | false |

Primary 的最终 audit actions：

```
["assert_fact", "permission_denied", "rollback", "set_goal", "plan", "act"]
```

`permission_denied` 被保留，`denied_entries_retained_after_rollback=1`。

---

## 4. 残余问题

- **audit 的 `version` 字段不是单调的**：primary 的 `final_audit_versions = [1, 2, 1, 2, 3, 4]`。这是因为 `AuditEntry.version` 记录的是**状态版本**，而 rollback 会把状态版本退回去。如果下游需要单调日志序号，应新增独立的 `audit_seq` 计数器，而不是复用状态版本。
- 本轮没有改 `AuditEntry` schema；加 `audit_seq` 会改变审计 JSON 的字段，属于兼容性变更，需要单独评估。
- 修复只覆盖 `CognitiveState.rollback`；`VersionedKnowledgeBase`、`TruthMaintenanceSystem` 的 rollback 语义未变。

---

## 5. 可以 / 不可以声称

**可以声称**

1. `rollback` 后 `permission_denied` 审计条目不再丢失；primary completeness 从 0.800 提升到 1.000。
2. 状态字段的 rollback exactness 不变（6/6 字段相等）。
3. 权限矩阵与 `permission_violations=0` 不变。

**不可以声称**

1. 不可以声称审计日志已经 tamper-evident；本轮只修了“rollback 不截断”，没有做哈希链/签名。
2. 不可以声称审计版本号单调；`version` 仍可能回退。
3. 不可以把单 episode 的修复外推到多 episode 生产负载或对抗性删除。

---

## 6. 产物

```
pss/runtime/state.py                          # rollback 不再恢复 audit
tests/test_runtime_audit_append_only.py       # 2 个回归
eval/results/upper_bound/u5_fixed/audit_completeness.json
```

复现：

```bash
python -m pytest tests/test_runtime_audit_append_only.py -q
python -u -m pss.eval.scripts.run_upper_bound_u5 \
  --out eval/results/upper_bound/u5_fixed/audit_completeness.json
```

---

## 7. 一句话

**rollback 恢复状态，但不恢复审计；primary completeness 从 0.800 到 1.000。** 这修的是“审计可被回滚抹掉”的语义洞，不是完整的 tamper-evidence。
