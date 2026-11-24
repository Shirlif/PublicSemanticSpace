# Phase 24 — Gap closure 7: runtime permission profiles

> Date: 2026-11-24
> Scope: `PSS-认知层架构定位.md` §5, row 7 (audit)
> Data: deterministic runtime actions (synthetic)
> Result artifacts: `eval/results/runtime_audit/`

---

## 1. What changed

Row 7 target: **versions / rollback / permissions**. Versions, snapshots,
rollback and audit were already in `CognitiveState` (Phase 16). This phase adds
the missing authority layer.

| Component | Change |
|---|---|
| `PermissionProfile` | `pss/runtime/permissions.py`: read / ingest / write_facts / plan / act / rollback |
| profiles | `read_only`, `service`, `operator`, `admin` |
| enforcement | `PSSRuntime._require()` in ingest / assert_fact / set_goal / plan / act / rollback / explain / summary |
| audit | denied actions append a `permission_denied` entry before raising |
| summary | `permissions` record included |
| default | `admin` for backward compatibility |

---

## 2. Permission matrix

| profile | read | ingest | write facts | plan | act | rollback | denied audit entries |
|---|---:|---:|---:|---:|---:|---:|---:|
| `read_only` | yes | no | no | no | no | no | 5 |
| `service` | yes | yes | no | no | no | no | 4 |
| `operator` | yes | yes | yes | yes | yes | no | 1 |
| `admin` | yes | yes | yes | yes | yes | yes | 0 |

- `read_only` can inspect state but cannot change it.
- `service` can ingest perception but cannot write facts or plan.
- `operator` can write facts, plan and act; rollback is reserved for `admin`.
- `admin` has full authority.
- Every denied action is recorded as a `permission_denied` audit entry; the
  exception then propagates.

---

## 3. Findings

1. **The authority layer is explicit and auditable.** Permissions are a small
   dataclass, not scattered conditionals; `summary.permissions` reports the
   active profile.
2. **Rollback is admin-only.** Operator workflows can write/plan/act, but
   undoing history requires elevated authority.
3. **Denials are visible.** A denied write/plan/rollback leaves an audit trail,
   so a failed action is attributable to the profile, not a silent no-op.
4. **Backward compatibility is preserved.** The default profile is `admin`;
   existing experiments and tests are unchanged.
5. **This closes the permission part of row 7**, not the whole audit story:
   authentication, per-resource ACLs and signed audit chains remain future
   work.

---

## 4. Boundaries

- Profiles are coarse (global, not per-object/fact).
- The caller chooses the profile; there is no authenticated identity binding.
- No cryptographic signature on the runtime audit log (PackageManager has
  signing, but the runtime audit is in-process).
- Single-process simulation; no concurrent writers.
- Synthetic actions, no real robot.

---

## 5. Gap-table status

| 层 | 状态 |
|---|---|
| 编码器 | 部分（DINOv2 默认，CLIP/VLM 缺权重） |
| 投影 | 部分（DINOv2 默认路径已校准；共享头/开放集/fallback/CLIP-VLM 未解决） |
| 实体绑定 | 部分（persistent slots + object attention 选项，benchmark 饱和） |
| 关系 | 部分（typed heads 默认 + 可学习/版本化阈值；contact 特征受限） |
| 规划 | 达标（learned heuristic，expansions 减半，wall-clock 小域变慢） |
| 语言 | 部分（真实 1.5B decoder，6/7，citations 不足） |
| 审计 | 达标（版本/回滚/权限/拒绝审计） |

Reproduce:

```bash
python -m pss.eval.scripts.run_runtime_audit
```
