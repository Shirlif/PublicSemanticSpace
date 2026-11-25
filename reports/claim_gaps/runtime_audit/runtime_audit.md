# Row 7 — runtime permission profiles

> generated: 2026-11-25 16:24:11 UTC

| profile | read | ingest | write_facts | plan | act | rollback | denied audit entries |
|---|---|---|---|---|---|---|---:|
| read_only | yes | no | no | no | no | no | 5 |
| service | yes | yes | no | no | no | no | 4 |
| operator | yes | yes | yes | yes | yes | no | 1 |
| admin | yes | yes | yes | yes | yes | yes | 0 |

## Verdict

- `read_only` can read but cannot ingest, write, plan, act or roll back.
- `service` can ingest perception but cannot write facts or plan.
- `operator` can write facts, plan and act; rollback requires `admin`.
- `admin` can do everything, including rollback.
- Every denied action is recorded as a `permission_denied` audit entry.
- This closes the permission part of row 7; versions/rollback/audit were
  already present from Phase 16.
