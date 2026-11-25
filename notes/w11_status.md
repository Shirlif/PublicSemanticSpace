# P6 / W11–W12 进度 — 记忆与成本（S6/S7/S8）

> 开始：2026-11-25
> 状态：T35–T38 完成（dev / oracle 为主）。

## T35 S6 长程记忆

- 场景：3 个物体、5 个连续子 episode、每段 3 s；子 episode 之间物理改位
  （`specs/s6_memory.yaml` 的 `task.session.relocations`）。
- `scripts/run_s6.py`：一次启动、一个 stage、连续 5 段；每段写一个 bundle
  （`ep0..ep4`），共享同一 session。
- `pss/detect/session_replay.py`：**一个 PSS runtime 跨子 episode 连续 ingest**，
  状态不重置；GT 只用于评分侧对齐。

## T36 记忆指标（`pss/metrics/memory.py`）

| 指标 | 值 |
|---|---|
| edited_accuracy | **1.0000** |
| old_forgetting | **1.0000**（forgetting_rate 0.0） |
| id_retention | **1.0000** |
| rollback_exactness | **1.0000** |
| rollback_changed_state | true（说明编辑确实改了状态，再逐位还原） |

- 表：`scorecards/S6.md` / `S6.json`。
- 说明：S6 当前用 oracle 检测；真实检测 + 图像回访是下一步。

## T37 S7 校准

- `pss/metrics/calibration.py`：检测分数校准（按 3D 位置匹配 GT，判断检测是否正确）。
- 在 `s1_occlusion_real`（GroundingDINO + DINOv2）上：

| 指标 | 值 |
|---|---|
| accuracy | 0.8087 |
| mean confidence | 0.6370 |
| Brier | 0.1438 |
| ECE | 0.2011 |
| selective acc @50% | 0.9391 |
| selective acc @80% | 0.9103 |

- 表：`scorecards/S7.md` / `S7.json`。
- **缺口**：这是**检测器**分数校准，不是 PSS 置信度校准；`conflict F1` 需要 PSS
  uncertainty trace，当前 replay 关闭了 uncertainty，故为 null。

## T38 S8 成本

- `pss/metrics/cost.py` 读 `pss_trace.parquet`：

| 指标 | 值 |
|---|---|
| calls | 100 |
| latency p50 | 0.144 ms |
| latency p95 | 0.187 ms |
| entities mean | 4.0 |
| facts mean | 0.0 |
| footprint | 未测量 |
| tokens | 0（无 LLM） |

- 表：`scorecards/S8.md` / `S8.json`。
- **缺口**：内存 footprint 未打点；token 为 0 是因为 replay 没有语言层。

## 复现

```powershell
D:\isaac\env\Scripts\python.exe scripts\run_s6.py --force
python scripts\score_s6.py
python scripts\score_s7.py
python scripts\score_s8.py
```

## 下一步

1. S6 用 `frozen_real` + 关键帧回访，而不是 oracle；这会直接检验“物理改位后旧事实是否
   被覆盖、回滚是否逐字段还原”。
2. S7 打开 `PSSRuntime` 的 uncertainty，产出 PSS 置信度/弃权，再算 conflict F1。
3. S8 用 `psutil`/`tracemalloc` 记录 footprint；语言层接入后统计 token。
