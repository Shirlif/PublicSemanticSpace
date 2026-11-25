# P7 / W13–W14 进度 — official 跑测准备

> 开始：2026-11-25
> 状态：A4 预注册完成；A9/A10 机制在 dev 上跑通；T40/T41 需要 3090 Ti，本机无法执行。

## T39 A4 预注册：完成

- 文件：`eval/prereg_isaac_v1.md`
- 冻结内容：假设、套件/指标/阈值、seeds、降采样、设备与版本、停止规则、分析口径、归档清单。
- 明确声明：
  - 主结论只来自 `frozen_real`；
  - oracle 只作标定；
  - S6 oracle 记忆与 S7 检测器校准都是**有限范围证据**；
  - PSS 置信度/弃权（abstain、conflict F1）为探索性。

## T40 环境对齐：机制完成，official 待执行

- `scripts/check_environment.py --device dev|official --freeze <freeze>`：
  检查 GPU、驱动、PSS/Isaac Python、Isaac Sim 版本；official 额外强制 3090 Ti。
- dev 实测：全部 PASS。
- official 待 3090 Ti 上执行：
  ```powershell
  python scripts\check_environment.py --device official --freeze freeze\p7-official.json
  ```
  在 3090 上先写 `freeze/p7-official.json`（Isaac/PhysX/驱动/GPU），再跑此命令。

## T41 official 全量：**未执行（硬件不在本机）**

- 4070 上不能产出 official 数字；本报告不编造。
- 待 3090 Ti 执行的命令矩阵：
  ```text
  S1/S2/S3/S6: 5 seeds, 20 objects, device official
  S4/S5:       3 seeds, full 9 variants / 10 fault classes
  detect:      frozen_real over rendered RGB-D
  replay:      full + 8 ablations
  score:       pss score --device official
  ```
- 需要先把 S1/S2/S3/S6 正式 spec 切到相机轨迹（ADR-001），再开 official。

## T42 A9 复算：dev 机制通过

- `pss verify-recompute --tag dev-p7`
- 结果：`verify/dev-p7.json`，所有 dev 套件 `ok=true`，`mismatches=[]`。
- 覆盖：`s1_occlusion_20`（5 seeds）、`s1_occlusion_{oracle,noisy,real}`、`s2_twins*`。
- 说明：这是**同机复算**；跨机 4070←3090 逐位一致要在 official 跑完后做。

## T43 A10 归档：dev 机制通过

- `pss report --tag dev-p7` → `reports/dev-p7/`
- 包含：`freeze.json`、`specs/`、`prereg.md`、`scorecards/`、
  `bundles.index.parquet`、`verify.json`、`overrides.jsonl`、`notes.md`。
- 说明：这是 dev 归档演练；official tag 的 8 件套在 3090 跑完后用同一命令生成。

## 已知偏差（写进正式报告）

1. official 数字尚不存在；当前所有表都是 dev。
2. S6 用 oracle 检测；S7 是检测器校准，不是 PSS 校准。
3. S2 abstain precision、conflict F1 未实现。
4. S4/S5 是脚本基线，PSS 尚未接入执行回路。
5. 相机轨迹只在 `s1_camera_motion` 验证；S1/S2/S3/S6 正式 spec 尚未切换。

## 下一步（在 3090 Ti 上）

1. 生成 `freeze/p7-official.json`，跑 `check_environment --device official`。
2. 把 S1/S2/S3/S6 spec 切到 `cameras[].motion`，重跑 dev 冒烟。
3. official 全量生成 → detect → replay → score。
4. `pss verify-recompute --tag isaac-p7-v1`（跨机复算用 4070）。
5. `pss report --tag isaac-p7-v1`，检查 8 件套。
