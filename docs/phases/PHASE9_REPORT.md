# Phase 9 报告：学习权限、安全护栏、降级与审计

> 日期：2026-11-20
> 对应计划：`公共语义空间-落地执行计划-v1.md` → Phase 9
> 状态：**L1→审核→L2 OTA 闭环跑通；Ed25519 包/人工门签名；L3/L4 实际 payload；阈值标定；审计含关系链+护栏+计划回放；混沌序列通过**
> 环境：torch 2.10.0+cu126 / Python 3.13.7

---

## 1. 交付物

| 交付 | 路径 | 状态 |
|---|---|---|
| L1 实例记忆 + 脱敏提案 | `pss/learning/l1_store/` | ✅ |
| L2 原型包（hash、增量、原子激活、回滚） | `pss/learning/l2_package/` | ✅ |
| L3/L4 版本注册表 | `pss/learning/ota/` | ✅ |
| 审核流水线（聚合/预审/回归/人工占位） | `pss/learning/review/` | ✅ |
| 三层护栏 | `pss/safety/__init__.py` | ✅ |
| 四级降级 FSM | `pss/safety/degradation.py` | ✅ |
| 审计快照 + 回放 | `pss/safety/audit.py` | ✅ |
| Phase 9 CLI | `pss/learning/scripts/run_phase9.py` | ✅ |
| 指标与报告 | `checkpoints/phase9/` | ✅ |
| 测试 | `tests/test_learning.py`、`test_safety.py`、`test_review.py` | ✅ 93 passed（全仓） |

---

## 2. 学习闭环（L1 → 审核 → L2 OTA）

以未见类别 `cloth` 为例：

| 项 | 结果 |
|---|---|
| L1 记录 | 28 条（只写本地） |
| 提案 | 28 样本，`scrubbed=True` |
| 预审（Phase 5 阈值） | **reject**（类间距离低） |
| 回归门 | **new_acc = 1.000**（20 条留出） |
| 审核结论 | **accept** |
| 激活版本 | `v1.1`，类别含 `cloth` |
| 回滚 | `v1.0`，`cloth` 消失 |
| 损坏包 | **hash 校验拒绝** |

> 这是 Phase 5 结论的延续：预审的距离阈值是占位值，会误杀可用类别；回归集准确率才是当前的决定性门。阈值标定仍是未解项（9.10）。

### 2.1 预审阈值标定与人工门（2026-11-22）

**预审阈值**：不再用固定 0.25。`review_proposal` 默认按模态计算

```text
threshold_m = max(0.01, 0.15 * min_nearest_neighbor_distance_m)
```

当前 bank 上：

| 模态 | 阈值 | cloth 最小类间距离 | 预审 |
|---|---:|---:|---|
| vision | 0.0874 | 0.2183 | pass |
| tactile | 0.0100 | 0.0759 | pass |
| geometry | 0.0158 | 0.0194 | pass |

预审从 **reject 变为 accept**。0.15 是"重复检测"因子，不是新类检测门；真正的验收仍是回归集准确率（`new_acc=1.000`）。

**人工门**：新增文件化审批记录。

- `write_human_decision(path, proposal_id, reviewer, decision)` 写 JSON；
- `review_proposal(..., human_decision=..., require_human=True)` 读取记录；
- accept 记录允许进入 L2 OTA；reject 记录返回 `human_rejected`，`accepted=False`。

本次演示：accept 记录 -> `accepted=True`；reject 记录 -> `veto_accepted=False`，原因 `human_rejected`。人工记录现在用 **Ed25519 签名**：`write_human_decision(..., private_key=...)` / `load_human_decision(..., trusted_public_key=...)`，错误公钥和篡改记录都会被拒绝。

局限：

- 0.15 因子只在当前 bank 和 cloth 候选上验证，尚未做多类别扫描。
- 人工门是签名文件，没有 UI、身份提供方或硬件 token；仍是单人或模拟复核。
- 预审仍是诊断项；回归门才是当前决策门。

### 2.2 Ed25519 包签名与 L3/L4 实际 payload（2026-11-22）

**L2 签名**：

- 后端生成 Ed25519 keypair，保存在 `checkpoints/phase9/keys/`。
- `PrototypePackage.save(..., private_key=...)` 对 manifest（不含签名/公钥字段）签名；`load(..., trusted_public_key=...)` 用固定公钥验证。
- 结果：`signature_verified=True`，错误公钥拒绝，未签名包拒绝。
- sha256 仍负责内容完整性；Ed25519 负责发布者身份。

**L3/L4 payload**：

- `AdapterRegistry` / `RuleRegistry` 现在有存储根：`checkpoints/phase9/l3/`、`checkpoints/phase9/l4/`。
- 安装时写入 `payload.json`、`files/*` 和 `manifest.json`；激活时校验文件 hash。
- 篡改 `a2/files/vision.pt` 后 `verify` 返回 False，激活被拒绝。
- payload 仍是小配置（adapter 描述、规则 JSON），不是训练权重。

局限：

- 私钥存本地 PEM，没有 HSM/KMS 或密钥轮换。
- 公钥固定是调用方传参，没有证书链/信任根。

---

## 3. L3 / L4 权限

| 层 | 激活 | 回滚 |
|---|---|---|
| L3 适配器 | `a2` | `a1` |
| L4 认知规则 | `r2` | `r1` |

- 端侧只读激活版本；安装、激活、回滚都在后台/离线侧。
- 版本包带 payload hash；hash 不匹配拒绝激活。

---

## 4. 三层护栏

| 场景 | allowed | hard veto | score | 原因 |
|---|---|---|---|---|
| 人距 0.2 m 且 target_is_human | ❌ | ✅ | 0 | `human_in_zone` |
| 正常推杯子（force 2） | ✅ | ❌ | 0.440 | — |
| safety_coeff = 0 | ❌ | ❌ | -0.100 | `safety_coeff_zero` |
| human_comfort 基础 | ✅ | ❌ | 0.665 | — |
| human_comfort + 夜间模式 | ✅ | ❌ | **0.755** | 场景原型提高舒适权重 |
| LLM 来源的危险动作 | ❌ | ✅ | 0 | `human_in_zone` |

硬安全层独立于价值打分：即使动作来自 LLM，也直接否决。

**阈值标定（2026-11-22）**：`calibrate_min_score()` 用 200 个合成 safe/unsafe 候选扫描 `min_score`，得到 **0.189**，balanced accuracy **1.000**（safe 均值 0.536，unsafe 均值 -0.183）。这是合成标签，不是人类安全标注；硬安全黑名单仍独立生效。

---

## 5. 四级降级（debounce=2）

| 步骤 | 信号 | 等级 | 提案 | 探索 | 运动 | 人工 |
|---|---|---|---|---|---|---|
| 0 | 正常 | normal | ✅ | ✅ | ✅ | ❌ |
| 1 | 投影漂移（第 1 帧） | normal | ✅ | ✅ | ✅ | ❌ |
| 2 | 投影漂移（第 2 帧） | **warning** | ✅ | ✅ | ✅ | ❌ |
| 3 | 相机失效（第 1 帧） | warning | ✅ | ✅ | ✅ | ❌ |
| 4 | 相机失效（第 2 帧） | **degraded** | ❌ | ❌ | ✅ | ❌ |
| 5 | 过热（第 1 帧） | degraded | ❌ | ❌ | ✅ | ❌ |
| 6 | 过热（第 2 帧） | **emergency** | ❌ | ❌ | ❌ | ✅ |
| 7 | 恢复（第 1 帧） | emergency | ❌ | ❌ | ❌ | ✅ |
| 8 | 恢复（第 2 帧） | **normal** | ✅ | ✅ | ✅ | ❌ |
| 9 | 离线 | normal | ❌ | ✅ | ✅ | ❌ |

- Warning：继续运行，允许提案与探索；
- Degraded：暂停提案与探索，保留安全运动；
- Emergency：停止运动，要求人工；
- 离线：本地感知/原型/安全继续，仅禁止提案上传。

### 5.1 混沌故障序列

`tests/test_safety.py` 新增 120 步混合故障序列：正常 / 投影漂移 / 相机失效 / 过热循环注入。

- 全程无崩溃，健康等级始终在 normal/warning/degraded/emergency 四档内。
- 过热阶段进入 emergency，运动禁止且要求人工。
- 故障撤销后 3 次正常检查恢复到 normal。
- 离线状态下提案始终禁止。

---

## 6. 审计

- 每次运行追加 1 条决策快照，回放校验通过（本次日志 10 条）。
- 候选打分：`task` 0.440、`comfort` 0.680 → 决策 `comfort`。
- `AuditLog.verify` 现在重算四层：价值打分、护栏 verdict（hard_veto / allowed）、计划执行（laundry 12 步）、关系链（2 个实体 → 4 条关系事实）。
- 篡改决策字段、护栏字段、计划动作或关系事实后，`AuditLog.verify` 返回 False（测试覆盖）。

快照包含：观测、实体、情感、动机、候选、打分明细、护栏结果与输入、计划、关系链、健康等级、价值向量。

仍未回放：原始像素 → 实体的感知前端。审计覆盖"实体状态 → 关系 → 计划 → 决策 + 护栏"，不覆盖视觉/触觉专家推理。

---

## 7. 冻结的决策

| # | 决策 |
|---|---|
| 1 | L1 只写本地；提案必须脱敏 |
| 2 | L2 包用 sha256 + Ed25519；激活原子替换；回滚保留历史 |
| 3 | L3/L4 端侧只读；安装/激活/回滚在后台；实际 payload 文件校验 hash |
| 4 | 预审阈值按模态标定为 `0.15 * min_nn`；回归集准确率仍是当前决策门 |
| 5 | 人工门是文件化审批记录；accept/reject 都进入审计 |
| 6 | 硬安全独立、并行、不可绕过 |
| 7 | 降级用 debounce；不同等级分别控制提案/探索/运动 |
| 8 | 审计快照可回放；价值打分、护栏 verdict、计划执行都能从快照复算 |

---

## 8. 已知边界

| 边界 | 说明 |
|---|---|
| 签名密钥 | Ed25519 已接；私钥本地 PEM，没有 HSM/KMS 或轮换 |
| L3/L4 payload | 已安装实际文件并校验 hash；仍是配置，不是训练权重 |
| 阈值 | 预审按模态标定，护栏 `min_score=0.189` 用合成数据标定；价值向量权重仍是设计值 |
| 人工门 | Ed25519 签名记录；没有 UI、身份提供方或双人复核 |
| 故障 | FSM + 120 步混沌序列；不是真实硬件故障注入 |
| 审计回放 | 覆盖实体→关系→计划→决策+护栏；不覆盖原始像素→实体的感知前端 |
| LLM 注入是规则模拟 | 未接真实 LLM |

---

## 9. 下一步：Phase 10（预注册评测与基线对比）

| 工作项 | 说明 |
|---|---|
| 预注册指标 | 绑定误差、ID 保持、关系 F1、状态误差、属性混淆、焦点召回 |
| 基线 B1 | 强 VLM + 同一套技能 |
| 基线 B2 | 行为克隆策略 |
| 消融 | 无槽 / 跨模态相加 / 无场景图 / 随机投影 |
| 杀死标准 | K1–K4，先写后跑 |

Phase 9 的审计与降级数据可以直接进入评测集。
