# PSS × Isaac Sim 实施计划

> 日期：2026-11-25
> 依据：`PSS-IsaacSim-评测平台设计.md`（架构与锚点 A0–A10）、`PSS-架构评测指标.md`
> 原则：**每周只交付一个可验收物；验收不过不进下一周。**
> 用法：按 §2 的周计划推进，按 §7 的任务清单勾选；任何偏离写进 `notes/`。

> **ADR-001（2026-11-25，平台决策）**：评测目标是 PSS 的空间感知与空间记忆，不是人形运动控制。
> 感知/记忆主平台改为 **固定桌面 + 脚本化相机轨迹**；G1 仅做 `base_fixed` 上肢操作；
> 人形腿行走策略从关键路径移除，T28/T34 取消，改为相机轨迹与可选轮式附表。
> 详见 `notes/ADR-001-platform.md`。

---

## 0. 起点盘点

| 项 | 现状 | 影响 | 第几天处理 |
|---|---|---|---|
| Python | 3.13.7 | **Isaac Sim 捆绑 Python 为 3.10/3.11，与 3.13 不兼容** | D1 双环境 |
| torch | 2.10.0+cu126，CUDA 可用 | 检测器与编码器可用 | — |
| mujoco | 4.0.1 | 快速层可用 | — |
| git | **未初始化**（有 `.gitignore`，无 `.git`） | A0 无 commit 可冻结 | **D0** |
| Isaac Sim | 未安装 | P1 阻塞项 | D1 |
| `pss/spec/` | 不存在 | A1 阻塞项 | D2 |
| `pss/metrics/` | 不存在（指标散在 `pss/eval/*.py`） | A8 阻塞项 | W2 |
| 4070 Laptop | 8 GB | dev 机 | 已有 |
| 3090 Ti | 24 GB | official 机 | P4 起用 |

> 两条环境事实决定了架构：**Python 版本必须分裂**，所以 Isaac 进程与 PSS 进程天然分离，ZMQ 不是可选项而是必需项；
> **检测属于 PSS 侧环境**（torch 在 3.13 上可用），所以离线批处理方案正好落在正确的进程里。

---

## 1. 总路线

| 周 | 里程碑 | 锚点 | 可验收物 |
|---|---|---|---|
| D0 | 版本可冻结 | A0 | `git init` + 首次提交 + `freeze.json` |
| W1 | 一条 episode 跑通 | A1–A3 | `bundles/dev/.../seed0` 可读 |
| W2–W3 | 实体层竖切（S1） | A3 | S1 的 `oracle` 档 IDF1 表 |
| W4–W5 | 真实检测 + 孪生（S2） | A6 | 三档分数表 |
| W6–W7 | 关系与 OOD（S3） | A7 | per-predicate F1 + consistency |
| W8–W10 | 固定基座执行层（S4/S5） | A5 | 执行成功率 + 10 类 recovery（`base_fixed` 主表） |
| W11–W12 | 记忆与成本（S6/S7/S8） | A7 | rollback exactness = 1.0 |
| W13–W14 | official 跑测 | A4–A10 | 完整 scorecard + `reports/<tag>/` |

---

## 2. 第 1 周：从零到一条 episode

本周只做一件事：**让一条 10 秒的 episode 从仿真落到磁盘，并且能被读出来检查。**
不做检测、不做指标、不做机器人。

### D0（半天）：建仓与冻结

```powershell
git init
git add .gitignore
git commit -m "chore: init repo with gitignore"
```
（`data/`、`checkpoints/`、`_external/` 已在 `.gitignore` 内，不会进仓。）

产出 `freeze.json` 骨架——字段按 `PSS-IsaacSim-评测平台设计.md` §8.3 A0：

```json
{
  "tag": "p1-w1",
  "date": "2026-11-25",
  "code": {"commit": "<sha>"},
  "python": {"pss": "3.13.7", "isaac": "TBD"},
  "sim": {"isaac_sim": "TBD", "isaac_lab": "TBD", "physx": "TBD"},
  "driver": "616.64",
  "gpu": {"dev": "RTX 4070 Laptop 8GB", "official": "RTX 3090 Ti 24GB"},
  "detector": {"pss": "TBD", "weights_sha256": "TBD"}
}
```

### D1：双环境 + Isaac 安装

| 环境 | Python | 用途 | 位置 |
|---|---|---|---|
| `pss` | 3.13.7（现有） | PSS、检测器、指标 | 现有 |
| `isaac` | 3.10/3.11（按 Isaac Sim 版本要求） | 仅仿真 | conda env，D 盘 |

验收：`isaac` 环境能跑通 Isaac Sim 自带的最小示例场景，**headless 模式**（`--headless`），
显存占用记录到 `notes/env.md`。

> 8 GB 上第一次启动就可能 OOM。先记录基线占用，再谈并行度。

### D2：spec schema + 第一个 spec（A1）

新建 `pss/spec/`：

```
pss/spec/
  __init__.py
  schema.py          # dataclass + JSON Schema 导出
  loader.py          # load / hash
  cli.py             # pss spec lint
specs/
  s1_occlusion.yaml  # 第一个 spec
```

`s1_occlusion.yaml` 先最小化：4 个物体、固定相机、无机器人、无故障。
字段按设计文档 §3.1，但允许留空并在 `notes` 里标注“W2 补”。

验收：`python -m pss.spec.cli lint specs/s1_occlusion.yaml` 输出 spec hash，并追加到 `specs/registry.jsonl`。

### D3：USD 构建（A2）

最小场景：桌面 + 4 个立方体 + 1 个固定相机。不加机械臂、不加光照扰动。

产出 `pss/isaac/scene_builder.py`，把 spec 编译成 USD，缓存到 `~/.cache/pss_usd/<hash>/`。

验收：同一 spec 连编两次，prim 树 hash 一致。

### D4：跑通 1 条 episode（A3 雏形）

```
pss sim run specs/s1_occlusion.yaml --episodes 1 --seed 0 --device dev
```

产出 `bundles/dev/s1_occlusion/<hash8>/seed0/`，包含：

- `manifest.json`（含 `device_class=dev`、spec hash、GPU）
- `obs_frames.parquet`（10 Hz × 10 s = 100 行）
- `gt.parquet`（每物理帧 × 4 实例）
- `events.parquet`（空表也要有 schema）

### D5：读 bundle 验证三件事

```python
import pandas as pd
gt = pd.read_parquet(".../gt.parquet")
frames = pd.read_parquet(".../obs_frames.parquet")

assert gt.instance_id.nunique() == 4
assert len(frames) == 100
assert gt.visibility_ratio.max() > 0.9 and gt.visibility_ratio.min() < 1.0
```

第三条最关键：**可见率必须变化**。如果全程 1.0，说明遮挡/渲染通路没接对，S1 后面全是假的。

### W1 完成定义（DoD）

- [x] git 有首次提交，`freeze.json` 存在且字段齐全（`isaac` 版本已填）
- [x] 双环境可用，Isaac headless 启动成功，显存基线已记录
- [x] `pss spec lint` 可跑，产出 hash
- [x] 一条 episode 的 bundle 可读，四个断言全过
- [x] `notes/p1_week1.md` 记录：显存、单条 episode 耗时、踩到的坑

**不过不进 W2。**

---

## 3. 第 2–3 周：实体层竖切（S1）

| 天 | 任务 | 产出 |
|---|---|---|
| W2 D1–2 | 加 G1，先 `base_fixed` + 只做碰撞体 | 机器人可加载，不渲染 |
| W2 D3 | 双频循环：物理 120 Hz / 观测 10 Hz | `pss/isaac/loop.py` |
| W2 D4 | GT oracle 全字段导出（§3.4） | `gt.parquet` 补可见率、遮挡对 |
| W2 D5 | ZMQ obs/gt 双通道 + 进程隔离 | `pss/isaac/bus.py` |
| W3 D1–2 | 遮挡物注入 + 重入轨迹 | `faults.py` 的 `occlusion_insert` |
| W3 D3 | `pss/metrics/` 首批：IDF1、retention、switches、duplicate | `pss/metrics/entity.py` |
| W3 D4 | `oracle` 检测档（GT 框直接下发） | 窄链路端到端 |
| W3 D5 | 跑 S1 的 5 条 episode，出第一张表 | `scorecards/S1_oracle.md` |

**DoD**：`oracle` 档 IDF1 ≥ 0.95。**低于 0.95 说明指标实现或 GT 有 bug，不许往下走**——这是标定关卡，不是能力关卡。

---

## 4. 第 4–5 周：真实检测 + 孪生（S2）

| 任务 | 产出 |
|---|---|
| GroundingDINO Swin-T 接入，**离线批处理** | `pss/detect/groundingdino.py` |
| `noisy` 档（漏检/混淆/位姿噪声） | `pss/detect/noise.py` |
| 孪生资产（`twin_group`） | `assets/` + spec 双实例 |
| 三档对比表 | `scorecards/S1_{oracle,noisy,real}.md` |
| S2：duplicate rate + abstain precision | `scorecards/S2.md` |

**DoD**：三档分数都在，且 `real` 档的 IDF1 明显低于 `oracle`——如果两者接近，说明检测器没真正接入（在偷偷用 GT）。

---

## 5. 第 6–7 周：关系与 OOD（S3）

| 任务 | 产出 |
|---|---|
| GT 关系生成（10 谓词） | `gt_relations.py` |
| PSS 侧关系推断接 ZMQ | `pss/runtime` 接入 |
| per-predicate F1 + consistency violation | `pss/metrics/relation.py` |
| 组合 OOD 划分（按谓词组合切 test） | `specs/s3_relation_ood.yaml` |

**DoD**：consistency violation 可量化；OOD 子集与 ID 子集分表报。

---

## 6. 第 8–10 周：固定基座执行层（S4/S5）

> ADR-001：不做人形腿行走。感知/记忆的视角变化由 `cameras[].motion` 提供；
> 执行层只做 `base_fixed` 上肢。需要房间级移动时，另开轮式差速附表。

| 任务 | 产出 |
|---|---|
| G1 全身资产 + `base_fixed` 上肢 | `freeze.json` 补 `locomotion_policy: not_applicable` |
| 技能库 `skills_g1_v1` | `pss/isaac/skills.py`（reach/grasp/place/release） |
| 平行夹爪/吸盘代理 | 抓取成功率 ≥ 0.9；代理必须标注 |
| S4 闭环（脚本基线→PSS 接入） | `scorecards/S4.md` |
| 故障引擎 10 类 | `pss/isaac/faults.py`（camera/base/path 标注事件级） |
| S5 | `scorecards/S5.md`（按类分报，`base_fixed` 单表） |
| 轮式附表（可选） | `scorecards/S4_mobile.md`：差速底座，无 RL 策略 |

**DoD**：无故障时 `base_fixed` execution success ≥ 0.8；10 类故障各有 recovery 数字；
`base_disturb` / `path_blocked` 在主表只记事件，轮式附表另行物理注入。

---

## 7. 第 11–12 周：记忆与成本（S6/S7/S8）

| 任务 | 产出 |
|---|---|
| 多 episode 连续会话 + 物理改位 | `specs/s6_memory.yaml` |
| 事实编辑 + 回滚 | `pss/metrics/memory.py` |
| S7 校准（复用 S2/S5 样本） | `scorecards/S7.md` |
| S8 成本 | `pss/metrics/cost.py` |

**DoD**：rollback exactness = 1.0；forgotten-old 与 edited 分报。

---

## 8. 第 13–14 周：official 跑测

1. 写 `eval/prereg_isaac_v1.md`（A4）：指标、阈值、降采样、停止规则。
2. 3090 Ti 环境按 `freeze.json` 逐项对齐（A0 复检）。
3. 完整八步链路跑一遍（`PSS-IsaacSim-评测平台设计.md` §8.2.2）。
4. 出 `scorecards/`、`reports/<tag>/`（A8–A10）。
5. 4070 复算验证（A9）。

**DoD**：`reports/<tag>/` 八件齐全，`pss report --tag <tag>` 可一键重出全部分数。

---

## 9. 任务清单

### P1（W1）
- [x] T01 `git init` + 首次提交
- [x] T02 `freeze.json` 骨架
- [x] T03 建 `isaac` conda 环境（Python 按 Isaac 要求）
- [x] T04 装 Isaac Sim，headless 跑通最小示例
- [x] T05 记录显存基线与启动耗时
- [x] T06 `pss/spec/schema.py` + `loader.py` + `cli.py`
- [x] T07 `specs/s1_occlusion.yaml`（最小版）
- [x] T08 `pss/isaac/scene_builder.py`（桌面 + 4 物体 + 相机）
- [x] T09 bundle 写入器（manifest + 4 张表）
- [x] T10 单条 episode 跑通 + 四个断言

### P2（W2–W3）
- [x] T11 G1 加载（`base_fixed`，仅碰撞体）
- [x] T12 双频循环
- [x] T13 GT oracle 全字段
- [x] T14 ZMQ 双通道 + 进程隔离
- [x] T15 `occlusion_insert` 故障
- [x] T16 `pss/metrics/entity.py`
- [x] T17 `oracle` 检测档
- [x] T18 S1 首表（IDF1 ≥ 0.95）
- [x] T44 `cameras[].motion`：spec schema + runner 相机轨迹插值（ADR-001 主平台）
- [x] T45 ADR-001 平台决策记录（固定桌面 + 相机轨迹；人形腿移出关键路径）

### P3（W4–W5）
- [x] T19 GroundingDINO 接入（离线批）
- [x] T20 `noisy` 档
- [x] T21 孪生资产
- [x] T22 三档对比表
- [x] T23 S2 表

### P4（W6–W7）
- [x] T24 GT 关系生成
- [x] T25 `pss/metrics/relation.py`
- [x] T26 S3 组合 OOD spec
- [x] T27 S3 表

### P5（W8–W10）
- [x] T28 G1 全身资产 + `base_fixed` 上肢（ADR-001：`locomotion_policy: not_applicable`）
- [x] T29 技能库
- [x] T30 平行夹爪抓取 / 吸盘代理（资产无平行夹爪，报表标 `suction_proxy`）
- [x] T31 S4 闭环（脚本基线；PSS planner 接入是下一步）
- [x] T32 故障引擎 10 类
- [x] T33 S5 表
- [ ] T34 `wheeled_differential` 附表（可选，未做；不再依赖 G1 locomotion）

### P6（W11–W12）
- [x] T35 S6 多 episode 记忆
- [x] T36 `pss/metrics/memory.py`
- [x] T37 S7 校准
- [x] T38 S8 成本

### P7（W13–W14）
- [ ] T39 预注册 A4
- [ ] T40 3090 Ti 环境对齐
- [ ] T41 official 全量跑测
- [ ] T42 复算验证 A9
- [ ] T43 `reports/<tag>/` 归档 A10

---

## 10. 止损点

| 触发条件 | 动作 |
|---|---|
| W3 结束 `oracle` IDF1 < 0.95 | 停，查 GT 与指标实现，不许调 PSS |
| W5 结束 `real` 与 `oracle` 差距 < 0.1 | 停，怀疑检测器未真正接入 |
| W10 结束抓取成功率 < 0.8 | 停，换抓取方案（吸盘/简化夹爪），不硬调 |
| 任务开始依赖人形腿行走策略 | 停，按 ADR-001 改固定基座/相机轨迹，不接 RL 腿策略 |
| 任何一周 DoD 未过 | 本周任务顺延，不顺延后续周 |
| 任一锚点回退 | 其后锚点全部重置 |

---

## 11. 现在立刻做的三件事

1. **`git init` 并首次提交**（10 分钟）——这是 A0 的前提，也是“以后能追溯”的唯一办法。
2. **建 `isaac` 环境并装 Isaac Sim**（半天到一天）——这是唯一的硬阻塞项，其他都能并行。
3. **写 `specs/s1_occlusion.yaml` 的第一版**（1 小时）——哪怕字段不全，先让 `pss spec lint` 有东西可跑。

三件事做完，W1 的 D0–D2 就算完成，剩下的都是照着做。
