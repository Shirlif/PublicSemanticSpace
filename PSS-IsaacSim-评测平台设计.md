# PSS × Isaac Sim 评测平台设计

> 日期：2026-11-24
> 依据：`PSS-架构评测指标.md` §2–§8、`公共语义空间-落地执行计划-v1.md` E/F
> 原则：**把评测做成与仿真器解耦的测量仪器；Isaac 是可替换的物理后端，不是评测本身。**
> 目的：回答“哪些指标在 Isaac 上测、怎么测、测得可不可信”，而不是给仿真器做集成验收。

---

## 0. 一句话

> 我们需要的不是“把 PSS 接到 Isaac 上跑通”，而是**一台对物理后端不敏感的测量仪器**：
> 指标定义、真值防火、样本划分、复现信息都固定在评测核心里，
> Isaac 只负责产出 episode，且它的所有参数都被 spec hash 固定。

---

## 1. 四条原则

| # | 原则 | 违反后果 |
|---|---|---|
| P1 | **PSS 永远拿不到 GT。** GT 只进评分器，不进认知层 | 实体层 IDF1 变高但无意义；文档 §8 第 9 条 |
| P2 | **episode bundle 是评测原子。** 仿真只产 bundle，指标全部离线算 | 换种子/消融要重跑仿真，3–5 seeds 做不起 |
| P3 | **世界用声明式 spec 描述，编译成 USD，带 hash。** | 报表无法追溯，场景 bug 被误读成架构结论 |
| P4 | **两层成本。** Isaac = 高保真层；MuJoCo = 快速层（消融/多种子/CI） | 9 变体 × 5 种子全上 Isaac，工期与显存都不允许 |

> P4 不是“最小改造”的妥协，而是成本决策：文档 §4.2 要求 9 个消融变体、§4.4 要求 3–5 seeds，
> 在 8 GB 显存上全量跑 Isaac 不成立。两层共享同一份指标代码与 bundle 格式，数字才可比。
>
> 叠加在 P4 之上的是**设备分层**：4070 Laptop 只做流程验证（`device_class=dev`），
> 正式数字全部来自 3090 Ti（`device_class=official`），两者不可合并统计（见 §10）。

---

## 2. 目标架构

```
┌──────────────── Isaac Sim 进程 (GPU) ──────────────────┐
│  world_spec.yaml ──hash──▶ USD 场景构建 (Isaac Lab)    │
│                                                        │
│  physics 120Hz ◀──▶ G1 技能层(行走策略 + 上肢 OSC)     │
│  perception 10Hz ──▶ RTX RGB-D + instance seg          │
│  fault engine: 按时间线注入                            │
│  GT oracle: 实例ID / 6DoF / 可见率 / 接触 / 关系事实   │
└──────────┬──────────────────────────┬──────────────────┘
           │ obs socket               │ gt socket
           ▼                          ▼
   ┌───────────────┐          ┌───────────────┐
   │ Detector 档位 │          │ GT store      │
   │ oracle /      │          │ (immutable)   │
   │ noisy / real  │          └───────┬───────┘
   └───────┬───────┘                  │
           ▼                          │
   ┌───────────────┐                  │
   │ PSSRuntime    │                  │
   │ ingest → plan │                  │
   │ → act         │                  │
   └───────┬───────┘                  │
           ▼                          ▼
   ┌────────────────────────────────────────────┐
   │ episode bundle (parquet ×5 + manifest.json)│
   └───────────────┬────────────────────────────┘
                   ▼
   ┌────────────────────────────────────────────┐
   │ 离线指标引擎 → scorecard (md/json)         │
   └────────────────────────────────────────────┘
```

防火墙是**进程级**的：PSS 进程只连 obs socket，GT socket 只连写盘。不做成 code review 约定。

---

## 3. 组件规范

### 3.1 World Spec

一份 YAML 描述一个场景族。`spec hash` 是全实验的主键，进 manifest、进报表、进 bundle 目录名。

```yaml
spec_version: 1
id: tabletop_occlusion_v1
task_family: manipulation
split: test                       # train/val/test，按“资产实例”划，不按类别划

seed_policy:
  episode_seeds: [0, 1, 2, 3, 4]
  pose_sampling: latin_hypercube

assets:
  instances:
    - id: cup_blue_01
      class: cup
      usd: assets/cup_a.usd
      mass_kg: 0.08
      friction: {static: 0.8, dynamic: 0.6}
      restitution: 0.05
      twin_group: tg_cup_a       # 近同孪生组，用于 S2
      split: test
    - id: cup_blue_01t
      class: cup
      usd: assets/cup_a_twin.usd # 几何近同、纹理/材质不同
      twin_group: tg_cup_a
      split: test
  catalog_exclude: [assets/train_only_*]   # 禁止 test 出现的训练资产

scene:
  table: {size: [1.2, 0.8, 0.04], height_m: 0.75, friction: 0.7}
  workspace: {xy: [[-0.30, -0.25], [0.30, 0.25]], table_z: 0.77}
  bins: [{name: bin_a, pose: [0.25, 0.20, 0.80], size: [0.12, 0.12, 0.06]}]
  lighting: {preset: studio, perturb: {intensity: [0.6, 1.4], kelvin: [4000, 7000]}}

cameras:
  - {name: wrist, mount: robot.end_effector, fov_deg: 70, width: 640, height: 480, rate_hz: 10}
  - {name: overhead, pose: {pos: [0.0, 0.0, 1.30], ori: [1, 0, 0, 0]},
     fov_deg: 55, width: 640, height: 480, rate_hz: 10}

robot:
  model: unitree_g1                     # 29 DoF，Isaac Lab 官方资产；见 §3.3
  base_mode: base_fixed                 # base_fixed | base_mobile
  locomotion_policy: g1_loco_v1         # base_mobile 时必填且版本冻结
  arms_controller: {type: osc, rate_hz: 200}
  end_effector: {type: parallel_jaw, max_width_m: 0.08}   # 主线；Dex3 仅 S5 压力套件
  skill_library: skills_g1_v1           # reach/grasp/place/handover/walk_to/turn
  workspace_limits_m: [[-0.4, -0.4, 0.70], [0.4, 0.4, 1.30]]

physics:
  dt: 0.00833                     # 120 Hz
  substeps: 2
  solver: TGS
  contact_report: {enabled: true, max_contacts_per_pair: 8}
  determinism: {fixed_gpu: true, seed: 0}

task:
  id: move_cup_to_bin
  initial_facts: ["on(cup_blue_01, table)"]
  goal_facts: ["in(cup_blue_01, bin_a)"]
  success_check: {type: fact_set, tolerance_m: 0.02}
  time_limit_s: 60

faults:                            # 见 §6 故障谱系
  - {at_step: 180, kind: occlusion_insert,
     params: {occluder: board_01, region: [0.10, 0.0], duration_steps: 120}}
  - {at_step: 300, kind: object_relocate,
     params: {target: cup_blue_01, pose: [0.15, -0.10, 0.79]}}

perception:
  mode: offline                    # offline（录制后批处理）| in_loop（仅 S4/S5）
  detector_tier: frozen_real       # oracle | noisy | frozen_real
  detector: {name: groundingdino_swint, revision: "…", weights_sha256: "…"}
  in_loop_detector: {name: rtdetr_r18, revision: "…", weights_sha256: "…"}
  noise_profile: null              # tier=noisy 时必填，见 §3.5

relations_gt:
  predicates: [left_of, right_of, in_front_of, behind, on_top_of, inside,
               touching, near, above, below]
  compute_from: gt_poses           # 只允许 gt_poses / gt_contacts
  tolerance_m: 0.02

metrics:
  primary:   [idf1, id_retention, id_switches, duplicate_rate, reappearance_rate]
  leading:   [slot_purity, provenance_completeness]
  guardrail: [safety_violation_rate, permission_violation_count]

recording:
  obs_rate_hz: 10
  gt_rate_hz: 60
  bundle_format: parquet
  store_images: true               # false 时只存 camera 指针 + 帧号
```

规则：

- `split` 必须是**资产实例级**，同类别的不同实例可分属 train/test；杜绝同资产泄漏（对应文档 §4.3）。
- `relations_gt.compute_from` 只允许 GT 来源；PSS 侧的关系必须从感知位姿算，两者信息源不同才构成有效对比。
- `perception.detector_tier` 与 `detector.revision` 是主变量，任何报表必须显式标注。

### 3.2 Isaac 侧执行循环

**不用 `ManagerBasedRLEnv`。** 这是评测不是训练；RL env 的 observation/termination 抽象会迫使你把 PSS 语义塞进它的槽里。

用 `SimulationContext` + `InteractiveScene` + `Articulation` + `Camera`/`TiledCamera` + `ContactSensor`，自写循环。三条硬约束：

| 约束 | 值 | 理由 |
|---|---|---|
| 双频 | physics 120 Hz / 感知 10 Hz / 控制 200 Hz | 真机速率；单频同步会把渲染时间算进控制回路，延迟指标失真 |
| 无 UI | `--headless`，关掉 RTX path tracing / 降噪 | 8 GB 显存下 throughput 的唯一来源 |
| 并行 | dev 2–4 / official 8–16 | 人形 29 DoF + 接触计算；8 GB 上 4 是上限，24 GB 约 12 |

世界构建：`world_spec → USD` 一次性编译，缓存到 `~/.cache/pss_usd/<spec_hash>/`；只变更位姿的采样在运行期做，不重新编译。

### 3.3 人形机器人执行层

#### 选型：Unitree G1

| 项 | 决定 |
|---|---|
| 平台 | Unitree G1（29 DoF），Isaac Lab 官方资产 + 预训练行走 checkpoint |
| 末端 | 主线**平行夹爪**；Dex3 灵巧手只做 S5 的压力套件 |
| 控制 | 上肢 OSC；下肢行走策略冻结且版本锁定 |
| 基座 | `base_fixed` 与 `base_mobile` 两模式，分开报 |

为什么是 G1：Isaac Lab 有官方 USD 与预训练 locomotion 策略，社区资产、遥操作数据和故障注入经验最全，29 DoF 在 8 GB 上还能跑 2–4 个并行场景。其他候选都不如它省事：H1 上肢弱，GR-1 资产链路浅，OP3 没有操作能力。

#### 为什么主线用平行夹爪

多指灵巧抓取在仿真里成功率低、方差大。评测主体是认知层，底层抓取不可靠会让“执行失败”无法归因到 PSS。所以：

- 主线：平行夹爪 + 接触式抓取，目标成功率 ≥ 0.9；
- Dex3：单列压力套件，只报相对增量，不进主表。

#### 基座两模式（方法学关键）

| 模式 | 实现 | 用途 | 报表 |
|---|---|---|---|
| `base_fixed` | 基座焊死，只动上肢 | 消除平衡控制这个混杂因素 | 主表 |
| `base_mobile` | 冻结行走策略 + 上肢 OSC | 检验 loco-manipulation 下的规划与恢复 | 附表和 S5 专项 |

**必须分开报。** 人形摔倒导致的失败不是 PSS 失败；混在一张表里，任何结论都会被平衡控制器的方差淹没。这也符合文档 §4.1 的 capability-controlled 原则：消融只变 PSS，底层执行能力在两种模式下都必须恒定。

#### 技能抽象

PSS 不直接输出关节指令，而是输出技能序列：

```
reach(target) | grasp(obj) | place(obj, region) | handover(agent)
| walk_to(region) | turn(angle) | release()
```

每个技能由冻结的低层实现执行，返回 `status ∈ {ok, timeout, slipped, collision, unreachable}`。这样：

- `execution success` 衡量“技能序列是否达成目标”，不是“控制器调得好不好”；
- 技能失败可按类型计入 recovery，而不是笼统一个“失败”；
- 换机器人只换技能库实现，PSS 侧零改动。

### 3.4 GT Oracle 与防火墙

每帧导出，只写 GT store，不进 obs socket：

| 字段组 | 内容 |
|---|---|
| 身份 | `instance_id`、`class`、`twin_group` |
| 位姿 | `pose_xyz`、`pose_quat`、`bbox_3d` |
| 可见性 | `visibility_ratio`、`occluded_by[]`、`occluder_of[]` |
| 接触 | `contact_points[]`、`net_force_n`、`in_contact_with[]`、`grasped_by` |
| 关节 | `q`、`qd`、`tau` |
| 关系 | 由 `gt_poses` 按 spec 谓词生成的 `(subj, pred, obj)` 三元组 |

进程隔离：`pss_runtime` 进程只 `connect` obs 端点；`gt_writer` 进程只收 GT 端点。两进程无共享内存。

### 3.5 检测器：选型、三档、离线批处理

#### 选型

| 角色 | 模型 | 许可 | 理由 |
|---|---|---|---|
| 主档 `frozen_real` | **GroundingDINO Swin-T** | Apache 2.0 | 开放词表；S3 的未见组合不会退化成“检测器没这个词” |
| 快档 `frozen_real_fast` | **RT-DETR-R18**（PaddleDetection 官方实现） | Apache 2.0 | 实时，仅用于 S4/S5 闭环内 |
| 明确排除 | Ultralytics YOLOv8/v11 | **AGPL-3.0** | 与计划文档“主线只用 Apache/MIT”冲突；ultralytics 版 RT-DETR 同样是 AGPL，容易踩 |
| 可选辅助 | SAM2 | Apache 2.0 | 类别无关掩码，只在需要分割真值时用 |

三条理由：

1. 实体层评的是 ID 持久性，不是检测精度。检测器必须是**固定、可复现、开放词表**的背景条件，否则改动的是难度而不是能力。
2. COCO 封闭词表检测器遇到 S3 的未见组合会系统性失败，把 OOD 结论污染成“检测器没见过这个类别”。
3. AGPL 是实现层地雷：内部评测虽不构成分发，但主线依赖它会限制后续开源与商用路径。

#### 三档

| 档位 | 输入 | 用途 | 报告角色 |
|---|---|---|---|
| `oracle` | GT 框/位姿直接下发 | 指标上界标定 | 只报“天花板”，不进结论 |
| `noisy` | GT + 按 `visibility_ratio` 调制的漏检、类别混淆、位姿噪声 | 噪声强度作为自变量画曲线 | 支撑“增量随噪声单调”论断 |
| `frozen_real` | 固定版本检测器跑 RTX 渲染图 | **主结论** | 唯一可对外声称的证据 |

规则：**任何主结论必须来自 `frozen_real`。** `oracle` 的 IDF1 接近 1.0 是渲染质量，不是 PSS 能力。

#### 离线批处理

检测不在仿真回路里做，而是在 episode 录制完成后批量跑：

```
sim（录制 RGB-D + GT） → detect（离线批量） → PSS（离线重放） → 指标
```

这样能同时解决三件事：

- 检测器与 Isaac 不抢显存（8 GB 的主要瓶颈）；
- 检测结果写进 bundle，9 个消融变体重评**不需要重跑检测**；
- 换检测器版本只重算 detect 阶段，不动仿真与 GT。

唯一需要在线检测的是 S4/S5 闭环执行（PSS 动作依赖当前帧），用 `frozen_real_fast`。

`noisy` 的 profile：

```yaml
noise_profile:
  miss_rate: {base: 0.05, per_visibility_drop: 0.6}   # 可见率低于阈值的漏检
  class_confusion: [[cup, mug, 0.15], [bowl, plate, 0.08]]
  pose_sigma_m: 0.004
  false_positive_per_frame: 0.2
  id_ambiguity: {twin_group: 0.3}                     # 孪生组内 ID 互换概率
```

### 3.6 Episode bundle 字段

目录：`bundles/<suite>/<spec_id>/<spec_hash8>/<variant>/<tier>/seed<k>/`

`manifest.json`：

```json
{
  "bundle_version": "1.0",
  "episode_id": "s1_occ__seed03__frozen_real__full",
  "created_utc": "2026-11-24T12:00:00Z",
  "spec": {"id": "tabletop_occlusion_v1", "hash": "sha256:…"},
  "stack": {
    "isaac_sim": "5.x.y", "isaac_lab": "2.x.y", "physx": "…",
    "gpu": "RTX 3090 Ti 24GB", "driver": "616.64",
    "pss_commit": "…",
    "detector": {"name": "groundingdino", "sha256": "…"},
    "llm": {"name": "Qwen2.5-1.5B", "revision": "…"}
  },
  "variant": "full",
  "detector_tier": "frozen_real",
  "device_class": "official",
  "seed": 3,
  "outcome": {"task_success": true, "steps": 412, "wall_s": 38.2},
  "counts": {"obs_frames": 412, "gt_frames": 2472, "faults_injected": 2, "replans": 1},
  "files": {
    "obs_frames": "obs_frames.parquet", "obs_detections": "obs_detections.parquet",
    "gt": "gt.parquet", "actions": "actions.parquet",
    "events": "events.parquet", "pss_trace": "pss_trace.parquet"
  }
}
```

五张表（+ manifest）：

| 表 | 粒度 | 关键列 |
|---|---|---|
| `obs_frames` | 每感知 tick × 相机 | `step`、`t_s`、`camera`、`image_ref`、`depth_ref`、`robot_q[7]`、`ee_pose` |
| `obs_detections` | 每帧每检测 | `step`、`camera`、`det_id`、`bbox_xyxy`、`score`、`class_pred`、`pose_xyz`、`pose_quat`、`emb_ref` |
| `gt` | 每物理帧 × 实例 | §3.4 全字段 |
| `actions` | 每控制步 | `step`、`action_type`、`payload_json`、`executor_status`、`rejected_reason` |
| `events` | 事件驱动 | `t_s`、`kind`、`source`、`detail_json` |
| `pss_trace` | 每 PSS 调用 | `t_s`、`layer`、`latency_ms`、`state_version`、`entities_n`、`facts_n`、`expansions` |

`pss_trace` 是 T3 成本层的唯一来源；不进 trace 的层不能报延迟。

### 3.7 离线指标引擎

新建 `pss/metrics/`，每条指标一个函数 + 一份定义（公式、方向、所需表、bundle 字段依赖）：

```python
@metric(name="idf1", direction="max", uses=["gt", "obs_detections", "pss_trace"])
def idf1(bundle, spec) -> float: ...
```

约束：

- 指标实现**只有一份**，Isaac 层与 MuJoCo 层共用；
- 指标不允许读 `obs_frames` 以外的仿真内部状态；
- scorecard 生成器按文档 §7 模板输出，强制包含 `spec.hash`、`detector_tier`、`variant`、`seeds`。

---

## 4. 接入点

`pss/runtime/runtime.py:PSSRuntime` 已经是正确边界，不需要为 Isaac 改语义：

| runtime 方法 | Isaac 侧职责 |
|---|---|
| `ingest` / `ingest_sample` | obs socket 的检测结果 → 观测记录 |
| `assert_fact` | 任务下发（spec 的 `initial_facts`） |
| `set_goal` / `plan` | spec 的 `goal_facts` → Goal → Plan |
| `act` | Plan 步骤 → 末端/夹爪指令 |
| `rollback` | S6 记忆回滚、故障后恢复 |
| `explain` | §3.8 语言层，**不在 Isaac 回路内**，离线批量 |

**硬禁止**：runtime 内部不允许出现 `if simulator == "isaac"`。一旦出现，跨后端可比性失效。

---

## 5. 测试套件

S1/S2/S3/S6 是**离线套件**：仿真只跑一次，9 个消融变体共享同一批 bundle（GT 与变体无关），可用满 5 seeds。S4/S5 是**闭环套件**：PSS 在环，每个变体各跑一次仿真，official 配置 3 seeds，算力预算见 §10。

| 套件 | 场景 | 主指标 | 对应 | spec 规模（official） |
|---|---|---|---|---|
| **S1 遮挡 Re-ID** | G1 手臂/遮挡板扫过桌面，N≈20 物体 | IDF1、retention、switches、duplicate、reappearance | 指标 §3.2 P0-1 | 40 episodes × 5 seeds |
| **S2 孪生与弃权** | `twin_group` 资产对 | duplicate rate、abstain precision、risk-coverage | 指标 §3.2 §3.6 | 30 episodes × 5 seeds |
| **S3 关系 OOD** | 训练谓词 / 未见组合 | per-predicate F1、consistency violation、组合 OOD | 指标 §3.3 P0-3 | 60 episodes × 5 seeds |
| **S4 闭环操作** | G1 `base_fixed` 抓取-放置，目标以事实集表示 | task success、valid plan、expansions、execution success | 指标 §3.5 P0-4 | 50 episodes × 3 seeds |
| **S5 故障恢复** | G1 + §6 十类故障时间线 | recovery rate（分故障）、safety violation | 指标 §3.5 P0-4 | 10 类 × 10 episodes × 3 seeds（变体抽样见 §10） |
| **S6 长程记忆** | 多 episode 连续，物体被物理改变位置，事实编辑+回滚 | edited/old/forgetting/rollback exactness | 指标 §3.4 P0-2 | 100 步编辑序列 × 5 seeds |
| **S7 校准** | 复用 S2/S5 样本 | ECE/Brier、selective acc@coverage、conflict F1 | 指标 §3.6 P0-5 | 派生，无新增仿真 |
| **S8 成本** | 同成功率下延迟/内存 | p50/p95、footprint、token | 指标 §3.9 | 复用 S1/S4 |

S6 是 Isaac 唯一不可替代的记忆测试：物体被真实物理改变位置后，“旧事实是否仍正确”“新事实是否覆盖”“回滚是否逐字段还原”才有物理意义。

不搬的层：`PSS-架构评测指标.md` §3.7 审计、§3.8 语言/工具——与物理引擎无因果关系，留在现有 harness。

---

## 6. 故障谱系

文档要求 `recovery rate` 但未定义故障类型。本表即为定义，也是可对外声称的设计贡献。

| kind | 注入方式 | 期望 PSS 行为 | 主要指标 |
|---|---|---|---|
| `occlusion_insert` | 生成遮挡板 | 保持 ID，不重复建槽 | duplicate_rate、retention |
| `object_relocate` | 瞬移物体 | 检测位置冲突并更新事实 | conflict F1、old accuracy |
| `object_remove` | 移出工作区 | 标 dormant，不立即删除 | reappearance、provenance |
| `grasp_slip` | 降摩擦/加大加速度 | 重规划 | recovery rate、replans |
| `camera_dropout` | 停发 obs 200–800 ms | 降级、用最后状态 | fallback_rate、safety |
| `lighting_shock` | 光照突变 | 不崩、不幻觉 | entity hallucination |
| `twin_swap` | 孪生实例替换 | **应弃权**而非硬猜 | abstain precision |
| `joint_limit` | 指令越界 | 拒绝动作并记录 | permission violations |
| `base_disturb` | 对躯干施加外部推力 | 不摔倒，恢复站姿后继续 | safety violation、recovery |
| `path_blocked` | 行走路径放置障碍 | 绕行或明确放弃 | replans、recovery |

`recovery rate = 注入后仍达成目标 / 注入次数`，按 kind 分报，不报平均数。

---

## 7. 消融矩阵

文档 §4.2 的 9 个变体（`full` / `no_entity` / `no_relations` / `no_memory` / `no_active_set` / `no_uncertainty` / `no_audit` / `flat_context` / `no_tools`）在每个套件上各跑一遍。

- 变体通过 `PSSRuntime` 构造参数切换，**不通过改 Isaac 场景**；
- 离线套件里 `full` 与 `no_X` 共享同一批 bundle 的 GT（GT 与变体无关），消融不产生额外 GPU 成本；
- 闭环套件（S4/S5）无法共享 bundle：每个变体必须各自跑仿真。official 配置下 S4 跑全 9 变体 × 3 seeds；dev 配置只抽样 3 变体 × 1 seed，仅用于验证流水线，不进报表；
- delta 的置信区间用配对 bootstrap，按 episode 配对而非按 seed。

`flat_context` 与 `no_tools` 只在 S1/S4 上跑，其余套件无意义。

---

## 8. 执行流程与锚点

### 8.1 为什么要锚点

一个过夜批跑完，如果没人能回答“这个 IDF1 是哪个 spec、哪台卡、哪个检测档、哪个变体产的”，
那批数据就是废的。所以每个阶段必须落**可验证产物 + 可重入检查点**；
检查点串成锚点链，任何时刻都能从最后一个通过的锚点续跑，任何数字都能沿链回溯。

锚点是**顺序门禁**：前一个不通过，后一个不启动。这是本平台的工程纪律，不是建议。

### 8.2 单次评测的执行链路

> **适用范围**：下面的完整链路用于 official 跑测与每周预跑。
> P1–P3 建设期走**窄链路**（§8.2.1），目标是最快拿到一条端到端 episode，而不是走全流程。
> 把两者混为一谈，会导致在流水线还没成形时就卡在“检测覆盖率 100%”这类运行期门禁上。

#### 8.2.1 窄链路（建设期）

```
spec lint → USD build → 单 episode → 以 GT 当 detections → replay → 看数
```

| 走法 | 什么时候用 | 单次耗时 |
|---|---|---|
| 窄链路 | P1–P3 日常迭代 | 分钟级 |
| 完整链路 | P5 起、official 跑测、每周预跑 | 小时级 |

窄链路可以省掉检测（直接注入 `oracle` 档），但两条约束不能省：

- **GT 仍然只在评分侧**，窄链路不许把 GT 塞进 PSS 的观测（原则 P1 不破）；
- **bundle 格式与完整链路一致**，不能为了快另起一套字段。

省的是步骤，不是契约。

#### 8.2.2 完整链路

```
spec lint → USD build → 冒烟 → 生成 → 检测 → 重放 → 评分 → 复算
```

| # | 步骤 | 命令形状（约定，实现时固定，避免各脚本参数漂移） | 输入 → 产出 | 失败意味着 |
|---|---|---|---|---|
| 1 | spec 校验 | `pss spec lint specs/<id>.yaml` | yaml → `specs/registry.jsonl`（含 hash） | spec 本身错，不是架构问题 |
| 2 | USD 编译 | `pss sim build <spec>` | spec → `~/.cache/pss_usd/<hash>/` | 资产或物理参数错 |
| 3 | 冒烟 | `pss sim run <spec> --seed 0 --episodes 5 --device dev` | spec → 5 bundles | **流水线 bug，禁止继续** |
| 4 | 生成 | `pss sim run <spec> --seeds 0-4 --device official` | spec → N bundles | 计数不符 → 断点续跑 |
| 5 | 检测 | `pss detect bundles/<spec>/<tier>` | RGB → `detections.parquet` | 覆盖率 < 100% → 重跑该批 |
| 6 | 重放 | `pss replay bundles/<...> --variant <v>` | bundle + detections → `pss_trace.parquet` | PSS 抛异常 → 代码 bug |
| 7 | 评分 | `pss score --suite S1 --device official` | bundles → `scorecards/S1.json` | 混入 `dev` → 直接拒绝 |
| 8 | 复算 | `pss verify --bundles <...> --on dev` | 4070 重算 → 逐位比对 | **指标引擎非确定性，全线停** |

第 3 步和第 8 步是整条链路的两道保险：一道防“还没跑通就出数”，一道防“数出来了但对不上”。

#### 8.2.3 哪些顺序是硬的，哪些可以松

| 关系 | 判定 | 说明 |
|---|---|---|
| 生成 → 检测 | **硬依赖** | 检测读渲染帧 |
| 检测 → 重放 | **硬依赖** | PSS 消费检测，不消费原始图像 |
| GT 落盘 → 任何评分 | **硬依赖** | GT 必须在 rollout 结束时落盘，不能事后从仿真补算 |
| 冒烟（第 3 步） | **不可跳** | 跳过直接进 official 生成，是最大的一次性浪费 |
| 复算（第 8 步） | **不可跳** | 跳过等于放弃“数字可信”这一整条论证 |
| 生成与检测 | **可流水** | 批 1 在检测时批 2 继续生成，前提是每批自身锚点完整 |
| 检测与重放 | 可流水 | 同上，按批对齐 |
| 评分与下一批生成 | 可重叠 | 评分只读 bundle，不干扰仿真 |

一句话：**依赖关系不能松，调度可以并行**。判据是“这批的锚点是否自洽”，不是“是否按表格顺序敲命令”。

### 8.3 锚点链

| 锚点 | 名称 | 产出 | 通过判定 | 失败动作 | 状态 |
|---|---|---|---|---|---|
| **A0** | 版本冻结 | `freeze/<tag>.json` | 两机 Isaac / Isaac Lab / PhysX / driver / detector / LLM 版本逐项一致 | 修环境，不推进 | ☐ |
| **A1** | spec 冻结 | `specs/registry.jsonl` | schema 校验通过，hash 唯一 | 修 spec | ☐ |
| **A2** | USD 编译 | `~/.cache/pss_usd/<hash>/` | dev 与 official 构建出相同 prim 树 | 查资产与物理参数 | ☐ |
| **A3** | dev 冒烟 | `bundles/dev/...` | 5 套件各 1 seed × 5 eps，全部可读、PSS 无异常 | 修流水线 | ☐ |
| **A4** | 预注册 | `eval/prereg_isaac_v1.md` | 指标、阈值、降采样、停止规则**先于跑测**冻结 | 不许开跑 | ☐ |
| **A5** | 正式生成 | `bundles/official/...` | episode 计数 = spec 预期，manifest 字段齐全 | 断点续跑 | ☐ |
| **A6** | 检测归档 | `detections.parquet` | 覆盖率 100%，无缺帧 | 重跑该批 | ☐ |
| **A7** | 重放矩阵 | `pss_trace.parquet` | 变体 × 种子矩阵完整 | 补跑缺项 | ☐ |
| **A8** | 评分 | `scorecards/<suite>.md/json` | `device_class` 单一，CI 可算 | 拒绝出表 | ☐ |
| **A9** | 复算验证 | `verify/<tag>.json` | 4070 重算与 3090 Ti 逐位一致 | **停线**，修指标引擎 | ☐ |
| **A10** | 归档 | `reports/<tag>/` + git tag | 复现清单完整（§8.5） | — | ☐ |

状态列随工程推进勾选；任一锚点回退，其后所有锚点状态重置。

### 8.4 断点与重入

- 每个锚点产物自带 manifest，启动时先扫锚点决定从哪一步续，不靠人记；
- 命令**幂等**：同一 `(spec, seed, variant, tier)` 已完成则跳过，除非显式 `--force`；
- 任何 `--force` 覆盖写入 `reports/<tag>/overrides.jsonl`，否则 A10 不通过；
- 锚点产物缺失但下游存在 → 视为无效状态，需回退重跑（不允许“补个 manifest 了事”）。

### 8.5 复现清单

`reports/<tag>/` 必须包含：

```
freeze.json            版本五元组
specs/                 所有 spec + hash
prereg.md              预注册
scorecards/            每套件一表
bundles.index.parquet  episode → 路径 → 指标行号
verify.json            复算结果
overrides.jsonl        所有 --force 记录
notes.md               人工判断与已知偏差
```

一条命令重出全部分数：`pss report --tag <tag>`。做不到这条，A10 不通过。

### 8.6 回归锚点：golden bundles

固定 20 条 episode（每套件 2–4 条），**永久保留**原始图像与 GT，不随迭代删除。

- 任何代码改动后跑 `pss replay golden/ --variant full`；
- 指标必须落在 `±tol` 内，超差 = 行为变更；
- 行为变更必须在 `notes.md` 说明并记录新的 tol，**不得静默放过**。

golden 是“后期锚点”里最实用的一条：它让“这次改动有没有悄悄改变结论”变成一次十分钟的检查，而不是一次 45 小时的复跑。

### 8.7 日常节奏

| 周期 | 动作 | 对应锚点 |
|---|---|---|
| 每天 | dev 冒烟 + golden 回归 | A3 |
| 每次提交 | golden 回归必须通过才能进 A5 | A3 |
| 每周 | 挑一个套件做 official 预跑，校验吞吐与预算 | A5–A7 |
| 里程碑 | 完整锚点链走一遍，产出 tag | A0–A10 |

---

## 9. 交付节奏

| 阶段 | 周 | 交付 | 门禁 |
|---|---|---|---|
| P1 骨架 | 1–2 | spec schema、USD 编译、双频循环、socket 契约、bundle 格式、replay | 能产出一条完整 bundle 并离线读出 |
| P2 实体层 | 3–4 | S1 + S2，`oracle`/`noisy` 两档，scorecard | IDF1 在 oracle 档 ≥ 0.95（标定通过） |
| P3 感知真实性 | 5–6 | `frozen_real` 档、GT 关系、S3 | 主结论可在真实检测器下复现 |
| P4 执行层 | 7–10 | G1 资产、技能库、S4、故障引擎、S5 | 无故障时 execution success ≥ 0.8（`base_fixed`） |
| P5 记忆与成本 | 11–12 | S6/S7/S8 | 回滚 exactness = 1.0 |
| P6 正式运行 | 13–14 | 预注册 + 5/3 seeds + 完整 scorecard | 见 §11 |

**P1 不做完不许碰指标。** 否则把场景 bug 当架构结论。

1–2 人，约 14 周。人形比固定臂多约一周：资产接入、技能库、行走策略冻结验证。

机器分工按阶段走：P1–P3 在 4070 上迭代；**P4 起闭环套件的迭代放 3090 Ti**（8 GB 跑人形闭环，一晚上只够几十个 episode），4070 保留给离线重放、指标调试与跨机重算验证。

---

## 10. 硬件、分工与许可

两台机器，职责不同，**结果不可混用**。

| 机器 | 用途 | `device_class` |
|---|---|---|
| RTX 4070 Laptop 8 GB | 流程验证、小场景调试、离线重算 | `dev` |
| RTX 3090 Ti 24 GB | 正式跑测 | `official` |

### 硬性规则

1. **正式结果必须在单机上完整跑完。** Ampere 与 Ada 的浮点与物理实现不同，跨卡的 episode 不能合并统计。
2. **两台的 Isaac Sim / Isaac Lab / PhysX / 驱动版本必须完全一致。** 否则 GT 本身就不可比；任一端升级即视为新 tier，需重跑。
3. bundle manifest 带 `device_class` 与 `gpu`；scorecard 生成器**拒绝**混合 `dev` 与 `official` 输入。
4. 4070 上的职责是冒烟：5 个套件各 1 seed × 5 episodes，验证流水线、显存边界与 bundle 可读性。

### 两个配置

| 项 | dev（4070 8 GB） | official（3090 Ti 24 GB） |
|---|---|---|
| 并行场景 | 2–4 | 8–16 |
| 分辨率 | 480×640 | 640×480 起，可到 720p |
| 非闭环 seeds | 1 | 5 |
| 闭环 seeds | 1 | 3 |
| S4 消融变体 | 抽样 3 个 | 全 9 个 |
| S5 故障类 | 3 类 | 10 类 |
| 预计时长 | 3–5 h | 约 45 h |

其他环境约束：

| 项 | 现实 | 决策 |
|---|---|---|
| 磁盘 | 4070 机 D 盘剩 85 GB | Isaac 装 D 盘；资产与权重按需拉取（见下“存储”） |
| 平台 | Windows 无官方 ROS2 bridge | Python + ZMQ，不用 ROS2 |
| 许可 | Isaac Sim / SimReady / G1 USD 为 NVIDIA 许可 | 计划文档“主线只用 Apache/MIT”需写明例外：仅用于评测，不进训练数据 |

### 算力预算（3090 Ti 24 GB，正式跑测）

| 项 | 估算 |
|---|---|
| 非闭环吞吐 | 8–16 并行、640×480、10 Hz 感知 ≈ 8–12 s/episode |
| S1+S2+S3 | (40+30+60) × 5 seeds = 650 episodes ≈ **2 h** |
| S6 长程序列 | 100 × 5 seeds × 5 子 episode ≈ 2500 episodes ≈ **7 h** |
| S4 闭环 | 9 变体 × 50 eps × 3 seeds = 1350 × ~45 s ≈ **17 h** |
| S5 闭环 | `full` 10 类 × 20 × 3 = 600；3 个关键变体 × 10 类 × 10 × 3 = 900；共 1500 × 45 s ≈ **19 h** |
| 检测批处理 | GroundingDINO 约 1–2 h，可与仿真并行 |
| 合计 | **约 45 GPU 小时**，两到三个过夜批 |

**24 GB 买到的是范围，不是时间。** 种子回到 5、S4 变体回到全 9 个、S5 回到 10 类，所以总时长与 8 GB 方案相近——但结论的置信区间窄得多。

### 存储与搬运

- 640×480 JPEG(q85) 约 40 KB/帧，10 Hz × 60 s ≈ 24 MB/episode。正式跑测约 5000 episodes ≈ **120 GB**，4070 机 D 盘 85 GB 装不下。
- 策略：**episode 结束后就地跑检测，只归档检测结果**；原始图像仅对 headline 套件（S1/S4）保留，或按 `store_images: per_suite` 白名单控制，约 20–30 GB。
- bundle 内一律相对路径，manifest 不含绝对路径，便于把 official bundle 拷回 4070 做离线重算与复核。
- 重算验证是 §11 第 7 条的前提：**用 4070 重算 3090 Ti 产出的 bundle，指标必须逐位一致**（GPU 只影响生成，不影响评分）。

---

## 11. 完成标准

“测试任务完成” = 以下同时成立：

1. 五个体检指标（`PSS-架构评测指标.md` §9）各有一张 scorecard，含 `full` 与 8 个消融的 delta；
2. 所有主结论来自 `frozen_real` 检测档；
3. 每张表带 seeds（Isaac 非闭环 5 / 闭环 3 / MuJoCo 快层 5）、95% CI、`spec hash`、bundle 路径；
4. 全部 official 数字来自同一台 3090 Ti、同一套 Isaac/PhysX/驱动版本；混入任何 `device_class=dev` 的 episode 即作废；
5. S5 的 recovery rate 按 §6 的 10 类分报，且 `base_fixed` 与 `base_mobile` 分表；
6. S6 的 rollback exactness = 1.0，且 forgotten-old 分报；
7. 同一批 bundle 可在无 GPU 机器上完整重算出同样的数字，且 4070 重算 3090 Ti bundle 的指标逐位一致；
8. 所有降采样（变体 × 种子 × 类目）在预注册文件中可查。

达不到第 7 条，就还是“跑通 Isaac”，不是“完成评测”。

---

## 12. 不做的事

1. 不把 `PSS-架构评测指标.md` §3.4 记忆、§3.7 审计、§3.8 语言搬进 Isaac——无因果关系，只增加周期。
2. 不在本期做 sim2real——文档列为 P2，且需要真机标定。
3. 不报 oracle 档的数字作为能力证据。
4. 不把 Isaac 的渲染延迟混进 T3 报表——单独 tier，单独标注。
5. 不为 Isaac 在 `PSSRuntime` 里加分支。
6. 不用 RL env 包装评测循环。

---

## 13. 对现有代码的处置

| 现有 | 处置 |
|---|---|
| `pss/sim/`（MuJoCo） | 保留为快速层；`scene.py:CATEGORY_SPECS` 迁到 spec schema 的 `assets` 段 |
| `pss/sim/generator.py` | 保留，只做训练数据生成；与评测流水线隔离 |
| `pss/eval/*.py` | 指标实现抽到 `pss/metrics/`；脚本退化为薄 CLI |
| `pss/eval/relational_closed_loop.py:_place_at` | 传送式放置，在 Isaac 层换成 G1 的 `grasp/place` 技能；旧数字标注为“传送代理” |
| `pss/runtime/` | 不动；作为唯一接入点 |
| `eval/preregistration.md` | 新增 Isaac 套件的预注册节（阈值先写后跑） |
