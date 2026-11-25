# P2 / W2 进度 — 实体层竖切（S1）

> 开始：2026-11-25
> 入口条件：W1 DoD 全过（见 `notes/p1_week1.md`）
> 状态：T11–T18 全部完成；oracle 标定门禁 IDF1 = 1.000（≥ 0.95）。

## 已完成

### T11 G1 加载（base_fixed，仅碰撞体）

- 资产：`_external/isaac_assets/G1`，155.4 MB / 29 个对象
  （`scripts/fetch_isaac_assets.py` 从 NVIDIA 公共 S3 镜像，manifest 记录目录 sha256）
- spec：`specs/g1_load.yaml`（`specs/registry.jsonl` 已登记）
- 构建：`pss sim build specs/g1_load.yaml` → 165 prims；`scripts/check_g1_scene.py` PASS
  - 151 个 robot prims，15 个 collision prims
  - articulation root：`/World/Robot/pelvis`
  - variants：Physics=SimplifiedPhysX，hands/sensor=None
  - visibility=invisible（碰撞体加载，不渲染）
- 关键坑：G1 5.1 资产的碰撞网格放在**同级 `/colliders` scope**，只引用 defaultPrim
  时会被丢弃；scene_builder 把 `/colliders/<link>` 逐 link 引用到
  `/World/Robot/<link>/pss_collisions`，并清空碰撞网格上越界的 material binding。

### T12 双频循环

- 物理 120 Hz / 观测 10 Hz；`SimulationContext.play()` 另有 2 次内部步进
- W1 空场景热跑 ~8 s；G1 场景首跑 ~23–40 s（PhysX/RTX 初始化）
- G1 1 s：120 物理步 / 10 观测帧 / GT 120 行

### T13 GT 全字段

- GT 增加接触字段：解析式 support/box-box 接触（W2 近似），`in_contact_with`、
  `contact_points`、`net_force_n`；cubes 在桌上 → `table`，0.4905 N。
- 关节字段不塞进 GT 实例表（避免 robot 变成第 5 个实例、污染 IDF1）：
  新增 `robot_state.parquet`，每物理帧 `base_pose` + 29 DoF `q/qd/tau`。
- 关系 GT 仍留到 W4（按计划 P4 T24）。

### T14 ZMQ obs/gt 双通道

- `pss/isaac/bus.py`：obs = PUB（允许多订阅者、可丢帧），gt = **PUSH/PULL**
  （GT 不允许丢；PUB/SUB 慢加入会丢消息，实测 120 条只收到 39 条）。
- `pss sim run ... --bus` 发布双通道；`scripts/gt_writer.py` 是独立进程，
  只连 GT 端点，验证 120/120 条 GT 消息全部落盘。
- 进程隔离是地址空间级：PSS 进程只给 obs 端点。

### T15 occlusion_insert 故障

- 运行时在 `/World/Faults/<name>` 插入静态遮挡板，到期移除；
  同时进入可见率采样，GT 的 `occluded_by` 反映故障。
- S1 spec：step 180 注入、持续 120 步；events 记录 `fault_injected` / `fault_removed`，
  manifest `faults_injected=1`。

### T16 `pss/metrics/entity.py`

- IDF1（全局二分匹配）、id_retention、id_switches、duplicate_rate、
  reappearance_rate、slot_purity；`@metric` 注册表记录方向与依赖表。
- 纯离线、无 GPU；W1/W2 与 MuJoCo 快层共用同一实现。

### T17 oracle 检测档

- `pss/detect/oracle.py`：GT 3D 框投影为 2D bbox，写 `obs_detections.parquet`；
  检测行与 GT 表分离。
- `pss/detect/replay.py`：PSS 只读 detections，写 `entity_trace.parquet` +
  `pss_trace.parquet`；oracle 档给每个实例完美的 one-hot appearance，
  tracker 用 appearance-dominant 关联（oracle 是上界标定，不是能力证据）。
- CLI：`pss detect <bundle>` / `pss replay <bundle>` / `pss score --suite ...`。

### T18 S1 首表（oracle）

- spec：`specs/s1_occlusion_20.yaml`，20 cubes，每 seed 分层随机布局，
  移动遮挡板 + `occlusion_insert`，G1 碰撞体加载。
- 5 条 episode（seeds 0–4）→ detect → replay → score。
- **IDF1 = 1.0000 ± 0.0000**（5 seeds），retention / reappearance = 1.0，
  switches / duplicate = 0。门禁 ≥ 0.95 通过。
- 表：`scorecards/S1_oracle.md` / `.json`；别名 `scorecards/s1_occlusion_20.*`。

## 复现

```powershell
$env:OMNI_KIT_ACCEPT_EULA = "YES"; $env:NO_PROXY = "*"
D:\isaac\env\Scripts\python.exe -m pss sim run specs\s1_occlusion_20.yaml --episodes 5 --seed 0 --device dev
foreach ($b in Get-ChildItem bundles\dev\s1_occlusion_20\*\seed*) {
  python -m pss detect $b.FullName
  python -m pss replay $b.FullName
}
python -m pss score --suite s1_occlusion_20 --device dev --label S1_oracle
```

## 已知限制 / 下一阶段入口

- oracle 的完美外观是上界假设；`noisy` / `frozen_real` 档在 W4/W5 接入。
- base_fixed 的 world fixed joint 尚未加；S1 的 G1 只加载不运动，S4 前必须补。
- 接触 GT 是解析式（桌面支撑 / AABB 接触），不是 PhysX contact report；
  S4/S5 前换真实 contact sensor。
- 资产引用在缓存 USD 中是绝对路径；跨机复算前在目标机重建 USD 缓存
  （prim hash 与路径无关，A2 仍可比较）。
- W3 的 oracle IDF1 门禁已过；下一阶段是 W4 的 `noisy`/`real` 检测器与孪生资产。
