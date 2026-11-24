# P2 / W2 进度 — 实体层竖切（S1）

> 开始：2026-11-25
> 入口条件：W1 DoD 全过（见 `notes/p1_week1.md`）
> 本周只交付 T11–T18；未过 T18（oracle IDF1 ≥ 0.95）不进 W3。

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
  `/World/Robot/<link>/pss_collisions`，并清空碰撞网格上越界的 material binding
  （否则 USD 每次 update 都重复解析坏路径，`play()` 长时间卡住）。
- 未决：base_fixed 的固定关节尚未加；W2 运行是 1 s 载入检查，不评估站立/操作。

### T12 双频循环

- `pss sim run specs/g1_load.yaml --episodes 1 --seed 0 --device dev --force`
- 1 s：120 物理步 / 10 观测帧 / GT 120 行；`SimulationContext.play()` 另有 2 次内部步进
- 首跑 G1 场景 wall ~40 s（含 PhysX/RTX 初始化）；W1 空场景热跑 ~8 s
- 注意：`g1_load` 无遮挡板，`verify` 的“可见率必须变化”断言按预期 FAIL；该断言只适用于 S1。

## 待办（按计划顺序）

- [ ] T13 GT 全字段（接触、关节 q/qd/tau、关系）——需要先定 robot 是否作为 GT 行
- [ ] T14 ZMQ obs/gt 双通道 + 进程隔离
- [ ] T15 `occlusion_insert` 故障
- [ ] T16 `pss/metrics/entity.py`（IDF1、retention、switches、duplicate、reappearance）
- [ ] T17 `oracle` 检测档（GT 框下发，窄链路）
- [ ] T18 S1 首表，oracle IDF1 ≥ 0.95（标定关卡）

## 阻塞与决策

- **T13 的 robot GT 行**：设计 §3.6 的 `gt` 是“每物理帧 × 实例”，而 §3.4 又列了关节字段。
  两条放在一起会迫使 robot 变成第 5 个“实例”，污染 IDF1 的实例计数。建议 robot 不进入
  `gt` 实例表，另开 `robot_state.parquet`（或只在 S4/S5 写），需要拍板。
- **base_fixed**：G1 资产没有内置固定基座；W3/S4 前需要加 world fixed joint 或用
  articulation 的 fixed-base 配置。当前只验证加载与循环。
- **资产引用绝对路径**：缓存的 scene.usd 里 G1 引用是绝对路径；跨机复算前需在目标机
  重建 USD 缓存（prim hash 与路径无关，A2 仍可比较）。official 跑测前应改成可搜索路径。
