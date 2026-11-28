# PSS 兼容性与模态扩展任务计划

> 日期：2026-11-28
> 状态：W0–W7 核心实现完成（2026-11-28）；剩余真实种子集、时间戳生产接入与灰度上线
> 范围：把“具体要维护的兼容性清单”落地为可执行任务；不改 `PUBLIC_DIM`、归一化、绑定算子
> 关联：`pss/contracts/schema.py`、`pss/prototypes/build.py`、`pss/prototypes/joint_readout.py`、`pss/inference.py`、`pss/memory/perception.py`、`pss/runtime/`
> 上游：`contracts/README.md`、`docs/experiments/PSS-手工视觉专家与真实图像复核.md`、`docs/experiments/PSS-真实触觉专家复核.md`

---

## 0. 一页纸摘要

**目标**：让“新增一种感官模态”变成**配置 + 适配器 + 契约测试**，而不是改 PSS 核心。上游传感器怎么变，下游的原型库、读出、绑定、记忆、规划都不需要重写。

**核心原则**：

1. **核心稳定**：公共空间、绑定算子、实体/关系契约保持稳定。
2. **模态插件化**：模态是注册表里的一条数据，不是代码里的一个 tuple。
3. **契约测试优先**：兼容性必须被测试，不能被声明。

**非目标**：

- 不修改 `PUBLIC_DIM`、归一化、Hadamard 绑定、`SLOT_COUNT`（属于大版本变更）。
- 不为尚不存在的模态预建大量抽象。
- 不做大规模数据采集；只做最小种子集与公开数据适配。

**成功指标**：

| 指标 | 目标 |
|---|---|
| 新模态上线工作量 | ≤ 1 人日（YAML + adapter + 投影头 + 测试），核心代码零改动 |
| 3 模态金标准 | 重构前后输出逐位一致（固定 seed） |
| 部分模态 | 1/2/3 模态查询均可用；缺失模态不抛异常 |
| 旧 artifact | 0.1.1 的 bank/projection/readout 全部可加载 |
| 兼容性矩阵 | 13 项扰动全部通过（见 §6） |
| 回归 | 现有测试全绿；无新增 flaky |

**排期**：约 36 人日。1 人约 7–8 周；2 人并行约 4 周。

---

## 1. 背景与判断

上游感官硬件迭代快、种类变多，这是趋势，但不等于“架构会自动兼容”。兼容性是**经验属性**：

- 新模态带来新的维度、量纲、时间尺度、噪声结构和缺失模式；
- 新精度会改变分布，投影、校准、开集阈值、可靠性权重都可能失效；
- 异步流会污染绑定，缺失模态会让原型库和联合读出退化。

因此本计划的目标不是“完美适配”，而是：

> **稳定核心 + 每模态插件 + 契约测试 + 优雅降级 + 快速适配。**

数据采集的定位也随之改变：不做重资产采集，只保留**最小真实种子集**作为架构的“单元测试”。

---

## 2. 现状审计

| # | 现状 | 位置 | 兼容性风险 |
|---|---|---|---|
| A1 | `MODALITIES` 是封闭 tuple | `pss/contracts/schema.py:22` | 加音频/6D 触觉要改多处代码 |
| A2 | `build_bank` 要求每类每模态都有样本，否则 `raise` | `pss/prototypes/build.py:27-39`（`:32` 抛错） | “分批纳入”会被直接挡住 |
| A3 | `JointReadout` 的 7 个 mask 硬编码 | `pss/prototypes/joint_readout.py:22, 50-60, 92-100` | 4 模态 15 个、8 模态 255 个，不能全训 |
| A4 | `PublicSemanticSpace.heads` 按固定 `MODALITIES` 构建 | `pss/inference.py:33, 79-83` | 新模态要改构造 |
| A5 | `Observation` 没有显式异步/延迟契约 | `pss/memory/perception.py:33-45` | 多传感器时间错位会污染绑定 |
| A6 | 投影/原型/校准各自版本，无统一清单 | `pss/projection/heads.py:124`、`pss/prototypes/bank.py:42`、`pss/prototypes/joint_readout.py` | 升级后旧向量失效，缺迁移/回滚路径 |
| A7 | 权重字典硬编码 | `pss/inference.py:141-153, 187-195` | 新模态没有权重，融合不可扩展 |
| A8 | demos/eval 脚本各自复制 `MODALITIES` | `pss/demos/`、`pss/eval/scripts/` | 重构后不一致；先加兼容 shim，后续再收口 |

**结论**：核心链路（contracts / prototypes / inference / memory / runtime）需要改造；demos/eval 只需兼容，不在本轮强制迁移。

### 2.1 清单与工作流映射

| 清单项 | 工作流 |
|---|---|
| `MODALITIES` 封闭 tuple | W1 模态注册表 |
| `build_bank` 每类每模态必须有样本 | W2 部分模态原型库 |
| `JointReadout` 的 7 个 mask 硬编码 | W3 动态联合读出 |
| `PublicSemanticSpace.heads` 按固定 MODALITIES 构建 | W4 动态空间与感知构造 |
| `Observation` 没有显式异步/延迟契约 | W5 时间与异步契约 |
| 投影/原型/校准各自版本 | W6 统一版本、迁移与回滚 |
| 兼容性如何被验证 | W0 金标准 + W7 契约测试与种子集 |

---

## 3. 目标架构

```
sensor stream
   │  (raw, timestamp, sensor_id)
   ▼
Adapter ──► Expert (encode -> native d) ──► ProjectionHead (-> 128d)
   │                                            │
   │                                            ▼
   │                                   PublicSemanticSpace
   │                                   ├─ PrototypeBank (per-modality prototypes)
   │                                   ├─ JointReadout (mask policy)
   │                                   └─ Calibration (temperature/reliability)
   ▼
Observation(modality, vector, t_capture, t_arrival, quality, valid)
   │
   ▼
Entity slots / Relations / Memory / Planning
```

三条边界：

1. **Adapter** 负责“新硬件 → 契约字段”，不改核心。
2. **Modality Registry** 负责“这个模态是什么、怎么校验、缺了怎么办”。
3. **Core** 只依赖注册表，不依赖具体模态名。

---

## 4. 工作流

### W0 — 基线与金标准（3 PD）

**目标**：在重构前冻结“当前 3 模态行为”，保证后续改造可回归。

任务：

1. 盘点所有 `MODALITIES` 硬编码（core / demos / eval），产出清单。
2. 生成金标准：
   - 固定 seed、固定数据子集，保存当前 `PrototypeBank` 原型、`JointReadout` 分数、`recognize` top-k。
   - 保存为 `tests/fixtures/golden_3modality.npz` + 哈希。
3. 增加 `tests/test_golden_contract.py`：重构前后输出 `np.allclose`（容差 1e-6）。
4. 记录当前版本：`SPACE_VERSION=0.1.1`、bank/projection/readout 文件哈希。

交付物：审计清单、金标准 fixture、金标准测试。

验收：

- `pytest tests/test_golden_contract.py -q` 通过。
- 任意重构后金标准差异为 0。

依赖：无。

---

### W1 — 模态注册表（5 PD）

**目标**：把 `MODALITIES` 从代码常量变成数据驱动注册表。

任务：

1. 新增 `contracts/modalities.yaml`（与 `experts.yaml` 同目录），字段见 §5.1。
2. 新增 `pss/contracts/modality.py`：
   - `ModalitySpec` dataclass；
   - `load_modalities(path) -> dict[str, ModalitySpec]`；
   - `default_modalities()` / `resolve_modalities(names)`；
   - 校验：必填字段、未知字段告警、`status` 枚举、`normalize` 枚举。
3. `pss/contracts/schema.py`：
   - 保留 `MODALITIES = (...) ` 作为**派生兼容常量**（从注册表默认值生成），标记 `DeprecationWarning`；
   - 新增 `get_modalities()` 作为唯一真源。
4. 新增 `tests/test_modality_registry.py`：
   - 加载默认 YAML；
   - 重复名报错；
   - 缺字段报错；
   - 未知模态名在 `resolve_modalities` 中报错。

交付物：YAML、`modality.py`、兼容 shim、测试。

验收：

- 新增一条 `audio`（`status: experimental`）后，`get_modalities()` 能看到它；
- 核心代码不再直接读 `MODALITIES`（用 `rg "MODALITIES" pss/contracts pss/prototypes pss/inference.py` 验证）；
- 金标准不变。

依赖：W0。

---

### W2 — 部分模态原型库（5 PD）

**目标**：允许某些类别缺少某些模态，按可用模态建原型。

任务：

1. `build_bank` 改为按 `(modality, category)` 收集样本：
   - 某类某模态无样本时，不抛错，记入 `coverage`；
   - 新增 `missing_policy`：`skip`（默认）/ `impute`（用类均值，需显式开启）/ `raise`。
2. `PrototypeBank` 增加字段：
   - `coverage: dict[str, dict[str, int]]`（modality → category → sample_count）；
   - `missing: dict[str, list[str]]`（modality → 缺失类别）；
   - `modality_versions: dict[str, str]`。
3. `recognize`：
   - 只对“该类别在该模态有原型”的模态计分；
   - 缺失模态不参与冲突检测；
   - 权重从注册表 `reliability_prior` 或校准记录读取，不再硬编码。
4. 热更新：
   - `add_class` / `update_class` 支持“只更新部分模态”；
   - 缺模态的类别标记 `partial=True`。
5. 新增 `tests/test_partial_modality_bank.py`：
   - 构造 3 组独立模态批次（类别覆盖不同）；
   - 建 bank、单模态/双模态/三模态识别；
   - 旧 bank 加载兼容。

交付物：`build_bank` 改造、`PrototypeBank` 字段、`recognize` 改造、测试。

验收：

- 某类只有 vision 样本时，bank 构建成功，tactile/geometry 记缺失；
- 只给 tactile 查询时，不因缺失 vision 原型而崩溃；
- 金标准不变。

依赖：W1。

---

### W3 — 动态联合读出（4 PD）

**目标**：`JointReadout` 不再假设“所有模态的所有子集都训练”。

任务：

1. `JointReadout.fit(..., mask_strategy=...)`，支持：
   - `all`：当前行为，适合 ≤3 模态；
   - `observed`：只训练训练集中实际出现的 mask；
   - `single_plus_all`：单模态 + 全模态，其他走回退（推荐默认，4+ 模态）；
   - `max_size=k`：限制 mask 大小。
2. 持久化 `mask_strategy`、`trained_masks`、`modality_versions`。
3. `_model_mask` 回退逻辑保留：查询可用模态不在已训练 mask 时，选最大可用的已训练子集。
4. 新增 `fit_partial`：支持“分开批次 + 类质心行”的近似训练（每类一行，拼接各模态类均值），并标记 `approximate=True`。
5. 新增 `tests/test_joint_readout_masks.py`：
   - 3 模态金标准不变；
   - 4 个 dummy 模态下 `all` = 15 个 mask，`single_plus_all` = 5 个；
   - 查询缺失模态时回退正确。

交付物：mask 策略、持久化字段、`fit_partial`、测试。

验收：

- 4 模态 `single_plus_all` 训练时间 < `all` 的 1/2；
- 回退结果与“训练过的最大子集”一致；
- 旧 `joint_readout.npz` 可加载（默认 `mask_strategy=all`）。

依赖：W1。

---

### W4 — 动态空间与感知构造（3 PD）

**目标**：`PublicSemanticSpace` / `PerceptionService` 按注册表构建，缺文件跳过并告警。

任务：

1. `PublicSemanticSpace.__init__`：
   - 从 `get_modalities()` 读取 active 模态；
   - 逐个尝试加载 `projection_dir/{m}.pt`，缺失则跳过并记录 `inactive_modalities`；
   - `weights` 从注册表/校准记录构建，不再硬编码。
2. `PerceptionService.describe()` 增加：
   - `active_modalities`、`inactive_modalities`、每个模态的 `version`、`calibration`。
3. `recognize` 的 `weights` 只对 query 中存在的模态取默认值。
4. 新增 `tests/test_dynamic_space.py`：
   - 加一个 dummy `audio` 模态 + 随机投影头，推理可运行；
   - 删除 tactile 投影文件，空间仍可用，`describe()` 标记 inactive；
   - 金标准不变。

交付物：动态构造、描述字段、测试。

验收：

- 新模态只加 YAML + 投影文件即可被识别；
- 缺文件不抛异常，降级可解释；
- 现有 3 模态结果不变。

依赖：W1、W2。

---

### W5 — 时间与异步契约（4 PD）

**目标**：把“时间戳、延迟、窗口、采样率”变成 Observation 的一等字段。

任务：

1. 新增 `ModalityObservation`（§5.4），`Observation` 增加：
   - `streams: dict[str, ModalityObservation]`；
   - `t_ref`（参考帧时间）、`time_skew_ms`；
   - 兼容属性：`raw` / `confidence` 仍可读（从 streams 派生）。
2. 时间校验：
   - `max_latency_ms` 来自注册表；
   - `|t_capture - t_ref| > max_latency_ms` → `valid=False`、`stale=True`；
   - `t_arrival < t_capture` → 报错（时钟域不一致）。
3. 绑定/融合：
   - 只用 `valid=True` 的流；
   - 冲突检测排除 stale 流；
   - `describe()` 报告 `stale_modalities`。
4. 新增 `tests/test_temporal_contract.py`：
   - 同步流：行为不变；
   - 触觉延迟 50/200/1000 ms：正确标记 stale；
   - 时钟倒退：报错；
   - 单模态缺失：降级。

交付物：Observation 扩展、时间校验、测试。

验收：

- 默认同步数据金标准不变；
- 异步注入不崩溃，且 stale 可解释；
- 运行时不因 stale 流改变实体 ID。

依赖：W1。

---

### W6 — 统一版本、迁移与回滚（4 PD）

**目标**：所有 artifact 带 `space_version` / `modality_versions`，不兼容时给出明确迁移路径。

任务：

1. 统一 artifact manifest（§5.5），写入：
   - `PrototypeBank`、`JointReadout`、`ProjectionHead`、`ModalityCalibration`。
2. 新增 `pss/contracts/versioning.py`：
   - `assert_compatible(artifact, space_version)`；
   - 版本规则：
     - **major**：`PUBLIC_DIM`、归一化、绑定算子、`SLOT_COUNT` → 旧向量作废；
     - **minor**：新增模态、新增专家、新增投影头 → 向后兼容；
     - **patch**：阈值/校准/元数据 → 不失效。
3. 新增 `pss/runtime/migrate.py`：
   - `migrate_bank` / `migrate_readout` / `migrate_projection`；
   - 只做“元数据补全 + 可安全重算”的迁移；维度/归一化不兼容时拒绝并给命令。
4. 回滚：
   - artifact registry 记录 checksum + 时间；
   - `rollback_artifact(name, version)`。
5. 新增 `tests/test_artifact_versioning.py`：
   - 0.1.1 旧 artifact 加载；
   - 0.2.0 新模态 artifact 加载；
   - 维度不匹配 → 明确报错；
   - 回滚可用。

交付物：manifest、versioning、migrate、测试。

验收：

- 旧 artifact 100% 可加载；
- 不兼容错误包含“为什么”和“怎么迁移”；
- 回滚后金标准恢复。

依赖：W1、W4。

---

### W7 — 契约测试与最小种子集（8 PD）

**目标**：把兼容性变成 CI 可执行的测试，而不是口头承诺。

任务：

1. `tests/test_modality_contract.py`：
   - 每个注册模态：形状、归一化、有限性、零向量、NaN/Inf、错误维度；
   - 批量/单样本一致性。
2. `tests/test_compatibility_matrix.py`（§6）：
   - 缺失、异步、量纲漂移、饱和、噪声、新模态、旧 artifact；
   - 每项断言“行为符合预期”，不是“不崩溃”而已。
3. 合成扰动生成器：
   - `pss/eval/scripts/gen_modality_perturbations.py`；
   - 固定 seed，输出 fixture。
4. 最小真实种子集（nightly，不进 CI）：
   - 30 物体 × 3 模态 × 2 会话 ≈ 1k 样本；
   - 数据卡：传感器、标定版本、采集日期、许可；
   - 只用于回归，不参与训练。
5. 文档：
   - `docs/plans/PSS-模态上线检查清单.md`（onboarding checklist）；
   - 更新 `contracts/README.md` 的版本规则。

交付物：测试、扰动生成器、种子集数据卡、上线清单。

验收：

- 新模态 onboarding 只改 YAML + adapter + 投影 + 测试；
- 兼容性矩阵 100% 通过；
- CI 时间增加 < 5 分钟。

依赖：W1–W6。

---

## 5. 关键接口设计

### 5.1 `contracts/modalities.yaml`

```yaml
schema_version: "0.2.0"
modalities:
  vision:
    kind: dense_rgb
    input_schema: {rgb: [3, 64, 64]}
    native_dim: 256
    preprocess: imagenet_normalize_v1
    normalize: l2
    time_unit: frame
    max_latency_ms: 50
    reliability_prior: 0.90
    missing_policy: skip
    calibration: temperature
    capabilities: [dense, object_crop]
    status: active
    version: "0.1.0"
  tactile:
    kind: contact_grid
    input_schema: {contact_grid: [8, 8]}
    native_dim: 64
    preprocess: contact_grid_v1
    normalize: l2
    time_unit: frame
    max_latency_ms: 20
    reliability_prior: 0.50
    missing_policy: skip
    calibration: temperature
    capabilities: [dense, contact]
    status: active
    version: "0.1.0"
  audio:
    kind: waveform
    input_schema: {waveform: [16000]}
    native_dim: 128
    preprocess: logmel_v1
    normalize: standardize
    time_unit: ms
    max_latency_ms: 100
    reliability_prior: 0.60
    missing_policy: skip
    calibration: temperature
    capabilities: [temporal]
    status: experimental
    version: "0.0.1"
```

> 说明：`schema_version` 是注册表自身格式的版本；`SPACE_VERSION` 在首次新增模态时由 `0.1.1 → 0.2.0`（minor，旧向量仍有效）。二者不要混用。

### 5.2 `ModalitySpec`

```python
@dataclass(frozen=True)
class ModalitySpec:
    name: str
    kind: str
    input_schema: dict
    native_dim: int | None
    preprocess: str
    normalize: Literal["l2", "none", "standardize"]
    time_unit: Literal["frame", "ms", "s"]
    max_latency_ms: float
    reliability_prior: float
    missing_policy: Literal["skip", "impute", "raise"]
    calibration: Literal["temperature", "none"]
    capabilities: tuple[str, ...]
    status: Literal["active", "experimental", "retired"]
    version: str
```

### 5.3 `PrototypeBank` 覆盖字段

```python
@dataclass
class PrototypeBank:
    categories: list[str]
    protos: dict[str, np.ndarray]          # modality -> (C, 128)
    stats: dict
    coverage: dict[str, dict[str, int]]    # modality -> category -> n
    missing: dict[str, list[str]]          # modality -> categories
    modality_versions: dict[str, str]
    version: str
```

### 5.4 `Observation` 时间字段

```python
@dataclass
class ModalityObservation:
    modality: str
    vector: np.ndarray
    t_capture: float
    t_arrival: float
    clock: str = "monotonic"
    sensor_id: str = ""
    valid: bool = True
    stale: bool = False
    quality: float = 1.0

@dataclass
class Observation:
    ...
    t_ref: float = 0.0
    streams: dict[str, ModalityObservation] = field(default_factory=dict)
    time_skew_ms: dict[str, float] = field(default_factory=dict)
    # 兼容属性：raw/confidence 从 streams 派生
```

### 5.5 artifact manifest

```yaml
artifact: public_space_v1
space_version: "0.2.0"
public_dim: 128
modalities: [vision, tactile, geometry]
projection_versions: {vision: "0.1.0", tactile: "0.1.0", geometry: "0.1.0"}
bank_version: "public_space_v1"
readout_version: "joint_readout_v1"
created_at: "2026-11-28T00:00:00Z"
checksums: {bank: "...", readout: "..."}
```

---

## 6. 兼容性矩阵

| # | 扰动 | 输入 | 预期行为 | 断言 |
|---|---|---|---|---|
| 1 | 缺失单模态 | 只有 vision | 用单模态原型/读出；不抛异常 | `recognize` 返回 top-k，`modalities=["vision"]` |
| 2 | 缺失多模态 | 只有 tactile | 同上；冲突检测跳过缺失 | `conflict=False` |
| 3 | 异步 | tactile 延迟 50 ms | 在预算内，正常融合 | `stale=False` |
| 4 | 异步超限 | tactile 延迟 1000 ms | 标记 stale，不参与融合 | `stale_modalities=["tactile"]` |
| 5 | 时钟倒退 | `t_arrival < t_capture` | 报契约错误 | `ContractError` |
| 6 | 量纲漂移 | 特征 ×0.5 / ×2 | 按校准/标准化处理，或明确拒绝 | 分数变化可解释，不 NaN |
| 7 | 饱和 | 全为最大值 | 有限；标记 `saturated` | 无 NaN/Inf |
| 8 | NaN/Inf | 注入 NaN | 拒绝并报错 | `ContractError` |
| 9 | 错误维度 | 127d | 拒绝并报错 | `dim` 期望 128 |
| 10 | 零向量 | 全 0 | 视为缺失，不参与 | 不崩溃 |
| 11 | 新模态 | dummy `audio` | 只加 YAML + adapter + 投影即可 | 识别可用；旧模态结果不变 |
| 12 | 旧 artifact | 0.1.1 bank/readout | 可加载 | 金标准一致 |
| 13 | 版本不兼容 | 0.1.1 → 1.0.0（维度/归一化变更） | 明确拒绝 + 迁移命令 | 错误含迁移提示 |

---

## 7. 版本与迁移策略

| 变更 | 版本 | 后果 | 迁移 |
|---|---|---|---|
| `PUBLIC_DIM` / 归一化 / 绑定 / `SLOT_COUNT` | **major** | 全部历史向量作废 | 全量重算，不能在线迁移 |
| 新增模态 / 新专家 / 新投影头 | **minor** | 旧向量仍有效 | 加 manifest；旧 artifact 可加载 |
| 阈值 / 校准 / 元数据 | **patch** | 不失效 | 直接覆盖 |

规则：

1. 任何 artifact 写入 `space_version` 与 `modality_versions`。
2. `assert_compatible` 在加载时执行；不兼容报错必须包含：
   - 当前版本 vs artifact 版本；
   - 不兼容字段；
   - 迁移命令或“需重建”的说明。
3. 回滚以 artifact 为单位，不以代码为单位。
4. `SPACE_VERSION` 的变更记录写在 `contracts/README.md`。

---

## 8. 里程碑与排期

| 里程碑 | 周 | 内容 | 交付 |
|---|---|---|---|
| M0 | W1 | W0 基线 + W1 注册表 | 金标准、YAML、`modality.py` |
| M1 | W2–W3 | W2 部分模态 bank | 覆盖字段、识别降级 |
| M2 | W4 | W3 动态读出 | mask 策略、回退 |
| M3 | W5 | W4 动态空间 + W5 时间契约 | 动态构造、stale 处理 |
| M4 | W6 | W6 版本/迁移/回滚 | manifest、migrate、rollback |
| M5 | W7–W8 | W7 契约测试 + 种子集 + 文档 | CI 矩阵、上线清单 |

人力估算：

| 工作流 | 人日 |
|---|---:|
| W0 | 3 |
| W1 | 5 |
| W2 | 5 |
| W3 | 4 |
| W4 | 3 |
| W5 | 4 |
| W6 | 4 |
| W7 | 8 |
| **合计** | **36** |

---

## 9. 风险与缓解

| 风险 | 影响 | 缓解 |
|---|---|---|
| 过度抽象 | 核心读出差、没人能验证 | 核心只依赖注册表；每模态只加插件；不做未知模态的预建 |
| 金标准漂移 | 重构悄悄改变结果 | 固定 seed/fixture；CI 逐位比较 |
| 动态 mask 性能 | 训练/推理变慢 | `single_plus_all` 默认；按需训练；缓存 |
| demos/eval 硬编码 | 重构后不一致 | `MODALITIES` 派生 shim；本轮不强制迁移 |
| 时间契约改动大 | runtime/tracker 回归 | 字段可选、默认同步；分阶段启用 |
| 数据许可 | 种子集不能发布 | 内部使用；记录 license；不进公开包 |
| 版本迁移复杂 | 旧 artifact 不可用 | 先支持“只读加载”，再做迁移工具 |

---

## 10. Definition of Done

- [ ] 新模态 onboarding 只需：YAML + adapter + 投影头 + 测试；核心零改动。
- [ ] 3 模态金标准逐位一致。
- [ ] 部分模态 bank/recognize 可用；缺失不抛异常。
- [ ] `JointReadout` 支持 `single_plus_all` / `observed`，旧 artifact 可加载。
- [ ] `Observation` 有时间戳/stale 契约；异步注入可解释。
- [ ] 所有 artifact 带 `space_version`；不兼容错误含迁移路径。
- [ ] 兼容性矩阵 13 项全绿。
- [ ] 现有测试全绿；CI 增量 < 5 分钟。
- [ ] 文档更新：`contracts/README.md`、`docs/plans/PSS-模态上线检查清单.md`、本计划的状态。

---

## 11. 非目标

- 不修改 `PUBLIC_DIM`、归一化、Hadamard 绑定、`SLOT_COUNT`。
- 不重写 demos/eval 的全部 `MODALITIES` 用法（只加兼容 shim）。
- 不为尚未出现的模态预建复杂抽象。
- 不做大规模真实数据采集；只做最小种子集。
- 不改变现有识别/开集/热更新的算法结论。

---

## 12. 附录

### 12.1 涉及文件

- `pss/contracts/schema.py`
- `pss/contracts/modality.py`（新增）
- `contracts/modalities.yaml`（新增）
- `pss/prototypes/build.py`
- `pss/prototypes/bank.py`
- `pss/prototypes/joint_readout.py`
- `pss/inference.py`
- `pss/memory/perception.py`
- `pss/runtime/perception.py`
- `pss/runtime/runtime.py`
- `pss/projection/heads.py`
- `pss/projection/calibration.py`
- `pss/contracts/versioning.py`（新增）
- `pss/runtime/migrate.py`（新增）
- `tests/test_golden_contract.py`（新增）
- `tests/test_modality_registry.py`（新增）
- `tests/test_partial_modality_bank.py`（新增）
- `tests/test_joint_readout_masks.py`（新增）
- `tests/test_dynamic_space.py`（新增）
- `tests/test_temporal_contract.py`（新增）
- `tests/test_artifact_versioning.py`（新增）
- `tests/test_compatibility_matrix.py`（新增）
- `docs/plans/PSS-模态上线检查清单.md`（新增）

### 12.2 快速验证命令

```bash
python -m pytest tests/test_golden_contract.py \
  tests/test_modality_registry.py \
  tests/test_partial_modality_bank.py \
  tests/test_joint_readout_masks.py -q

python -m pytest tests/test_temporal_contract.py \
  tests/test_artifact_versioning.py \
  tests/test_compatibility_matrix.py -q
```

### 12.3 新模态上线检查清单（摘要）

- [ ] 在 `contracts/modalities.yaml` 登记（含 `native_dim`、`max_latency_ms`、`missing_policy`）
- [ ] 实现 adapter（`encode`）
- [ ] 训练/加载投影头（128d）
- [ ] 准备校准集（temperature/reliability）
- [ ] 通过 `tests/test_modality_contract.py`
- [ ] 通过兼容性矩阵（至少 1/2/3/4 项）
- [ ] 更新 artifact manifest
- [ ] 更新 `describe()` 与文档
- [ ] 记录许可与数据卡

---

## 13. 实现状态（2026-11-28）

| 工作流 | 状态 | 主要交付 |
|---|---|---|
| W0 基线 | 完成 | `pss/eval/scripts/freeze_golden_contract.py`、`tests/fixtures/golden_3modality.npz`、`tests/test_golden_contract.py` |
| W1 注册表 | 完成 | `contracts/modalities.yaml`、`pss/contracts/modality.py`、`MODALITIES` 派生、`tests/test_modality_registry.py`、`tests/test_modality_contract.py` |
| W2 部分模态 | 完成 | `build_bank` 支持 `train_idx` 字典与 `coverage/missing/imputed`、`recognize` 按可用模态融合、`tests/test_partial_modality_bank.py` |
| W3 动态读出 | 完成 | `all/single_plus_all/max_size/observed` mask 策略、`fit_class_centroids`、部分模态热更新、`tests/test_joint_readout_masks.py` |
| W4 动态空间 | 完成 | `load_available_heads`、动态权重/`describe()`、`PerceptionService.describe()`、`tests/test_dynamic_space.py` |
| W5 时间契约 | 完成（契约层） | `ModalityObservation`、`Observation.add_stream/refresh_staleness/valid_raw`、tracking 使用有效流、`tests/test_temporal_contract.py` |
| W6 版本迁移 | 完成 | `pss/contracts/versioning.py`、artifact `space_version`、`pss/runtime/migrate.py`、`pss/runtime/artifacts.py`、`tests/test_artifact_versioning.py` |
| W7 契约测试 | 完成（合成） | 13 项兼容性矩阵、模态契约测试、`docs/plans/PSS-模态上线检查清单.md` |

**尚未完成**：

- 真实种子集（30 物体 × 3 模态 × 2 会话）需要硬件采集；当前矩阵是合成扰动。
- 真实感知生产者尚未填充 `t_capture/t_arrival`；契约已就绪，需在 `PerceptionService.extract_frame` 接入。
- 首次真正新增模态时，`SPACE_VERSION` 从 `0.1.1` 升到 `0.2.0`；本轮未改版本号。
- `pss/demos/`、`pss/eval/scripts/` 仍使用兼容常量 `MODALITIES`，未强制迁移。
- `ArtifactRegistry` 已可用，但尚未接入所有加载路径；当前是显式调用。

**验证命令**：

```bash
python -m pytest tests/test_golden_contract.py tests/test_modality_registry.py \
  tests/test_modality_contract.py tests/test_partial_modality_bank.py \
  tests/test_joint_readout_masks.py tests/test_dynamic_space.py \
  tests/test_temporal_contract.py tests/test_artifact_versioning.py \
  tests/test_compatibility_matrix.py -q
```

---

## 14. 变更记录

| 日期 | 变更 |
|---|---|
| 2026-11-28 | 初稿：W0–W7、兼容性矩阵、版本策略、排期 |
