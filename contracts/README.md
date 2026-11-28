# 公共语义空间 · 接口契约 v0

> 版本：`0.1.1`
> 状态：Phase 1 冻结。改动维度、归一化、绑定算子属于**大版本变更**，会让全部历史实体向量作废（原文 9.1.1）。

## 1. 公共语义空间

| 项 | 值 |
|---|---|
| 维度 `PUBLIC_DIM` | 128（基准）；256 / 512 作对照 |
| 数据类型 | `float64`（训练/推理内部可降为 `float32`，需单独评估） |
| 归一化 | 每模态原始公共向量 L2 归一化为单位范数 |
| 槽数 `SLOT_COUNT` | 4 |
| 槽基向量 | Hadamard 矩阵行，单位化后分量恒为 `±1/√D`，两两正交 |
| 绑定方案 `BINDING_SCHEME` | `hadamard`（HRR 仅作审计备选，未采用） |

槽 = **工作记忆位置**，不是实体身份。实体身份由场景图分配并持久化（原文 7.3）。

## 2. 绑定算子

```
bind(f, s)        = f ⊙ s
unbind(b, s)      = b / safe(s)
safe(s)_i         = sign(s_i) · max(|s_i|, eps)
```

- `eps = 1e-6` 是默认值；Hadamard / Rademacher 槽的分量绝对值恒为 `1/√D`，不会触发保护。
- 原 Demo 使用 `np.where(|s|<eps, eps, s)`，会把近零**负**分量翻成正的，造成符号错误；v0 已改为保号 clamp。
- 单分量可逆：`unbind(bind(f,s),s) = f`。
- 同槽多分量相加**不可分离**：`unbind(bind(a,s)+bind(b,s),s) = a+b`，无法反推 `a`、`b`。

## 3. 实体（Entity）

```python
@dataclass
class Entity:
    id: str
    pose: np.ndarray | None           # (7,) xyz+quat 或 (6,)
    slot: int | None                  # bound 向量当前绑定的工作记忆槽
    bound: dict[str, np.ndarray]      # modality -> bound vector（分模态，禁止相加）
    raw: dict[str, np.ndarray]        # modality -> 未绑定公共特征（L2 归一）
    conf: dict[str, float]            # modality -> 置信度 [0,1]
    prototype: str | None
    state: dict
    parent: str | None                # 碎片溯源
    provenance: dict
    last_seen: float
    status: active | dormant | fragmented | missing | historical
```

规则：

1. `bound` 与 `raw` 的每个向量维度必须是 `PUBLIC_DIM`。
2. `bound` 是**工作记忆的临时表示**，只在 `slot` 指定的槽下有效；持久表示是 `raw` + `pose`。实体换槽时必须用 `rebuild_for_slot` 从 `raw` 重新绑定，**不能**拿旧 `bound` 配新槽解释。
3. `bound` 中**不得**存放跨模态相加的结果。跨模态融合只能通过置信度加权匹配、关系推理或世界模型完成。
4. `conf` 缺省视为 0；下游融合时缺省模态不参与。
5. 实体状态变化（完整 → 破碎）只改 `state`/`status`，不换 `id`；碎片是新的实体，`parent` 指回原实体（原文 7.4）。

## 4. 关系（Relation）

```python
@dataclass
class Relation:
    subject: str
    predicate: str
    object: str
    confidence: float
    bound_parts: dict[str, np.ndarray]   # {"subj":..., "rel":..., "obj":...}
    fact: np.ndarray | None              # 仅供快速检索的求和版
```

- `bound_parts` 必须同时包含 `subj` / `rel` / `obj` 三个分量，每个分量独立 bind 到对应角色槽。
- `fact` 是三个分量的和，**只用于相似度检索**；需要拆分量时必须用 `bound_parts`，不能从 `fact` 反推。
- 关系置信度参与规划器的软前置条件判断；不满足时可拒绝执行并报告阻塞（原文 6.7）。

## 5. 原型节点（PrototypeNode）

字段对齐原文 9.6.3：

- `node_id`、`parent_node`、`category_name`
- `modality_protos`：分模态原型向量，每个单位范数
- `sample_count`、`variance`、`confidence_threshold`、`confusable_with`
- `safety_tag`、`version`

规则：

1. 原型必须来自真实样本聚类，不能手工写数字（原文 9.6.3）。
2. 原型向量按模态分别存储；识别时按模态置信度加权匹配，不做跨模态相加。
3. 端侧对原型库**只读**；新增/更新走离线审核 + OTA（原文 9.6.1）。

## 6. 版本规则

| 变更 | 性质 | 后果 |
|---|---|---|
| 改 `PUBLIC_DIM`、归一化、绑定算子 | 外壳规约变更（大版本） | 全部历史实体向量作废，需全量迁移 |
| 换某模态专家 + 其投影头 | 映射函数变更（小版本） | 其他模态、原型库、下游不动（原文 9.1.1） |
| 新增原型包 | 数据更新 | 不触发反向传播，可单独回滚 |

当前契约版本：`SPACE_VERSION = "0.1.1"`。

### 变更记录

| 版本 | 变更 |
|---|---|
| 0.1.0 | Phase 0：维度、归一化、Hadamard 槽、绑定算子、Entity/Relation/PrototypeNode |
| 0.1.1 | Phase 1：Entity 增加 `slot`；明确 `bound` 为临时表示、`raw` 为持久表示；新增 `BINDING_SCHEME`；HRR 对照后确认保留 Hadamard |

## 7. 容量与 rebind 预算（Phase 0 实测）

来自 `bench/binding_stability.py`（完整网格见 `bench/results/binding_stability.md`）：

| 项 | 实测 | 契约约束 |
|---|---|---|
| D=128，K=5 叠加检索 | top-1 = 1.000 | 单向量叠加建议 ≤ 8 个分量 |
| D=128，K=20 叠加检索 | top-1 ≈ 0.77 | 超过 8 个分量时升 D 或分开存储 |
| D=512，K=20 叠加检索 | top-1 = 1.000 | 高容量场景用 512 |
| Hadamard 往返误差 | ≈ `noise × sqrt(cycles)` | 误差达 10% 前必须 rebind |
| 1% 噪声 × 100 轮 | ≈ 10% | 约 100 轮触发 rebind |
| 5% 噪声 × 10 轮 | ≈ 16% | 约 10 轮触发 rebind |
| Gaussian 槽 K=5 容量 | top-1 ≈ 0.35–0.40 | **禁止**使用高斯槽 |

> 注意：容量测试是"所有事实共用一个向量"的**压力测试**。本架构默认的实体存储是
> 分模态、分实体存放，单个实体的叠加量≈该模态属性数（约 5 个），处于 D=128 的安全区。
> 只有把大量关系压进一个向量时才需要升 D 或改用逐条存储。

常量：`RECOMMENDED_MAX_SUPERPOSITION = 8`，`REBIND_ERROR_BUDGET = 0.10`。

## 8. Phase 1 绑定方案对照结论

完整网格见 `bench/results/phase1_binding.md`。

| 项 | Hadamard | HRR | 结论 |
|---|---:|---:|---|
| 无噪声往返误差（1 轮） | ~5e-17 | ~1.0 | Hadamard 精确 |
| 10 轮往返误差 | ~5e-17 | ~1e6 | HRR 发散 |
| 100 轮往返误差 | ~5e-17 | ~1e80 | HRR 不可用于迭代 |
| 容量 D=128 K=20 | 0.775 | 0.667 | 无一致优势 |
| 容量 D=256 K=20 | 0.925 | 0.958 | 互有胜负 |
| 容量 D=512 K=50 | 0.833 | 0.790 | 无一致优势 |
| 跨角色串扰（D=128，max） | 0.153 | 0.310 | Hadamard 更低 |

**决策：保留 Hadamard。** 原文 6.1 说"循环卷积容量更大但只是近似可逆"；实测进一步表明，
HRR 不仅近似，而且**迭代下误差指数发散**，不能用于 rebind 循环。HRR 保留在
`pss/binding/hrr.py` 作为审计对照，不作为默认方案。

### 关系三元组的实测边界

| 解码方式 | 单条关系相对误差 | 20 条关系叠加后 |
|---|---:|---:|
| 独立 `parts` | ~5e-17（精确） | ~5e-17（逐条） |
| 单个 `fact` | ~1.41 | — |
| 叠加 `memory` | — | ~7.9 |

因此：`fact` 只能用于快速相似度检索；任何需要取回具体角色的场景都必须读 `parts`。

## 9. 两个 YAML 的分工（2026-11-28）

| 文件 | 角色 | 谁读它 | 改了会怎样 |
|---|---|---|---|
| `contracts/modalities.yaml` | **模态真源**：有哪些模态、维度、归一化、时间预算、缺失策略 | `pss/contracts/modality.py`；`MODALITIES` 由此派生 | 新增 `active` 模态会立刻进入 `get_modalities()` |
| `contracts/experts.yaml` | **专家登记/评审清单**：候选、许可、评测记录 | 人工评审、`bench_expert_zoo` 的配置 | **不会**自动改变生产；生产仍由 `pss/encoders/registry.py` 和 `pss/experts/` 按 `kind` 构建 |

规则：

1. 新增模态：先改 `modalities.yaml`，再加 adapter、投影头、校准与契约测试。
2. 新增专家：同时改 `experts.yaml`（登记）与 `pss/encoders/registry.py` / `pss/experts/`（执行）；两者缺一视为未完成。
3. `tests/test_experts_contract.py` 会检查登记的模态是否存在、维度/许可/状态是否完整，防止 YAML 与代码漂移。

## 10. 版本与 artifact 强制（2026-11-28）

当前 `SPACE_VERSION = "0.1.1"`。首次真正新增一个 `active` 模态时升到 `0.2.0`；本轮只准备了注册表和测试，未改版本号。

| 变更 | 版本 | 后果 | 迁移 |
|---|---|---|---|
| `PUBLIC_DIM` / 归一化 / 绑定 / `SLOT_COUNT` | **major** | 全部历史向量作废 | 全量重算，不能在线迁移 |
| 新增模态 / 新专家 / 新投影头 | **minor** | 旧向量仍有效 | 旧 artifact 可加载；写 manifest |
| 阈值 / 校准 / 元数据 | **patch** | 不失效 | 直接覆盖 |

加载强制：

- `PrototypeBank.load`、`JointReadout.load`、`load_projection` 默认调用
  `assert_compatible(artifact.space_version, SPACE_VERSION)`；
- 旧复现包或迁移工具可显式传 `strict=False`；
- 不兼容错误包含 artifact 名、两个版本号和迁移提示。

工具：

- `python -m pss.runtime.migrate --kind bank|readout|projection --path ...`：只补元数据，不改向量；
- `pss.runtime.artifacts.ArtifactRegistry`：checksum 记录、版本解析与回滚。

变更记录：

| 版本 | 变更 |
|---|---|
| 0.1.2（规划） | 模态注册表、部分模态原型库、动态读出、时间契约、artifact 版本强制；`SPACE_VERSION` 仍为 0.1.1 |
| 0.2.0（规划） | 首个新增 `active` 模态启用时升级；旧 0.1.1 artifact 仍兼容 |
