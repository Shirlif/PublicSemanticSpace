# PSS 模态上线检查清单

> 配套：`docs/plans/PSS-兼容性与模态扩展任务计划.md`（W7 交付物）
> 原则：新增模态只改“注册表 + 适配器 + 投影头 + 校准 + 测试”，不改 PSS 核心。

---

## 1. 登记

- [ ] 在 `contracts/modalities.yaml` 增加一条 spec：
  - `kind`、`input_schema`、`native_dim`
  - `preprocess`、`normalize`
  - `time_unit`、`max_latency_ms`
  - `reliability_prior`（0–1）
  - `missing_policy`（默认 `skip`）
  - `calibration`（默认 `temperature`）
  - `capabilities`、`status`、`version`
- [ ] 若模态尚不稳定，`status: experimental`；它不会进入默认 `MODALITIES`。
- [ ] 运行 `python -c "from pss.contracts import get_modalities; print(get_modalities(include_experimental=True))"` 确认注册成功。

## 2. 适配器与专家

- [ ] 实现 `encode(x) -> (B, native_dim)`，输出有限、形状稳定。
- [ ] 在 `pss/encoders/registry.py` 或 `pss/experts/` 注册。
- [ ] `ExpertContract` 校验：`modality`、`version`、`dim`、`encode`、`health`。
- [ ] 单样本/批量一致性；空输入、零向量、NaN 的行为明确。

## 3. 投影与校准

- [ ] 训练/加载 `projection_dir/<modality>.pt`，输出 `PUBLIC_DIM=128`、L2 归一化。
- [ ] `ProjectionHead.space_version` 与当前 `SPACE_VERSION` 一致。
- [ ] 在验证集上拟合 `temperature` / `reliability`，写入校准记录。
- [ ] 校准集与训练集、测试集按实例隔离。

## 4. 原型库与读出

- [ ] `build_bank(..., train_idx={modality: indices})` 使用该模态自己的样本集。
- [ ] 缺失类别按 `missing_policy` 处理；检查 `coverage` / `missing` / `imputed`。
- [ ] `JointReadout.fit(..., mask_strategy=...)`：
  - ≤3 模态：`all`
  - ≥4 模态：`single_plus_all` 或 `max_size`
  - 分开批次：`fit_class_centroids`（标记 `approximate`）
- [ ] 单模态/双模态/全模态查询均可用；缺失模态不抛异常。

## 5. 时间契约

- [ ] 生产者填 `t_capture`、`t_arrival`、`clock`、`sensor_id`，通过
  `Observation.stamp_streams({modality: (t_capture, t_arrival)}, t_ref=...)`
  一次性写入（`extract_frame` / `PerceptionService.observe` 也接受 `timings`）。
- [ ] `max_latency_ms` 与注册表一致；超限流标记 stale。
- [ ] `Observation.valid_raw()` 只返回可用流；tracker、`EntityLayer` token、
  `pad_observations`、runtime 缺失模态检测都已默认使用它。
- [ ] 回归：stale 流不得改变实体 ID（`tests/test_temporal_contract.py`）。
- [ ] 真实驱动层用传感器时间覆盖默认的同步时间戳；不要用帧号冒充采集时间。

## 6. 版本与 artifact

- [ ] bank / readout / projection 均写入 `space_version`。
- [ ] `ArtifactManifest` 记录 `modalities`、`projection_versions`、`checksums`。
- [ ] 加载时 `assert_compatible`；不兼容错误包含迁移提示。
- [ ] `ArtifactRegistry.register` 保存可回滚副本；`rollback` 验证通过。

## 7. 契约测试

- [ ] `tests/test_modality_contract.py`（新增模态的专属用例）
- [ ] `tests/test_compatibility_matrix.py`（13 项）
- [ ] `tests/test_golden_contract.py`（3 模态金标准不变）
- [ ] `tests/test_temporal_contract.py`（同步/异步/时钟倒退）
- [ ] `tests/test_artifact_versioning.py`（旧 artifact / 版本不兼容 / 回滚）
- [ ] `tests/test_runtime_semantics_lock.py`（已知运行时语义缺口，改行为时同步更新）

## 7.1 最小种子集（回归用，不训练）

- [ ] 真实采集：30 物体 × 3 模态 × 2 会话，每格 ≥ 6 样本。
- [ ] 目录符合 `objects/<object_id>/<modality>/session-<n>.npz`，
  每格含 `features`、`timestamps_ns`、`sample_ids`。
- [ ] `manifest.json` 的 `kind` 为 `real`，并填传感器、标定版本/日期、许可。
- [ ] `python -m pss.eval.seedset validate --root <path>` 通过。
- [ ] 数据卡随种子集保存；不提交到公开仓库。
- [ ] CI 只跑 `build-synthetic`（30 × 3 × 2 = 180 格 / 1080 样本）验证布局，
  不把合成数据当真实能力证据。

## 8. 文档与发布

- [ ] 更新 `contracts/README.md` 的变更记录。
- [ ] 更新 `docs/` 的模态说明与数据卡（许可、传感器、标定版本）。
- [ ] `PerceptionService.describe()` / `PublicSemanticSpace.describe()` 能看到新模态。
- [ ] 若为 minor 变更：`SPACE_VERSION` 从 `0.1.1` → `0.2.0`；旧 artifact 仍可加载。
- [ ] 若为 major 变更：明确“旧向量作废”，走全量重建。

## 9. 失败回滚

- [ ] 保留上一个可用的 bank/readout/projection。
- [ ] `ArtifactRegistry.rollback(name, version)` 可恢复。
- [ ] 回滚后跑一遍金标准与兼容性矩阵。

---

## 最小命令

```bash
# 兼容性契约门禁（快，CI 用）
make compat

# L0 属性套件（NumPy，CI 用）
make property-l0

# 种子集布局（合成；真实采集后把 --root 指向真实目录）
make seedset-synthetic
make seedset-validate
python -m pss.eval.seedset validate --root <real-seedset> --min-objects 30

# 单独跑某一组
python -m pytest tests/test_golden_contract.py \
  tests/test_modality_contract.py \
  tests/test_compatibility_matrix.py \
  tests/test_temporal_contract.py \
  tests/test_artifact_versioning.py -q
```
