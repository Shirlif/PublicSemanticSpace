# L2 复现包

> **日期**：2026-11-26
> **范围**：公开的 L2 bundle replay 证据（E12）如何下载、校验、复现。
> **相关**：`scripts/pack_repro_bundles.py`、`scripts/fetch_repro_bundles.py`、`scripts/reproduce_l2.py`、`reports/repro/REPRODUCTION.md`。

---

## 1. 为什么需要单独的复现包

`.gitignore` 排除了 `bundles/`、`data/`、`checkpoints/`、`_external/`。一个干净的 clone 只有代码、spec、scorecard 和报告，**没有 L2 bundle**，因此无法直接复现 E12 的 bundle replay 数字。

复现包把 E12 需要的输入表单独发布为 release asset，不提交进 git：

- 33 个 bundle；
- 约 5.5 MB（压缩后）；
- 4 个 bundle 带 GroundingDINO / DINOv2 embedding；
- 每个文件带 SHA256，包内带 `PACK_MANIFEST.json`。

## 2. 下载与校验

```bash
python scripts/fetch_repro_bundles.py \
  --url https://github.com/Shirlif/PublicSemanticSpace/releases/download/v0.1.1/pss-l2-bundles-v0.1.1.tar.gz \
  --sha256 9bd50f418d21e3372c3b77bdb300e9ec73926979a24b8cf70e2f324c34c7984e \
  --out bundles/repro
```

也可以直接指定本地归档：

```bash
python scripts/fetch_repro_bundles.py \
  --archive dist/repro/pss-l2-bundles-v0.1.1.tar.gz \
  --sha256 9bd50f418d21e3372c3b77bdb300e9ec73926979a24b8cf70e2f324c34c7984e \
  --out bundles/repro
```

脚本会：

1. 校验归档 SHA256；
2. 拒绝绝对路径、`..`、符号链接和设备文件；
3. 校验 `PACK_MANIFEST.json` 中每个文件的大小和 SHA256；
4. 解压到 `bundles/repro/pss-bundles-v0.1.1/`。

当前归档：

| 字段 | 值 |
|---|---|
| 归档 | `pss-l2-bundles-v0.1.1.tar.gz` |
| 大小 | 5,463,441 bytes |
| SHA256 | `9bd50f418d21e3372c3b77bdb300e9ec73926979a24b8cf70e2f324c34c7984e` |
| 文件数 | 244 |
| bundle 数 | 33 |
| 带 embedding 的 bundle | 4 |

同一份清单也写在：

- `docs/repro/PSS_BUNDLES_v0.1.1.json`
- `docs/repro/SHA256SUMS`

## 3. 复现

```bash
make reproduce-l2
# 等价于
python scripts/reproduce_l2.py --root bundles/repro --out reports/repro
```

输出：

| 文件 | 内容 |
|---|---|
| `reports/repro/REPRODUCTION.md` | 汇总报告与 PASS/FAIL |
| `reports/repro/REPRODUCED.json` | 机器可读的复现状态 |
| `reports/repro/l2_memory/REPORT.md` | E12 全量 bundle 表 |
| `reports/repro/l2_perception/REPORT.md` | 4 个带 embedding bundle 的表 |

期望结果（valid bundles，19/33）：

| variant | n | ID retention | switches |
|---|---:|---:|---:|
| full | 19 | 1.000 | 0.00 |
| no_memory | 19 | 0.827 | 0.56 |
| no_entity | 19 | 0.270 | 47.47 |
| no_relations | 19 | 1.000 | 0.00 |

脚本会按报告精度比较（retention 3 位小数，switches 2 位小数），不一致时退出码非 0。

## 4. 包内容与排除项

### 包含

每个 bundle 的输入表：

- `manifest.json`
- `obs_frames.parquet`
- `obs_detections.parquet`
- `gt.parquet`
- `robot_state.parquet`
- `actions.parquet`
- `events.parquet`
- `detect_meta.json`（如果有）

### 默认排除

| 内容 | 原因 |
|---|---|
| `images/` | replay 只读 parquet；排除渲染帧可减小体积，并避免再分发 Isaac Sim 渲染资产 |
| `entity_trace.parquet` | replay 派生输出，运行时会重新生成 |
| `pred_states.parquet` | 同上 |
| `pred_relations.parquet` | 同上 |
| `pss_trace.parquet` | 同上 |

如果需要重新跑检测（GroundingDINO），用 `--include-images` 重新打包：

```bash
python scripts/pack_repro_bundles.py --root bundles/dev --out dist/repro --include-images
```

## 5. 4 个带 embedding 的 bundle 是 null result

`reports/repro/l2_perception/REPORT.md` 会显示：

| variant | n | ID retention |
|---|---:|---:|
| full | 4 | 1.000 |
| no_memory | 4 | 1.000 |
| no_entity | 4 | 1.000 |
| no_relations | 4 | 1.000 |

这不是复现失败。这 4 个 bundle（`s1_hard`、`s1_occlusion_real`、`s2_twins`、`s1_camera_real`）检测密集，记忆消融没有信号；它们证明的是 **detector → runtime 路径可用**，不是记忆层有贡献。记忆信号来自全量集合里的 occlusion / oracle bundle。

这也说明下一步 F2（感知级 mid-stream expert swap）需要**设计新的感知 bundle**，而不是直接复用这 4 个。

## 6. 维护者：重新打包与发布

```bash
make repro-pack
# 等价于
python scripts/pack_repro_bundles.py --root bundles/dev --out dist/repro --version 0.1.1
```

产物：

- `dist/repro/pss-l2-bundles-v0.1.1.tar.gz`（gitignored）
- `dist/repro/PSS_BUNDLES_v0.1.1.json`
- `dist/repro/SHA256SUMS`
- 同时写入 `docs/repro/`，供仓库引用

发布记录：

- **v0.1.1** 已发布：<https://github.com/Shirlif/PublicSemanticSpace/releases/tag/v0.1.1>，
  asset 为 `pss-l2-bundles-v0.1.1.tar.gz`。

后续版本的发布步骤（需要维护者手动执行）：

1. 把 `dist/repro/pss-l2-bundles-vX.Y.Z.tar.gz` 上传到 GitHub Release 或 Zenodo；
2. 用上传后的 URL 更新 `docs/reproducibility.md` 和 release notes；
3. 确认 `docs/repro/SHA256SUMS` 与上传文件一致；
4. 如果版本变化，更新 `docs/repro/PSS_BUNDLES_vX.Y.Z.json` 和本页数字。

不要把 bundle 提交进 git；`bundles/` 在 `.gitignore` 中是有意为之。

## 7. 已知边界

- 复现包只覆盖 **E12 bundle replay**，不覆盖 S4-PSS fresh Isaac 运行（需要 Isaac Sim 环境）。
- 4 个感知 bundle 的记忆消融是 null result，见 §5。
- 图像默认不包含；需要重跑检测时用 `--include-images`。
- 包内 `manifest.json` 保留原始生成时的 `pss_commit`；它可能早于当前 HEAD，这是正常的来源记录。
