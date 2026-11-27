# PSS 文档索引

> 2026-11-26 整理：根目录只保留项目入口、作者、许可证和引用元数据；设计、实验、计划、阶段报告统一放到 `docs/`。
> 本文件是文档总入口；完整阶段报告索引见 [`phases/README.md`](phases/README.md)。

---

## 1. 按主题

### 架构与定位 — `docs/architecture/`

| 文档 | 内容 |
|---|---|
| `PSS-认知层架构定位.md` | PSS 与 Transformer 的分层关系：Transformer 负责感知/语言，PSS 负责绑定、关系、记忆、规划；五优先级与能力上限 |
| `PSS-认知运行时与Harness.md` | 运行时（runtime）与 harness 的区别，什么时候 PSS 只是外壳 |
| `PSS-架构评测指标.md` | 分层指标（T0–T3）、哪些指标不属于 PSS、怎么判断一个指标该不该算 PSS 的分 |
| `公共语义空间推理架构-推演记录-11.19.md` | 最早的完整推演长文（约 235 KB）：感知、绑定、记忆、规划、与 Atlas/VLA 的对比 |
| `公共语义空间-工作总结-v1.md` | 阶段工作总结与关键结论 |

### 实验 — `docs/experiments/`

| 文档 | 内容 |
|---|---|
| `PSS_VS_TRANSFORMER_EXPERIMENT.md` | PSS vs Transformer 的实验协议、指标、杀死标准 |
| `PSS_VS_TRANSFORMER_RESULTS.md` | 六维对比结果：闭集、re-ID、开放集、增量更新、冲突、规划、scaling |
| `PSS-五优先级最终结论.md` | 五个优先级的结论，以及“明确不能声称”的清单 |
| `PSS-读出与推理优化总报告.md` | A 线（读出）与 B 线（关系/搜索）的统一结论：混合顺序、读出上限、覆盖与搜索 |

### 计划 — `docs/plans/`

| 文档 | 内容 |
|---|---|
| `公共语义空间-落地执行计划-v1.md` | 最早的落地执行计划（Phase 0–10 的来源） |
| `PSS-优化任务清单.md` | 长期优化任务清单与逐项验收记录（约 90 KB） |
| `PSS-IsaacSim-实施计划.md` | Isaac Sim 评测平台的实施计划 |
| `PSS-IsaacSim-评测平台设计.md` | Isaac Sim 平台的 spec、bundle、指标设计 |
| `PSS-后续研究方向评估.md` | 对照论文 §7–§8 与 claim-gap 证据，对外部未来方向清单做事实核对与任务优先级排序 |
| `PSS-读出与推理优化执行方案.md` | 归因 v2 之后的优化方案：A0 闸门召回、双头读出、度量蒸馏、关系头覆盖与配对基线；含 P0 实测结果 |

### 阶段报告 — `docs/phases/`

按 Phase 顺序归档的工程报告，共 30+ 篇。完整索引见 [`phases/README.md`](phases/README.md)。

### 背景 — `docs/background/`

| 文档 | 内容 |
|---|---|
| `从婴儿的感官到抽象符号-11.22.md` | 架构的认知科学背景长文：感官表征、符号、逻辑与算力 |

### 发布 — `docs/releases/`

| 文档 | 内容 |
|---|---|
| `RELEASE_NOTES_v0.1.0.md` | v0.1.0 发布说明 |
| `RELEASE_NOTES_v0.1.1.md` | v0.1.1：L2 公开复现包与校验脚本 |

### 复现 — `docs/reproducibility.md`

| 文档 | 内容 |
|---|---|
| `reproducibility.md` | L2 复现包的下载、SHA256 校验、`make reproduce-l2` 与期望结果 |
| `repro/PSS_BUNDLES_v0.1.1.json` | 复现包清单与逐文件哈希 |
| `repro/SHA256SUMS` | 归档 SHA256 |

### 其他既有目录

| 路径 | 内容 |
|---|---|
| `notes/` | 工作笔记、claim–evidence 映射、实验设计、文献综述 |
| `reports/` | claim-gap 报告、property validation 报告、dev 运行记录 |
| `eval/` | 预注册、评测套件说明、spatial 评测协议 |
| `scorecards/` | 可复现的评测结果（JSON/Markdown） |
| `specs/` | 世界 spec（S1/S2/S3/S4/S6） |

---

## 2. 推荐阅读顺序

1. 根目录 `README.md`：项目定位、架构、结果、边界。
2. `docs/architecture/PSS-认知层架构定位.md`：分层认知架构与 PSS 的职责边界。
3. `docs/experiments/PSS_VS_TRANSFORMER_RESULTS.md`：目前最强的对照证据。
4. `docs/experiments/PSS-五优先级最终结论.md`：哪些结论成立、哪些不能声称。
5. `notes/PAPER_WRITING_MAP.md`：claim–evidence 映射与写作口径。
6. `reports/claim_gaps/INDEX.md`：claim-gap 实验索引与关键数字。
7. 需要实现细节时，按主题查 `docs/phases/`。

---

## 3. 维护约定

- **根目录只放**：`README.md`、`AUTHORS.md`、`NOTICE`、`LICENSE`、`CITATION.cff`（以及构建配置）。
- **新架构/设计文档** → `docs/architecture/`。
- **新计划/任务清单** → `docs/plans/`。
- **新实验协议/结论** → `docs/experiments/`；原始结果放 `reports/` 或 `eval/results/`。
- **新阶段报告** → `docs/phases/`，命名 `PHASE<N>_<TOPIC>_REPORT.md`。
- **工作笔记、临时分析** → `notes/`。
- **可复现的评测结果** → `scorecards/`。
- **生成数据/权重** → `data/`、`checkpoints/`（已 gitignore，不要提交）。
- 新文档尽量在开头写清：日期、范围、依据、关联实验编号。

---

## 4. 旧路径说明

本次整理把根目录的 `PHASE*.md`、`PSS-*.md`、`PSS_VS_TRANSFORMER_*.md`、`公共语义空间*.md`、`从婴儿的感官到抽象符号-11.22.md`、`CATEGORY_PRETRAIN_REPORT.md`、`RELEASE_NOTES_v0.1.0.md` 移入了 `docs/`。

历史文档里如果出现裸文件名，按以下规则解析：

| 旧文件名 | 新位置 |
|---|---|
| `PHASE*_REPORT.md`、`PHASE*_GAP_CLOSURE_*.md`、`PHASE*_EVAL_SUITE.md`、`PHASE*_OFFICIAL_DATA.md` 等 | `docs/phases/` |
| `CATEGORY_PRETRAIN_REPORT.md` | `docs/phases/` |
| `PSS-认知层架构定位.md`、`PSS-认知运行时与Harness.md`、`PSS-架构评测指标.md` | `docs/architecture/` |
| `公共语义空间推理架构-推演记录-11.19.md`、`公共语义空间-工作总结-v1.md` | `docs/architecture/` |
| `PSS_VS_TRANSFORMER_EXPERIMENT.md`、`PSS_VS_TRANSFORMER_RESULTS.md`、`PSS-五优先级最终结论.md` | `docs/experiments/` |
| `公共语义空间-落地执行计划-v1.md`、`PSS-优化任务清单.md`、`PSS-IsaacSim-*.md` | `docs/plans/` |
| `从婴儿的感官到抽象符号-11.22.md` | `docs/background/` |
| `RELEASE_NOTES_v0.1.0.md` | `docs/releases/` |

历史报告内部的交叉引用保持原文，不再逐篇改路径；需要跳转时以本索引为准。

---

## 5. 不要把什么放这里

- 生成的数据集、模型权重（`data/`、`checkpoints/`）。
- 原始评测输出（`eval/results/`，已 gitignore）。
- 个人临时草稿（放 `notes/` 或本地）。
- 对外发布用的论文稿（如需要，单独建 `paper/`，当前只有 `paper/figures/`）。
