# PSS vs LLM：空间感知与空间规划预注册（S1）

> 日期：2026-11-24
> 状态：**预注册。先冻结本文件，再跑 S1。**
> 关联：`PSS-认知层架构定位.md`、`PSS_VS_TRANSFORMER_EXPERIMENT.md`、`eval/preregistration.md`
> 范围：S1 只做 **T-text 空间感知**；S2 图像档、S3 闭环规划、S4 外部效度另立文件。

---

## 1. 主张与不主张

### 主张

> 在**同一份场景状态**、**同一套空间谓词**、**同一评测协议**下，PSS 的显式关系层 + 精确几何 + 规划器，在**关系级空间感知**和**可执行空间规划**上，优于**裸 LLM**，并在**未见关系组合 / 未见布局**上优于 **LLM + 通用工具**。

### 不主张

- PSS 在图像识别、开放世界常识、语言生成上优于 LLM/VLM；
- PSS 在像素级感知上更强；
- PSS 在自然语言歧义、指代消解上更强；
- 本实验能外推到前沿多模态大模型，除非 S2 跑了对应基线。

### 为什么限定“关系级空间感知”

`PSS-认知层架构定位.md` §1 已把分工写死：Transformer 负责“看懂/听懂”，PSS 负责“绑定、关系、规划、记忆”。§5 的现状表也显示：编码器“部分”、关系层“部分”、规划“达标”。所以本预注册把“空间感知”定义为：

> 给定实体、位姿、尺寸和相机，判断/计算实体之间的空间关系、参照系、度量、遮挡与容纳；不包含从像素识别实体。

这与 §3.3 的第 2–3 层一致，也避免和 VLM 的主场硬碰。

---

## 2. 三个公平性锁

### 锁一：同一观测预算

| 档 | LLM 拿到 | PSS 拿到 | 测什么 |
|---|---|---|---|
| **T-text**（S1） | 物体列表 + 位姿 + 尺寸 + 相机，**不含关系** | 同一份状态 | 纯关系推理 |
| **T-image**（S2） | RGB（+深度） | RGB-D + 分割 | 端到端空间感知 |
| **T-tools**（S1/S3） | 同一份状态 + 几何/规划工具 | 同一份状态 + 关系层/规划器 | 认知层 vs 工具层 |

三档分开报，不混表。

### 锁二：同一动作空间

S3 闭环统一用 `push / lift / place / navigate` 原语；LLM 与 PSS 都通过同一 skill controller 执行。禁止 PSS 用原语、LLM 用关节。

### 锁三：真实基线是 LLM + tools

裸 LLM 不是主基线。主基线是：

- LLM + 场景 JSON + 计算器/几何工具；
- LLM + CoT；
- LLM + 工具 + 规划器（hybrid）。

如果 PSS 只赢裸 LLM、输给 LLM+tools，结论必须写成：

> PSS 是更小、更快、可审计的工具层实现，不是“空间智能更强”。

---

## 3. S1 任务族

所有任务都由程序化 3D 桌面场景生成，GT 由几何谓词计算。坐标统一：世界系 +x 右、+y 远离观察者、+z 上；单位米。

### 3.1 空间感知（T-text）

| 族 | 问题形式 | 答案 | 说明 |
|---|---|---|---|
| `relation` | “obj_1 在 obj_2 左边吗（世界系/观察者系）？” | yes/no | left/right/front/behind/near/contact |
| `vertical` | “obj_1 在 obj_2 上面吗？” | yes/no | 需要堆叠场景 |
| `frame` | 同一对物体，分别在 world / viewer / object 系下问 | yes/no | 只保留两系答案不同的样本 |
| `transitive` | “已知 A 在 B 左、B 在 C 左，A 在 C 左吗？” | yes/no | 真结论与反向结论各半 |
| `count` | “obj_2 左边有几个物体？” / “容器里有几个物体？” | 整数 | 0..N-1 |
| `ordinal` | “观察者系下从左数第 k 个杯子是哪个？” | object id | 要求同类别 ≥ k 个 |
| `occlusion` | “obj_1 被 obj_2 挡住了吗（cam0）？” | yes/no | 线段-包围球近似 |
| `metric` | “obj_1 比 obj_2 离相机更近吗？” / “obj_1 放得进 obj_2 吗？” | yes/no | 度量/容纳 |
| `perspective` | “从 cam1 看，obj_1 在 obj_2 左边吗？” | yes/no | 换相机参照系 |

### 3.2 空间规划（S3，本文件只冻结定义）

| 族 | 任务 | 主指标 |
|---|---|---|
| `route` | 避障到达目标区域 | 成功率、步数、碰撞 |
| `rearrange` | 把物体移动到目标关系集合 | 目标满足率、步数 |
| `stack` | 按稳定性/顺序堆叠 | 成功率、倒塌次数 |
| `pack` | 装进容器且不溢出 | 成功率、空间利用率 |
| `search` | 遮挡后找到目标 | 找到率、步数 |
| `replan` | 注入障碍/失败后恢复 | 恢复率、重规划次数 |
| `language` | 自然语言目标 → 计划 | 成功率（只报 hybrid） |

---

## 4. 数据切分与 OOD

| 切分 | 生成方式 | 用途 |
|---|---|---|
| `train` | 3–4 物体，自由布局，已见类别 | few-shot 示例、学习关系头训练 |
| `test_id` | 同分布 | 主表 |
| `test_layout` | 3×3 网格布局 | 布局 OOD |
| `test_count` | 5–6 物体 | 数量 OOD |
| `test_object` | 未见类别（bottle/can/jar） | 类别 OOD |
| `test_frame` | 只问 object 参照系 | 参照系 OOD（S1.5） |

硬约束：

- 训练/测试场景种子不重叠；
- 测试集不进入 few-shot 示例；
- 每个族、每个 split 至少 100 题；bool 族正负各半；
- 序列化场景**不含任何关系词或距离字段**，只含位姿、尺寸、相机。

---

## 5. 基线

| 编号 | 基线 | 说明 |
|---|---|---|
| B0 | Random | 随机 yes/no、随机计数、随机 object |
| B1 | LLM zero-shot | 场景 JSON + 问题，无 CoT |
| B2 | LLM CoT | 要求先写推理再答 |
| B3 | LLM + tools | 见 §5.1，分 low / mid / oracle 三档 |
| B4 | PSS exact | 精确几何谓词 + 传递闭包 |
| B5 | PSS learned | `SpatialRelationHead` / typed heads，不用 GT 谓词 |
| B6 | PSS + LLM hybrid | PSS 提供状态/工具/审计，LLM 负责语言与常识 |
| B7 | Human ceiling | 每族抽 20 题人工，仅作上限参考 |

S1 必跑 B0/B1/B2/B3/B4；B5 在关系头覆盖谓词后补；B6 在 S3 跑。

### 5.1 B3 工具档位

工具层是确定性的、白名单的，且**不读取 `query.answer`**。档位必须在跑之前固定，并写进报告。

| 档位 | 工具 | 测什么 |
|---|---|---|
| `low` | `get_object`、`get_camera`、`get_axes`、`delta`、`project`、`distance`、`surface_gap`、`horizontal_distance`、`z_difference` | 模型能否自己组合几何 |
| `mid` | low + `inside`、`fits`、`closer` | 模型能否选择并绑定直接谓词 |
| `oracle` | mid + `occluded`、`count`、`relation`、`ordinal` | **工具辅助上限，不是空间推理性基线**；这些工具直接返回答案 |

`oracle` 档如果追平 PSS exact，按 KS5 处理：结论写成“PSS 是可审计/低延迟的工具层实现”，不能写成“PSS 空间推理更强”。

工具调用协议（写在 system prompt 里，解析器只接受这个格式）：

```text
TOOL: {"name": "<tool>", "args": {...}}
```

每次调用后，工具返回 `TOOL_RESULT: {"ok": true, "result": ...}` 或 `{"ok": false, "error": ...}`。最多 `--llm-max-steps` 次；超限后强制要求 `Answer: ...`。报告必须同时给出：

- 最终准确率；
- 每题平均工具调用数；
- 工具错误率（未知工具、非法参数、非法 object id）；
- 是否出现“没调工具直接答”的样本比例。

---

## 6. 指标

| 指标 | 定义 |
|---|---|
| Accuracy | 每题 exact match |
| Macro-F1 | bool 族，按 yes/no 两类 |
| Count MAE | `count` 族绝对误差 |
| Invalid rate | 解析失败/拒答/多答案 |
| Latency p50/p95 | 单题端到端，含序列化 |
| Token cost | 输入+输出 token，按 1k 计 |
| Abstention | 明确说“不确定”的比例 |
| Consistency | 反对称/传递约束违反率 |
| Tool calls/item | B3 专属，平均工具调用次数 |
| Tool error rate | B3 专属，未知工具/非法参数/非法 id 的比例 |
| No-tool answer rate | B3 专属，未调用任何工具直接作答的比例 |

报告必须分 **family × split × baseline** 三张表；不允许只报总平均。

---

## 7. 杀死标准

| 编号 | 条件 | 触发后的动作 |
|---|---|---|
| **KS1** | PSS exact 对 LLM+tools 在 `relation/transitive/metric` 上优势 < 10 点 | 收窄主张为“实时/可审计/低成本”，不再声称空间推理更强 |
| **KS2** | PSS exact 在 `frame` 或 `perspective` 上不优于 LLM+tools | 说明参照系不是差异化能力，删除该主张 |
| **KS3** | PSS learned 在噪声/遮挡/OOD 上低于 LLM text | 学习关系层不成立，回退精确几何 |
| **KS4** | PSS 在 `language` 族低于 LLM | 不做端到端语言规划，只做 hybrid |
| **KS5** | LLM+tools 全面追平 PSS | 架构价值改写为“工具层 + 审计 + 延迟”，停止“更强”叙事 |

---

## 8. 协议细节

- LLM：temperature=0，max_new_tokens=32，固定 prompt 模板；每个基线跑 3 个种子（若 API 非确定）并报均值/方差；
- 同一题集、同一顺序、同一序列化；prompt 模板写入仓库，不临时改；
- 解析器只接受 `yes/no`、整数、`obj_\d+`；其余记 invalid；
- 所有结果写入 `eval/results/spatial_s1/`，失败项照实报告；
- 预注册冻结后，新增任务族只能作为 exploratory，不能进主表。

---

## 9. 本阶段能证明什么，不能证明什么

**能证明**：在给定状态、给定谓词、给定协议下，显式关系结构是否比语言先验 + 通用工具更准、更稳、更便宜。

**不能证明**：真实世界空间智能、视觉 grounding、开放世界常识、人形全身操作、sim-to-real。

**与 P5b 的关系**：P5b 已证明显式关系规划对 transformer BC 的组合泛化优势（1.000 vs 0.434）。S1 把基线换成 LLM，检验同一优势是否仍成立；不重复 P5b 的结论。

---

## 10. 执行顺序

1. 冻结本文件；
2. 生成 `train / test_id / test_layout / test_count / test_object`；
3. 跑 B0/B4，确认数据与评测器无泄漏（exact 应接近 1.0，random 接近机会水平）；
4. 接 B1/B2/B3，跑 3 个种子；
5. 出表；按 KS1–KS5 决定是否进入 S2/S3。

---

## 11. 复现命令

```bash
# 冒烟：每族 10 题，4 个切分，约 360 题
make spatial-s1-smoke

# 正式：每族 100 题，4 个切分，约 3600 题
make spatial-s1

# 本地 LLM 基线（默认 local_files_only，不会触发下载）
python -m pss.eval.spatial.scripts.run_s1 \
  --out eval/results/spatial_s1_llm \
  --n-per-family 20 --seed 0 \
  --llm-model Qwen/Qwen2.5-1.5B-Instruct \
  --llm-device cuda --llm-max-new-tokens 16

# B3 工具循环（low / mid / oracle）
python -m pss.eval.spatial.scripts.run_s1 \
  --out eval/results/spatial_s1_tools \
  --n-per-family 20 --seed 0 \
  --llm-model Qwen/Qwen2.5-7B-Instruct \
  --llm-device cuda --llm-max-new-tokens 128 \
  --llm-tools mid --llm-max-steps 4

# 允许下载（仅在明确需要时）
python -m pss.eval.spatial.scripts.run_s1 \
  --llm-model Qwen/Qwen2.5-7B-Instruct --llm-allow-download
```

产物：

```
eval/results/spatial_s1/
  report.md            # 汇总表
  results.json         # 指标
  rows_*.jsonl         # 逐题记录（答案、预测、原始输出、延迟）
  suite/*.jsonl        # 冻结题集（scene + query + answer）
```

注意：`rows_*.jsonl` 含答案，只能用于离线分析，不能作为 LLM 的输入。

---

## 12. 实现位置

| 文件 | 职责 |
|---|---|
| `pss/eval/spatial/schema.py` | 场景/查询数据结构、无泄漏序列化、严格答案解析 |
| `pss/eval/spatial/predicates.py` | 精确几何谓词（world/viewer/object 参照系） |
| `pss/eval/spatial/generator.py` | 程序化场景与 9 个任务族生成器 |
| `pss/eval/spatial/baselines.py` | PSS exact、random、naive-world、LLM 包装、B3 tool loop |
| `pss/eval/spatial/tools.py` | 确定性工具执行器、档位、调用协议解析 |
| `pss/eval/spatial/runner.py` | 指标、逐题记录、Markdown 报告 |
| `pss/eval/spatial/scripts/run_s1.py` | CLI |
| `tests/test_spatial_suite.py` | 确定性、无泄漏、exact 一致性、解析器测试 |
| `tests/test_spatial_tools.py` | 工具执行、档位、协议解析、循环终止测试 |

S1 的 exact 基线只证明“标签与谓词一致、题集无泄漏”，不构成能力结论。LLM 基线必须在报告中写清模型、量化方式、few-shot 数和 token 预算。


