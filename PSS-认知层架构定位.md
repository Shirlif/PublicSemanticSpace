# PSS 与 Transformer 的分层认知架构

> 日期：2026-11-24
> 定位：把 PSS 从“transformer 的 harness”重新表述为“感知之上的认知运行时”
> 依据：四波实验、P5b、公平归因、object attention、长历史/知识热更新实验
> 关联：`PSS-认知运行时与Harness.md`

---

## 1. 核心判断

> **Transformer 负责“看懂/听懂”，MLP 投影负责“对齐到同一语义空间”，PSS 负责“把对象绑起来、算关系、做规划、管记忆”，输出头负责“把思考变成动作/语言/行为”。**

这不是 transformer 的替代，而是：

```
sense → align → bind → reason → act
```

PSS 的位置不是“另一个 transformer”，而是 **认知运行时（cognitive runtime）**。

---

## 2. 完整分层

```
传感器 / 本体感觉 / 触觉 / 视觉 / 音频 / 语言输入
        │
        ▼
模态 Transformer 编码器（冻结或慢更新）
        │
        ▼
MLP / adapter 投影头 → 公共语义空间（128-d 或更高）
        │
        ▼
PSS 认知层
  ├─ 实体绑定：object slots / scene graph / persistent ID
  ├─ 关系计算：spatial / semantic / tactile / temporal / task
  ├─ 知识记忆：facts / prototypes / episodes / versions
  ├─ 逻辑规划：rules / operators / planner / world model
  └─ 不确定性 / 冲突 / 弃权 / 审计
        │
        ▼
输出头
  ├─ 动作头：push / grasp / navigate
  ├─ 行为头：skill / option / policy
  ├─ 语言头：LLM decoder / structured prompt
  └─ 价值/安全头：risk / permission / rollback
```

---

## 3. 每一层的职责

### 3.1 模态 Transformer：只做编码，不背历史

- 视觉：DINOv2 / CLIP / SigLIP / VLM；
- 触觉：tactile transformer / CNN；
- 音频、本体感觉、语言：各自编码器；
- **按帧/按对象编码**，不要把整段历史塞进 transformer；
- 历史交给 PSS 的实体记忆，而不是 KV cache。

这一步直接绕开“上下文越来越长”的问题。

### 3.2 投影头：不是简单 MLP，而是对齐层

- 多模态对比学习 / 锚点对齐；
- 模态缺失/噪声下的校准；
- 保留原始特征，投影只做 adapter，不把信息压死；
- 输出要带 reliability / uncertainty，而不是一个裸向量。

当前 PSS 的 128-d 公共空间是对的，但它只是**感知层对齐**，不能单独承载逻辑。

### 3.3 PSS 认知层：至少三层表示

1. **感知语义空间**：128-d 向量、原型、类别；
2. **实体层**：object slot、track ID、pose、属性、置信度；
3. **关系/事实层**：scene graph、facts、rules、goal、plan。

只有第 1 层不够；PSS 的价值在第 2–3 层。

### 3.4 输出头：语言不是 MLP 能干的

- 动作/行为：小 head 或 policy 可以；
- 语言：需要 pretrained decoder / LLM；
- PSS 给 LLM 的应该是**结构化状态 + 工具接口**，不是让 MLP 直接“翻译”；
- 语言输出头本质上是另一个大模型，PSS 负责给它可信的上下文。

---

## 4. 这套架构怎么回答“智力上限”

- **感知/语言能力**：由 transformer/VLM 决定，可以很大、可以预训练；
- **知识/事实**：由 PSS 外部存储决定，可热更新；
- **逻辑/关系**：由 PSS 的关系层/规则/planner 决定，可以学习、可以版本化；
- **工作记忆/控制**：由 PSS 的 active-set 决定，成本可控；
- **输出能力**：由动作头/语言 decoder 决定。

所以智力上限不是由“PSS 小”决定，而是由：

> **最强编码器 + 最强解码器 + PSS 的关系/逻辑层是否可学习。**

当前 PSS 的上限低，是因为编码器小、逻辑手写；不是因为分层本身。

---

## 5. 当前实现和目标的差距

> 下表已按 Phase 11–24 的实际实现回写（2026-11-24）。
> `部分` = 架构路径已接但精度/覆盖受限；`达标` = 有可证伪实验支持。

| 层 | 目标 | 当前（Phase 11–24 后） | 状态 |
|---|---|---|---|
| 编码器 | CLIP/DINOv2/VLM | runtime 默认 frozen DINOv2（SmallCNN 为 fallback）；CLIP 代码就绪但本地无权重；VLM 未接 | 部分 |
| 投影 | 对齐 + 校准 | DINOv2 默认路径：PCA-whiten 96（vision+geometry）+ residual，reliability 进 uncertainty；但识别/re-ID 共用头、开放集未超 bank、SmallCNN fallback 仍是 plain、CLIP/VLM 未验证 | 部分 |
| 实体绑定 | object attention / persistent slots | `EntityLayer` persistent slots（ID/生命周期/属性/置信度/provenance）；object attention 作为可切换关联选项；hand/graph/attention 均可选 | 部分 |
| 关系 | typed object attention + 学习规则 | typed heads（spatial/semantic/tactile）为默认；relation logits 进 scene graph；几何阈值可学习/版本化；temporal/task 未训练 | 部分 |
| 规划 | 学习 + 搜索 | A* 搜索保留；learned cost-to-go heuristic（可选，expansions 减半）；versioned operators；手写 heuristic 为 fallback | 达标 |
| 语言 | LLM decoder | 本地 Qwen2.5-1.5B decoder + 结构化状态/tools/audit；7 查询 0.857，citations/枚举仍不足 | 部分 |
| 审计 | 版本/回滚/权限 | `CognitiveState` version/snapshot/rollback/audit；四档权限 read-only/service/operator/admin；拒绝动作审计 | 达标 |

详细结果见 `PHASE11`–`PHASE24` 报告与 `PSS-五优先级最终结论.md` §8。

这也解释了实验现象：

- **re-ID 公平归因后 PSS 不占优**：它在抢编码器/输入层的活；
- **P5b 关系组合 PSS 强**：它在做认知层的活；
- **object attention 追平但没赢**：关系层还是浅的，没有关系监督；
- **长历史 PSS 强**：active-set 是认知层的正确设计。

---

## 6. PSS 与 Transformer 的特性对比

| 维度 | Transformer | PSS |
|---|---|---|
| 计算单位 | token | object / entity |
| 表示 | 稠密分布式向量 | 结构化对象槽 + 原型 + 事实 |
| 知识 | 隐式存在参数里 | 外部显式存储、可版本化 |
| 逻辑 | attention/FFN 隐式学习 | 规则/算子/planner 显式 |
| 记忆 | 上下文窗口 / KV cache | scene graph / FactStore，持久 |
| 更新 | 微调、replay、EWC | 原子 fact/prototype 更新、回滚 |
| 计算复杂度 | O(L²)，随历史增长 | O(k²)，随 active-set 增长 |
| 泛化 | 开放世界、语言、感知强 | 组合、关系、状态、审计强 |
| 数据需求 | 大规模预训练 | 小数据 + 规则可启动 |
| 可解释性 | attention map，局部 | 事实、规则、版本、审计链 |
| 主要失败 | 幻觉、遗忘、上下文膨胀 | 感知/检索错误、规则脆弱 |
| 扩展方式 | 参数/数据/算力 scaling | 知识/实体/规则/技能 scaling |

---

## 7. 优化顺序

### 第一优先：换编码器，不动 PSS 接口

- 冻结 CLIP/DINOv2；
- 触觉换更强编码器；
- 投影头做对齐/校准；
- 测 P5b、re-ID、开放集。

### 第二优先：把关系层变成可学习

- object attention + relation labels；
- typed heads：spatial / semantic / tactile / temporal / task；
- 输出 relation logits 进 scene graph；
- 不再手写边特征。

### 第三优先：学习 active-set / write policy

- 学“哪些对象/事实值得注意”；
- 学“什么值得写入记忆”；
- 不确定时回退到更大上下文。

### 第四优先：输出头升级

- 动作/行为头保留；
- 语言输出接 LLM decoder；
- PSS 提供结构化状态、工具、审计，而不是直接生成语言。

### 第五优先：端到端 / bilevel 训练

- 编码器冻结，训练投影 + 绑定 + 关系；
- 任务损失 + 一致性/审计约束；
- 否则 PSS 永远只是 harness。

---

## 8. 什么会让 PSS 变成 harness，什么会让它成为架构

### 会变成 harness 的做法

- 只把 transformer 包一层 Python；
- 逻辑全部手写；
- active-set 用 oracle；
- 输出头是简单 MLP；
- 训练信号只来自外部模型。

### 会变成架构的做法

- 学 write / read / reason；
- 关系层有可学习参数和监督；
- active-set 由 learned retrieval 决定；
- 投影/绑定/关系可端到端或 bilevel 训练；
- PSS 自己的状态和计算图决定输出。

---

## 9. 一句话

> **Transformer 提供能力，PSS 提供可编辑知识、显式逻辑、持久记忆和 active-set 控制。PSS 不该和 transformer 抢感知，它应该站在感知之上做认知；它的上限不取决于“是否小”，而取决于关系层是否可学习、检索是否可靠、编码器/解码器是否足够强。**
