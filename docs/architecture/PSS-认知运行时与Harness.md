# 认知运行时与 Harness 的区别

> 日期：2026-11-24
> 关联文档：`PSS-认知层架构定位.md`
> 目的：区分“把模型接起来的胶水层”和“拥有状态与语义的系统层”

---

## 1. 一句话

> **Harness 是“把模型接起来”的胶水层；认知运行时（cognitive runtime）是“拥有状态、状态转换和计算语义”的系统层。Harness 可以是认知运行时的一部分，但认知运行时不等于 harness。**

---

## 2. 两个概念

### 2.1 Harness

围绕模型的外部脚手架：

- 拼 prompt；
- 调工具/API；
- 解析输出；
- 循环、重试、超时；
- 把检索结果塞进上下文；
- 记录日志。

它通常**不拥有状态语义**：状态往往是聊天历史、字符串、JSON；它不定义“实体是什么、事实如何冲突、计划如何验证”。

典型例子：

- LangChain agent loop；
- 一个 Python 脚本：读文件 → 调 LLM → 解析 JSON → 调下一个工具；
- 评测框架里的 adapter。

### 2.2 Cognitive runtime

“认知”的运行环境：

- 拥有**类型化、持久化状态**：实体、关系、事实、目标、计划、版本；
- 定义**状态转换**：绑定、更新、合并、冲突解析、提交、回滚；
- 定义**不变量**：一致性、权限、审计、生命周期；
- 负责**调度与资源**：active-set、检索、计算预算、优先级；
- 可以有**自己的可学习模块**：write/read/reason 策略。

典型类比：

- 操作系统：进程、内存、调度、权限；
- 数据库：事务、索引、一致性、回滚；
- 规划器：状态、算子、目标、搜索。

---

## 3. 核心区别

| 维度 | Harness | Cognitive runtime |
|---|---|---|
| 主要职责 | 连接、调用、解析 | 状态、转换、一致性、调度 |
| 状态 | 上下文/字符串/临时 JSON | 类型化、持久、可版本化 |
| 语义 | 很少定义 | 定义实体/事实/关系/计划语义 |
| 计算 | 主要转发给模型 | 自己执行状态转换/检索/规划 |
| 失败归因 | prompt/工具接线 | 状态、关系、规划、检索 |
| 可替换性 | 换模型即可 | 换模型是换一个 compute kernel |
| 学习 | 通常不训练 | 可以有 learned policy/module |
| 审计 | 日志 | 事务、版本、回滚、权限 |

---

## 4. 重叠在哪里

两者都会做：

- 调用模型；
- 管理上下文；
- 调度工具；
- 记录日志。

所以一个**很薄的 cognitive runtime** 看起来就像一个 harness。  
一个**很厚的 harness** 也可能包含 memory、retriever、planner，边界会模糊。

区别不在“有没有代码”，而在：

> **它是否拥有状态语义和状态转换的权威。**

---

## 5. 用 PSS 举例

### 5.1 现在哪些部分是 harness 性质

- 把 encoder 输出拼成 token；
- 调 GraphMatcher / Transformer；
- 把 planner 的 action 传给 ActionHead；
- 串联 perception → tracker → planner。

这些是 orchestration。

### 5.2 哪些部分是 runtime 性质

- `FactStore`：多源、优先级、冲突、版本、回滚、审计；
- `SceneGraph`：实体生命周期、track、关系；
- `Tracker`：状态转换、dormant/historical；
- `RuleEngine` / planner：显式逻辑、goal、plan；
- `PackageManager`：版本、签名、激活、回滚。

这些不是“胶水”，是**状态与语义**。

### 5.3 哪些部分还不够 runtime

- 逻辑多是手写；
- active-set 多是 oracle/精确检索；
- write/read policy 不是学出来的；
- 关系层还是浅层边特征。

所以当前 PSS 是：

> **一个带 runtime 骨架的系统，但认知计算还没有完全学起来。**

---

## 6. 判断标准

问四个问题：

1. **它是否拥有类型化、持久的状态？**  
   是 → runtime；只是字符串/上下文 → harness。

2. **它是否定义状态转换和不变量？**  
   是 → runtime；只是转发 → harness。

3. **它是否自己做检索/规划/一致性检查？**  
   是 → runtime；全部交给模型 → harness。

4. **失败时，问题出在状态/语义，还是 prompt/接线？**  
   前者 → runtime；后者 → harness。

---

## 7. 和“架构”的关系

- **Harness**：系统集成层，不产生独立智能。
- **Cognitive runtime**：系统层，拥有状态和计算语义，可以产生独立能力。
- **Model architecture**：runtime 里的计算本身可学习，能泛化。

三者关系：

```
Model architecture  ⊂  Cognitive runtime  ⊂  完整系统
Harness 是完整系统里的连接层，可以很薄，也可以很厚
```

PSS 的目标不是“做一个更厚的 harness”，而是：

> **让 runtime 里的 write/read/reason 变成可学习计算，同时保留外部知识、显式逻辑、审计和 active-set。**

---

## 8. 对 PSS 研究主张的影响

### 如果 PSS 只是 harness

能主张：

- 系统集成价值；
- 知识热更新、审计、回滚；
- active-set 成本优势；
- 与强模型组合的工程价值。

不能主张：

- 更高智力上限；
- 更强泛化；
- 新模型架构。

### 如果 PSS 是 cognitive runtime

能主张：

- 状态/语义层独立于模型参数；
- 知识可热更新、逻辑可版本化；
- 计算随 active-set，而不是历史长度；
- 可控、可审计、可回滚。

### 如果 PSS 是 model architecture

还需要：

- write/read/reason 可学习；
- 端到端或 bilevel 训练；
- 与同算力 transformer 比较；
- 证明结构化归纳偏置带来更高上限。

---

## 9. 一句话

> **Harness 管“怎么接”，cognitive runtime 管“状态是什么、怎么变、谁来算”。两者会重叠，但本质不同：harness 是胶水，runtime 是语义与状态的主人。PSS 现在有 runtime 的骨架，但只有把 write/read/reason 学起来，才不只是 harness。**
