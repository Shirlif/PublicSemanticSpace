# PSS ContactEvent 层计划：从几何碰撞到多感官事件

> 日期：2026-11-29
> 上游：`docs/plans/PSS-三维场景建模实验计划-v2.md`、`docs/research/PSS-扫描到语义到标定到变化检测.md`
> 目的：把“碰撞/接触”从**几何谓词**升级为**事件**：几何预测、多感官证据、时序分割、冲突解决、主动感知、PSS 审计。
> 状态：C0–C6 已执行（2026-11-29）；覆盖扩展 48-cell 已完成。
> **工程目标达成**：ContactEvent 层 opt-in、可审计、事件/证据/路由可写入 runtime；23 项 contact-event 测试 + 56 项 runtime 回归通过。
> **研究假设 H（fusion ≥ 最好单通道 +0.05）未成立**：clean force 是 GT oracle；noisy force 下 C2 drop/collide strict +0.229，但 48-cell 覆盖只有 **1/48** 有 +0.05，47/48 `force_noisy` 不输 fusion。
> **「force_noisy 默认」仅限接触任务**；距离/关系/材质/遮挡任务见 `docs/plans/PSS-任务条件化路由实验计划.md`。

---

## 0. 一句话

**游戏引擎用精确 transform 算碰撞；具身智能没有精确 transform，只能从多感官证据推断“是否接触、何时、在哪、多强、是否可信”。** ContactEvent 层就是这件事的架构：几何/3D 状态做**预测**，force/tactile/audio/vision/proprio 做**证据**，PSS 做**融合、冲突、审计**；输出不是 bool，而是带来源、强度、时间、置信度的 `ContactEvent`。

---

## 1. 为什么需要这一层

### 1.1 几何碰撞的局限

- 几何碰撞是**状态谓词**：给定精确位姿/形状，算距离/穿透；
- 具身智能的位姿是估计的，带噪声、部分观测、延迟；
- 碰撞不只是“重叠”，还是**事件**：onset、持续时间、参与者、强度、后果；
- 人判断碰撞用多感官：声音、视觉突变、触觉/力、本体感觉、材料/因果先验。

### 1.2 目标

把碰撞从：

```text
collision = geometry_overlap(T_a, T_b)  # bool
```

升级为：

```text
ContactEvent(
  subject, object,
  onset, offset,
  location, intensity,
  confidence, status,
  evidence = {geometry_pred, force, tactile, audio, vision, proprio}
)
```

---

## 2. 数据模型

### 2.1 ContactEvidence

```python
@dataclass
class ContactEvidence:
    source: str          # geometry | force | tactile | audio | vision | proprio
    timestamp: float
    score: float         # 0..1, evidence for contact
    intensity: float | None
    location: np.ndarray | None
    reliability: float   # per-source calibration
    metadata: dict
```

### 2.2 ContactEvent

```python
@dataclass
class ContactEvent:
    event_id: str
    subject: str
    object: str
    onset: float
    offset: float | None
    location: np.ndarray | None
    intensity: float | None
    confidence: float
    status: str          # predicted | confirmed | contradicted | resolved
    evidence: list[ContactEvidence]
    version: int
    provenance: dict
```

### 2.3 状态机

```text
predicted  --force/tactile/audio confirm--> confirmed
predicted  --window expires, no evidence--> expired
confirmed  --contradicting evidence-------> review
review     --active perception------------> confirmed / rejected
```

---

## 3. 证据通道

| 通道 | 来源 | 在 MuJoCo 里 | 现实局限 |
|---|---|---|---|
| **geometry_pred** | 3D 位姿/形状，预测距离/穿透/相对速度 | 有（scene model） | 位姿/标定误差；遮挡 |
| **force** | 接触法向力/冲量 | 有（`SensorRig.contacts()`） | 真实力传感器噪声/带宽 |
| **tactile** | 触觉图/压力分布 | 有 proxy（`tactile_maps()`） | 真实触觉阵列覆盖有限 |
| **vision** | 运动突变、遮挡、形变 | 有 RGB-D/分割 | 遮挡、光照、运动模糊 |
| **audio** | 冲击声、频谱、能量 | **没有**；可合成代理 | 真实音频混响/噪声 |
| **proprio** | 关节力矩/电流/反作用力 | 没有；可后续接 | 真实机器人动力学 |

**原则**：每个通道是一个 `ContactEvidence`，有自己的 reliability/latency；不要把它们压成一个标量就丢掉来源。

---

## 4. 架构与 PSS 集成

```text
MuJoCo / 真实传感器
  → 证据提取器（geometry/force/tactile/audio/vision/proprio）
  → 时序分割（onset/offset，hysteresis）
  → 证据融合（Bayesian / Dempster-Shafer / logistic）
  → ContactEvent（status/confidence/evidence）
  → PSS：CognitiveState / UncertaintyService / RelationFactLayer / audit
  → 主动感知（probe/tap/re-observe）
```

### 4.1 新增模块

```
pss/events/model.py          # ContactEvent / ContactEvidence / 状态机
pss/events/extractors.py     # 各通道 evidence 提取
pss/events/segmenter.py      # onset/offset、hysteresis、事件合并/拆分
pss/events/fusion.py         # Bayesian/DS/logistic 融合 + 校准
pss/events/layer.py          # ContactEventLayer：接 runtime
pss/eval/scripts/run_contact_event_bench.py
tests/test_contact_event.py
```

### 4.2 与现有 PSS 的关系

- `Entity`：subject/object 的位姿、协方差、provenance；
- `RelationFactLayer`：`contact` 边可以来自 ContactEvent；
- `UncertaintyService`：confidence 低 → review/abstain；
- `CognitiveState`：事件写入 audit，支持版本/回滚；
- `PSSRuntime`：事件可触发 `act/stop/probe`。

---

## 5. 模拟与数据

### 5.1 场景

复用 `pss/sim` 的 MuJoCo 桌面：

- 物体：box/cylinder/sphere/ellipsoid；
- 动作：`push` / `lift` / `place` / `wait`（`pss/sim/actions.py`）；
- 场景：drop、push-into、stack、near-miss、occlusion；
- 每帧：RGB/depth/seg、物体位姿、contact force、tactile proxy。

### 5.2 Ground truth

- 接触事件：`SensorRig.contacts()` 中 `normal_force > threshold` 的连续区间；
- onset/offset：力首次/最后超过阈值的时间；
- 参与者：`geom_to_instance` 映射；
- 强度：峰值法向力/冲量；
- 位置：接触点 `contact.pos`。

### 5.3 音频（可选）

- 没有真实音频；用接触力/冲量合成**代理**冲击声（衰减正弦/噪声），明确标注“合成，不是真实音频”；
- 只用于验证“音频作为证据通道”的接口，不用于声称真实声学性能。

---

## 6. 方法

### 6.1 证据提取

- **geometry_pred**：由 3D 位姿/形状算表面距离/相对速度；`score = sigmoid((eps - distance)/τ)`；intensity = 预测冲量。
- **force**：`normal_force` 阈值/归一化；intensity = 峰值/冲量。
- **tactile**：触觉图能量/峰值；intensity = 能量。
- **vision**：光流/质心速度突变、遮挡比例；score = 运动突变。
- **audio**：onset 检测 + 频带能量；score = 冲击检测概率。
- **proprio**：力矩/电流突变。

### 6.2 时序分割

- 每个通道先做单通道 onset/offset（阈值 + hysteresis + 最小持续时间）；
- 多通道事件在时间窗内合并（IoU/时间重叠）；
- 避免抖动：最小持续时间、冷却时间、状态机。

### 6.3 融合

- **Bayesian**：`p(contact | e) ∝ p(contact) ∏ p(e_i | contact) / p(e_i | no-contact)`；每个通道的 likelihood 在 train/cal 上校准。
- **Dempster-Shafer**：处理“不知道/冲突”；适合 audio 缺失、tactile 未覆盖。
- **Logistic**：特征 = 各通道 score/reliability/latency；简单、可校准。
- 输出：`confidence` + `status`（predicted/confirmed/contradicted/review）。

### 6.4 冲突处理

| 情况 | 解释 | 动作 |
|---|---|---|
| geometry predicted，force/tactile 确认 | 正常接触 | confirmed |
| geometry predicted，force/tactile 无 | 预测但未发生（near-miss/延迟） | predicted → expired / review |
| force/tactile 有，geometry 无 | 模型/标定错，或未建模接触 | contradicted → review/audit |
| audio 有，force 无 | 远处/其他物体/误检 | 降低 confidence |
| 多通道互相矛盾 | 不确定性高 | 主动感知 |

### 6.5 主动感知

- 不确定时：probe/tap/re-observe/换视角；
- 选择动作最大化信息增益（简化：优先补缺失通道/降低最大不确定性）；
- 评测：主动感知后的 event F1/延迟/代价。

---

## 7. 指标与预注册 bar

### 7.1 事件级指标

| 指标 | 定义 | bar（clean / occlusion） |
|---|---|---|
| Event precision / recall / F1 | 时间容差内匹配 GT 事件 | ≥0.95 / ≥0.85 |
| Onset latency | 检测 onset − GT onset | ≤50 ms / ≤100 ms |
| Offset latency | 检测 offset − GT offset | ≤100 ms |
| Intensity MAE | 峰值力/冲量相对误差 | ≤10% |
| Location error | 接触点 3D 距离 | ≤2 cm |
| False alarm | no-contact/near-miss 上的误报 | ≤5% |
| Conflict resolution | 冲突场景的正确处理率 | ≥0.90 |
| ECE | confidence 校准 | ≤0.05 |

### 7.2 下游指标

- 及时停止/避让成功率；
- 抓取/放置任务成功率；
- 安全违规次数；
- 事件到动作的延迟。

### 7.3 对照臂

| arm | 说明 |
|---|---|
| `geometry_only` | 只用 3D 几何预测 |
| `force_only` | 只用接触力 |
| `tactile_only` | 只用触觉 proxy |
| `vision_only` | 只用视觉突变 |
| `audio_only` | 只用合成音频（可选） |
| `fusion` | 多通道 Bayesian/DS |
| `fusion_active` | fusion + 主动感知 |
| `oracle` | GT 事件 |

---

## 8. 实验阶段

### C0：schema + 分割 + GT

- `ContactEvent` / `ContactEvidence` / 状态机；
- 从 MuJoCo contact force 生成 GT 事件；
- 单通道 onset/offset 分割；
- 出口：事件表可复现。

### C1：单通道 baseline

- geometry/force/tactile/vision（audio 可选）；
- 每通道的 precision/recall/latency；
- 出口：知道每个通道的强弱与盲区。

### C2：融合与校准

- Bayesian/DS/logistic；
- train/cal 校准 likelihood/reliability；
- 出口：fusion ≥ 最好单通道 +0.05。

### C3：遮挡/冲突/近失

- 视觉遮挡、tactile 未覆盖、geometry 错、near-miss；
- 出口：冲突处理 ≥0.90；false alarm ≤5%。

### C4：主动感知

- probe/tap/re-observe；
- 出口：主动后 F1/延迟改善，代价可接受。

### C5：PSS 集成

- `ContactEventLayer` 写入 `CognitiveState`/`RelationFactLayer`；
- audit/provenance/rollback；
- 出口：runtime 可查询事件，旧测试不回归。

### C6：报告与 go/no-go

- 事件级主表、消融、失败项；
- 决策：ContactEvent 是否成为 runtime 默认接触层；几何/1D 是否 fallback。

---

## 9. 停止规则

| 观测 | 结论 | 动作 |
|---|---|---|
| `force_only` 已 ≥0.95 且 fusion 无增量 | 融合没有价值 | 报告；保留单通道 |
| fusion < 最好单通道 | 融合/校准有问题 | 修 likelihood/权重，不加新臂 |
| geometry_only 主导 | 事件层不需要复杂融合 | 简化；几何优先 |
| onset latency 超标 | 时序/窗口问题 | 简化 segmenter，不加通道 |
| 冲突处理 <0.90 | 证据不可靠 | 记录；不宣称多模态 |
| ECE >0.10 | 校准失败 | 不宣称 confidence |

**C2 结束先做 go/no-go；不通过不推进 C3–C5 的默认化。**

---

## 10. 风险与混杂

1. **GT 事件定义**：阈值/最小持续时间会改变 precision/recall；预注册。
2. **force/tactile 在 sim 里近 oracle**：真实传感器噪声/带宽未测；结果是上限。
3. **audio 是合成代理**：不能声称真实声学性能。
4. **通道延迟不同**：force 快、vision 慢；融合要建模延迟，否则 onset 偏。
5. **事件边界模糊**：碰撞有抖动/多次接触；hysteresis/合并规则要固定。
6. **类别不平衡**：大多数帧无接触；用事件级指标，不用帧级 accuracy。
7. **test 调参**：likelihood/权重/阈值在 train/cal 上选；test 一次。
8. **多重比较**：一个主指标（event F1）；Holm。

---

## 11. 交付物与时间盒

```
pss/events/model.py
pss/events/extractors.py
pss/events/segmenter.py
pss/events/fusion.py
pss/events/layer.py
pss/eval/scripts/run_contact_event_bench.py
tests/test_contact_event.py
eval/results/contact_event/
docs/experiments/PSS-ContactEvent-Report.md
```

| 天 | 内容 | 出口 |
|---|---|---|
| C1 | schema + GT + 单通道分割 | 事件表可复现 |
| C2 | 单通道 baseline | 通道强弱/盲区 |
| C3 | 融合 + 校准 | fusion ≥ 单通道 +0.05 |
| C4 | 遮挡/冲突/近失 | 冲突 ≥0.90、误报 ≤5% |
| C5 | 主动感知 | 主动后改善 |
| C6 | PSS 集成 + 报告 | 默认化决策 |

---

## 12. 边界

- MuJoCo 图元是 proxy；force/tactile 近 oracle；audio 是合成。
- 不是真实机器人/真实传感器；不声称 sim2real。
- 事件层不替代几何/3D 状态；几何是预测通道，事件层是融合/审计。
- 真实音频/触觉/本体感觉需要真实硬件或数据集；本计划先验证接口与融合逻辑。

---

## 13. 一句话

**ContactEvent 层把碰撞从 bool 变成事件：几何预测、force/tactile/audio/vision 证据、时序分割、冲突解决、主动感知、PSS 审计。** 先做 force+tactile+geometry 的融合，audio 作为接口代理；C2 结束 go/no-go，不通过就不默认化。
