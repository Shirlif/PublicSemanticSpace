# PSS 后续研究方向评估与任务优先级

> **日期**：2026-11-26
> **范围**：评估外部整理的《PSS 后续潜在研究方向》，对照论文 §7–§8、claim-gap 实验与当前代码，给出可执行的任务排序。
> **依据**：`paper/main.tex`、`notes/CLAIM_GAP_EXPERIMENTS.md`、`reports/claim_gaps/INDEX.md`、`eval/property_preregistration.md`、`README.md`。
> **一句话结论**：方向总体合理，但需要区分“论文缺口 / 已部分完成 / 猜想”，并按“能否改变论文结论 + 成本”重新排序。

---

## 0. 结论摘要

外部清单覆盖了论文的主要出口，但有三个问题：

1. **三类东西混在一起**：论文明确承认的缺口、仓库里已经部分做过的、纯猜想性扩展，被并列成同等优先级。
2. **部分事实已经过期**，尤其是 L2 感知链路与 bundle 的 embedding 状态。
3. **没有按“能否改变论文结论 / 成本”排序**，导致最该先做的可复现性问题没有出现。

最值得先做的不是清单里的“分布式 PSS”或“百万实体”，而是：

- **公开可复现工件**（成本最低、可信度收益最高）；
- **感知级 mid-stream expert swap**（把论文 Limitations 直接变成 Results）；
- **延迟 guard**（把已知的 2/14 尾延迟问题闭合）。

---

## 1. 论文原文的四个直接方向

论文没有独立的 Future Work 章节。四条“下一步”来自 §8 Conclusion（`paper/main.tex:826`），§7 Limitations（`paper/main.tex:801`）是另一份清单。

| 论文原文方向 | 当前状态 | 说明 |
|---|---|---|
| 感知驱动的 L2 harness | **部分具备，缺口变窄** | GroundingDINO 已写 DINOv2 embedding；35 个 detection bundle 中有 4 个带非空 `emb`。缺的是 mid-stream swap、更多感知 bundle、公开数据。 |
| 接入更多专家族 | **架构已支持，suite 未接** | `pss/encoders/registry.py` 已有 DINOv2 / CLIP / WT-MM；`ProjectionHead` 本就接受任意 `in_dim`。堵点在 property suite 的注册与校准路径。 |
| 真实机器人验证 | **未做，当前环境无硬件** | 论文明确 no real robot。短期折中是真实数据回放，而不是等真机。 |
| 竞争性非 PSS baseline | **只有 contract-free 基线** | E5 是 flat runtime，不是能力基线；还缺外部系统和统一 adapter。 |

---

## 2. 逐条核对

| 外部清单条目 | 判断 | 依据 / 修正 |
|---|---|---|
| 感知驱动完整 L2 测试框架 | ✅ 真缺口，但比清单说的近 | `pss/detect/groundingdino.py:249` 已生成 crop embedding。带非空 `emb` 的 bundle：`s1_hard` 556 行、`s1_occlusion_real` 460 行、`s2_twins` 407 行、`s1_camera_real` 285 行。README 中“当前 dev bundles 的 `obs_detections.emb` 为空”已经过期。 |
| 接入更多异构专家 | ✅ 但措辞不准 | 模型并不缺；缺的是不兼容宽度专家的注册、校准与准入路径。E11 已证明 3 族 zero-shot 0.186 → k=1 0.936。 |
| 真实机器人硬件部署 | ✅ 但当前不可执行 | 论文 §7 已列为限制。可先做真实 RGB-D / R-Gate 回放，作为外部效度的中间台阶。 |
| 开发非 PSS 竞争性基线 | ⚠️ 方向对，定性要改 | E5 是 **contract-free flat runtime**，不是“内部阉割 PSS”。它证明 suite 不是定制试卷，但不构成能力基线。 |
| 改进专家准入与校准 | ✅ 有价值 | 1-shot ridge 0.922 < prototype rebuild 0.964，说明校准不是万能药；在线校准 + drift 是自然下一步。 |
| 增强 TMS 版本化知识库 | ✅ 准确 | `TMSFactStore` 已包生产 `MultiSourceFactStore`（E10 全 1.00），但 call site 未全部迁移。 |
| 有界推理与尾延迟优化 | ✅ 精确 | `eval/property_preregistration.md:185`：2/14 traces 出现 52–63 ms 单帧尖峰，BAR 0.9993；实体预算 100%。 |
| 类型化关系自动分工 | ✅ 自然延伸 | E3 已给出 `hybrid_diag`；下一步是 learned gate + 域外自动降级。 |
| 校准鲁棒性 / 对抗样本 | ⚠️ 要拆开 | 普通 OOD 拒识已经很强（E16 AUROC 0.993）。真正的风险是**校准集投毒 / 对抗 embedding**，不是普通 OOD。 |
| 百万级实体 | ⚠️ 太远 | E8 只到 500 entities / 2000 relations、1000 facts；facts update 已近线性（500→30.5 ms，1000→52.9 ms）。先做 1e4–1e5 和增量依赖传播。 |
| 分布式 PSS / 多 Agent | ❌ 不建议现在做 | 不是论文缺口，单机契约还没闭；会引入一致性、同步、跨设备权限问题，成本高、证据链弱。 |
| 形式化证明契约不变量 | ⚠️ 可做但后置 | 适合作为 v0.2 的差异化。先做 TLA+/Alloy 的 swap + rollback + TMS 小模型，不要一上来证明全系统。 |
| LLM 作为语言专家接入 | ✅ 有叙事价值 | 应做成“一个专家 + TMS 冲突校验”的最小实验，不要扩成新架构。 |

两点需要特别修正：

- **“热插拔、TMS 更新、回滚只在 L0/L1 验证”不完全对。** E19/E20 已在 L2 replay 中做了 KB 更新、planner swap、审计回滚、遮挡绑定和冲突弃权。真正缺的是**感知级 expert swap**。
- **“当前负向对照组是内部阉割版本”** 应改为“E5 是 contract-free 内部基线；还缺外部竞争性系统”。这个区别审稿人会追问。

---

## 3. 清单遗漏的三个高价值缺口

### 3.1 公开可复现工件（最该先做）

`.gitignore` 排除了 `bundles/`、`data/`、`checkpoints/`、`_external/`。一个干净的 clone **无法复现 E12、E7、L2 replay 的任何数字**，但论文和 Zenodo DOI 已经把 bundle replay 当作证据。

现有可发布资产很小：33 个 bundle 去掉渲染图像和派生输出后，压缩包约 5.5 MB（含 4 个带 embedding 的 bundle）。这不是“做不了”，而是“还没做”。

### 3.2 感知级 mid-stream expert swap

现在有：

- 运行时级 KB 更新 / planner swap（E19）；
- runtime 跨层消融（E20）；
- 带 embedding 的离线 replay（E12/E7）。

没有：**在一段真实检测流中间换入一个感知专家**。这是论文“热插拔”主张在 L2 的最后一公里。

### 3.3 协议的外部有效性

论文自己说“不是竞争性 benchmark”。要把 protocol 单独作为方法贡献，需要：

- 一个 `SystemUnderTest` adapter 接口；
- 至少一个外部系统（哪怕是简单的 transformer-state baseline）跑同一套 property；
- 预注册它应该在哪几项失败。

---

## 4. 推荐任务

### P0：直接改变论文能说的话

#### F1 公开复现包

- **状态（2026-11-26）**：已完成并发布到 GitHub Release `v0.1.1`；E12 在干净临时目录从公开 URL 下载、校验、复现 PASS。归档：`pss-l2-bundles-v0.1.1.tar.gz`，SHA256 `9bd50f41…7984e`。
- **问题**：公开仓库缺少 bundle，E12/E7 不可复现。
- **做法**：
  - 发布 33 个 bundle（约 5.5 MB 压缩后）到 GitHub Release / Zenodo / Hugging Face，不提交进 git；
  - 附 manifest、SHA256、下载脚本；
  - 增加 `make reproduce-l2`，一条命令跑 E12 valid-bundle 表和 4-bundle 感知表；
  - 修正 README 中“`obs_detections.emb` 为空”的过期表述。
- **验收**：干净 clone + 一条命令，在 10 分钟内复现 full 1.000 / no_memory 0.827 / no_entity 0.270，以及 4-bundle 感知结果。
- **工作量**：1–2 天。

#### F2 感知级 mid-stream expert swap

- **状态（2026-11-26）**：已完成并合入 master（`b8e7140`）。4 个带 `emb` 的 bundle 上正例 SSR/RBE/INT/RCR 全 1.000；naive 负控制 RCR 4/4 为 0，INT_post 在 2/4 下降。报告：`reports/claim_gaps/perception_swap/REPORT.md`。限制：单 swap 点、单 seed；INT 为 baseline-relative；guard 只包 ingest，端到端 p99 最高约 11 ms。
- **问题**：L2 的 expert swap 仍停留在 runtime 级或离线 replay，没有在真实检测流中间换入。
- **做法**：
  - 在 4 个带 `emb` 的 bundle 上，于第 t 帧换入另一视觉族（例如 `smallcnn` → `dinov2`）；
  - 只向 PSS 喂 `obs_detections.parquet`，不打开 GT；
  - 测 SSR / RBE / INT / RCR，并对比 oracle pose 与检测 pose 的差异；
  - 加负控制：naive swap 必须在 INT/RCR 上失败。
- **验收**：swap 成功 100%、回滚精确、非目标模块不变；报告检测位姿 vs GT 位姿的 delta。
- **工作量**：3–5 天。

#### F3 延迟 guard

- **状态（2026-11-26）**：已完成并合入 master（`d49e6e7`，E21）。`DeadlineGuard` + `LatencyIsolation` + 确定性慢阶段注入；报告见 `reports/claim_gaps/latency_guard/REPORT.md`。
- **结果**：注入 30 ms 慢阶段时 BAR 0.944 → 0.999、p99 31.3 → 8.9 ms、ID retention 1.000；但 4/57 次无 guard 运行仍有 52–67 ms 尖峰，残余来自 OS 抢占 guard 线程。**硬 10 ms 墙钟上界未达成**；达成的是“bounded work + anytime degradation at the API boundary”。
- **问题**：2/14 traces 出现 52–63 ms 单帧尖峰，10 ms 界限下 BAR=0.9993。
- **做法**：
  - 在 replay 路径加 deadline guard 与 anytime 降级；
  - 记录 p95 / p99 / p999；
  - 加 guard-off 负控制，验证关闭 guard 时按预期超界。
- **验收**：10 ms 下 BAR=1.00；或给出有界降级曲线，尾延迟不再无界。
- **工作量**：2–3 天。

### P1：把“协议”变成可独立发表的方法

#### F4 异构宽度 expert adapter

- **状态（2026-11-26）**：已完成并合入 master（`8b7164b`）。DINOv2 384 / CLIP 768 / WT-MM 64 三宽度准入契约一致（reject/accept/review），vision k=1 = 0.983 / 1.000；报告：`reports/claim_gaps/families_width/REPORT.md`。CLIP 实际用 ViT-B/16（768），ViT-L/14 权重未齐；WT-MM k=1 仅 0.38–0.45（专家本身限制，非契约失败）。
- **问题**：E11 只接了 3 个同宽族；DINOv2 384/768、CLIP 768、WT-MM 尚未进 property suite。
- **做法**：
  - 在 registry / property suite 中接入不同 `in_dim` 的专家；
  - 每个专家独立投影头 + 校准集；
  - 补 k=0 / 1 / 5 / 20 的校准曲线。
- **验收**：3 种宽度下准入契约保持；k=1 vision ≥ 0.90；runtime 代码零改动。
- **工作量**：1–2 周。

#### F5 协议包 + 外部 baseline

- **状态（2026-11-26）**：前半段已完成并合入 master（`61edacf`）。新增零 PSS 依赖的 `pss_eval` 协议包、E5 flat 与 transformer-state adapter、runner 和可执行预注册；e5-flat 在 contract 上按预注册失败（SRR/RBE/AUD=0）。设计说明：`docs/plans/PSS-F5-协议包与预注册.md`。后半段（PSS 自身 adapter、真实 bundle 流、更多 property）未做。
- **问题**：E5 是内部 contract-free baseline，suite 尚未对外部系统开放。
- **做法**：
  - 抽出 `pss-eval`，定义 `SystemUnderTest` adapter；
  - 跑一个外部 transformer-state baseline 和 flat baseline；
  - 预注册它应在哪些指标失败。
- **验收**：外部系统按预注册方式通过/失败；suite 不依赖 PSS 内部类型。
- **工作量**：2–3 周。

#### F6 在线校准 + 投毒负控制

- **问题**：当前是一次性小集合校准；校准集投毒未被测试。
- **做法**：
  - 流式校准 + drift 检测；
  - 构造 poisoned calibration 负控制；
  - 量化校准开销–性能 Pareto 曲线。
- **验收**：用 ≤N 个样本恢复 one-shot 的 ≥90%；投毒能被检出或触发弃权。
- **工作量**：1–2 周。

### P2：扩展，等 P0/P1 闭合后再做

| 编号 | 任务 | 说明 |
|---|---|---|
| F7 | TMS 全量迁移 + 1e4 facts | 增量依赖传播、快照成本、call-site 迁移 |
| F8 | 关系层 learned gating + 域外降级 | E3 的自然延伸，避免朴素 hybrid 的 0.503 |
| F9 | 多 query + coverage loss 的 fact 选择 | E4 只测了单 query |
| F10 | 真实数据回放 / 真机 | 先接 R-Gate 或真实 RGB-D，再谈硬件 |

---

## 5. 不建议现在做

| 方向 | 原因 | 什么时候再考虑 |
|---|---|---|
| 分布式 PSS / 多 Agent | 不是论文缺口；单机契约尚未闭合，multi-agent 会引入一致性与同步问题 | 单机 swap / rollback / TMS 在 L2 稳定之后 |
| 百万级实体 | 当前只到 500 / 2000，先把 1e4–1e5 的退化曲线测出来 | 有明确的规模化用户或 benchmark 需求时 |
| 全系统形式化证明 | 成本高，且不会补当前的实证缺口 | 作为 v0.2 差异化，从 swap + rollback 小模型开始 |
| LLM expert zoo | 叙事吸引力大，但容易稀释“runtime + protocol”的主线 | 先做一个 LLM-as-expert + TMS 冲突校验的最小实验 |

---

## 6. 证据索引

| 主题 | 证据 |
|---|---|
| 论文 Limitations / Conclusion | `paper/main.tex:801`、`paper/main.tex:826` |
| L2 尾延迟 2/14 | `eval/property_preregistration.md:185` |
| 感知 embedding 生成 | `pss/detect/groundingdino.py:249` |
| 带 `emb` 的 bundle | `bundles/dev/s1_hard`、`s1_occlusion_real`、`s2_twins`、`s1_camera_real` |
| E12 记忆消融 | `reports/claim_gaps/l2_memory_all/REPORT.md`（valid 19/33：1.000 / 0.827 / 0.270） |
| E8 规模 | `reports/claim_gaps/scale/REPORT.md`（500 entities/2000 relations；1000 facts 52.9 ms） |
| E11 专家族 | `reports/claim_gaps/families/REPORT.md`（zero-shot 0.186；k=1 0.936） |
| E3 hybrid 关系 | `reports/claim_gaps/hybrid_relations/REPORT.md`（hybrid_diag 0.857 vs geo 0.828 @ noise 0.15） |
| E16 开放集 | `reports/claim_gaps/openset/p2_openset.md`（AUROC 0.993 / reject 0.958） |
| E19/E20 runtime | `reports/claim_gaps/l2_runtime_updates/REPORT.md`、`reports/claim_gaps/runtime_cross_layer/runtime.md` |
| E21 延迟 guard | `reports/claim_gaps/latency_guard/REPORT.md` |
| F2 感知级 swap | `reports/claim_gaps/perception_swap/REPORT.md` |
| F4 异构宽度 | `reports/claim_gaps/families_width/REPORT.md` |
| F5 协议包 | `pss_eval/`、`docs/plans/PSS-F5-协议包与预注册.md` |
| 可复现性缺口 | `.gitignore`（`bundles/`、`data/`、`checkpoints/`、`_external/`） |

---

## 7. 建议的下一步

如果只做一件事：**F1 + F2 一起做**。

- F1 让公开 release 真正可复现；
- F2 把 Limitations 里的“L2 依赖 oracle pose / 感知级 swap 未做”直接变成 Results。

这两项的边际收益高于再写一份 Future Work 段落，也高于任何 P2 扩展。

建议顺序：**F1 → F3 → F2 → F4 → F5 → F6**。

- F1 先建立可复现基线；
- F3 成本低、能立刻闭合已知负结果；
- F2 依赖 F1 的数据包和 F3 的延迟基线；
- F4/F5/F6 再进入“协议作为方法贡献”的阶段。

**进度（2026-11-26）**：F1、F3、F2、F4 已完成；F5 前半段完成。剩余：F6（在线校准 + 投毒负控制）、F5 后半段（PSS 自身 adapter + 真实数据流），之后才是 P2。
