# 阶段报告索引

> 按 Phase 归档的工程报告。这些报告记录“当时做了什么、结果如何、边界在哪”，不是运行入口。
> 返回文档总索引：[`../README.md`](../README.md)。

---

## Phase 0–10：骨架、冒烟与预注册评测

| 报告 | 内容 |
|---|---|
| `PHASE0_REPORT.md` | 契约、绑定算子、三个基线 demo、绑定稳定性基准 |
| `PHASE1_REPORT.md` | Hadamard vs HRR、槽池与 rebind、38 个测试 |
| `PHASE2_REPORT.md` | MuJoCo 桌面场景与数据集 v1（原始几何体、合成触觉） |
| `PHASE3_REPORT.md` | 视觉/触觉/几何专家冒烟（管线验证，不是能力结论） |
| `PHASE4_REPORT.md` | 跨模态投影与类别级对齐（V-T / V-G / T-G） |
| `PHASE5_REPORT.md` | 原型库、分层原型树、开放集阈值与冲突标记 |
| `PHASE6_REPORT.md` | 3D 场景图、追踪、工作记忆、遮挡 ID 保持 |
| `PHASE6_DECOMPOSITION_REPORT.md` | 场景图解体规则（计划 7.4） |
| `PHASE6_V2_REPORT.md` | canonical 栈验证（category_pretrain + phase45 projection） |
| `PHASE7_REPORT.md` | 关系编码、动作算子库、A* 规划、世界模型 |
| `PHASE8_REPORT.md` | 动作头、文本头、价值/目标头；文本头消融 |
| `PHASE9_REPORT.md` | 安全护栏、四级降级 FSM、审核流水线、审计回放 |
| `PHASE10_REPORT.md` | 预注册评测 E1–E6；杀死标准 K1–K4 全部触发 |
| `PHASE10_HARDENING_REPORT.md` | 完全遮挡基准、RD2 schema、VLM mock、SqueezeNet 骨干 |
| `PHASE10_PUBLIC_DATA_REPORT.md` | LeRobot v3.0 / WT-MM 公共数据适配 |
| `PHASE10_T1_INSTANCE_REPORT.md` | T1 实例级对照（合成数据） |
| `PHASE10_WTMM_TACTILE_REPORT.md` | WT-MM 触觉编码器与跨传感器迁移 |
| `CATEGORY_PRETRAIN_REPORT.md` | 类别识别预训练 v2 主线（20 类、融合 0.969） |

## Phase 11–15：五优先级

| 报告 | 内容 |
|---|---|
| `PHASE11_ENCODER_PRIORITY_REPORT.md` | 第一优先：换编码器、投影校准、开放集/re-ID |
| `PHASE12_TYPED_RELATIONS_REPORT.md` | 第二优先：typed relations（spatial/semantic/tactile） |
| `PHASE13_ACTIVE_WRITE_POLICY_REPORT.md` | 第三优先：active-set 与 write policy |
| `PHASE14_LANGUAGE_HEAD_REPORT.md` | 第四优先：语言头与结构化状态/工具/审计 |
| `PHASE15_BILEVEL_REPORT.md` | 第五优先：bilevel / 联合训练与一致性约束 |

## Phase 16–18：认知运行时收敛

| 报告 | 内容 |
|---|---|
| `PHASE16_RUNTIME_REPORT.md` | `CognitiveState`、`EntityLayer`、`RelationFactLayer`、`RuntimePlanner`、`OutputRouter` |
| `PHASE17_RUNTIME_INTEGRATION_REPORT.md` | 学习模块接入 runtime；versioned operators；70 帧遮挡 |
| `PHASE18_LEARNED_DEFAULTS_REPORT.md` | learned relations / active policy / write policy 设为默认 |

## Phase 19–24：Gap closure

| 报告 | 内容 |
|---|---|
| `PHASE19_GAP_CLOSURE_ENCODER_PROJECTION.md` | 编码器与投影默认路径（frozen DINOv2 + 校准） |
| `PHASE20_GAP_CLOSURE_BINDING.md` | object attention 作为可切换绑定选项 |
| `PHASE21_GAP_CLOSURE_RULES.md` | 可学习、可版本化的关系规则 |
| `PHASE22_GAP_CLOSURE_PLANNING.md` | learned cost-to-go heuristic + A* |
| `PHASE23_GAP_CLOSURE_LANGUAGE.md` | 本地 Qwen2.5-1.5B decoder 接入 |
| `PHASE24_GAP_CLOSURE_AUDIT.md` | runtime 四档权限（read_only/service/operator/admin） |

## Phase 25–30：评测套件与真实数据

| 报告 | 内容 |
|---|---|
| `PHASE25_EVAL_SUITE.md` | 评测套件 Phase A 骨架（schema → adapter → metrics → report） |
| `PHASE26_OFFICIAL_DATA.md` | TAO / CLEVR 官方数据接入与 oracle baseline |
| `PHASE27_CLEVR_IMAGE_RELATION.md` | CLEVR 图像关系与 CoGenT 属性组合 |
| `PHASE28_TAO_PSS.md` | TAO 官方帧 + PSS tracker（IDF1 0.979） |
| `PHASE29_DETECTOR_REPLACEMENT.md` | 用检测器替换 GT boxes（CenterNet-lite、COCO person） |
| `PHASE30_LEARNED_REID_DESIGN.md` | learned re-ID 设计与实施计划（设计稿，未实现） |
| `PHASE45_REPORT.md` | Phase 4/5 材质感知投影验证（融合 0.979、AUROC 0.993） |

---

## 读法建议

- 只关心结论：先读根目录 `README.md` 和 `../experiments/PSS-五优先级最终结论.md`。
- 关心某个能力：按主题跳转，例如实体/追踪看 Phase 6、10、17、20、28；关系看 Phase 7、12、21；规划看 Phase 7、22；语言看 Phase 14、23；审计看 Phase 9、24。
- 关心评测口径：读 Phase 10、25、26 和 `eval/property_preregistration.md`。
- 历史报告里的数字有日期和边界，不要脱离报告上下文直接引用。
