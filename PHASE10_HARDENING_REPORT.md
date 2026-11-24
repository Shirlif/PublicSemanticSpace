# Phase 10 加固报告（1–4）

> 日期：2026-11-20
> 对应预注册：`eval/preregistration.md`
> 目标：按 Phase 10 的 K1–K4，先修上游的四件事，而不是继续加模块。

---

## 1. 更难的遮挡基准（已做）

新增 `data/hard_occlusion`：

- 6 条序列 × 12 帧；第 4–7 帧用整块幕布罩住场景；
- 验证：覆盖帧的观测数为 **0**（此前小遮挡条只有部分遮挡）；
- 生成：`python -m pss.sim.scripts.generate_hard_occlusion --out data/hard_occlusion`

### E3 重跑（硬遮挡）

| 变体 | ID retention | switches |
|---|---:|---:|
| main | **1.000** | 0.00 |
| pose_only | 1.000 | 0.00 |
| appearance_only | 1.000 | 0.00 |
| **B1 文本状态代理** | **0.500** | 1.00 |
| no_scene_graph | 0.125 | 7.00 |

**K3 由触发变为未触发**：主方法相对 B1 的优势 **+0.500**。

需要诚实标注：B1 仍是文本状态代理，不是真实 VLM；但这个基准至少让"遮挡后身份丢失"的结构差异显现出来。

---

## 2. RD2 真实数据脚手架（已做，数据待采）

| 交付 | 路径 |
|---|---|
| Schema（帧/实例/同步/位姿/触觉） | `pss/rd2/schema.py` |
| 校验器（同步 ≤5 ms、consent、隐私、形状） | `pss/rd2/validate.py` |
| 保存/加载 + 转仿真 sample 格式 | `pss/rd2/adapter.py` |
| 采集协议（硬件/规模/切分/隐私/R-Gate） | `pss/rd2/protocol.md` |
| Mock 域偏移生成器 | `pss/rd2/mock.py` |
| CLI | `pss/rd2/scripts/validate.py`、`make_mock.py` |

Mock 数据集：8 episodes，RGB 亮度/噪声偏移、深度 5 mm 噪声、触觉 0.8×+0.05 偏移、位姿 3 mm 噪声；校验 **8/8 通过**。

> 这不是真实数据。RD2 真机采集与 E0 R-Gate 仍然未运行。

---

## 3. VLM 基线适配器（已做，真实 API 未调用）

| 交付 | 路径 |
|---|---|
| `VLMClient` 协议 + 文本解析 | `pss/eval/baselines/vlm.py` |
| Mock VLM（读 GT 可见性，模拟"只看图像"） | 同上 |
| 真实 VLM 可调用适配器（需调用方提供 API/授权） | `CallableVLMClient` |
| 文本状态跟踪基线 | 复用 `TextStateTracker` |

硬遮挡下 Mock VLM 结果：

| 指标 | 值 |
|---|---:|
| ID retention | 0.500 |
| ID switches | 1.00 |
| 重现成功 | 0.000 |
| 唤醒延迟 | 4 帧 |

> 真实 VLM 会把图像发送到外部服务，需要显式授权与 API key；本次**没有**调用任何外部 API。

---

## 4. 真实预训练视觉骨干（已做）

- 安装 `torchvision 0.25.0+cpu`，**torch 保持 2.10.0+cu126 未变**。
- ResNet18 预训练权重下载完成但 **hash 校验失败**（expected `f37072fd`，got `2cabfea…`），未使用。
- SqueezeNet1.1 预训练权重下载并校验成功，作为真实 ImageNet 预训练骨干。

| 骨干 | train | val | test |
|---|---:|---:|---:|
| SmallCNN（从零） | 1.000 | 1.000 | 0.958 |
| **SqueezeNet1.1（预训练）** | 1.000 | 0.969 | **0.979** |

- 提升 +2.1 个点；不大，但方向正确。
- 权重：`checkpoints/expert_pretrained_v1/vision_pretrained.pt`
- 命令：`python -m pss.encoders.scripts.train_pretrained_expert --backbone squeezenet1_1 --pretrained true`

> 尚未把 SqueezeNet 特征接回 Phase 4/5 全链路重训；这需要重跑投影头与原型库，属于下一步。

---

## 5. 更新后的杀死标准

| ID | 状态 | 说明 |
|---|---|---|
| K1 | ✅ 触发 | D=128 K=20 = 0.769；升 D=512 或逐条存储 |
| K2 | ✅ 触发 | V-T R@1 = 0.510 < 0.60；触觉仍弱 |
| K3 | ❌ **未触发** | 硬遮挡下 main 1.000 vs B1 0.500，+0.500 |
| K4 | ✅ 触发 | P4 V-T gate 未过 |

E5 其余 10 项 gate 仍通过。

---

## 6. 结论

1. **硬遮挡基准让 H1 第一次得到支持**：显式实体槽 + 休眠/re-ID 在完全遮挡后保住了身份；文本状态与无场景图基线都掉了。
2. **但 B1 是代理，不是真实 VLM**；H1 的最终结论仍需要真实 VLM 对比。
3. **触觉是当前唯一未过 gate 的模态**：V-T 0.510，K2/K4 仍然触发；RD2 真实同步数据是解药。
4. **预训练骨干有小幅收益**（0.958 → 0.979），但不足以修复触觉对齐。
5. RD2 schema/协议/校验器/mock 已就绪；真机采集与 E0 仍是开放项。

---

## 7. 下一步

1. **RD2 真机采集**：按 `pss/rd2/protocol.md` 采 500–2000 条实例级同步 episode。
2. **真实 VLM 基线**：用 `CallableVLMClient` 接一个已授权的 VLM，重跑硬遮挡 E3。
3. **SqueezeNet 全链路重训**：Phase 4 投影 + Phase 5 原型 + Phase 10 E2/E3。
4. **D 决策**：若关系库需要 K≥20 压缩存储，升 D=512；否则 128 + 逐条。
