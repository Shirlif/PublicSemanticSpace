# 视觉模型混淆检查（S1-hard）

> 日期：2026-11-25
> 目的：回答“结果是否被视觉模型输出误导”，而不是只做口头保证。

## 1. 已经有的保护

### GT 防火墙（代码级）

- `replay_bundle()` 只读 `obs_detections.parquet` / `obs_frames.parquet`，不读 `gt.parquet`。
  测试：`tests/test_gt_firewall.py` 把 GT 文件删掉后 replay 仍能跑。
- tracker 的关联代码不使用 `Observation.gt_id`（`pss/memory/tracking.py` 明确注释：
  “Association never uses GT ids”）。
- 评分侧才读 GT，并用 3D 位置做 pred→GT 匹配；GT 不回流到 PSS。
- Oracle 档是显式例外：`detect_oracle` 从 GT 生成框，`emb_ref=instance_id`，
  replay 用 one-hot appearance。它是**上界标定**，不能当能力证据。

### 同输入对比

PSS / B1 / B2 共用同一份 `obs_detections.parquet`、同一 GT、同一 0.05 m 匹配门限。
检测器偏差对三者是共同的，比较的是同一视觉输入下的身份逻辑。

### 检测元数据冻结

`detect_meta.json` 记录模型、权重 sha256、prompt、box/text threshold、NMS IoU、pose mode。
换检测器/阈值必须重跑，不能混表。

## 2. 仍然存在的风险

1. **单一检测器 / 单一嵌入模型**：GroundingDINO + DINOv2 的偏差会进入所有方法。
2. **PSS 使用 appearance，B1/B2 使用得少或不用**：PSS 的小幅领先可能来自嵌入，
   而不是结构。
3. **oracle 档直接使用 GT 身份**：任何 oracle 数值都不能作为能力证据。
4. **单 seed / 单 episode**：没有置信区间。
5. **session replay 在同一进程读 GT 做评分**（S6）：运行时是离线评分器，不是
   进程级防火墙；正式跑测应把评分拆到独立进程/文件权限。

## 3. 本次控制实验：打乱 appearance embedding

在 S1-hard（12 物体、真实检测、长遮挡、改位、孪生）上：

| 条件 | IDF1 | retention | duplicate | reappearance |
|---|---|---|---|---|
| PSS full（正常 DINOv2） | 0.5874 | 0.7088 | 0.7692 | 0.2308 |
| PSS full + 每帧打乱 embedding | **0.5159** | 0.6317 | 0.9231 | 0.1538 |
| B2 SORT（位置+0.3外观） | 0.5179 | 0.6032 | 0.5217 | 0.0769 |
| B1 text（无外观） | 0.5179 | 0.6032 | 0.5000 | 0.0769 |

**结论**：打乱外观后，PSS IDF1 从 0.587 掉到 0.516，正好落到 B1/B2 水平。
也就是说，PSS 在这条硬场景上的 **+0.07 IDF1 几乎全部来自 DINOv2 外观通道**，
不是来自“持久实体/记忆结构”本身。

这直接回答了“是否被视觉模型误导”：

- 结果确实**高度依赖视觉嵌入**；
- 当前不能把 +0.07 解释成 PSS 架构优势；
- 要证明结构优势，必须在**相同外观输入**下比较，或者把外观因素消掉。

## 4. 还需要做的控制

1. **相同外观输入的基线**：B2 用与 PSS 相同的 appearance 权重/同一嵌入，
   再看结构差异是否还存在。
2. **appearance 置换/随机化**：已做“帧内打乱”，还应做“全局随机向量”“全零”
   两种，确认不是打乱方式导致的伪影。
3. **检测器敏感性**：box threshold 0.2/0.3/0.4、prompt 变化、NMS 0.3/0.5，
   报 P/R 和指标；若结论随阈值翻转，说明检测器主导。
4. **第二个检测器**：RT-DETR 或颜色阈值检测器，复现同一结论。
5. **oracle-hard 结构隔离**：S1-hard 用 oracle 检测（one-hot appearance），
   让 PSS 与 B1/B2 共享完美外观，只比“长遮挡后的持久槽/重入”；若 PSS 仍不赢，
   结构优势不成立。
6. **多 seed**：5 seeds + 配对 bootstrap，报 CI。
7. **运行时-评分进程分离**：S6 的 GT 读取移到独立评分进程。

## 5. 当前可下的结论

- S1-hard 的 0.587 是“**GroundingDINO + DINOv2 + PSS full**”的联合结果，
  不是 PSS 结构单独的结果。
- 打乱外观后 PSS ≈ B2，说明在这条基准上，**PSS 的结构没有表现出超出
  通用跟踪器 + 外观嵌入的增益**。
- 在拿到第 4 节的同输入控制之前，不能声称“PSS 架构有优势”。
