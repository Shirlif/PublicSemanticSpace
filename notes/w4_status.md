# P3 / W4–W5 进度 — 真实检测 + 孪生（S2）

> 开始：2026-11-25
> 状态：T19、T20、T21、T22、T23 完成（S2 的 abstain 指标留缺口）。

## T19 GroundingDINO Swin-T（离线批处理）

- 模型：`IDEA-Research/grounding-dino-tiny`（689,359,096 B，
  sha256 `1a2412ef99bd74bcd3c2a246fa1e48581f8889a1300c9051974741314fc042f3`）
  放在 `_external/models/grounding-dino-tiny/`。
- 外观嵌入：`facebook/dinov2-small` CLS（384-d，L2 归一化），写进
  `obs_detections.parquet` 的新列 `emb`。
- 3D 位姿：bbox 中心 + 深度图反投影（相机内参由 fov/分辨率计算）。
  深度打到立方体顶面，故 z≈0.81；评分侧匹配门限 0.05 m。
- 采集：`pss sim run ... --store-images`；相机 120 Hz 传感器、按 10 Hz 观测
  tick 显式 render；暖机阻塞到首帧，W1 100/100 帧齐全。
- 25 GB 级 3.4 GB 下载教训复用：68 万级文件用并行 Range（~5 MB/s）。

## T20 noisy 档

- `pss/detect/noise.py`：按可见率调制的漏检、位姿高斯噪声、类别混淆、
  每帧误检；默认 profile 与设计 §3.5 一致。
- `pss detect <bundle> --tier noisy --seed 0`。

## T21 孪生资产

- `specs/s2_twins.yaml`：两组孪生（组内同色同尺寸，`twin_group`），
  `twin_swap` 故障在 step 300 交换 tg_a 两个实例的位姿。
- runner 新增 `twin_swap` 故障注入。

## T22 三档对比（S1，4 cubes / 1 seed / dev）

| tier | IDF1 | retention | switches | duplicate | reappearance |
|---|---|---|---|---|---|
| oracle | 1.0000 | 1.0000 | 0.000 | 0.0000 | 1.000 |
| noisy | 0.9424 | 1.0000 | 0.000 | 0.0000 | 1.000 |
| frozen_real | 0.8942 | 1.0000 | 0.000 | 0.0000 | 1.000 |

- real 与 oracle 差距 0.106 > 0.1，满足“检测器确实接入”的止损检查。
- 表：`scorecards/S1_tiers.md` 与 `S1_{oracle,noisy,real}.md`。

## T23 S2 表

| tier | IDF1 | duplicate_rate | twin_ambiguity_rate | abstain_precision |
|---|---|---|---|---|
| oracle | 1.0000 | 0.0000 | 0.0000 | n/a |
| frozen_real | 0.8190 | 0.6667 | 0.0000 | n/a |

- `twin_ambiguity_rate`（DINOv2 同组余弦 ≥0.98）为 0：外观可区分，但
  `twin_swap` 仍造成 IDF1 下降、duplicate 升到 0.667——位置交换后的硬关联
  是当前失败模式。
- **缺口**：`abstain_precision` 未实现；PSS 还没有孪生弃权策略。
  下一步是在 `UncertaintyService`/tracker 加入“同组外观不可分则弃权”，
  并用 S2 的 twin_swap 重测。表：`scorecards/S2.md`。

## 复现

```powershell
$env:NO_PROXY="*"; $env:OMNI_KIT_ACCEPT_EULA="YES"
# 1) 采图
D:\isaac\env\Scripts\python.exe -m pss sim run specs\s1_occlusion.yaml --force --store-images
# 2) oracle / noisy / real
python -m pss detect <bundle> --tier oracle
python -m pss detect <bundle> --tier noisy --seed 0
python -m pss detect <bundle> --tier frozen_real
python -m pss replay <bundle>
python -m pss score --suite <suite> --label <label>
```

## 已知限制

- 三档表是 dev 4-cube 场景，不是 official 20-cube × 5 seeds；official 跑测在 W13–W14。
- 深度反投影把物体顶面当作中心，z 偏差约 2 cm；official 前应改成 bbox 内深度
  分位数 + 物体包围盒先验。
- GroundingDINO 的 `labels` 在新版 transformers 返回整数 id，代码优先用
  `text_labels`。
- S2 的 abstain 策略、孪生资产真实纹理（当前同色同尺寸）仍是缺口。
