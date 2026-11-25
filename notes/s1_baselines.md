# S1 外部基线对比（B1 文本状态 / B2 SORT-lite）

> 日期：2026-11-25
> 数据：`s1_occlusion_20`（20 cubes, 5 seeds, oracle）、`s1_occlusion_real`（4 cubes, 1 seed）、
> `s1_camera_real`（4 cubes + 相机轨迹, 1 seed）
> 性质：dev，样本量不足以做显著性声明。

## 方法定义

- **PSS full**：当前完整实体层（pose + appearance；oracle 档用 one-hot，real 档用 DINOv2）。
- **B1 text-state**：没有持久实体槽；每帧检测与上一帧按位置/类别做匈牙利匹配，匹配不上就新建 ID。
  这是“文本状态/无持久槽”的代理，不是真实 VLM。
- **B2 SORT-lite**：恒速预测 + 匈牙利匹配 + 位置门限 + max_age=10；real 档可选 0.3 外观项
  （DeepSORT-lite），oracle 档无外观。

所有方法共用同一批 `obs_detections.parquet`、同一 GT、同一匹配门限（0.05 m）和同一指标实现。

## 结果 1：S1 20 cubes / 5 seeds / oracle

| method | IDF1 | retention | switches | duplicate | reappearance |
|---|---|---|---|---|---|
| PSS full | **1.0000** | 1.0000 | 0.00 | 0.0000 | 1.0000 |
| B1 text | 0.6038 | 0.6015 | 1.39 | 0.5816 | 0.0000 |
| B2 SORT | **1.0000** | 1.0000 | 0.00 | 0.0000 | 1.0000 |
| no_entity | 0.0113 | 0.0114 | 87.37 | 0.9887 | 0.0000 |

- PSS 显著优于 B1（IDF1 +0.396，CI 不跨 0）。
- **PSS 与 B2 打平**：oracle 下物体静止、遮挡间隔短，SORT 的位置预测足够。
  这说明这个场景**太简单**，不能证明 PSS 优于通用跟踪器。

## 结果 2：S1 4 cubes / 1 seed / frozen_real（静态相机）

| method | IDF1 | retention | duplicate |
|---|---|---|---|
| PSS full | 0.8942 | 1.0000 | 0.0000 |
| B1 text | 0.8942 | 1.0000 | 0.0000 |
| B2 SORT | 0.8942 | 1.0000 | 0.0000 |

- 三方打平。检测质量足够高、物体运动小，无持久槽的 B1 也能靠逐帧最近邻保持 ID。
- 这说明：**在简单静态场景，PSS 没有可测的优势**。

## 结果 3：S1 camera trajectory / 1 seed / frozen_real

| method | IDF1 | retention | switches | duplicate | reappearance |
|---|---|---|---|---|---|
| PSS full | **0.7406** | 0.8725 | 0.75 | 0.4000 | 0.7500 |
| B1 text | 0.5597 | 0.6120 | 2.00 | 0.6364 | 0.2500 |
| B2 SORT | 0.5631 | 0.6195 | 2.75 | 0.6364 | 0.2500 |
| no_entity | 0.0137 | 0.0181 | 59.25 | 0.9834 | 0.0000 |

- 相机移动时，PSS 比 B1/B2 高约 **+0.18 IDF1**、**+0.25 retention**、**+0.50 reappearance**。
- 但只有 **1 个 episode**，配对 bootstrap 的 CI 退化成一个点，**不能声称显著**。
- 方向性解释：视角变化让“逐帧位置匹配”失效，PSS 的外观通道和持久槽更有用。

## 结论

1. **现在不能声称 PSS 架构整体优于通用跟踪器。**
   - oracle 20-cube：B2 打平；
   - static real：B1/B2 打平；
   - camera real：PSS 领先，但只有 1 episode。

2. **能说的方向性结论**：
   - 无持久槽（B1/no_entity）在遮挡/gap 下明显退化；
   - 视角变化时，PSS 的 appearance + 持久槽比纯位置跟踪更有优势；
   - 但需要更多 seeds 和更难场景才能形成统计证据。

3. **下一步要形成可声称的优势，必须补**：
   - S1 20 cubes + 相机轨迹 + `frozen_real` × 5 seeds（当前只有 4 cubes ×1）；
   - 更长的遮挡 gap、物体运动、孪生干扰；
   - B2 调参公平化（外观权重、max_age）并报告敏感性；
   - 5 个以上 episode 的配对 bootstrap CI。

## 产物

- `scorecards/s1_occlusion_20_vs_baselines.md/json`
- `scorecards/s1_occlusion_real_vs_baselines.md/json`
- `scorecards/s1_camera_real_vs_baselines.md/json`
- `scripts/run_s1_baselines.py`
- `pss/baselines/trackers.py`
