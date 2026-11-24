# RD2 真实同步数据采集协议

> 目标：产出**同一实例、同一时刻**的视觉 / 触觉 / 位姿同步数据，用于：
> 1. 投影头对齐的裁判集（9.1.4）；
> 2. L2 原型准入的真实样本；
> 3. R-Gate 的 sim2real 冒烟测试。
>
> **真实数据是裁判，不是训练集替代品。** 仿真仍负责规模、真值与可控遮挡。

---

## 1. 硬件

| 项 | 要求 |
|---|---|
| 机械臂 | 桌面级 6/7 轴，重复定位 ≤ 1 mm |
| 视觉 | RGB-D（RealSense D4xx 或同级），≥ 30 FPS，深度 ≤ 2 m 有效 |
| 触觉 | DIGIT / GelSight 类光学触觉，≥ 30 FPS，8×8 或更高接触图 |
| 同步 | 硬件触发或 PTP；记录每帧时钟偏差 |
| 位姿 | AprilTag / 动捕 / SLAM，三选一并记录 `pose_source` |
| 标定 | 相机内参、手眼标定、触觉传感器位姿；写入 `calibration` |

---

## 2. 物体与实例

- 8 类：杯、盘、盒、瓶、布、积木、玩具水果、工具。
- 每类 3–6 个实例；每个实例有唯一 `instance_id`。
- 记录：类别、材质（硬/软、粗/滑）、尺寸、颜色。
- 实例级同步数据必须覆盖：同一实例的视-触-位姿配对，而不是“同类不同物”。

---

## 3. 采集规模

| 项 | 建议 |
|---|---|
| 实例数 | 20–50 |
| episode | 500–2000 |
| 每 episode 时长 | 5–10 s |
| 帧率 | 视觉/触觉 ≥ 30 Hz |
| 同步误差 | ≤ 5 ms（写入 `sync_offset_ns`） |
| 环境 | 至少 3 种光照/背景 |
| 动作 | 静态观察、抓取、放置、推动、轻微形变 |

---

## 4. 字段

每个 episode 目录：

```
ep_0000/
  meta.json      # episode_id, instance_id, category, calibration, consent, privacy_scrubbed, source
  frames.npz     # timestamps_ns, sync_offsets_ns, rgb, depth, mask, tactile, sensor_pose, object_pose, pose_source, environment
```

| 字段 | 形状 | 说明 |
|---|---|---|
| rgb | (T,H,W,3) uint8 | 视觉 |
| depth | (T,H,W) float32 | 米 |
| mask | (T,H,W) bool | 该实例的分割 |
| tactile | (T,8,8) float32 | 接触图 |
| sensor_pose | (T,7) float32 | xyz+quat |
| object_pose | (T,7) float32 | xyz+quat |
| timestamp_ns | (T,) int64 | 单调递增 |
| sync_offset_ns | (T,) int64 | 相对主时钟 |
| pose_source | (T,) str | mocap / apriltag / slam |

---

## 5. 切分

- **实例级**：同一类别的不同实例分到 train/val/test，禁止泄漏。
- **类别级**：留出 2 类作为 `test_category`。
- **环境级**：至少留出一种光照/背景作为 shift 测试。
- 数据卡记录：类别、实例、环境、同步误差分布、采集日期。

---

## 6. 隐私与合规

- 不采集人脸、私密空间、可识别个人信息。
- `consent=true`；`privacy_scrubbed=true`；回传只传特征与统计量，不传原始图像。
- 数据目录访问受控；对外发布前做二次脱敏。

---

## 7. 验收

1. `python -m pss.rd2.scripts.validate --path <dataset>` 全部通过。
2. R-Gate：
   - 仿真训触觉专家在真实接触图上的可分性 ≥ 仿真值 70%；
   - 仿真投影头在真实配对上的跨模态 R@1 ≥ max(0.6, 仿真值×0.7)；
   - 真实噪声下绑定还原误差 ≤ 预算 1.5 倍。
3. 未通过：先修对应环节，不进入 Phase 5/11 的能力声明。
