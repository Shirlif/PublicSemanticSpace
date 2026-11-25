# Dev 全链路 dry-run（4070）

> 日期：2026-11-25
> tag：`dev-smoke`
> 目的：在 4070 上验证 A1–A10 的非 official 链路，不产出 official 数字。

## 链路

```
spec lint → USD build → 1 episode（相机轨迹 + RGB-D）
→ oracle / frozen_real 检测 → replay → score → verify-recompute → report
```

- spec：`specs/s1_camera_motion.yaml`（4 cubes + 绕桌相机轨迹）
- bundle：`bundles/dev/s1_camera_motion/b4bb5f0e/seed0`
- 图像：100 帧 RGB + 100 帧 16-bit depth，0 缺帧
- 检测：
  - oracle → `bundles/dev/s1_camera_oracle/...`
  - frozen_real（GroundingDINO + DINOv2）→ `bundles/dev/s1_camera_real/...`

## 结果（dev，1 episode，不是 official）

| tier | IDF1 | retention | switches | duplicate | reappearance |
|---|---|---|---|---|---|
| oracle | 1.0000 | 1.0000 | 0.000 | 0.0000 | 1.000 |
| frozen_real | 0.7406 | 0.8725 | 0.750 | 0.4000 | 0.750 |

- oracle 是上界标定；real 在移动相机下明显下降，符合预期。
- real 用 **pose+appearance** 关联（`_real_runtime`）；oracle 用 appearance-only。

## 发现并修复的 bug

1. **深度/平面反投影 y 轴符号错误**：图像 v 向下、相机 y 向上，原实现少了负号，
   导致 real 检测的 y 关于相机镜像。
   - 修复前：camera-motion real IDF1 = 0.106；
   - 修复后：0.7406。
   - 影响：所有在此修复前生成的 real 档（S1 4-cube、S2）分数作废，需重跑。
2. **oracle 的 tracker 配置不适合 real 嵌入**：one-hot appearance 下 appearance-only
   最优；DINOv2 嵌入随视角变化，必须用 pose+appearance 默认配置。
3. **标号大小写冲突**：Windows 大小写不敏感，`s1_camera_oracle` 与 `S1_camera_oracle`
   是同一文件；label 改为 `S1_cam_oracle`。

## 产物

- `scorecards/s1_camera_oracle.*`、`s1_camera_real.*`
- `verify/dev-smoke.json`：所有 dev 套件 `ok=true`，`mismatches=[]`
- `reports/dev-smoke/`：freeze/specs/prereg/scorecards/bundles.index/verify/overrides/notes

## 结论

- **4070 可以完整跑流程**，A1–A10 的非 official 机制已打通。
- **不能产出 official 数字**：`device_class=dev`，1 episode，非 5 seeds；A5/A7 仍需 3090 Ti。
- 正式跑测前需要：
  1. 用修复后的 backprojection 重跑所有 real 档（本文件之后已重跑 S1 4-cube/S2）；
  2. 把 S1/S2/S3/S6 正式 spec 切到 `cameras[].motion`；
  3. 在 3090 上生成 `freeze/p7-official.json` 并跑 `check_environment --device official`。
