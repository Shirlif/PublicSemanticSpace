# 公共数据最小验证报告：WT-MM

> 日期：2026-11-20
> 目标：用 WT-MM/lerobot_tactile 做一次 **B 类**（类别级/抓取对象级）的触觉 + 视觉最小闭环验证。
> 结论：**适配器可用；WT-MM 的视-触数据确实同步且可学习；但仍是 B 类，不是 C 类锚点。**

---

## 1. 数据集

| 项 | 值 |
|---|---|
| Repo | `WT-MM/lerobot_tactile` |
| 许可证 | **Apache-2.0**（可商用） |
| 机器人 | SO-100 触觉跟随臂 |
| 规模 | 86 episodes / 103,541 帧 / 30 fps |
| 模态 | RGB 640×480 + 触觉 12×32 + 6-D 关节/action |
| RD2 层级 | B 类；无物体 6D 位姿/实例 ID |

## 2. 最小对

避免下载 34.7 MB 视频，选用数据/视频索引错位的一对最小分片：

```
data/chunk-000/file-004.parquet             73,835 bytes
videos/observation.images.cam/chunk-000/file-005.mp4   11.4 MB
```

- Episode 50；**1027 帧**；30 fps；视频 1027 帧，触觉 1027 行，**逐帧对齐**。
- 校验：`meta/episodes/chunk-000/file-004.parquet` 中 `videos/.../file_index=5`，长度 1027。

## 3. 适配器

| 能力 | 实现 |
|---|---|
| LeRobot v3.0 | Parquet 逐帧 + `meta/info.json` + `meta/episodes/*` |
| 触觉/state | `observation.tactile.*`、`observation.state` 自动识别 |
| 下载 | 默认 `https://hf-mirror.com`，直接流式 HTTP |
| 断点续传 | `.part` 文件 + Range 请求 |
| 冒烟 | 触觉统计、时序一致性、线性 AE、视频-触觉 InfoNCE 对齐 |

CLI：

```bash
python -m pss.rd2.public.scripts.download_smoke \
  --dataset wt_mm --root data/public/wt_mm_pair --limit 1027 \
  --video data/public/wt_mm_pair/videos/observation.images.cam/chunk-000/file-005.mp4 \
  --out eval/results/public_wt_mm_pair_smoke.json
```

## 4. 结果

| 指标 | 值 |
|---|---:|
| 帧数 / episode | 1027 / 50 |
| 触觉形状 | 12×32 |
| 接触帧比例 | 0.367 |
| 触觉范围 | 0–1 |
| 时序余弦：相邻 / 随机 | **0.963 / 0.409**（gap 0.554） |
| 线性 AE 验证 MSE | 0.000117 |
| AE latent std | 0.080（非退化） |
| **视频-触觉对齐 R@1** | **0.211** |
| 打乱触觉后的 R@1 | 0.0019 |
| 随机基线（1/1027） | 0.00097 |

**对齐 R@1 是随机基线的约 217 倍**，说明：

1. WT-MM 的相机视频与触觉阵列是**真正逐帧同步**的；
2. 一个极小的线性双塔模型就能从触觉检索到对应视频帧，远超随机；
3. 适配器读出的触觉/视频配对没有错位。

## 5. 这验证了什么 / 没验证什么

**验证了：**
- LeRobot v3.0 适配器（Parquet + 视频）可用；
- 公开触觉数据能训练出非退化的触觉编码器；
- 视-触同步配对是真实可学的，B 类投影对齐有数据基础；
- 国内镜像入口 + 断点续传可把最小分片拉下来。

**没验证：**
- 物体 6D 位姿/实例 ID：WT-MM 没有，不能当 C 类锚点；
- 跨传感器迁移：12×32 触觉阵列与我们的合成触觉/其他传感器不同；
- 生产级 V-T 模型：R@1 0.211 是 1027 帧上的线性冒烟，不是最终能力；
- E0/R-Gate：仍需自采同传感器数据。

## 6. 国内镜像说明

- CLI 默认 `--endpoint https://hf-mirror.com`；
- `hf_hub_download` 与镜像的 xet 元数据不兼容，改为直接 HTTP 流式；
- 实测镜像会 307 到海外 CDN，速度仍受出口带宽限制（约 10–30 KiB/s）；
- 大文件支持 `.part` 断点续传；多线程 Range 会被 CDN SSL 限流，不建议。

## 7. 下一步

1. 用 WT-MM 的触觉帧预训练一个 **触觉编码器**（替换当前合成触觉 MLP），再重跑 Phase 4/5；
2. 若要更接近机器人的高分辨率触觉，再申请 Daimon-Infinity（Gated，非商用）或使用 OpenGraphLabs（CC-BY-4.0，人手）；
3. RD2 自采 C 类种子集仍是 E0/R-Gate 的前置条件，公开数据不能替代。
