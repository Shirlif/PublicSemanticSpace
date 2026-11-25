# 类别识别预训练报告（v2）

> 日期：2026-11-20
> 原则：预训练阶段只解决一件事——**让空间模型具备识别物体类别的能力**。实例 ID、追踪、绑定、具身部署都往后放。

---

## 1. 目标与验收口径

| 指标 | 目标 | 本次结果 |
|---|---|---:|
| 闭集 top-1（16 类） | ≥ 0.95 | **0.969**（融合） |
| 类别数 | 20–50 | **20**（16 已见 + 4 未见） |
| 类内变化 | 每类 ≥ 20 个变化 | **24 变化 × 4 视角** |
| 开放集拒绝 | 未见类别能说"不认识" | 未见拒绝 **0.818**，AUROC **0.960** |
| 鲁棒性 | 视角/光照分层 | 每视角 **0.958–0.979** |

---

## 2. 数据

`data/category_pretrain`，MuJoCo 桌面场景，原始几何体代理。

| 项 | 值 |
|---|---:|
| 类别 | 20 |
| 已见类别 | 16 |
| 未见类别（开放集） | 4：`cloth`、`tool`、`sponge`、`sheet` |
| 每类变化 | 24 |
| 每变化视角 | 4（随机相机 + 随机光照） |
| 样本文件 | 480 个 `.npz` |
| 帧数 | 1,920 |
| 划分 | 变化 index `v % 8 < 6` → train；`== 6` → val；`== 7` → test；未见类别 → `test_category` |
| 训练 / 验证 / 测试 | 1,152 / 192 / 192 帧 |
| 未见类别 | 384 帧 |

已见类别：

```
ball, block, bottle, bowl, box, brick, can, carton,
cup, egg, fruit, jar, mug, plate, rod, saucer
```

类内变化维度：尺度 0.85–1.15、颜色抖动 ±0.10、材质抖动 ±0.15、随机 yaw、随机相机、随机光照。
视觉裁剪保留长宽比（`crop_preserve_aspect=True`），避免把高杯子和扁盘子都压成同一个方形。

---

## 3. 感官专家

`checkpoints/category_pretrain_experts`

| 专家 | train | val | test | 说明 |
|---|---:|---:|---:|---|
| 视觉 SmallCNN | 0.970 | 0.943 | **0.943** | 64×64 裁剪，保留长宽比 |
| 几何手工特征 + LR | 0.931 | 0.911 | **0.911** | 16 维，深度+分割 |
| 触觉 MLP（材质代理） | 0.305 | 0.312 | **0.245** | 随机基线 0.0625，约 4× 随机 |

触觉仍然弱，因为它是材质参数生成的代理，不是真实接触测量。这一阶段不指望它承担类别识别。

---

## 4. 公共空间与原型库

`checkpoints/category_pretrain_projection` + `checkpoints/category_pretrain_prototypes`

融合权重由**训练集内部 3 折交叉验证**选出（按变化分组，测试集不参与）：

```
{'vision': 1.0, 'tactile': 0.0, 'geometry': 1.0}
```

| 查询 | val | test |
|---|---:|---:|
| **融合 flat（主）** | 0.958 | **0.969** |
| 融合 + 原型树 | 0.865 | 0.807 |
| 仅视觉 | 0.953 | 0.969 |
| 仅几何 | 0.896 | 0.870 |
| 仅触觉 | 0.286 | 0.203 |

关键结论：

- **类别识别成立**：16 类闭集 0.969，和视觉单模态持平。
- **触觉没有贡献**：CV 把触觉权重压到 0。合成触觉的类别信号太弱，这是数据问题，不是融合问题。
- **几何没有超过视觉**：V+G 与 V 持平；几何在深度/尺寸上更稳，但当前类别靠形状+颜色已可分。
- **原型树在 16 类上过剪枝**：0.807 vs flat 0.969。类别预训练阶段用 flat 检索，不要开树。

### 每类测试准确率

| 类别 | acc | 类别 | acc |
|---|---:|---|---:|
| ball | 1.000 | jar | 0.917 |
| block | 0.917 | mug | 1.000 |
| bottle | 1.000 | plate | 1.000 |
| bowl | 1.000 | rod | 1.000 |
| box | 1.000 | saucer | 1.000 |
| brick | 1.000 | cup | **0.833** |
| can | 1.000 | egg | 0.917 |
| carton | 1.000 | fruit | 0.917 |

弱项集中在 `cup / mug / jar / can` 这组圆柱容器，以及 `block / box / brick` 这组方块。原因是原始几何体只靠长宽比和颜色区分，视觉裁剪后差异变小；增加纹理、背景和难负样本后会改善。

### 每视角测试准确率

| view | 0 | 1 | 2 | 3 |
|---|---:|---:|---:|---:|
| acc | 0.979 | 0.958 | 0.979 | 0.958 |

视角/光照随机化下没有明显塌陷，说明模型没有只记住单一相机。

---

## 5. 开放集

| 指标 | 值 |
|---|---:|
| 阈值（val max score 5 分位） | 0.878 |
| 已见误拒率 | 0.042 |
| 未见拒绝率 | **0.818** |
| 开放集 AUROC | **0.960** |
| 已见平均 max score | 0.955 |
| 未见平均 max score | 0.756 |

可用的第一版，但还没到生产级。下一步可以用 **top1−top2 margin** 或每模态校准替代单一 max score，目标是把未见拒绝率提到 0.95 以上、已见误拒保持在 5% 以下。

---

## 6. 这次修正了什么

1. **类别数 6 → 16（总 20）**：从 6 类冒烟扩到 16 类闭集。
2. **类内变化 3 → 24**：原来每类只有 3 个固定变体，现在连续采样尺度/颜色/材质。
3. **裁剪保留长宽比**：原来 tight crop 后 resize 到方形，杯子和盘子形状被压平；现在先扩成方形再 resize。
4. **融合权重不再拍脑袋**：固定权重会被弱触觉拖累（0.807）；改为训练集内部 3 折 CV，自动把触觉权重压到 0。
5. **加每类/每视角拆解和开放集 AUROC**：不再只报一个总数。

---

## 7. 复现

```bash
# 1. 生成类别预训练集
python -m pss.sim.scripts.generate_category_pretrain \
  --out data/category_pretrain --variations 24 --views 4 --seed 0

# 2. 训练感官专家
python -m pss.encoders.scripts.train_experts \
  --data data/category_pretrain --out checkpoints/category_pretrain_experts \
  --epochs 30 --size 64 --seed 0

# 3. 训练投影头
python -m pss.projection.scripts.train_projections \
  --data data/category_pretrain --experts checkpoints/category_pretrain_experts \
  --out checkpoints/category_pretrain_projection \
  --epochs 60 --batch-size 64 --temp 0.07 --seed 0

# 4. 建原型库并评测
python -m pss.prototypes.scripts.build_prototypes \
  --data data/category_pretrain --experts checkpoints/category_pretrain_experts \
  --projections checkpoints/category_pretrain_projection \
  --out checkpoints/category_pretrain_prototypes --seed 0
```

或：

```bash
make category-pretrain
```

---

## 8. 下一步（仍只做类别识别）

| 优先级 | 动作 | 目的 |
|---|---|---|
| P0 | 给 `cup/mug/jar/can`、`block/box/brick` 加纹理/背景随机化和难负样本 | 把弱项从 0.83 拉到 0.95 |
| P0 | 开放集改用 margin 分数 + 每模态校准 | 未见拒绝 0.818 → 0.95+ |
| P1 | 类别数扩到 30–50，每类变化扩到 40+ | 验证容量和扩展性 |
| P1 | 加 D=256/512 容量对照 | 为后续更多类别定维度 |
| P2 | 领域随机化（桌面颜色、背景、光照强度） | 缩小 sim2real 的视觉差距 |

实例 ID、追踪、绑定、真机部署仍不在本阶段范围内。
