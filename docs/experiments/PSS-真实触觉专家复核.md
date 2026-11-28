# PSS 真实触觉专家复核：手工接触特征 vs 学习编码器

> 日期：2026-11-28
> 范围：在真实触觉数据上回答“非神经网络特征提取是否够用”
> 关联：`pss/encoders/tactile/handcrafted.py`、`pss/eval/scripts/run_tactile_real_bench.py`、`contracts/experts.yaml`
> 上游：`docs/phases/PHASE10_PUBLIC_DATA_REPORT.md`、`docs/phases/PHASE10_WTMM_TACTILE_REPORT.md`

---

## 0. 一句话

**在真实 GelSight 的粗糙度/纹理上，0 参数手工接触特征是最强的（0.692 / 0.744），超过真实 GelSight 预训练 CNN（0.641）和 WT-MM CNN（0.667）。** 但硬度（0.462，多数类 0.438）和 WT-MM 夹爪状态（0.364）不是手工特征的强项——前者需要动态/力信号，后者由合成 MLP 拿到 0.605。结论依旧是任务条件化的，不是“手工全面替代学习”。

---

## 1. 实现

### 1.1 手工触觉描述子

`pss/encoders/tactile/handcrafted.py`，0 参数，兼容两类真实输入：

- **PhysiCLeAR**：5 帧 64×64 RGB GelSight；
- **WT-MM**：12×32 接触网格。

每个 clip 输出一个向量：

- 逐帧：强度统计（均值/方差/分位数）、梯度幅值、LBP、HOG、4×4 空间网格；
- 接触代理：`mean + std` 阈值以上的比例、加权质心、扩散度（尺度无关）；
- 时序：相邻帧平均变化、时间标准差、质心漂移；
- RGB 输入额外加 3 通道均值/方差。

维度随输入形状变化：**8×8 = 118、64×64 RGB = 406、12×32 = 586**。

接入 registry（`kind="handcrafted"`）与 `contracts/experts.yaml` 的 `tactile_handcrafted_v0`。

### 1.2 评测脚本

`pss/eval/scripts/run_tactile_real_bench.py`：

- `--dataset physiclear`：官方 object-level train/val/test（333/36/39 clips），任务为 hardness/roughness/texture 三档；
- `--dataset wtmm`：本地缓存的 3 个 episode，按 episode 内时间顺序 70/15/15 切分，5 帧窗口，任务为夹爪开/合；
- 只拟合线性探针（StandardScaler + LogisticRegression），专家权重冻结；
- 报告 top-1、macro-F1、多数类基线、单 clip 延迟、参数量。

---

## 2. 合成 8×8 对照（`data/category_pretrain`，1920 视图，3 seeds）

| 专家 | 闭集 top-1 | 开集 AUROC | 属性 R² 均值 | 延迟 | 参数量 |
|---|---:|---:|---:|---:|---:|
| `tactile_mlp_v0` | **0.220 ± 0.002** | 0.242 | 0.309 | 0.32 ms | 33,088 |
| `tactile_handcrafted_v0` | 0.184 ± 0.019 | **0.264** | **0.369** | 0.56 ms | **0** |

属性分项：手工的 hardness 0.921 / roughness 0.870，均高于 MLP 的 0.877 / 0.668。也就是说，在合成域上分类准确率 MLP 略高，但物理属性方向手工更强——与真实 GelSight 的结论一致。

---

## 3. PhysiCLeAR（真实 GelSight）

| 任务 | 专家 | test top-1 | test macro-F1 | 参数量 |
|---|---|---:|---:|---:|
| **hardness** | 手工 | 0.462 | 0.335 | **0** |
| | 合成 MLP | **0.538** | 0.233 | 33,088 |
| | PhysiCLeAR CNN | 0.385 | **0.366** | 258,048 |
| | WT-MM CNN | 0.462 | 0.375 | 27,680 |
| | 多数类 | 0.438 | — | — |
| **roughness** | 手工 | **0.692** | **0.687** | **0** |
| | 合成 MLP | 0.385 | 0.189 | 33,088 |
| | PhysiCLeAR CNN | 0.641 | 0.619 | 258,048 |
| | WT-MM CNN | 0.667 | 0.615 | 27,680 |
| | 多数类 | 0.402 | — | — |
| **texture** | 手工 | **0.744** | **0.670** | **0** |
| | 合成 MLP | 0.282 | 0.147 | 33,088 |
| | PhysiCLeAR CNN | 0.641 | 0.651 | 258,048 |
| | WT-MM CNN | 0.667 | 0.652 | 27,680 |
| | 多数类 | 0.468 | — | — |

成本（单 clip，5 帧，batch=1）：手工 6.70 ms / 0 参数；合成 MLP 0.76 ms；PhysiCLeAR CNN 0.73 ms；WT-MM CNN 0.89 ms。

读法：

1. **粗糙度/纹理：手工最强**（0.692 / 0.744，macro-F1 也最高）。纹理本来就是梯度/LBP 的直接统计量，手工描述子的归纳偏置正好匹配。
2. **硬度：没有专家真正学会**。合成 MLP 的 top-1 0.538 高于多数类，但 macro-F1 只有 0.233，说明它主要押注多数类；手工 0.462 也仅略高于多数类 0.438。单帧静态 RGB 可能不足以判断硬度，需要动态按压/力信号。
3. **真实预训练 CNN 不是自动赢家**：PhysiCLeAR CNN 在粗糙度/纹理上低于手工，说明它的自监督目标（时序对比 + 重建）没有把“材质属性”压到线性可读的方向。
4. **手工的延迟高是工程问题**：6.7 ms 来自 Python 循环；向量化后应能到 1 ms 量级。参数量优势是结构性的（0 vs 2.6×10⁵）。

---

## 4. WT-MM（真实 12×32 接触网格）

任务：5 帧窗口预测夹爪开/合（按训练集中位数二值化）。

| 专家 | val top-1 | test top-1 | test macro-F1 | 参数量 | 延迟 |
|---|---:|---:|---:|---:|---:|
| 合成 MLP | 0.651 | **0.605** | **0.593** | 33,088 | 0.51 ms |
| 手工 | 0.569 | 0.364 | 0.330 | **0** | 3.68 ms |
| WT-MM CNN | 0.568 | 0.328 | 0.300 | 27,680 | 0.53 ms |
| 多数类 | — | 0.537 | — | — | — |

读法：

1. **合成 MLP 最好（0.605）**：它在合成 8×8 接触图上受过监督训练，接触模式与夹爪状态相关，迁移到这个弱目标反而有效。
2. **WT-MM CNN 最差（0.328）**：它的自监督目标是时序对比 + 重建，不保证保留夹爪状态；这也是“预训练目标决定特征可用性”的又一例。
3. **手工 0.364**：接触统计量对夹爪开合不够判别，可能因为窗口内自由空间帧占多数、统计被稀释。
4. **绝对夹爪位置不可线性预测**：所有专家的 R² 都小于 0，因此不报告回归，只报告开/合分类。这是诚实的负结果。

---

## 5. 结论

| 问题 | 答案 |
|---|---|
| 手工触觉特征能打吗？ | 能。在真实 GelSight 的粗糙度/纹理上是最强的，且 0 参数。 |
| 能全面替代学习编码器吗？ | 不能。硬度、夹爪状态上手工不是最好；硬度的共同失败说明输入信号本身不足。 |
| 最强方案是什么？ | 任务条件化：纹理/粗糙度 → 手工；接触/夹爪状态 → 监督过的小 MLP；真实预训练 CNN 不自动最强。 |
| 成本最低？ | 结构上手工（0 参数）；当前实现延迟不是最低，需向量化。 |

这与视觉侧的结论一致：**特征提取不必须是神经网络，但“哪个方案最好”取决于任务、标签和信号，而不是模型大小。**

---

## 6. 边界

- PhysiCLeAR 测试只有 39 clips，object-level 划分；macro-F1 方差大。
- 标签是三档 ordinal rank，不是物体身份；不能外推成“触觉物体识别”。
- WT-MM 本地只有 3 个 episode（约 4k 帧），按 episode 内时间切分；跨 episode 泛化未测。
- 夹爪开/合是弱语义目标，触觉可能直接编码接触，而不能说明材质/物体理解。
- PhysiCLeAR / Octopi 的权重与数据无 LICENSE，按研究用途使用，不进主线发布。
- 手工描述子当前是 Python 实现，延迟不是理论下界。

---

## 7. 复现

```bash
# 合成 8x8 对照
python -m pss.encoders.scripts.bench_expert_zoo \
  --data data/category_pretrain --experts contracts/experts.yaml \
  --out eval/results/expert_zoo_tactile_handcrafted \
  --only tactile_mlp_v0 tactile_handcrafted_v0 --seeds 0 1 2

# 真实 GelSight 属性
python -m pss.eval.scripts.run_tactile_real_bench \
  --dataset physiclear --out eval/results/tactile_real_bench --device cuda

# 真实 WT-MM 接触网格
python -m pss.eval.scripts.run_tactile_real_bench \
  --dataset wtmm --out eval/results/tactile_real_wtmm --device cuda
```

测试：

```bash
python -m pytest tests/test_handcrafted_tactile.py -q
```

---

## 8. 改动文件

- `pss/encoders/tactile/handcrafted.py`（新增）
- `pss/encoders/tactile/__init__.py`（导出）
- `pss/encoders/registry.py`（新增 tactile `handcrafted` kind）
- `pss/eval/scripts/run_tactile_real_bench.py`（新增）
- `contracts/experts.yaml`（登记 `tactile_handcrafted_v0`）
- `tests/test_handcrafted_tactile.py`（新增）
