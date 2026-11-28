# PSS 手工视觉专家与真实图像复核

> 日期：2026-11-28
> 范围：回答“特征提取是否必须用神经网络”，并在真实图像文件上复核 SmallCNN / DINOv2 / 手工特征三条路线
> 关联：`pss/encoders/vision/handcrafted.py`、`pss/encoders/scripts/bench_expert_zoo.py`、`pss/eval/scripts/run_vision_real_bench.py`、`contracts/experts.yaml`
> 上游：`eval/expert_zoo.md`、`docs/phases/PHASE11_ENCODER_PRIORITY_REPORT.md`

---

## 0. 一句话

**神经网络不是提取物体特征向量的唯一途径。** 在封闭合成域上，0 参数的 HSV+HOG+LBP 手工描述子拿到闭集 **0.964**，几乎追平 DINOv2（0.965），但开放集只有 0.519（接近随机）；在真实图像上，DINOv2 在形状/材质/尺寸和分布偏移下明显更强，手工特征在材质/尺寸上仍能以 1/200 的参数量打平。结论是**任务条件化的效率前沿**，不存在单一“最强方案”。

---

## 1. 实现

### 1.1 手工视觉专家

`pss/encoders/vision/handcrafted.py`：

- HSV 直方图（12/8/8）；
- HOG（16×16 cell，9 方向，2×2 block L2 归一化）；
- 8 邻域 LBP（uniform 映射 10 bin，2×2 空间网格）；
- 4×4 平均 RGB 网格；
- 输出 **440 维**，**0 参数**，纯 NumPy + Pillow，无预训练权重。

接入点：

- registry 新增 `kind="handcrafted"`；
- zoo 新增 `HandcraftedVisionAdapter`，与 CNN 专家共用同一套分割裁剪；
- `contracts/experts.yaml` 登记 `vision_handcrafted_v0`（R0，planned，非主线对照）。

### 1.2 真实图像复核脚本

`pss/eval/scripts/run_vision_real_bench.py`：

- `--dataset clevr`：CLEVR-CoGenT PNG（渲染图），按 CoGenT 协议 trainA/valA/valB；
- `--dataset tao`：TAO / Charades **真实视频帧**，按 GT box 裁剪，视频级 70/15/15 划分；
- 三个专家共用同一 crop，只拟合线性探针（StandardScaler + LogisticRegression），不改专家权重；
- 报告闭集 top-1、macro-F1、val→test 变化、单样本延迟、参数量。

---

## 2. 合成基准（`data/category_pretrain`，1920 视图，3 seeds）

| 专家 | 闭集 top-1 | 开集 AUROC | 属性 R² 均值 | 延迟 p50 | 峰值显存 | 参数量 |
|---|---:|---:|---:|---:|---:|---:|
| `vision_smallcnn_v0` | **0.990 ± 0.004** | **0.752** | **0.688** | 1.43 ms | 45.8 MB | 274,560 |
| `vision_handcrafted_v0` | 0.964 ± 0.004 | 0.519 | 0.592 | 1.13 ms | 9.6 MB | **0** |
| `vision_dinov2_vit_s14` | 0.965 ± 0.006 | 0.605 | 0.495 | 6.27 ms | 294.2 MB | 22,056,576 |

读法：

1. **闭集上手工特征几乎追平 DINOv2**（差 0.1 个点，seed 标准差 0.4–0.6），比 SmallCNN 低 2.6 个点。
2. **开集是分水岭**：手工 0.519 ≈ 随机；DINOv2 0.605；SmallCNN 0.752。类别可分 ≠ 能拒识新类。
3. **属性**：手工 0.592 高于 DINOv2 的 0.495，低于 SmallCNN 的 0.688；尺寸类属性（size_x/y）三者都很好（0.92 左右）。
4. **成本**：零参数、约 1/5 显存、比 DINOv2 快 5.6 倍；延迟与 SmallCNN 同量级（Python 循环，未向量化）。

> 结论：在封闭、干净的合成域，**大部分类别信息确实在经典图像统计里**；学习带来的优势主要体现在开放集和属性几何，而不是闭集 top-1。

---

## 3. 真实图像文件：CLEVR-CoGenT（2000/800/800）

> 注意：CLEVR 是**渲染图**，不是真实照片；它提供真实 PNG 文件、对象元数据和 CoGenT 组合偏移。valB 是“未见过的形状/颜色/材质组合”。

| 任务 | 专家 | valA | valB（shift） | 变化 | shift macro-F1 |
|---|---|---:|---:|---:|---:|
| `front_shape`（3 类） | SmallCNN | 0.882 | **0.350** | −0.532 | 0.324 |
| | DINOv2 | 0.995 | **0.998** | +0.003 | 0.997 |
| | 手工 | 0.991 | 0.884 | −0.107 | 0.881 |
| `front_material`（2 类） | SmallCNN | 0.922 | 0.914 | −0.009 | 0.914 |
| | DINOv2 | 0.991 | **0.986** | −0.005 | 0.986 |
| | 手工 | 0.998 | 0.985 | −0.013 | 0.985 |
| `front_size`（2 类） | SmallCNN | 0.986 | 0.979 | −0.007 | 0.979 |
| | DINOv2 | 0.999 | 0.998 | −0.001 | 0.997 |
| | 手工 | **1.000** | **1.000** | 0.000 | 1.000 |
| `majority_color`（8 类） | SmallCNN | 0.347 | 0.316 | −0.031 | 0.311 |
| | DINOv2 | 0.242 | 0.206 | −0.036 | 0.189 |
| | 手工 | 0.244 | 0.244 | 0.000 | 0.230 |

成本（batch=1，对象 crop）：SmallCNN 2.08 ms / 274,560 参数；DINOv2 9.19 ms / 22,056,576；手工 1.39 ms / **0**。

读法：

1. **形状 + 组合偏移是 SmallCNN 的灾难**：valA 0.882 → valB 0.350。任务自训的小 CNN 学到了训练域的形状-外观相关，换组合就崩。这正是“特征几何会决定 PSS 下游”的直接证据。
2. **DINOv2 对组合偏移几乎无损**：0.995 → 0.998。
3. **手工特征在材质/尺寸上与 DINOv2 打平**（0.985 vs 0.986；1.000 vs 0.998），参数量为 0；形状上差 11 个点。
4. `majority_color` 三个专家都在 0.2–0.3，接近 8 类随机（0.125）和多数类（0.226）。这个标签是**整场景多数颜色**，而输入只是前景对象 crop——标签与输入不匹配，应视为负对照，不是专家失败。

---

## 4. 真实照片：TAO / Charades（12 类，2303/1112/971）

> 真实视频帧，GT box 裁剪；视频级划分，无跨 split 泄漏；第三 split 是留出测试集，不是域偏移。

| 专家 | val top-1 | test top-1 | test macro-F1 | 参数量 | 延迟 p50 |
|---|---:|---:|---:|---:|---:|
| `dinov2_small` | 0.425 | **0.456** | **0.392** | 22,056,576 | 9.50 ms |
| `handcrafted_v0` | 0.295 | 0.285 | 0.239 | **0** | 1.58 ms |
| `smallcnn_v0` | 0.183 | 0.244 | 0.213 | 274,560 | 2.71 ms |

多数类基线 0.087。读法：

1. **真实照片上冻结预训练骨干明显最强**（0.456 vs 0.285 vs 0.244）。
2. 手工特征仍**高于合成域训出来的 SmallCNN**；后者在真实域基本没有迁移价值。
3. 绝对准确率都不高：12 个粗粒度类别、遮挡/尺度/背景变化大、裁剪切自跟踪标注而非检测器。这里比较的是“同一 crop 下的特征质量”，不是产品级识别率。

---

## 5. 回答原来的问题

### 5.1 “最高准确率”没有单一答案

| 场景 | 最高准确率方案 | 依据 |
|---|---|---|
| 封闭合成域、固定类别 | 任务自训 SmallCNN | 0.990 vs 0.965 / 0.964 |
| 开放/偏移的真实图像 | 冻结 DINOv2 | 形状 shift 0.998；真实照片 0.456 |
| 材质 / 尺寸等物理属性 | 手工特征 ≈ DINOv2 | 0.985 / 1.000，0 参数 |
| 嵌入式、极低硬件 | 手工特征 | 0 参数、~10 MB、1.1–1.6 ms |

### 5.2 “最快 / 最省硬件”

排序稳定：手工 ≪ 小 MLP/小 CNN ≪ DINOv2 ≪ 大模型。手工特征的延迟不是理论下界（当前是 Python 循环），但已比 DINOv2 快 5–6 倍，且零参数、零权重加载。

### 5.3 前提修正

“只要能提取出来，方法和过程没啥影响”**只在接口层成立**。反例就在本轮数据里：

- 同一个 SmallCNN，合成域 0.990，真实组合偏移下 0.350；
- 同一个 DINOv2，换投影头 re-ID 可从 0.963 到 0.993（Phase 11）。

特征是否**可迁移、可拒识、可跨模态对齐**，直接决定 PSS 读出的上限。方法有影响，而且影响在“提取之后、读出之前”。

---

## 6. 边界

- CLEVR 是渲染图；TAO/Charades 是真实照片但只有 12 个粗类，且只用了 Charades 子集（本地有视频的分片）。
- 没有真实同步的 RGB + 深度 + 分割 + 触觉 + 属性；RD2 真实数据缺口仍未关闭。
- 真实图像bench 目前是单 seed；视频级划分的方差尚未量化。下一步应跑 3 seeds 或按视频做交叉验证。
- TAO 的 GT box 需要从原始分辨率缩放到 480p 帧；检测器误差未建模。
- `majority_color` 不是可从前景 crop 学到的标签，只能当负对照。

---

## 7. 复现

```bash
# 合成 zoo：三个视觉专家
python -m pss.encoders.scripts.bench_expert_zoo \
  --data data/category_pretrain \
  --experts contracts/experts.yaml \
  --out eval/results/expert_zoo_handcrafted \
  --only vision_smallcnn_v0 vision_dinov2_vit_s14 vision_handcrafted_v0 \
  --seeds 0 1 2

# 真实图像文件：CLEVR-CoGenT
python -m pss.eval.scripts.run_vision_real_bench \
  --dataset clevr --out eval/results/real_vision_bench \
  --n-train 2000 --n-val 800 --n-test 800

# 真实照片：TAO / Charades
python -m pss.eval.scripts.run_vision_real_bench \
  --dataset tao --out eval/results/real_vision_photo_bench \
  --min-samples 100 --min-per-split 5 --max-per-class 200
```

测试：

```bash
python -m pytest tests/test_handcrafted_vision.py \
  tests/test_vision_real_bench.py tests/test_bench_expert_zoo.py -q
```

---

## 8. 改动文件

- `pss/encoders/vision/handcrafted.py`（新增）
- `pss/encoders/vision/__init__.py`（导出）
- `pss/encoders/registry.py`（新增 `handcrafted` kind）
- `pss/encoders/scripts/bench_expert_zoo.py`（新增 adapter 与注册）
- `pss/eval/scripts/run_vision_real_bench.py`（新增）
- `contracts/experts.yaml`（登记 `vision_handcrafted_v0`）
- `tests/test_handcrafted_vision.py`、`tests/test_vision_real_bench.py`、`tests/test_bench_expert_zoo.py`
