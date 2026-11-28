# PSS U1c：CLIP 双塔 VLM 代理（16 类 category）

> 日期：2026-11-28
> 上游：`docs/experiments/PSS-架构上界复核-第二阶段.md` §2、`pss/eval/scripts/run_upper_bound_u1b.py`
> 结果目录：`eval/results/upper_bound/u1c/`
> 状态：已实际执行；CLIP 文本塔加载成功，未触发 Qwen2.5 回退。本报告所有数字来自本轮 `u1c.json` 与终端输出。

---

## 0. 一句话

工作区没有 Qwen-VL/LLaVA/InternVL 权重，因此 U1c 用**冻结 CLIP ViT-B/16 视觉塔 + CLIP 文本塔**做 two-tower 代理，而不是真实 VLM。16 类闭集上 `proto` 与 `lda` 都达到 **0.973**，U1b 的 LDA 增益（384 类上 +0.110）在本任务上**消失**；CLIP 文本零样本闭集 **0.292**（高于 1/16=0.0625 的机会，但远低于 LDA），开集 **AUROC 0.492 / reject 0.023**（随机水平）；`dual_head` 在 calibration 上选到原型开集头，test 上 **AUROC 0.958 / reject 0.744**，未通过 `u_real_vlm_proxy` 的 0.99/0.95 门槛。结论：**文本条件化读出不能匹配学习 LDA，双头在此任务上也不成立。**

---

## 1. 设置

| 项 | 取值 |
|---|---|
| vision tower | `openai/clip-vit-base-patch16`（冻结，768-d pooler + 512-d CLIP joint projection） |
| text tower | 同一快照的 CLIP text encoder（512-d joint space） |
| 文本模板 | `a photo of a {}.` / `a close-up photo of a {}.` / `a {} on a table.` / `a small {}.`，4 个模板的归一化均值 |
| 任务 | `data/category_pretrain_instance8`，`task_split(regime="category")` |
| seen 类 | 16 类：ball/block/bottle/bowl/box/brick/can/carton/cup/egg/fruit/jar/mug/plate/rod/saucer |
| held-out 类 | cloth/tool/sponge/sheet（只用于开集，不参与训练或文本原型） |
| closed-set 划分 | 按 variation index：train 2304 行 / val 384 / test 384 |
| open-set 划分 | held-out 768 行，按 U1b 方式交错切成 cal 384 / test 384 |
| 投影 seeds | 0 / 1 / 2；`transform="res"`、128-d、120 epochs |
| shrinkage grid | 0.001 / 0.05 / 0.1 / 0.3 / 0.5 / 0.7 / 0.9，在 val seen + held cal 上选 |

四个臂：

- `proto`：U1b 配方，学习投影后多模态原型 cosine（CLIP vision + 合成 tactile/geometry）。
- `lda`：U1b 配方，学习投影 → 逐模态 PCA 白化 → concat → pooled-covariance LDA；shrinkage 在 cal 上选。
- `text_zero_shot`：CLIP joint image embedding 与 16 个类名文本原型的 cosine，无学习投影。
- `dual_head`：闭集用 `lda`；开集用 cal 上 AUROC 更高的 `text` 或 `proto` 头（test 半不参与选择）。

CLIP 视觉特征复用 U1b 缓存（`u1b_clip/paired_features.npz`，3840×768）；若缓存缺失会重新编码。512-d joint 特征由 `visual_projection` 作用于 768-d pooler 得到，已与 `CLIPModel.get_image_features` 数值对齐（cos=1.0）。

---

## 2. 主表（3 投影种子，mean ± std）

| arm | closed-set acc | AUROC | unseen reject @95 |
|---|---:|---:|---:|
| `proto` | 0.973 ± 0.002 | 0.958 ± 0.009 | 0.744 ± 0.031 |
| `lda`（shrinkage on cal） | **0.973 ± 0.001** | **0.966 ± 0.002** | **0.873 ± 0.004** |
| `text_zero_shot` | 0.292 ± 0.000 | 0.492 ± 0.000 | 0.023 ± 0.000 |
| `dual_head`（LDA 闭集 + proto 开集） | 0.973 ± 0.001 | 0.958 ± 0.009 | 0.744 ± 0.031 |

- `lda − proto = +0.0000`（精确差 1.1×10⁻¹⁶）：闭集增益消失。
- `dual` 的闭集沿用 LDA，开集在三个 seed 的 cal 上都选到 `proto`，因此和 `proto` 开集完全相同。
- 文本臂与 seed 无关，std 为 0。

---

## 3. 每 seed 明细

| seed | shrinkage | open head | text cal AUROC | proto cal AUROC | proto acc | lda acc | text acc | dual AUROC | dual reject |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.001 | proto | 0.473 | 0.950 | 0.971 | 0.974 | 0.292 | 0.946 | 0.729 |
| 1 | 0.001 | proto | 0.473 | 0.965 | 0.977 | 0.974 | 0.292 | 0.958 | 0.716 |
| 2 | 0.001 | proto | 0.473 | 0.975 | 0.971 | 0.971 | 0.292 | 0.969 | 0.786 |

三个 seed 的 shrinkage 都选最小 0.001，与 U1b 的 CLIP 结果一致：开集问题不是协方差收缩能修的。

---

## 4. 预注册检查

`u_real_vlm_proxy` 映射到 `dual_head`（LDA 闭集 + text/proto 开集）：

- 检查 arm 数：2（`u_real_proto` 无 bar，`u_real_vlm_proxy` 3 项）
- 通过：**False**
- 失败项：
  - `u_real_vlm_proxy.auroc`：期望 ≥ 0.990，实测 **0.9575**
  - `u_real_vlm_proxy.unseen_reject_at_95`：期望 ≥ 0.950，实测 **0.7439**
- 通过项：`accuracy_delta_vs_proto = +0.0000`（bar 为 ≥ 0.0，恰好压线）。

失败照实保留，未调整 bar。

---

## 5. 判读

1. **16 类闭集饱和，LDA 增益消失。** U1b 在 384 类 view-instance 上得到 proto 0.812 / LDA 0.922（+0.110）；本任务 proto 与 LDA 都是 0.973。A4 的“先混合再判别”增益依赖类数与细粒度，不是换任务就自动成立。

2. **CLIP 文本零样本在该数据上不可用。** 16 类闭集只有 0.292（高于 1/16=0.0625 的机会，但远低于 LDA 的 0.973），开集 AUROC 0.492（随机）、reject 0.023。可能原因：MuJoCo primitive 渲染（128×128、无真实纹理）不在 CLIP 图文预训练分布内；类名之间有语义重叠（cup/mug/bowl/saucer）；文本原型只覆盖 seen 类名。这是**代理数据与代理文本塔的联合失败**，不能直接外推成“VLM 没有文本条件化能力”。

3. **双头拆分没有收益。** cal 上 text AUROC 0.473、proto 0.950–0.975，三个 seed 都选 proto；但 test 上 proto 开集（0.958 / 0.744）反而低于 LDA 单头（0.966 / 0.873）。也就是说“LDA 闭集 + 原型开集”在这里不如“LDA 单头”。U1b 的 DINOv2 双头之所以成立，是因为原型开集强于 LDA 开集；本任务上这个前提不成立。即便把 LDA 也加入开集候选，0.966 / 0.873 仍低于 0.99 / 0.95，因此失败不是 dual 的选择规则造成的。

4. **开集比 U1b 难，且是任务属性。** U1b 384 类的 LDA 开集是 0.997 / 0.995；本任务的 4 个 held-out 类别（cloth/tool/sponge/sheet）与 seen 类别语义接近，LDA 只有 0.966 / 0.873、proto 0.958 / 0.744。开集退化来自类间语义距离，不是 CLIP 视觉塔单独的问题。

5. **文本塔加载成功，但结果仍是代理结论。** CLIP text encoder 正常加载（`text_source=clip_text`，512-d），没有触发 Qwen2.5-1.5B 回退。即使如此，U1c 也只能说明“在这个合成 16 类任务上，CLIP 双塔代理的文本零样本与双头都不可用”，不能说明真实 VLM 的行为。

---

## 6. 可以 / 不可以声称

**可以声称**

1. CLIP ViT-B/16 视觉塔 + CLIP 文本塔可以在 16 类 category 任务上跑通，文本原型可生成、可评估。
2. 16 类闭集上 `proto` 与 `lda` 都达到 0.973，U1b 的 LDA 增益在该任务上消失。
3. CLIP 文本零样本在该合成 primitive 数据上不可用：闭集 0.292，开集 AUROC 0.492 / reject 0.023（开集为随机水平）。
4. `dual_head` 的 cal 选择机制可执行；但 test 上原型开集不如 LDA 单头，双头无收益。
5. 本轮 `u_real_vlm_proxy` 预注册失败：AUROC 0.958、reject 0.744，低于 0.99/0.95。

**不可以声称**

1. 不可以声称“真实 VLM 已复现”——这是 two-tower 代理，工作区没有 Qwen-VL/LLaVA/InternVL 权重。
2. 不可以声称文本条件化读出匹配或超过学习 LDA；实测文本零样本 0.292 vs LDA 0.973。
3. 不可以把 16 类数字与 U1b 的 384 类数字直接相减；任务、类数、开集难度都不同。
4. 不可以声称双头在开集上有效；本任务上单头 LDA 更好。
5. 不可以把“CLIP 文本零样本失败”外推为“所有 VLM 的文本条件化都失败”；只测了 CLIP 双塔与 16 个短类名。

---

## 7. 边界

- **two-tower 代理，不是真实 VLM**：没有 VLM 权重，文本塔只覆盖 16 个 seen 类名。
- **16 类 category，不是 384 类 view-instance**：闭集更容易、开集更难，结论不能与 U1b 直接互换。
- **文本原型只覆盖 seen 类名**：held-out 的 cloth/tool/sponge/sheet 只用于开集，没有文本原型。
- **`proto`/`lda` 复用 U1b 多模态读出**：CLIP vision + 合成 tactile/geometry；LDA 不是纯视觉。文本塔只在 CLIP 视觉/文本 joint space 工作。
- **模板固定**：4 个模板的均值，未做 prompt 搜索；模板在运行前固定，未用 test 调参。
- **3 个投影种子**只覆盖投影训练方差，不覆盖数据划分方差。
- **cal/test 切分**沿用 U1b 的 held 交错切法；cal 只用于 shrinkage 与开集头选择，test 只报一次。

---

## 8. 产物与复现

```
pss/eval/scripts/run_upper_bound_u1c.py     # 新建：CLIP 双塔代理 + 4 arms
eval/results/upper_bound/u1c/u1c.json       # 全部数字与预注册检查
eval/results/upper_bound/u1c/U1C_REPORT.md  # 脚本自动生成的简版报告
eval/results/upper_bound/u1c/paired_features.npz  # CLIP 768-d + joint 512-d 缓存
docs/experiments/PSS-U1c-vlm-proxy.md       # 本报告
```

复现命令：

```bash
python -u -m pss.eval.scripts.run_upper_bound_u1c \
  --out eval/results/upper_bound/u1c \
  --projection-seeds 0 1 2 --projection-epochs 120 --device cuda
```

---

## 9. 一句话

**U1c 是一个诚实的负结果：CLIP 双塔代理能跑通，但 16 类闭集上 LDA 增益消失，CLIP 文本零样本闭集仅 0.292、开集为随机水平，双头开集未达预注册门槛；不能据此声称真实 VLM 的文本条件化读出成立。**
