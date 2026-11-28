# PSS-Boundary 决策 memo

> 日期：2026-11-28
> 范围：噪声边界契约的最终决策。**本文不再新增实验**；所有数字来自已完成的运行。
> 上游：`docs/plans/PSS-问题重构.md`、`docs/experiments/PSS-Boundary-Phase4.md`
> 状态：最终决策（single-frame contract GO with class-balanced Bayes；stress 需多帧或接受 overall 回退）。
>
> **2026-11-28 补充**：三维场景建模已测（`PSS-Scene-Model-Report.md`）。H1/H2 假：3D 3 视角 0.686 赢不了 1D 64 帧 0.886；NO-GO 默认化。**本 memo 的类平衡 Bayes 仍是 fallback**，没有被 scene model 替换。

---

## 0. 决策

**在契约传感器条件下（σ≤0.01 / 3°+1%），轴谓词使用“类平衡经验 Bayes 边界”，不用自然先验 Bayes，也不用全局温度 soft dead zone。**

具体配置：

- `GridBayesBoundary(eps=0.02, bins=257)`，先验按 pair type 反频率加权，`strength=0.5`（在训练 cal 上选，未在 test 调）。
- `sigma` 传 **pair-difference 噪声**：per-object σ 时是 `σ√2`。
- `axis_mode="bayes"`；`hard` 作为无先验/无 σ 时的 fallback。
- 压力档 σ=0.04 / 5°+2%：单帧 strength 0.5 不达标；用多帧（M≥64 @σ=0.04，M≥4 @σ=0.01）或接受 full-balance 的 overall 回退。

---

## 1. 证据总表

| 场景 | arm | near-worst | overall | 判定 |
|---|---|---:|---:|---|
| 像素契约 σ=0.01 | `hard_clean`（U6 baseline） | 0.685 | 0.968 | 参照 |
| 像素契约 σ=0.01 | 自然先验 grid Bayes（加噪输入） | 0.634 | 0.969 | NO-GO |
| 像素契约 σ=0.01 | 自然先验 KDE（加噪输入） | 0.662 | 0.970 | NO-GO |
| **像素契约 σ=0.01** | **类平衡 Bayes，strength 0.5（cal）** | **0.848** | **0.962** | **GO** |
| canonical 3°+1% | 自然先验 grid Bayes | 0.693 | 0.964 | NO-GO |
| **canonical 3°+1%** | **类平衡 Bayes，strength 0.5（transfer）** | **0.871** | **0.957** | **GO** |
| canonical 5°+2% | 类平衡 Bayes，strength 0.5 | 0.750 | 0.924 | 低于 0.80；需多帧/full balance |
| 像素 stress σ=0.04 | 类平衡 strength 0.5 | 0.052 | 0.890 | NO-GO |
| 像素 stress σ=0.04 | full balance KDE（post-hoc） | 0.922 | 0.636 | Pareto，超出 2pt overall 守卫 |
| 像素 stress σ=0.04 | 多帧 M=64（自然先验） | 0.846 | 保留 | GO（传感器级） |
| 像素契约 σ=0.01 | 多帧 M=4（自然先验） | 0.844 | 保留 | GO |

关键校准事实：

- strength 0.5 是在**像素训练 10% cal split** 上选出的；canonical 3°+1% 没有重新调 strength，直接迁移，仍然 0.871，说明不是过拟合。
- canonical 3°+1% 的 overall 从自然先验 0.964 掉到 0.957（−0.7 pt），在 2 pt 守卫内。
- 自然先验的 0.948/0.971 是**干净输入 / overall**，不是加噪 near-cell；0.948 不再引用。

---

## 2. Ship 什么

| 组件 | 决策 |
|---|---|
| 轴谓词 | `AxisPredicate(mode="bayes")` + 类平衡 `GridBayesBoundary`（strength 0.5） |
| 先验 | 启动时在 clean 训练对上 fit；换场景/换单位必须重 fit |
| σ | pair-difference σ；per-object σ 时乘 √2 |
| fallback | `axis_mode="hard"`：无先验/无 σ/先验漂移时 |
| 弃权 | 接口已存在（`axis_abstain`）；**selective accuracy/coverage 尚未在 balanced Bayes 上标定**，上线前补 |
| 压力档 | 多帧 M≥64 @σ=0.04 或 M≥4 @σ=0.01；或接受 full-balance 的 overall 回退 |
| runtime 默认 | 仍是 `off`（兼容）；部署配置显式切 `bayes` |
| 审计 | `origin=bayes_geometry` / `hard_geometry`；`axis_mode`/`sigma` 可查 |

---

## 3. 不能声称什么

1. 不能声称 0.948 是契约结果；它是干净输入 + 宽似然的诊断。
2. 不能声称自然先验 Bayes/KDE 解决了 near-cell；两者 0.63–0.69，NO-GO。
3. 不能声称 full-balance 是免费收益；它把 stress near 拉到 0.922，但 overall 掉到 0.636。
4. 不能声称 sim2real；canonical 仍是 CLEVR 渲染 + 针孔相机代理，不是真实 RGB-D 标定。
5. 不能声称 runtime 默认已经用 Bayes；库默认是 `off`。
6. 不能声称弃权已经校准；balanced Bayes 的 selective curve 未测。
7. 不能声称多帧在真实运动/配准误差下成立；当前是独立噪声 + 静态 pair。

---

## 4. Operating points（Pareto）

| 目标 | 选择 | near | overall | 适用 |
|---|---|---:|---:|---|
| 契约内、overall 守卫 | 类平衡 strength 0.5 | 0.848–0.871 | 0.957–0.962 | **默认推荐** |
| 最差 cell 优先 | full balance / KDE | 0.975 | 0.917 | 只有在 diagonal 非安全关键时 |
| 压力档、overall 保留 | 多帧 M=64 | 0.846 | 保留 | 静态/可配准场景 |
| 最低成本 | hard fallback | 0.685 | 0.968 | 无先验/低风险场景 |

**没有单一“最强”**：near-cell 与 overall 在交换；契约的 2 pt overall 守卫选中 strength 0.5。

---

## 5. 仍未解决

1. **真实 RGB-D**：像素/3D 代理 → 真实深度/位姿噪声。
2. **canonical 5°+2%**：0.750；多帧或 full balance 未在 canonical 上复测。
3. **弃权曲线**：balanced Bayes 的 selective accuracy/coverage。
4. **先验漂移**：换场景/换传感器后的重估与监控。
5. **多帧配准**：真实运动、遮挡、深度缺失下的帧对齐。
6. **审计 version 单调化**：`audit_seq` 未做。

---

## 6. 复现

```bash
# 像素契约：类平衡，strength 在训练 cal 上选
python -u -m pss.eval.scripts.run_boundary_bayes --calibrate-strength \
  --out eval/results/boundary_bench/bayes_calibrated

# canonical 3°+1%：同一 strength 迁移
python -u -m pss.eval.scripts.run_boundary_canonical --balance-strength 0.5 \
  --out eval/results/boundary_canonical_balanced

# 多帧
python -u -m pss.eval.scripts.run_boundary_multiframe \
  --out eval/results/boundary_bench/multiframe_extended
```

---

## 7. 一句话

**Ship 类平衡 Bayes（strength 0.5，cal 选出）：像素契约 0.848/0.962，canonical 3°+1% 0.871/0.957，overall 回退 ≤0.7 pt；压力档用多帧 M≥64 或接受 full-balance 的 overall 回退。** 自然先验和 0.948 都不再引用；真实 RGB-D 与弃权曲线仍是开放项。
