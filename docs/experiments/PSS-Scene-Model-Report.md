# PSS 三维场景建模报告（D6）：go/no-go

> 日期：2026-11-28
> 计划：`docs/plans/PSS-三维场景建模实验计划.md`
> 结果：`eval/results/scene_bench/{s1,s1_m64,s2,s2_outliers,s3,s3_v1,s3_v3,s3_v5}/`
> 状态：D1–D5 已执行。**H1 假，H2 假，H3 部分成立；按停止规则不采纳 3D 场景层为默认，1D 类平衡 Bayes 仍是 fallback。**
>
> **2026-11-28 限定/撤回**：本报告的“3D”管线是**虚拟针孔投影 + 每物体每视角一个中心点**，没有使用 MuJoCo 等真实 3D 环境；它也没有用可见表面/形状，只做了点三角化。因此本报告的结论只能写成“在已知中心点投影下，3D 点三角化 3 视角不如 1D 64 视角”，**不能**外推为“3D 建模不如 1D”。v2 已改用仓库的 MuJoCo 桌面环境 + 真实 RGB-D/分割/表面点云 + 形状拟合，见 `docs/plans/PSS-三维场景建模实验计划-v2.md`。

---

## 0. 一句话

**三维场景建模没有“用 3 帧抵 64 帧”。** 3 视角 near-aligned 0.686，1D 64 视角 0.886；即使 3D 用 64 视角也只有 0.900，仅 +0.014。根因不是求解器，是**深度噪声**：σ_depth=1% 在 11.4 m 上是 0.114 m，和 `eps_xy=0.12` 同阶，near 窗口 `[0.06,0.18]` 比单次深度噪声还窄。三维模型只是重新组织信息，不能创造信息；near-boundary 被观测精度卡死。**下一步是改观测契约（深度 ≤0.5% 或视角 ≥32 / 真实 RGB-D），不是继续加模型。**

---

## 1. S1：三维位置估计（决定性 go/no-go）

16 视角、3 seeds、100 test scenes：

| arm | near-aligned | pose median | pose p90 |
|---|---:|---:|---:|
| `3d_1view` | 0.579 ± 0.009 | 8.6 cm | 14.5 cm |
| `3d_3view` | **0.686 ± 0.019** | 4.9 cm | 8.1 cm |
| `3d_5view` | 0.702 ± 0.045 | 3.8 cm | 6.2 cm |
| `3d_all`（16） | 0.846 ± 0.011 | 2.2 cm | 3.2 cm |
| 1D `backproject_mean` @64 | **0.886 ± 0.007** | 1.6 cm | 2.3 cm |
| oracle | 1.000 | 0 | 0 |

**H1：假。** 预注册方向 `3d_3view ≥ 1D-64 + 0.05` 要求 0.936，实测 0.686；绝对 bar 0.90 也未达到。bar 未调，`eps_xy` 未改。

### 1.2 诊断

- **不是求解器**：同一 `solve_scene_v2` 的 xy RMS 与深度噪声下限 `σ_d·cos(elev)/√N ≈ 0.100/√N` 逐行吻合（3 视角 0.052 vs 0.058；16 视角 0.023 vs 0.025）。
- **是观测噪声 + 视角数**：σ_depth=1% @11.4 m = 0.114 m；near 窗口 [0.06,0.18] 比单次深度噪声还窄。
- **视角几何不是主因**：16 环上 3/5 视角换均匀分布无显著改善。
- **对应/离群/约束不参与**：outlier=0、关联已知、桌面先验只影响 z。
- **结构性上限**：3D 64 视角 near 0.900，对 1D-64 0.886 只 +0.014；+0.05 在本模拟器不可达。

---

## 2. S2：约束与消融

5 视角、3 seeds：

| arm | near | overall | pose median |
|---|---:|---:|---:|
| `3d_3view_noconstraint` | 0.761 ± 0.019 | 0.9825 | 4.03 cm |
| `3d_3view_constraints` | 0.765 ± 0.016 | 0.9826 | 3.62 cm |
| `3d_3view_nodepth` | 0.748 ± 0.018 | 0.9814 | 4.33 cm |
| oracle | 1.000 | 1.000 | 0 |

- **H2：假。** 约束 +0.4 pt，落在 seed 波动内。原因：`solve_scene` 已有桌面先验；不穿透与 near-alignment 几何不重合。
- 约束把 `same_level` 0.933→1.000、`support` 0.990→1.000；三维独有查询确实受益。
- 10% 离群：near 0.713（−5.2 pt），overall 0.979（−0.37 pt）；near 最脆弱。
- 去深度：near −1.35 pt、pose +3 mm；深度有贡献但有限。

---

## 3. S3：关系 belief 与校准

| 视角 | near | overall | bench ECE | pooled ECE | near 子集 ECE |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.5865 | 0.9554 | 0.0486 | 0.0182 | 0.2262 |
| 3 | 0.6557 | 0.9721 | 0.0369 | 0.0056 | 0.1054 |
| 5 | 0.7612 | 0.9825 | 0.0274 | 0.0042 | 0.0473 |

- **H3：主口径成立**（bench ECE ≤0.05），但 **near 子集在 3 视角 ECE 0.105 >0.05**；只有 5 视角 0.047 刚压线。不能声称 near 关系已校准。
- 1 视角协方差尾部失控：29.6% 物体超 99.9% χ²₃ 界，5.1% Mahalanobis >1000；3 视角均值 2.98（尺度正确），5 视角 2.23（略保守）。
- CLEVR 上 `same_level`/`support` 恒真、`contact`/`collision` 恒假；这些关系的 ECE 不代表关系层校准。
- 延迟 11.9 ms median / 18.2 ms p90，≤50 ms 达标。

---

## 4. S4：PSS 接入

- `pss/scene/pss_bridge.py` 把场景模型写成 `Entity`（pos、covariance、provenance、scene_id）和 `Relation`（aligned_x/y、same_level、contact、collision、support；`origin="scene_model"`）。
- 5 场景 demo：40 entities / 324 relations；`same_level` 278、`support` 38、`aligned_x/y` 8；audit/provenance/covariance 校验全过。
- 1D fallback 仍是 `axis_runtime_mode="off"`，未被替换。
- 测试：`tests/test_scene_runtime.py` 6 项 + 既有 12 项 = 18 项通过。
- 但 pose median 4.3–6.2 cm，高于计划 1 cm 守卫；这是 S1 估计器的观测噪声上限。

---

## 5. 假设判定

| 假设 | bar | 实测 | 判定 |
|---|---|---|---|
| **H1 维度收益** | `3d_3view ≥ 1D-64 +0.05` | 0.686 vs 0.886（差 −0.20） | **假** |
| **H2 约束收益** | constraints ≥ noconstraint +0.03 | +0.0042 | **假** |
| **H3 不确定度可校准** | ECE ≤0.05 | 3 视角 0.0369（主）/ 0.105（near 子集） | **部分成立**：全局过，near 不过 |
| **H4 观测预算** | 3 视角 ≥0.90；5 视角 ≥0.95 | 0.686 / 0.702 | **假** |
| pose 守卫 | median ≤1 cm、p90 ≤3 cm | 3.6–4.9 cm / 5.7–8.1 cm | **假** |

---

## 6. Go / No-Go

### 6.1 决策

**NO-GO：不把 3D 场景层设为默认几何层。**

- 1D 类平衡 Bayes（strength 0.5）仍是 fallback。
- 3D bridge 保留为 opt-in；可用于三维独有查询（same_level / contact / collision / support）和一致性检查，但不用于 near-aligned 的主判据。
- 不改变 runtime 默认；不把 `axis_mode="bayes"` 换成 scene model。

### 6.2 停止规则执行

- S1 完成，H1 假 → 按 §9 停止推进为默认方案。
- S2–S4 因并行执行已完成；结果作为 closure 报告，不提升为默认。
- 不再加新臂；不再调 `eps_xy` 或 bar。

---

## 7. 对“回到三维”这个判断的正确读法

### 7.1 支持的部分

- **三维独有查询确实能答好**：`same_level` 0.933→1.000、`support` 0.990→1.000；1D 谓词根本回答不了这些。
- **per-observation 效率更高**：3D 16 视角 0.846，而 3D 1 视角只有 0.579；三维联合估计比单帧好。
- **接入可行**：covariance/provenance/audit 都能落到 PSS，不需要改 state schema。
- **信息在坐标里**：oracle 1.000；问题不是任务不可解，是观测不够准。

### 7.2 不支持的部分

- **三维模型不能创造信息**：近边界精度由 `σ_depth/√N` 决定；3 视角不可能替代 64 帧。
- **“3 视角赢 64 帧”是错的**：0.686 vs 0.886；即使 3D 64 视角也只 +0.014。
- **约束不是免费杠杆**：桌面先验已经编码支撑；不穿透和近对齐不重合。
- **全局 ECE 会掩盖 near 子集**：3 视角 near ECE 0.105，不能说“已校准”。

### 7.3 正确的结论

**三维建模是正确的表示层，但不是观测层的替代品。** 在 `σ_depth=1%`、`eps_xy=0.12` 的契约下，near-boundary 是**观测精度受限**的；换任何模型（1D/3D/Bayes/KDE）都绕不过 `σ/eps`。要提升，只能改观测：

- 深度 ≤0.5%（σ/eps ≈0.25）或视角 ≥32/64；
- 真实 RGB-D 的深度/位姿标定；
- 多帧配准/深度融合；
- 或者在任务上接受“near-aligned 弃权”，用 3D 模型回答它擅长的 `same_level`/`support`/`contact`/`collision`。

---

## 8. 失败与残余

1. H1/H2/H4 与 pose 守卫均失败；照实保留，bar 未调。
2. H3 只在全局 ECE 成立；near 子集 3 视角 0.105。
3. 1 视角协方差尾部失控（29.6% 超 χ²₃ 99.9% 界）。
4. CLEVR 的 `same_level`/`support` 恒真、`contact`/`collision` 恒假；这些关系的标签没有区分度。
5. 多视角是虚拟针孔投影，不是完整渲染；真实检测器/分割器/关联未测。
6. 物体用球形半径近似，不是网格碰撞。

---

## 9. 下一步（按优先级）

1. **改观测契约再复测 H1**：σ_depth=0.5% 或 32/64 视角；如果 3D 3 视角仍赢不了 1D 同视角，H1 才真正稳健地假。
2. **真实 RGB-D**：用真实深度/位姿噪声替换像素/3D 代理；这是唯一能改变 `σ/eps` 的外部变量。
3. **三维独有查询的下游价值**：`same_level`/`support`/`contact`/`collision` 接规划/抓取，证明 3D 模型的增量不在 near-aligned，而在这些查询。
4. **弃权**：near-boundary 低置信时弃权；用 3D 协方差做 risk-coverage，而不是硬判。
5. **PSS bridge 保持 opt-in**：等 1–3 有结果，再讨论默认化。

---

## 10. 产物

```
pss/scene/{model,clevr,observation,metrics,bench_api,optimizer}.py   # D1
pss/scene/{optimizer_v2,arms_3d}.py                                  # D2/S1
pss/scene/{constraints,arms_constraints}.py                          # D3/S2
pss/scene/{uncertainty,arms_uncertainty}.py                          # D4/S3
pss/scene/pss_bridge.py + run_scene_runtime.py + tests               # D5/S4
pss/eval/scripts/run_scene_bench.py
docs/experiments/PSS-Scene-S1-3D-Optimizer.md
docs/experiments/PSS-Scene-S2-Constraints.md
docs/experiments/PSS-Scene-S3-Uncertainty.md
docs/experiments/PSS-Scene-Runtime.md
eval/results/scene_bench/{s1,s1_m64,s2,s2_outliers,s3,s3_v1,s3_v3,s3_v5}/
```

---

## 11. 一句话

**3D 场景建模是正确表示层，但不是观测层的替代品：在 σ_depth=1% 下，3 视角 0.686 赢不了 1D 64 帧 0.886；near-boundary 由 `σ_depth/√N` 卡死。NO-GO 默认化；1D 类平衡 Bayes 仍是 fallback；下一步改观测契约，不再加模型。**
