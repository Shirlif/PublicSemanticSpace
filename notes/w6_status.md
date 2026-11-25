# P4 / W6–W7 进度 — 关系与 OOD（S3）

> 开始：2026-11-25
> 状态：T24–T27 完成；S3 表为**管线校验**，不是 OOD 能力证据（见缺口）。

## T24 GT 关系生成

- `pss/relations/__init__.py`：10 谓词的几何判定
  （left/right/front/behind/above/below/near/touching/on_top_of/inside）。
- runner 每物理帧从 GT 位姿生成 `(subj, pred, obj)`，写入 `gt.relations` 的
  `"pred(obj)"` 字符串；只读 GT，不进 PSS。
- 只用 GT poses/contacts（`relations_gt.compute_from`）。

## T25 关系指标

- `pss/metrics/relation.py`：
  - per-predicate TP/FP/FN、P/R/F1、micro/macro F1；
  - consistency violations：非对称谓词双向、逆谓词同对、传递闭包缺失。
- `pss/metrics/relation_score.py` + `pss score-relations`：
  用 `_match_states` 把 PSS track 映射到 GT 实例，再对齐关系三元组。

## T26 S3 组合 OOD spec

- `specs/s3_relations_ood.yaml`：8 个基础 cubes + 2 个堆叠 cubes；
  `relations_gt.ood_combinations` 声明 `[on_top_of,behind]`、
  `[touching,left_of]`、`[above,in_front_of]` 三组未见组合。
- 1 episode / 5 s / oracle，无机器人无图像。

## T27 S3 表

- `scorecards/s3_relations_ood_relations.md/json`：
  - macro F1 = 1.0000，micro F1 = 1.0000（10 谓词）
  - PSS consistency violations = 0，GT violations = 0

## 关键缺口（必须如实说明）

当前 PSS 的关系推断是**几何规则**，与 GT 使用同一套谓词函数，所以 S3 的
F1=1.0 只证明“关系 GT、匹配、指标、consistency 检查”这条管线正确，
**不构成对未见组合的泛化证据**。真正的 S3 需要：
1. 用 `checkpoints/` 里的 learned relation head（或训练一个新头）从感知
   embedding 预测关系；
2. 按 `ood_combinations` 切 ID/OOD 子集分表报；
3. 在 OOD 子集上看 F1 下降。

这是 W6 未完成的能力项，不能把当前表当作 OOD 结论。
