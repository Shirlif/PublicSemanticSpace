"""Token autoregression vs explicit vector state (rebuilt from section 6.6).

The physics question is the same one as in the document:

    苹果在 x=0，杯子在 x=2.2，苹果以 1.0 m/s 向右运动，碰撞阈值 0.5 m。
    3 秒后苹果在哪？会碰到杯子吗？

Path A simulates the *structure* of token autoregression: every step re-reads
the whole growing context, so attention cost is O(N^2). This is NOT a real LLM;
KV cache would avoid recomputation but would not remove the linear growth of
the context (6.6 boundary).

Path B keeps an explicit state: two objects, and only the apple's position is
updated each step. Attribute vectors are stored once and never touched.

Run: ``python -m pss.demos.token_vs_vector_reasoning``
"""

from __future__ import annotations

import re

import numpy as np

from pss._console import ensure_utf8

STEPS = 3
DT = 1.0
V0 = 1.0
CUP_X = 2.2
RADIUS = 0.25
THRESH = 0.5

SCENE_TEXT = (
    "场景描述：桌面上有一个苹果，颜色为红色，表面光滑，形状为近似球形，"
    "半径为0.25米，初始位置在x=0米，y=0米；桌子右侧x=2.2米处有一个杯子，"
    "颜色为蓝色，形状为圆柱形，表面光滑，半径为0.25米，位置固定不动。"
    "苹果在t=0时刻被向右推动，初速度为1.0米每秒，方向沿x轴正方向；"
    "忽略摩擦力与空气阻力。问题：3秒后苹果在什么位置，是否会碰到杯子？"
    "碰撞判定阈值为两物体中心距离小于0.5米。"
)

_CJK = re.compile(r"[一-鿿]")
_WORD = re.compile(r"[A-Za-z0-9_.=+\-]+")


def count_tokens(text: str) -> int:
    """A simple, deterministic token counter for the simulation.

    Counts each CJK character as one token and each ASCII/number run as one.
    The absolute numbers will differ from the document because the exact
    serialization differs; the *growth pattern* is what matters.
    """
    return len(_CJK.findall(text)) + len(_WORD.findall(text))


def _verdict_text(t: int, x: float, dist: float, hit: bool) -> str:
    passed = x > CUP_X
    return (
        f"第{t}秒结束：苹果沿x轴正方向移动，当前坐标为x={x:.1f}米；"
        f"杯子保持在x={CUP_X:.1f}米不动；两物体中心距离为{dist:.1f}米，"
        f"{'小于' if hit else '大于'}碰撞阈值{THRESH:.1f}米，"
        f"因此本时刻{'发生碰撞' if hit else '尚未碰撞'}。"
        + ("苹果已越过杯子所在位置。" if passed else "")
    )


def run_token_path() -> tuple[list[dict], int]:
    context = SCENE_TEXT
    rows = []
    total = 0
    for t in range(1, STEPS + 1):
        n = count_tokens(context)
        cost = n * n
        total += cost
        x = V0 * t
        dist = abs(CUP_X - x)
        hit = dist < THRESH
        conclusion = _verdict_text(t, x, dist, hit)
        rows.append(
            {
                "t": t,
                "tokens": n,
                "cost": cost,
                "x": x,
                "dist": dist,
                "hit": hit,
                "conclusion": conclusion,
            }
        )
        context = context + conclusion
    return rows, total


def run_vector_path() -> list[dict]:
    apple_pos = np.array([0.0, 0.0])
    velocity = np.array([V0, 0.0])
    cup_pos = np.array([CUP_X, 0.0])
    rows = []
    for t in range(1, STEPS + 1):
        # In-place state update: only the changed object is touched.
        apple_pos += velocity * DT
        dist = float(np.linalg.norm(apple_pos - cup_pos))
        hit = dist < THRESH
        rows.append(
            {
                "t": t,
                "x": float(apple_pos[0]),
                "dist": dist,
                "hit": hit,
                "dims_touched": 2,
            }
        )
    return rows


def run(verbose: bool = True) -> dict:
    token_rows, token_total = run_token_path()
    vector_rows = run_vector_path()
    answers_match = all(
        token_rows[i]["hit"] == vector_rows[i]["hit"] for i in range(STEPS)
    )
    first_collision = next((r["t"] for r in vector_rows if r["hit"]), None)

    if verbose:
        print("=== 对比实验：Token 自回归 vs 向量空间推理（Phase 0 重建）===")
        print()
        print("题目：苹果 x=0，杯子 x=2.2，苹果以 1.0 m/s 向右运动；")
        print("      3 秒后苹果在哪？会碰到杯子吗？（碰撞阈值 0.5 m）")
        print()
        print("路径 A：Token 自回归（结构模拟）")
        print(f"  {'t':>2}  {'输入 token':>10}  {'注意力开销 N²':>14}  结论")
        for r in token_rows:
            print(
                f"  {r['t']:>2}  {r['tokens']:>10}  {r['cost']:>14,}  "
                f"x={r['x']:.1f}, 距离={r['dist']:.1f}, "
                f"{'会碰到' if r['hit'] else '还没碰到'}"
            )
        print(f"  累计注意力计算量：{token_total:,}")
        print()
        print("路径 B：向量空间（显式工作记忆）")
        print(f"  {'t':>2}  {'苹果 x':>8}  {'距离':>6}  判定         本步计算维度")
        for r in vector_rows:
            print(
                f"  {r['t']:>2}  {r['x']:>8.1f}  {r['dist']:>6.2f}  "
                f"{'会碰到杯子' if r['hit'] else '还没碰到':<12} 仅位置 2 维（跳过 192 维属性）"
            )
        print()
        print("量化对比")
        print(f"  每步输入 token：{token_rows[0]['tokens']} → "
              f"{token_rows[1]['tokens']} → {token_rows[2]['tokens']}（线性增长）")
        print("  向量状态：始终 2 个客体，单步只更新位置 2 维（恒定）")
        print(f"  累计注意力计算量：{token_total:,}")
        print(f"  向量路径总更新维度：{sum(r['dims_touched'] for r in vector_rows)}")
        print()
        print(f"物理答案一致：{'✅' if answers_match else '❌'}；"
              f"首次碰撞时刻：t={first_collision}")
        print()
        print("边界（6.6）：路径 A 为结构模拟，不是真实 LLM；路径 B 为朴素状态更新，")
        print("不是学到的世界模型。结构性差异（序列重读 vs 原地更新）是真实的。")

    return {
        "token_rows": token_rows,
        "vector_rows": vector_rows,
        "token_total": token_total,
        "token_counts": [r["tokens"] for r in token_rows],
        "vector_ops": sum(r["dims_touched"] for r in vector_rows),
        "first_collision_t": first_collision,
        "answers_match": answers_match,
    }


if __name__ == "__main__":
    ensure_utf8()
    run()
