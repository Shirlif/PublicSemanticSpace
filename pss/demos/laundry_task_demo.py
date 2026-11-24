"""Long-horizon task demo: relations + action operators (rebuilt from 6.7).

State is a set of relation tuples, not if/else branches. Every action checks
its preconditions, mutates the relation set, and the caller prints a snapshot.
This is the logic-level skeleton; perception, learned preconditions, and
exception recovery are deliberately out of scope for Phase 0.

Run: ``python -m pss.demos.laundry_task_demo``
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pss._console import ensure_utf8

WASHER = "washer"
DRYING_RACK = "drying_rack"


@dataclass
class Scene:
    facts: set[tuple] = field(default_factory=set)

    def add(self, *fact: str) -> None:
        self.facts.add(tuple(fact))

    def remove(self, *fact: str) -> None:
        self.facts.discard(tuple(fact))

    def has(self, *fact: str) -> bool:
        return tuple(fact) in self.facts

    def replace(self, old: tuple, new: tuple) -> None:
        self.facts.discard(tuple(old))
        self.facts.add(tuple(new))

    def snapshot(self) -> list[str]:
        return sorted(":".join(f) for f in self.facts)


def initial_scene() -> Scene:
    s = Scene()
    for fact in [
        ("clothes", "in", "basket"),
        ("clothes", "dry"),
        ("detergent", "on", "counter"),
        ("door", "closed"),
        ("machine", "stopped"),
        ("water", "drained"),
        ("drying_rack", "empty"),
    ]:
        s.add(*fact)
    return s


def pick(s: Scene, obj: str) -> tuple[bool, str]:
    for loc in ("basket", "counter", "drying_rack"):
        for prep in ("in", "on"):
            if s.has(obj, prep, loc):
                s.remove(obj, prep, loc)
                s.add(obj, "held")
                return True, f"{obj} 从 {loc} 拿起"
    return False, f"{obj} 不在可拿起的位置（前置条件不满足）"


def open_door(s: Scene, target: str = WASHER) -> tuple[bool, str]:
    if not s.has("door", "closed"):
        return False, "门不是关闭状态"
    if not s.has("machine", "stopped"):
        return False, "机器未停止，不能开门"
    s.replace(("door", "closed"), ("door", "open"))
    return True, "门已打开"


def close_door(s: Scene, target: str = WASHER) -> tuple[bool, str]:
    if not s.has("door", "open"):
        return False, "门不是打开状态"
    s.replace(("door", "open"), ("door", "closed"))
    return True, "门已关闭"


def put_in(s: Scene, obj: str, dest: str) -> tuple[bool, str]:
    if not s.has(obj, "held"):
        return False, f"{obj} 未被拿起"
    if not s.has("door", "open"):
        return False, "门未打开，不能放入"
    s.remove(obj, "held")
    s.add(obj, "in", dest)
    return True, f"{obj} 已放入 {dest}"


def pour(s: Scene, src: str, dest: str) -> tuple[bool, str]:
    if not s.has(src, "held"):
        return False, f"{src} 未被拿起"
    if not s.has("door", "open"):
        return False, "门未打开，不能倾倒"
    s.remove(src, "held")
    s.add(src, "in", dest)
    return True, f"{src} 已倒入 {dest}"


def fill_water(s: Scene, dest: str) -> tuple[bool, str]:
    if not s.has("door", "closed"):
        return False, "门未关闭，不能注水"
    if not s.has("water", "drained"):
        return False, "已有水或水源未排空"
    if not s.has("clothes", "in", dest):
        return False, "衣服不在洗衣机内"
    s.replace(("water", "drained"), ("water", "in", dest))
    return True, "已注水"


def press_start(s: Scene, dest: str) -> tuple[bool, str]:
    required = [
        ("door", "closed"),
        ("water", "in", dest),
        ("clothes", "in", dest),
        ("detergent", "in", dest),
        ("machine", "stopped"),
    ]
    for fact in required:
        if not s.has(*fact):
            return False, f"缺少前置条件: {':'.join(fact)}"
    s.replace(("machine", "stopped"), ("machine", "running"))
    return True, "机器启动"


def wait(s: Scene, dest: str) -> tuple[bool, str]:
    if not s.has("machine", "running"):
        return False, "机器未在运行"
    s.replace(("machine", "running"), ("machine", "stopped"))
    s.remove("water", "in", dest)
    s.add("water", "drained")
    s.remove("clothes", "dry")
    s.remove("clothes", "dirty")
    s.add("clothes", "clean")
    return True, "洗涤完成：机器停止、水排空、衣服变干净"


def take_out(s: Scene, obj: str, dest: str) -> tuple[bool, str]:
    if not s.has(obj, "in", dest):
        return False, f"{obj} 不在 {dest} 内"
    if not s.has("door", "open"):
        return False, "门未打开，不能取出"
    s.remove(obj, "in", dest)
    s.add(obj, "held")
    return True, f"{obj} 已取出"


def hang(s: Scene, obj: str, rack: str) -> tuple[bool, str]:
    if not s.has(obj, "held"):
        return False, f"{obj} 未被拿起"
    if not s.has(rack, "empty"):
        return False, f"{rack} 已被占用"
    s.remove(obj, "held")
    s.remove(rack, "empty")
    s.add(obj, "on", rack)
    return True, f"{obj} 已挂到 {rack}"


PLAN = [
    ("拿起衣服", lambda s: pick(s, "clothes")),
    ("打开洗衣机门", lambda s: open_door(s)),
    ("把衣服放进洗衣机", lambda s: put_in(s, "clothes", WASHER)),
    ("拿起洗衣液", lambda s: pick(s, "detergent")),
    ("倒入洗衣液", lambda s: pour(s, "detergent", WASHER)),
    ("关闭洗衣机门", lambda s: close_door(s)),
    ("注水", lambda s: fill_water(s, WASHER)),
    ("按下开始", lambda s: press_start(s, WASHER)),
    ("等待洗完", lambda s: wait(s, WASHER)),
    ("再次打开门", lambda s: open_door(s)),
    ("取出衣服", lambda s: take_out(s, "clothes", WASHER)),
    ("挂到晾衣架", lambda s: hang(s, "clothes", DRYING_RACK)),
]


def run(verbose: bool = True) -> dict:
    scene = initial_scene()
    log = []
    if verbose:
        print("=== 长时序任务 Demo：关系表征 + 动作算子（Phase 0 重建）===")
        print(f"初始场景：{' | '.join(scene.snapshot())}")
        print()

    for i, (name, action) in enumerate(PLAN, 1):
        ok, msg = action(scene)
        log.append({"step": i, "action": name, "ok": ok, "msg": msg})
        if verbose:
            print(f"[{i:>2}] {name} … {'✅' if ok else '❌'} {msg}")
            print(f"     场景：{' | '.join(scene.snapshot())}")
        if not ok:
            break

    goal_reached = scene.has("clothes", "on", DRYING_RACK)

    # A deliberate precondition rejection, to show the blocking path.
    failure_scene = initial_scene()
    rejected, reject_msg = press_start(failure_scene, WASHER)

    if verbose:
        print()
        print(f"目标达成（clothes on drying_rack）：{'✅' if goal_reached else '❌'}")
        print(f"前置条件拒绝演示：按下开始 → {'✅ 被拒绝' if not rejected else '❌ 未拒绝'}（{reject_msg}）")
        print()
        print("边界（6.7）：动作序列是手写的；前置条件是规则；状态由代码直接修改；")
        print("未接感知前端，也未从交互数据学习前置条件。验证的是结构，不是学习。")

    return {
        "steps": len(log),
        "goal_reached": goal_reached,
        "log": log,
        "facts": scene.snapshot(),
        "rejection": {"ok": rejected, "msg": reject_msg},
    }


if __name__ == "__main__":
    ensure_utf8()
    run()
