"""Procedural 3D tabletop generator for the T-text spatial suite.

One scene per item keeps serialization simple and prevents cross-item leakage.
All answers are computed by ``predicates.py``; the scene JSON never contains a
relation, distance, or answer.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from . import predicates as P
from .schema import Camera, Item, ObjectSpec, Query, Scene

TABLE = (-0.30, 0.30, -0.20, 0.20)

CATEGORY_SPECS: dict[str, dict] = {
    "cup": {"half": (0.040, 0.040, 0.050)},
    "mug": {"half": (0.045, 0.045, 0.045)},
    "bowl": {"half": (0.070, 0.070, 0.030)},
    "plate": {"half": (0.090, 0.090, 0.008)},
    "box": {"half": (0.060, 0.045, 0.035)},
    "block": {"half": (0.035, 0.035, 0.035)},
    "bin": {"half": (0.080, 0.080, 0.060), "container": True, "inner_radius": 0.060},
    "bottle": {"half": (0.030, 0.030, 0.080)},
    "can": {"half": (0.035, 0.035, 0.060)},
    "jar": {"half": (0.050, 0.050, 0.070)},
}

TRAIN_CATEGORIES = ("cup", "mug", "bowl", "plate", "box", "block", "bin")
UNSEEN_CATEGORIES = ("bottle", "can", "jar")

CAMERAS = (
    Camera("cam0", (-0.70, -0.90, 0.80), (0.0, 0.0, 0.05)),
    Camera("cam1", (0.85, -0.55, 0.75), (0.0, 0.0, 0.05)),
)

DIRECTIONAL = ("left_of", "right_of", "front_of", "behind")
BOOL_FAMILIES = (
    "relation",
    "vertical",
    "frame",
    "transitive",
    "occlusion",
    "metric",
    "perspective",
)
ALL_FAMILIES = (
    "relation",
    "vertical",
    "frame",
    "transitive",
    "count",
    "ordinal",
    "occlusion",
    "metric",
    "perspective",
)

PHRASE = {
    "left_of": "to the left of",
    "right_of": "to the right of",
    "front_of": "in front of",
    "behind": "behind",
    "above": "above",
    "below": "below",
    "contact": "touching",
    "near": "near",
    "inside": "inside",
    "on": "on top of",
}


def _spec(category: str) -> dict:
    return CATEGORY_SPECS[category]


def _make_object(idx: int, category: str, pos: tuple[float, float, float], yaw: float) -> ObjectSpec:
    spec = _spec(category)
    half = tuple(float(v) for v in spec["half"])
    return ObjectSpec(
        id=f"obj_{idx}",
        category=category,
        pos=(float(pos[0]), float(pos[1]), float(pos[2])),
        half=half,
        yaw=float(yaw),
        container=bool(spec.get("container", False)),
        inner_radius=float(spec.get("inner_radius", 0.0)),
    )


def _rest_z(category: str) -> float:
    return float(_spec(category)["half"][2]) + 0.001


def _clamp_xy(x: float, y: float, radius: float) -> tuple[float, float]:
    x_min, x_max, y_min, y_max = TABLE
    return (
        float(np.clip(x, x_min + radius + 0.01, x_max - radius - 0.01)),
        float(np.clip(y, y_min + radius + 0.01, y_max - radius - 0.01)),
    )


def _free_positions(n: int, rng: np.random.Generator, radii: list[float]) -> list[tuple[float, float]]:
    x_min, x_max, y_min, y_max = TABLE
    placed: list[tuple[tuple[float, float], float]] = []
    for radius in radii:
        for _ in range(400):
            x = float(rng.uniform(x_min + radius, x_max - radius))
            y = float(rng.uniform(y_min + radius, y_max - radius))
            if all(math.hypot(x - px, y - py) > radius + pr + 0.02 for (px, py), pr in placed):
                placed.append(((x, y), radius))
                break
        else:
            # Fall back to a grid slot.
            cols = int(math.ceil(math.sqrt(n)))
            rows = int(math.ceil(n / cols))
            slot = len(placed)
            gx = x_min + (slot % cols + 0.5) * (x_max - x_min) / cols
            gy = y_min + (slot // cols + 0.5) * (y_max - y_min) / rows
            placed.append(((float(gx), float(gy)), radius))
    return [xy for xy, _radius in placed]


def _grid_positions(n: int, rng: np.random.Generator) -> list[tuple[float, float]]:
    x_min, x_max, y_min, y_max = TABLE
    cells = [(i, j) for i in range(3) for j in range(3)]
    rng.shuffle(cells)
    out = []
    for i, j in cells[:n]:
        x = x_min + (i + 0.5) * (x_max - x_min) / 3
        y = y_min + (j + 0.5) * (y_max - y_min) / 3
        out.append((float(x + rng.uniform(-0.01, 0.01)), float(y + rng.uniform(-0.01, 0.01))))
    return out


def make_scene(
    rng: np.random.Generator,
    scene_id: str,
    split: str = "test",
    ood: str = "id",
    n_objects: int | None = None,
    force_container: bool = False,
    force_stack: bool = False,
    force_inside: bool = False,
    force_occlusion: bool = False,
) -> Scene | None:
    if n_objects is None:
        n_objects = int(rng.integers(3, 5)) if ood != "count" else int(rng.integers(5, 7))
    n_objects = max(2, min(n_objects, 6))
    pool = UNSEEN_CATEGORIES if ood == "object" else TRAIN_CATEGORIES
    categories = [str(rng.choice(pool)) for _ in range(n_objects)]

    # Only inject a support/container that exists in this split's vocabulary.
    # Injecting a seen "bin" into the unseen-object split would leak the split.
    if (force_container or force_inside) and "bin" in pool:
        if "bin" not in categories:
            categories[0] = "bin"
    if force_stack and not any(c in {"box", "block", "jar", "can", "bottle"} for c in categories):
        categories[1] = "box"

    radii = [float(max(_spec(c)["half"][0], _spec(c)["half"][1])) for c in categories]
    if ood == "layout":
        xy = _grid_positions(n_objects, rng)
    else:
        xy = _free_positions(n_objects, rng, radii)

    objects = []
    for idx, (category, (x, y), radius) in enumerate(zip(categories, xy, radii)):
        x, y = _clamp_xy(x, y, radius)
        objects.append(
            _make_object(idx, category, (x, y, _rest_z(category)), float(rng.uniform(0, 2 * math.pi)))
        )

    if force_inside:
        objects = _place_inside(objects, rng)
        if objects is None:
            return None
    if force_stack:
        objects = _place_stack(objects, rng)
        if objects is None:
            return None
    if force_occlusion:
        objects = _place_occluded_pair(objects, rng)
        if objects is None:
            return None

    return Scene(
        scene_id=scene_id,
        objects=tuple(objects),
        cameras=CAMERAS,
        table=TABLE,
        split=split,
        ood=ood,
    )


def _place_inside(objects: list[ObjectSpec], rng: np.random.Generator) -> list[ObjectSpec] | None:
    containers = [i for i, obj in enumerate(objects) if obj.container]
    if not containers:
        return None
    bin_idx = int(rng.choice(containers))
    bin_obj = objects[bin_idx]
    candidates = [
        i
        for i, obj in enumerate(objects)
        if i != bin_idx and obj.radius < bin_obj.inner_radius - 0.005 and obj.height <= bin_obj.height
    ]
    if not candidates:
        return None
    idx = int(rng.choice(candidates))
    obj = objects[idx]
    margin = max(0.002, bin_obj.inner_radius - obj.radius - 0.004)
    dx, dy = rng.uniform(-margin, margin, size=2)
    z = bin_obj.bottom + obj.half[2] + 0.002
    moved = obj.moved((bin_obj.pos[0] + dx, bin_obj.pos[1] + dy, z))
    out = list(objects)
    out[idx] = moved
    return out


def _place_stack(objects: list[ObjectSpec], rng: np.random.Generator) -> list[ObjectSpec] | None:
    supports = [
        i
        for i, obj in enumerate(objects)
        if obj.category in {"box", "block", "jar", "can", "bottle"} and obj.top <= 0.20
    ]
    if not supports:
        return None
    support_idx = int(rng.choice(supports))
    support = objects[support_idx]
    candidates = [
        i
        for i, obj in enumerate(objects)
        if i != support_idx and obj.radius <= support.radius and obj.category not in {"bin", "plate"}
    ]
    if not candidates:
        return None
    idx = int(rng.choice(candidates))
    obj = objects[idx]
    dx, dy = rng.uniform(-0.008, 0.008, size=2)
    z = support.top + obj.half[2] + 0.001
    moved = obj.moved((support.pos[0] + dx, support.pos[1] + dy, z))
    out = list(objects)
    out[idx] = moved
    return out


def _place_occluded_pair(objects: list[ObjectSpec], rng: np.random.Generator) -> list[ObjectSpec] | None:
    order = sorted(range(len(objects)), key=lambda i: objects[i].radius, reverse=True)
    if len(order) < 2:
        return None
    cam = np.asarray(CAMERAS[0].pos, dtype=float)
    for front_idx in order:
        front = objects[front_idx]
        for back_idx in order[::-1]:
            if back_idx == front_idx:
                continue
            back = objects[back_idx]
            direction = np.asarray(front.pos, dtype=float) - cam
            direction[2] = 0.0
            norm = float(np.linalg.norm(direction))
            if norm < 1e-6:
                continue
            direction = direction / norm
            distance = front.radius + back.radius + 0.02
            target = np.asarray(front.pos, dtype=float) + direction * distance
            x, y = _clamp_xy(float(target[0]), float(target[1]), back.radius)
            moved = back.moved((x, y, _rest_z(back.category)))
            trial = list(objects)
            trial[back_idx] = moved
            scene = Scene("trial", tuple(trial), CAMERAS, TABLE)
            if P.occluded(scene, moved, front, "cam0"):
                return trial
    return None


def _qid(counter: list[int]) -> str:
    counter[0] += 1
    return f"q_{counter[0]:06d}"


def _frame_phrase(frame: str) -> str:
    if frame == "world":
        return "in the world frame (+x right, +y away from the viewer)"
    if frame == "viewer":
        return "from the viewer's perspective (cam0)"
    if frame == "object":
        return "in the subject's own frame"
    if frame.startswith("viewer:"):
        return f"from the {frame.split(':', 1)[1]} perspective"
    return f"in the {frame} frame"


def _pick_pair(scene: Scene, rng: np.random.Generator) -> tuple[ObjectSpec, ObjectSpec]:
    ids = rng.choice(len(scene.objects), size=2, replace=False)
    return scene.objects[int(ids[0])], scene.objects[int(ids[1])]


def _direction_label(scene: Scene, a: ObjectSpec, b: ObjectSpec, predicate: str, frame: str) -> bool:
    return P.relation(scene, a, b, predicate, frame)


def make_relation_query(
    scene: Scene, rng: np.random.Generator, want: bool, counter: list[int], frame_choices: Iterable[str]
) -> Query | None:
    a, b = _pick_pair(scene, rng)
    frame = str(rng.choice(list(frame_choices)))
    predicate = str(rng.choice(("left_of", "right_of", "front_of", "behind", "contact", "near")))
    label = _direction_label(scene, a, b, predicate, frame)
    if label != want:
        return None
    if predicate in {"contact", "near"}:
        question = f"Is {a.id} {PHRASE[predicate]} {b.id}?"
    else:
        question = f"Is {a.id} {PHRASE[predicate]} {b.id} {_frame_phrase(frame)}?"
    return Query(
        query_id=_qid(counter),
        family="relation",
        kind="bool",
        question=question,
        answer="yes" if label else "no",
        answer_format="Answer with exactly one word: yes or no.",
        meta={"op": "relation", "subject": a.id, "object": b.id, "predicate": predicate, "frame": frame},
    )


def make_vertical_query(scene: Scene, rng: np.random.Generator, want: bool, counter: list[int]) -> Query | None:
    a, b = _pick_pair(scene, rng)
    if rng.random() < 0.5:
        a, b = b, a
    predicate = "above" if rng.random() < 0.5 else "below"
    label = P.relation(scene, a, b, predicate, "world")
    if label != want:
        return None
    question = f"Is {a.id} {predicate} {b.id} in the world frame?"
    return Query(
        query_id=_qid(counter),
        family="vertical",
        kind="bool",
        question=question,
        answer="yes" if label else "no",
        answer_format="Answer with exactly one word: yes or no.",
        meta={"op": "relation", "subject": a.id, "object": b.id, "predicate": predicate, "frame": "world"},
    )


def make_frame_query(
    scene: Scene, rng: np.random.Generator, want: bool, counter: list[int], frame_choices: Iterable[str]
) -> Query | None:
    a, b = _pick_pair(scene, rng)
    predicate = str(rng.choice(DIRECTIONAL))
    frame = str(rng.choice(list(frame_choices)))
    label = _direction_label(scene, a, b, predicate, frame)
    if label != want:
        return None
    # This family is only meaningful when the named frame disagrees with a
    # fixed alternative frame. Without this filter, most items are just
    # "world left/right" in disguise.
    other_frame = "viewer" if frame == "world" else "world"
    other_label = _direction_label(scene, a, b, predicate, other_frame)
    if label == other_label:
        return None
    question = f"Is {a.id} {PHRASE[predicate]} {b.id} {_frame_phrase(frame)}?"
    return Query(
        query_id=_qid(counter),
        family="frame",
        kind="bool",
        question=question,
        answer="yes" if label else "no",
        answer_format="Answer with exactly one word: yes or no.",
        meta={
            "op": "relation",
            "subject": a.id,
            "object": b.id,
            "predicate": predicate,
            "frame": frame,
            "frame_sensitive": True,
        },
    )


def make_transitive_query(
    scene: Scene, rng: np.random.Generator, want: bool, counter: list[int], frame_choices: Iterable[str]
) -> Query | None:
    if len(scene.objects) < 3:
        return None
    frame = str(rng.choice(list(frame_choices)))
    right, _front, _up = P.frame_axes(scene, frame)
    ordered = sorted(scene.objects, key=lambda obj: float(np.dot(np.asarray(obj.pos), -right)), reverse=True)
    a, b, c = ordered[0], ordered[1], ordered[2]
    for x, y in ((a, b), (b, c), (a, c)):
        if abs(float(np.dot(np.asarray(x.pos) - np.asarray(y.pos), right))) < 0.01:
            return None
    # Premises are true by construction; the conclusion is either A left C or C left A.
    subject, target = (a, c) if want else (c, a)
    label = P.relation(scene, subject, target, "left_of", frame)
    if label != want:
        return None
    premise_frame = "world frame" if frame == "world" else "viewer perspective"
    question = (
        f"Given that {a.id} is to the left of {b.id} ({premise_frame}) and "
        f"{b.id} is to the left of {c.id} ({premise_frame}), "
        f"is {subject.id} to the left of {target.id}?"
    )
    return Query(
        query_id=_qid(counter),
        family="transitive",
        kind="bool",
        question=question,
        answer="yes" if want else "no",
        answer_format="Answer with exactly one word: yes or no.",
        meta={
            "op": "transitive",
            "subject": subject.id,
            "object": target.id,
            "predicate": "left_of",
            "frame": frame,
            "premises": [[a.id, b.id], [b.id, c.id]],
        },
    )


def make_count_query(scene: Scene, rng: np.random.Generator, counter: list[int], frame_choices: Iterable[str]) -> Query | None:
    if len(scene.objects) < 3:
        return None
    predicate = str(rng.choice(("left_of", "right_of", "front_of", "behind", "inside", "near")))
    if predicate == "inside":
        containers = [obj for obj in scene.objects if obj.container]
        if not containers:
            return None
        reference = containers[int(rng.integers(0, len(containers)))]
        frame = "world"
    else:
        reference = scene.objects[int(rng.integers(0, len(scene.objects)))]
        frame = str(rng.choice(list(frame_choices)))
    answer = P.count_relation(scene, reference, predicate, frame)
    if predicate == "near" and answer == 0:
        return None
    if predicate == "inside":
        question = f"How many objects are inside {reference.id}? Count all other objects."
    else:
        question = (
            f"How many objects are {PHRASE[predicate]} {reference.id} "
            f"{_frame_phrase(frame)}? Count all other objects."
        )
    return Query(
        query_id=_qid(counter),
        family="count",
        kind="count",
        question=question,
        answer=str(answer),
        answer_format="Answer with a single integer.",
        meta={"op": "count", "reference": reference.id, "predicate": predicate, "frame": frame},
    )


def make_ordinal_query(scene: Scene, rng: np.random.Generator, counter: list[int], frame_choices: Iterable[str]) -> Query | None:
    counts: dict[str, int] = {}
    for obj in scene.objects:
        counts[obj.category] = counts.get(obj.category, 0) + 1
    eligible = [category for category, n in counts.items() if n >= 2]
    if not eligible:
        return None
    category = str(rng.choice(eligible))
    k_max = min(3, counts[category])
    k = int(rng.integers(1, k_max + 1))
    frame = str(rng.choice(list(frame_choices)))
    answer = P.ordinal_from_left(scene, category, k, frame)
    if answer is None:
        return None
    right, _front, _up = P.frame_axes(scene, frame)
    projections = [
        float(np.dot(np.asarray(obj.pos), -right)) for obj in scene.objects if obj.category == category
    ]
    if len(set(round(v, 4) for v in projections)) != len(projections):
        return None
    question = (
        f"Which {category} is the {k}-th from the left {_frame_phrase(frame)}? "
        "The leftmost one is 1st."
    )
    return Query(
        query_id=_qid(counter),
        family="ordinal",
        kind="object",
        question=question,
        answer=answer,
        answer_format="Answer with exactly one object id, for example obj_2.",
        meta={"op": "ordinal", "category": category, "k": k, "frame": frame},
    )


def make_occlusion_query(scene: Scene, rng: np.random.Generator, want: bool, counter: list[int]) -> Query | None:
    a, b = _pick_pair(scene, rng)
    label = P.occluded(scene, a, b, "cam0")
    if label != want:
        return None
    question = f"Is {a.id} occluded by {b.id} from camera cam0?"
    return Query(
        query_id=_qid(counter),
        family="occlusion",
        kind="bool",
        question=question,
        answer="yes" if label else "no",
        answer_format="Answer with exactly one word: yes or no.",
        meta={"op": "occluded", "subject": a.id, "object": b.id, "camera": "cam0"},
    )


def make_metric_query(scene: Scene, rng: np.random.Generator, want: bool, counter: list[int]) -> Query | None:
    subtype = str(rng.choice(("closer", "fits")))
    if subtype == "fits":
        containers = [obj for obj in scene.objects if obj.container]
        if not containers:
            return None
        container = containers[int(rng.integers(0, len(containers)))]
        candidates = [obj for obj in scene.objects if obj.id != container.id]
        if not candidates:
            return None
        obj = candidates[int(rng.integers(0, len(candidates)))]
        label = P.fits(obj, container)
        if label != want:
            return None
        question = f"Does {obj.id} fit inside {container.id}?"
        return Query(
            query_id=_qid(counter),
            family="metric",
            kind="bool",
            question=question,
            answer="yes" if label else "no",
            answer_format="Answer with exactly one word: yes or no.",
            meta={"op": "fits", "subject": obj.id, "object": container.id},
        )
    a, b = _pick_pair(scene, rng)
    label = P.closer_than(scene, a, b, "cam0")
    if label != want:
        return None
    question = f"Is {a.id} closer to camera cam0 than {b.id}?"
    return Query(
        query_id=_qid(counter),
        family="metric",
        kind="bool",
        question=question,
        answer="yes" if label else "no",
        answer_format="Answer with exactly one word: yes or no.",
        meta={"op": "closer", "subject": a.id, "object": b.id, "camera": "cam0"},
    )


def make_perspective_query(scene: Scene, rng: np.random.Generator, want: bool, counter: list[int]) -> Query | None:
    a, b = _pick_pair(scene, rng)
    predicate = str(rng.choice(DIRECTIONAL))
    label = P.relation(scene, a, b, predicate, "viewer:cam1")
    if label != want:
        return None
    question = f"From camera cam1's perspective, is {a.id} {PHRASE[predicate]} {b.id}?"
    return Query(
        query_id=_qid(counter),
        family="perspective",
        kind="bool",
        question=question,
        answer="yes" if label else "no",
        answer_format="Answer with exactly one word: yes or no.",
        meta={"op": "relation", "subject": a.id, "object": b.id, "predicate": predicate, "frame": "viewer:cam1"},
    )


def _make_query(
    scene: Scene,
    family: str,
    want: bool | None,
    rng: np.random.Generator,
    counter: list[int],
    ood: str,
) -> Query | None:
    frame_choices = ("object",) if ood == "frame" and family in {"relation", "frame"} else ("world", "viewer")
    if family == "relation":
        return make_relation_query(scene, rng, bool(want), counter, frame_choices)
    if family == "vertical":
        return make_vertical_query(scene, rng, bool(want), counter)
    if family == "frame":
        return make_frame_query(scene, rng, bool(want), counter, frame_choices)
    if family == "transitive":
        return make_transitive_query(scene, rng, bool(want), counter, frame_choices)
    if family == "count":
        return make_count_query(scene, rng, counter, frame_choices)
    if family == "ordinal":
        return make_ordinal_query(scene, rng, counter, frame_choices)
    if family == "occlusion":
        return make_occlusion_query(scene, rng, bool(want), counter)
    if family == "metric":
        return make_metric_query(scene, rng, bool(want), counter)
    if family == "perspective":
        return make_perspective_query(scene, rng, bool(want), counter)
    raise ValueError(f"unknown family: {family}")


def generate_suite(
    n_per_family: int = 100,
    seed: int = 0,
    split: str = "test",
    ood: str = "id",
    families: Iterable[str] = ALL_FAMILIES,
) -> list[Item]:
    rng = np.random.default_rng(seed)
    counter = [0]
    items: list[Item] = []
    scene_counter = [0]

    def new_scene(family: str, want: bool | None) -> Scene | None:
        scene_counter[0] += 1
        # Count items sometimes get a container with an object inside; the rest
        # stay free-layout. Forcing it every time would bias the count labels.
        force_inside = family == "count" and rng.random() < 0.4
        return make_scene(
            rng,
            scene_id=f"{split}_{ood}_{scene_counter[0]:05d}",
            split=split,
            ood=ood,
            force_container=force_inside,
            force_stack=family == "vertical",
            force_inside=force_inside,
            force_occlusion=family == "occlusion" and bool(want),
        )

    for family in families:
        if family in BOOL_FAMILIES:
            per_label = max(1, n_per_family // 2)
            for want in (True, False):
                made = 0
                attempts = 0
                limit = max(200, per_label * 80)
                while made < per_label and attempts < limit:
                    attempts += 1
                    scene = new_scene(family, want)
                    if scene is None:
                        continue
                    query = _make_query(scene, family, want, rng, counter, ood)
                    if query is None:
                        continue
                    items.append(Item(scene=scene, query=query))
                    made += 1
        else:
            made = 0
            attempts = 0
            limit = max(200, n_per_family * 80)
            while made < n_per_family and attempts < limit:
                attempts += 1
                scene = new_scene(family, None)
                if scene is None:
                    continue
                query = _make_query(scene, family, None, rng, counter, ood)
                if query is None:
                    continue
                items.append(Item(scene=scene, query=query))
                made += 1
    return items
