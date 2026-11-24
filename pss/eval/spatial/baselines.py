"""Baselines for the T-text spatial suite.

Predictors take an ``Item`` and return a raw answer string (or None). The
runner parses/normalizes; the exact solver recomputes the answer from geometry
independently of the generator's stored label, which is the leakage check.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from . import predicates as P
from .schema import Item, parse_answer, parse_final_answer, render_prompt, scene_to_json
from .tools import execute_tool, parse_tool_call, tool_manifest


@dataclass
class Prediction:
    answer: str | None
    raw: str = ""
    latency_ms: float = 0.0
    trace: list[dict[str, Any]] | None = None


def solve_exact(item: Item) -> str | None:
    """Recompute the answer from the scene. No access to ``query.answer``."""
    scene = item.scene
    meta = item.query.meta
    op = meta.get("op")
    try:
        if op == "relation":
            a = scene.obj(meta["subject"])
            b = scene.obj(meta["object"])
            value = P.relation(scene, a, b, meta["predicate"], meta.get("frame", "world"))
            return "yes" if value else "no"
        if op == "transitive":
            a = scene.obj(meta["subject"])
            b = scene.obj(meta["object"])
            value = P.relation(scene, a, b, meta.get("predicate", "left_of"), meta.get("frame", "world"))
            return "yes" if value else "no"
        if op == "count":
            ref = scene.obj(meta["reference"])
            value = P.count_relation(scene, ref, meta["predicate"], meta.get("frame", "world"))
            return str(int(value))
        if op == "ordinal":
            value = P.ordinal_from_left(scene, meta["category"], int(meta["k"]), meta.get("frame", "viewer"))
            return value
        if op == "occluded":
            a = scene.obj(meta["subject"])
            b = scene.obj(meta["object"])
            value = P.occluded(scene, a, b, meta.get("camera", "cam0"))
            return "yes" if value else "no"
        if op == "closer":
            a = scene.obj(meta["subject"])
            b = scene.obj(meta["object"])
            value = P.closer_than(scene, a, b, meta.get("camera", "cam0"))
            return "yes" if value else "no"
        if op == "fits":
            a = scene.obj(meta["subject"])
            b = scene.obj(meta["object"])
            value = P.fits(a, b)
            return "yes" if value else "no"
    except Exception:
        return None
    return None


class ExactSolver:
    name = "pss_exact"

    def __call__(self, item: Item) -> str | None:
        return solve_exact(item)


class RandomBaseline:
    def __init__(self, seed: int = 0):
        self.name = "random"
        self.rng = random.Random(seed)

    def __call__(self, item: Item) -> str:
        kind = item.query.kind
        if kind == "bool":
            return self.rng.choice(("yes", "no"))
        if kind == "count":
            return str(self.rng.randint(0, max(0, len(item.scene.objects) - 1)))
        return self.rng.choice([obj.id for obj in item.scene.objects])


class WorldAxisOnlySolver:
    """A diagnostic strawman: always uses world axes, ignores reference frames
    and occlusion. It exists to show the suite discriminates; it is not an LLM.
    """

    name = "naive_world"

    def __call__(self, item: Item) -> str | None:
        scene = item.scene
        meta = item.query.meta
        op = meta.get("op")
        try:
            if op == "relation":
                a = scene.obj(meta["subject"])
                b = scene.obj(meta["object"])
                value = P.relation(scene, a, b, meta["predicate"], "world")
                return "yes" if value else "no"
            if op == "transitive":
                a = scene.obj(meta["subject"])
                b = scene.obj(meta["object"])
                value = P.relation(scene, a, b, meta.get("predicate", "left_of"), "world")
                return "yes" if value else "no"
            if op == "count":
                ref = scene.obj(meta["reference"])
                value = P.count_relation(scene, ref, meta["predicate"], "world")
                return str(int(value))
            if op == "ordinal":
                value = P.ordinal_from_left(scene, meta["category"], int(meta["k"]), "world")
                return value
            if op == "occluded":
                return "no"
            if op == "closer":
                a = scene.obj(meta["subject"])
                b = scene.obj(meta["object"])
                return "yes" if P.closer_than(scene, a, b) else "no"
            if op == "fits":
                a = scene.obj(meta["subject"])
                b = scene.obj(meta["object"])
                return "yes" if P.fits(a, b) else "no"
        except Exception:
            return None
        return None


class LLMPredictor:
    """Wrap any ``generate(prompt) -> str`` callable."""

    def __init__(
        self,
        generate: Callable[[str], str],
        name: str = "llm",
        few_shot: Iterable[Item] = (),
        system: str | None = None,
        mode: str = "zero_shot",
    ):
        self.name = name
        self.generate = generate
        self.few_shot = tuple(few_shot)
        self.system = system
        if mode not in {"zero_shot", "cot"}:
            raise ValueError(f"unknown llm mode: {mode}")
        self.mode = mode
        self.last_raw = ""

    def __call__(self, item: Item) -> str | None:
        self.last_raw = ""
        prompt = render_prompt(item, self.few_shot)
        if self.mode == "cot":
            prompt += (
                "\n\nThink step by step. End with a final line exactly in the form "
                "`Answer: <your answer>`."
            )
        if self.system:
            prompt = self.system.rstrip() + "\n\n" + prompt
        raw = self.generate(prompt)
        self.last_raw = str(raw)
        return parse_final_answer(raw, item.query.kind)


TOOL_SYSTEM = """You are a spatial reasoning agent for a tabletop scene.

You may call deterministic tools. Tool access is limited to the list below.

Protocol:
1. To call a tool, output exactly one line and nothing else:
   TOOL: {"name": "<tool>", "args": {...}}
2. After a tool call you receive a line beginning with TOOL_RESULT.
3. When you know the answer, output exactly one final line:
   Answer: <your answer>
4. Do not write code. Do not invent tool names or object ids. Do not call more tools than necessary.

Example:
Question: Is obj_0 touching obj_1?
TOOL: {"name": "surface_gap", "args": {"a": "obj_0", "b": "obj_1"}}
TOOL_RESULT: {"ok": true, "result": {"surface_gap": -0.001}}
Answer: yes"""


class ToolLoopPredictor:
    """A small ReAct-style tool loop over the whitelisted tool layer.

    ``generate`` may accept either a string prompt or a list of chat messages.
    ``load_transformers_generator`` marks itself with ``supports_messages``;
    a plain string callable gets a rendered transcript instead.
    """

    def __init__(
        self,
        generate: Callable[[Any], str],
        profile: str = "mid",
        max_steps: int = 6,
        name: str = "",
        supports_messages: bool | None = None,
    ):
        self.generate = generate
        self.profile = profile
        self.max_steps = int(max_steps)
        self.name = name or f"llm_tools:{profile}"
        if supports_messages is None:
            supports_messages = bool(getattr(generate, "supports_messages", False))
        self.supports_messages = supports_messages
        self.last_raw = ""
        self.last_trace: list[dict[str, Any]] = []
        # Validate the profile early; a typo should fail at construction.
        tool_manifest(self.profile)

    def _call(self, messages: list[dict[str, str]]) -> str:
        if self.supports_messages:
            return str(self.generate(messages))
        lines = []
        for message in messages:
            lines.append(f"{message['role'].upper()}: {message['content']}")
        lines.append("ASSISTANT:")
        return str(self.generate("\n".join(lines)))

    def __call__(self, item: Item) -> str | None:
        self.last_raw = ""
        self.last_trace = []
        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": TOOL_SYSTEM
                + "\n\nAvailable tools:\n"
                + tool_manifest(self.profile),
            },
            {
                "role": "user",
                "content": (
                    "Scene:\n"
                    + scene_to_json(item.scene)
                    + f"\n\nQuestion: {item.query.question}\n"
                    + item.query.answer_format
                ),
            },
        ]
        outputs: list[str] = []
        for step in range(self.max_steps):
            text = self._call(messages)
            outputs.append(text)
            call = parse_tool_call(text)
            if call is not None:
                outcome = execute_tool(item, call.name, call.args, self.profile)
                self.last_trace.append(
                    {
                        "step": step + 1,
                        "name": call.name,
                        "args": call.args,
                        "ok": bool(outcome.get("ok")),
                        "result": outcome.get("result"),
                        "error": outcome.get("error"),
                        "elapsed_ms": outcome.get("elapsed_ms"),
                    }
                )
                messages.append({"role": "assistant", "content": text.strip()})
                messages.append(
                    {
                        "role": "user",
                        "content": "TOOL_RESULT: " + json.dumps(outcome, ensure_ascii=False),
                    }
                )
                continue
            answer = parse_final_answer(text, item.query.kind)
            if answer is not None:
                self.last_raw = "\n---\n".join(outputs)
                return answer
            messages.append({"role": "assistant", "content": text.strip()})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Invalid response. Output exactly one TOOL: line, or a final "
                        "`Answer: <answer>` line."
                    ),
                }
            )
        messages.append(
            {"role": "user", "content": "No more tools. Output only `Answer: <answer>`."}
        )
        text = self._call(messages)
        outputs.append(text)
        self.last_raw = "\n---\n".join(outputs)
        return parse_final_answer(text, item.query.kind)


def load_transformers_generator(
    model_name: str,
    device: str = "auto",
    max_new_tokens: int = 32,
    local_files_only: bool = True,
) -> Callable[[str], str]:
    """Lazy local HF generator. Imports transformers/torch only when called."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
    kwargs = {"local_files_only": local_files_only}
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype="auto" if device != "cpu" else torch.float32,
            **kwargs,
        )
    except TypeError:  # older transformers use torch_dtype
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto" if device != "cpu" else torch.float32,
            **kwargs,
        )
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    def generate(prompt) -> str:
        if isinstance(prompt, list):
            messages = prompt
        else:
            messages = [{"role": "user", "content": str(prompt)}]
        if getattr(tokenizer, "chat_template", None):
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages) + "\nASSISTANT:"
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = output[0, inputs["input_ids"].shape[1] :]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    generate.supports_messages = True  # type: ignore[attr-defined]
    return generate


def timed(predictor: Callable[[Item], str | None], item: Item) -> Prediction:
    start = time.perf_counter()
    error = ""
    try:
        raw = predictor(item)
    except Exception as exc:  # a failed call is an invalid answer, not a crash
        raw = None
        error = f"{type(exc).__name__}: {exc}"
    elapsed = (time.perf_counter() - start) * 1000.0
    raw_text = getattr(predictor, "last_raw", None)
    if raw_text is None:
        raw_text = raw if isinstance(raw, str) else ""
    trace = getattr(predictor, "last_trace", None)
    if raw is None:
        return Prediction(answer=None, raw=str(raw_text) or error, latency_ms=elapsed, trace=trace)
    answer = parse_answer(str(raw), item.query.kind)
    return Prediction(answer=answer, raw=str(raw_text), latency_ms=elapsed, trace=trace)
