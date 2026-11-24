"""T-text spatial perception suite (S1).

See ``eval/spatial/preregistration.md``. The package is numpy-only.
"""

from .baselines import (
    ExactSolver,
    LLMPredictor,
    RandomBaseline,
    ToolLoopPredictor,
    WorldAxisOnlySolver,
    solve_exact,
)
from .generator import ALL_FAMILIES, generate_suite, make_scene
from .runner import evaluate_predictor, write_jsonl, write_report
from .schema import Camera, Item, ObjectSpec, Query, Scene, parse_answer, render_prompt, scene_to_json
from .tools import TOOL_PROFILES, execute_tool, parse_tool_call, tool_manifest

__all__ = [
    "ALL_FAMILIES",
    "Camera",
    "ExactSolver",
    "Item",
    "LLMPredictor",
    "ObjectSpec",
    "Query",
    "RandomBaseline",
    "Scene",
    "TOOL_PROFILES",
    "ToolLoopPredictor",
    "WorldAxisOnlySolver",
    "evaluate_predictor",
    "execute_tool",
    "generate_suite",
    "make_scene",
    "parse_answer",
    "parse_tool_call",
    "render_prompt",
    "scene_to_json",
    "solve_exact",
    "tool_manifest",
    "write_jsonl",
    "write_report",
]
