#!/usr/bin/env bash
set -euo pipefail
export PYTHONIOENCODING=utf-8

echo "== Phase 0: tests =="
python -m pytest

echo "== Phase 0: demos =="
python -m pss.demos.semantic_space_demo
python -m pss.demos.token_vs_vector_reasoning
python -m pss.demos.laundry_task_demo

echo "== Phase 0/1: binding benchmarks =="
python bench/binding_stability.py
python bench/phase1_binding.py
