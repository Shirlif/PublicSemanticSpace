$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "== Phase 0: tests ==" -ForegroundColor Cyan
python -m pytest

Write-Host "== Phase 0: demos ==" -ForegroundColor Cyan
python -m pss.demos.semantic_space_demo
python -m pss.demos.token_vs_vector_reasoning
python -m pss.demos.laundry_task_demo

Write-Host "== Phase 0/1: binding benchmarks ==" -ForegroundColor Cyan
python bench/binding_stability.py
python bench/phase1_binding.py
