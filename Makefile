.PHONY: demo test bench bench-quick bench-phase1 data expert-data train-experts projection prototypes memory phase7 phase8 phase9 phase10 hard-occlusion rd2-mock pretrained-expert public-smoke-wt public-smoke-opengraph public-smoke-daimon tactile-wtmm projection-wtmm t1-instance t1-diagnostics public-space-bank public-space-demo category-pretrain-data category-pretrain-train category-pretrain-projection category-pretrain-prototypes category-pretrain encoder-priority encoder-priority-smoke typed-relations active-policy language-head bilevel runtime runtime-phase17 runtime-defaults runtime-encoder runtime-binding runtime-rules runtime-planning runtime-language runtime-audit suite-smoke suite clevr-image tao-frames tao-pss repro-pack repro-fetch reproduce-l2 all

demo:
	python -m pss.demos.semantic_space_demo
	python -m pss.demos.token_vs_vector_reasoning
	python -m pss.demos.laundry_task_demo

test:
	python -m pytest

bench:
	python bench/binding_stability.py

bench-quick:
	python bench/binding_stability.py --quick

bench-phase1:
	python bench/phase1_binding.py

data:
	python -m pss.sim.scripts.generate_dataset --out data/v1 --scale small --seed 0

expert-data:
	python -m pss.sim.scripts.generate_dataset --out data/expert_v1 --scale expert --seed 0

train-experts:
	python -m pss.encoders.scripts.train_experts --data data/expert_v1 --out checkpoints/expert_v1 --epochs 30 --size 64 --seed 0

projection:
	python -m pss.projection.scripts.train_projections --data data/expert_v1 --experts checkpoints/expert_v1 --out checkpoints/projection_v1 --epochs 60 --batch-size 64 --temp 0.07 --seed 0

prototypes:
	python -m pss.prototypes.scripts.build_prototypes --data data/expert_v1 --experts checkpoints/expert_v1 --projections checkpoints/projection_v1 --out checkpoints/prototypes_v1 --seed 0

memory:
	python -m pss.memory.scripts.run_tracking --data data/v1 --experts checkpoints/expert_v1 --projections checkpoints/projection_v1 --out checkpoints/memory_v1 --occlusion 4 --episodes 2 --seed 0

tracking-v2-data:
	python -m pss.sim.scripts.generate_dataset --out data/tracking_v2 --scale small --seed 0 --tactile-mode material

memory-v2: tracking-v2-data
	python -m pss.memory.scripts.run_tracking --data data/tracking_v2 --experts checkpoints/category_pretrain_experts --projections checkpoints/phase45_projection --out checkpoints/memory_v2_gt --split test --occlusion 100 --episodes 100 --seed 0

memory-v2-proto: tracking-v2-data
	python -m pss.memory.scripts.run_tracking --data data/tracking_v2 --experts checkpoints/category_pretrain_experts --projections checkpoints/phase45_projection --prototypes checkpoints/phase45_prototypes --out checkpoints/memory_v2_proto --split test --occlusion 100 --episodes 100 --seed 0

memory-decomposition:
	python -m pss.memory.scripts.run_decomposition_demo

phase7:
	python -m pss.reasoning.scripts.run_phase7 --data data/v1 --out checkpoints/phase7 --episodes 8 --epochs 100

phase7-v2-data:
	python -m pss.sim.scripts.generate_dataset --out data/phase7_v2 --scale expert --seed 0 --tactile-mode material

phase7-v2: phase7-v2-data
	python -m pss.reasoning.scripts.run_phase7 --data data/phase7_v2 --out checkpoints/phase7_v2 --episodes 8 --epochs 100

relations-v2-more:
	python -m pss.sim.scripts.generate_dataset --out data/phase7_v2_more --scale full --seed 2 --tactile-mode material
	python -m pss.reasoning.scripts.run_relation_eval --data data/phase7_v2_more --episodes 32 --out eval/results/relations_v2_more

phase7-v2-more:
	python -m pss.reasoning.scripts.run_phase7 --data data/phase7_v2_more --out checkpoints/phase7_v2_more --episodes 32 --epochs 300

closed-loop:
	python -m pss.reasoning.scripts.run_closed_loop --out eval/results/closed_loop

closed-loop-no-anomaly:
	python -m pss.reasoning.scripts.run_closed_loop --out eval/results/closed_loop_no_anomaly --no-anomaly

tracking-bench-v1:
	python -m pss.sim.scripts.generate_tracking_bench --out data/tracking_bench_v1 --train 200 --val 40 --test 60 --seed 0

pss-vs-transformer:
	python -m pss.eval.scripts.run_pss_vs_transformer --data data/tracking_bench_v1 --experts checkpoints/category_pretrain_experts --projections checkpoints/phase45_projection --out eval/results/pss_vs_transformer --seeds 5 --epochs 200

pss-learned-metric:
	python -m pss.eval.scripts.run_pss_learned_metric --data data/tracking_bench_v1 --experts checkpoints/category_pretrain_experts --projections checkpoints/phase45_projection --out eval/results/pss_vs_transformer --seeds 5 --epochs 200

p4-scaling:
	python -m pss.eval.scripts.run_scaling_benchmark --out eval/results/pss_vs_transformer

p2-openset:
	python -m pss.eval.scripts.run_p2_openset --data data/category_pretrain --experts checkpoints/category_pretrain_experts --projections checkpoints/phase45_projection --prototypes checkpoints/phase45_prototypes --out eval/results/pss_vs_transformer --seeds 5 --epochs 200

modality-conflict:
	python -m pss.eval.scripts.run_modality_conflict --data data/category_pretrain --experts checkpoints/category_pretrain_experts --projections checkpoints/phase45_projection --prototypes checkpoints/phase45_prototypes --out eval/results/pss_vs_transformer --seeds 5 --epochs 200

p3-incremental:
	python -m pss.eval.scripts.run_p3_incremental --data data/category_pretrain --experts checkpoints/category_pretrain_experts --projections checkpoints/phase45_projection --prototypes checkpoints/phase45_prototypes --out eval/results/pss_vs_transformer --seeds 5 --base-epochs 200 --finetune-epochs 50 --seed 0

phase8:
	python -m pss.heads.scripts.run_phase8 --data data/v1 --experts checkpoints/expert_v1 --projections checkpoints/projection_v1 --prototypes checkpoints/prototypes_v1 --out checkpoints/phase8 --trials 10 --seed 0

phase9:
	python -m pss.learning.scripts.run_phase9 --data data/expert_v1 --experts checkpoints/expert_v1 --projections checkpoints/projection_v1 --prototypes checkpoints/prototypes_v1 --out checkpoints/phase9

phase10:
	python -m pss.eval.scripts.run_phase10 --data data/expert_v1 --occlusion-data data/hard_occlusion --experts checkpoints/expert_v1 --projections checkpoints/projection_v1 --prototypes checkpoints/prototypes_v1 --out eval/results --seed 0

hard-occlusion:
	python -m pss.sim.scripts.generate_hard_occlusion --out data/hard_occlusion --sequences 6 --frames 12 --seed 0

occlusion-v3:
	python -m pss.sim.scripts.generate_occlusion_v3 --out data/occlusion_v3 --sequences 6 --frames 24 --seed 0

e3-occlusion-v3: occlusion-v3
	python -m pss.eval.scripts.run_e3_occlusion --data data/occlusion_v3 --split test --experts checkpoints/category_pretrain_experts --projections checkpoints/phase45_projection --out eval/results/occlusion_v3_gt
	python -m pss.eval.scripts.run_e3_occlusion --data data/occlusion_v3 --split test --experts checkpoints/category_pretrain_experts --projections checkpoints/phase45_projection --prototypes checkpoints/phase45_prototypes --out eval/results/occlusion_v3_proto

occlusion-v3-samecat:
	python -m pss.sim.scripts.generate_occlusion_v3 --out data/occlusion_v3_samecat --sequences 6 --frames 24 --category cup --variation-specs --seed 0

e3-occlusion-v3-samecat: occlusion-v3-samecat
	python -m pss.eval.scripts.run_e3_occlusion --data data/occlusion_v3_samecat --split test --experts checkpoints/category_pretrain_experts --projections checkpoints/phase45_projection --out eval/results/occlusion_v3_samecat_gt
	python -m pss.eval.scripts.run_e3_occlusion --data data/occlusion_v3_samecat --split test --experts checkpoints/category_pretrain_experts --projections checkpoints/phase45_projection --prototypes checkpoints/phase45_prototypes --out eval/results/occlusion_v3_samecat_proto

occlusion-v3-samecat-bottle:
	python -m pss.sim.scripts.generate_occlusion_v3 --out data/occlusion_v3_samecat_bottle --sequences 6 --frames 24 --category bottle --variation-specs --seed 0

e3-occlusion-v3-samecat-bottle: occlusion-v3-samecat-bottle
	python -m pss.eval.scripts.run_e3_occlusion --data data/occlusion_v3_samecat_bottle --split test --experts checkpoints/category_pretrain_experts --projections checkpoints/phase45_projection --out eval/results/occlusion_v3_samecat_bottle_gt
	python -m pss.eval.scripts.run_e3_occlusion --data data/occlusion_v3_samecat_bottle --split test --experts checkpoints/category_pretrain_experts --projections checkpoints/phase45_projection --prototypes checkpoints/phase45_prototypes --out eval/results/occlusion_v3_samecat_bottle_proto

occlusion-v3-near-identical:
	python -m pss.sim.scripts.generate_occlusion_v3 --out data/occlusion_v3_near_identical --sequences 6 --frames 24 --category cup --near-identical --seed 0

e3-occlusion-v3-near-identical: occlusion-v3-near-identical
	python -m pss.eval.scripts.run_e3_occlusion --data data/occlusion_v3_near_identical --split test --experts checkpoints/category_pretrain_experts --projections checkpoints/phase45_projection --out eval/results/occlusion_v3_near_identical_gt
	python -m pss.eval.scripts.run_e3_occlusion --data data/occlusion_v3_near_identical --split test --experts checkpoints/category_pretrain_experts --projections checkpoints/phase45_projection --prototypes checkpoints/phase45_prototypes --out eval/results/occlusion_v3_near_identical_proto

rd2-mock:
	python -m pss.rd2.scripts.make_mock --out data/rd2_mock --data data/expert_v1 --episodes 8 --seed 0
	python -m pss.rd2.scripts.validate --path data/rd2_mock

pretrained-expert:
	python -m pss.encoders.scripts.train_pretrained_expert --data data/expert_v1 --out checkpoints/expert_pretrained_v1 --backbone squeezenet1_1 --pretrained true --epochs 30 --size 64 --seed 0

public-smoke-wt:
	python -m pss.rd2.public.scripts.download_smoke --dataset wt_mm --root data/public/wt_mm_pair --limit 1027 --video data/public/wt_mm_pair/videos/observation.images.cam/chunk-000/file-005.mp4 --out eval/results/public_wt_mm_pair_smoke.json

public-smoke-opengraph:
	python -m pss.rd2.public.scripts.download_smoke --dataset opengraph_ego_tactile --root data/public/opengraph --limit 1500 --out eval/results/public_opengraph_smoke.json

public-smoke-daimon:
	python -m pss.rd2.public.scripts.download_smoke --dataset daimon_infinity --root data/public/daimon --limit 500 --out eval/results/public_daimon_smoke.json

tactile-wtmm:
	python -m pss.encoders.scripts.train_tactile_wtmm --pair-root data/public/wt_mm_pair --extra-root data/public/wt_mm --video data/public/wt_mm_pair/videos/observation.images.cam/chunk-000/file-005.mp4 --out checkpoints/tactile_wtmm --epochs 30 --batch-size 128 --out-dim 64 --seed 0

projection-wtmm:
	python -m pss.encoders.scripts.train_projection_wtmm --data data/expert_v1 --experts checkpoints/expert_v1 --tactile checkpoints/tactile_wtmm/tactile_wtmm.pt --out checkpoints/projection_wtmm --epochs 60 --batch-size 64 --temp 0.07 --seed 0

t1-instance:
	python -m pss.projection.scripts.t1_instance_experiment --data data/expert_v1 --epochs 60 --seed 0 --out eval/results/t1_instance.json

t1-diagnostics:
	python -m pss.rd2.scripts.t1_diagnostics --source expert_v1 --data data/expert_v1 --out eval/results/t1_diagnostics_expert_v1.json

category-pretrain-data:
	python -m pss.sim.scripts.generate_category_pretrain --out data/category_pretrain --variations 24 --views 4 --seed 0

category-pretrain-train:
	python -m pss.encoders.scripts.train_experts --data data/category_pretrain --out checkpoints/category_pretrain_experts --epochs 30 --size 64 --seed 0

category-pretrain-projection:
	python -m pss.projection.scripts.train_projections --data data/category_pretrain --experts checkpoints/category_pretrain_experts --out checkpoints/category_pretrain_projection --epochs 60 --batch-size 64 --temp 0.07 --seed 0

category-pretrain-prototypes:
	python -m pss.prototypes.scripts.build_prototypes --data data/category_pretrain --experts checkpoints/category_pretrain_experts --projections checkpoints/category_pretrain_projection --out checkpoints/category_pretrain_prototypes --seed 0

category-pretrain: category-pretrain-data category-pretrain-train category-pretrain-projection category-pretrain-prototypes

# First-priority encoder swap (docs/architecture/PSS-认知层架构定位.md §7):
# frozen DINOv2/CLIP, conv tactile, calibrated projections; runs open-set,
# re-ID and P5b. CLIP is skipped unless its weights are cached or
# --allow-download is passed.
encoder-priority:
	python -m pss.eval.scripts.run_encoder_priority --stage all --device cuda

encoder-priority-smoke:
	python -m pss.eval.scripts.run_encoder_priority --stage all --only base --device cpu --seeds 1 --epochs 20 --p5b-seeds 1 --p5b-epochs 20

# Second priority: learnable typed relations. Trains spatial/semantic/tactile
# heads, writes relation logits into the scene graph, compares hand vs learned
# vs hybrid edges, and runs P5b with a learned spatial relation predicate.
typed-relations:
	python -m pss.eval.scripts.run_typed_relations --stage all --device cuda

# Third priority: learned active-set and write policies. Trains a listwise
# candidate selector with calibrated fallback, a write/review policy, and a
# sparse long-history fact active set.
active-policy:
	python -m pss.eval.scripts.run_active_policy --stage all --device cuda

# Fourth priority: language output as an LLM adapter over structured PSS state,
# tools and audit. No local LLM is required for the harness.
language-head:
	python -m pss.eval.scripts.run_language_head --device cpu

# Fifth priority: end-to-end / bilevel training with frozen encoders.
# Trains projections + binder + relations + matcher jointly; reports
# recognition, re-ID, relation F1 and consistency/audit metrics.
bilevel:
	python -m pss.eval.scripts.run_bilevel --device cuda --epochs 60 --seeds 1

# Phase 16: cognitive runtime integration. One typed state, entity/relation/
# uncertainty/planning/output services, and a cross-layer closed loop.
runtime:
	python -m pss.eval.scripts.run_runtime

# Phase 17: learned relations, write gate, versioned operators and long
# occlusion through the runtime.
runtime-phase17:
	python -m pss.eval.scripts.run_runtime_phase17 --device cpu --sequences 12

# Phase 18: learned relations, active-set and write policies as runtime
# defaults, with per-component ablations against the hand-written fallbacks.
runtime-defaults:
	python -m pss.eval.scripts.run_runtime_defaults

# Gap closure row 1-2: DINOv2 encoder + calibrated projection as runtime
# defaults, compared with the SmallCNN fallback.
runtime-encoder:
	python -m pss.eval.scripts.run_runtime_encoder --device cuda --sequences 5

# Gap closure row 3: object attention as an EntityLayer/runtime associator.
runtime-binding:
	python -m pss.eval.scripts.run_runtime_binding --device cuda --sequences 8

# Gap closure row 4: learnable/versioned geometric relation rules.
runtime-rules:
	python -m pss.eval.scripts.run_runtime_rules

# Gap closure row 5: learned cost-to-go heuristic with A* search.
runtime-planning:
	python -m pss.eval.scripts.run_runtime_planning --device cpu --train 4000 --test 400

# Gap closure row 6: a real local decoder (Qwen2.5-1.5B if cached) in the
# runtime language path, with the tool-loop convergence guard.
runtime-language:
	python -m pss.eval.scripts.run_runtime_language --device cuda

# Gap closure row 7: read_only/service/operator/admin runtime permissions.
runtime-audit:
	python -m pss.eval.scripts.run_runtime_audit

# Phase A evaluation-suite skeleton: schema + bfcl/openood/zsre adapters +
# metrics + runner. --smoke uses built-in episodes; real runs need official data.
suite-smoke:
	python -m pss.eval.suite.runner --smoke --out eval/results/suite

suite:
	python -m pss.eval.suite.runner --root data/public --out eval/results/suite

# CLEVR-CoGenT image relation + property composition (DINOv2 crops).
clevr-image:
	python -m pss.eval.scripts.run_clevr_image --device cuda --train 8000 --val 3000

# TAO official frames: extract Charades frames and compare IoU vs PSS tracker.
tao-frames:
	python -m pss.eval.scripts.prepare_tao_frames --videos 5
tao-pss:
	python -m pss.eval.scripts.run_tao_pss --device cuda --videos 5

phase45-projection:
	python -m pss.projection.scripts.train_projections --data data/category_pretrain --experts checkpoints/category_pretrain_experts --out checkpoints/phase45_projection --epochs 120 --batch-size 256 --temp 0.07 --seed 0 --material-weight 1.0

phase45-prototypes:
	python -m pss.prototypes.scripts.build_prototypes --data data/category_pretrain --experts checkpoints/category_pretrain_experts --projections checkpoints/phase45_projection --out checkpoints/phase45_prototypes --seed 0

phase45: phase45-projection phase45-prototypes
	python -m pss.eval.scripts.run_phase45 --data data/category_pretrain --experts checkpoints/category_pretrain_experts --projections checkpoints/phase45_projection --prototypes checkpoints/phase45_prototypes --out eval/results/phase45

anchor-retrain-vision:
	python -m pss.projection.scripts.anchor_retrain --data data/category_pretrain --experts checkpoints/category_pretrain_experts --projections checkpoints/phase45_projection --prototypes checkpoints/phase45_prototypes --modality vision --out checkpoints/anchor_retrain_vision --epochs 60

anchor-retrain-tactile:
	python -m pss.projection.scripts.anchor_retrain --data data/category_pretrain --experts checkpoints/category_pretrain_experts --projections checkpoints/phase45_projection --prototypes checkpoints/phase45_prototypes --modality tactile --out checkpoints/anchor_retrain_tactile --epochs 60

phase5-package:
	python -m pss.prototypes.scripts.package_ops --action run-all --category sheet --root checkpoints/prototype_packages --out eval/results/phase5_package

instance-view:
	python -m pss.projection.scripts.instance_view_experiment --data data/category_pretrain --experts checkpoints/category_pretrain_experts --projections checkpoints/phase45_projection --out checkpoints/instance_view --epochs 60

instance-data:
	python -m pss.sim.scripts.generate_category_pretrain --out data/category_pretrain_instance --variations 24 --views 4 --seed 0

instance-experts:
	python -m pss.encoders.scripts.train_experts --data data/category_pretrain_instance --out checkpoints/category_pretrain_instance_experts --epochs 30 --size 64 --seed 0

instance-projection:
	python -m pss.projection.scripts.train_projections --data data/category_pretrain_instance --experts checkpoints/category_pretrain_instance_experts --out checkpoints/category_pretrain_instance_projection --epochs 120 --batch-size 256 --temp 0.07 --seed 0 --material-weight 1.0

instance-view-v2: instance-data instance-experts instance-projection
	python -m pss.projection.scripts.instance_view_experiment --data data/category_pretrain_instance --experts checkpoints/category_pretrain_instance_experts --projections checkpoints/category_pretrain_instance_projection --out checkpoints/instance_view_v2 --epochs 60

instance8-data:
	python -m pss.sim.scripts.generate_category_pretrain --out data/category_pretrain_instance8 --variations 24 --views 8 --seed 0

instance8-experts:
	python -m pss.encoders.scripts.train_experts --data data/category_pretrain_instance8 --out checkpoints/category_pretrain_instance8_experts --epochs 30 --size 64 --seed 0

instance8-projection:
	python -m pss.projection.scripts.train_projections --data data/category_pretrain_instance8 --experts checkpoints/category_pretrain_instance8_experts --out checkpoints/category_pretrain_instance8_projection --epochs 120 --batch-size 256 --temp 0.07 --seed 0 --material-weight 1.0

instance-view-v8: instance8-data instance8-experts instance8-projection
	python -m pss.projection.scripts.instance_view_experiment --data data/category_pretrain_instance8 --experts checkpoints/category_pretrain_instance8_experts --projections checkpoints/category_pretrain_instance8_projection --out checkpoints/instance_view_v8 --epochs 80 --train-views 0 1 2 3 --test-views 4 5 6 7

objectfolder-query:
	python -c "import numpy as np; q=np.load('_external/ObjectFolder/demo/vision_demo.npy'); np.save('_research/vision_query_2.npy', q[:2])"

objectfolder-render:
	for n in 50 60 70 80 90 100; do \
		python -m pss.rd2.scripts.objectfolder_r1 --root _external/ObjectFolder/ObjectFolder1-100 --renderer _external/ObjectFolder --out eval/results/objectfolder --limit $$n --touches 4 --vision-query _research/vision_query_2.npy --render-only; \
	done

objectfolder-r1: objectfolder-query objectfolder-render
	python -m pss.rd2.scripts.objectfolder_r1 --root _external/ObjectFolder/ObjectFolder1-100 --renderer _external/ObjectFolder --out eval/results/objectfolder --limit 100 --touches 4 --vision-query _research/vision_query_2.npy --no-render

objectfolder-tactile:
	python -m pss.rd2.scripts.train_objectfolder_tactile --renders eval/results/objectfolder/renders --out eval/results/objectfolder_tactile --checkpoint-dir checkpoints/objectfolder_tactile --epochs 40 --folds 5

objectfolder-symmetric:
	python -m pss.rd2.scripts.train_objectfolder_symmetric --renders eval/results/objectfolder/renders --out eval/results/objectfolder_symmetric --checkpoint-dir checkpoints/objectfolder_symmetric --epochs 40 --folds 5

objectfolder-render-t12: objectfolder-query
	for n in 20 40 60 80 100; do \
		python -m pss.rd2.scripts.objectfolder_r1 --root _external/ObjectFolder/ObjectFolder1-100 --renderer _external/ObjectFolder --out eval/results/objectfolder --limit $$n --touches 12 --vision-query _research/vision_query_2.npy --render-only; \
	done

objectfolder-tactile-t12: objectfolder-render-t12
	python -m pss.rd2.scripts.train_objectfolder_tactile --renders eval/results/objectfolder/renders --out eval/results/objectfolder_tactile_t12 --checkpoint-dir checkpoints/objectfolder_tactile_t12 --epochs 40 --folds 5

objectfolder-symmetric-t12: objectfolder-tactile-t12
	python -m pss.rd2.scripts.train_objectfolder_symmetric --renders eval/results/objectfolder/renders --out eval/results/objectfolder_symmetric_t12 --checkpoint-dir checkpoints/objectfolder_symmetric_t12 --tactile-baseline eval/results/objectfolder_tactile_t12/tactile_encoder_metrics.json --epochs 40 --folds 5

expert-zoo:
	python -m pss.encoders.scripts.bench_expert_zoo --data data/category_pretrain --experts contracts/experts.yaml --out eval/results/expert_zoo

physiclear-cache:
	python -m pss.encoders.scripts.train_physiclear --prepare-only --videos _external/octopi/dataset --repo-root _external/octopi --data data/physiclear

physiclear-train:
	python -m pss.encoders.scripts.train_physiclear --videos _external/octopi/dataset --repo-root _external/octopi --data data/physiclear --out checkpoints/tactile_physiclear --epochs 30

tactile-r2: physiclear-cache physiclear-train

public-space-bank:
	python -m pss.demos.public_semantic_space_demo --build-bank

public-space-demo:
	python -m pss.demos.public_semantic_space_demo --image data/demo_input.png --tactile data/demo_tactile.npy

spatial-s1:
	python -m pss.eval.spatial.scripts.run_s1 --out eval/results/spatial_s1 --n-per-family 100 --seed 0

spatial-s1-smoke:
	python -m pss.eval.spatial.scripts.run_s1 --out eval/results/spatial_s1_smoke --n-per-family 10 --seed 0

spatial-s1-tools-smoke:
	python -m pss.eval.spatial.scripts.run_s1 --out eval/results/spatial_s1_tools_smoke --n-per-family 3 --seed 0 --split test_id:id --llm-model Qwen/Qwen2.5-1.5B-Instruct --llm-device cuda --llm-max-new-tokens 128 --llm-tools mid --llm-max-steps 4

repro-pack:
	python scripts/pack_repro_bundles.py --root bundles/dev --out dist/repro --version 0.1.1

repro-fetch:
	python scripts/fetch_repro_bundles.py --out bundles/repro

reproduce-l2:
	python scripts/reproduce_l2.py --root bundles/repro --out reports/repro

all: test demo bench bench-phase1
