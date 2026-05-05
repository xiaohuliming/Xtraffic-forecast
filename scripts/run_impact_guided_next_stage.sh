#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
COMMON_CACHE="${ROOT_DIR}/outputs/full_candidate_stgnn_heatmap_model/first_pass/full_candidate_samples.h5"
NEXT_ROOT="${ROOT_DIR}/outputs/impact_guided_next_stage"
LEARNED_DUAL_CACHE="${NEXT_ROOT}/full_candidate_stgnn_learned_normal_dual/full_candidate_samples.h5"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_impact_guided_next_stage.sh normal-smoke
  bash scripts/run_impact_guided_next_stage.sh normal
  bash scripts/run_impact_guided_next_stage.sh learned-normal-smoke
  bash scripts/run_impact_guided_next_stage.sh learned-normal-delta-smoke
  bash scripts/run_impact_guided_next_stage.sh learned-normal-dual-smoke
  bash scripts/run_impact_guided_next_stage.sh learned-normal-uncertainty-smoke
  bash scripts/run_impact_guided_next_stage.sh learned-normal-decay-smoke
  bash scripts/run_impact_guided_next_stage.sh learned-normal-fullregion-smoke
  bash scripts/run_impact_guided_next_stage.sh smoke
  bash scripts/run_impact_guided_next_stage.sh full
  bash scripts/run_impact_guided_next_stage.sh learned-normal-full
  bash scripts/run_impact_guided_next_stage.sh learned-normal-delta-full
  bash scripts/run_impact_guided_next_stage.sh learned-normal-dual-full
  bash scripts/run_impact_guided_next_stage.sh learned-normal-uncertainty-full
  bash scripts/run_impact_guided_next_stage.sh learned-normal-decay-full
  bash scripts/run_impact_guided_next_stage.sh learned-normal-uncertainty-seeds
  bash scripts/run_impact_guided_next_stage.sh learned-normal-decay-seeds
  bash scripts/run_impact_guided_next_stage.sh learned-normal-decay-groups
  bash scripts/run_impact_guided_next_stage.sh learned-normal-fullregion-diagnostics
  bash scripts/run_impact_guided_next_stage.sh diagnostics
  bash scripts/run_impact_guided_next_stage.sh ablations

Modes:
  normal-smoke  Fast Alameda-only learned normal STGNN run.
  normal        Train learned normal STGNN branches for all three regions.
  learned-normal-smoke  Rebuild a tiny incident cache with the learned normal branch and train one epoch.
  learned-normal-delta-smoke  Same as learned-normal-smoke, but feed normal_delta to the residual STGNN.
  learned-normal-dual-smoke  Same as learned-normal-delta-smoke, plus dual historical residual input.
  learned-normal-uncertainty-smoke  Reuse dual smoke cache and append abs(normal_delta) disagreement features.
  learned-normal-decay-smoke  Reuse dual smoke cache and add the temporal decay head.
  learned-normal-fullregion-smoke  Tiny Alameda cache rebuild using full-region normal inference.
  smoke      Fast Alameda-only run that rebuilds a tiny cache and checks the pipeline.
  full       Reuse the existing full-candidate HDF5 cache and run the current main model.
  learned-normal-full  Rebuild incident cache with the learned normal branch, then run the main model.
  learned-normal-delta-full  Rebuild learned-normal cache and train residual STGNN with normal_delta input.
  learned-normal-dual-full  Rebuild learned-normal cache and train with normal_delta + dual history input.
  learned-normal-uncertainty-full  Reuse dual cache and train with normal_delta + abs(normal_delta) + dual history.
  learned-normal-decay-full  Reuse dual cache and train with the temporal decay head.
  learned-normal-uncertainty-seeds  Reuse dual cache and run additional seeds for robustness.
  learned-normal-decay-seeds  Reuse dual cache and run additional temporal-decay seeds.
  learned-normal-decay-groups  Compare no-decay and decay models by severity, recovery, and horizon.
  learned-normal-fullregion-diagnostics  Compare local candidate-subgraph and full-region normal inference.
  diagnostics  Compare old statistical-normal and learned-normal residual caches.
  ablations  Reuse the existing cache and run the most important graph/auxiliary ablations.
EOF
}

run_full_candidate() {
  local output_dir="$1"
  shift
  "${PYTHON_BIN}" scripts/train_full_candidate_stgnn_heatmap_model.py \
    --event-root outputs/impact_labels_aggregated/region_area_sensor_window \
    --raw-label-dir outputs/impact_labels \
    --cache-path "${COMMON_CACHE}" \
    --output-dir "${output_dir}" \
    --regions Alameda ContraCosta Orange \
    --input-steps 12 \
    --horizon-steps 12 \
    --max-candidate-nodes 36 \
    --candidate-pm-radius 5.0 \
    --anchor-pm-radius 2.0 \
    --epochs 5 \
    --batch-size 192 \
    --hidden-dim 96 \
    --dropout 0.10 \
    --lr 0.001 \
    --weight-decay 0.0001 \
    --event-aux-weight 0.05 \
    --node-aux-weight 0.03 \
    --seed 7 \
    --max-train-samples 20000 \
    --device auto \
    "$@"
}

run_learned_normal_uncertainty() {
  local output_dir="$1"
  local seed="$2"
  "${PYTHON_BIN}" scripts/train_full_candidate_stgnn_heatmap_model.py \
    --event-root outputs/impact_labels_aggregated/region_area_sensor_window \
    --raw-label-dir outputs/impact_labels \
    --output-dir "${output_dir}" \
    --cache-path "${LEARNED_DUAL_CACHE}" \
    --normal-model-dir "${NEXT_ROOT}/normal_stgnn_forecaster" \
    --normal-infer-batch-size 256 \
    --regions Alameda ContraCosta Orange \
    --input-steps 12 \
    --horizon-steps 12 \
    --max-candidate-nodes 36 \
    --candidate-pm-radius 5.0 \
    --anchor-pm-radius 2.0 \
    --epochs 5 \
    --batch-size 192 \
    --hidden-dim 96 \
    --dropout 0.10 \
    --lr 0.001 \
    --weight-decay 0.0001 \
    --graph-layers 2 \
    --graph-mode undirected \
    --graph-sigma 3.0 \
    --heatmap-aux-weight 0.0 \
    --event-aux-weight 0.05 \
    --node-aux-weight 0.03 \
    --use-normal-delta \
    --use-normal-delta-abs \
    --use-dual-hist-residual \
    --seed "${seed}" \
    --max-train-samples 20000 \
    --device auto
}

run_learned_normal_decay() {
  local output_dir="$1"
  local seed="$2"
  "${PYTHON_BIN}" scripts/train_full_candidate_stgnn_heatmap_model.py \
    --event-root outputs/impact_labels_aggregated/region_area_sensor_window \
    --raw-label-dir outputs/impact_labels \
    --output-dir "${output_dir}" \
    --cache-path "${LEARNED_DUAL_CACHE}" \
    --normal-model-dir "${NEXT_ROOT}/normal_stgnn_forecaster" \
    --normal-infer-batch-size 256 \
    --regions Alameda ContraCosta Orange \
    --input-steps 12 \
    --horizon-steps 12 \
    --max-candidate-nodes 36 \
    --candidate-pm-radius 5.0 \
    --anchor-pm-radius 2.0 \
    --epochs 5 \
    --batch-size 192 \
    --hidden-dim 96 \
    --dropout 0.10 \
    --lr 0.001 \
    --weight-decay 0.0001 \
    --graph-layers 2 \
    --graph-mode undirected \
    --graph-sigma 3.0 \
    --heatmap-aux-weight 0.0 \
    --event-aux-weight 0.05 \
    --node-aux-weight 0.03 \
    --use-normal-delta \
    --use-normal-delta-abs \
    --use-dual-hist-residual \
    --use-temporal-decay-head \
    --seed "${seed}" \
    --max-train-samples 20000 \
    --device auto
}

mode="${1:-}"
case "${mode}" in
  normal-smoke)
    "${PYTHON_BIN}" scripts/train_normal_stgnn_forecaster.py \
      --output-dir "${NEXT_ROOT}/normal_stgnn_smoke" \
      --regions Alameda \
      --input-steps 12 \
      --horizon-steps 12 \
      --sample-stride 144 \
      --epochs 1 \
      --batch-size 2 \
      --hidden-dim 32 \
      --graph-layers 1 \
      --graph-topk 4 \
      --graph-sigma 1.5 \
      --max-train-samples 64 \
      --max-eval-samples 32 \
      --device auto
    ;;
  normal)
    "${PYTHON_BIN}" scripts/train_normal_stgnn_forecaster.py \
      --output-dir "${NEXT_ROOT}/normal_stgnn_forecaster" \
      --regions Alameda ContraCosta Orange \
      --input-steps 12 \
      --horizon-steps 12 \
      --sample-stride 6 \
      --epochs 5 \
      --batch-size 8 \
      --hidden-dim 64 \
      --graph-layers 2 \
      --graph-topk 8 \
      --graph-sigma 1.5 \
      --max-train-samples 12000 \
      --max-eval-samples 4096 \
      --device auto
    ;;
  smoke)
    "${PYTHON_BIN}" scripts/train_full_candidate_stgnn_heatmap_model.py \
      --event-root outputs/impact_labels_aggregated/region_area_sensor_window \
      --raw-label-dir outputs/impact_labels \
      --output-dir "${NEXT_ROOT}/smoke_alameda" \
      --cache-path "${NEXT_ROOT}/smoke_alameda/full_candidate_samples.h5" \
      --rebuild-cache \
      --regions Alameda \
      --input-steps 12 \
      --horizon-steps 12 \
      --max-candidate-nodes 16 \
      --candidate-pm-radius 5.0 \
      --anchor-pm-radius 2.0 \
      --epochs 1 \
      --batch-size 128 \
      --hidden-dim 48 \
      --graph-layers 1 \
      --graph-mode undirected \
      --graph-sigma 1.0 \
      --heatmap-aux-weight 0.0 \
      --event-aux-weight 0.02 \
      --node-aux-weight 0.02 \
      --seed 7 \
      --max-train-samples 512 \
      --device auto
    ;;
  learned-normal-smoke)
    "${PYTHON_BIN}" scripts/train_full_candidate_stgnn_heatmap_model.py \
      --event-root outputs/impact_labels_aggregated/region_area_sensor_window \
      --raw-label-dir outputs/impact_labels \
      --output-dir "${NEXT_ROOT}/learned_normal_smoke_alameda" \
      --cache-path "${NEXT_ROOT}/learned_normal_smoke_alameda/full_candidate_samples.h5" \
      --rebuild-cache \
      --normal-model-dir "${NEXT_ROOT}/normal_stgnn_forecaster" \
      --normal-infer-batch-size 256 \
      --regions Alameda \
      --input-steps 12 \
      --horizon-steps 12 \
      --max-candidate-nodes 16 \
      --max-cache-samples-per-split 192 \
      --candidate-pm-radius 5.0 \
      --anchor-pm-radius 2.0 \
      --epochs 1 \
      --batch-size 128 \
      --hidden-dim 48 \
      --graph-layers 1 \
      --graph-mode undirected \
      --graph-sigma 1.0 \
      --heatmap-aux-weight 0.0 \
      --event-aux-weight 0.02 \
      --node-aux-weight 0.02 \
      --seed 7 \
      --max-train-samples 512 \
      --device auto
    ;;
  learned-normal-delta-smoke)
    "${PYTHON_BIN}" scripts/train_full_candidate_stgnn_heatmap_model.py \
      --event-root outputs/impact_labels_aggregated/region_area_sensor_window \
      --raw-label-dir outputs/impact_labels \
      --output-dir "${NEXT_ROOT}/learned_normal_delta_smoke_alameda" \
      --cache-path "${NEXT_ROOT}/learned_normal_delta_smoke_alameda/full_candidate_samples.h5" \
      --rebuild-cache \
      --normal-model-dir "${NEXT_ROOT}/normal_stgnn_forecaster" \
      --normal-infer-batch-size 256 \
      --regions Alameda \
      --input-steps 12 \
      --horizon-steps 12 \
      --max-candidate-nodes 16 \
      --max-cache-samples-per-split 192 \
      --candidate-pm-radius 5.0 \
      --anchor-pm-radius 2.0 \
      --epochs 1 \
      --batch-size 128 \
      --hidden-dim 48 \
      --graph-layers 1 \
      --graph-mode undirected \
      --graph-sigma 1.0 \
      --heatmap-aux-weight 0.0 \
      --event-aux-weight 0.02 \
      --node-aux-weight 0.02 \
      --use-normal-delta \
      --seed 7 \
      --max-train-samples 512 \
      --device auto
    ;;
  learned-normal-dual-smoke)
    "${PYTHON_BIN}" scripts/train_full_candidate_stgnn_heatmap_model.py \
      --event-root outputs/impact_labels_aggregated/region_area_sensor_window \
      --raw-label-dir outputs/impact_labels \
      --output-dir "${NEXT_ROOT}/learned_normal_dual_smoke_alameda" \
      --cache-path "${NEXT_ROOT}/learned_normal_dual_smoke_alameda/full_candidate_samples.h5" \
      --rebuild-cache \
      --normal-model-dir "${NEXT_ROOT}/normal_stgnn_forecaster" \
      --normal-infer-batch-size 256 \
      --regions Alameda \
      --input-steps 12 \
      --horizon-steps 12 \
      --max-candidate-nodes 16 \
      --max-cache-samples-per-split 192 \
      --candidate-pm-radius 5.0 \
      --anchor-pm-radius 2.0 \
      --epochs 1 \
      --batch-size 128 \
      --hidden-dim 48 \
      --graph-layers 1 \
      --graph-mode undirected \
      --graph-sigma 1.0 \
      --heatmap-aux-weight 0.0 \
      --event-aux-weight 0.02 \
      --node-aux-weight 0.02 \
      --use-normal-delta \
      --use-dual-hist-residual \
      --seed 7 \
      --max-train-samples 512 \
      --device auto
    ;;
  learned-normal-uncertainty-smoke)
    "${PYTHON_BIN}" scripts/train_full_candidate_stgnn_heatmap_model.py \
      --event-root outputs/impact_labels_aggregated/region_area_sensor_window \
      --raw-label-dir outputs/impact_labels \
      --output-dir "${NEXT_ROOT}/learned_normal_uncertainty_smoke_alameda" \
      --cache-path "${NEXT_ROOT}/learned_normal_dual_smoke_alameda/full_candidate_samples.h5" \
      --normal-model-dir "${NEXT_ROOT}/normal_stgnn_forecaster" \
      --normal-infer-batch-size 256 \
      --regions Alameda \
      --input-steps 12 \
      --horizon-steps 12 \
      --max-candidate-nodes 16 \
      --max-cache-samples-per-split 192 \
      --candidate-pm-radius 5.0 \
      --anchor-pm-radius 2.0 \
      --epochs 1 \
      --batch-size 128 \
      --hidden-dim 48 \
      --graph-layers 1 \
      --graph-mode undirected \
      --graph-sigma 1.0 \
      --heatmap-aux-weight 0.0 \
      --event-aux-weight 0.02 \
      --node-aux-weight 0.02 \
      --use-normal-delta \
      --use-normal-delta-abs \
      --use-dual-hist-residual \
      --seed 7 \
      --max-train-samples 512 \
      --device auto
    ;;
  learned-normal-decay-smoke)
    "${PYTHON_BIN}" scripts/train_full_candidate_stgnn_heatmap_model.py \
      --event-root outputs/impact_labels_aggregated/region_area_sensor_window \
      --raw-label-dir outputs/impact_labels \
      --output-dir "${NEXT_ROOT}/learned_normal_decay_smoke_alameda" \
      --cache-path "${NEXT_ROOT}/learned_normal_dual_smoke_alameda/full_candidate_samples.h5" \
      --normal-model-dir "${NEXT_ROOT}/normal_stgnn_forecaster" \
      --normal-infer-batch-size 256 \
      --regions Alameda \
      --input-steps 12 \
      --horizon-steps 12 \
      --max-candidate-nodes 16 \
      --max-cache-samples-per-split 192 \
      --candidate-pm-radius 5.0 \
      --anchor-pm-radius 2.0 \
      --epochs 1 \
      --batch-size 128 \
      --hidden-dim 48 \
      --graph-layers 1 \
      --graph-mode undirected \
      --graph-sigma 1.0 \
      --heatmap-aux-weight 0.0 \
      --event-aux-weight 0.02 \
      --node-aux-weight 0.02 \
      --use-normal-delta \
      --use-normal-delta-abs \
      --use-dual-hist-residual \
      --use-temporal-decay-head \
      --seed 7 \
      --max-train-samples 512 \
      --device auto
    ;;
  learned-normal-fullregion-smoke)
    "${PYTHON_BIN}" scripts/train_full_candidate_stgnn_heatmap_model.py \
      --event-root outputs/impact_labels_aggregated/region_area_sensor_window \
      --raw-label-dir outputs/impact_labels \
      --output-dir "${NEXT_ROOT}/learned_normal_fullregion_smoke_alameda" \
      --cache-path "${NEXT_ROOT}/learned_normal_fullregion_smoke_alameda/full_candidate_samples.h5" \
      --rebuild-cache \
      --normal-model-dir "${NEXT_ROOT}/normal_stgnn_forecaster" \
      --normal-inference-scope full \
      --normal-infer-batch-size 16 \
      --regions Alameda \
      --input-steps 12 \
      --horizon-steps 12 \
      --max-candidate-nodes 16 \
      --max-cache-samples-per-split 32 \
      --candidate-pm-radius 5.0 \
      --anchor-pm-radius 2.0 \
      --epochs 1 \
      --batch-size 64 \
      --hidden-dim 48 \
      --graph-layers 1 \
      --graph-mode undirected \
      --graph-sigma 1.0 \
      --heatmap-aux-weight 0.0 \
      --event-aux-weight 0.02 \
      --node-aux-weight 0.02 \
      --use-normal-delta \
      --use-normal-delta-abs \
      --use-dual-hist-residual \
      --seed 7 \
      --max-train-samples 128 \
      --device auto
    ;;
  full)
    run_full_candidate "${NEXT_ROOT}/full_candidate_stgnn_main" \
      --graph-layers 2 \
      --graph-mode undirected \
      --graph-sigma 3.0 \
      --heatmap-aux-weight 0.0
    ;;
  learned-normal-full)
    "${PYTHON_BIN}" scripts/train_full_candidate_stgnn_heatmap_model.py \
      --event-root outputs/impact_labels_aggregated/region_area_sensor_window \
      --raw-label-dir outputs/impact_labels \
      --output-dir "${NEXT_ROOT}/full_candidate_stgnn_learned_normal" \
      --cache-path "${NEXT_ROOT}/full_candidate_stgnn_learned_normal/full_candidate_samples.h5" \
      --rebuild-cache \
      --normal-model-dir "${NEXT_ROOT}/normal_stgnn_forecaster" \
      --normal-infer-batch-size 256 \
      --regions Alameda ContraCosta Orange \
      --input-steps 12 \
      --horizon-steps 12 \
      --max-candidate-nodes 36 \
      --candidate-pm-radius 5.0 \
      --anchor-pm-radius 2.0 \
      --epochs 5 \
      --batch-size 192 \
      --hidden-dim 96 \
      --dropout 0.10 \
      --lr 0.001 \
      --weight-decay 0.0001 \
      --graph-layers 2 \
      --graph-mode undirected \
      --graph-sigma 3.0 \
      --heatmap-aux-weight 0.0 \
      --event-aux-weight 0.05 \
      --node-aux-weight 0.03 \
      --seed 7 \
      --max-train-samples 20000 \
      --device auto
    ;;
  learned-normal-delta-full)
    "${PYTHON_BIN}" scripts/train_full_candidate_stgnn_heatmap_model.py \
      --event-root outputs/impact_labels_aggregated/region_area_sensor_window \
      --raw-label-dir outputs/impact_labels \
      --output-dir "${NEXT_ROOT}/full_candidate_stgnn_learned_normal_delta" \
      --cache-path "${NEXT_ROOT}/full_candidate_stgnn_learned_normal_delta/full_candidate_samples.h5" \
      --rebuild-cache \
      --normal-model-dir "${NEXT_ROOT}/normal_stgnn_forecaster" \
      --normal-infer-batch-size 256 \
      --regions Alameda ContraCosta Orange \
      --input-steps 12 \
      --horizon-steps 12 \
      --max-candidate-nodes 36 \
      --candidate-pm-radius 5.0 \
      --anchor-pm-radius 2.0 \
      --epochs 5 \
      --batch-size 192 \
      --hidden-dim 96 \
      --dropout 0.10 \
      --lr 0.001 \
      --weight-decay 0.0001 \
      --graph-layers 2 \
      --graph-mode undirected \
      --graph-sigma 3.0 \
      --heatmap-aux-weight 0.0 \
      --event-aux-weight 0.05 \
      --node-aux-weight 0.03 \
      --use-normal-delta \
      --seed 7 \
      --max-train-samples 20000 \
      --device auto
    ;;
  learned-normal-dual-full)
    "${PYTHON_BIN}" scripts/train_full_candidate_stgnn_heatmap_model.py \
      --event-root outputs/impact_labels_aggregated/region_area_sensor_window \
      --raw-label-dir outputs/impact_labels \
      --output-dir "${NEXT_ROOT}/full_candidate_stgnn_learned_normal_dual" \
      --cache-path "${NEXT_ROOT}/full_candidate_stgnn_learned_normal_dual/full_candidate_samples.h5" \
      --rebuild-cache \
      --normal-model-dir "${NEXT_ROOT}/normal_stgnn_forecaster" \
      --normal-infer-batch-size 256 \
      --regions Alameda ContraCosta Orange \
      --input-steps 12 \
      --horizon-steps 12 \
      --max-candidate-nodes 36 \
      --candidate-pm-radius 5.0 \
      --anchor-pm-radius 2.0 \
      --epochs 5 \
      --batch-size 192 \
      --hidden-dim 96 \
      --dropout 0.10 \
      --lr 0.001 \
      --weight-decay 0.0001 \
      --graph-layers 2 \
      --graph-mode undirected \
      --graph-sigma 3.0 \
      --heatmap-aux-weight 0.0 \
      --event-aux-weight 0.05 \
      --node-aux-weight 0.03 \
      --use-normal-delta \
      --use-dual-hist-residual \
      --seed 7 \
      --max-train-samples 20000 \
      --device auto
    ;;
  learned-normal-uncertainty-full)
    run_learned_normal_uncertainty "${NEXT_ROOT}/full_candidate_stgnn_learned_normal_uncertainty" 7
    ;;
  learned-normal-decay-full)
    run_learned_normal_decay "${NEXT_ROOT}/full_candidate_stgnn_learned_normal_decay" 7
    ;;
  learned-normal-uncertainty-seeds)
    run_learned_normal_uncertainty "${NEXT_ROOT}/full_candidate_stgnn_learned_normal_uncertainty_seed_11" 11
    run_learned_normal_uncertainty "${NEXT_ROOT}/full_candidate_stgnn_learned_normal_uncertainty_seed_23" 23
    ;;
  learned-normal-decay-seeds)
    run_learned_normal_decay "${NEXT_ROOT}/full_candidate_stgnn_learned_normal_decay_seed_11" 11
    run_learned_normal_decay "${NEXT_ROOT}/full_candidate_stgnn_learned_normal_decay_seed_23" 23
    ;;
  learned-normal-decay-groups)
    "${PYTHON_BIN}" scripts/evaluate_decay_group_metrics.py \
      --cache-path "${LEARNED_DUAL_CACHE}" \
      --no-decay-model "${NEXT_ROOT}/full_candidate_stgnn_learned_normal_uncertainty/model.pt" \
      --decay-model "${NEXT_ROOT}/full_candidate_stgnn_learned_normal_decay/model.pt" \
      --no-decay-metrics "${NEXT_ROOT}/full_candidate_stgnn_learned_normal_uncertainty/metrics.json" \
      --decay-metrics "${NEXT_ROOT}/full_candidate_stgnn_learned_normal_decay/metrics.json" \
      --output-dir "${NEXT_ROOT}/decay_group_analysis" \
      --batch-size 256 \
      --device auto
    ;;
  learned-normal-fullregion-diagnostics)
    "${PYTHON_BIN}" scripts/diagnose_full_region_normal_inference.py \
      --normal-model-dir "${NEXT_ROOT}/normal_stgnn_forecaster" \
      --output-dir "${NEXT_ROOT}/full_region_normal_diagnostics" \
      --regions Alameda ContraCosta Orange \
      --input-steps 12 \
      --horizon-steps 12 \
      --max-candidate-nodes 36 \
      --candidate-pm-radius 5.0 \
      --anchor-pm-radius 2.0 \
      --max-samples-per-split 256 \
      --batch-size 32 \
      --seed 7 \
      --device auto
    ;;
  diagnostics)
    "${PYTHON_BIN}" scripts/diagnose_learned_normal_residual_cache.py \
      --old-cache outputs/full_candidate_stgnn_heatmap_model/first_pass/full_candidate_samples.h5 \
      --new-cache "${NEXT_ROOT}/full_candidate_stgnn_learned_normal/full_candidate_samples.h5" \
      --old-metrics outputs/full_candidate_stgnn_heatmap_model/ablation_sigma_3_00_undirected/metrics.json \
      --new-metrics "${NEXT_ROOT}/full_candidate_stgnn_learned_normal/metrics.json" \
      --output-dir "${NEXT_ROOT}/learned_normal_residual_diagnostics"
    ;;
  ablations)
    run_full_candidate "${NEXT_ROOT}/ablation_no_graph" \
      --graph-layers 0 \
      --graph-mode undirected \
      --graph-sigma 3.0 \
      --heatmap-aux-weight 0.0
    run_full_candidate "${NEXT_ROOT}/ablation_directional_graph" \
      --graph-layers 2 \
      --graph-mode directional \
      --graph-sigma 3.0 \
      --heatmap-aux-weight 0.0
    run_full_candidate "${NEXT_ROOT}/ablation_heatmap_aux" \
      --graph-layers 2 \
      --graph-mode undirected \
      --graph-sigma 3.0 \
      --heatmap-aux-weight 0.1
    ;;
  *)
    usage
    exit 1
    ;;
esac
