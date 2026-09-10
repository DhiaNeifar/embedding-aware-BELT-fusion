#!/usr/bin/env bash
# Controlled clean-test sweep: only the transmitted PQ-STF query-state rate
# changes.  Runs sequentially because all jobs require the same GPU.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="${1:-$ROOT/outputs/pq_stf_low_rate_sweep}"
mkdir -p "$OUTPUT_ROOT"

COMMON=(
  --opencood-config "$ROOT/models/pointpillar_late_fusion/config.yaml"
  --detector-dir "$ROOT/models/pointpillar_late_fusion"
  --uncertainty-checkpoint "$ROOT/outputs/belt_uncertainty_balanced/uncertainty_epoch10.pth"
  --trackformer-checkpoint "$ROOT/outputs/spatial_trackformer_association_head_128d/spatial_trackformer_best.pth"
  --trackformer-root "$ROOT/external/trackformer"
  --trackformer-protocol propagated
  --data-root /mnt/external/workspace/public/dataset/opv2v
  --split test
  --workers 4
  --association-distance 5.0
  --embedding-distance 5.0
  --embedding-min-similarity 0.90
  --track-query-score-threshold 0.5
  --device cuda
)

run_raw() {
  python -m embedding_aware_belt_fusion.evaluation.belt_fusion \
    "${COMMON[@]}" \
    --output-dir "$OUTPUT_ROOT/raw_1024B"
}

run_fixed() {
  local stages="$1"
  local name="$2"
  python -m embedding_aware_belt_fusion.evaluation.belt_fusion \
    "${COMMON[@]}" \
    --residual-message-codebook "$ROOT/outputs/codebooks/pq_stf_residual_query_state_8_16_32.pth" \
    --residual-stages "$stages" \
    --output-dir "$OUTPUT_ROOT/$name"
}

run_adaptive() {
  python -m embedding_aware_belt_fusion.evaluation.belt_fusion \
    "${COMMON[@]}" \
    --residual-message-codebook "$ROOT/outputs/codebooks/pq_stf_residual_query_state_8_16_32.pth" \
    --adaptive-rate-thresholds 0.61 0.64 \
    --output-dir "$OUTPUT_ROOT/adaptive_1_2_4B"
}

run_raw
run_fixed 1 fixed_1B
run_fixed 2 fixed_2B
run_fixed 3 fixed_4B
run_adaptive

python "$ROOT/scripts/plot_pq_stf_low_rate_sweep.py" \
  --metrics-root "$OUTPUT_ROOT"
