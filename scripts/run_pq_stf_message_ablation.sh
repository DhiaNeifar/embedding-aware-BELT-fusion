#!/usr/bin/env bash
# Controlled PQ-STF message-value diagnostic on the official clean test set.
# Only the 256-D propagated query-state content changes across runs.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
OUTPUT_ROOT="${ROOT}/outputs/pq_stf_message_content_ablation_clean_test"
PYTHON_BIN="${PYTHON_BIN:-/mnt/external/backup/dhia/conda_envs/opencood-blackwell/bin/python}"
mkdir -p "${OUTPUT_ROOT}"

COMMON=(
  -m embedding_aware_belt_fusion.evaluation.belt_fusion
  --opencood-config "${ROOT}/models/pointpillar_late_fusion/config.yaml"
  --detector-dir "${ROOT}/models/pointpillar_late_fusion"
  --uncertainty-checkpoint "${ROOT}/outputs/belt_uncertainty_balanced/uncertainty_epoch10.pth"
  --trackformer-checkpoint "${ROOT}/outputs/spatial_trackformer_association_head_128d/spatial_trackformer_best.pth"
  --trackformer-root "${ROOT}/external/trackformer"
  --trackformer-protocol propagated
  --trackformer-box-fusion score-weighted
  --trackformer-grouping ego-centric
  --data-root /mnt/external/workspace/public/dataset/opv2v
  --split test
  --workers 4
  --association-distance 5.0
  --embedding-distance 5.0
  --embedding-min-similarity 0.90
  --track-query-score-threshold 0.5
  --device cuda
)

for mode in none zero permute; do
  echo "Running PQ-STF query-state ablation: ${mode}"
  "${PYTHON_BIN}" "${COMMON[@]}" \
    --message-content-ablation "${mode}" \
    --output-dir "${OUTPUT_ROOT}/${mode}"
done
