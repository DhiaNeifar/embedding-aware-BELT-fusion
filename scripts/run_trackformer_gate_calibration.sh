#!/usr/bin/env bash
# Calibrate propagated Spatial TrackFormer association gates on held-out
# clean OPV2V validation scenarios. Test data must not be used here.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
OUTPUT_ROOT="${ROOT}/outputs/trackformer_gate_calibration_clean_validation"

# A match must pass BOTH gates:
#   distance <= DISTANCE metres and cosine similarity >= SIMILARITY.
DISTANCES=(5 8 10)
SIMILARITIES=(0.50 0.60 0.70 0.80 0.90)

mkdir -p "${OUTPUT_ROOT}"
export PYTHONPATH="${ROOT}/external/OpenCOOD${PYTHONPATH:+:${PYTHONPATH}}"

for distance in "${DISTANCES[@]}"; do
  for similarity in "${SIMILARITIES[@]}"; do
    tag="d${distance}_s${similarity/./p}"
    echo "Evaluating distance=${distance} m, minimum cosine=${similarity}"
    python -m embedding_aware_belt_fusion.evaluation.belt_fusion \
      --opencood-config "${ROOT}/models/pointpillar_late_fusion/config.yaml" \
      --detector-dir "${ROOT}/models/pointpillar_late_fusion" \
      --uncertainty-checkpoint "${ROOT}/outputs/belt_uncertainty_balanced/uncertainty_epoch10.pth" \
      --trackformer-checkpoint "${ROOT}/outputs/spatial_trackformer_association_head_128d/spatial_trackformer_best.pth" \
      --trackformer-root "${ROOT}/external/trackformer" \
      --trackformer-protocol propagated \
      --data-root /mnt/external/workspace/public/dataset/opv2v \
      --output-dir "${OUTPUT_ROOT}/${tag}" \
      --split train \
      --scenario-split-file "${ROOT}/outputs/spatial_trackformer_association_head_128d/scenario_split.json" \
      --scenario-role validation \
      --workers 4 \
      --position-noise-std 0 \
      --heading-noise-std-deg 0 \
      --time-delay-ms 0 \
      --noise-seed 20 \
      --association-distance 5.0 \
      --embedding-distance "${distance}" \
      --embedding-min-similarity "${similarity}" \
      --track-query-score-threshold 0.5 \
      --device cuda
  done
done
