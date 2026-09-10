#!/usr/bin/env bash
# Run all five combined geometry-noise points concurrently for one model.
#
# Usage:
#   bash scripts/run_belt_geometry_noise_sweep.sh naive
#   bash scripts/run_belt_geometry_noise_sweep.sh spatial_trackformer_clean
#   bash scripts/run_belt_geometry_noise_sweep.sh spatial_trackformer_annealed
#   bash scripts/run_belt_geometry_noise_sweep.sh spatial_trackformer_annealed_codebook

set -euo pipefail

MODEL="${1:?Choose naive, spatial_trackformer_clean, spatial_trackformer_annealed, or spatial_trackformer_annealed_codebook}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
OUTPUT_ROOT="${ROOT}/outputs/belt_geometry_noise_sweep/${MODEL}"
mkdir -p "\${OUTPUT_ROOT}"

COMMON=(
  -m embedding_aware_belt_fusion.evaluation.belt_fusion
  --opencood-config "${ROOT}/models/pointpillar_late_fusion/config.yaml"
  --detector-dir "${ROOT}/models/pointpillar_late_fusion"
  --uncertainty-checkpoint "${ROOT}/outputs/belt_uncertainty_balanced/uncertainty_epoch10.pth"
  --data-root /mnt/external/workspace/public/dataset/opv2v
  --split test
  --workers 4
  --association-distance 5.0
  --embedding-distance 5.0
  --embedding-min-similarity -1.0
  --track-query-score-threshold 0.5
  --time-delay-ms 0
  --noise-seed 20
  --device cuda
)

case "${MODEL}" in
  naive)
    MODEL_ARGS=()
    ;;
  spatial_trackformer_clean)
    MODEL_ARGS=(
      --trackformer-checkpoint "${ROOT}/outputs/spatial_trackformer_association_head_128d/spatial_trackformer_best.pth"
      --trackformer-root "${ROOT}/external/trackformer"
      --trackformer-protocol propagated
    )
    ;;
  spatial_trackformer_annealed)
    MODEL_ARGS=(
      --trackformer-checkpoint "${ROOT}/outputs/spatial_trackformer_curriculum_strong_128d/spatial_trackformer_best.pth"
      --trackformer-root "${ROOT}/external/trackformer"
      --trackformer-protocol propagated
    )
    ;;
  spatial_trackformer_annealed_codebook)
    MODEL_ARGS=(
      --trackformer-checkpoint "${ROOT}/outputs/spatial_trackformer_curriculum_strong_128d/spatial_trackformer_best.pth"
      --trackformer-root "${ROOT}/external/trackformer"
      --trackformer-protocol propagated
      --message-codebook "${ROOT}/outputs/codebooks/annealed_query_state_8x256.pth"
    )
    ;;
  *)
    echo "Unknown model: ${MODEL}" >&2
    exit 2
    ;;
esac

TAGS=(clean p1_h5 p2_h10 p4_h20 p10_h30)
POSITIONS=(0 1 2 4 10)
HEADINGS=(0 5 10 20 30)
PIDS=()

for index in "${!TAGS[@]}"; do
  tag="${TAGS[index]}"
  position="${POSITIONS[index]}"
  heading="${HEADINGS[index]}"
  (
    export PYTHONPATH="${ROOT}/external/OpenCOOD${PYTHONPATH:+:${PYTHONPATH}}"
    python "${COMMON[@]}" "${MODEL_ARGS[@]}" \
      --position-noise-std "${position}" \
      --heading-noise-std-deg "${heading}" \
      --output-dir "${OUTPUT_ROOT}/${tag}"
  ) &
  PIDS+=("$!")
  echo "Started ${MODEL}/${tag}: position std=${position} m, heading std=${heading} deg"
done

status=0
for pid in "${PIDS[@]}"; do
  wait "${pid}" || status=1
done
exit "${status}"
