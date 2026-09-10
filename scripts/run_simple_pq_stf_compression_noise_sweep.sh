#!/usr/bin/env bash
# Measure fixed-rate PQ-STF-only association under combined localization noise.
#
# The clean point is reused from existing outputs.  This script evaluates the
# remaining four geometry-noise points sequentially, so it is safe to resume
# after an interruption and does not contend for a single GPU.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
OUTPUT_ROOT="${ROOT}/outputs/simple_pq_stf_compression_noise_sweep"
CODEBOOK="${ROOT}/outputs/codebooks/pq_stf_residual_query_state_8_16_32.pth"

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
  --data-root /mnt/external/workspace/public/dataset/opv2v
  --split test
  --workers 4
  --embedding-distance 5.0
  --embedding-min-similarity 0.90
  --track-query-score-threshold 0.5
  --time-delay-ms 0
  --noise-seed 20
  --device cuda
)

run_one() {
  local tag="$1"
  local position="$2"
  local heading="$3"
  local rate="$4"
  local stages="$5"
  local output_dir="${OUTPUT_ROOT}/${tag}/${rate}"

  if [[ -f "${output_dir}/metrics.json" ]]; then
    echo "Already complete: ${tag}/${rate}"
    return
  fi

  local message_args=()
  if [[ "${stages}" != "raw" ]]; then
    message_args=(
      --residual-message-codebook "${CODEBOOK}"
      --residual-stages "${stages}"
    )
  fi
  echo "Running ${tag}/${rate}: position std=${position} m, heading std=${heading} deg"
  PYTHONPATH="${ROOT}/external/OpenCOOD${PYTHONPATH:+:${PYTHONPATH}}" \
    python "${COMMON[@]}" "${message_args[@]}" \
      --position-noise-std "${position}" \
      --heading-noise-std-deg "${heading}" \
      --output-dir "${output_dir}"
}

# Combined position/heading localization-noise conditions from the earlier
# robustness figure.  Clean is deliberately not repeated.
TAGS=(p1_h5 p2_h10 p4_h20 p10_h30)
POSITIONS=(1 2 4 10)
HEADINGS=(5 10 20 30)

for index in "${!TAGS[@]}"; do
  run_one "${TAGS[index]}" "${POSITIONS[index]}" "${HEADINGS[index]}" raw raw
  run_one "${TAGS[index]}" "${POSITIONS[index]}" "${HEADINGS[index]}" fixed_4B 3
  run_one "${TAGS[index]}" "${POSITIONS[index]}" "${HEADINGS[index]}" fixed_2B 2
  run_one "${TAGS[index]}" "${POSITIONS[index]}" "${HEADINGS[index]}" fixed_1B 1
done

python "${ROOT}/scripts/plot_simple_pq_stf_compression_noise_sweep.py"
