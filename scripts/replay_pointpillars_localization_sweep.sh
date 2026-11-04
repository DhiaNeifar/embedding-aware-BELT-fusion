#!/usr/bin/env bash
# Replay our own pipeline (naive/ego/BELT/AlignFormer) from already-built
# caches across the localization-std sweep, registering anything not yet in
# summary.csv. Read-only replay is fast (no detector forward pass).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
CACHE_ROOT="${ROOT}/outputs/sweep_frame_caches"
RESULTS_DIR="${ROOT}/outputs/noise_sweep_results/pointpillars_pipeline/runs"
PYTHON_BIN="${PYTHON_BIN:-/mnt/external/backup/dhia/conda_envs/opencood-blackwell/bin/python}"
MAX_PARALLEL="${MAX_PARALLEL:-6}"   # concurrent noise points
WORKERS="${WORKERS:-2}"             # DataLoader workers PER point
mkdir -p "${RESULTS_DIR}"

COMMON=(
  -m embedding_aware_belt_fusion.evaluation.belt_fusion
  --opencood-config "${ROOT}/models/pointpillar_late_fusion/config.yaml"
  --detector-dir "${ROOT}/models/pointpillar_late_fusion"
  --uncertainty-checkpoint "${ROOT}/outputs/belt_uncertainty_balanced/uncertainty_epoch10.pth"
  --trackformer-checkpoint "${ROOT}/outputs/spatial_trackformer_p2_h10_full_128d/spatial_trackformer_best.pth"
  --trackformer-root "${ROOT}/external/trackformer"
  --trackformer-protocol propagated
  --trackformer-box-fusion score-weighted
  --trackformer-grouping all-agent-symmetric-pair
  --all-agent-pair-head-checkpoint "${ROOT}/outputs/all_agent_pair_head_p2_h10_full_pq_retrain/all_agent_pair_head_best.pth"
  --ransac-localization-correction
  --pair-head-min-probability 0.5
  --data-root /mnt/external/workspace/public/dataset/opv2v
  --split test
  --workers "${WORKERS}"
  --noise-seed 20
  --global-sort-detections
  --device cuda
)

POINTS=(
  "localization_std_0.2m|0.2"
  "localization_std_0.4m|0.4"
  "localization_std_0.6m|0.6"
  "localization_std_0.8m|0.8"
  "localization_std_1.0m|1.0"
  "localization_std_1.2m|1.2"
  "localization_std_1.4m|1.4"
  "localization_std_1.6m|1.6"
  "localization_std_1.8m|1.8"
)

run_point() {
  local name="$1" pos="$2"
  local cache_dir="${CACHE_ROOT}/${name}"
  local metrics_path="${RESULTS_DIR}/${name}/metrics.json"
  local log_path="${RESULTS_DIR}/${name}.log"

  echo "[replay] ${name} (position_std=${pos}m)"
  {
    "${PYTHON_BIN}" "${COMMON[@]}" \
      --position-noise-std "${pos}" \
      --heading-noise-std-deg 0 \
      --time-delay-ms 0 \
      --frame-cache-mode read-only \
      --frame-cache-dir "${cache_dir}" \
      --output-dir "${RESULTS_DIR}/${name}"

    "${PYTHON_BIN}" -c "
import sys
sys.path.insert(0, '${ROOT}/scripts')
from sweep_results import record_pointpillars_pipeline_run
rows = record_pointpillars_pipeline_run(
    '${metrics_path}', sweep_dimension='localization_std_m', sweep_value=${pos},
)
print(f'recorded {len(rows)} rows for ${name}')
"
    echo "[done] ${name}"
  } > "${log_path}" 2>&1
}

running=0
for point in "${POINTS[@]}"; do
  IFS='|' read -r name pos <<< "${point}"
  run_point "${name}" "${pos}" &
  running=$((running + 1))
  if [ "${running}" -ge "${MAX_PARALLEL}" ]; then
    wait -n
    running=$((running - 1))
  fi
done
wait
echo "All localization-std replays done."
