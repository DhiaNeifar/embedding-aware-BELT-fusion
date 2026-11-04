#!/usr/bin/env bash
# Build every still-missing frame cache for our own pipeline (naive late
# fusion / ego-only / BELT-geometry / AlignFormer, all four computed in one
# pass) across the full localization/heading/comm-delay sweep, and record
# each run's results into outputs/noise_sweep_results/summary.csv.
#
# Already-built caches (outputs/sweep_frame_caches/) are skipped, so this is
# safe to re-run -- it only fills in what's missing.
#
# Each noise point costs one full GPU pass (~10-15 min on a clean run).
# This script does NOT need RoCo's training to finish -- it's entirely our
# own pipeline, independent of CoAlign/RoCo.
#
# Points run MAX_PARALLEL at a time (default 4) as separate background
# processes sharing the one GPU -- a single run doesn't saturate it (it's
# partly CPU/TrackFormer/RANSAC-bound), so a few concurrent runs fill the
# idle gaps instead of one run just waiting on CPU work serially. Override
# with: MAX_PARALLEL=N bash scripts/run_pointpillars_pipeline_noise_sweep.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
CACHE_ROOT="${ROOT}/outputs/sweep_frame_caches"
RESULTS_DIR="${ROOT}/outputs/noise_sweep_results/pointpillars_pipeline/runs"
PYTHON_BIN="${PYTHON_BIN:-/mnt/external/backup/dhia/conda_envs/opencood-blackwell/bin/python}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"
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

# name | position_std_m | heading_std_deg | delay_ms | record_dimensions(csv of dim=value pairs)
POINTS=(
  "baseline_clean|0.0|0|0|localization_std_m=0.0,heading_std_deg=0.0,comm_delay_ms=0"
  "localization_std_2.0m|2.0|0|0|localization_std_m=2.0"
  "heading_std_1deg|0.0|1|0|heading_std_deg=1.0"
  "heading_std_2deg|0.0|2|0|heading_std_deg=2.0"
  "heading_std_3deg|0.0|3|0|heading_std_deg=3.0"
  "heading_std_4deg|0.0|4|0|heading_std_deg=4.0"
  "heading_std_5deg|0.0|5|0|heading_std_deg=5.0"
  "heading_std_6deg|0.0|6|0|heading_std_deg=6.0"
  "heading_std_7deg|0.0|7|0|heading_std_deg=7.0"
  "heading_std_8deg|0.0|8|0|heading_std_deg=8.0"
  "heading_std_9deg|0.0|9|0|heading_std_deg=9.0"
  "heading_std_10deg|0.0|10|0|heading_std_deg=10.0"
  "comm_delay_100ms|0.0|0|100|comm_delay_ms=100"
  "comm_delay_200ms|0.0|0|200|comm_delay_ms=200"
  "comm_delay_300ms|0.0|0|300|comm_delay_ms=300"
  "comm_delay_400ms|0.0|0|400|comm_delay_ms=400"
  "comm_delay_500ms|0.0|0|500|comm_delay_ms=500"
)

run_point() {
  local name="$1" pos="$2" head="$3" delay="$4" dims="$5"
  local cache_dir="${CACHE_ROOT}/${name}"
  local metrics_path="${RESULTS_DIR}/${name}/metrics.json"
  local log_path="${RESULTS_DIR}/${name}.log"

  if [ -d "${cache_dir}" ] && [ -f "${metrics_path}" ]; then
    echo "[skip] ${name}: cache + results already exist"
    return 0
  fi

  echo "[start] ${name}  (position_std=${pos}m heading_std=${head}deg delay=${delay}ms) -> ${log_path}"
  {
    "${PYTHON_BIN}" "${COMMON[@]}" \
      --position-noise-std "${pos}" \
      --heading-noise-std-deg "${head}" \
      --time-delay-ms "${delay}" \
      --frame-cache-mode read-write \
      --frame-cache-dir "${cache_dir}" \
      --output-dir "${RESULTS_DIR}/${name}"

    IFS=',' read -ra dim_pairs <<< "${dims}"
    for pair in "${dim_pairs[@]}"; do
      dim="${pair%%=*}"
      value="${pair##*=}"
      "${PYTHON_BIN}" -c "
import sys
sys.path.insert(0, '${ROOT}/scripts')
from sweep_results import record_pointpillars_pipeline_run
rows = record_pointpillars_pipeline_run(
    '${metrics_path}',
    sweep_dimension='${dim}', sweep_value=${value},
)
print(f'recorded {len(rows)} rows for ${name} (${dim}=${value})')
"
    done
    echo "[done] ${name}"
  } > "${log_path}" 2>&1
}

running=0
for point in "${POINTS[@]}"; do
  IFS='|' read -r name pos head delay dims <<< "${point}"

  run_point "${name}" "${pos}" "${head}" "${delay}" "${dims}" &

  running=$((running + 1))
  if [ "${running}" -ge "${MAX_PARALLEL}" ]; then
    wait -n
    running=$((running - 1))
  fi
done
wait

echo "All pointpillars-pipeline sweep points present. Per-point logs: ${RESULTS_DIR}/<name>.log"
