#!/usr/bin/env bash
# One command, all 6 methods, all 3 noise dimensions. Skips anything already
# registered in summary.csv, so it's always safe to just re-run this after
# any partial progress -- it only computes what's missing.
#
# Methods covered:
#   - pointpillars_pipeline: naive_late_fusion, pointpillars_ego_only, belt,
#     alignformer (all 4 computed together per noise point, frame-cache based)
#   - coalign, roco (each: inference.py + association diagnostics)
#
# Dimensions: localization_std_m (0.0-2.0), heading_std_deg (0.0-10.0),
# comm_delay_ms (0-500). The "0" point of heading/delay is identical to the
# "0" point of localization (no noise at all), so for coalign/roco it's
# registered by re-using their existing localization_std_m=0.0 results
# instead of recomputing -- zero extra GPU/CPU cost.
#
# Why the CPU-bound stretch (CoAlign/RoCo association diagnostics) takes a
# while even though GPU inference is already fast: see comments further
# down. Each process is capped to THREADS_PER_JOB BLAS threads so
# MAX_PARALLEL jobs run concurrently without oversubscribing the CPU.
#
# Usage: bash scripts/run_full_noise_sweep.sh [MAX_PARALLEL] [THREADS_PER_JOB]
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
CONDA_ENVS_ROOT="/mnt/external/backup/dhia/conda_envs"
OPENCOOD_PYTHON_BIN="${CONDA_ENVS_ROOT}/opencood-blackwell/bin/python"
PP_PYTHON_BIN="${CONDA_ENVS_ROOT}/opencood-blackwell/bin/python"

MAX_PARALLEL="${1:-6}"
THREADS_PER_JOB="${2:-3}"
RESULTS_ROOT="${ROOT}/outputs/noise_sweep_results"
CACHE_ROOT="${ROOT}/outputs/sweep_frame_caches"
PP_RESULTS_DIR="${RESULTS_ROOT}/pointpillars_pipeline/runs"
WORKERS="${WORKERS:-2}"

mkdir -p "${PP_RESULTS_DIR}"
for pipeline in coalign roco; do
  for sub in localization_runs heading_runs delay_runs; do
    mkdir -p "${RESULTS_ROOT}/${pipeline}/${sub}"
  done
done

declare -A TREE=( [coalign]="${ROOT}/external/CoAlign" [roco]="${ROOT}/external/RoCo_merged" )
declare -A PYTHON_BIN=( [coalign]="${CONDA_ENVS_ROOT}/coalign-blackwell/bin/python" [roco]="${CONDA_ENVS_ROOT}/roco-blackwell/bin/python" )
declare -A CLEAN_CONFIG=( [coalign]="${ROOT}/models/Zayd Model_clean_test/config.yaml" [roco]="${ROOT}/models/RoCo_clean_test/config.yaml" )
declare -A CHECKPOINT=( [coalign]="${ROOT}/models/Zayd Model_clean_test/net_epoch_bestval_at25.pth" [roco]="${ROOT}/models/RoCo_clean_test/net_epoch_bestval_at23.pth" )
declare -A METHOD_NAME=( [coalign]="Zayd Model" [roco]="RoCo (epoch 23)" )

# ---- ask summary.csv what's already done, once, up front -----------------
COVERAGE="$("${OPENCOOD_PYTHON_BIN}" -c "
import csv
rows = list(csv.DictReader(open('${RESULTS_ROOT}/summary.csv')))
have = set()
for r in rows:
    have.add((r['pipeline'], r['method'], r['sweep_dimension'], r['sweep_value']))

pp_methods = ['naive_late_fusion', 'pointpillars_ego_only', 'belt', 'alignformer']
def pp_done(dim, val):
    return all(('pointpillars_pipeline', m, dim, val) in have for m in pp_methods)

for dim, val in [('localization_std_m', '0.0')] + [('localization_std_m', f'{v:.1f}') for v in [0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0]] + [('heading_std_deg', f'{v:.1f}') for v in range(0,11)] + [('comm_delay_ms', str(v)) for v in [0,100,200,300,400,500]]:
    if pp_done(dim, val):
        print(f'pointpillars|{dim}|{val}')

for pipeline, method in [('coalign','Zayd Model'), ('roco','RoCo (epoch 23)')]:
    for dim, val in [('localization_std_m', f'{v:.1f}') for v in [0.0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0]] + [('heading_std_deg', f'{v:.1f}') for v in range(0,11)] + [('comm_delay_ms', str(v)) for v in [0,100,200,300,400,500]]:
        if (pipeline, method, dim, val) in have:
            print(f'{pipeline}|{dim}|{val}')
")"
declare -A ALREADY_DONE
while IFS= read -r line; do
  [ -n "${line}" ] && ALREADY_DONE["${line}"]=1
done <<< "${COVERAGE}"

is_done() { [ -n "${ALREADY_DONE[$1|$2|$3]:-}" ]; }

# ---- pointpillars_pipeline job table (one run = all 4 methods) -----------
# name | position_std_m | heading_std_deg | delay_ms | record_dimensions
PP_POINTS=(
  "baseline_clean|0.0|0|0|localization_std_m=0.0,heading_std_deg=0.0,comm_delay_ms=0"
  "localization_std_0.2m|0.2|0|0|localization_std_m=0.2"
  "localization_std_0.4m|0.4|0|0|localization_std_m=0.4"
  "localization_std_0.6m|0.6|0|0|localization_std_m=0.6"
  "localization_std_0.8m|0.8|0|0|localization_std_m=0.8"
  "localization_std_1.0m|1.0|0|0|localization_std_m=1.0"
  "localization_std_1.2m|1.2|0|0|localization_std_m=1.2"
  "localization_std_1.4m|1.4|0|0|localization_std_m=1.4"
  "localization_std_1.6m|1.6|0|0|localization_std_m=1.6"
  "localization_std_1.8m|1.8|0|0|localization_std_m=1.8"
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

PP_COMMON=(
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
  --pair-head-min-probability 0.5
  --ransac-localization-correction
  --data-root /mnt/external/workspace/public/dataset/opv2v
  --split test
  --workers "${WORKERS}"
  --noise-seed 20
  --global-sort-detections
  --device cuda
)

# ---- coalign/roco: heading + comm_delay point lists -----------------------
HEADING_POINTS=(1 2 3 4 5 6 7 8 9 10)
DELAY_POINTS=(100 200 300 400 500)
LOCALIZATION_POINTS=(0.0 0.2 0.4 0.6 0.8 1.0 1.2 1.4 1.6 1.8 2.0)

# ---- build the unified job list -------------------------------------------
# Each entry: kind|key1|key2|key3  (kind decides how run_job interprets it)
JOBS=()

for point in "${PP_POINTS[@]}"; do
  IFS='|' read -r name pos head delay dims <<< "${point}"
  # A pointpillars point can feed multiple dimensions at once (baseline_clean
  # feeds all 3); only skip it if every one of those dims is already done.
  all_done=1
  IFS=',' read -ra dim_pairs <<< "${dims}"
  for pair in "${dim_pairs[@]}"; do
    dim="${pair%%=*}"; val="${pair##*=}"
    is_done "pointpillars" "${dim}" "${val}" || all_done=0
  done
  [ "${all_done}" -eq 1 ] && continue
  JOBS+=("pp|${name}|${pos}|${head}|${delay}|${dims}")
done

for pipeline in coalign roco; do
  # localization_std_m
  for v in "${LOCALIZATION_POINTS[@]}"; do
    is_done "${pipeline}" "localization_std_m" "${v}" && continue
    JOBS+=("cr|${pipeline}|localization|${v}|localization_std_${v}m")
  done
  # heading_std_deg (0.0 handled via cheap reuse below, not here)
  for v in "${HEADING_POINTS[@]}"; do
    is_done "${pipeline}" "heading_std_deg" "${v}.0" && continue
    JOBS+=("cr|${pipeline}|heading|${v}|heading_std_${v}deg")
  done
  # comm_delay_ms (0 handled via cheap reuse below, not here)
  for v in "${DELAY_POINTS[@]}"; do
    is_done "${pipeline}" "comm_delay_ms" "${v}" && continue
    JOBS+=("cr|${pipeline}|delay|${v}|comm_delay_${v}ms")
  done
  # cheap reuse: heading=0.0 / delay=0 are identical to localization=0.0,
  # register from that existing result instead of recomputing.
  if ! is_done "${pipeline}" "heading_std_deg" "0.0"; then
    JOBS+=("cr_reuse|${pipeline}|heading_std_deg|0.0|reuse_heading0")
  fi
  if ! is_done "${pipeline}" "comm_delay_ms" "0"; then
    JOBS+=("cr_reuse|${pipeline}|comm_delay_ms|0|reuse_delay0")
  fi
done

TOTAL=${#JOBS[@]}
if [ "${TOTAL}" -eq 0 ]; then
  echo "Nothing to do -- every method/dimension/value is already in summary.csv."
  exit 0
fi

STATUS_DIR="$(mktemp -d)"
trap 'rm -rf "${STATUS_DIR}"' EXIT

job_label() {
  # unique, filesystem-safe id for a job's status file
  echo "$1" | tr '|/ ' '___'
}

run_pp_job() {
  local name="$1" pos="$2" head="$3" delay="$4" dims="$5" status_file="$6"
  local cache_dir="${CACHE_ROOT}/${name}"
  local metrics_path="${PP_RESULTS_DIR}/${name}/metrics.json"
  local log_path="${PP_RESULTS_DIR}/${name}.log"

  echo "RUNNING pipeline" > "${status_file}"
  (
    set -e
    "${PP_PYTHON_BIN}" "${PP_COMMON[@]}" \
      --position-noise-std "${pos}" \
      --heading-noise-std-deg "${head}" \
      --time-delay-ms "${delay}" \
      --frame-cache-mode read-write \
      --frame-cache-dir "${cache_dir}" \
      --output-dir "${PP_RESULTS_DIR}/${name}"

    echo "RUNNING register" > "${status_file}"
    IFS=',' read -ra dim_pairs <<< "${dims}"
    for pair in "${dim_pairs[@]}"; do
      dim="${pair%%=*}"; value="${pair##*=}"
      "${OPENCOOD_PYTHON_BIN}" -c "
import sys
sys.path.insert(0, '${ROOT}/scripts')
from sweep_results import record_pointpillars_pipeline_run
record_pointpillars_pipeline_run(
    '${metrics_path}', sweep_dimension='${dim}', sweep_value=${value},
)
"
    done
    echo "DONE" > "${status_file}"
  ) > "${log_path}" 2>&1 || echo "FAILED (see ${log_path})" > "${status_file}"
}

run_cr_job() {
  local pipeline="$1" dimension="$2" value="$3" name="$4" status_file="$5"
  local subdir dim_key sweep_value
  case "${dimension}" in
    localization) subdir="localization_runs"; dim_key="localization_std_m"; sweep_value="${value}" ;;
    heading)      subdir="heading_runs"; dim_key="heading_std_deg"; sweep_value="${value}.0" ;;
    delay)        subdir="delay_runs"; dim_key="comm_delay_ms"; sweep_value="${value}" ;;
  esac
  local results_dir="${RESULTS_ROOT}/${pipeline}/${subdir}"
  local model_dir="${ROOT}/models/${pipeline}_eval_${name}"
  local assoc_json="${results_dir}/${name}_association.json"
  local log_path="${results_dir}/${name}.log"

  echo "RUNNING inference" > "${status_file}"
  (
    set -e
    mkdir -p "${model_dir}"
    ln -sf "${CHECKPOINT[$pipeline]}" "${model_dir}/$(basename "${CHECKPOINT[$pipeline]}")"

    case "${dimension}" in
      localization)
        "${PYTHON_BIN[$pipeline]}" -c "
import opencood.hypes_yaml.yaml_utils as yaml_utils
hypes = yaml_utils.load_yaml('${CLEAN_CONFIG[$pipeline]}', None)
hypes['noise_setting']['add_noise'] = True
hypes['noise_setting']['args'] = {'pos_std': ${value}, 'rot_std': 0.0, 'pos_mean': 0, 'rot_mean': 0}
yaml_utils.save_yaml(hypes, '${model_dir}/config.yaml')
"
        ;;
      heading)
        "${PYTHON_BIN[$pipeline]}" -c "
import opencood.hypes_yaml.yaml_utils as yaml_utils
hypes = yaml_utils.load_yaml('${CLEAN_CONFIG[$pipeline]}', None)
hypes['noise_setting']['add_noise'] = True
hypes['noise_setting']['args'] = {'pos_std': 0.0, 'rot_std': ${value}.0, 'pos_mean': 0, 'rot_mean': 0}
yaml_utils.save_yaml(hypes, '${model_dir}/config.yaml')
"
        ;;
      delay)
        "${PYTHON_BIN[$pipeline]}" -c "
import opencood.hypes_yaml.yaml_utils as yaml_utils
hypes = yaml_utils.load_yaml('${CLEAN_CONFIG[$pipeline]}', None)
hypes['noise_setting']['add_noise'] = False
hypes['wild_setting'] = {'async': True, 'async_mode': 'sim', 'async_overhead': ${value}}
yaml_utils.save_yaml(hypes, '${model_dir}/config.yaml')
"
        ;;
    esac

    cd "${TREE[$pipeline]}"
    OMP_NUM_THREADS="${THREADS_PER_JOB}" OPENBLAS_NUM_THREADS="${THREADS_PER_JOB}" MKL_NUM_THREADS="${THREADS_PER_JOB}" \
      "${PYTHON_BIN[$pipeline]}" -u opencood/tools/inference.py \
      --model_dir "${model_dir}" \
      --fusion_method intermediate \
      --save_vis_interval 1000000 \
      --note "${dimension}_${value}"

    echo "RUNNING association-diagnostics" > "${status_file}"
    eval_yaml=$(ls "${model_dir}"/eval_*.yaml | head -1)

    OMP_NUM_THREADS="${THREADS_PER_JOB}" OPENBLAS_NUM_THREADS="${THREADS_PER_JOB}" MKL_NUM_THREADS="${THREADS_PER_JOB}" \
      "${PYTHON_BIN[$pipeline]}" -u opencood/tools/evaluate_association_diagnostics.py \
      --hypes_yaml "${model_dir}/config.yaml" \
      > "${assoc_json}" || true
    # tqdm's stderr stream occasionally gets corrupted (stray bytes) in a way
    # that makes the process exit non-zero even though the real stdout JSON
    # is complete and valid; judge success by the JSON content, not the exit
    # code, so a cosmetic logging glitch doesn't throw away a finished run.
    python3 -c "
text = open('${assoc_json}').read()
import json; json.loads(text[text.index('{'):])
" || { echo "association-diagnostics produced no valid JSON" >&2; exit 1; }

    echo "RUNNING register" > "${status_file}"
    "${OPENCOOD_PYTHON_BIN}" -c "
import sys
sys.path.insert(0, '${ROOT}/scripts')
from sweep_results import record_coalign_or_roco_run
record_coalign_or_roco_run(
    '${pipeline}', '${METHOD_NAME[$pipeline]}', '${eval_yaml}',
    sweep_dimension='${dim_key}', sweep_value=${sweep_value},
    association_json='${assoc_json}',
)
"
    echo "DONE" > "${status_file}"
  ) > "${log_path}" 2>&1 || echo "FAILED (see ${log_path})" > "${status_file}"
}

run_cr_reuse_job() {
  # heading=0.0 / delay=0 are the same "no noise" condition as the existing
  # clean_test predictions -- no inference.py re-run needed, just run
  # association diagnostics once against the existing clean config (shared
  # across both reuse calls, guarded by a lock so it only computes once).
  local pipeline="$1" dim_key="$2" sweep_value="$3" status_file="$4"
  local clean_dir="${ROOT}/models/${METHOD_NAME[$pipeline]}_clean_test"
  [ "${pipeline}" = "roco" ] && clean_dir="${ROOT}/models/RoCo_clean_test"
  [ "${pipeline}" = "coalign" ] && clean_dir="${ROOT}/models/Zayd Model_clean_test"
  local assoc_json="${RESULTS_ROOT}/${pipeline}/localization_runs/clean_association.json"
  local lock_dir="${assoc_json}.lock"
  local ready_marker="${assoc_json}.ready"
  local log_path="${RESULTS_ROOT}/${pipeline}/localization_runs/reuse_${dim_key}.log"

  echo "RUNNING association-diagnostics" > "${status_file}"
  (
    set -e
    eval_yaml=$(ls "${clean_dir}"/eval_*.yaml 2>/dev/null | head -1)
    [ -n "${eval_yaml}" ] || { echo "no eval_yaml found at ${clean_dir}" >&2; exit 1; }

    if mkdir "${lock_dir}" 2>/dev/null; then
      OMP_NUM_THREADS="${THREADS_PER_JOB}" OPENBLAS_NUM_THREADS="${THREADS_PER_JOB}" MKL_NUM_THREADS="${THREADS_PER_JOB}" \
        "${PYTHON_BIN[$pipeline]}" -u "${TREE[$pipeline]}/opencood/tools/evaluate_association_diagnostics.py" \
        --hypes_yaml "${clean_dir}/config.yaml" \
        > "${assoc_json}" || true
      # judge success by JSON validity, not exit code -- see comment in
      # run_cr_job for why the process can exit non-zero despite valid output.
      python3 -c "
text = open('${assoc_json}').read()
import json; json.loads(text[text.index('{'):])
" || { echo "association-diagnostics produced no valid JSON" >&2; exit 1; }
      touch "${ready_marker}"
    else
      # another reuse job for this pipeline is already computing it; wait
      # for its completion marker, not just non-empty output (the diagnostics
      # script prints a "Dataset dir: ..." line almost instantly, well before
      # the real JSON is written, so file-size alone is not a safe signal).
      until [ -f "${ready_marker}" ]; do sleep 5; done
    fi

    echo "RUNNING register" > "${status_file}"
    "${OPENCOOD_PYTHON_BIN}" -c "
import sys
sys.path.insert(0, '${ROOT}/scripts')
from sweep_results import record_coalign_or_roco_run
record_coalign_or_roco_run(
    '${pipeline}', '${METHOD_NAME[$pipeline]}', '${eval_yaml}',
    sweep_dimension='${dim_key}', sweep_value=${sweep_value},
    association_json='${assoc_json}',
)
"
    echo "DONE" > "${status_file}"
  ) > "${log_path}" 2>&1 || echo "FAILED (see ${log_path})" > "${status_file}"
}

dispatch() {
  local job="$1"
  local kind="${job%%|*}"
  local rest="${job#*|}"
  local label; label="$(job_label "${job}")"
  local status_file="${STATUS_DIR}/${label}.status"
  echo "QUEUED" > "${status_file}"
  case "${kind}" in
    pp)
      IFS='|' read -r name pos head delay dims <<< "${rest}"
      run_pp_job "${name}" "${pos}" "${head}" "${delay}" "${dims}" "${status_file}"
      ;;
    cr)
      IFS='|' read -r pipeline dimension value name <<< "${rest}"
      run_cr_job "${pipeline}" "${dimension}" "${value}" "${name}" "${status_file}"
      ;;
    cr_reuse)
      IFS='|' read -r pipeline dim_key sweep_value name <<< "${rest}"
      run_cr_reuse_job "${pipeline}" "${dim_key}" "${sweep_value}" "${status_file}"
      ;;
  esac
}

# ---- launch everything with MAX_PARALLEL throttling -----------------------
running=0
for job in "${JOBS[@]}"; do
  dispatch "${job}" &
  running=$((running + 1))
  if [ "${running}" -ge "${MAX_PARALLEL}" ]; then
    wait -n
    running=$((running - 1))
  fi
done

# ---- live dashboard, redrawn in place (no scrollback growth) -------------
describe_job() {
  local job="$1"
  local kind="${job%%|*}"
  local rest="${job#*|}"
  case "${kind}" in
    pp)       IFS='|' read -r name _ _ _ _ <<< "${rest}"; echo "pointpillars|${name}" ;;
    cr)       IFS='|' read -r pipeline dimension value _ <<< "${rest}"; echo "${pipeline}|${dimension}=${value}" ;;
    cr_reuse) IFS='|' read -r pipeline dim_key sweep_value _ <<< "${rest}"; echo "${pipeline}|${dim_key}=${sweep_value} (reuse)" ;;
  esac
}

job_log_path() {
  local job="$1"
  local kind="${job%%|*}"
  local rest="${job#*|}"
  case "${kind}" in
    pp)
      IFS='|' read -r name _ _ _ _ <<< "${rest}"
      echo "${PP_RESULTS_DIR}/${name}.log"
      ;;
    cr)
      IFS='|' read -r pipeline dimension value name <<< "${rest}"
      local subdir
      case "${dimension}" in
        localization) subdir="localization_runs" ;;
        heading)      subdir="heading_runs" ;;
        delay)        subdir="delay_runs" ;;
      esac
      echo "${RESULTS_ROOT}/${pipeline}/${subdir}/${name}.log"
      ;;
    cr_reuse)
      IFS='|' read -r pipeline dim_key _ _ <<< "${rest}"
      echo "${RESULTS_ROOT}/${pipeline}/localization_runs/reuse_${dim_key}.log"
      ;;
  esac
}

draw_dashboard() {
  local done_count=0
  clear
  echo "Full noise sweep -- all 6 methods, all 3 dimensions (MAX_PARALLEL=${MAX_PARALLEL}, ${THREADS_PER_JOB} threads/job)"
  echo "================================================================================"
  printf "%-60s %s\n" "JOB" "STATUS"
  for job in "${JOBS[@]}"; do
    local label; label="$(job_label "${job}")"
    local status_file="${STATUS_DIR}/${label}.status"
    local status
    status=$(cat "${status_file}" 2>/dev/null || echo "QUEUED")
    if [[ "${status}" == DONE* || "${status}" == FAILED* ]]; then
      done_count=$((done_count + 1))
    elif [[ "${status}" == RUNNING* ]]; then
      local log_path; log_path="$(job_log_path "${job}")"
      local pct
      pct=$(tr '\r' '\n' < "${log_path}" 2>/dev/null | grep -a -oE '[0-9]+%\|' | tail -1)
      [ -n "${pct}" ] && status="${status} (${pct%\|})"
    fi
    printf "%-60s %s\n" "$(describe_job "${job}")" "${status}"
  done
  echo "================================================================================"
  echo "Completed: ${done_count} / ${TOTAL}"
}

while :; do
  draw_dashboard
  remaining=0
  for job in "${JOBS[@]}"; do
    local_label="$(job_label "${job}")"
    status=$(cat "${STATUS_DIR}/${local_label}.status" 2>/dev/null || echo "QUEUED")
    [[ "${status}" == DONE* || "${status}" == FAILED* ]] || remaining=$((remaining + 1))
  done
  [ "${remaining}" -eq 0 ] && break
  sleep 5
done

wait
draw_dashboard
echo "All jobs finished."
