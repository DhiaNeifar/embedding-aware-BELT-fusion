#!/usr/bin/env bash
# Run CoAlign's or RoCo's own inference.py + association diagnostics across
# the localization-std sweep (0.2m - 2.0m; 0.0 already done separately).
# Noise is independent of the precalc'd stage1_boxes.json (confirmed: it's
# computed on clean data), so no re-precalc is needed per point -- only the
# noise_setting in a per-point config copy changes.
#
# Usage: bash scripts/run_coalign_roco_localization_sweep.sh <coalign|roco> [MAX_PARALLEL] [WORKERS]
set -euo pipefail

PIPELINE="${1:?Usage: $0 <coalign|roco> [MAX_PARALLEL]}"
MAX_PARALLEL="${2:-${MAX_PARALLEL:-3}}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
RESULTS_DIR="${ROOT}/outputs/noise_sweep_results/${PIPELINE}/localization_runs"
mkdir -p "${RESULTS_DIR}"

CONDA_ENVS_ROOT="/mnt/external/backup/dhia/conda_envs"
OPENCOOD_PYTHON_BIN="${CONDA_ENVS_ROOT}/opencood-blackwell/bin/python"

case "${PIPELINE}" in
  coalign)
    TREE="${ROOT}/external/CoAlign"
    PYTHON_BIN="${CONDA_ENVS_ROOT}/coalign-blackwell/bin/python"
    CLEAN_CONFIG="${ROOT}/models/Zayd Model_clean_test/config.yaml"
    CHECKPOINT="${ROOT}/models/Zayd Model_clean_test/net_epoch_bestval_at25.pth"
    METHOD_NAME="Zayd Model"
    ;;
  roco)
    TREE="${ROOT}/external/RoCo_merged"
    PYTHON_BIN="${CONDA_ENVS_ROOT}/roco-blackwell/bin/python"
    CLEAN_CONFIG="${ROOT}/models/RoCo_clean_test/config.yaml"
    CHECKPOINT="${ROOT}/models/RoCo_clean_test/net_epoch_bestval_at23.pth"
    METHOD_NAME="RoCo (epoch 23)"
    ;;
  *)
    echo "Unknown pipeline: ${PIPELINE} (expected coalign or roco)" >&2
    exit 1
    ;;
esac

POINTS=(0.2 0.4 0.6 0.8 1.0 1.2 1.4 1.6 1.8 2.0)

run_point() {
  local pos="$1"
  local name="localization_std_${pos}m"
  local model_dir="${ROOT}/models/${PIPELINE}_eval_${name}"
  local log_path="${RESULTS_DIR}/${name}.log"
  local assoc_json="${RESULTS_DIR}/${name}_association.json"

  if [ -s "${assoc_json}" ] && ls "${model_dir}"/eval_*.yaml >/dev/null 2>&1; then
    echo "[skip] ${PIPELINE} ${name}: already have eval_yaml + association json"
    return 0
  fi

  {
    echo "[start] ${PIPELINE} ${name} (position_std=${pos}m)"
    mkdir -p "${model_dir}"
    ln -sf "${CHECKPOINT}" "${model_dir}/$(basename "${CHECKPOINT}")"

    "${PYTHON_BIN}" -c "
import opencood.hypes_yaml.yaml_utils as yaml_utils
hypes = yaml_utils.load_yaml('${CLEAN_CONFIG}', None)
hypes['noise_setting']['add_noise'] = True
hypes['noise_setting']['args'] = {'pos_std': ${pos}, 'rot_std': 0.0, 'pos_mean': 0, 'rot_mean': 0}
yaml_utils.save_yaml(hypes, '${model_dir}/config.yaml')
"

    cd "${TREE}"
    "${PYTHON_BIN}" -u opencood/tools/inference.py \
      --model_dir "${model_dir}" \
      --fusion_method intermediate \
      --save_vis_interval 1000000 \
      --note "localization_${pos}m"

    eval_yaml=$(ls "${model_dir}"/eval_*.yaml | head -1)

    "${PYTHON_BIN}" -u opencood/tools/evaluate_association_diagnostics.py \
      --hypes_yaml "${model_dir}/config.yaml" \
      > "${RESULTS_DIR}/${name}_association.json"

    "${OPENCOOD_PYTHON_BIN}" -c "
import sys
sys.path.insert(0, '${ROOT}/scripts')
from sweep_results import record_coalign_or_roco_run
row = record_coalign_or_roco_run(
    '${PIPELINE}', '${METHOD_NAME}', '${eval_yaml}',
    sweep_dimension='localization_std_m', sweep_value=${pos},
    association_json='${RESULTS_DIR}/${name}_association.json',
)
print('recorded:', row)
"
    echo "[done] ${PIPELINE} ${name}"
  } > "${log_path}" 2>&1
}

running=0
for pos in "${POINTS[@]}"; do
  run_point "${pos}" &
  running=$((running + 1))
  if [ "${running}" -ge "${MAX_PARALLEL}" ]; then
    wait -n
    running=$((running - 1))
  fi
done
wait
echo "All ${PIPELINE} localization-std points done. Logs: ${RESULTS_DIR}/*.log"
