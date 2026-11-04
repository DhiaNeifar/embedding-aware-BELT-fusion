#!/usr/bin/env bash
# Noise sweep on OPV2V, V2X-Real and CooperScene with the same protocol
# (8 sigmas x 20 seeds, Chenyi's full Propose/Verify/Abstain config), AP by
# CAV-count bucket. See scripts/reproduce_multidataset_sweep.py.
#
# Usage: bash scripts/run_multidataset_sweep.sh STEP [WORKERS]
#   v2xreal-cache   GPU, once: detector over V2X-Real test (Chenyi's net_epoch13)
#   v2xreal         V2X-Real sweep, Chenyi's V2X-Real checkpoint + FreeAlign calibration
#   cooperscene     CooperScene sweep, zero-shot transfer of the V2X-Real checkpoint + calibration
#   cooperscene-opv2v   same, transferring the OPV2V checkpoint + calibration instead
#   opv2v           OPV2V through the same driver (bucket table on the identical protocol)
# Each step skips itself if its output already exists.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "${ROOT}"
PY="/mnt/external/backup/dhia/conda_envs/opencood-blackwell/bin/python"
STEP="${1:?step: v2xreal-cache | v2xreal | cooperscene | cooperscene-opv2v | opv2v}"
WORKERS="${2:-20}"

V2X_DET="models/v2xreal/late_fusion/pointpillar"
V2X_FA="models/v2xreal/freealign_calibration.json"
V2X_AF="models/v2xreal/alignformer_boxes_only.pth"
OPV2V_FA="models/opv2v/freealign_calibration.json"
OPV2V_AF="models/opv2v/alignformer_boxes_only.pth"
COOPER_DET_CONFIG="models/cooperscene/late_fusion/pointpillar/config.yaml"
COOPER_PREDS="/home/dhia/cooperscene_preds/pp/TownCooperScene"
OUT="outputs/multidataset_sweep"
mkdir -p "${OUT}"

sweep() {  # sweep OUTPUT_JSON ARGS...
  local output="$1"; shift
  if [[ -f "${output}" ]]; then echo "exists, skipping: ${output}"; return; fi
  "${PY}" scripts/reproduce_multidataset_sweep.py "$@" --seeds 20 --workers "${WORKERS}" --output "${output}"
}

case "${STEP}" in
  v2xreal-cache)
    if [[ -f outputs/v2xreal_geometry_test_cache/manifest.json ]]; then
      echo "cache exists; delete outputs/v2xreal_geometry_test_cache to rebuild"; exit 0
    fi
    "${PY}" scripts/build_v2xreal_geometry_cache.py -- --cache-mode geometry_only \
      --opencood-config "${V2X_DET}/config.yaml" --detector-dir "${V2X_DET}" \
      --data-root /mnt/external/workspace/public/dataset/v2x-real --split test \
      --output-dir outputs/v2xreal_geometry_test_cache --workers 8 --resume
    ;;
  v2xreal)
    sweep "${OUT}/v2xreal_test_20seeds.json" --dataset v2xreal \
      --opencood-config "${V2X_DET}/config.yaml" \
      --data-root /mnt/external/workspace/public/dataset/v2x-real --split test \
      --geometry-cache-dir outputs/v2xreal_geometry_test_cache \
      --frame-table-cache outputs/v2xreal_test_frame_table.pt \
      --freealign-calibration "${V2X_FA}" --alignformer-pva-checkpoint "${V2X_AF}"
    ;;
  cooperscene)
    sweep "${OUT}/cooperscene_test_20seeds_v2xrealckpt.json" --dataset cooperscene \
      --opencood-config "${COOPER_DET_CONFIG}" --data-root "${COOPER_PREDS}" --split test \
      --geometry-cache-dir outputs/cooperscene_geometry_test_cache \
      --frame-table-cache outputs/cooperscene_test_frame_table.pt \
      --freealign-calibration "${V2X_FA}" --alignformer-pva-checkpoint "${V2X_AF}"
    ;;
  cooperscene-opv2v)
    sweep "${OUT}/cooperscene_test_20seeds_opv2vckpt.json" --dataset cooperscene \
      --opencood-config "${COOPER_DET_CONFIG}" --data-root "${COOPER_PREDS}" --split test \
      --geometry-cache-dir outputs/cooperscene_geometry_test_cache \
      --frame-table-cache outputs/cooperscene_test_frame_table.pt \
      --freealign-calibration "${OPV2V_FA}" --alignformer-pva-checkpoint "${OPV2V_AF}"
    ;;
  opv2v)
    sweep "${OUT}/opv2v_test_20seeds.json" --dataset opv2v \
      --opencood-config configs/pointpillar_late_fusion_r140.yaml \
      --data-root /mnt/external/workspace/public/dataset/opv2v --split test \
      --geometry-cache-dir outputs/r140_geometry_test_cache \
      --frame-table-cache outputs/opv2v_frame_table.pt \
      --freealign-calibration "${OPV2V_FA}" --alignformer-pva-checkpoint "${OPV2V_AF}"
    ;;
  *) echo "unknown step ${STEP}"; exit 2 ;;
esac
