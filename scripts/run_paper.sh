#!/usr/bin/env bash
# Every run behind the paper's exhibits (CVPR/sec/9_ledger.tex), OPV2V and V2X-Real. Nothing here reads the
# noise level: only the simulator in alignformer_ext/propose_cache.py does. Every step skips or resumes what exists.
#
# Usage: bash scripts/run_paper.sh STEP [WORKERS]
#
# Data, built once (already on disk):
#   v2xreal-cache, v2xreal-val-cache   GPU: V2X-Real late-fusion detector over test / val (per-agent boxes)
#   v2xreal-table, v2xreal-val-table   CPU: poses + ground truth per frame (no LiDAR read)
#   opv2v-propose, v2xreal-propose     CPU: Propose + FreeAlign per (sigma, seed), 8 sigmas x 20 seeds
#   v2xreal-val-propose                CPU: same on V2X-Real val, sigmas 0 / 0.4 / 1 x 5 seeds
#   odometry                           CPU: KISS-ICP on every vehicle's own scans (kiss_icp_env)
# Selection, V2X-Real validation only:
#   select                             assumed odometry accuracy and window length
# Exhibits:
#   opv2v-late, v2xreal-late           T3: late fusion, every sigma and seed
#   opv2v-async-cache, v2xreal-async-cache   delayed messages, 0 / 100 / 200 / 400 ms
#   opv2v-delay, v2xreal-delay         T6: late fusion under unknown delay
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "${ROOT}"
PY="/mnt/external/backup/dhia/conda_envs/opencood-blackwell/bin/python"
KISS="outputs/kiss_icp_env/bin/python"
DATA="/mnt/external/workspace/public/dataset"
STEP="${1:?step, see the header of this file}"
WORKERS="${2:-20}"

OPV2V_CFG="configs/pointpillar_late_fusion_r140.yaml"
OPV2V_AF="models/opv2v/alignformer_boxes_only.pth"
OPV2V_FA="models/opv2v/freealign_calibration.json"
V2X_DET="models/v2xreal/late_fusion/pointpillar"
V2X_AF="models/v2xreal/alignformer_boxes_only.pth"
V2X_FA="models/v2xreal/freealign_calibration.json"
V2X_RANGE="102.4 51.2"
MAIN="alignformer_c alignformer_tk"
mkdir -p outputs/paper

v2xreal_cache() {  # split output
  if [[ -f "$2/manifest.json" ]]; then echo "exists: $2"; return; fi
  "${PY}" scripts/build_v2xreal_geometry_cache.py -- --cache-mode geometry_only \
    --opencood-config "${V2X_DET}/config.yaml" --detector-dir "${V2X_DET}" \
    --data-root "${DATA}/v2x-real" --split "$1" --output-dir "$2" --workers 8 --resume
}
v2xreal_table() {  # split geometry-cache frame-table
  "${PY}" scripts/reproduce_multidataset_sweep.py --dataset v2xreal --frame-table-only \
    --opencood-config "${V2X_DET}/config.yaml" --data-root "${DATA}/v2x-real" --split "$1" \
    --geometry-cache-dir "$2" --frame-table-cache "$3" \
    --freealign-calibration "${V2X_FA}" --alignformer-pva-checkpoint "${V2X_AF}"
}

case "${STEP}" in
  v2xreal-cache)     v2xreal_cache test outputs/v2xreal_geometry_test_cache ;;
  v2xreal-val-cache) v2xreal_cache val outputs/v2xreal_geometry_val_cache ;;
  v2xreal-table)     v2xreal_table test outputs/v2xreal_geometry_test_cache outputs/v2xreal_test_frame_table.pt ;;
  v2xreal-val-table) v2xreal_table val outputs/v2xreal_geometry_val_cache outputs/v2xreal_val_frame_table.pt ;;
  opv2v-propose)
    "${PY}" scripts/build_propose_cache.py --frame-table outputs/opv2v_frame_table.pt --checkpoint "${OPV2V_AF}" \
      --freealign-calibration "${OPV2V_FA}" --output-dir outputs/propose_cache_opv2v --workers "${WORKERS}"
    ;;
  v2xreal-propose)
    "${PY}" scripts/build_propose_cache.py --frame-table outputs/v2xreal_test_frame_table.pt --checkpoint "${V2X_AF}" \
      --freealign-calibration "${V2X_FA}" --output-dir outputs/propose_cache_v2xreal --workers "${WORKERS}"
    ;;
  v2xreal-val-propose)
    "${PY}" scripts/build_propose_cache.py --frame-table outputs/v2xreal_val_frame_table.pt --checkpoint "${V2X_AF}" \
      --freealign-calibration "${V2X_FA}" --sigmas 0 0.4 1 --seeds 5 \
      --output-dir outputs/propose_cache_v2xreal_val --workers "${WORKERS}"
    ;;
  odometry)
    "${KISS}" scripts/build_odometry_table.py --split-dir "${DATA}/v2x-real/val" --output outputs/odometry/v2xreal_val.npz --workers "${WORKERS}"
    "${KISS}" scripts/build_odometry_table.py --split-dir "${DATA}/v2x-real/test" --output outputs/odometry/v2xreal_test.npz --workers "${WORKERS}"
    "${KISS}" scripts/build_odometry_table.py --split-dir "${DATA}/opv2v/test" --output outputs/odometry/opv2v_test.npz --workers "${WORKERS}"
    ;;
  select)
    "${PY}" scripts/alignformer_variants_sweep.py --cache-dir outputs/propose_cache_v2xreal_val \
      --frame-table outputs/v2xreal_val_frame_table.pt --checkpoint "${V2X_AF}" \
      --opencood-config "${V2X_DET}/config.yaml" --eval-range ${V2X_RANGE} \
      --odometry outputs/odometry/v2xreal_val.npz --geometry-cache outputs/v2xreal_geometry_val_cache \
      --variants alignformer_c alignformer_odo1 alignformer_odo4 alignformer_odo8 \
                 alignformer_tk_w3 alignformer_tk_w10 alignformer_tk alignformer_tk_w50 \
      --workers "${WORKERS}" --output outputs/paper/select_v2xreal_val.json
    ;;
  opv2v-late)
    "${PY}" scripts/alignformer_variants_sweep.py --cache-dir outputs/propose_cache_opv2v \
      --frame-table outputs/opv2v_frame_table.pt --checkpoint "${OPV2V_AF}" --opencood-config "${OPV2V_CFG}" \
      --odometry outputs/odometry/opv2v_test.npz --geometry-cache outputs/r140_geometry_test_cache \
      --variants ${MAIN} --workers "${WORKERS}" --output outputs/paper/late_opv2v.json
    ;;
  v2xreal-late)
    "${PY}" scripts/alignformer_variants_sweep.py --cache-dir outputs/propose_cache_v2xreal \
      --frame-table outputs/v2xreal_test_frame_table.pt --checkpoint "${V2X_AF}" \
      --opencood-config "${V2X_DET}/config.yaml" --eval-range ${V2X_RANGE} \
      --odometry outputs/odometry/v2xreal_test.npz --geometry-cache outputs/v2xreal_geometry_test_cache \
      --variants ${MAIN} --workers "${WORKERS}" --output outputs/paper/late_v2xreal.json
    ;;
  opv2v-async-cache)
    "${PY}" scripts/build_async_cache.py --delays 0 1 2 4 --compensate --output-dir outputs/async_opv2v --workers "${WORKERS}"
    ;;
  v2xreal-async-cache)
    "${PY}" scripts/build_async_cache.py --frame-table outputs/v2xreal_test_frame_table.pt \
      --geometry-cache outputs/v2xreal_geometry_test_cache --checkpoint "${V2X_AF}" \
      --freealign-calibration "${V2X_FA}" --delays 0 1 2 4 --compensate \
      --output-dir outputs/async_v2xreal --workers "${WORKERS}"
    ;;
  opv2v-delay)
    "${PY}" scripts/latency_sweep.py --async-dir outputs/async_opv2v --odometry outputs/odometry/opv2v_test.npz \
      --workers "${WORKERS}" \
      --output outputs/paper/delay_opv2v.json
    ;;
  v2xreal-delay)
    "${PY}" scripts/latency_sweep.py --async-dir outputs/async_v2xreal --frame-table outputs/v2xreal_test_frame_table.pt \
      --geometry-cache outputs/v2xreal_geometry_test_cache --checkpoint "${V2X_AF}" --freealign-calibration "${V2X_FA}" \
      --opencood-config "${V2X_DET}/config.yaml" --eval-range ${V2X_RANGE} \
      --odometry outputs/odometry/v2xreal_test.npz --workers "${WORKERS}" --output outputs/paper/delay_v2xreal.json
    ;;
  *) echo "unknown step ${STEP}" >&2; exit 2 ;;
esac
