#!/usr/bin/env bash
# AlignFormer variants (C++ Verify) on V2X-Real and CooperScene: same 8 sigmas x 20 seeds as OPV2V.
#
# Usage: bash scripts/run_variants_other_datasets.sh STEP [WORKERS]
#   v2xreal-cache        GPU, once: Chenyi's V2X-Real detector over the test split
#   v2xreal-table        CPU, once: poses + ground truth (no LiDAR read)
#   v2xreal-propose      Propose + FreeAlign cache, Chenyi's V2X-Real checkpoint + calibration
#   v2xreal-sweep        score uncorrected / oracle / FreeAlign / alignformer / alignformer_c
#   cooperscene-propose  same, zero-shot transfer of the V2X-Real checkpoint + calibration
#   cooperscene-sweep
#   v2xreal-multiway-sweep  joint (multi-marginal Sinkhorn) matching of all agents per frame (alignformer_ext/multiway.py)
#   v2xreal-async-cache    stale CAV messages (0 / 100 / 200 / 400 ms) with sender velocities (build_async_cache.py)
#   v2xreal-latency-sweep  ideas A + D under delay: latency estimated from boxes, staleness-aware scores (latency_sweep.py)
#   cooperscene-cal-sweep  FreeAlign and AlignFormer with the dataset's scan-vs-pose offsets as calibration
#                        (outputs/calibration/cooperscene_scan_offsets.json, scripts/measure_scan_pose_offsets.py)
# Validation splits (select alignformer_tk's window before reporting test; OPV2V has no independent val split):
#   v2xreal-val-cache    GPU, once: detector over V2X-Real val
#   v2xreal-val-table    CPU, once
#   v2xreal-val-propose  Propose + FreeAlign cache, sigmas 0 / 0.4 / 1 x 5 seeds
#   v2xreal-val-select   window 3 / 10 / 30 / 50 on val
#   cooperscene-val-propose, cooperscene-val-select   same on CooperScene validate
# Every step resumes / skips what already exists.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "${ROOT}"
PY="/mnt/external/backup/dhia/conda_envs/opencood-blackwell/bin/python"
STEP="${1:?step}"
WORKERS="${2:-20}"

V2X_DET="models/v2xreal/late_fusion/pointpillar"
V2X_FA="models/v2xreal/freealign_calibration.json"
V2X_AF="models/v2xreal/alignformer_boxes_only.pth"
VARIANTS="alignformer_tk alignformer_tg alignformer_tgo freealign_tk"
SELECT="alignformer_c alignformer_tk_w3 alignformer_tk_w10 alignformer_tk alignformer_tk_w50"
COOPER_DET_CONFIG="models/cooperscene/late_fusion/pointpillar/config.yaml"
mkdir -p outputs/variants

case "${STEP}" in
  v2xreal-cache)
    bash scripts/run_multidataset_sweep.sh v2xreal-cache
    ;;
  v2xreal-table)
    "${PY}" scripts/reproduce_multidataset_sweep.py --dataset v2xreal --frame-table-only \
      --opencood-config "${V2X_DET}/config.yaml" \
      --data-root /mnt/external/workspace/public/dataset/v2x-real --split test \
      --geometry-cache-dir outputs/v2xreal_geometry_test_cache \
      --frame-table-cache outputs/v2xreal_test_frame_table.pt \
      --freealign-calibration "${V2X_FA}" --alignformer-pva-checkpoint "${V2X_AF}"
    ;;
  v2xreal-propose)
    "${PY}" scripts/build_propose_cache.py --frame-table outputs/v2xreal_test_frame_table.pt \
      --checkpoint "${V2X_AF}" --freealign-calibration "${V2X_FA}" \
      --output-dir outputs/propose_cache_v2xreal --workers "${WORKERS}" --cavcav
    ;;
  v2xreal-sweep)
    "${PY}" scripts/alignformer_variants_sweep.py --cache-dir outputs/propose_cache_v2xreal \
      --frame-table outputs/v2xreal_test_frame_table.pt --checkpoint "${V2X_AF}" \
      --opencood-config "${V2X_DET}/config.yaml" --eval-range 102.4 51.2 \
      --variants ${VARIANTS} --workers "${WORKERS}" --output outputs/variants/v2xreal_tgo_20seeds.json
    ;;
  cooperscene-propose)
    "${PY}" scripts/build_propose_cache.py --frame-table outputs/cooperscene_test_frame_table.pt \
      --checkpoint "${V2X_AF}" --freealign-calibration "${V2X_FA}" \
      --output-dir outputs/propose_cache_cooperscene --workers "${WORKERS}" --cavcav
    ;;
  cooperscene-sweep)
    "${PY}" scripts/alignformer_variants_sweep.py --cache-dir outputs/propose_cache_cooperscene \
      --frame-table outputs/cooperscene_test_frame_table.pt --checkpoint "${V2X_AF}" \
      --opencood-config "${COOPER_DET_CONFIG}" --eval-range 102.4 40 \
      --variants ${VARIANTS} --workers "${WORKERS}" --output outputs/variants/cooperscene_tgo_20seeds.json
    ;;
  v2xreal-multiway-sweep)
    "${PY}" scripts/alignformer_variants_sweep.py --cache-dir outputs/propose_cache_v2xreal \
      --frame-table outputs/v2xreal_test_frame_table.pt --checkpoint "${V2X_AF}" \
      --opencood-config "${V2X_DET}/config.yaml" --eval-range 102.4 51.2 \
      --variants alignformer_mm alignformer_tg alignformer_tg_mm alignformer_tg_mmt alignformer_tg_mmj \
      --workers "${WORKERS}" --output outputs/variants/v2xreal_multiway_20seeds.json
    ;;
  v2xreal-async-cache)
    "${PY}" scripts/build_async_cache.py --frame-table outputs/v2xreal_test_frame_table.pt \
      --geometry-cache outputs/v2xreal_geometry_test_cache --checkpoint "${V2X_AF}" \
      --freealign-calibration "${V2X_FA}" --delays 0 1 2 4 --compensate \
      --output-dir outputs/async_v2xreal --workers "${WORKERS}"
    ;;
  v2xreal-latency-sweep)
    "${PY}" scripts/latency_sweep.py --async-dir outputs/async_v2xreal --frame-table outputs/v2xreal_test_frame_table.pt \
      --geometry-cache outputs/v2xreal_geometry_test_cache --checkpoint "${V2X_AF}" \
      --freealign-calibration "${V2X_FA}" \
      --opencood-config "${V2X_DET}/config.yaml" --eval-range 102.4 51.2 \
      --workers "${WORKERS}" --output outputs/variants/v2xreal_latency_20seeds.json
    ;;
  cooperscene-cal-sweep)
    "${PY}" scripts/alignformer_variants_sweep.py --cache-dir outputs/propose_cache_cooperscene \
      --frame-table outputs/cooperscene_test_frame_table.pt --checkpoint "${V2X_AF}" \
      --opencood-config "${COOPER_DET_CONFIG}" --eval-range 102.4 40 \
      --variants alignformer_tg alignformer_tg_cal freealign_cal \
      --calibration outputs/calibration/cooperscene_scan_offsets.json \
      --workers "${WORKERS}" --output outputs/variants/cooperscene_cal_20seeds.json
    ;;
  v2xreal-val-cache)
    if [[ -f outputs/v2xreal_geometry_val_cache/manifest.json ]]; then
      echo "cache exists; delete outputs/v2xreal_geometry_val_cache to rebuild"; exit 0
    fi
    "${PY}" scripts/build_v2xreal_geometry_cache.py -- --cache-mode geometry_only \
      --opencood-config "${V2X_DET}/config.yaml" --detector-dir "${V2X_DET}" \
      --data-root /mnt/external/workspace/public/dataset/v2x-real --split val \
      --output-dir outputs/v2xreal_geometry_val_cache --workers 8 --resume
    ;;
  v2xreal-val-table)
    "${PY}" scripts/reproduce_multidataset_sweep.py --dataset v2xreal --frame-table-only \
      --opencood-config "${V2X_DET}/config.yaml" \
      --data-root /mnt/external/workspace/public/dataset/v2x-real --split val \
      --geometry-cache-dir outputs/v2xreal_geometry_val_cache \
      --frame-table-cache outputs/v2xreal_val_frame_table.pt \
      --freealign-calibration "${V2X_FA}" --alignformer-pva-checkpoint "${V2X_AF}"
    ;;
  v2xreal-val-propose)
    "${PY}" scripts/build_propose_cache.py --frame-table outputs/v2xreal_val_frame_table.pt \
      --checkpoint "${V2X_AF}" --freealign-calibration "${V2X_FA}" --sigmas 0 0.4 1 --seeds 5 \
      --output-dir outputs/propose_cache_v2xreal_val --workers "${WORKERS}"
    ;;
  v2xreal-val-select)
    "${PY}" scripts/alignformer_variants_sweep.py --cache-dir outputs/propose_cache_v2xreal_val \
      --frame-table outputs/v2xreal_val_frame_table.pt --checkpoint "${V2X_AF}" \
      --opencood-config "${V2X_DET}/config.yaml" --eval-range 102.4 51.2 \
      --variants ${SELECT} --workers "${WORKERS}" --output outputs/variants/v2xreal_val_select.json
    ;;
  cooperscene-val-propose)
    "${PY}" scripts/build_propose_cache.py --frame-table outputs/cooperscene_validate_frame_table.pt \
      --checkpoint "${V2X_AF}" --freealign-calibration "${V2X_FA}" --sigmas 0 0.4 1 --seeds 5 \
      --output-dir outputs/propose_cache_cooperscene_val --workers "${WORKERS}"
    ;;
  cooperscene-val-select)
    "${PY}" scripts/alignformer_variants_sweep.py --cache-dir outputs/propose_cache_cooperscene_val \
      --frame-table outputs/cooperscene_validate_frame_table.pt --checkpoint "${V2X_AF}" \
      --opencood-config "${COOPER_DET_CONFIG}" --eval-range 102.4 40 \
      --variants ${SELECT} --workers "${WORKERS}" --output outputs/variants/cooperscene_val_select.json
    ;;
  *) echo "unknown step ${STEP}"; exit 2 ;;
esac
