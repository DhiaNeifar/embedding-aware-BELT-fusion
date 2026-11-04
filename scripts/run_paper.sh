#!/usr/bin/env bash
# Runs behind the paper's exhibits, one step at a time (CVPR/sec/9_ledger.tex). Nothing here reads the noise level.
#   odometry   KISS-ICP tables for the test and validation splits (CPU, kiss_icp_env)
#   select     the odometry accuracy to assume, on V2X-Real validation only
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "${ROOT}"
PY="/mnt/external/backup/dhia/conda_envs/opencood-blackwell/bin/python"
KISS="outputs/kiss_icp_env/bin/python"
DATA="/mnt/external/workspace/public/dataset"
STEP="${1:?step: odometry | select}"
WORKERS="${2:-20}"

case "${STEP}" in
  odometry)
    "${KISS}" scripts/build_odometry_table.py --split-dir "${DATA}/v2x-real/val" --output outputs/odometry/v2xreal_val.npz --workers "${WORKERS}"
    "${KISS}" scripts/build_odometry_table.py --split-dir "${DATA}/v2x-real/test" --output outputs/odometry/v2xreal_test.npz --workers "${WORKERS}"
    "${KISS}" scripts/build_odometry_table.py --split-dir "${DATA}/opv2v/test" --output outputs/odometry/opv2v_test.npz --workers "${WORKERS}"
    ;;
  select)
    "${PY}" scripts/alignformer_variants_sweep.py --cache-dir outputs/propose_cache_v2xreal_val \
      --frame-table outputs/v2xreal_val_frame_table.pt --checkpoint models/v2xreal/alignformer_boxes_only.pth \
      --opencood-config models/v2xreal/late_fusion/pointpillar/config.yaml --eval-range 102.4 51.2 \
      --odometry outputs/odometry/v2xreal_val.npz --geometry-cache outputs/v2xreal_geometry_val_cache \
      --variants alignformer_c alignformer_odo1 alignformer_odo4 alignformer_odo8 \
      --workers "${WORKERS}" --output outputs/paper/select_odometry_v2xreal_val.json
    ;;
  *) echo "unknown step ${STEP}" >&2; exit 2 ;;
esac
