#!/usr/bin/env bash
# T5: noise the method never saw, late fusion, 5 seeds, both datasets. CPU only. Every step runs in the foreground
# with its own tqdm bar. Caches that exist are skipped, so a rerun resumes.
#   drift / bias / mixed at sigma = 1 (alignformer_ext/noise_kinds.py), beyond training = benchmark noise at sigma = 5
# Output: outputs/paper/noise_<dataset>_<kind>.json; the paper's T5 and F8 fill in when they appear.
set -euo pipefail
cd "$(dirname "$0")/.."
P=/mnt/external/backup/dhia/conda_envs/opencood-blackwell/bin/python

declare -A TABLE=([opv2v]=outputs/opv2v_frame_table.pt [v2xreal]=outputs/v2xreal_test_frame_table.pt)
declare -A AF=([opv2v]=models/opv2v/alignformer_boxes_only.pth [v2xreal]=models/v2xreal/alignformer_boxes_only.pth)
declare -A FA=([opv2v]=models/opv2v/freealign_calibration.json [v2xreal]=models/v2xreal/freealign_calibration.json)
declare -A CFG=([opv2v]=configs/pointpillar_late_fusion_r140.yaml [v2xreal]=models/v2xreal/late_fusion/pointpillar/config.yaml)
declare -A RANGE=([opv2v]="140 40" [v2xreal]="102.4 51.2")
declare -A GEO=([opv2v]=outputs/r140_geometry_test_cache [v2xreal]=outputs/v2xreal_geometry_test_cache)

for ds in opv2v v2xreal; do
  for kind in drift bias mixed beyond; do
    cache="outputs/propose_cache_${ds}_${kind}"
    if [[ "${kind}" == beyond ]]; then noise=(--sigmas 5); else noise=(--sigmas 1 --noise-kind "${kind}"); fi
    echo "=== ${ds} ${kind}: Propose cache"
    "${P}" scripts/build_propose_cache.py --frame-table "${TABLE[$ds]}" --checkpoint "${AF[$ds]}" \
      --freealign-calibration "${FA[$ds]}" "${noise[@]}" --seeds 5 --output-dir "${cache}" \
      --odometry "outputs/odometry/${ds}_test.npz" --geometry-cache "${GEO[$ds]}" --workers 20
    echo "=== ${ds} ${kind}: late fusion"
    "${P}" scripts/alignformer_variants_sweep.py --cache-dir "${cache}" --frame-table "${TABLE[$ds]}" \
      --checkpoint "${AF[$ds]}" --opencood-config "${CFG[$ds]}" --eval-range ${RANGE[$ds]} \
      --odometry "outputs/odometry/${ds}_test.npz" --geometry-cache "${GEO[$ds]}" \
      --variants alignformer_c alignformer_tk --workers 20 --output "outputs/paper/noise_${ds}_${kind}.json"
  done
done
echo "T5 done"
