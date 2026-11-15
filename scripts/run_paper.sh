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
#   odometry                           CPU: KISS-ICP on V2X-Real scans; OPV2V from its recorded poses (kiss_icp_env)
# Selection, V2X-Real validation only:
#   select                             assumed odometry accuracy and window length
# Exhibits:
#   opv2v-late, v2xreal-late           T3: late fusion, every sigma and seed
#   opv2v-poses                        CPU: corrected poses of all four plug-ins (FreeAlign, CoAlign, RoCo, AlignFormer)
#   v2xreal-poses                      CPU: same for V2X-Real (AlignFormer, FreeAlign; CoAlign and RoCo need stage-1 first)
#   v2xreal-copy                       disk: V2X-Real train + val to the NVMe drive (~100 GB), for stage-1 training
#   opv2v-copy                         disk: OPV2V train to the NVMe drive (~101 GB), for the OPV2V stage-1 detector
#   opv2v-stage1                       GPU: CoAlign / RoCo stage-1 detector on OPV2V (same recipe; last epoch is used)
#   opv2v-stage1-boxes                 GPU: its boxes + uncertainty on OPV2V test (minutes)
#   opv2v-poses-pgo                    CPU: CoAlign and RoCo OPV2V poses from those boxes, then late fusion (4 plug-ins)
#   opv2v-hosts-stage1                 GPU: the 7 OPV2V networks, CoAlign / RoCo conditions only, merged into hosts/
#   late-20seeds                       CPU: late fusion, all four plug-ins, 20 seeds, both datasets
#   v2xreal-hosts                      GPU, T1: AttFuse / F-Cooper on V2X-Real (QuantV2X) x 8 sigmas, seed 0
#   v2xreal-stage1                     GPU: CoAlign / RoCo stage-1 detector trained on V2X-Real (CoAlign's code, 30 epochs)
#   v2xreal-stage1-boxes               GPU: its boxes + uncertainty on V2X-Real test (minutes)
#   v2xreal-poses-pgo                  CPU: CoAlign and RoCo poses on V2X-Real, then late fusion with all four plug-ins
#   opv2v-dryrun                       GPU: every OPV2V network x every pose condition on 50 frames (minutes)
#   opv2v-timing                       GPU: all 7 OPV2V networks, all frames, sigma 1, seed 0 (real timing, ~1 h)
#   opv2v-hosts                        GPU, T1: 7 OPV2V networks x 8 sigmas, seed 0, true / bare / four plug-ins
#   delay                              CPU, T6: delayed-message caches + late fusion under unknown delay, both datasets
#   revision-nov12                     CPU: assessment 2026-11-11 runs (pooled baselines; T6 with clean staleness table + oracle row)
#   revision-5draws                    GPU+CPU: five noise draws for T1 (12 hosts) and T6, T4 with 5 draws, then all tables
#   sigma-std                          GPU+CPU: the sigma standard 0/0.5/1/1.5/2, every missing run, into tagged files
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "${ROOT}"
ENVS="/mnt/external/backup/dhia/conda_envs"
PY="${ENVS}/opencood-blackwell/bin/python"
KISS="outputs/kiss_icp_env/bin/python"
DATA="/mnt/external/workspace/public/dataset"
NVME_V2X="/media/dhia/workingspace/Public/v2x-real"   # NVMe copy of V2X-Real train/val for stage-1 training
NVME_OPV2V="/media/dhia/workingspace/Public/opv2v"    # NVMe copy of OPV2V test (and train, for stage-1 training)
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

# Progress of the networks that run in the background (two at a time, output in their own log files): every 5 s,
# one line per network in place, the newest progress bar from its log (or "done" / "FAILED" once it ended).
RUN_LOGS=()
DRAWN=0
draw_progress() {
  local f line width
  width=$(( $(tput cols 2>/dev/null || echo 120) - 1 ))
  (( DRAWN > 0 )) && printf '\033[%dA' "${DRAWN}"
  for f in "${RUN_LOGS[@]}"; do
    line="$(tail -c 3000 "${f}" 2>/dev/null | tr '\r' '\n' | grep -E '%\||^(done|FAILED|exists)' | tail -1 || true)"
    printf '\033[2K%s\n' "$(echo "${line:-$(basename "${f}" .log): starting}" | cut -c1-"${width}")"
  done
  DRAWN=${#RUN_LOGS[@]}
}
progress_until() {  # wait, showing progress, until at most $1 background jobs run (the hardware logger is one)
  while (( $(jobs -rp | wc -l) > $1 )); do
    draw_progress
    sleep 5
  done
  if (( $1 <= 1 )); then draw_progress; fi
  return 0
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
    # V2X-Real: real LiDAR odometry. OPV2V: the simulator's recorded poses (exact relative motion, stated in the paper)
    "${KISS}" scripts/build_odometry_table.py --source kiss-icp --split-dir "${DATA}/v2x-real/val" --output outputs/odometry/v2xreal_val.npz --workers "${WORKERS}"
    "${KISS}" scripts/build_odometry_table.py --source kiss-icp --split-dir "${DATA}/v2x-real/test" --output outputs/odometry/v2xreal_test.npz --workers "${WORKERS}"
    "${KISS}" scripts/build_odometry_table.py --source recorded --split-dir "${DATA}/opv2v/test" --output outputs/odometry/opv2v_test.npz --workers "${WORKERS}"
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
  opv2v-poses)
    # all four plug-ins, sigmas 0 / .2 / .4 / .6 / .8 / 1 / 1.5 / 2, 5 seeds (CoAlign and RoCo in their own environments)
    "${PY}" scripts/export_poses.py --dataset opv2v --workers "${WORKERS}" --overwrite
    "${ENVS}/coalign-blackwell/bin/python" scripts/export_poses_pgo.py --method coalign --dataset opv2v --workers "${WORKERS}" --overwrite
    "${ENVS}/roco-blackwell/bin/python" scripts/export_poses_pgo.py --method roco --dataset opv2v --workers "${WORKERS}" --overwrite
    ;;
  v2xreal-poses) "${PY}" scripts/export_poses.py --dataset v2xreal --workers "${WORKERS}" --overwrite ;;
  v2xreal-copy)
    # V2X-Real train + val (~100 GB) from the HDD to the NVMe drive: stage-1 training reads every agent's scan per
    # sample, and on the HDD that is 1.2 s of a 1.3 s iteration. Same files; test stays where it is.
    mkdir -p "${NVME_V2X}"
    rsync -a --info=progress2 "${DATA}/v2x-real/train" "${DATA}/v2x-real/val" "${NVME_V2X}/"
    ;;
  opv2v-copy)
    # OPV2V train (~101 GB) from the HDD to the NVMe drive, for the OPV2V stage-1 detector (CoAlign / RoCo).
    # test is already there. OPV2V has no validate split on this machine (validate -> test), so none is copied.
    mkdir -p "${NVME_OPV2V}"
    rsync -a --info=progress2 "${DATA}/opv2v/train" "${NVME_OPV2V}/"
    ;;
  opv2v-stage1)
    # CoAlign / RoCo stage 1 on OPV2V: same recipe as V2X-Real, OPV2V data settings, from the NVMe copy of train.
    # Refuses to start on an incomplete copy. The checkpoint used later is the last epoch (no OPV2V validation set).
    want=$(find "${DATA}/opv2v/train" -type f | wc -l); have=$(find "${NVME_OPV2V}/train" -type f 2>/dev/null | wc -l)
    if [[ "${have}" != "${want}" ]]; then echo "OPV2V train copy has ${have} of ${want} files: run opv2v-copy first"; exit 1; fi
    bash scripts/log_hardware.sh outputs/paper/stage1_opv2v_hardware.log &
    LOGGER=$!
    trap 'kill ${LOGGER} 2>/dev/null' EXIT
    PYTHONPATH=src OMP_NUM_THREADS=1 "${ENVS}/coalign-blackwell/bin/python" scripts/coalign_v2xreal.py train \
      --hypes-yaml models/opv2v/intermediate_fusion/coalign_stage1/stage1_pointpillar_uncertainty.yaml --workers 16
    ;;
  opv2v-stage1-boxes)
    # its boxes + uncertainty on OPV2V test, from the LAST epoch of the newest OPV2V stage-1 run (minutes)
    dir="$(ls -dt external/CoAlign/opencood/logs/opv2v_point_pillar_uncertainty_late_fusion_* | head -1)"
    echo "stage-1 run: ${dir}"
    PYTHONPATH=src "${ENVS}/coalign-blackwell/bin/python" scripts/coalign_v2xreal.py boxes --dataset opv2v \
      --stage1-dir "${dir}" --checkpoint "${dir}/net_epoch29.pth" --split test
    ;;
  opv2v-poses-pgo)
    # CoAlign and RoCo poses on OPV2V from our stage-1 boxes, then late fusion with all four plug-ins.
    # The previous files (labmate's stage-1 boxes) are kept in outputs/backups/ first.
    mkdir -p outputs/backups/opv2v_labmate_stage1
    for m in coalign roco; do
      [[ -d outputs/backups/opv2v_labmate_stage1/${m} ]] || cp -r outputs/poses/opv2v/${m} outputs/backups/opv2v_labmate_stage1/${m}
    done
    cp -n outputs/paper/late_opv2v_plugins.json outputs/backups/opv2v_labmate_stage1/ 2>/dev/null || true
    "${ENVS}/coalign-blackwell/bin/python" scripts/export_poses_pgo.py --method coalign --dataset opv2v --workers "${WORKERS}" --overwrite
    "${ENVS}/roco-blackwell/bin/python" scripts/export_poses_pgo.py --method roco --dataset opv2v --workers "${WORKERS}" --overwrite
    "${PY}" scripts/late_from_poses.py --dataset opv2v --workers "${WORKERS}" --output outputs/paper/late_opv2v_plugins.json
    ;;
  opv2v-hosts-stage1)
    # The 7 OPV2V networks again, CoAlign and RoCo conditions only (with true / bare, which must come out identical),
    # one pass per network, two networks at a time; then swapped into outputs/paper/hosts (originals backed up).
    mkdir -p outputs/paper/hosts_stage1
    bash scripts/log_hardware.sh outputs/paper/hosts_stage1/hardware.log &
    LOGGER=$!
    trap 'kill ${LOGGER} 2>/dev/null' EXIT
    run_stage1_host() {
      local host="$1" sigma missing=0
      for sigma in 0 0.2 0.4 0.6 0.8 1 1.5 2; do
        [[ -f "outputs/paper/hosts_stage1/opv2v_${host}_sigma_${sigma}.json" ]] || missing=1
      done
      if (( ! missing )); then echo "exists, skipping: ${host} (all 8 sigmas)" > "outputs/paper/hosts_stage1/opv2v_${host}.log"; return; fi
      OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "${PY}" scripts/run_host.py --dataset opv2v --host "${host}" \
        --plugins coalign roco --sigmas 0 0.2 0.4 0.6 0.8 1 1.5 2 --seeds 1 --workers $(( WORKERS / 2 )) \
        --output "outputs/paper/hosts_stage1/opv2v_${host}_sigma_{sigma}.json" > "outputs/paper/hosts_stage1/opv2v_${host}.log" 2>&1 \
        && echo "done: ${host}" >> "outputs/paper/hosts_stage1/opv2v_${host}.log" || echo "FAILED: ${host} (see outputs/paper/hosts_stage1/opv2v_${host}.log)" >> "outputs/paper/hosts_stage1/opv2v_${host}.log"
    }
    for host in cobevt v2xvit attfuse coalign fcooper v2vam where2comm; do
      RUN_LOGS+=("outputs/paper/hosts_stage1/opv2v_${host}.log")
      run_stage1_host "${host}" &
      progress_until 2   # the hardware logger + two networks
    done
    progress_until 1   # until only the hardware logger is left
    "${PY}" scripts/merge_host_plugins.py --new outputs/paper/hosts_stage1 --main outputs/paper/hosts \
      --backup outputs/paper/hosts_labmate_stage1 --plugins coalign roco
    ;;
  v2xreal-hosts)
    # T1 on V2X-Real, seed 0: AttFuse, F-Cooper, V2X-ViT (run in QuantV2X, external/QuantV2X; weights load exactly,
    # perfect-pose AP reproduces QuantV2X's own evaluation) x 8 sigmas x true / bare / four plug-ins. One pass per
    # network, one file per (network, sigma), two networks at a time; a network with all 8 files is skipped.
    mkdir -p outputs/paper/hosts_v2xreal
    bash scripts/log_hardware.sh outputs/paper/hosts_v2xreal/hardware.log &
    LOGGER=$!
    trap 'kill ${LOGGER} 2>/dev/null' EXIT
    run_v2x_host() {
      local host="$1" sigma missing=0
      for sigma in 0 0.2 0.4 0.6 0.8 1 1.5 2; do
        [[ -f "outputs/paper/hosts_v2xreal/v2xreal_${host}_sigma_${sigma}.json" ]] || missing=1
      done
      if (( ! missing )); then echo "exists, skipping: ${host} (all 8 sigmas)" > "outputs/paper/hosts_v2xreal/v2xreal_${host}.log"; return; fi
      PYTHONPATH=src OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "${ENVS}/coalign-blackwell/bin/python" scripts/run_host_qv2x.py \
        --host "${host}" --sigmas 0 0.2 0.4 0.6 0.8 1 1.5 2 --seeds 1 --workers $(( WORKERS / 2 )) \
        --output "outputs/paper/hosts_v2xreal/v2xreal_${host}_sigma_{sigma}.json" > "outputs/paper/hosts_v2xreal/v2xreal_${host}.log" 2>&1 \
        && echo "done: ${host}" >> "outputs/paper/hosts_v2xreal/v2xreal_${host}.log" || echo "FAILED: ${host} (see outputs/paper/hosts_v2xreal/v2xreal_${host}.log)" >> "outputs/paper/hosts_v2xreal/v2xreal_${host}.log"
    }
    # V2X-ViT is not in this list yet: public QuantV2X lacks the private version's padding of the feature map to a
    # multiple of 16 before its window attention (config: fusion_pad_multiple), so it does not run as is.
    for host in attfuse fcooper; do
      RUN_LOGS+=("outputs/paper/hosts_v2xreal/v2xreal_${host}.log")
      run_v2x_host "${host}" &
      progress_until 2   # the hardware logger + two networks
    done
    progress_until 1   # until only the hardware logger is left
    ;;
  late-20seeds)
    # Late fusion, all four plug-ins, 20 seeds, both datasets (CPU). Pose files that exist are kept (seeds 0-4),
    # the missing seeds 5-19 are added; run AFTER opv2v-poses-pgo so OPV2V CoAlign/RoCo use the new stage-1 boxes.
    for ds in opv2v v2xreal; do
      "${PY}" scripts/export_poses.py --dataset "${ds}" --seeds 20 --workers "${WORKERS}"
      "${ENVS}/coalign-blackwell/bin/python" scripts/export_poses_pgo.py --method coalign --dataset "${ds}" --seeds 20 --workers "${WORKERS}"
      "${ENVS}/roco-blackwell/bin/python" scripts/export_poses_pgo.py --method roco --dataset "${ds}" --seeds 20 --workers "${WORKERS}"
      "${PY}" scripts/late_from_poses.py --dataset "${ds}" --seeds 20 --workers "${WORKERS}" \
        --output "outputs/paper/late_${ds}_plugins_20seeds.json"
    done
    ;;
  v2xreal-stage1)
    # CoAlign / RoCo stage 1 on V2X-Real: CoAlign's own train.py and recipe (30 epochs), V2X-Real data settings.
    # Prints one line per iteration. Checkpoints go to external/CoAlign/opencood/logs/v2xreal_point_pillar_uncertainty_*.
    # Resume after an interruption: add --model-dir <that log folder> to the line below.
    root=""; [[ -d "${NVME_V2X}/train" ]] && root="--data-root ${NVME_V2X}"
    bash scripts/log_hardware.sh outputs/paper/stage1_v2xreal_hardware.log &
    LOGGER=$!
    trap 'kill ${LOGGER} 2>/dev/null' EXIT
    # 16 loader workers, normal priority: on the NVMe copy this makes the GPU the limit (~0.11 s per iteration)
    PYTHONPATH=src OMP_NUM_THREADS=1 "${ENVS}/coalign-blackwell/bin/python" scripts/coalign_v2xreal.py train ${root} --workers 16
    ;;
  v2xreal-stage1-boxes)
    # stage-1 boxes + uncertainty on V2X-Real test, from the newest stage-1 log folder (minutes)
    dir="$(ls -dt external/CoAlign/opencood/logs/v2xreal_point_pillar_uncertainty_late_fusion_* | head -1)"
    echo "stage-1 run: ${dir}"
    PYTHONPATH=src "${ENVS}/coalign-blackwell/bin/python" scripts/coalign_v2xreal.py boxes --stage1-dir "${dir}" --split test
    ;;
  v2xreal-poses-pgo)
    # CoAlign and RoCo poses on V2X-Real (needs v2xreal-poses and v2xreal-stage1-boxes), then late fusion with all
    # four plug-ins on the same 8 sigmas x 5 seeds, as on OPV2V
    "${ENVS}/coalign-blackwell/bin/python" scripts/export_poses_pgo.py --method coalign --dataset v2xreal --workers "${WORKERS}" --overwrite
    "${ENVS}/roco-blackwell/bin/python" scripts/export_poses_pgo.py --method roco --dataset v2xreal --workers "${WORKERS}" --overwrite
    "${PY}" scripts/late_from_poses.py --dataset v2xreal --workers "${WORKERS}" --output outputs/paper/late_v2xreal_plugins.json
    ;;
  opv2v-dryrun)
    for host in attfuse coalign cobevt fcooper v2vam v2xvit where2comm; do
      "${PY}" scripts/run_host.py --dataset opv2v --host "${host}" --plugins freealign alignformer --sigmas 0 1 --seeds 1 \
        --max-frames 50 --workers "${WORKERS}" \
        --output "outputs/paper/dryrun/opv2v_${host}.json"
    done
    "${PY}" scripts/summarize_hosts.py outputs/paper/dryrun
    ;;
  opv2v-timing)
    # all 7 OPV2V networks, all 2170 frames, sigma 1, seed 0: true / bare / FreeAlign / CoAlign / RoCo / AlignFormer.
    # Measures the real time per network and projects the full run (144 conditions per frame). Skips networks already
    # done. Logs GPU power and temperatures every 5 s to outputs/paper/timing/hardware.log (the machine reset twice).
    mkdir -p outputs/paper/timing
    bash scripts/log_hardware.sh outputs/paper/timing/hardware.log &
    LOGGER=$!
    trap 'kill ${LOGGER} 2>/dev/null' EXIT
    for host in attfuse v2xvit coalign cobevt fcooper v2vam where2comm; do
      out="outputs/paper/timing/opv2v_${host}.json"
      if [[ -f "${out}" ]]; then echo "exists, skipping: ${out}"; continue; fi
      # gentle on a shared machine: low CPU priority, 2 threads in the main process
      OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 \
        "${PY}" scripts/run_host.py --dataset opv2v --host "${host}" --sigmas 1 --seeds 1 --workers "${WORKERS}" --output "${out}"
    done
    "${PY}" scripts/summarize_hosts.py outputs/paper/timing --full-conditions 144
    ;;
  opv2v-hosts)
    # T1, seed 0: every OPV2V network at every sigma, true / bare / FreeAlign / CoAlign / RoCo / AlignFormer.
    # Each network makes ONE pass over the frames for all 8 sigmas (each frame read once, the true pose run once)
    # and writes one file per (network, sigma). Two networks run at a time, each with half the workers.
    # A network whose 8 files all exist is skipped; if any is missing the network is redone (one pass).
    mkdir -p outputs/paper/hosts
    bash scripts/log_hardware.sh outputs/paper/hosts/hardware.log &
    LOGGER=$!
    trap 'kill ${LOGGER} 2>/dev/null' EXIT
    run_one_host() {
      local host="$1" sigma missing=0
      for sigma in 0 0.2 0.4 0.6 0.8 1 1.5 2; do
        [[ -f "outputs/paper/hosts/opv2v_${host}_sigma_${sigma}.json" ]] || missing=1
      done
      if (( ! missing )); then echo "exists, skipping: ${host} (all 8 sigmas)" > "outputs/paper/hosts/opv2v_${host}.log"; return; fi
      OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "${PY}" scripts/run_host.py --dataset opv2v --host "${host}" \
        --sigmas 0 0.2 0.4 0.6 0.8 1 1.5 2 --seeds 1 --workers $(( WORKERS / 2 )) \
        --output "outputs/paper/hosts/opv2v_${host}_sigma_{sigma}.json" > "outputs/paper/hosts/opv2v_${host}.log" 2>&1 \
        && echo "done: ${host}" >> "outputs/paper/hosts/opv2v_${host}.log" || echo "FAILED: ${host} (see outputs/paper/hosts/opv2v_${host}.log)" >> "outputs/paper/hosts/opv2v_${host}.log"
    }
    for host in attfuse v2xvit coalign cobevt fcooper v2vam where2comm; do
      RUN_LOGS+=("outputs/paper/hosts/opv2v_${host}.log")
      run_one_host "${host}" &
      progress_until 2   # the hardware logger + two networks
    done
    progress_until 1   # until only the hardware logger is left
    "${PY}" scripts/summarize_hosts.py outputs/paper/hosts
    ;;
  delay)
    # T6, seed 0: late fusion under unknown delay (0 / 100 / 200 / 400 ms) at sigma 0 and 1, OPV2V then V2X-Real.
    # Rebuilds the delayed-message caches (senders now track with their own odometry, never true poses) into
    # outputs/async/, then scores FreeAlign and AlignFormer.
    "${PY}" scripts/build_async_cache.py --frame-table outputs/opv2v_frame_table.pt \
      --geometry-cache outputs/r140_geometry_test_cache --odometry outputs/odometry/opv2v_test.npz \
      --checkpoint "${OPV2V_AF}" --freealign-calibration "${OPV2V_FA}" --sigmas 0 1 --seeds 1 --delays 0 1 2 4 \
      --compensate --output-dir outputs/async/opv2v --workers "${WORKERS}"
    "${PY}" scripts/build_async_cache.py --frame-table outputs/v2xreal_test_frame_table.pt \
      --geometry-cache outputs/v2xreal_geometry_test_cache --odometry outputs/odometry/v2xreal_test.npz \
      --checkpoint "${V2X_AF}" --freealign-calibration "${V2X_FA}" --sigmas 0 1 --seeds 1 --delays 0 1 2 4 \
      --compensate --output-dir outputs/async/v2xreal --workers "${WORKERS}"
    "${PY}" scripts/latency_sweep.py --async-dir outputs/async/opv2v --odometry outputs/odometry/opv2v_test.npz \
      --workers "${WORKERS}" --output outputs/paper/delay_opv2v.json
    "${PY}" scripts/latency_sweep.py --async-dir outputs/async/v2xreal --frame-table outputs/v2xreal_test_frame_table.pt \
      --geometry-cache outputs/v2xreal_geometry_test_cache --odometry outputs/odometry/v2xreal_test.npz \
      --checkpoint "${V2X_AF}" --freealign-calibration "${V2X_FA}" \
      --opencood-config "${V2X_DET}/config.yaml" --eval-range ${V2X_RANGE} \
      --workers "${WORKERS}" --output outputs/paper/delay_v2xreal.json
    ;;
  revision-nov12)
    # Assessment 2026_11_11 (revision of 2026-11-12). CPU only, existing caches, no new noise draws.
    # 1-2) EXP-003, ablation of pooling: FreeAlign / CoAlign / RoCo per-frame corrections, alone and through
    #      AlignFormer's pooling + window Abstain; seeds 0-4, sigma 0 / 0.4 / 1 / 2 (sigma 0 has one draw).
    POOLED="freealign_tk coalign_frame coalign_tk roco_frame roco_tk alignformer_c alignformer_tk"
    ALIGNFORMER_POSES_DATASET=opv2v "${PY}" scripts/alignformer_variants_sweep.py --cache-dir outputs/propose_cache_opv2v \
      --frame-table outputs/opv2v_frame_table.pt --checkpoint "${OPV2V_AF}" --opencood-config "${OPV2V_CFG}" \
      --odometry outputs/odometry/opv2v_test.npz --geometry-cache outputs/r140_geometry_test_cache \
      --variants ${POOLED} --sigmas 0 0.4 1 2 --max-seeds 5 --workers "${WORKERS}" --output outputs/paper/pooled_opv2v.json
    ALIGNFORMER_POSES_DATASET=v2xreal "${PY}" scripts/alignformer_variants_sweep.py --cache-dir outputs/propose_cache_v2xreal \
      --frame-table outputs/v2xreal_test_frame_table.pt --checkpoint "${V2X_AF}" \
      --opencood-config "${V2X_DET}/config.yaml" --eval-range ${V2X_RANGE} \
      --odometry outputs/odometry/v2xreal_test.npz --geometry-cache outputs/v2xreal_geometry_test_cache \
      --variants ${POOLED} --sigmas 0 0.4 1 2 --max-seeds 5 --workers "${WORKERS}" --output outputs/paper/pooled_v2xreal.json
    # 3-4) EXP-007 + oracle: T6 with the staleness table fitted without test data (OPV2V causal, V2X-Real validation)
    #      and the known-delay oracle row (analysis only). Same seed-0 caches as the paper. The old test-fitted files
    #      are kept as delay_<ds>_testtable.json.
    for ds in opv2v v2xreal; do
      [[ -f "outputs/paper/delay_${ds}_testtable.json" ]] || cp "outputs/paper/delay_${ds}.json" "outputs/paper/delay_${ds}_testtable.json"
    done
    "${PY}" scripts/latency_sweep.py --async-dir outputs/async/opv2v --odometry outputs/odometry/opv2v_test.npz \
      --staleness-table causal --workers "${WORKERS}" --output outputs/paper/delay_opv2v.json
    "${PY}" scripts/latency_sweep.py --async-dir outputs/async/v2xreal --frame-table outputs/v2xreal_test_frame_table.pt \
      --geometry-cache outputs/v2xreal_geometry_test_cache --odometry outputs/odometry/v2xreal_test.npz \
      --checkpoint "${V2X_AF}" --freealign-calibration "${V2X_FA}" \
      --opencood-config "${V2X_DET}/config.yaml" --eval-range ${V2X_RANGE} \
      --staleness-table other --staleness-from outputs/v2xreal_val_frame_table.pt outputs/v2xreal_geometry_val_cache \
        outputs/odometry/v2xreal_val.npz \
      --workers "${WORKERS}" --output outputs/paper/delay_v2xreal.json
    # 5) regenerate every table body (and the pooled supplementary table) from outputs/paper/
    "${PY}" scripts/make_paper_exhibits.py
    ;;
  revision-5draws)
    # Every noisy result in the paper is the mean over five noise draws (seeds 0-4). Draw 0 of Tab. 1 already exists
    # (outputs/paper/hosts*, Nov 9), so this runs draws 1-4 only (--first-seed 1) at the sigmas Tab. 1 shows with noise
    # (1, 2); make_paper_exhibits.py merges both. Finished stages are skipped, so a restart only does what is missing.
    # Live progress: bash scripts/watch_5draws.sh
    S5="1 2"
    mkdir -p outputs/paper/hosts_5draws outputs/paper/hosts_v2xreal_5draws
    host_done() { [[ -f "$1_sigma_1.json" && -f "$1_sigma_2.json" ]]; }
    opv2v_host5() {
      local out="outputs/paper/hosts_5draws/opv2v_$1"
      if host_done "${out}"; then echo "done (skipped, exists): $1" > "${out}.log"; return; fi
      PYTHONPATH=src OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "${PY}" scripts/run_host.py --dataset opv2v --host "$1" \
        --sigmas ${S5} --first-seed 1 --seeds 5 --workers $(( WORKERS / 2 )) \
        --output "${out}_sigma_{sigma}.json" > "${out}.log" 2>&1 \
        && echo "done: $1" >> "${out}.log" || echo "FAILED: $1" >> "${out}.log"
    }
    v2x_host5() {
      local out="outputs/paper/hosts_v2xreal_5draws/v2xreal_$1"
      if host_done "${out}"; then echo "done (skipped, exists): $1" > "${out}.log"; return; fi
      PYTHONPATH=src OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "${ENVS}/coalign-blackwell/bin/python" scripts/run_host_v2xreal.py \
        --host "$1" --sigmas ${S5} --first-seed 1 --seeds 5 --workers $(( WORKERS / 2 )) \
        --output "${out}_sigma_{sigma}.json" > "${out}.log" 2>&1 \
        && echo "done: $1" >> "${out}.log" || echo "FAILED: $1" >> "${out}.log"
    }
    JOBS=(o:cobevt v:v2xvit o:v2xvit o:coalign v:attfuse o:attfuse v:where2comm o:fcooper v:fcooper o:v2vam v:v2vam o:where2comm)
    echo "[$(date +%H:%M)] hosts: 12 networks, sigma 1 and 2, draws 1-4, two at a time"
    for job in "${JOBS[@]}"; do
      while (( $(jobs -rp | wc -l) >= 2 )); do sleep 10; done
      echo "[$(date +%H:%M)] started ${job}"
      if [[ "${job%%:*}" == "o" ]]; then opv2v_host5 "${job#*:}" & else v2x_host5 "${job#*:}" & fi
      sleep 1
    done
    wait
    echo "[$(date +%H:%M)] hosts finished"
    if grep -h "FAILED" outputs/paper/hosts_5draws/*.log outputs/paper/hosts_v2xreal_5draws/*.log; then
      echo "a host network failed, see its log; stopping before the delay stage"; exit 1
    fi
    # T6: delayed-message caches for sigma 1, draws 1-4 (existing caches are skipped), then the sweep with the clean
    # staleness table, unless the result already has five draws. The one-draw files are kept as delay_<ds>_1draw.json.
    five_draws() { "${PY}" -c "import json,sys; d=json.load(open('$1')); sys.exit(0 if len(d['per_seed']['d4']['1'])>=5 else 1)" 2>/dev/null; }
    echo "[$(date +%H:%M)] delay: caches and sweep"
    "${PY}" scripts/build_async_cache.py --frame-table outputs/opv2v_frame_table.pt \
      --geometry-cache outputs/r140_geometry_test_cache --odometry outputs/odometry/opv2v_test.npz \
      --checkpoint "${OPV2V_AF}" --freealign-calibration "${OPV2V_FA}" --sigmas 1 --seeds 5 --delays 0 1 2 4 \
      --compensate --output-dir outputs/async/opv2v --workers "${WORKERS}"
    "${PY}" scripts/build_async_cache.py --frame-table outputs/v2xreal_test_frame_table.pt \
      --geometry-cache outputs/v2xreal_geometry_test_cache --odometry outputs/odometry/v2xreal_test.npz \
      --checkpoint "${V2X_AF}" --freealign-calibration "${V2X_FA}" --sigmas 1 --seeds 5 --delays 0 1 2 4 \
      --compensate --output-dir outputs/async/v2xreal --workers "${WORKERS}"
    for ds in opv2v v2xreal; do
      [[ -f "outputs/paper/delay_${ds}_1draw.json" ]] || cp "outputs/paper/delay_${ds}.json" "outputs/paper/delay_${ds}_1draw.json"
    done
    five_draws outputs/paper/delay_opv2v.json && echo "delay opv2v: five draws exist, skipped" || \
    "${PY}" scripts/latency_sweep.py --async-dir outputs/async/opv2v --odometry outputs/odometry/opv2v_test.npz \
      --staleness-table causal --workers "${WORKERS}" --output outputs/paper/delay_opv2v.json
    five_draws outputs/paper/delay_v2xreal.json && echo "delay v2xreal: five draws exist, skipped" || \
    "${PY}" scripts/latency_sweep.py --async-dir outputs/async/v2xreal --frame-table outputs/v2xreal_test_frame_table.pt \
      --geometry-cache outputs/v2xreal_geometry_test_cache --odometry outputs/odometry/v2xreal_test.npz \
      --checkpoint "${V2X_AF}" --freealign-calibration "${V2X_FA}" \
      --opencood-config "${V2X_DET}/config.yaml" --eval-range ${V2X_RANGE} \
      --staleness-table other --staleness-from outputs/v2xreal_val_frame_table.pt outputs/v2xreal_geometry_val_cache \
        outputs/odometry/v2xreal_val.npz \
      --workers "${WORKERS}" --output outputs/paper/delay_v2xreal.json
    # pose-error table over the same five draws (the 20-draw files are kept as pose_accuracy_<ds>_20draws.json)
    echo "[$(date +%H:%M)] pose error"
    for ds in opv2v v2xreal; do
      [[ -f "outputs/paper/pose_accuracy_${ds}_20draws.json" ]] || cp "outputs/paper/pose_accuracy_${ds}.json" "outputs/paper/pose_accuracy_${ds}_20draws.json"
      "${PY}" -c "import json,sys; sys.exit(0 if json.load(open('outputs/paper/pose_accuracy_${ds}.json'))['seeds']==5 else 1)" 2>/dev/null \
        && echo "pose ${ds}: five draws exist, skipped" \
        || "${PY}" scripts/pose_accuracy.py --dataset "${ds}" --sigma 1 --seeds 5 --output "outputs/paper/pose_accuracy_${ds}.json"
    done
    # every table body, all at five draws
    "${PY}" scripts/make_paper_exhibits.py
    echo "[$(date +%H:%M)] revision-5draws finished"
    ;;
  sigma-std)
    # The paper's sigma standard 0 / 0.5 / 1 / 1.5 / 2 (authors' decision, 2026-11-12 night). Only what is missing is
    # run, into NEW files (tag s05s15), so no existing result is overwritten; make_paper_exhibits.py merges them.
    #   1 CPU  Propose + FreeAlign caches at sigma 0.5 (5 draws), both datasets
    #   2 CPU  corrected poses of all four plug-ins at sigma 0.5 (5 draws), both datasets
    #   3 GPU  the 12 networks: sigma 0.5 draws 0-4 and sigma 1.5 draws 1-4 (V2VAM 1.5 comes from the authors' run),
    #          two at a time in the background, while 4-6 run on the CPU
    #   4 CPU  late fusion of the four plug-ins at 0.5; one-frame, ablation and pooled variants at 0.5 and 1.5
    #   5 CPU  audits E2 (classical starts), E5 (Abstain harm), E0 (bootstrap, branches), E3 (odometry) at the new levels
    #   6      every table and figure
    # Finished files are skipped, so a restart only does what is missing.
    TAG=s05s15
    W2=$(( WORKERS / 2 ))
    echo "[$(date +%H:%M)] 1/6 Propose caches at sigma 0.5"
    "${PY}" scripts/build_propose_cache.py --frame-table outputs/opv2v_frame_table.pt --checkpoint "${OPV2V_AF}" \
      --freealign-calibration "${OPV2V_FA}" --sigmas 0.5 --seeds 5 --output-dir outputs/propose_cache_opv2v --workers "${WORKERS}"
    "${PY}" scripts/build_propose_cache.py --frame-table outputs/v2xreal_test_frame_table.pt --checkpoint "${V2X_AF}" \
      --freealign-calibration "${V2X_FA}" --sigmas 0.5 --seeds 5 --output-dir outputs/propose_cache_v2xreal --workers "${WORKERS}"
    echo "[$(date +%H:%M)] 2/6 plug-in poses at sigma 0.5"
    for ds in opv2v v2xreal; do
      "${PY}" scripts/export_poses.py --dataset "${ds}" --sigmas 0.5 --seeds 5 --workers "${WORKERS}"
      "${ENVS}/coalign-blackwell/bin/python" scripts/export_poses_pgo.py --method coalign --dataset "${ds}" --sigmas 0.5 --seeds 5 --workers "${WORKERS}"
      "${ENVS}/roco-blackwell/bin/python" scripts/export_poses_pgo.py --method roco --dataset "${ds}" --sigmas 0.5 --seeds 5 --workers "${WORKERS}"
    done

    echo "[$(date +%H:%M)] 3/6 networks start in the background (logs in outputs/paper/hosts*_5draws/*_${TAG}.log)"
    mkdir -p outputs/paper/hosts_5draws outputs/paper/hosts_v2xreal_5draws
    host_std() {   # dataset network
      local dir py script extra out log
      if [[ "$1" == "opv2v" ]]; then dir=outputs/paper/hosts_5draws; py="${PY}"; script="scripts/run_host.py"; extra="--dataset opv2v"
      else dir=outputs/paper/hosts_v2xreal_5draws; py="${ENVS}/coalign-blackwell/bin/python"; script="scripts/run_host_v2xreal.py"; extra=""; fi
      out="${dir}/$1_$2"; log="${out}_${TAG}.log"; : > "${log}"
      if [[ ! -f "${out}_sigma_0.5.json" ]]; then
        PYTHONPATH=src OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "${py}" ${script} ${extra} --host "$2" --sigmas 0.5 \
          --first-seed 0 --seeds 5 --workers "${W2}" --output "${out}_sigma_{sigma}.json" >> "${log}" 2>&1 \
          || { echo "FAILED: $1 $2 sigma 0.5" >> "${log}"; return; }
      fi
      if [[ "$2" != "v2vam" && ! -f "${out}_sigma_1.5.json" ]]; then
        PYTHONPATH=src OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "${py}" ${script} ${extra} --host "$2" --sigmas 1.5 \
          --first-seed 1 --seeds 5 --workers "${W2}" --output "${out}_sigma_{sigma}.json" >> "${log}" 2>&1 \
          || { echo "FAILED: $1 $2 sigma 1.5" >> "${log}"; return; }
      fi
      echo "done: $1 $2" >> "${log}"
    }
    (
      for job in opv2v:cobevt v2xreal:v2xvit opv2v:v2xvit opv2v:coalign v2xreal:attfuse opv2v:attfuse \
                 v2xreal:where2comm opv2v:fcooper v2xreal:fcooper opv2v:v2vam v2xreal:v2vam opv2v:where2comm; do
        while (( $(jobs -rp | wc -l) >= 2 )); do sleep 10; done
        host_std "${job%%:*}" "${job#*:}" &
        sleep 2
      done
      wait
    ) &
    HOSTS_PID=$!

    echo "[$(date +%H:%M)] 4/6 late fusion, ablation and pooled variants at the new levels"
    for ds in opv2v v2xreal; do
      [[ -f "outputs/paper/late_${ds}_plugins_${TAG}.json" ]] || \
        "${PY}" scripts/late_from_poses.py --dataset "${ds}" --sigmas 0.5 --seeds 5 --workers "${W2}" \
          --output "outputs/paper/late_${ds}_plugins_${TAG}.json"
    done
    STAGES="propose_only propose_verify alignformer_c alignformer_pool alignformer_tk"
    POOLED="freealign_tk coalign_frame coalign_tk roco_frame roco_tk alignformer_c alignformer_tk"
    [[ -f "outputs/paper/ablation_opv2v_${TAG}.json" ]] || "${PY}" scripts/alignformer_variants_sweep.py \
      --cache-dir outputs/propose_cache_opv2v --frame-table outputs/opv2v_frame_table.pt --checkpoint "${OPV2V_AF}" \
      --opencood-config "${OPV2V_CFG}" --odometry outputs/odometry/opv2v_test.npz --geometry-cache outputs/r140_geometry_test_cache \
      --variants ${STAGES} --sigmas 0.5 1.5 --max-seeds 5 --workers "${W2}" --output "outputs/paper/ablation_opv2v_${TAG}.json"
    [[ -f "outputs/paper/ablation_v2xreal_${TAG}.json" ]] || "${PY}" scripts/alignformer_variants_sweep.py \
      --cache-dir outputs/propose_cache_v2xreal --frame-table outputs/v2xreal_test_frame_table.pt --checkpoint "${V2X_AF}" \
      --opencood-config "${V2X_DET}/config.yaml" --eval-range ${V2X_RANGE} --odometry outputs/odometry/v2xreal_test.npz \
      --geometry-cache outputs/v2xreal_geometry_test_cache \
      --variants ${STAGES} --sigmas 0.5 1.5 --max-seeds 5 --workers "${W2}" --output "outputs/paper/ablation_v2xreal_${TAG}.json"
    [[ -f "outputs/paper/pooled_opv2v_${TAG}.json" ]] || ALIGNFORMER_POSES_DATASET=opv2v "${PY}" scripts/alignformer_variants_sweep.py \
      --cache-dir outputs/propose_cache_opv2v --frame-table outputs/opv2v_frame_table.pt --checkpoint "${OPV2V_AF}" \
      --opencood-config "${OPV2V_CFG}" --odometry outputs/odometry/opv2v_test.npz --geometry-cache outputs/r140_geometry_test_cache \
      --variants ${POOLED} --sigmas 0.5 1.5 --max-seeds 5 --workers "${W2}" --output "outputs/paper/pooled_opv2v_${TAG}.json"
    [[ -f "outputs/paper/pooled_v2xreal_${TAG}.json" ]] || ALIGNFORMER_POSES_DATASET=v2xreal "${PY}" scripts/alignformer_variants_sweep.py \
      --cache-dir outputs/propose_cache_v2xreal --frame-table outputs/v2xreal_test_frame_table.pt --checkpoint "${V2X_AF}" \
      --opencood-config "${V2X_DET}/config.yaml" --eval-range ${V2X_RANGE} --odometry outputs/odometry/v2xreal_test.npz \
      --geometry-cache outputs/v2xreal_geometry_test_cache \
      --variants ${POOLED} --sigmas 0.5 1.5 --max-seeds 5 --workers "${W2}" --output "outputs/paper/pooled_v2xreal_${TAG}.json"

    echo "[$(date +%H:%M)] 5/6 audits at the new levels"
    PAPER_TAG=${TAG} PAPER_SIGMAS="0.5 1.5" "${PY}" scripts/audit_e2_propose.py --splits opv2v_test v2xreal_test v2xreal_test_opv2v_checkpoint --workers "${W2}"
    PAPER_TAG=${TAG} PAPER_SIGMAS="0.5 1.5 2" "${PY}" scripts/audit_e5_decision.py --workers "${W2}"
    PAPER_TAG=${TAG} PAPER_SIGMAS="0.5 1.5" "${PY}" scripts/audit_e0_bootstrap.py --workers "${W2}"
    PAPER_TAG=${TAG} PAPER_SIGMAS="0.5 1.5" "${PY}" scripts/audit_e3_odometry.py --workers "${W2}"

    echo "[$(date +%H:%M)] waiting for the networks (one line each, refreshed every 5 s)"
    RUN_LOGS=()
    for job in opv2v:cobevt v2xreal:v2xvit opv2v:v2xvit opv2v:coalign v2xreal:attfuse opv2v:attfuse \
               v2xreal:where2comm opv2v:fcooper v2xreal:fcooper opv2v:v2vam v2xreal:v2vam opv2v:where2comm; do
      d=outputs/paper/hosts_5draws; [[ "${job%%:*}" == "v2xreal" ]] && d=outputs/paper/hosts_v2xreal_5draws
      RUN_LOGS+=("${d}/${job%%:*}_${job#*:}_${TAG}.log")
    done
    while kill -0 "${HOSTS_PID}" 2>/dev/null; do draw_progress; sleep 5; done
    draw_progress
    if grep -h "FAILED" outputs/paper/hosts_5draws/*_${TAG}.log outputs/paper/hosts_v2xreal_5draws/*_${TAG}.log; then
      echo "a network failed, see its log"; exit 1
    fi
    echo "[$(date +%H:%M)] 6/6 tables and figures"
    "${PY}" scripts/make_paper_exhibits.py
    "${PY}" scripts/make_f1_teaser.py
    echo "[$(date +%H:%M)] sigma-std finished"
    ;;
  *) echo "unknown step ${STEP}" >&2; exit 2 ;;
esac
