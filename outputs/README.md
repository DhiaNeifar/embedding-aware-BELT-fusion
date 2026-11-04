# Outputs

Everything here is derived data and results of the current pipeline. Model weights live in `models/`.

| path | what | written by | read by |
|---|---|---|---|
| `r140_geometry_test_cache/`, `v2xreal_geometry_{test,val}_cache/`, `cooperscene_geometry_{test,validate}_cache/` | per-agent local boxes and scores from each dataset's late-fusion detector (detector run once) | `build_*_geometry_cache.py`, `run_multidataset_sweep.sh` | frame tables |
| `opv2v_frame_table.pt`, `v2xreal_{test,val}_frame_table.pt`, `cooperscene_{test,validate}_frame_table.pt` | per frame: world poses, ground truth, cached boxes | `reproduce_*_sweep.py` | every sweep |
| `cooperscene_stage/` | staged CooperScene frames used to build its frame tables | `reproduce_multidataset_sweep.py` | same |
| `propose_cache_{opv2v,v2xreal,cooperscene}[_val]/` | AlignFormer Propose + FreeAlign outputs per (sigma, seed) | `build_propose_cache.py` | `alignformer_variants_sweep.py` |
| `async_opv2v/`, `async_v2xreal/` | delayed-message caches (0/100/200/400 ms) | `build_async_cache.py` | `async_sweep.py`, `latency_sweep.py` |
| `calibration/` | CooperScene scan-vs-pose offsets (reported as a dataset fault) | `measure_scan_pose_offsets.py` | `--calibration` runs |
| `variants/` | 20-seed results of every variant sweep | `alignformer_variants_sweep.py`, `latency_sweep.py` | paper tables, website |
| `multidataset_sweep/` | reproduction of Chenyi's protocol on the three datasets | `reproduce_multidataset_sweep.py` | |
| `sweep_full_20seeds_v2.json`, `sweep_cavs_{ge3,lt3}_v2.json` | reference OPV2V sweep (all / >=3 / <3 CAVs) | `reproduce_opv2v_sweep.py` | `check_verify_core.py` |
| `noise_sweep_results/summary.csv` | earlier CoAlign / RoCo OPV2V noise sweep (summary only) | `sweep_results.py` | |
