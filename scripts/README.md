# Scripts

Every run behind the paper goes through `run_paper.sh STEP` (steps listed in its header). The Python scripts it calls:

| script | what it does |
|---|---|
| `build_v2xreal_geometry_cache.py` | GPU, once: each agent's boxes from the V2X-Real late-fusion detector |
| `reproduce_multidataset_sweep.py --frame-table-only` | V2X-Real frame tables (poses, ground truth, cached boxes) |
| `reproduce_opv2v_sweep.py` | OPV2V frame table, and the shared helpers every script imports as `base` |
| `build_propose_cache.py` | Propose (the unmodified network) and FreeAlign per (sigma, seed); the only place noise is drawn |
| `build_odometry_table.py` | KISS-ICP on every vehicle's own scans (runs in `outputs/kiss_icp_env`) |
| `alignformer_variants_sweep.py` | late fusion: Verify, Abstain, pooling over time, NMS and AP (C++ `verify_core`, `fusion_core`) |
| `build_async_cache.py`, `latency_sweep.py` | late fusion under unknown delay |
| `check_verify_core.py` | the C++ Verify reproduces the Python reference |
| `convert_chenyi_alignformer_checkpoint.py` | how `models/*/alignformer_boxes_only.pth` were made from Chenyi's checkpoints |
