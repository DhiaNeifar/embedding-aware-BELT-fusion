# Models

One folder per dataset, with the same layout in each:

```
<dataset>/
  late_fusion/<detector>/        per-agent LiDAR detector; its boxes are what AlignFormer and FreeAlign align
  alignformer_boxes_only.pth     AlignFormer pose head (Chenyi's stage-2 "B, boxes-only", converted to alignformer_pva)
  freealign_calibration.json     FreeAlign parameters selected on validation (the baseline exactly as in the paper)
  intermediate_fusion/<method>/  config.yaml + checkpoint of each intermediate-fusion host
```

## OPV2V

| path | what | origin |
|---|---|---|
| `late_fusion/pointpillar_r140/` | PointPillars late fusion, +/-140.8 m | Chenyi's bundle (byte-identical) |
| `alignformer_boxes_only.pth` | pose head, from `r140/stage2_B_matched_boxes_only/best.pth` | `scripts/convert_chenyi_alignformer_checkpoint.py` |
| `freealign_calibration.json` | `r140/freealign_calibration_valap_result.json` | Chenyi's bundle |
| `intermediate_fusion/{attfuse,coalign,cobevt,fcooper,v2vam,v2xvit,where2comm}` | the paper's Table 5 baselines (OpenCOOD) | Chenyi's bundle, `models/opv2v/` |
| `intermediate_fusion/coalign_boxalign/` | CoAlign with its own pose graph (box_align) + its `stage1_boxes.json` | ours |
| `intermediate_fusion/roco/` | RoCo, epoch 23; runs in `external/RoCo_merged` | ours |

## V2X-Real

| path | what | origin |
|---|---|---|
| `late_fusion/pointpillar/` | PointPillars late fusion, epoch 13 | Chenyi's bundle |
| `alignformer_boxes_only.pth` | pose head, from `v2xreal/stage2_B_boxes_only/best.pth` | converted |
| `freealign_calibration.json` | `v2xreal/freealign_calibration_valap_thr1_result.json` | Chenyi's bundle |
| `intermediate_fusion/` | attfuse, attfuse_encdec, codefilling_base, codefilling_codebook, cosdh, cost, ermvp, fcooper, nofusion, v2vam, v2vnet, v2xvit, where2comm (`<method>_v2xreal_clean` runs) | jigsawcomm-v2x |

## CooperScene

| path | what | origin |
|---|---|---|
| `late_fusion/pointpillar/` | PointPillars late fusion, epoch 15 | `AdverCPM_hf/models/detectors/cooperscene_pointpillars` |
| AlignFormer / FreeAlign | none of their own: the V2X-Real pose head and calibration are used zero-shot | |
| `intermediate_fusion/` | the 18 runs listed in `intermediate_fusion/README.md` (folder names drop the `_cooperscene` suffix) | jigsawcomm-v2x |

## Code that runs them

- Late fusion: this repo (`scripts/`, `src/embedding_aware_belt_fusion/alignformer_ext`), with `external/OpenCOOD`.
- OPV2V intermediate: OpenCOOD checkpoints, evaluated in the paper by Chenyi's
  `embedding_aware_belt_fusion.alignformer.baselines` (repo `embedding-aware-BELT-fusion`, branch `alignformer`).
  CoAlign box_align: `external/CoAlign`. RoCo: `external/RoCo_merged`.
- V2X-Real / CooperScene intermediate: trained in repo **jigsawcomm-v2x**, branch `baselines-package`, commit `a2f45a0`, which is
  not on this machine. Most are stock OpenCOOD networks under other key names and can load through a small adapter, checked
  against their reported clean AP. Where2comm's and CoST's fusion modules are not in stock OpenCOOD.
  The configs use `${COOPERSCENE_ROOT}` or absolute dataset paths that must be substituted before evaluation.

Chenyi's full provenance bundle (stage-1 heads, shrinkage calibrations, SHA256SUMS) stays in
`/mnt/external/workspace/dhia/embedding-aware-BELT-fusion/models/`.
