# Plan-0 proposal embedding training

## Implemented pipeline

The extractor runs the frozen official OpenCOOD PointPillar checkpoint on clean,
same-timestamp multi-agent OPV2V frames. For each CAV it:

1. preserves flattened anchor indices through score filtering and local rotated NMS;
2. maps each retained anchor to its exact 384-dimensional pre-head BEV cell feature;
3. assigns predicted proposals one-to-one to local GT using rotated BEV IoU;
4. stores stable CARLA object IDs, agent IDs, boxes, scores, IoUs, and FP16 features.

The embedding dataset keeps the best-IoU proposal per `(object, agent)` and uses:

- proposals of the same object from different agents as positives;
- other objects in the same frame as natural negatives;
- scenario-disjoint training and validation splits.

The model is:

```text
384 -> 256 -> 128 -> 64 -> L2 normalization
```

and is trained with cross-agent supervised contrastive loss. Same-agent proposals
are never treated as positives.

## Refresh the editable installation

```bash
conda activate /mnt/external/backup/dhia/conda_envs/opencood-blackwell
cd /mnt/external/workspace/dhia/embedding-aware-BELT-fusion
python -m pip install -e . --no-deps
```

## Extract the full training cache

This is frozen-detector inference, not detector training. Noise is forcibly disabled
for feature extraction and GT identity assignment.

```bash
extract-opencood-proposals \
  --opencood-config models/pointpillar_late_fusion/config.yaml \
  --detector-dir models/pointpillar_late_fusion \
  --data-root /mnt/external/workspace/public/dataset/opv2v \
  --output-dir outputs/proposal_cache_train \
  --split train \
  --minimum-gt-iou 0.5 \
  --shard-frames 100 \
  --device cuda
```

Do not add `--frame-stride` for the full cache. The extractor writes a manifest and
sharded `.pt` records; all are ignored by Git.

For a quick extractor smoke test:

```bash
extract-opencood-proposals \
  --opencood-config models/pointpillar_late_fusion/config.yaml \
  --detector-dir models/pointpillar_late_fusion \
  --data-root /mnt/external/workspace/public/dataset/opv2v \
  --output-dir outputs/proposal_cache_smoke \
  --split train \
  --minimum-gt-iou 0.5 \
  --shard-frames 8 \
  --max-frames 8 \
  --frame-stride 1000 \
  --device cuda
```

## Train the embedding

After full extraction finishes:

```bash
train-proposal-embedding \
  --cache-dir outputs/proposal_cache_train \
  --output-dir outputs/proposal_embedding \
  --epochs 30 \
  --batch-size 16 \
  --workers 4 \
  --embedding-dim 128 \
  --learning-rate 0.001 \
  --weight-decay 0.0001 \
  --temperature 0.07 \
  --validation-fraction 0.2 \
  --seed 42 \
  --device cuda
```

`--batch-size` counts frames, not proposals. Each frame contributes a variable number
of matched proposals.

Monitor:

- decreasing training and validation contrastive loss;
- increasing cross-agent top-1 retrieval accuracy;
- positive cosine similarity separating from negative cosine similarity.

The output directory contains one embedding checkpoint per epoch and `history.json`.

## Verification completed

- A two-frame real-data extraction verified anchor/feature/GT alignment.
- A six-frame, multi-scenario cache verified scenario-disjoint data loading.
- Real proposals included many identities observed by two or three CAVs.
- A two-epoch embedding smoke run completed loss, validation retrieval, and
  checkpoint generation.

## Next stage

After training, calibrate an embedding-similarity rejection threshold on validation
scenarios and compare:

1. original OpenCOOD geometry/NMS association;
2. embedding-only Hungarian matching;
3. combined geometry, uncertainty, and embedding matching.
