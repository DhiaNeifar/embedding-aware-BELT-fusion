# Frozen PointPillar uncertainty training

## Implemented stage

`FrozenPointPillarWithUncertainty` wraps the pretrained OpenCOOD `PointPillar`.
The following detector modules remain frozen and in evaluation mode:

- PillarVFE;
- point-pillar scatter;
- BEV backbone;
- original classification head (`psm`);
- original regression-mean head (`rm`).

The wrapper consumes the shared `[B,384,H,W]` `spatial_features_2d` tensor and adds:

- a small shared `3x3` convolutional projection;
- two-outcome evidence per anchor (background and vehicle);
- seven regression log-variance values per anchor.

For the official configuration with two anchors per cell:

```text
psm             [B,  2, H, W]  frozen classification logits
rm              [B, 14, H, W]  frozen regression means
evidence        [B,  2, 2, H, W]
alpha           [B,  2, 2, H, W]
cls_uncertainty [B,  2, H, W]
reg_log_var     [B, 14, H, W]
```

Classification uses a two-outcome Dirichlet because OPV2V has one foreground class
but anchor training still requires background evidence. Regression uses the existing
OpenCOOD encoded anchor targets and trains only at positive anchors.

## Install the parent project

In the Blackwell-compatible environment:

```bash
conda activate /mnt/external/backup/dhia/conda_envs/opencood-blackwell
cd /mnt/external/workspace/dhia/embedding-aware-BELT-fusion
python -m pip install -e . --no-deps
```

## One-batch GPU smoke run

This writes a disposable checkpoint after one training batch and one validation
batch:

```bash
train-belt-uncertainty \
  --opencood-config models/pointpillar_late_fusion/config.yaml \
  --detector-dir models/pointpillar_late_fusion \
  --output-dir outputs/belt_uncertainty_smoke \
  --epochs 1 \
  --batch-size 1 \
  --workers 0 \
  --max-train-batches 1 \
  --validation-batches 1
```

Confirm that:

- the progress bar reports finite regression and classification losses;
- `training_metadata.json` names the epoch-30 detector checkpoint;
- `uncertainty_epoch1.pth` is created;
- only uncertainty-head weights appear in that checkpoint.

## Full initial training

```bash
train-belt-uncertainty \
  --opencood-config models/pointpillar_late_fusion/config.yaml \
  --detector-dir models/pointpillar_late_fusion \
  --output-dir outputs/belt_uncertainty \
  --epochs 10 \
  --batch-size 2 \
  --workers 4 \
  --learning-rate 0.001 \
  --weight-decay 0.0001 \
  --hidden-channels 128 \
  --kl-weight 0.001 \
  --negative-weight 0.25 \
  --validation-batches 100 \
  --seed 42
```

The clean OpenCOOD configuration is intentional for this first stage. Agent pose
noise affects cross-agent coordinate transformation and association, whereas the
local detector's aleatoric uncertainty heads should first learn from native-frame
detection residuals. Noisy or mixed training is a later controlled ablation.

Resume with:

```bash
train-belt-uncertainty \
  --opencood-config models/pointpillar_late_fusion/config.yaml \
  --detector-dir models/pointpillar_late_fusion \
  --output-dir outputs/belt_uncertainty \
  --epochs 10 \
  --resume outputs/belt_uncertainty/uncertainty_epoch5.pth
```

## Verification completed

A real CPU smoke pass through one OPV2V training sample and the official epoch-30
checkpoint produced:

```text
spatial_features_2d  (1, 384, 100, 176)
psm                  (1,   2, 100, 176)
rm                   (1,  14, 100, 176)
evidence             (1,   2, 2, 100, 176)
reg_log_var           (1,  14, 100, 176)
positive anchors      31
```

Synthetic tests verify tensor layouts, finite loss, frozen-detector gradient
isolation, detector evaluation mode, and uncertainty-checkpoint round trips.

## Next implementation after training

The checkpoint alone does not change AP. The next adapter must carry anchor indices
through OpenCOOD score filtering, box decoding, projection, abnormal-box filtering,
and NMS so each retained proposal has its matching evidence and covariance. Only
then can the corrected BELT association/fusion consume these fields and be evaluated
against the reproduced noisy baseline.
