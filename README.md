# Embedding-Aware BELT-Fusion

This project studies robust cooperative LiDAR detection on OPV2V.  It starts
from a pretrained OpenCOOD PointPillars late-fusion detector, reproduces the
effect of localization and delay noise, adds uncertainty-aware BELT-style box
fusion, and investigates a Spatial TrackFormer association embedding.  A
separate SWFormer detector study is included as a backbone exploration.

The repository contains code and experiment metadata, but not OPV2V, model
checkpoints, or cached features.  The required upstream projects are Git
submodules:

- `external/OpenCOOD` — PointPillars detector and late-fusion baseline.
- `external/BELT-Fusion` — reference BELT-Fusion implementation.
- `external/trackformer` — reference TrackFormer implementation.
- `external/OpenCOOD-SWFormer` — SWFormer-based OpenCOOD fork.

Clone with:

```bash
git clone --recurse-submodules <repository-url>
```

The experiments below assume OPV2V is available at
`/mnt/external/workspace/public/dataset/opv2v`.

## PointPillars and BELT-style uncertainty-aware fusion

OpenCOOD PointPillars late fusion was made operational on the RTX 5090 system.
The pretrained epoch-30 checkpoint reproduces the expected clean OPV2V
late-fusion baseline.  In the noisy condition, non-ego agents receive Gaussian
position and heading perturbations plus delay: 0.2 m, 0.2 degrees, and 100 ms
(seed 20).

| Method | Condition | AP@0.3 | AP@0.5 | AP@0.7 |
|---|---|---:|---:|---:|
| PointPillars naive late fusion | Clean | 0.867 | 0.859 | 0.782 |
| PointPillars naive late fusion | Noisy | 0.860 | 0.712 | 0.294 |
| BELT-style geometry/uncertainty fusion | Noisy | 0.851 | 0.752 | 0.415 |
| BELT-style fusion + Spatial TrackFormer association | Noisy | 0.846 | 0.743 | 0.423 |

The uncertainty-aware fusion step raises AP@0.7 by 0.121 over noisy naive
late fusion.  The current TrackFormer association integration gives a smaller
additional AP@0.7 increase of 0.008; it improves strict-IoU localization but
slightly reduces AP@0.3 and AP@0.5.  Therefore, the association mechanism is
promising but is not yet the main source of the robustness gain.

For the PointPillars reproduction and noise protocol, see
[the baseline record](docs/baseline_reproduction.md).  For the frozen detector
uncertainty head, see [uncertainty training](docs/uncertainty_training.md).

## Spatial TrackFormer association embedding

Each CAV processes its local LiDAR with PointPillars.  For every detected box,
we retain **all** PointPillars BEV cells covered by that box rather than one
anchor cell.  The ROI cells, their relative positions, ego-frame box geometry,
and detector confidence form the proposal input to the Spatial TrackFormer.
The model produces a normalized 128-dimensional embedding; equal physical
objects observed by different CAVs should be close in this space.

The training cache is built from same-timestamp, cross-agent OPV2V proposals
with physical object IDs used only as supervision.  The final
geometry-conditioned association head reached **0.891 cross-agent Top-1
association accuracy** on the clean, held-out official OPV2V test split.

### Association robustness to noise

The frozen clean association model was evaluated on the same 2,170 official
OPV2V test frames while scaling all three perturbations together.  At
`alpha = 1`, the perturbation is 0.2 m position noise, 0.2 degrees heading
noise, and 100 ms delay.  Values use one fixed noise seed (20), so this is a
controlled sweep rather than a multi-seed confidence interval.

| Noise multiplier | Position / heading / delay | Cross-agent Top-1 |
|---:|---|---:|
| 0.0 | clean | 0.8909 |
| 0.5 | 0.1 m / 0.1 deg / 50 ms | 0.8908 |
| 1.0 | 0.2 m / 0.2 deg / 100 ms | 0.8909 |
| 1.5 | 0.3 m / 0.3 deg / 150 ms | 0.8905 |
| 2.0 | 0.4 m / 0.4 deg / 200 ms | 0.5814 |
| 2.5 | 0.5 m / 0.5 deg / 250 ms | 0.5811 |
| 3.0 | 0.6 m / 0.6 deg / 300 ms | 0.4464 |
| 4.0 | 0.8 m / 0.8 deg / 400 ms | 0.3604 |
| 5.0 | 1.0 m / 1.0 deg / 500 ms | 0.2984 |

![Cross-agent Top-1 association accuracy versus noise severity](outputs/association_noise_sweep/association_top1_vs_noise.png)

The curve demonstrates strong resistance over the BELT reference noise level,
followed by a clear degradation at more severe perturbations.  Association
Top-1 measures whether the correct cross-agent object has the highest embedding
similarity; it is not itself a detection AP score.

### Geometry-only ablation

The following ablation removes delay entirely and isolates CAV position and
heading errors.  The same multiplier is applied to either position, heading,
or both, depending on the curve being evaluated.

| Alpha | Position-noise std. | Heading-noise std. | Delay |
|---:|---:|---:|---:|
| 0 | 0.0 m | 0.0 deg | 0 ms |
| 1 | 0.2 m | 0.2 deg | 0 ms |
| 2 | 0.4 m | 0.4 deg | 0 ms |
| 3 | 0.6 m | 0.6 deg | 0 ms |
| 4 | 0.8 m | 0.8 deg | 0 ms |
| 5 | 1.0 m | 1.0 deg | 0 ms |

![Position-only, heading-only, and combined geometry-noise association sweep](outputs/geometry_only_noise_sweep/geometry_only_association_top1.png)

With delay removed, geometry-only perturbations produce only a small Top-1
change through `alpha = 5`; combined position and heading error is the most
damaging of the three conditions.  This isolates delay as the primary cause of
the sharp degradation in the earlier combined noise sweep.

Implementation and cache details are in
[TrackFormer training](docs/trackformer_training.md).

## SWFormer detector study

SWFormer replaces the dense PointPillars BEV backbone with sparse-window
Transformer processing.  In this repository it is currently a **single-sweep,
per-CAV detector**: each CAV runs SWFormer locally, then OpenCOOD late fusion
transforms and merges the decoded boxes.  It has not yet been used as the
backbone for the Spatial TrackFormer proposal cache.

The table below reports globally confidence-sorted AP on the same official
OPV2V test split.  Global sorting is required when calculating a dataset-level
precision–recall curve; the default frame-order accumulation is not a valid
cross-detector AP comparison here.

| Detector checkpoint | AP@0.3 | AP@0.5 | AP@0.7 |
|---|---:|---:|---:|
| PointPillars late fusion, epoch 30 | 0.910 | 0.905 | 0.856 |
| SWFormer late fusion, epoch 112 (best validation) | 0.692 | 0.691 | 0.682 |
| SWFormer late fusion, epoch 119 (last) | 0.690 | 0.689 | 0.681 |

The SWFormer implementation trains stably in this run, but its detector AP is
substantially below the PointPillars reference.  The current result should be
treated as a baseline for controlled SWFormer ablations—not evidence that
SWFormer improves OPV2V late fusion.

![PointPillars and SWFormer precision--recall curves](outputs/swformer_epoch119_eval/pointpillars_vs_swformer_pr_curves.png)

![SWFormer training and validation loss](outputs/swformer_epoch119_eval/swformer_training_validation_loss.png)

### SWFormer training

The commands use the OpenCOOD-SWFormer fork and do **not** enable mixed
precision (`--half`).  Each completed epoch writes an archival
`net_epoch<N>.pth` checkpoint and atomically refreshes `latest.pth`, which also
contains optimizer, scheduler, and random-number-generator state for resume.

#### Early fusion

Early fusion first transforms participating CAV point clouds into the ego frame
and detects from the merged cloud.  The committed early-fusion YAML is already
configured for the local OPV2V path; batch size 1 is deliberately conservative
because a merged cloud uses considerably more memory than a per-CAV cloud.

```bash
PYTHONPATH=external/OpenCOOD-SWFormer \
python -u external/OpenCOOD-SWFormer/opencood/tools/train.py \
  --hypes_yaml external/OpenCOOD-SWFormer/opencood/hypes_yaml/swformer_early_fusion.yaml \
  --epochs 128 \
  --batch-size 1 \
  --temporal-sweeps 1
```

#### Late fusion: validated single-sweep configuration

This command reproduces the settings of the final completed late-fusion run:
single local LiDAR sweep, batch size 8, 128 epochs, Adam peak learning rate
`2e-4`, 8-epoch warm-up from `5e-5`, cosine decay to `1e-6`, gradient clipping
at 5, and a temporal window of one LiDAR sweep.  The learning-rate, augmentation,
loss, and clipping settings are stored in
`swformer_late_fusion.yaml`; the explicit command-line values make the batch,
epoch, and temporal-sweep choices unambiguous.

```bash
PYTHONPATH=external/OpenCOOD-SWFormer \
python -u external/OpenCOOD-SWFormer/opencood/tools/train.py \
  --hypes_yaml external/OpenCOOD-SWFormer/opencood/hypes_yaml/swformer_late_fusion.yaml \
  --epochs 128 \
  --batch-size 8 \
  --temporal-sweeps 1
```

#### Late-fusion temporal-window ablation

Use the same validated settings and change only `--temporal-sweeps`.  For
example, `--temporal-sweeps 3` uses the current LiDAR frame plus two previous
local frames.  It changes the VFE input representation, so start a new run;
do not resume a one-sweep checkpoint with three sweeps.

```bash
PYTHONPATH=external/OpenCOOD-SWFormer \
python -u external/OpenCOOD-SWFormer/opencood/tools/train.py \
  --hypes_yaml external/OpenCOOD-SWFormer/opencood/hypes_yaml/swformer_late_fusion.yaml \
  --epochs 128 \
  --batch-size 8 \
  --temporal-sweeps 3
```

#### Resume an interrupted late-fusion run

Use the same architecture arguments used at the start of the run.  The trainer
loads `latest.pth` when it is present and continues from the next completed
epoch.

```bash
PYTHONPATH=external/OpenCOOD-SWFormer \
python -u external/OpenCOOD-SWFormer/opencood/tools/train.py \
  --hypes_yaml external/OpenCOOD-SWFormer/opencood/hypes_yaml/swformer_late_fusion.yaml \
  --model_dir external/OpenCOOD-SWFormer/opencood/logs/<run-directory> \
  --epochs 128 \
  --batch-size 8 \
  --temporal-sweeps 1
```

## Reproducibility notes

- Keep detector, split, noise seed, and AP accumulation protocol fixed when
  comparing methods.
- Report global confidence-sorted AP for PointPillars/SWFormer detector
  comparisons.
- Report association Top-1 separately from final detection AP; they measure
  different stages of the cooperative-perception pipeline.
- The noise sweep currently uses one seed.  A final paper result should repeat
  each severity across multiple seeds and report mean and variation.
