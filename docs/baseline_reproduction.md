# OpenCOOD PointPillars late-fusion baseline

## Reproduced results

Evaluation uses the official OpenCOOD PointPillars late-fusion epoch-30 checkpoint,
the OPV2V Default Towns test split, and OpenCOOD's original non-global AP sorting.

| Setting | AP@0.3 | AP@0.5 | AP@0.7 |
|---|---:|---:|---:|
| Clean | 0.867 | 0.859 | 0.782 |
| Noisy | 0.864 | 0.768 | 0.318 |
| BELT-Fusion README late-fusion baseline | not reported | 0.662 | 0.414 |
| BELT-Fusion README BELT-Fusion | not reported | 0.724 | 0.447 |

The clean AP@0.7 reproduces OpenCOOD's published 0.781 result within display
precision.

## Noise protocol used

The evaluated noisy configuration is
`models/pointpillar_late_fusion_noisy/config.yaml`:

```yaml
wild_setting:
  seed: 20
  async: true
  async_mode: sim
  async_overhead: 100
  loc_err: true
  xyz_std: 0.2
  ryp_std: 0.2
```

In the pinned OpenCOOD implementation:

- `async_mode: sim` and `async_overhead: 100` select the preceding frame for every
  non-ego CAV because OPV2V is treated as 10 Hz.
- `xyz_std: 0.2` is Gaussian pose noise in metres.
- `ryp_std: 0.2` is Gaussian angular-pose noise in degrees; only yaw is applied by
  `BaseDataset.add_loc_noise`.
- Noise is applied only to non-ego CAV poses.
- `add_loc_noise` resets NumPy's RNG to `seed` on every call. Consequently the same
  sampled offset is reused rather than drawing independent noise per CAV/frame.

These values match the magnitudes stated in BELT-Fusion's README: 0.2 m position
noise, 0.2 degree heading noise, and 100 ms delay. The README does not state its
random seed, fixed-versus-random delay mode, noise resampling policy, OpenCOOD
revision, checkpoint, or AP implementation. Therefore the 0.662/0.414 values are
comparison targets, not evidence that the present run is misconfigured.

## BELT-Fusion integration gate

The OpenCOOD `PointPillar` checkpoint outputs:

- `psm`: per-anchor classification logits;
- `rm`: per-anchor box regression.

The BELT-Fusion fusion interface additionally requires:

- decoded per-agent proposals before cross-agent NMS;
- classification evidence / Dirichlet parameters;
- per-proposal regression log variance or covariance;
- classification uncertainty.

Those uncertainty quantities are learned outputs and are absent from the pretrained
OpenCOOD checkpoint. They cannot be reconstructed faithfully from confidence scores.
The BELT-Fusion submodule does not contain a compatible trained checkpoint or a
wired OpenCOOD detector.

The minimum faithful next implementation is:

1. Add a parent-repository OpenCOOD adapter that exposes the 384-channel
   `spatial_features_2d` tensor and preserves per-CAV decoded proposals.
2. Add dense, anchor-aligned evidential-classification and heteroscedastic-regression
   branches initialized alongside the frozen pretrained PointPillars heads.
3. Train and validate those uncertainty branches on OPV2V while initially freezing
   the detector.
4. Calibrate covariance and evidential uncertainty on validation data.
5. Feed decoded boxes, scores, covariance, and evidence into a corrected
   parent-repository BELT fusion adapter.
6. Verify that its original-association mode reproduces the noisy OpenCOOD baseline
   before evaluating uncertainty-aware fusion.

The upstream BELT-Fusion prototype must not be inserted unchanged: its regression
quantifier contains an undefined variable, its multi-agent matching is simplified,
and its box fusion modifies dimensions by center displacement. These issues need
isolated tests and corrected parent-repository implementations.
