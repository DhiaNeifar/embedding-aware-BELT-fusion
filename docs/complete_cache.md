# Complete PointPillars research cache

Schema version 3 preserves every downstream-relevant PointPillars result without
duplicating OPV2V's existing raw point-cloud and YAML files.

Per agent and timestamp it stores:

- the native, unpooled `384x100x176` FP16 BEV tensor;
- dense classification logits and anchor-regression maps;
- every decoded/NMS-retained proposal, score, anchor index, and 384-D anchor-cell
  feature;
- physical-object IDs, IoUs, exact local GT boxes, and all native BEV cell indices
  covered by each rotated proposal;
- local-to-ego transformation and original YAML/PCD paths.

The manifest defines every field, coordinate conventions, LiDAR range, voxel
size, box order, dtype, source checkpoint, and configuration. `inventory.json`
is produced by the validator after checking tensor alignment, shapes, ROI bounds,
GT rows, transformations, and source files.

The complete cache is intentionally large. The verified sample uses approximately
0.039 GiB per frame, or approximately 250 GiB for 6,374 training frames.
