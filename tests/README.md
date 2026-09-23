# Tests

Run with `pytest tests/`.

- `test_belt_fusion.py`, `test_uncertainty_integration.py`: proposal schemas,
  feature-to-box alignment, covariance decoding, evidential fusion.
- `test_geometry_association.py`, `test_embedding_pipeline.py`,
  `test_spatial_trackformer.py`: geometry/embedding association and
  TrackFormer pipeline behavior.
- `test_pair_head_association.py`: constrained (Hungarian) assignment
  rejects invalid pairs without letting them block a valid pair elsewhere
  (the bug fixed in `0497cf4`), and the all-agent symmetric pair-head
  grouping mode matches two source CAVs to each other directly, without an
  ego correspondence.
- `test_localization_ransac.py`: RANSAC SE(2) pose correction -- identity at
  zero noise, translation/yaw/combined recovery, inlier/outlier rejection,
  `correspondences_by_source` filtering, and `apply_se2_correction`.
