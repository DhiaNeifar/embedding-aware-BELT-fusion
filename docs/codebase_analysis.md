# Codebase analysis

## Scope and pinned revisions

This analysis covers BELT-Fusion `511d621013481aea3b4dfc62d6b0705106a304bf`
and TrackFormer `e468bf156b029869f6de1be358bc11cd1f517f3c`. Neither submodule was
modified.

## Executive finding

The checked-in BELT-Fusion is a research prototype, not a complete executable
PointPillars/OPV2V stack. It defines uncertainty heads, a simplified fusion module,
dataset shells, and illustrative scripts. It does **not** define the configured
`BELTFusion`, `PointPillarsBackbone`, or `FPN` classes; produce detector proposals
from point clouds; transform detections between agent frames; inject pose noise; or
implement real 3D NMS. Several scripts expect a non-existent `data["features"]`.
Consequently, extracting real proposal features requires first connecting a trained
detector implementation/checkpoint and the dataset's actual schema.

### Known defects relevant to reuse

- `RegressionUncertaintyQuantifier.forward` in
  `external/BELT-Fusion/belt_fusion/models/fusion_modules/uncertainty_fusion.py`
  refers to `self_epsilon`, which is undefined (it should be `self.epsilon`).
- `build_dataset` passes the config key `type` through `**dataset_cfg`, but
  `OPV2VDataset.__init__` and `DAIRV2XDataset.__init__` do not accept `type`.
- Config pipelines are lists of dictionaries, while dataset `__getitem__` calls each
  element as a function.
- OPV2V train/val/test all use `train_pipeline`; no OPV2V-specific test pipeline is
  supplied.
- `fuse_matched_pairs` changes box dimensions by the center displacement. That is
  not a valid box fusion rule and must be validated before final AP experiments.
- Multi-agent matching is only ego-to-each-other-agent, and global `processed`
  bookkeeping prevents an ego proposal from collecting observations from more than
  one partner.

These are upstream facts, not assumptions. Plan 0 should wrap and test the minimal
required seam rather than silently repairing unrelated behavior.

## BELT-Fusion

### Entry points and OPV2V data

- `external/BELT-Fusion/tools/train.py`: `parse_args`, `main`. Selects OPV2V with
  `--dataset opv2v`, constructs `OPV2VDataset`, then trains only
  `ProbabilisticDetectionHead`. It assumes `data["features"]` already exists and
  explicitly describes backbone integration as simplified.
- `external/BELT-Fusion/tools/test.py`: `parse_args`, `main`,
  `compute_covariance_from_log_var`. It constructs the uncertainty head and
  `UncertaintyAwareAdaptiveFusion`, assumes agent feature tensors are present, and
  serializes results. It does not build PointPillars.
- `external/BELT-Fusion/configs/belt_fusion_pointpillars_opv2v.py`: inherits the
  DAIR-V2X config, selects `OPV2VDataset`, one `Car` class, range
  `[-70.4,-70.4,-2,70.4,70.4,4]`, and pickle files named
  `opv2v_infos_{train,val,test}.pkl`.
- `external/BELT-Fusion/belt_fusion/datasets/opv2v_dataset.py`:
  `OPV2VDataset.get_data_info` expects `ego_lidar_path`, optional
  `connected_vehicles[*].{lidar_path,pose}`, and optional top-level
  `gt_bboxes_3d`/`gt_labels_3d`. It exposes timestamp but no scenario ID, agent ID,
  shared object ID, transforms, or visibility.

The local raw OPV2V directory must be inspected read-only in the next integration
step. The repository does not include an info-file generator, so the expected pickle
schema cannot yet be asserted to match that dataset.

### PointPillars and feature availability

`external/BELT-Fusion/configs/belt_fusion_pointpillars_dairv2x.py` declares:

- model type `BELTFusion`;
- backbone type `PointPillarsBackbone`, stages `[64,128,256]`;
- `FPN` neck with 256 output channels and three outputs;
- `ProbabilisticDetectionHead(in_channels=256, num_regs=7)`.

No implementation or registry entry exists for the model, backbone, or neck.
`ProbabilisticDetectionHead.forward` accepts an already object-specific tensor
`(N, in_channels)` and feeds the exact same tensor to regression and classification
branches. Thus the ideal "immediately before heads" feature seam is the argument
`x`, but no upstream code constructs proposal-aligned `x`. No RoI extractor is
present. No BEV tensor is exposed by executable BELT-Fusion code.

### Detection and uncertainty structures

`external/BELT-Fusion/belt_fusion/models/uncertainty_heads/probabilistic_head.py`:

- `ProbabilisticRegressionHead.forward(x)` returns `reg_mean (N,7)` and
  `reg_log_var (N,7)`.
- `EvidentialClassificationHead.forward(x)` returns non-negative `evidence`,
  `alpha=evidence+1`, and `cls_uncertainty=K/sum(alpha)`.
- `ProbabilisticDetectionHead.forward(x)` returns a dictionary with
  `reg_mean`, `reg_log_var`, `evidence`, `alpha`, and `cls_uncertainty`.

`tools/test.py` converts `reg_log_var` into diagonal `(N,7,7)` covariance and
constructs one per-agent detection dictionary:
`boxes`, `scores=alpha/sum(alpha)`, `covariances`, `evidence`.

`UncertaintyAwareAdaptiveFusion.forward` consumes a list of those dictionaries.
Its fused output dictionaries contain `boxes`, `centers`, `reg_uncertainty`,
`cls_belief`, `cls_uncertainty`, and `pred_class`. There is no formal proposal
class/dataclass and no scenario/frame/agent metadata.

### Coordinates, matching, and fusion

- **Common frame:** not implemented. `OPV2VDataset` returns connected-agent poses,
  but no code consumes them or transforms boxes. `match_objects` therefore assumes
  its input boxes already share coordinates.
- **Cost:** `UncertaintyAwareAdaptiveFusion.match_objects` computes pairwise
  Euclidean distance between box XY centers.
- **Assignment:** the same method calls SciPy
  `linear_sum_assignment(dist_matrix.cpu().numpy())`, then rejects pairs at
  distance >= 5 m.
- **Fusion handoff:** `UncertaintyAwareAdaptiveFusion.forward` slices matched
  `boxes`, `covariances`, and `evidence`, then calls `fuse_matched_pairs`.
- **Regression uncertainty:** `RegressionUncertaintyQuantifier.forward` computes a
  Mahalanobis discrepancy from paired boxes and their average covariance.
- **Classification uncertainty:** `ClassificationUncertaintyQuantifier` derives
  evidential belief/uncertainty and `ds_fusion_two_agents` applies Dempster-Shafer
  fusion.
- **Pose noise:** no cooperative pose-noise path exists. The only
  `translation_std=[0,0,0]` is a standard single-cloud augmentation in the DAIR
  test config; it is not agent-pose noise. Heading noise is absent.

The exact association adapter seam is therefore
`UncertaintyAwareAdaptiveFusion.match_objects`; the cleanest Plan-0 integration is
a parent-repository subclass/wrapper overriding matching and adding embeddings to
the per-agent dictionaries, while delegating matched-pair fusion unchanged.

## TrackFormer

### Decoder embeddings

- `external/trackformer/src/trackformer/models/detr.py`: `DETR.forward` obtains
  decoder states `hs` and publishes the final pre-normalization decoder state as
  `out["hs_embed"] = hs_without_norm[-1]`.
- `external/trackformer/src/trackformer/models/deformable_detr.py`:
  `DeformableDETR.forward` publishes `out["hs_embed"] = hs[-1]`.
- Both use the same query states to produce `pred_logits` and `pred_boxes`, making
  them object-query-specific representations.

### Identity preservation

`external/trackformer/src/trackformer/models/detr_tracking.py`:

- `DETRTrackingBase.forward` runs preceding frames and matches their predictions to
  preceding-frame targets.
- `add_track_queries_to_targets` uses ground-truth `track_ids` to find identities
  present in both frames. It copies selected previous `hs_embed` states into
  `target["track_query_hs_embeds"]`, carries detached previous boxes, and constructs
  track-query masks including optional false positives/negatives.
- `DETR.forward` prepends these states as decoder targets; deformable TrackFormer
  does the analogous concatenation in
  `DeformableTransformer.forward` in `deformable_transformer.py`.

At online inference, `external/trackformer/src/trackformer/models/tracker.py`
maintains active/inactive tracks and query state across sequential frames. This is
temporal state propagation, not a standalone metric embedding system.

### Supervision

- `external/trackformer/src/trackformer/models/matcher.py`:
  `HungarianMatcher.forward` combines class, L1 box, and generalized-IoU costs.
  When track queries exist, it hard-constrains a valid track query to its known
  current-frame target and prevents false track queries from matching.
- `external/trackformer/src/trackformer/models/detr.py`: `SetCriterion.forward`
  applies classification (`loss_ce`, optionally focal), L1 box (`loss_bbox`), and
  generalized-IoU (`loss_giou`) losses after Hungarian assignment, including
  auxiliary decoder layers. False-positive track queries receive no-object
  supervision/weighting.

There is no explicit contrastive, triplet, or re-identification loss over
`hs_embed`. Identity is learned implicitly because the previous object's query
state is routed to, and supervised against, the same track ID in the next frame.

### Reuse decision

Reasonable concepts to reuse:

- object-specific latent features shared by prediction heads;
- identity labels to route supervision;
- hard positives plus injected hard false-positive queries;
- one-to-one assignment with an explicit no-match/rejection mechanism;
- attention/query embeddings as conceptual evidence that compact object state can
  carry identity.

Do not directly reuse:

- previous-frame query propagation, active/inactive track state, temporal motion,
  track aging, frame sampling, or 2D normalized boxes;
- TrackFormer's detector architecture or hard track-query constraints;
- its loss as if it were an explicit metric loss.

Plan 0 compares independent, same-timestamp proposals from different agents. It
should train a small metric projection on 3D-detector features, preserving agent
grouping, rather than importing TrackFormer's temporal pipeline.

## Adapter/modification map

Parent-repository additions (no upstream edits):

1. `src/.../integration/detector_adapter.py`: connect the actual PointPillars
   detector/checkpoint, return decoded proposals, uncertainty fields, BEV features,
   and transform metadata.
2. `src/.../features/bev_roi.py`: proposal-aligned rotated BEV pooling.
3. `src/.../data/proposal_record.py` and `pair_dataset.py`: schema, GT assignment,
   grouped cross-agent sampling.
4. `src/.../embeddings/head.py` and `losses.py`: projection head and supervised
   contrastive objective.
5. `src/.../matching/costs.py` and `assignment.py`: normalized costs, Hungarian
   assignment, and rejection.
6. `src/.../integration/fusion_adapter.py`: subclass/wrap
   `UncertaintyAwareAdaptiveFusion`, replace only `match_objects`, and pass accepted
   pairs into `fuse_matched_pairs`.
7. `src/.../evaluation/association.py`, `detection.py`, and `communication.py`.
8. `scripts/extract_proposals.py`, `train_embedding.py`, and
   `evaluate_pose_noise.py`.

An upstream patch is only justified if the eventual detector hides the necessary BEV
feature; prefer a forward hook or wrapper first. Any patch must be isolated,
documented, and submitted separately.

## Unresolved questions

1. Which trained PointPillars implementation and checkpoint produced the intended
   BELT-Fusion detections? This repository contains neither.
2. Does local OPV2V expose stable physical object IDs across CAV YAML files, and
   what are its coordinate conventions and split layout?
3. Are uncertainty-head weights available and calibrated for that detector?
4. Is the desired association ego-centric pairwise matching or true multi-agent
   clustering? Existing fusion supports only the former reliably.
5. Does final evaluation require preserving the upstream fusion formula despite the
   dimensional-update defect noted above?
