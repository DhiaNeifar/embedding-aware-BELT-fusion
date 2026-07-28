# Plan-0 design

## Boundary and invariant

Plan 0 changes association, not detection or uncertainty fusion:

`OPV2V -> per-agent detector -> proposal feature -> embedding -> cost matrix ->
Hungarian + rejection -> existing uncertainty fusion`.

Ground-truth identity is derived in a noise-free common frame. Pose noise is applied
only to the estimated agent-to-common transform used by geometry-based matching, so
it cannot change correspondence labels.

## Proposal feature recommendation

### Recommended first choice: rotated BEV feature pooling

Pool the highest-resolution PointPillars neck/BEV map inside each **predicted** box
footprint (for example rotated RoIAlign or a deterministic grid sample to `3x3`),
then flatten or average-project it. This is the simplest reliable Plan-0 choice
because PointPillars naturally produces a spatial BEV feature map, pooling makes it
object-specific, and no second-stage RoI feature is present in the inspected
BELT-Fusion code.

The feature must be sampled in the detector's native agent frame using the unnoised
predicted box. That prevents artificial pose noise from corrupting appearance while
allowing transformed geometry to degrade as intended.

This recommendation is conditional: first connect the real detector and verify the
feature tensor's level, stride, channels, coordinate mapping, proposal decoding, and
checkpoint compatibility on one frame.

### Alternatives

- **Input to `ProbabilisticDetectionHead`:** semantically ideal because it is shared
  immediately before uncertainty heads. In the current snapshot `x` is already
  `(N,C)`, but no code creates it. Use it if the real detector exposes a genuine
  proposal-aligned vector.
- **RoI feature:** usually strongest object-specific candidate, but no RoI module is
  present and PointPillars is normally single-stage. Adding a new RoI network is not
  minimal Plan 0.
- **Dense per-anchor head feature:** easy to obtain near convolutional heads but
  alignment through anchor decoding/NMS is implementation-specific and may be less
  stable than box-region pooling.
- **Box/score/uncertainty vector:** useful as an ablation, not the primary embedding
  input; it would mostly relearn geometry and collapse under the target pose errors.

## Proposal record

Use one versioned record per decoded, post-NMS predicted proposal:

| Field | Type / meaning |
|---|---|
| `schema_version` | integer |
| `scenario_id`, `frame_id`, `timestamp`, `agent_id` | stable source identity |
| `proposal_id` | unique within `(scenario, frame, agent)` |
| `box_agent` | float32 `[x,y,z,l,w,h,yaw]` in native agent coordinates |
| `box_common_clean` | clean transformed box for GT labelling only |
| `predicted_class`, `score` | class integer and calibrated confidence |
| `evidence`, `alpha` | evidential vectors when available |
| `regression_covariance` | float32 `[7,7]`, with coordinate/frame declared |
| `classification_uncertainty` | scalar |
| `detector_feature` | pooled object-specific vector plus feature-spec ID |
| `gt_object_id` | stable physical ID or null |
| `gt_iou` | IoU to assigned GT |
| `gt_valid` | true only above positive threshold |
| `visibility`, `num_points`, `occlusion`, `truncation` | nullable metadata |
| `agent_to_common_clean` | transform provenance/reference |
| `source_checkpoint`, `source_config` | reproducibility identifiers |

Prefer sharded tensor storage plus a compact indexed manifest; never commit it.
Avoid duplicating the dense BEV map. Record schema, coordinate conventions, feature
stride, pooling rule, and hash/checkpoint provenance in shard metadata.

## GT assignment and cross-agent examples

For each `(scenario, timestamp, agent)`:

1. Transform predicted boxes to a noise-free common frame.
2. Compute class-compatible oriented BEV or 3D IoU against common-frame GT.
3. Perform one-to-one prediction-to-GT assignment (Hungarian on `1-IoU`), then
   accept only IoU >= 0.5. Treat IoU < 0.3 as background/unmatched and the interval
   `[0.3,0.5)` as ignore initially. Thresholds remain configurable and should be
   checked against detector recall.
4. Use OPV2V's stable physical actor ID if verified. If the local schema lacks one,
   do not manufacture identity from box proximity; implement and validate an
   explicit scene-level GT identity adapter first.

Construct samples only within the same scenario and timestamp:

- positive: different-agent proposals with equal valid `gt_object_id`;
- easy negative: different valid object IDs;
- hard negative: different IDs, same class, overlapping/nearby in the clean common
  frame and preferably high detector confidence;
- optional unmatched negatives: high-confidence detections below the ignore band,
  never positives.

Keep a grouped training batch keyed by `(scenario, frame)` with agent IDs intact.
Sample positives across diverse baselines/viewpoints and cap repeated pairs so large
agent groups do not dominate. Split by scenario, never by proposal/pair, to prevent
leakage.

## Embedding model and loss

Minimal head:

`feature -> Linear(256) -> ReLU -> Linear(128) -> Linear(D) -> L2 normalize`,
with `D in {32,64,128}` and default 64. Input dimension is resolved from the verified
feature extractor. Start with a frozen detector and train only this head.

Use supervised contrastive loss as the primary objective. A grouped batch can
contain multiple observations of one physical object and many natural negatives, so
all valid same-ID cross-agent views become positives without arbitrary triplet
selection. Restrict positives to different agents; mask same proposal/self pairs;
ignore invalid GT matches. Temperature defaults to 0.07 and must be validated.
Hard-negative sampling remains useful at the batch construction level. Report a
pairwise BCE or triplet objective only as an ablation.

## Matching integration

For two agent proposal sets, compute:

- original geometry `G_ij`: current XY-center distance, retained exactly for the
  baseline;
- embedding `E_ij = 1 - cosine(e_i,e_j)`;
- uncertainty `U_ij`: a separately specified pairwise uncertainty/discrepancy
  measure. Do not call post-match Mahalanobis output an existing matching cost—the
  current BELT-Fusion matcher does not use uncertainty.

Modes:

```text
original:       C = G
embedding_only: C = E
combined:       C = lambda_g * norm(G)
                    + lambda_e * norm(E)
                    + lambda_u * norm(U)
```

For Plan 0, set `lambda_u=0` until `U` is defined and calibrated. Normalize geometry
and uncertainty using validation-set robust scales (fixed percentiles or clipped
CDFs), not per-matrix min/max, which changes meaning with scene composition.
Cosine cost already has a known range; optionally map it to `[0,1]`.

Apply class incompatibility masks, run SciPy Hungarian assignment, then reject each
assigned pair if `cost > max_cost` or cosine similarity is below the configured
minimum. Calibrate rejection thresholds on validation association F1; never tune on
the test set. Preserve the original 5 m rejection for the exact baseline.

The parent fusion adapter should output the tuple format currently consumed by
`UncertaintyAwareAdaptiveFusion.forward` and delegate accepted pairs to
`fuse_matched_pairs`. Unit-test matching separately because the upstream multi-agent
bookkeeping is pairwise and not a general clusterer.

## Evaluation

### Association level

For every evaluated agent pair, define a predicted match as accepted by assignment
and a true match as equal valid physical ID. Aggregate micro and macro:

- precision, recall, F1;
- correct assignments / accepted assignments and / eligible GT correspondences;
- false matches and misses;
- results by translation/heading noise, range, visibility, and agent pair.

Count proposals with no valid GT separately so detector errors are not confused with
association errors. Freeze proposal sets across matching modes and noise levels.

### Cooperative perception

Run identical detections and unchanged fusion for original, embedding-only, and
combined matching. Report AP@0.5 and AP@0.7 under the Cartesian product (or a clearly
declared one-axis-at-a-time subset) of:

- translation std: `0, 0.1, 0.2, 0.5, 1.0 m`;
- heading std: `0, 0.2, 0.5, 1.0, 2.0 degrees`.

Use deterministic seeds and multiple noise realizations with confidence intervals.
Validate the transform convention by injecting a known deterministic offset before
any stochastic sweep. Expensive experiments remain out of scope until the pipeline
smoke test passes.

### Communication

Report both incremental embedding bytes and total declared payload:

`bytes/object = box fields + evidence/score + covariance representation + embedding`.

For embedding dimension `D`, the incremental cost is `4D` bytes in FP32 and `2D`
bytes in FP16: at `D=64`, 256 and 128 bytes respectively. Total bytes depend on
whether covariance is dense (49 values) or diagonal (7 values), and on protocol
headers; report both representation and assumptions explicitly.

## Staged implementation checklist

- [x] Pin clean BELT-Fusion and TrackFormer submodules.
- [x] Add parent package layout, external-path template, Plan-0 config, artifact
  ignores, and codebase/design documentation.
- [ ] Read-only inventory one local OPV2V scenario; document IDs, transforms,
  coordinate conventions, visibility fields, and split structure.
- [ ] Identify/provide the trained PointPillars implementation, detector config,
  checkpoint, and uncertainty checkpoint.
- [ ] Build a one-frame detector adapter smoke test; assert decoded proposal and
  uncertainty tensor shapes.
- [ ] Expose the highest-resolution BEV feature via wrapper/hook and validate
  box-to-feature coordinates visually/numerically on one frame.
- [ ] Implement proposal schema and rotated BEV pooling with synthetic unit tests.
- [ ] Extract a tiny, non-committed multi-agent shard; verify physical IDs and IoU
  assignment manually.
- [ ] Implement grouped pair sampling, projection head, and supervised contrastive
  loss; overfit a tiny shard as a sanity check.
- [ ] Implement cost normalization, three matching modes, Hungarian rejection, and
  association metrics with synthetic tests.
- [ ] Wrap the upstream fusion seam and verify the original mode reproduces baseline
  matches exactly.
- [ ] Implement deterministic coordinate/noise tests, then a small evaluation
  sweep.
- [ ] Only after all gates pass, generate full proposal shards and train Plan 0.

## Next smallest coding task

Implement a read-only OPV2V schema/transform inventory script and a detector-adapter
interface test. The detector/checkpoint choice is the gating dependency: feature
extraction code written before resolving it would encode unsupported tensor and
coordinate assumptions.

## Principal risks

1. The upstream repository is incomplete, so the true detector seam may live in an
   unprovided codebase.
2. Stable cross-agent object identity and transform conventions must be confirmed in
   the local OPV2V layout.
3. Uncertainty outputs/checkpoints may not be calibrated or aligned to decoded
   proposals.
4. Pose noise can accidentally leak into feature sampling or labels unless clean and
   noisy transforms are separated.
5. Existing fusion contains prototype limitations/defects that could dominate final
   AP even if association improves.
