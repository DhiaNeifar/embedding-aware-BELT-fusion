# Close the gap to Chenyi's actual CVPR draft, then attack the real bottleneck

## Context

Found Chenyi's current, in-progress CVPR submission: an untracked directory
`AlignFormer_2027/` inside `/mnt/external/workspace/dhia/embedding-aware-BELT-fusion`
(never committed to git, discovered by searching the working tree directly).
It is a complete draft (abstract, method, experiments, discussion) describing
his CURRENT method and protocol — not PQ-STF/BELT/TrackFormer, which his own
paper never mentions. This supersedes everything in the previous plan that
was based on inference from his old code rather than his actual current
results.

**Point 1 — PQ-STF/IE-STF/annealing/BELT/TrackFormer are confirmed dead.**
Chenyi's method ("\paper" in the draft, i.e. his own "AlignFormer") sends only
the 7-float detected boxes, exactly what the architecture we already ported
as `alignformer_pva` does. Confirmed by reading his method section directly:
Propose (SuperGlue-style Sinkhorn + heading-augmented weighted Kabsch) →
Verify (gated two-point RANSAC re-solve, gates 6/3/1.5/0.75/0.5m) → Abstain
(Wald test, agreement rule with a consensus floor). Hyperparameters match
what we ported almost exactly (d=256, 4 layers, 4 heads, heading_lambda=2m,
20 Sinkhorn iterations) — **the architecture port was correct and is current.
No reason to revisit PQ-STF/BELT/TrackFormer; they do not appear anywhere in
his own current comparison.**

**Point 2 — where his method falls short, and where OURS falls short of his.**
He already ran exactly the "compare to oracle, quantify the gap" study the
user asked for (`sec/4_experiments.tex`, Tables `main`/`buckets`/`association`):
- Association is NOT the weak point: geometry-only Sinkhorn gets 99.4% Top-1
  correct; verified correspondences hold 0.94 precision at *every* noise
  level and reach the oracle-correspondence residual (0.41m) on dense pairs.
- The real, persistent gap to oracle is the **sparse bucket** — pairs sharing
  only 1-2 objects (23% of V2X-Real test frames): AP@0.7 there is ~0.25
  against oracle's 0.426, a gap that holds at every noise level, because one
  or two points give too little geometric evidence to fit a precise pose
  from, not because matching is wrong. Dense pairs (3+ shared, 67% of frames)
  are already essentially at the oracle's precision.
- **Separately: our own measured numbers on the same OPV2V dataset/noise
  points (AP@0.7 0.661/0.366/0.451 at 0.2m/1.0m/2.0m+10°) are far below his
  reported ~0.85 flat**, even though the architecture matches. Checked our
  own config against his tuned one and found concrete, fixable gaps:
  - We used `variance_model=UNWEIGHTED` (uncalibrated) throughout — his own
    ablation states the fitted inverse-variance weighting is specifically
    "what makes the difference" on OPV2V (unweighted loses 0.012 AP to
    FreeAlign there; weighted recovers it). This was Step 3 of the old plan
    and was never done.
  - Our `refine.py:DEFAULT_GATES_M = (2.0, 1.0, 0.5)` (3 gates) vs. his tuned
    `(6, 3, 1.5, 0.75, 0.5)` (5 gates, starts much wider — his own ablation
    explains why: "the proposal can be metres off on the pairs that matter
    most").
  - `AgreementConfig.consensus_floor` defaults to `0` (off) in our port; his
    tuned value is 4, worth +0.012 AP on V2X-Real's sweep mean on its own.
  - Our integration's `--alignformer-pva-abstain-level` default (0.05) vs.
    his tuned `alpha=0.2` (selected on validation among {0.5,0.2,0.1,0.05,0.01}).
  - We trained single-stage (match_nll + corner_loss jointly from epoch 0);
    he trains match_nll first (stage 1, noise ≤0.5m) then the pose head
    under the full curriculum (stage 2).
  - Our post-correction multi-agent grouping (`alignformer_pva/inference.py:
    associate_simple_by_geometry` + `fuse_groups_score_weighted`, built new
    for this integration since no embedding-based grouping existed) has
    never been checked against an oracle-correspondence ceiling the way his
    Fig. `residuals` sanity-checks his own fit.

  **This is almost certainly most of our gap to his numbers, and it is all
  tuning/calibration on code that already exists — not new architecture.**

**Point 3 — a new embedding (ROI padding/context, LiDAR color) was already
tried by Chenyi, in a stronger form, and documented not to help.**
`sec/4_experiments.tex` ("Appearance does not help, and why") and
`sec/3_method.tex` ("What is transmitted"): a cross-agent contrastive head on
DINOv2 patch features with hard negatives reaches AUC 0.885-0.90 on exactly
the sparse pairs we'd hope an embedding helps — far beyond what a simple
ROI+padding+color embedding could realistically achieve — and fed into the
matcher it *costs* 0.011 AP on the sweep mean. Mechanism, stated directly:
geometry alone already gets 99.4% of associations right, verified
correspondences already hit oracle residual on dense pairs, so "there is
almost nothing for appearance to fix," and appearance tokens get displaced
by the very pose error the system exists to correct. A plainer 128-d
per-object LiDAR embedding (closer in spirit to our idea) gave a small,
inconsistent benefit (+0.013 on validation, within noise on test, no stable
sign) and was removed as not worth 18x the bandwidth. On "a proper
association method, not top-1 classification": Sinkhorn/SuperGlue-style soft
assignment with dustbins — which `alignformer_pva` already implements — *is*
the literature-grounded answer the user asked for; no further literature
search is needed on this point, it's already resolved by the architecture we
ported.

**Decision: stop the context-padded-embedding experiment** (was in progress,
now halted — background training/eval processes killed). It would very
likely reproduce Chenyi's documented negative result rather than close our
actual gap, which per Point 2 is calibration/tuning, not representation.

**Point 4 — CAV-CAV error sharing.** Chenyi's own Limitations section: he
tested multi-agent pose-graph consistency and "it added nothing here because
a third agent rarely bridges a sparse pair," but flags it "may help on
denser deployments" — his evaluation is two-agent-pair only (V2X-Real: one
CAV + infra per pair; OPV2V: ego + one CAV). **Checked our own OPV2V complete
cache directly: our frames have 3 agents each (ego + 2 CAVs), not 2** — i.e.
exactly the denser setting he flags as untested. This is a legitimate, not
yet disproven, direction for us specifically, but only worth pursuing after
Point 2's calibration gap is closed (no point fusing multiple noisy estimates
from an uncalibrated base).

**Point 5 — C++ for the non-GPU post-processing.** Nothing in Chenyi's draft
bears on this; it's a separate, valid engineering point. Our own per-frame
Python evaluation loop (`evaluation/belt_fusion.py`) is the slow part of
every sweep (10-30+ min per noise point today), and we are about to adopt a
much denser protocol (8 sigma values × 5 seeds, per Chenyi's own protocol,
vs. our current 3 points) — this will make the slowness much more costly.
Detector inference already writes to a cache (boxes/scores/gt_ids), so the
pattern the user has used successfully before — infer once in Python, cache
it, then do noise injection + NMS + fusion + AP in a fast compiled path —
applies directly here.

## Plan

### Phase 0 — Adopt Chenyi's protocol (do this first; it's how every later phase is judged)

Rebuild our OPV2V evaluation sweep to match `sec/4_experiments.tex`'s
protocol exactly, so our numbers are directly comparable to his and to each
other phase-over-phase:
- Noise sweep σ ∈ {0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0} (both position-std-m
  and heading-std-deg set to σ, matching his "σ metres in x/y and σ degrees
  in yaw" convention), 5 seeds on test (3 on validation), paired per-seed
  differences as the statistic — not our own ad hoc 3-point/mixed-heading
  sweep.
- Metrics: AP@0.3/0.5/0.7 per σ and the sweep mean, **plus the shared-object
  bucket breakdown** (0 / 1-2 / 3+ shared ground-truth objects between ego
  and the CAV) — this is what actually shows where a change helps.
- Rows: `uncorrected` (today's naive late fusion), `oracle` (correct every
  CAV with its true pose — check if `evaluation/belt_fusion.py` already has
  an oracle-pose correction path to reuse before building one), and
  `alignformer_pva`.
- This phase is infrastructure only (sweep script + bucket-breakdown
  reporting), no model changes.

### Phase 1 — Close our own gap to Chenyi's reported numbers on the SAME architecture

Apply the concrete, already-identified fixes from Point 2, in order of
expected impact, re-running Phase 0's sweep after each to attribute the gain:
1. **Calibrate the inverse-variance model** (`alignformer_pva/variance.py`,
   Step 3 from the old plan, never done): measure cross-agent centre/heading
   residual of the same ground-truth object under the TRUE pose, binned by
   detector confidence, on held-out validation data from our own detector
   (mirrors his "0.53m RMS low-confidence decile vs 0.21m high-confidence
   decile" measurement) and fit the two-parameter power law
   `CorrespondenceVarianceModel` expects, replacing `UNWEIGHTED`.
2. **Retune `refine.py`/`abstain.py` defaults on our own validation split**,
   the same way his Table `ablation` does: widen `DEFAULT_GATES_M` toward his
   `(6, 3, 1.5, 0.75, 0.5)`, turn on `AgreementConfig.consensus_floor` (try 4
   first), sweep the Wald `abstain_level`/`alpha` among his candidate set
   `{0.5, 0.2, 0.1, 0.05, 0.01}` — select by validation AP, not guessed
   defaults.
3. **Switch `train.py` to a two-stage curriculum**: stage 1 trains
   trunk+matcher on `match_nll` alone with noise capped at 0.5m; stage 2
   warm-starts from it and trains the pose head under the full 0→2m
   curriculum. (`train.py` is already additive/flag-driven; add this as a
   `--stage` option rather than a rewrite.)
4. **Audit the post-correction fusion/grouping step**: check
   `alignformer_pva/inference.py`'s `associate_simple_by_geometry` +
   `fuse_groups_score_weighted` against an oracle-correspondence ceiling
   (feed ground-truth matches directly into the same fusion code, the same
   sanity check his Fig. `residuals` does) to confirm it isn't silently
   capping our AP independent of alignment quality.

Acceptance gate: does our own `alignformer_pva`, same architecture, land
close to his reported curve (flat ~0.83-0.85 AP@0.7 on OPV2V) under Phase 0's
protocol? This is the real "did we faithfully reproduce his result" check
before building anything past it.

### Phase 2 — Attack the actual remaining gap: the sparse-pair bucket

Only after Phase 1's gate passes. Per Point 2, the real headroom (even in
Chenyi's own numbers) is pairs sharing 1-2 objects, where there's too little
single-frame geometric evidence for a precise fit. Two directions, both
explicitly justified by his own results rather than by hypothesis:
- **Temporal accumulation (idea 4)**: a sparse pair this frame may have been
  a denser pair last frame or will be next frame. Accumulating shared-object
  evidence across consecutive frames turns a sparse single-frame problem into
  a denser multi-frame one. Adapt `embeddings/trackformer.py`'s unused
  time-axis track-query propagation (see prior investigation) or a simpler
  running accumulation of matched correspondences, keyed by `gt_ids` first
  (ground-truth continuity, to measure the upper bound) before any real
  tracker.
- **Multi-CAV error sharing (idea 2)**: justified specifically because our
  OPV2V frames have 3 agents (not Chenyi's 2), the exact "denser deployment"
  condition his own limitations section flags as untested. Extend
  correspondence computation to CAV-CAV pairs (`embeddings/all_agent_association.py:all_agent_pairs`
  already computes these, unused for localization today — see
  `integration/localization.py:correspondences_by_source`'s ego-only
  restriction), get multiple independent pose-error estimates per CAV, fuse
  by precision-weighting (same algebraic family as `integration/belt_fusion.py:fuse_group`,
  applied to pose-error estimates instead of box observations).

GNN: still deprioritized for the matching/embedding step (Chenyi's own
negative-result pattern generalizes). Could revisit only as an optional
fusion mechanism inside idea 2's multi-estimate combination, compared against
the precision-weighted baseline, not before.

### Phase 3 (parallel engineering, not gated on the above) — C++ fast path

Once Phase 0's cache-driven sweep format is settled, build a small compiled
(pybind11 or similar) module that takes cached per-frame boxes/scores (we
already have the `complete`/`geometry_only` caches with this), and does noise
injection + the alignment correction application + NMS + AP accumulation —
the part of `evaluation/belt_fusion.py`'s per-frame loop that doesn't need
the GPU/detector. Mirrors the user's own prior pattern (infer once in
Python, cache it, post-process fast in C++). This pays for itself as soon as
Phase 0's 8σ × 5-seed protocol replaces our 3-point sweep.

## Explicitly out of scope / deprioritized

- A new context-padded or camera-color embedding for the base
  association/pose step — already shown not to help, in a stronger form than
  we were building (Point 3).
- Any further PQ-STF/IE-STF/BELT/TrackFormer work.
- GNN for matching/embedding.
- Any changes to the labmate's repo (`embedding-aware-BELT-fusion`,
  `AlignFormer_2027/` included) — read-only reference, never modified.

## Verification

- Phase 0: the new sweep script reproduces the OLD 3-point numbers at the
  overlapping σ values (0.2, 1.0, 2.0) as a sanity check before trusting the
  new bucket-breakdown reporting.
- Phase 1: each of the 4 fixes is measured independently (re-run the sweep
  after each), not applied all at once blind — exactly so we know which one
  actually closes the gap, mirroring Chenyi's own ablation discipline.
- Phase 2: acceptance is a measured AP gain specifically in the sparse-pair
  bucket (not just the overall mean) at ≥1 noise level, consistent with
  where Point 2 says the real headroom is.
- Full test suite (currently 178 passing) stays green throughout; existing
  `naive_late`/`ego_only`/`belt_geometry` sweep outputs remain unaffected by
  any `alignformer_pva`-only change.

## Update (2026-10-17) — scope narrowed, concrete next idea, in progress

Superseding Phase 2's generic "temporal accumulation" with a specific,
literature-grounded mechanism discussed and approved with the user:

**The idea**: instead of treating every frame independently, maintain a
running, per-CAV empirical distribution of past pose corrections (an
extension of the James-Stein shrinkage already in `alignformer_pva/shrinkage.py`,
generalized from one fixed global prior to a per-CAV, accumulated one). This
is a Bayesian MAP prior fed into two places, not a parallel fourth stage:
- **Propose**: one more term in the weighted Kabsch solve, alongside the
  existing detector-confidence-based weighting (`variance.py`).
- **Abstain**: (a) a richer, CAV-specific reference variance for the Wald
  test, and (b) a smarter fallback — when the test says "don't trust this
  frame," fall back to the learned historical correction instead of to zero
  correction (today's behavior), which is itself just a naive, unexamined
  prior.

Motivation (from a literature pass, not Chenyi's paper — he treats pose
error as a stateless per-frame unknown): GNSS multipath/NLOS is a major real
cause of this error and is well-documented as non-i.i.d. across time/location
(can exceed 100m in dense urban NLOS; existing GNSS literature already uses
cross-vehicle consistency checks to fight exactly this). Caveat: the error is
reported as non-Gaussian/jumpy (tied to local geometry), not smooth drift —
so this is framed as "characterize the distribution and weight by precision,"
not "track a smoothly-evolving state," which is why it lives inside the
existing Procrustes/Abstain weighting machinery rather than a new Kalman
filter.

**Scope decision**: OPV2V only, for now. V2X-Real is deferred until the user
obtains a trained backbone for it; our own cache/extraction pipeline is
OPV2V-format-specific and would need a (small, not from-scratch) adapter for
a new dataset regardless.

**Inference reuse checked and rejected**: investigated reusing cached
PointPillars inference from the `GenCPM`/`OpenCPM` repos (same checkpoint,
byte-identical per MD5). Not usable here: that cache is run over GenCPM's own
synthetic CARLA scenes, not the official OPV2V train/validate/test split
Chenyi's numbers and our own prior sweep are measured on, and has a different
(variable 5-8, or fixed-15) agent count per scenario rather than OPV2V's 3.
Using it would make results incomparable to the one number that matters here
(Chenyi's reported OPV2V curve). **Decision: skip it, reuse our own existing
cache** (`outputs/pointpillars_complete_cache_train_v3`, 6,374 official train
frames, noise-free, already verified; plus the existing official-test-split
frame caches under `outputs/frame_cache/`) — zero new inference needed.

**Chenyi's reported OPV2V numbers are now the standing baseline/target**
(Table `opv2v` in `AlignFormer_2027/sec/4_experiments.tex`, AP@0.7 at
σ=0/0.2/0.4/0.6/0.8/1.0/1.5/2.0): uncorrected `.8964/.5978/.3158/.2154/.1844/
.1763/.1799/.1898`; FreeAlign (AP-selected) `.8406/.8344/.8236/.8196/.8178/
.8175/.8174/.8172`; his full AlignFormer `.8526/.8433/.8320/.8282/.8277/
.8264/.8262/.8252`. Every future number we report should be measured under
his exact protocol (Phase 0) and placed next to this table directly.

**Two parallel engineering streams now in progress** (independent, no file
overlap):
1. C++ fast post-processing path (Phase 3) — needed to make the full 8σ × 5-seed
   protocol tractable (naive Python: ~8h/dataset; the goal is minutes).
2. The per-CAV accumulated-prior idea above, implemented in `alignformer_pva`,
   trained and unit-tested on the existing cache.

Once both land: run the full Chenyi protocol on OPV2V test, report AP@0.3/0.5/0.7
per σ and the sweep mean next to his table above, plus the shared-object
bucket breakdown. This is the acceptance gate for everything going forward.

## Result — Phase 1, point 1 (variance calibration), done

Per the user's decision, deferred the per-CAV accumulated-prior idea (it
needs an estimable source of persistent, CAV-specific heterogeneity that the
current, unmodified noise model doesn't have) and did the already-planned,
guaranteed-real-leverage fix instead: calibrated `alignformer_pva/variance.py`
on our own detector/data.

**New file**: `alignformer_pva/calibrate_variance.py` — measures detector
disagreement (centre + heading, folded for the 180° ambiguity) between
ego/CAV detections of the same `gt_id` under the TRUE pose, on the
scenario-disjoint validation role, fits the same Gamma-deviance MLE
"score_additive" power law the labmate's own script uses (methodology
inspected read-only via `git show`, not his numbers). **Found and fixed a
real bug during this work**: the heading-fold helper (`_fold_sign`) used
`unsqueeze(0)` instead of `unsqueeze(1)` before calling
`procrustes.heading_orientation`, which collapses every matched pair into one
shared `(1, K, K)` cross-comparison matrix and reads row 0 instead of the
diagonal — invisible in the original single-pair unit tests (K=1 collapses
the distinction) but corrupting every frame with 2+ matched objects. Caught
because the first calibration run's yaw RMS (33-64°) was 5-10x the labmate's
own reference range; after the fix it reads 3.5-9.8°, matching his range.
Added a regression test (`test_collect_pair_correspondences_folds_each_pair_independently`)
with 2 objects needing different folds, which the bug-reproducing version
fails and the fix passes.

Fitted (score_reference=0.4): `sigma_translation_m=0.2355`,
`translation_exponent=1.153` (labmate: 1.13), `sigma_yaw_deg=4.443`,
`yaw_exponent=1.351` (labmate: 1.93) — same order of magnitude as his, as
expected for a different detector/data.

Wired additively into `train.py` (`--variance-config`, stored in the
checkpoint so eval-time loading needs no separate flag) and retrained
(`outputs/alignformer_pva_train_variance_v1`, same 30-epoch/cache/split as
the geometry-only baseline). Validation position error improved 0.578m ->
0.546m. Measured AP on the same 3-point sweep as the baseline:

| noise | AP@0.3 (base -> new) | AP@0.5 (base -> new) | AP@0.7 (base -> new) |
|---|---|---|---|
| 0.2m | 0.834 -> 0.835 | 0.793 -> 0.796 | 0.661 -> 0.666 |
| 1.0m | 0.828 -> 0.831 | 0.700 -> 0.722 | 0.366 -> **0.420** |
| 2.0m/10° | 0.615 -> 0.617 | 0.559 -> 0.553 | 0.451 -> 0.442 |

Real, modest, mixed improvement — concentrated at the medium-noise point
(+0.054 AP@0.7), roughly flat at low noise, a small regression at the
highest noise point. Consistent with Chenyi's own finding that this specific
fix is real but not a game-changer (~0.01-0.02 AP on his data). 200/200
tests passing throughout.

**Next**: still a meaningful gap to Chenyi's reported flat ~0.83-0.85 AP@0.7
curve. Remaining items from Phase 1's original list (retune refine/abstain
gate schedule and thresholds; two-stage training; audit the post-correction
fusion/grouping step) are still open and are the next candidates.

## Result — Phase 1, point 2 (retune refine/abstain hyperparameters), done

Exposed the previously-hardcoded Verify/Abstain knobs as new, additive CLI
flags on `evaluation/belt_fusion.py` (`--alignformer-pva-refine-gates-m`,
`--alignformer-pva-agreement-tolerance-m`, `--alignformer-pva-consensus-floor`;
`--alignformer-pva-abstain-level` already existed), defaults unchanged so
nothing regresses when omitted (confirmed byte-identical via the
`regression_check_*` runs). Built 2 new validation-split frame caches
(σ=1.0m, σ=2.0m+10°) and ran a cumulative ablation on the
variance-calibrated checkpoint, mirroring Chenyi's own ablation-table
practice (one change at a time, not a grid search):

| validation, AP@0.7 | σ=1.0m | σ=2.0m+10° |
|---|---|---|
| baseline (old defaults) | 0.441 | 0.326 |
| + wide gates (6,3,1.5,0.75,0.5) | 0.444 | 0.411 |
| + agreement tolerance 1.0m, consensus_floor=4 (alone) | 0.468 | 0.334 |
| + both combined | **0.472** | **0.451** |
| + abstain_level=0.5 (vs default 0.05) | 0.472 (no change) | n/a (run didn't complete) |

Winner: wide gates + consensus_floor=4 combined, abstain_level left at its
existing default (0.05) — the abstain-level sweep showed no benefit at the
one point it completed, and a second data point's eval run didn't finish
cleanly (cut short by an environment gap, not a code issue; not re-run since
the signal from gates+consensus_floor was already decisive).

Confirmed on the held-out TEST split (the real, reportable number):

| noise | AP@0.3 | AP@0.5 | AP@0.7 |
|---|---|---|---|
| 0.2m | 0.840 | 0.798 | 0.647 |
| 1.0m | 0.836 | 0.743 | **0.543** |
| 2.0m/10° | 0.673 | 0.634 | **0.521** |

Full progression on AP@0.7: geometry-only baseline (0.661/0.366/0.451) ->
+variance calibration (0.666/0.420/0.442) -> +tuned gates/consensus
(0.647/0.543/0.521). Against the other methods at the same test points
(unaffected by this change): `ego_only` (0.574/0.567/0.565) is still ahead
of `alignformer_pva` at every point, but the gap closed sharply — from 0.123
to 0.044 at the highest noise point — and `alignformer_pva` now clearly
leads `naive_late` (0.747/0.208/0.268) everywhere and is close to
`belt_geometry` (0.735/0.591/0.372). Small, real regression at the lowest
noise point (0.666 -> 0.647) worth tracking but not alarming given the gains
elsewhere. 211/211 tests passing throughout.

**Next candidates**: audit the post-correction fusion/grouping step against
an oracle-correspondence ceiling (cheap, never done, possibly a large lever
independent of the model); two-stage training curriculum; the full 8σ×5-seed
Chenyi protocol with bucket breakdown (still only using the faster 3-point
proxy sweep for iteration speed).

## Reporting convention change (user directive)

Per explicit user instruction: stop reporting `naive_late`/`belt_geometry` as
the comparison set going forward. These were this project's own legacy
baselines, not Chenyi's. The comparisons that matter from here on are
`ego_only` (the sanity bar — fusion is pointless if it doesn't beat using
only your own sensor, even under noise) and `oracle`/`uncorrected`/`FreeAlign`
(Chenyi's own comparison set). Both are now ported and measured in our own
harness (see below) rather than quoted from his paper.

## Result — FreeAlign ported, calibrated, and measured in our own harness

The user confirmed FreeAlign (Lei et al., ICRA 2024) already exists as a
careful, from-paper reimplementation (not the authors' code) in the
labmate's repo, `embedding-aware-BELT-fusion`, `origin/alignformer` branch:
`src/embedding_aware_belt_fusion/alignformer/freealign.py` +
`freealign_calibration.py` + `scripts/calibrate_freealign.py` (fetched
read-only via `git show`, never checked out, never modified).

Ported into a new, deliberately separate top-level package
`src/embedding_aware_belt_fusion/freealign/` (not nested inside
`alignformer_pva/` — this is the competitor baseline, not our architecture;
kept clearly distinct per the user's request for easy navigation):
`freealign.py` (the algorithm: salient-object graph + edge-distance matching,
GNN-free per the labmate's own ablation and shipped config; MASS subgraph
search; its own independent closed-form SE(2) solver, deliberately not
sharing `alignformer_pva.procrustes.weighted_se2_kabsch`, so nothing of our
own architecture's policy leaks into the competitor), `calibrate_freealign.py`
(recalibrated on OUR OWN validation-role data, not the labmate's fitted
constants — same discipline as `alignformer_pva/calibrate_variance.py`).
Integrated additively into `evaluation/belt_fusion.py` as a new `--freealign`
/ `--freealign-calibration` opt-in row; existing methods confirmed unaffected
when omitted. New tests: `tests/test_freealign.py`,
`tests/test_freealign_calibration.py` (25 new, following
`test_alignformer_freealign.py`'s structure, including its cross-check that
the closed-form core agrees with `weighted_se2_kabsch` at uniform weights).
236/236 tests passing.

Measured on the same 3-point test sweep as every other number in this
document (AP@0.7, our own harness/detector/split, no paper numbers quoted):

| noise | ego_only | freealign | alignformer_pva |
|---|---|---|---|
| 0.2m | 0.574 | **0.651** | 0.647 |
| 1.0m | 0.567 | **0.547** | 0.543 |
| 2.0m/10° | 0.565 | **0.541** | 0.521 |

**Two findings, both real and both important**:
1. FreeAlign consistently edges out our `alignformer_pva` by a small margin
   (0.004-0.020 AP@0.7) at every point — the opposite of what Chenyi's own,
   fully-finished implementation shows against his own FreeAlign. Expected,
   not alarming: our port is not yet as finished as his (two-stage training
   and the fusion-step audit below are still open).
2. **Neither alignment method clears the `ego_only` bar except at the lowest
   noise point.** At 1.0m and 2.0m, both FreeAlign and `alignformer_pva`
   score BELOW plain no-fusion. Since this affects an independently
   implemented competitor too, not just ours, the likely shared culprit is
   downstream of both alignment methods — most plausibly the fusion/grouping/
   NMS step every method's corrected boxes pass through, not something
   specific to either pose estimator. **This makes the fusion-step audit the
   clear top priority next**, promoted above two-stage training and the full
   protocol sweep.

## Result — the detector-range bug: confirmed, fix in progress (detector training launched, not yet complete)

**The single most important methodological gap found in this project so
far.** Every number measured above (`naive_late`, `ego_only`,
`belt_geometry`, `alignformer_pva`, `freealign` — all of it) was measured
with our detector at `cav_lidar_range: [-70.4, -40, -3, 70.4, 40, 1]`
(`models/pointpillar_late_fusion/config.yaml`). **Chenyi's own reported
OPV2V numbers were measured with a detector trained at DOUBLE that range,**
`[-140.8, -40, -3, 140.8, 40, 1]` — confirmed directly from his own training
config, fetched read-only from the labmate's repo
(`git show origin/alignformer:configs/alignformer_detector_r140.yaml` in
`embedding-aware-BELT-fusion`) and cross-checked against his own real result
artifact (`git show origin/alignformer:outputs/alignformer/r140/freealign_test_result.json`,
whose `cav_lidar_range` field is literally `[-140.8, -40, -3, 140.8, 40, 1]`
and whose `ap.oracle.ap_70` is 0.8964 — exactly his paper's reported
uncorrected σ=0 number). His own config's comment states the mechanism
directly: OpenCOOD's evaluated `GT_RANGE` is ±140 m for every published row
regardless of detector range, and "416 of 4152 sampled ground-truth boxes
(10.0%) lie beyond |x| = 70.4 m and are invisible to the narrow detector,
capping its recall near 90% in every published row." **This is almost
certainly the dominant reason our numbers have been far below Chenyi's
throughout this whole project** — a structural recall ceiling from training
the detector at the wrong range, not a bug in our alignment code. (Also
discovered along the way: our own "working" ±70.4 m checkpoint,
`models/pointpillar_late_fusion/net_epoch30.pth`, has an mtime of
2021-09-08 — it is OpenCOOD's own stock published checkpoint, not something
this project ever trained. There is no existing from-scratch training run on
this project's own data to compare against; the ±140.8 m run below is this
project's first.)

**Step 1 done — new detector config built and training launched.**
`configs/pointpillar_late_fusion_r140.yaml`: a line-for-line copy of
OpenCOOD's stock `external/OpenCOOD/opencood/hypes_yaml/point_pillar_late_fusion.yaml`
with only `cav_lidar_range` widened to ±140.8 m and `root_dir`/`validate_dir`
pointed at our own OPV2V split (`/mnt/external/workspace/public/dataset/opv2v/train`
/ `.../test`, matching `models/pointpillar_late_fusion/config.yaml`'s
existing convention exactly) — same discipline as Chenyi's own r140 config,
which changes nothing else either. Architecture, batch size (2), epoch count
(15), optimizer, and LR schedule are all identical to the narrow config and
to his r140 config.

**Shim assumption checked, not needed.** Chenyi's own training wrapper
(`origin/alignformer:scripts/train_late_fusion.py`, read for methodology
only) exists solely to work around ASCII `.pcd` decoding on a slow **USB**
disk on his machine (his own docstring: "about a day" unshimmed vs. 2h19m
shimmed). Our OPV2V tree is also ASCII `.pcd` on a rotational disk
(`/dev/sda`, `ROTA=1`, confirmed via `lsblk`), but it is an internal 18TB
drive, not USB over a slow bus. Timed directly before committing to a long
run: a 150-second smoke run of OpenCOOD's stock
`external/OpenCOOD/opencood/tools/train.py` (no shim) against the new r140
config reached a stable ~2.5-3.0 it/s at batch_size=2 with the stock
`num_workers=8` (confirmed I/O-bound, not GPU-bound: GPU utilization reads
0% at random samples, workers sit in Linux `D`-state disk-wait, 4.3 GB VRAM
used of 32 GB available). At that rate, one 3187-iteration training epoch is
~20 minutes and one 1085-iteration validation epoch (official OPV2V test
split, `eval_freq=1`) is ~7 minutes — **~27 min/epoch, ~7 hours for all 15
epochs** — long, but tractable directly with OpenCOOD's stock entrypoint, no
shim needed or built.

**Training launched** (started 2026-10-22, ~15:27 local):

```
nohup /mnt/external/backup/dhia/conda_envs/opencood-blackwell/bin/python -u \
  external/OpenCOOD/opencood/tools/train.py \
  --hypes_yaml configs/pointpillar_late_fusion_r140.yaml \
  > outputs/pointpillar_late_fusion_r140_train/train.log 2>&1 &
disown
```

- PID: `1094917` (check alive: `ps -p 1094917 -o pid,etime,%cpu`)
- Live log: `outputs/pointpillar_late_fusion_r140_train/train.log` (tqdm
  progress lines; `tail -f` to watch, or
  `grep -oE "\[epoch [0-9]+\]\[[0-9]+/3187\]" outputs/pointpillar_late_fusion_r140_train/train.log | tail -1`
  for the latest position)
- Checkpoints + per-epoch val loss land in (OpenCOOD's own `logs/` dir,
  gitignored inside the `external/OpenCOOD` submodule, not committed
  anywhere):
  `external/OpenCOOD/opencood/logs/point_pillar_late_fusion_r140_2026_10_22_15_27_38/`
  (`net_epoch1.pth` ... `net_epoch15.pth`, `save_freq=1`, plus a dumped
  `config.yaml` and tensorboard event files)
- Confirmed stable through iteration 220/3187 of epoch 0 (loss curve moving,
  no errors/NaNs) before this hand-back; **not yet complete** — ETA roughly
  2026-10-22 ~22:30-23:30 local (~7h from launch) for all 15 epochs.
- The existing ±70.4 m checkpoint/caches/results above are untouched and
  unaffected — this is new, additional work, not a replacement.

**Remaining steps, in order, once training finishes (none started yet)**:

1. **Validate the new detector before trusting it** (per this project's own
   hard-learned discipline — do not build on an unverified detector):
   check `external/OpenCOOD/opencood/logs/point_pillar_late_fusion_r140_.../config.yaml`'s
   dumped per-epoch val loss log (or tensorboard event file) for a sensible
   monotone-ish decrease (compare shape against Chenyi's own reference curve
   quoted in his r140 config's comment: his loss went 0.85 → 0.37 over 15
   epochs, minimum at the LAST epoch). Then run detector inference once over
   a small held-out slice (e.g. `extract_proposals.py --cache-mode
   geometry_only --max-frames 50`) and sanity-check recall/box quality is
   not garbage before proceeding. **Do not skip this gate.**
2. **Pick the checkpoint** (last epoch unless the loss curve says otherwise,
   per Chenyi's own finding that his wide-range run's minimum was the last
   epoch, not an earlier one like the narrow run). Copy it + a cleaned hypes
   yaml into `models/pointpillar_late_fusion_r140/` (config.yaml +
   `net_epoch15.pth`), mirroring `models/pointpillar_late_fusion/`'s existing
   layout so every script that takes `--opencood-config`/`--detector-dir`
   works unchanged, just pointed at the new directory.
3. **Rebuild caches at r140** (new detector, new paths, `_r140` suffix,
   nothing old touched):
   - `python -m embedding_aware_belt_fusion.data.extract_proposals --cache-mode geometry_only --opencood-config models/pointpillar_late_fusion_r140/config.yaml --detector-dir models/pointpillar_late_fusion_r140 --data-root /mnt/external/workspace/public/dataset/opv2v --split train --output-dir outputs/alignformer_pva_geometry_train_cache_r140` —
     equivalent of the existing `outputs/alignformer_pva_geometry_train_cache`.
   - Three test-split frame caches via
     `python -m embedding_aware_belt_fusion.evaluation.belt_fusion --frame-cache-mode read-write --opencood-config models/pointpillar_late_fusion_r140/config.yaml --detector-dir models/pointpillar_late_fusion_r140 ...`
     at the same 3 noise points as every number above (0.2m/0°, 1.0m/0°,
     2.0m+10°, `--noise-seed 20`), writing to
     `outputs/frame_cache/opv2v_test_p0p2_h0_seed20_r140` /
     `_p1p0_h0_seed20_r140` / `_p2_h10_seed20_r140`.
   - The scenario split file does NOT need rebuilding — it is purely
     scenario-name-based (`outputs/spatial_trackformer_association_head_128d/scenario_split.json`),
     independent of detector range, and is already reused as the default in
     `calibrate_variance.py`/`calibrate_freealign.py`.
4. **Recalibrate against the new cache** (same scripts, new cache/outputs,
   nothing rebuilt):
   - `python -m embedding_aware_belt_fusion.alignformer_pva.calibrate_variance --cache-dir outputs/alignformer_pva_geometry_train_cache_r140 --output-json outputs/alignformer_pva_variance_calibration_r140.json`
   - `python -m embedding_aware_belt_fusion.alignformer_pva.train --cache-dir outputs/alignformer_pva_geometry_train_cache_r140 --output-dir outputs/alignformer_pva_train_variance_r140 --variance-config outputs/alignformer_pva_variance_calibration_r140.json` —
     same 30-epoch recipe already used for `alignformer_pva_train_variance_v1`.
   - `python -m embedding_aware_belt_fusion.freealign.calibrate_freealign --cache-dir outputs/alignformer_pva_geometry_train_cache_r140 --output-json outputs/freealign_calibration_r140.json`.
   - Apply the already-found-to-help gate/abstain tuning
     (`--alignformer-pva-refine-gates-m 6,3,1.5,0.75,0.5
     --alignformer-pva-agreement-tolerance-m 1.0
     --alignformer-pva-consensus-floor 4`) when evaluating; worth a quick
     independent check at the new range if time permits (wider range → more
     far-field, lower-confidence boxes → these gate/consensus values may not
     transfer unchanged), but reasonable to carry forward as a starting
     point rather than re-deriving from scratch.
5. **Measure and report**: the 3-point sweep (0.2m/0°, 1.0m/0°, 2.0m+10°) at
   r140 for `ego_only`, `freealign`, and `alignformer_pva` only (per this
   doc's reporting-convention change above — `naive_late`/`belt_geometry` are
   not reported going forward), placed directly next to Chenyi's own
   reported OPV2V curve (`CVPR/sec/4_experiments.tex`, Table `opv2v`, AP@0.7
   at σ=0→2.0m): uncorrected `.8964→.1898`, FreeAlign(AP-selected)
   `.8406→.8172`, his AlignFormer `.8526→.8252` — to see how much of the gap
   the range fix alone closes.

**Honest status at hand-back**: steps above are fully specified (exact
commands, exact new paths) but **not executed** — detector training is the
long pole and was still ~7h from done at hand-back time. Nothing past
"launch training" has been run. No caches rebuilt, no recalibration done, no
new numbers measured yet.

## Update (2026-10-29, cont.) — CARLA/GenCPM dataset located; uncertainty-into-Abstain and LiDAR-surrounding-context given full treatment

### (a) The GenCPM CARLA dataset — found, and it is NOT CooperScene

It is already on disk, downloaded: `/mnt/external/workspace/dhia/AdverCPM_hf/`
(the Hugging-Face "AdverCPM" release GenCPM's own `configs/paths.example.yaml`
points `carla_frames` at). Confirmed directly (tar listings, README):

- 30 CARLA episodes across 5 towns (Town01/02/03/06: 5 each, Town04: 10),
  1500 ticks of 50ms each, detections written every 2 ticks (100ms).
- **15 connected agents (vehicles + RSUs) per episode, "spread over a 60m
  cluster"** — vs. OPV2V's typical 2-4 CAVs per scenario and V2X-Real's
  2-agent pairs. This is the "definitely more CAVs per scenario" density
  the user described, confirmed by direct inspection, not assumption.
- `raw/Town*/<episode>/<vehicle>.tar` → OPV2V-layout raw sensor data (4 RGB
  cameras, LiDAR `.pcd`, pose/calibration/ground-truth `.yaml`) — same
  format our pipeline already parses for OPV2V.
- `frames/pointpillars/Town*/<episode>.tar.gz` → **already-computed**
  per-agent PointPillars detections (`<frame>_preds.yaml` + `_score.npy`,
  OPV2V box convention: `location`/`center`/`extent`/`angle`, `lidar_pose`
  in world frame) — exactly the cache format our `extract_proposals.py` /
  `alignformer_pva` dataset already consumes, so no new detector inference
  is needed, only a path/layout adapter to read this tree instead of
  OpenCOOD's own OPV2V directory structure.
- `frames/second/` is the same frames with a second detector (SECOND),
  useful later for a detector-robustness check but not needed for this.

**Integration plan, small and reversible, per the standing requirement to
validate on only a few scenarios first:**
1. New, additive loader (new file, e.g.
   `alignformer_pva/carla_source.py` or a new `--source carla` branch in
   `extract_proposals.py`, not a rewrite) that reads
   `AdverCPM_hf/frames/pointpillars/<Town>/<episode>/` into the same
   `FrameRecord`/cache schema `reproduce_opv2v_sweep.py` already consumes —
   the per-frame dict keys (agent id -> boxes/scores/pose) are structurally
   the same shape as what we already build for OPV2V, just sourced from a
   different directory layout.
2. Smoke-test on **2-3 episodes** first (e.g. Town03 s301/s302, one from
   Town04 for a second map), confirming: per-tick agent count and spacing
   (need to check how many of the 15 are actually within our detector's
   140m/40m range of a given ego at a given tick — 15 agents over "a 60m
   cluster" strongly suggests most frames will have far more simultaneous
   CAVs in range than OPV2V's 970/869/111/101 distribution, but this needs
   confirming per-frame, not assumed from the episode-level count alone).
3. Only once that confirms real 3+/4+/5+-CAV density: run the EXISTING,
   unmodified Chenyi-converted checkpoint and the new `alignformer_context`
   checkpoint through `reproduce_opv2v_sweep.py`'s existing per-bucket
   reporting on this new source, no retraining required for this first
   check — this alone will show whether the dense-CAV weakness gets worse
   (as predicted) and by how much, and gives a much richer test bed for
   validating any of the fixes below than OPV2V's ~10% ≥3-CAV slice.
4. Training on CARLA data (if warranted after step 3) is a separate,
   later decision — not needed to use it as a harder evaluation set first.

### (b) Uncertainty into Verify AND Abstain — the full treatment, not just gate width

What was actually tried (`select_adaptive_gates` in `reproduce_opv2v_sweep.py`):
a single GLOBAL confidence probe per correction (via `abstention_threshold`
at level=0.5, never abstaining) choosing between exactly two fixed gate
schedules for Verify. This never touched Abstain's own decision threshold,
and collapsed Propose's entire per-pair confidence signal into one binary
choice. That is why it under-delivered — it wasn't a real test of the idea.

What Propose actually computes but currently throws away: the Sinkhorn
transport plan gives, per matched (ego-object, CAV-object) pair, a full soft
assignment distribution, not just a top-1 index. A pair with 0.95 mass on
one candidate is categorically more trustworthy than one split 0.4/0.35/0.25
across three — this per-pair concentration (e.g. row entropy, or top1/top2
mass ratio) is exactly a measurement of the diagnosed "blur" mechanism
itself, at the place it originates, not an indirect proxy for it. It is
currently computed inside Sinkhorn and discarded before Verify/Abstain ever
see it.

Two distinct, genuinely new mechanisms (both additive, both reuse
Propose/Verify/Abstain's existing math unchanged):

1. **Per-correspondence (not global) Verify gating.** Thread the per-pair
   Sinkhorn concentration score through to Verify's RANSAC re-solve and use
   it to set each correspondence's OWN inlier gate, not one global schedule
   choice for the whole correction: sharp/confident pairs get a tight gate
   immediately (Propose already found the right pair — a wide search gate
   only risks Verify locking onto a nearby-but-wrong candidate within that
   wide radius, which is the actual blur-to-error mechanism diagnosed
   earlier), ambiguous/blurred pairs keep today's wide schedule so recall on
   genuinely hard pairs doesn't regress. This directly targets the
   mechanism (ambiguous correspondences getting a falsely generous search
   radius) rather than picking between two global schedules.

2. **Confidence-weighted Abstain threshold.** Right now Abstain's Wald test
   only looks at the POST-Verify fit's own covariance against a fixed-alpha
   critical value — it has no way to tell a fit that came from unambiguous
   correspondences apart from one that came from barely-resolved,
   near-tied ones even if both happen to produce a similarly tight final
   statistic (a fit can get "lucky" on a blurred match). Fix: scale the
   critical threshold (equivalently, the effective alpha) by a monotone
   function of the mean/min Sinkhorn concentration among the inlier
   correspondences that fed the final fit — ambiguous-but-passing fits get
   a STRICTER threshold (abstain more readily), unambiguous fits keep
   today's threshold. This is the part of the user's idea ("Verify AND
   Abstain making an informed decision based on Propose") that was not
   touched at all by the gate-width experiment.

Both are additive changes confined to `refine.py`/`abstain.py` call sites
(new optional per-correspondence-weight argument, default-off so existing
behavior is reproduced exactly when unset) — Sinkhorn/Kabsch/Wald math
itself is untouched, honoring "core method stays unchanged."

Honest risk: strictly more hyperparameters than the already-tried global
schedule swap (a concentration->gate mapping, a concentration->alpha
mapping), both needing validation-set tuning rather than a guessed default,
so this costs more iteration cycles than the first attempt — but it is the
first version that actually uses Propose's per-pair confidence rather than
one scalar global probe, which is what was asked for.

### (c) LiDAR surrounding-object feature extraction — distinct from DINOv2 and the 128-d ROI embedding

Both of Chenyi's already-tried, already-negative directions encode the
OBJECT ITSELF: DINOv2 encodes the object's own RGB crop (ruled out —
appearance tokens get displaced by the very pose error being corrected);
the 128-d LiDAR embedding pools points inside the object's own 3D box (small,
inconsistent benefit, dropped). Both are low-SNR by construction: a vehicle's
own LiDAR return is sparse at range and is itself corrupted by the exact pose
error the system exists to correct.

The user's (and their professor's) framing — "surrounding objects," wider
neighborhood, trees/signs — is a different thing: using STATIC SCENE
STRUCTURE around a candidate match, not the moving object's own footprint.
Why this is mechanistically different, not just "try appearance again":

- Pose drift is a single rigid transform applied to the CAV's WHOLE point
  cloud. Static structure near a candidate object (curbs, poles, building
  corners, parked cars) is DENSE, has strong, repeatable local geometry
  (corners, verticality, planarity), and both ego's and the CAV's LiDAR see
  it independently of whether either one's detector fires a box on it. A
  local neighborhood descriptor (10-20m radius around the candidate pair)
  is a much richer, less ambiguous fingerprint than one object's collapsed
  (x, y, heading, size) 7-DOF box, which is exactly what Sinkhorn's blur
  degrades in dense scenes.
- It targets the diagnosed failure mode directly: the blur happens when
  several candidates are plausible under the box-only descriptor. A static
  context descriptor is a genuine extra discriminating signal exactly
  where ambiguity exists, which is the opposite situation from Chenyi's
  DINOv2 test (sparse 2-agent pairs, already at 99.4% geometric top-1 —
  "almost nothing for appearance to fix" there). Dense (3+ CAV) scenes are
  explicitly the UNTESTED regime his own limitations section flags.
- Should be used as a tie-breaker specifically for near-tied Sinkhorn mass
  (the same per-pair concentration signal from (b)), not as a global
  always-on extra feature — minimizes risk of repeating the "small,
  inconsistent, not worth the bandwidth" outcome by only paying the cost
  where there is ambiguity to resolve.

Concrete descriptor choice, deliberately NOT a learned/trained encoder like
DINOv2 (avoids needing new training data or risking the same "pose-error
entangled with content" failure): rotation/pose-error-invariant handcrafted
local statistics computed from the raw point cloud around the candidate
location — height/density histograms in concentric rings, or FPFH-style
local surface normals/curvature histograms. These describe STATIC structure
that doesn't move with the pose error being corrected, unlike an RGB crop or
an in-box LiDAR embedding.

Feasibility, stated honestly:
- Needs raw point clouds per frame, which the current
  `alignformer_pva_geometry_train_cache` does NOT retain (boxes/scores
  only) — this is new data engineering (a caching pass over
  `raw/.../<frame>.pcd`), not a reuse of existing cache, and is the main
  cost of trying this at all.
- Precomputable and cacheable exactly like detector output, since static
  context around a location doesn't depend on which noise draw perturbed
  the CAV's reported pose — compatible with the standing "cache inference,
  no repeated detector runs" requirement, just a new cache artifact.
- FPFH-style descriptors are not differentiable/trainable end-to-end in
  this form (fine — they are meant as a fixed tie-breaking feature, not a
  learned embedding); the height/density-histogram variant is trivially
  differentiable if a learned combination weight is ever wanted later.
- Sequence this AFTER (b): (b) only touches a signal that already exists
  (Sinkhorn's own output) and needs no new data; (c) needs new data
  engineering and is more speculative. Before any training integration,
  validate cheaply on ~20-50 known-blurred dense frames: does a surrounding-
  context descriptor actually separate the true candidate from the
  nearby-plausible one in cases Propose currently gets wrong? That answer
  should come before deciding whether to build the full pipeline.

### Priority order recommended (not yet started — awaiting go-ahead)

1. **(b) Verify/Abstain confidence propagation** — no new data needed, pure
   additive code on existing cache, directly targets the diagnosed
   mechanism, cheapest to test and revert.
2. **(a) CARLA/AdverCPM smoke test (steps 1-3 above, no training)** — cheap
   (loader + a few-episode sweep), and gives a much richer dense-CAV test
   bed to validate (b) and (c) against than OPV2V's ~10% slice.
3. **(c) LiDAR surrounding-context descriptor, validation-only first** — the
   20-50-frame separability check before any caching/training investment,
   gated on (b) and (a) both being in hand so there is a proper dense-CAV
   test bed and an existing confidence signal to combine it with.

All three remain additive/opt-in per the standing reversibility requirement;
nothing in `alignformer_pva/{trunk,model,dataset,train}.py`,
`evaluation/belt_fusion.py`, or Chenyi's checkpoint path changes.

## Update (2026-10-29, later) — Multi-dataset sweep built (V2X-Real, CooperScene); CooperScene σ=0 finding; ideas list

### What was built (all new files; nothing existing modified)

- `scripts/reproduce_multidataset_sweep.py` — imports `reproduce_opv2v_sweep.py`
  as a module and calls its own `run_condition` per (σ, seed, CAV-count
  bucket): same noise model, same `FULL_*` Propose/Verify/Abstain config, same
  FreeAlign, same fusion_core AP. Adds `--dataset {opv2v,v2xreal,cooperscene}`,
  per-bucket AP (exact count + exact pooled `all`/`3+` merges of TP/FP/score
  lists, not averages), an `ego_only` row, keyed (scenario, frame) geometry
  attach. No delay conditions (bucket subsets break frame adjacency).
- `src/embedding_aware_belt_fusion/multidataset/v2xreal.py` — Chenyi's V2X-Real
  adapter, ported read-only (`origin/alignformer` f6718fc), camera hook dropped.
- `src/embedding_aware_belt_fusion/multidataset/sources.py` — per-dataset
  dataset construction, CooperScene ego/range/staging, stub-LiDAR frame tables.
- `scripts/build_v2xreal_geometry_cache.py` — runs unmodified
  `extract_proposals --cache-mode geometry_only` through the adapter.
- `scripts/build_cooperscene_geometry_cache.py` — converts the cached GenCPM
  detections (`~/cooperscene_preds/pp/TownCooperScene`, CooperScene-trained
  PointPillars net_epoch15) to the same cache schema. No inference needed.
  Built: `outputs/cooperscene_geometry_{test,validate}_cache` (1312 / 802 frames).
- `outputs/alignformer_chenyi_boxesonly_v2xreal.pth` — Chenyi's V2X-Real
  `stage2_B_boxes_only/best.pth` via the existing converter (strict load OK).
- `scripts/run_multidataset_sweep.sh` — the long runs (below).
- `tests/test_multidataset_sources.py` — 4 tests; suite 240/240.

### Dataset decisions (each one checked, not assumed)

- **V2X-Real**: Chenyi's own detector (`net_epoch13`, his val-loss pick), his
  boxes-only checkpoint, his FreeAlign calibration (`valap_thr1`). His final
  V2X-Real AlignFormer row uses exactly our `FULL_*` config (RANSAC gates
  6/3/1.5/0.75/0.5, weighted, Wald 0.2, agreement 1.0 m, floor 4 — read from
  `B_boxes_only_icprw_test_result.json`), so no per-dataset preset exists.
  **Acceptance gate** (his 5-seed test, AP@0.7, σ=1 m): oracle .4224,
  uncorrected .158, FreeAlign .332, AlignFormer .3445; σ=0: AlignFormer .3921.
- **CooperScene**: agent `0` is the RSU (static pose, 2.7 m up); OpenCOOD would
  make it ego by sort order, so ego = vehicle `1` (vehicle-centric, as
  V2X-Real; `--ego-agent` overrides). Object ids are global across agents
  (same id → same world location), so OpenCOOD's union-by-id GT is valid.
  The cached detections were post-processed under stock `GT_RANGE` and contain
  nothing beyond |y| = 40 m (max 39.98 m over 3815 sampled boxes), so GT is
  cut at x ±102.4 / y ±40 — otherwise every method pays for objects no
  detector output could contain. No in-domain AlignFormer checkpoint or
  FreeAlign calibration exists: both methods are **zero-shot transfers**
  (V2X-Real ckpt+calibration by default; OPV2V ckpt+calibration as a second
  variant).
- **Ground truth without LiDAR**: GT depends only on yaml params, so frame
  tables use a stub point reader. Verified identical (ego pose, CAV poses,
  GT boxes) to the real-LiDAR `outputs/opv2v_frame_table.pt` at 6 frames.
- **Bucket merge verified**: OPV2V σ=0 through the new driver reproduces
  `sweep_full_20seeds_v2.json` to 6 decimals for every method/IoU. OPV2V
  buckets: 0:119, 1:970, 2:869, 3:111, 4:101 (the 119 zero-CAV frames were
  never broken out before).

### CooperScene: full test split, σ=0 (one deterministic condition — a real result, not a sample)

1312 frames; CAV-count buckets 2:169, **3:1143** (87% of frames have 3 CAVs in
range, vs ~10% on OPV2V — the dense regime we wanted).

| AP@0.7 / 0.5 / 0.3 (all frames) | ego_only | oracle | FreeAlign | AlignFormer |
|---|---|---|---|---|
| σ=0 | .175/.356/.497 | **.516/.747/.832** | .109/.341/.537 | .199/.493/.754 |

At σ=0 any correction can only hurt; both transferred methods destroy most
of fusion's gain (FreeAlign falls below ego-only). Diagnosis (1/5 of frames
for the per-pair shifts, all frames for the residuals):
- **Agent 3 shares a median of 1 detected object with ego** → unalignable
  pair. FreeAlign (no abstention) applies a median **60 m** shift, >2 m on
  80% of pairs. AlignFormer abstains on 59% but still shifts >2 m on 13%.
- **The dataset poses are fine**: detection→GT residual under the yaml pose is
  0.33–0.55 m for every agent (ego's own: 0.41 m); a mild speed dependence
  for the moving agent 2 (0.51 m moving vs 0.39 m still, corr 0.45).
- **Open**: both methods apply a consistent ~0.67 m shift to agent 2 (≥0.5 m
  on ~80% of pairs) that the residuals say is *wrong*. Two independent
  methods agreeing on a wrong correction suggests a shared input bias
  (candidates: a ~1° yaw bias from repetitive parked rows; a per-agent
  detector box-centre bias). Not yet explained.

Conclusion so far: on CooperScene the binding constraint is **abstention on
low-overlap pairs**, not estimation precision. Caveat: zero-shot transfer
for both methods; no claim about AlignFormer-vs-FreeAlign on CooperScene
until the σ sweep and the in-domain-calibration variant exist.

### Runs to launch (user; tqdm progress; each step skips if output exists)

```
bash scripts/run_multidataset_sweep.sh v2xreal-cache        # GPU, once
bash scripts/run_multidataset_sweep.sh v2xreal 20           # then compare to the acceptance gate above
bash scripts/run_multidataset_sweep.sh cooperscene 20       # ~58 s/condition single-core; ~10 min on 20 workers
bash scripts/run_multidataset_sweep.sh cooperscene-opv2v 20 # second transfer variant
bash scripts/run_multidataset_sweep.sh opv2v 20             # OPV2V bucket table on the identical driver
```
Outputs: `outputs/multidataset_sweep/*.json` (per bucket: per-seed AP, mean,
std; `meta` records ckpt/calibration/ranges/ego). End-of-run stdout prints
AP@0.7 tables per bucket with AF−FA.

### Improvement ideas — prioritized running list

Each: target → where the measured gap is → what was tried → next step, risk.

1. **Abstention ceiling diagnostic (do first; CPU, cached, cheap).**
   Target: split each dataset's AF-vs-oracle gap into "would be fixed by
   perfect abstention" vs "estimation error on pairs that should be
   corrected". Gap evidence: CooperScene σ=0 .516→.199; OPV2V σ=0 .882→.848;
   V2X-Real σ=0 .422→.392 — all are pure abstention failures (truth is
   identity). Never measured: an oracle-abstain row (keep the correction
   only when its error vs true pose < τ, else identity) per bucket and σ.
   Risk: none; it is a measurement. Payoff: tells us how much items 2–3 can
   possibly buy before building them.
2. **Per-pair Sinkhorn confidence into Verify AND Abstain** (scoped in (b)
   above). New evidence makes Abstain the half that matters most: the
   CooperScene failure is a pair with ~1 shared object, which is exactly
   what Sinkhorn concentration / non-dustbin mass measures. Concrete
   abstain features: number of sharp (top1/top2 ratio > r) correspondences,
   total matched mass, inlier count after Verify; add to the Wald gate as a
   minimum-evidence condition first (one hyperparameter), then the
   concentration-scaled threshold. Tune on V2X-Real val + CooperScene
   validate, report on test. Risk: more hyperparameters; transfer of
   thresholds across datasets is itself a question (report both per-dataset
   and single shared setting).
3. **Cycle-consistency abstention with 3+ CAVs.** With ego, A, B all
   corrected pairwise, the loop ego→A→B→ego must close; a wrong pair breaks
   it. Distinct from what was tried: the post-hoc consensus *averaged
   boxes*; Chenyi's pose graph (fill/joint) was flat on V2X-Real val
   (+.0013/−.0009 AP@0.7 at σ=1) but V2X-Real rarely has 3 agents in range,
   so cycles barely existed. CooperScene (87% 3-CAV) is the first test bed
   where cycles exist on most frames. Needs CAV↔CAV pair estimates (one
   extra Propose call per CAV pair; boxes already cached). Risk: if two of
   three pairs are wrong the loop can still close by coincidence — measure
   loop-residual separability vs pair error first (cheap) before using it.
4. **Explain the agent-2 shared 0.67 m wrong shift** (CooperScene). Decompose
   the estimated correction into yaw vs translation and check range
   dependence (yaw bias → shift ∝ range). If it is a per-agent detector
   centre bias, it is a dataset property worth stating; if it is a yaw bias
   from periodic parked rows, it is the "blur" mechanism in its purest form.
   Cheap.
5. **In-domain calibration on CooperScene** (fairness of the transfer
   comparison): FreeAlign `calibrate_freealign` and AlignFormer
   `calibrate_variance` on `cooperscene_geometry_validate_cache` (needs
   gt-matched fields the converter does not yet write — add from the yaml
   labels). Fine-tuning AlignFormer on CooperScene train would need detector
   inference on train (GenCPM only ran validate/test). Deferred behind 1–3.
6. **Shared-object buckets alongside CAV-count buckets.** On CooperScene the
   CAV count barely varies; shared-object count (Chenyi's axis: 0 / 1–2 /
   3+) is what discriminates. Needs pair-level evaluation (ego + one CAV)
   since frames mix buckets. Moderate.
7. **LiDAR static-context descriptor** (scoped in (c)). Unchanged, but a new
   risk: CooperScene's failure is periodic parked rows, and static structure
   in a parking lot is periodic too — the descriptor's separability must be
   validated specifically on those frames, not just on OPV2V.
8. **AdverCPM CARLA (15 agents)**: the GenCPM layout is the same family as
   the CooperScene cache, so its converter is a small variant of
   `build_cooperscene_geometry_cache.py` (boxes come from `_preds.yaml`
   world-frame objects if `_pred.npy` is absent). Gives a controlled
   (simulated, exact-pose) dense test bed to separate density from
   real-data effects seen on CooperScene.
