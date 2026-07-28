# PointPillars–TrackFormer embedding training

This pipeline uses TrackFormer's actual `Transformer` implementation with a
PointPillars spatial BEV memory. It is distinct from the earlier per-proposal MLP
baseline.

For each agent, the frozen PointPillars detector supplies:

- a 384-channel BEV map, adaptively pooled from `100x352` to `13x44`;
- decoded proposals and their local PointPillars features;
- proposal boxes, scores, agent identity, timestamp, and ground-truth object ID.

The adapter projects the BEV map to TrackFormer's 256-dimensional model space.
Proposal features initialize decoder content queries, and 3D boxes initialize
query positional embeddings. Source-agent decoder states are detached and
prepended to the target agent as TrackFormer track queries. Known cross-agent
object identities hard-assign valid track queries, unmatched source proposals
provide no-object/false-positive supervision, and remaining target objects are
Hungarian-matched to ordinary object queries. Training uses object/no-object
classification, seven-parameter box L1, and BEV generalized-IoU losses.
TrackFormer's decoder states are projected and normalized to 128 dimensions.

The spatial pooling is necessary for tractability. A full FP16 map is about
25.8 MiB per agent and vanilla encoder self-attention is quadratic over 35,200
spatial locations. The `13x44` memory is about 0.42 MiB per agent and contains
572 spatial locations.
