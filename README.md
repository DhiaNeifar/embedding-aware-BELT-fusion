# Embedding-aware BELT-Fusion

Plan-0 research scaffold for learned cross-agent proposal embeddings that augment
BELT-Fusion's Hungarian association cost while leaving uncertainty-aware fusion
unchanged.

The upstream projects are pinned as clean Git submodules:

- `external/BELT-Fusion`
- `external/trackformer`

Clone with:

```bash
git clone --recurse-submodules <repository-url>
```

The OPV2V dataset is external and must not be copied into this repository. Create a
local config from the committed template:

```bash
cp configs/paths.example.yaml configs/local.paths.yaml
```

Then set `opv2v_root` to `/mnt/external/workspace/public/dataset/opv2v` (or the
appropriate path on another machine).

The current upstream BELT-Fusion snapshot is a prototype and does not contain a
wired PointPillars backbone, OPV2V preprocessing, coordinate-frame transformation,
or pose-noise implementation. Read [the codebase analysis](docs/codebase_analysis.md)
and [the Plan-0 design](docs/plan0_design.md) before implementing extraction.

The parent repository now supplies the missing frozen-OpenCOOD uncertainty training
adapter. See [uncertainty training](docs/uncertainty_training.md) for the smoke and
full training commands.

The Plan-0 proposal cache and supervised-contrastive embedding pipeline are also
implemented. See [embedding training](docs/embedding_training.md).

No dataset, checkpoints, proposal records, or experiment outputs are tracked.
