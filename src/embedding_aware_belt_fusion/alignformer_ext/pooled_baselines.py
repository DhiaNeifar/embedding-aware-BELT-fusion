"""Information-matched temporal baselines (assessment 2026_11_11, EXP-003): another plug-in's per-frame corrections
pooled over time with exactly AlignFormer's pooling and window Abstain (alignformer_tk's settings).

This is an ablation of the pooling stage, not a new version of the baselines: the published baselines stay single-frame
in the main tables. It asks whether pooling helps any per-frame corrector as much as it helps AlignFormer's.

* ``freealign_tk``: FreeAlign's per-pair corrections, refitted from its cached corrected boxes
  (poses.freealign_corrections, the same refit export_poses.py writes for the hosts);
* ``coalign_tk``, ``roco_tk``: the pose-graph corrections exported to ``outputs/poses/<dataset>/<plugin>/`` and matched
  to the cached pairs by (frame index, CAV id).

The dataset of the exported files comes from ``ALIGNFORMER_POSES_DATASET`` (opv2v or v2xreal); the run stops instead of
guessing. Leak check: the per-frame inputs are each plug-in's own corrections under the reported pose. ``sigma`` and
``seed`` only name the file of the same noise draw, as for every plug-in; nothing reads truth, the noise level or delay.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from embedding_aware_belt_fusion.alignformer_ext import poses, variants

POSES_ROOT = Path(os.environ.get("ALIGNFORMER_POSES_ROOT", "outputs/poses"))
DATASETS = ("opv2v", "v2xreal")
KEPT_TOLERANCE = 1e-6


def _dataset() -> str:
    dataset = os.environ.get("ALIGNFORMER_POSES_DATASET")
    if dataset not in DATASETS:
        raise RuntimeError("coalign_tk / roco_tk need ALIGNFORMER_POSES_DATASET=opv2v or v2xreal")
    return dataset


def freealign_per_frame(inputs: variants.PairInputs) -> np.ndarray:
    """(P, 3) FreeAlign's correction per cached pair, zero where it kept the reported pose."""
    return poses.freealign_corrections(inputs.cache)


def _exported(plugin: str):
    def per_frame(inputs: variants.PairInputs) -> np.ndarray:
        cache, records = inputs.cache, inputs.records
        path = poses.condition_path(POSES_ROOT, _dataset(), plugin, float(cache["sigma"]), int(cache["seed"]))
        with np.load(path) as data:
            index = {(int(f), str(c)): np.asarray(x, np.float32)
                     for f, c, x in zip(data["frame_index"], data["cav_id"], data["correction"])}
        out = np.zeros((len(cache["frame"]), 3), np.float32)
        missing = 0
        for p in range(len(cache["frame"])):
            f = int(cache["frame"][p])
            cav = sorted(records[f].cav_boxes)[int(cache["cav_rank"][p])]
            correction = index.get((f, str(cav)))
            if correction is None:
                missing += 1
                continue
            out[p] = correction
        if missing:
            raise RuntimeError(f"{path}: {missing} of {len(out)} cached pairs have no exported correction")
        # A kept reported pose is exported as round-off (|x| < 1e-9; real corrections are > 1e-3). Pooling counts any
        # nonzero row as a measurement, so snap round-off to an exact abstention, as alignformer_c's zeros are.
        out[np.abs(out).max(axis=1) < KEPT_TOLERANCE] = 0.0
        return out

    per_frame.__name__ = f"{plugin}_per_frame"
    return per_frame


def _pooled(per_frame):
    def run(inputs: variants.PairInputs) -> np.ndarray:
        from embedding_aware_belt_fusion.alignformer_ext import temporal

        settings = temporal.TemporalSettings(odometry_sd=(0.0, 0.0), odometry_rel=variants._LOAM, consistency="adaptive")
        return temporal.temporal_decisions(inputs, settings, measurements=per_frame(inputs))

    run.__doc__ = f"{per_frame.__name__} pooled over time with alignformer_tk's settings."
    return run


BASELINES = {
    # one frame, for a paired comparison on the same draws (FreeAlign's one-frame row is the sweep's "freealign")
    "coalign_frame": _exported("coalign"),
    "roco_frame": _exported("roco"),
    # pooled with AlignFormer's pooling and window Abstain
    "freealign_tk": _pooled(freealign_per_frame),
    "coalign_tk": _pooled(_exported("coalign")),
    "roco_tk": _pooled(_exported("roco")),
}
