"""Per-dataset plumbing for the multi-dataset noise sweep.

``scripts/reproduce_opv2v_sweep.py`` fixes everything that defines the
experiment -- noise model, Propose/Verify/Abstain configs, FreeAlign, late
fusion, AP -- and reads only two dataset-shaped things: an OpenCOOD
late-fusion dataset (world poses + ground truth, via ``build_frame_table``)
and a geometry cache (per-agent local boxes/scores, detector run once). This
module supplies those two things for V2X-Real and CooperScene, so the sweep
itself runs unchanged on all three datasets.

What differs per dataset, and where it is handled:

* **OPV2V** -- stock OpenCOOD, stock ranges. Nothing here.
* **V2X-Real** -- Chenyi's adapter (``multidataset.v2xreal``): ``.bin``
  LiDAR, vehicle-class filter, vehicle-centric ego, V2X-Real GT/COM range.
* **CooperScene** -- OPV2V-format yaml, so stock ``LateFusionDataset``, but:
  - agent ``0`` is the roadside unit (static pose, 2.7 m up) and OpenCOOD
    would make it ego by sorted order; the ego is moved to a vehicle
    (``COOPERSCENE_DEFAULT_EGO``), the same vehicle-centric protocol as
    V2X-Real;
  - the detections are the cached GenCPM ones (``gencpm infer`` with
    ``AdverCPM_hf/models/detectors/cooperscene_pointpillars``), whose
    post-processing ran under OpenCOOD's stock ``GT_RANGE`` and so contain
    nothing beyond |y| = 40 m in any agent's frame (measured: max |y| 39.98 m
    over 3815 sampled boxes). Ground truth is cut at the same |y| so no
    method is charged for objects no detector output could contain; x uses
    the detector's own +/-102.4 m.
  - those cache dirs mix ``<ts>_preds.yaml`` in with ``<ts>.yaml`` and
    OpenCOOD lists every ``*.yaml`` as a frame, so the dataset is built over
    a yaml-only symlink tree (``stage_cooperscene_split``).

Ground truth never depends on LiDAR (``generate_object_center`` /
``generate_gt_bbx`` read only yaml params), so frame tables are built with a
stub point-cloud reader (``stub_lidar``) instead of reading ~44 GB of ASCII
``.pcd``. ``tests/test_multidataset_sources.py`` checks this against the
saved OPV2V frame table built with real LiDAR.
"""

from __future__ import annotations

import contextlib
import os
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from embedding_aware_belt_fusion.multidataset import v2xreal

DATASETS = ("opv2v", "v2xreal", "cooperscene")

COOPERSCENE_RSU_ID = "0"
COOPERSCENE_DEFAULT_EGO = "1"
COOPERSCENE_GT_RANGE = [-102.4, -40.0, -15.0, 102.4, 40.0, 15.0]
COOPERSCENE_COM_RANGE = 70

# A few points well inside every detector range and outside mask_ego_points'
# ego box, so voxelization has something to do and returns quickly.
_STUB_POINTS = np.array(
    [[10.0, 0.0, -1.0, 0.5], [20.0, 5.0, -1.0, 0.5], [-15.0, -5.0, -1.0, 0.5]],
    dtype=np.float32,
)


@contextlib.contextmanager
def stub_lidar() -> Iterator[None]:
    """Make ``pcd_utils.pcd_to_np`` return a fixed 3-point cloud, restored on exit."""
    from opencood.utils import pcd_utils

    original = pcd_utils.pcd_to_np
    pcd_utils.pcd_to_np = lambda _path: _STUB_POINTS.copy()
    try:
        yield
    finally:
        pcd_utils.pcd_to_np = original


def set_ego(scenario_database: "OrderedDict", ego_id: str) -> "OrderedDict":
    """Every scenario reordered with ``ego_id`` first and flagged ego.

    OpenCOOD takes the first agent's timestamps as the frame list and the
    ``ego`` flag as the ego, so both move together. Raises if a scenario has
    no such agent rather than silently keeping OpenCOOD's choice.
    """
    reordered = OrderedDict()
    for scenario_index, agents in scenario_database.items():
        if ego_id not in agents:
            raise KeyError(f"scenario {scenario_index} has no agent {ego_id!r} (agents: {list(agents)})")
        order = [ego_id] + [cav_id for cav_id in agents if cav_id != ego_id]
        scenario = OrderedDict()
        for cav_id in order:
            content = OrderedDict(agents[cav_id])
            content["ego"] = cav_id == ego_id
            scenario[cav_id] = content
        reordered[scenario_index] = scenario
    return reordered


def stage_cooperscene_split(gencpm_root: Path, split: str, stage_root: Path) -> Path:
    """Yaml-only symlink tree ``<stage_root>/<split>/<episode>/<agent>/<ts>.yaml``.

    ``gencpm_root`` is the GenCPM layout dir holding ``<split>_<scene>_s<k>``
    episodes (e.g. ``~/cooperscene_preds/pp/TownCooperScene``). Idempotent.
    """
    split_dir = stage_root / split
    episodes = sorted(p for p in gencpm_root.iterdir() if p.is_dir() and p.name.startswith(f"{split}_"))
    if not episodes:
        raise FileNotFoundError(f"no {split}_* episodes under {gencpm_root}")
    for episode in episodes:
        for agent_dir in sorted(p for p in episode.iterdir() if p.is_dir()):
            out = split_dir / episode.name / agent_dir.name
            out.mkdir(parents=True, exist_ok=True)
            for name in os.listdir(agent_dir):
                if name.endswith(".yaml") and not name.endswith("_preds.yaml"):
                    link = out / name
                    if not link.is_symlink():
                        link.symlink_to((agent_dir / name).resolve())
    return split_dir


def build_sweep_dataset(
    dataset: str, hypes: Dict, split_dir: Path, ego_id: Optional[str] = None
):
    """The OpenCOOD late-fusion dataset ``build_frame_table`` reads, for ``dataset``.

    Also sets OpenCOOD's module-level ``GT_RANGE``/``COM_RANGE`` (read at call
    time by ground-truth generation and ``build_frame_table``'s comm filter).
    """
    import opencood.data_utils.datasets as opencood_datasets

    hypes["validate_dir"] = str(Path(split_dir).resolve())
    if dataset == "opv2v":
        built = opencood_datasets.build_dataset(hypes, visualize=False, train=False)
    elif dataset == "v2xreal":
        built = v2xreal.build_dataset(hypes, visualize=False, train=False)
    elif dataset == "cooperscene":
        opencood_datasets.GT_RANGE = list(COOPERSCENE_GT_RANGE)
        opencood_datasets.COM_RANGE = COOPERSCENE_COM_RANGE
        hypes["fusion"]["core_method"] = "LateFusionDataset"
        built = opencood_datasets.build_dataset(hypes, visualize=False, train=False)
    else:
        raise ValueError(f"unknown dataset {dataset!r}; expected one of {DATASETS}")
    if ego_id is not None:
        built.scenario_database = set_ego(built.scenario_database, ego_id)
        built.len_record = v2xreal._len_record(built.scenario_database)
    elif dataset == "cooperscene":
        raise ValueError("cooperscene needs an explicit ego_id (agent 0 is the RSU)")
    return built


def frame_keys(dataset) -> List[Tuple[str, str, str]]:
    """``(scenario_id, frame_id, ego_id)`` per dataset index, as ``extract_proposals`` names them."""
    keys = []
    previous_end = 0
    for scenario_index, end in enumerate(dataset.len_record):
        scenario = dataset.scenario_database[scenario_index]
        first = next(iter(scenario.values()))
        timestamps = [k for k in first.keys() if k != "ego"]
        ego_id = next(cav_id for cav_id, content in scenario.items() if content["ego"])
        scenario_name = Path(first[timestamps[0]]["yaml"]).parents[1].name
        for offset in range(end - previous_end):
            keys.append((scenario_name, str(timestamps[offset]), str(ego_id)))
        previous_end = end
    return keys


def attach_geometry_by_key(records: Sequence, keys: Sequence[Tuple[str, str, str]], geometry_index: Dict) -> None:
    """Like ``reproduce_opv2v_sweep.attach_geometry``, but looked up by (scenario, frame) and ego id.

    The positional version relies on cache shard order matching dataset
    order; keyed lookup cannot silently misalign and fails loudly on a gap.
    CAVs outside COM_RANGE (absent from ``record.cav_poses``) are skipped, as
    ``extract_proposals`` never caches them in the first place;
    ``run_condition`` assumes every cached CAV has a pose.
    """
    for record, (scenario_id, frame_id, ego_id) in zip(records, keys):
        agents = geometry_index.get((scenario_id, frame_id))
        if agents is None:
            raise KeyError(f"geometry cache has no frame {scenario_id}/{frame_id}")
        if ego_id not in agents:
            raise KeyError(f"geometry cache frame {scenario_id}/{frame_id} has no ego agent {ego_id!r}")
        for agent_id, agent in agents.items():
            if agent_id == ego_id:
                record.ego_boxes = agent["boxes"]
                record.ego_scores = agent["scores"]
            elif agent_id in record.cav_poses:
                record.cav_boxes[agent_id] = agent["boxes"]
                record.cav_scores[agent_id] = agent["scores"]
