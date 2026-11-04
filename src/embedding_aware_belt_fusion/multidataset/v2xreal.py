"""V2X-Real through OpenCOOD's late-fusion dataset: an adapter, not a fork.

Ported read-only (``git show``) from the labmate's repo
``embedding-aware-BELT-fusion``, branch ``origin/alignformer`` (f6718fc),
``src/embedding_aware_belt_fusion/alignformer/v2xreal.py`` -- the adapter
Chenyi's own V2X-Real detector, cache and AlignFormer checkpoint were built
with, so ground truth, ego choice and ranges here match his. Verbatim except
the camera hook (``frame_cameras``), dropped: it needs his camera-feature
module, which this repo does not have and the boxes-only sweep never uses.

V2X-Real was recorded by the lab that wrote OpenCOOD and its per-frame yaml
is OpenCOOD's yaml (``lidar_pose``, ``vehicles`` with ``location`` /
``angle`` / ``extent``), so ``LateFusionDataset`` already indexes the
scenarios, projects the boxes and builds the labels. Three things differ,
and each is handled in the smallest place that can handle it:

1. **LiDAR is ``<ts>.bin``**, float32 (x, y, z, intensity), never ``.pcd``.
   ``basedataset`` builds the ``.pcd`` name unconditionally and reads it
   through ``opencood.utils.pcd_utils.pcd_to_np`` -- the one module slot the
   training script already patches for its NVMe cache. The adapter patches
   the same slot and defers to the previous reader for a ``.pcd`` that
   really exists, so OPV2V keeps working in the same process.
2. **The ``vehicles`` block is every annotated object**: pedestrians,
   riders, buses, trash cans. OpenCOOD's ``project_world_objects`` filters
   by position only, so without a class filter a pedestrian is a
   ground-truth vehicle. The filter is GenComm's ``vehicle`` super-class
   verbatim (``opencood/data_utils/__init__.py:1-7`` in
   ``/media/chenyi/basement2/repos/HetPoison/GenComm``). Trucks and buses
   are that codebase's ``truck`` super-class with its own anchor; a 12.7 m
   bus does not fit a car anchor, so they are out of the ground truth on
   purpose, for every arm and for FreeAlign alike.
3. **Two roadside units per scenario.** ``basedataset`` rotates ONE negative
   id to the end of its sorted list, so with folders ``-1, -2, 1, 2`` its ego
   is ``-2``. The vehicle-centric protocol (GenComm
   ``v2xreal_basedataset.py:207-218``) puts vehicles first; the adapter
   reorders the scenario database after construction and moves the ``ego``
   flag to the lowest-numbered vehicle.

Ranges are V2X-Real's, not OPV2V's: ``GT_RANGE`` and ``COM_RANGE`` are
module constants of ``opencood.data_utils.datasets`` read at call time, so
they are set from the config when the dataset is built, and never edited
in the vendored tree.

Nothing here mutates the dict OpenCOOD hands over; OpenCOOD keeps
references to what it is given.
"""

from __future__ import annotations

import copy
import functools
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

import numpy as np
import yaml

CORE_METHOD = "V2XRealLateFusionDataset"

# GenComm opencood/data_utils/__init__.py:1-7, SUPER_CLASS_MAP["vehicle"].
VEHICLE_TYPES = frozenset({"Car", "PoliceCar", "LongVehicle"})

# GenComm opencood/data_utils/datasets/__init__.py:44 and every V2X-Real
# stage-1 yaml. Wide in z because infrastructure LiDARs sit ~4 m up.
V2XREAL_GT_RANGE = [-102.4, -51.2, -15, 102.4, 51.2, 15]
V2XREAL_COM_RANGE = 70

# (scenario, agent, timestamp) whose LiDAR was corrupt in the archive itself
# and was deleted on 2026-09-28 (memory: v2x-real-dataset-state). The yaml
# and cameras for that frame still exist, so an indexer that lists yamls
# will reach it; the shim refuses it by name rather than by a missing file.
CORRUPT_FRAMES = frozenset({("2023-04-04-15-58-18_30_0", "1", "000112")})

# Per-process. A training item loads the same four frame yamls ~5 times;
# 64 entries cover a DataLoader worker's recent items without holding a
# split's worth of parsed dicts in every worker.
YAML_CACHE_SIZE = 64

POINT_FLOATS = 4
POINT_BYTES = POINT_FLOATS * np.dtype(np.float32).itemsize

PcdReader = Callable[[str], np.ndarray]


# ----------------------------------------------------------------------------
# LiDAR
# ----------------------------------------------------------------------------


def load_lidar_bin(path: Path | str) -> np.ndarray:
    """``(N, 4)`` float32 from a V2X-Real ``.bin``, non-finite rows dropped.

    Real files carry occasional NaN rows, and the frame that had to be
    deleted had 294 of them plus infinities. A NaN reaching the voxelizer is
    a silent garbage pillar, so rows are dropped here, not passed on.
    """
    path = Path(path)
    size = path.stat().st_size
    if size % POINT_BYTES != 0:
        raise ValueError(
            f"{path} is {size} bytes, not a multiple of {POINT_BYTES}: truncated?"
        )
    points = np.fromfile(path, dtype=np.float32).reshape(-1, POINT_FLOATS)
    finite = np.isfinite(points).all(axis=1)
    return points[finite]


def bin_pcd_to_np(real_reader: PcdReader) -> PcdReader:
    """A ``pcd_to_np`` that reads ``<ts>.bin`` when ``<ts>.pcd`` does not exist.

    Returned rather than installed so a test can hold it without touching
    module state; :func:`install_shims` installs it.
    """

    def read(pcd_file: str) -> np.ndarray:
        pcd = Path(pcd_file)
        if pcd.exists():
            return real_reader(pcd_file)
        _refuse_corrupt(pcd)
        bin_path = pcd.with_suffix(".bin")
        if bin_path.exists():
            return load_lidar_bin(bin_path)
        raise FileNotFoundError(f"neither {pcd} nor {bin_path} exists")

    return read


def _no_pcd_reader(pcd_file: str) -> np.ndarray:
    raise FileNotFoundError(f"V2X-Real has no .pcd files; {pcd_file} should not exist")


def _refuse_corrupt(pcd: Path) -> None:
    key = (pcd.parent.parent.name, pcd.parent.name, pcd.stem)
    if key in CORRUPT_FRAMES:
        raise FileNotFoundError(
            f"{pcd.with_suffix('.bin')} was corrupt in the archive and deleted; "
            "skip this frame (alignformer.v2xreal.CORRUPT_FRAMES)"
        )


# ----------------------------------------------------------------------------
# Frame yaml
# ----------------------------------------------------------------------------
#
# Profiled on the NVMe copy: 0.83 s of a 0.93 s training item was pure-Python
# yaml parsing, 21 loads per item for 4 agents, most of them the same four
# files reloaded by basedataset's distance calculation and parameter loading.
# libyaml's C loader with OpenCOOD's float resolver parses the same text
# several times faster; a bounded per-process cache removes the repeats.

_BaseLoader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)


class _FrameLoader(_BaseLoader):  # type: ignore[misc,valid-type]
    """libyaml with the implicit float resolver ``yaml_utils.load_yaml`` adds."""


# Verbatim from opencood/hypes_yaml/yaml_utils.py: without it YAML 1.1 reads
# `1e-10` as a string.
_FrameLoader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(
        r"""^(?:
         [-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\.[0-9_]*
        |[-+]?\.(?:inf|Inf|INF)
        |\.(?:nan|NaN|NAN))$""",
        re.X,
    ),
    list("-+0123456789."),
)


@functools.lru_cache(maxsize=YAML_CACHE_SIZE)
def _parse_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as stream:
        return yaml.load(stream, Loader=_FrameLoader)


def fast_load_yaml(file: str, opt: Optional[Any] = None) -> Dict[str, Any]:
    """Drop-in for ``yaml_utils.load_yaml`` on frame yamls.

    A deep copy of the cached parse, because OpenCOOD is free to edit what
    it is handed. Hypes loading (``opt.model_dir``, ``yaml_parser``) is
    OpenCOOD's job and is deferred to it.
    """
    if opt is not None and getattr(opt, "model_dir", None):
        from opencood.hypes_yaml.yaml_utils import load_yaml

        return load_yaml(file, opt)
    return copy.deepcopy(_parse_yaml(str(file)))


# ----------------------------------------------------------------------------
# Classes and agents
# ----------------------------------------------------------------------------


def filter_vehicles(params: Dict[str, Any]) -> Dict[str, Any]:
    """A new params dict whose ``vehicles`` keeps only :data:`VEHICLE_TYPES`."""
    vehicles = {
        object_id: dict(entry)
        for object_id, entry in params["vehicles"].items()
        if entry.get("obj_type") in VEHICLE_TYPES
    }
    filtered = dict(params)
    filtered["vehicles"] = vehicles
    return filtered


def order_agents(agent_ids: Iterable[str]) -> Tuple[str, ...]:
    """Vehicle-centric order: vehicles ascending, then roadside units.

    Numeric, not lexical (``"10"`` after ``"2"``). Refuses a scenario with no
    vehicle, because the protocol has no ego for it.
    """
    ids = list(agent_ids)
    vehicles = sorted((i for i in ids if int(i) >= 0), key=int)
    infrastructure = sorted((i for i in ids if int(i) < 0), key=lambda i: -int(i))
    if not vehicles:
        raise ValueError(f"no vehicle agent among {ids}: vehicle-centric ego undefined")
    return tuple(vehicles + infrastructure)


# ----------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------


def _dataset_class():
    """Built lazily so importing this module never imports OpenCOOD."""
    from opencood.data_utils.datasets.late_fusion_dataset import LateFusionDataset

    class V2XRealLateFusionDataset(LateFusionDataset):
        """OpenCOOD's late-fusion dataset with the three V2X-Real differences."""

        def __init__(self, params, visualize, train=True):
            super().__init__(params, visualize, train)
            database = _drop_corrupt_timestamps(self.scenario_database)
            self.scenario_database = _vehicle_centric(database)
            self.len_record = _len_record(self.scenario_database)

        def retrieve_base_data(self, idx, cur_ego_pose_flag=True):
            data = super().retrieve_base_data(idx, cur_ego_pose_flag)
            filtered = OrderedDict()
            for cav_id, content in data.items():
                entry = dict(content)
                entry["params"] = filter_vehicles(content["params"])
                filtered[cav_id] = entry
            return filtered

        # The three hooks the cache builder asks for when it assembles one
        # agent-frame from paths, outside retrieve_base_data.

        @staticmethod
        def frame_params(yaml_path: Path | str) -> Dict[str, Any]:
            return filter_vehicles(fast_load_yaml(str(yaml_path)))

        @staticmethod
        def frame_points(lidar_path: Path | str) -> np.ndarray:
            return bin_pcd_to_np(_no_pcd_reader)(str(lidar_path))

    return V2XRealLateFusionDataset


def _drop_corrupt_timestamps(scenario_database: "OrderedDict") -> "OrderedDict":
    """Every scenario without the timestamps of :data:`CORRUPT_FRAMES`.

    Dropped for EVERY agent of the scenario, not only the one whose LiDAR was
    corrupt: ``retrieve_base_data`` looks each agent up by the ego's timestamp
    key, so a timestamp missing from one agent is a ``KeyError`` for the frame.
    A refusal at read time (``bin_pcd_to_np``) stays as the backstop.
    """
    corrupt = {}
    for scenario, agent, timestamp in CORRUPT_FRAMES:
        corrupt.setdefault(scenario, set()).add(timestamp)
    cleaned = OrderedDict()
    for scenario_index, agents in scenario_database.items():
        scenario_name = _scenario_name(agents)
        drop = corrupt.get(scenario_name, set())
        cleaned[scenario_index] = OrderedDict(
            (cav_id, OrderedDict((k, v) for k, v in content.items() if k not in drop))
            for cav_id, content in agents.items()
        )
    return cleaned


def _scenario_name(agents: "OrderedDict") -> str:
    first = next(iter(agents.values()))
    yaml_path = next(v["yaml"] for k, v in first.items() if k != "ego")
    return Path(yaml_path).parents[1].name


def _len_record(scenario_database: "OrderedDict") -> list:
    """Cumulative ego-frame counts, as ``basedataset.__init__`` builds them."""
    record, total = [], 0
    for agents in scenario_database.values():
        first = next(iter(agents.values()))
        total += sum(1 for k in first if k != "ego")
        record.append(total)
    return record


def _vehicle_centric(scenario_database: "OrderedDict") -> "OrderedDict":
    """Every scenario reordered by :func:`order_agents`, ego on the first."""
    reordered = OrderedDict()
    for scenario_index, agents in scenario_database.items():
        order = order_agents(agents.keys())
        scenario = OrderedDict()
        for position, cav_id in enumerate(order):
            content = OrderedDict(agents[cav_id])
            content["ego"] = position == 0
            scenario[cav_id] = content
        reordered[scenario_index] = scenario
    return reordered


# ----------------------------------------------------------------------------
# Building
# ----------------------------------------------------------------------------


def install_shims(hypes: Dict[str, Any]) -> None:
    """Patch the LiDAR and yaml readers; set OpenCOOD's ranges from ``hypes['v2xreal']``.

    Idempotent: the reader is wrapped once (a second call finds the wrapper
    already installed), and the ranges are plain assignments.
    """
    import opencood.data_utils.datasets as opencood_datasets
    from opencood.utils import pcd_utils

    if getattr(pcd_utils.pcd_to_np, "__v2xreal_shim__", False) is False:
        shim = bin_pcd_to_np(pcd_utils.pcd_to_np)
        shim.__v2xreal_shim__ = True  # type: ignore[attr-defined]
        pcd_utils.pcd_to_np = shim

    import opencood.data_utils.datasets.basedataset as basedataset
    import opencood.data_utils.datasets.late_fusion_dataset as late_fusion

    basedataset.load_yaml = fast_load_yaml
    late_fusion.load_yaml = fast_load_yaml

    ranges = hypes.get("v2xreal")
    if not ranges:
        raise ValueError(
            f"a {CORE_METHOD} config needs a 'v2xreal' block with gt_range and comm_range"
        )
    opencood_datasets.GT_RANGE = list(ranges["gt_range"])
    opencood_datasets.COM_RANGE = ranges["comm_range"]


def build_dataset(hypes: Dict[str, Any], visualize: bool = False, train: bool = True):
    """OpenCOOD's ``build_dataset``, plus :data:`CORE_METHOD`.

    Every other ``core_method`` goes to OpenCOOD unchanged, so the OPV2V path
    is bit-identical to before this module existed.
    """
    import opencood.data_utils.datasets as opencood_datasets

    if hypes["fusion"]["core_method"] != CORE_METHOD:
        return opencood_datasets.build_dataset(hypes, visualize, train)
    install_shims(hypes)
    return _dataset_class()(params=hypes, visualize=visualize, train=train)


__all__ = [
    "CORE_METHOD",
    "CORRUPT_FRAMES",
    "V2XREAL_COM_RANGE",
    "V2XREAL_GT_RANGE",
    "VEHICLE_TYPES",
    "bin_pcd_to_np",
    "build_dataset",
    "fast_load_yaml",
    "filter_vehicles",
    "install_shims",
    "load_lidar_bin",
    "order_agents",
]
