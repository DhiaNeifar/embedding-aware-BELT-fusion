"""V2X-Real inside CoAlign's OpenCOOD fork (also used by RoCo): an adapter, not a fork.

CoAlign and RoCo are learning-based, so on V2X-Real they need their own stage-1 detector (single-agent boxes with
uncertainty), trained on V2X-Real in CoAlign's own code. CoAlign's code knows OPV2V, V2X-Sim, DAIR-V2X and V2XSet,
not V2X-Real. This module adds V2X-Real WITHOUT editing ``external/CoAlign``: it defines a base dataset and installs
it into ``opencood.data_utils.datasets.build_dataset`` at run time (``scripts/coalign_v2xreal.py``).

The V2X-Real rules are the ones of our late-fusion adapter (``multidataset/v2xreal.py``), reused, not re-derived, so
frames, ego, CAVs and ground truth are the same as everywhere else in the paper:

1. LiDAR is ``<ts>.bin`` (float32 x, y, z, intensity), read through the ``pcd_utils.pcd_to_np`` slot, non-finite
   rows dropped; the one corrupt frame of the archive is refused by name and removed from the index.
2. Ground truth is the ``vehicle`` super-class only (Car, PoliceCar, LongVehicle).
3. Vehicle-centric agents: vehicles first (numeric order), then roadside units; the ego is the lowest vehicle.

Ranges come from the config, as in every CoAlign config: ``comm_range`` and ``postprocess.gt_range``. The V2X-Real
values (70 m, +-102.4 x +-51.2 m) are in ``models/v2xreal/intermediate_fusion/coalign_boxalign/*.yaml``.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict

from embedding_aware_belt_fusion.multidataset import v2xreal as rules

DATASET_NAME = "v2xreal"


def _base_dataset_class():
    """Built lazily, so importing this module never imports CoAlign's OpenCOOD."""
    import opencood.data_utils.datasets.basedataset.opv2v_basedataset as opv2v_base

    class V2XREALBaseDataset(opv2v_base.OPV2VBaseDataset):
        """CoAlign's OPV2V base dataset with the three V2X-Real differences."""

        def reinitialize(self):
            super().reinitialize()
            database = rules._drop_corrupt_timestamps(self.scenario_database)
            self.scenario_database = rules._vehicle_centric(database)
            self.len_record = rules._len_record(self.scenario_database)

        def retrieve_base_data(self, idx):
            data = super().retrieve_base_data(idx)
            filtered = OrderedDict()
            for cav_id, content in data.items():
                entry = OrderedDict(content)
                entry["params"] = rules.filter_vehicles(content["params"])
                filtered[cav_id] = entry
            return filtered

        def frame_key(self, idx):
            """(scenario folder name, timestamp, agent ids in order) of dataset index ``idx``, for parity checks."""
            scenario_index = next(i for i, end in enumerate(self.len_record) if idx < end)
            database = self.scenario_database[scenario_index]
            local = idx if scenario_index == 0 else idx - self.len_record[scenario_index - 1]
            timestamp = self.return_timestamp_key(database, local)
            return rules._scenario_name(database), timestamp, tuple(database.keys())

    return V2XREALBaseDataset


def install() -> None:
    """Make CoAlign's ``build_dataset`` accept ``fusion.dataset: v2xreal``. Idempotent.

    Also installs the LiDAR reader (``.bin``) and the fast frame-yaml reader. OPV2V and the other datasets keep
    going to CoAlign's own ``build_dataset`` unchanged.
    """
    import opencood.data_utils.datasets as datasets
    import opencood.data_utils.datasets.basedataset.opv2v_basedataset as opv2v_base
    from opencood.utils import pcd_utils

    if getattr(datasets.build_dataset, "__v2xreal__", False):
        return
    if not getattr(pcd_utils.pcd_to_np, "__v2xreal_shim__", False):
        shim = rules.bin_pcd_to_np(pcd_utils.pcd_to_np)
        shim.__v2xreal_shim__ = True
        pcd_utils.pcd_to_np = shim
    # The fast frame-yaml reader is for V2X-Real files only: OPV2V's frame yamls carry python/numpy tags that only
    # CoAlign's own (full) loader reads, so every other file keeps going to it.
    original_load_yaml = opv2v_base.load_yaml

    def load_yaml(file, opt=None):
        if "v2x-real" in str(file):
            return rules.fast_load_yaml(file, opt)
        return original_load_yaml(file, opt)

    opv2v_base.load_yaml = load_yaml

    base_cls = _base_dataset_class()
    original = datasets.build_dataset

    def build_dataset(dataset_cfg: Dict[str, Any], visualize: bool = False, train: bool = True):
        if dataset_cfg["fusion"]["dataset"] != DATASET_NAME:
            return original(dataset_cfg, visualize, train)
        fusion_name = dataset_cfg["fusion"]["core_method"]
        assert fusion_name in ("late", "intermediate", "intermediate2stage", "early"), fusion_name
        fusion_func = getattr(datasets, "get" + fusion_name.capitalize() + "FusionDataset")
        return fusion_func(base_cls)(params=dataset_cfg, visualize=visualize, train=train)

    build_dataset.__v2xreal__ = True
    datasets.build_dataset = build_dataset
    datasets.V2XREALBaseDataset = base_cls


__all__ = ["DATASET_NAME", "install"]
