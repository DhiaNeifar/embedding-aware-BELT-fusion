#!/usr/bin/env python3
"""Load each V2X-Real network checkpoint into jigsawcomm-v2x (the code it was trained with) and report missing or
unexpected weights. Load only: no data, no inference.

    cd external/jigsawcomm-v2x && \
      /mnt/external/backup/dhia/conda_envs/coalign-blackwell/bin/python ../../scripts/check_v2xreal_networks.py \
      attfuse fcooper v2vam v2xvit where2comm
"""

from __future__ import annotations

import sys
import tempfile
import types
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
CODE = REPO / "external" / "jigsawcomm-v2x"
sys.modules.setdefault("efficientnet_pytorch", types.SimpleNamespace(EfficientNet=None))   # camera-only dependency
sys.modules.setdefault("pypcd", types.SimpleNamespace(pypcd=None))                          # V2X-Real is .bin
sys.path.insert(0, str(CODE))

from opencood.hypes_yaml.yaml_utils import load_yaml  # noqa: E402
from opencood.tools import train_utils  # noqa: E402


def main() -> None:
    for name in sys.argv[1:] or ["attfuse", "fcooper", "v2vam", "v2xvit", "where2comm"]:
        folder = REPO / "models" / "v2xreal" / "intermediate_fusion" / name
        try:
            config = Path(tempfile.mkdtemp()) / "config.yaml"
            config.write_text((folder / "config.yaml").read_text())
            hypes = load_yaml(str(config))
            model = train_utils.create_model(hypes)
            checkpoint = sorted(folder.glob("net_epoch*.pth"))[-1]
            missing, unexpected = model.load_state_dict(torch.load(checkpoint, map_location="cpu"), strict=False)
            status = "LOADS EXACTLY" if not missing and not unexpected else "MISMATCH"
            print(f"{name:12s} {status:14s} {hypes['model']['core_method']:28s} {checkpoint.name}  "
                  f"missing {len(missing)} {missing[:2]}  unexpected {len(unexpected)} {unexpected[:2]}", flush=True)
        except Exception as error:  # noqa: BLE001
            print(f"{name:12s} FAILS          {type(error).__name__}: {str(error)[:160]}", flush=True)


if __name__ == "__main__":
    main()
