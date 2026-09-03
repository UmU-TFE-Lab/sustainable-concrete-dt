"""Configuration and serialization helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def load_json(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_run_config(path: Path) -> Dict[str, Any]:
    path = Path(path).resolve()
    config = load_json(path)
    root = path.parents[1]
    config["_root"] = str(root)
    for key in ("data_path", "engineering_config", "lca_config", "output_dir"):
        config[key] = str((root / config[key]).resolve())
    return config


def write_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
