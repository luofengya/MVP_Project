from __future__ import annotations
import yaml
from pathlib import Path
from typing import Any


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_configs(chunking_config_path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    chunk_cfg = load_yaml(chunking_config_path)
    manifest_path = chunk_cfg["manifest_path"]
    manifest_cfg = load_yaml(manifest_path)
    return chunk_cfg, manifest_cfg
