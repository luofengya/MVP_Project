from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.core.models import DocumentRecord


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def load_documents_from_manifest(manifest_cfg: dict[str, Any]) -> list[DocumentRecord]:
    docs: list[DocumentRecord] = []
    for item in manifest_cfg.get("documents", []):
        docs.append(
            DocumentRecord(
                doc_id=item["doc_id"],
                path=item["path"],
                relative_path=item.get("relative_path", ""),
                file_name=item.get("file_name", Path(item["path"]).name),
                source_kind=item["source_kind"],
                device_family=item["device_family"],
                brand=item["brand"],
                language=item["language"],
                doc_title=item["doc_title"],
                doc_type=item["doc_type"],
                doc_role=item["doc_role"],
                source_priority=item["source_priority"],
                retrieval_boost=float(item.get("retrieval_boost", 1.0)),
                status=item.get("status", "active"),
                tags=item.get("tags", []),
                notes=item.get("notes", ""),
            )
        )
    return docs


def load_configs(chunking_config_path: str | Path) -> tuple[dict[str, Any], dict[str, Any], list[DocumentRecord]]:
    chunk_cfg = load_yaml(chunking_config_path)
    manifest_path = chunk_cfg["manifest_path"]
    manifest_cfg = load_yaml(manifest_path)
    documents = load_documents_from_manifest(manifest_cfg)
    return chunk_cfg, manifest_cfg, documents
