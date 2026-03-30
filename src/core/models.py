from dataclasses import dataclass, field
from typing import Any

@dataclass
class DocumentRecord:
    doc_id: str
    path: str
    relative_path: str
    file_name: str
    source_kind: str
    device_family: str
    brand: str
    language: str
    doc_title: str
    doc_type: str
    doc_role: str
    source_priority: str
    retrieval_boost: float = 1.0
    status: str = "active"
    tags: list[str] = field(default_factory=list)
    notes: str = ""

@dataclass
class PageRecord:
    doc_id: str
    page_num: int
    text: str
    source_kind: str

@dataclass
class ChunkRecord:
    chunk_id: str
    text: str
    metadata: dict[str, Any]
