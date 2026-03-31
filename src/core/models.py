from __future__ import annotations

from dataclasses import dataclass, field, asdict
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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PageRecord:
    doc_id: str
    page_num: int
    text: str
    source_kind: str
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ChunkRecord:
    chunk_id: str
    text: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HeadingInfo:
    level: str
    text: str


@dataclass
class SectionBlock:
    text: str
    h1: str = ""
    h2: str = ""
    h3: str = ""
    page_num: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationIssue:
    level: str
    chunk_id: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)