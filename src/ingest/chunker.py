from __future__ import annotations

import re
from typing import Any

from src.core.models import DocumentRecord, PageRecord, ChunkRecord, SectionBlock
from src.core.utils import chunk_text_by_size, clean_empty_values
from src.ingest.heading_parser import split_by_headings
from src.ingest.field_extractors import enrich_structured_fields


def resolve_chunker_name(doc: DocumentRecord, cfg: dict) -> str:
    overrides = cfg.get("doc_overrides", {})
    if doc.doc_id in overrides and "chunker" in overrides[doc.doc_id]:
        return overrides[doc.doc_id]["chunker"]

    for chunker_name, chunker_cfg in cfg["chunkers"].items():
        if doc.doc_type in chunker_cfg.get("applies_to", []):
            return chunker_name

    raise ValueError(f"No chunker matched for doc_type={doc.doc_type}")


def build_base_metadata(doc: DocumentRecord, page: PageRecord, cfg: dict) -> dict[str, Any]:
    return {
        "device_family": doc.device_family,
        "brand": doc.brand,
        "language": doc.language,
        "doc_id": doc.doc_id,
        "doc_title": doc.doc_title,
        "doc_type": doc.doc_type,
        "doc_role": doc.doc_role,
        "source_kind": doc.source_kind,
        "source_priority": doc.source_priority,
        "retrieval_boost": doc.retrieval_boost,
        "page_start": page.page_num,
        "page_end": page.page_num,
        "keywords_zh": [],
    }


def apply_metadata_profile(metadata: dict[str, Any], profile_name: str, cfg: dict) -> dict[str, Any]:
    allowed = set(cfg["metadata_profiles"][profile_name]["add_fields"])
    filtered = {k: v for k, v in metadata.items() if k in allowed}
    return clean_empty_values(filtered)


def make_chunk_id(doc: DocumentRecord, page_start: int, local_index: int, cfg: dict) -> str:
    template = cfg["naming"]["chunk_id_template"]
    return template.format(doc_id=doc.doc_id, page_start=page_start, local_index=f"{local_index:03d}")


def _chunk_long_block(block_text: str, target_chars: int, overlap_chars: int) -> list[str]:
    return chunk_text_by_size(block_text, target_chars=target_chars, overlap_chars=overlap_chars)


def _extract_section_titles(block: SectionBlock) -> dict[str, str]:
    return {
        "section_h1": block.h1,
        "section_h2": block.h2,
        "section_h3": block.h3,
    }


def chunk_manual(doc: DocumentRecord, pages: list[PageRecord], cfg: dict) -> list[ChunkRecord]:
    chunk_cfg = cfg["chunkers"]["manual_chunker"]
    base_cfg = chunk_cfg["base"]
    target_chars = int(base_cfg["target_chars"])
    overlap_chars = int(base_cfg["overlap_chars"])
    profile_name = chunk_cfg["metadata_profile"]

    chunks: list[ChunkRecord] = []

    for page in pages:
        blocks = split_by_headings(page.text, cfg, page.page_num)
        if not blocks:
            blocks = [SectionBlock(text=page.text, page_num=page.page_num)]

        local_index = 1
        for block in blocks:
            parts = _chunk_long_block(block.text, target_chars, overlap_chars)
            for part in parts:
                metadata = build_base_metadata(doc, page, cfg)
                metadata.update(_extract_section_titles(block))
                metadata = enrich_structured_fields(part, metadata, cfg)
                metadata = apply_metadata_profile(metadata, profile_name, cfg)

                chunks.append(
                    ChunkRecord(
                        chunk_id=make_chunk_id(doc, page.page_num, local_index, cfg),
                        text=part,
                        metadata=metadata,
                    )
                )
                local_index += 1

    return chunks


def chunk_steps(doc: DocumentRecord, pages: list[PageRecord], cfg: dict) -> list[ChunkRecord]:
    chunk_cfg = cfg["chunkers"]["step_chunker"]
    target_chars = int(chunk_cfg["base"]["target_chars"])
    overlap_chars = int(chunk_cfg["base"]["overlap_chars"])
    profile_name = chunk_cfg["metadata_profile"]

    step_patterns = [re.compile(p) for p in chunk_cfg["workflow_rules"]["step_patterns"]]
    chunks: list[ChunkRecord] = []

    for page in pages:
        lines = [line.strip() for line in page.text.splitlines() if line.strip()]
        step_blocks: list[str] = []
        current: list[str] = []

        for line in lines:
            if any(p.match(line) for p in step_patterns) and current:
                step_blocks.append("\n".join(current))
                current = [line]
            else:
                current.append(line)

        if current:
            step_blocks.append("\n".join(current))

        if not step_blocks:
            step_blocks = _chunk_long_block(page.text, target_chars, overlap_chars)

        for idx, block in enumerate(step_blocks, start=1):
            metadata = build_base_metadata(doc, page, cfg)
            metadata["step_no"] = idx
            metadata = enrich_structured_fields(block, metadata, cfg)
            metadata = apply_metadata_profile(metadata, profile_name, cfg)

            chunks.append(
                ChunkRecord(
                    chunk_id=make_chunk_id(doc, page.page_num, idx, cfg),
                    text=block,
                    metadata=metadata,
                )
            )

    return chunks


def chunk_codes(doc: DocumentRecord, pages: list[PageRecord], cfg: dict) -> list[ChunkRecord]:
    chunk_cfg = cfg["chunkers"]["code_chunker"]
    profile_name = chunk_cfg["metadata_profile"]
    code_patterns = chunk_cfg["code_rules"]["event_code_patterns"]

    code_regex = re.compile("|".join(f"({p})" for p in code_patterns), re.IGNORECASE)
    chunks: list[ChunkRecord] = []

    for page in pages:
        text = page.text
        matches = list(code_regex.finditer(text))

        if not matches:
            block_list = [text]
        else:
            block_list = []
            for i, m in enumerate(matches):
                start = m.start()
                end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
                block = text[start:end].strip()
                if block:
                    block_list.append(block)

        for idx, block in enumerate(block_list, start=1):
            metadata = build_base_metadata(doc, page, cfg)
            metadata = enrich_structured_fields(block, metadata, cfg)
            metadata = apply_metadata_profile(metadata, profile_name, cfg)

            chunks.append(
                ChunkRecord(
                    chunk_id=make_chunk_id(doc, page.page_num, idx, cfg),
                    text=block,
                    metadata=metadata,
                )
            )

    return chunks


def chunk_topics(doc: DocumentRecord, pages: list[PageRecord], cfg: dict) -> list[ChunkRecord]:
    chunk_cfg = cfg["chunkers"]["topic_chunker"]
    target_chars = int(chunk_cfg["base"]["target_chars"])
    overlap_chars = int(chunk_cfg["base"]["overlap_chars"])
    profile_name = chunk_cfg["metadata_profile"]
    topic_keywords = chunk_cfg["topic_rules"]["topic_keywords"]

    chunks: list[ChunkRecord] = []

    for page in pages:
        blocks = split_by_headings(page.text, cfg, page.page_num)
        if not blocks:
            blocks = [SectionBlock(text=page.text, page_num=page.page_num)]

        local_index = 1
        for block in blocks:
            parts = _chunk_long_block(block.text, target_chars, overlap_chars)
            for part in parts:
                metadata = build_base_metadata(doc, page, cfg)
                metadata.update(_extract_section_titles(block))

                for topic_name, keywords in topic_keywords.items():
                    if any(kw.lower() in part.lower() for kw in keywords):
                        metadata["comm_topic"] = topic_name
                        break

                metadata = enrich_structured_fields(part, metadata, cfg)
                metadata = apply_metadata_profile(metadata, profile_name, cfg)

                chunks.append(
                    ChunkRecord(
                        chunk_id=make_chunk_id(doc, page.page_num, local_index, cfg),
                        text=part,
                        metadata=metadata,
                    )
                )
                local_index += 1

    return chunks


def chunk_cases(doc: DocumentRecord, pages: list[PageRecord], cfg: dict) -> list[ChunkRecord]:
    chunk_cfg = cfg["chunkers"]["case_chunker"]
    target_chars = int(chunk_cfg["base"]["target_chars"])
    overlap_chars = int(chunk_cfg["base"]["overlap_chars"])
    profile_name = chunk_cfg["metadata_profile"]

    chunks: list[ChunkRecord] = []

    for page in pages:
        parts = _chunk_long_block(page.text, target_chars, overlap_chars)
        for idx, part in enumerate(parts, start=1):
            metadata = build_base_metadata(doc, page, cfg)
            metadata = enrich_structured_fields(part, metadata, cfg)
            metadata = apply_metadata_profile(metadata, profile_name, cfg)

            chunks.append(
                ChunkRecord(
                    chunk_id=make_chunk_id(doc, page.page_num, idx, cfg),
                    text=part,
                    metadata=metadata,
                )
            )

    return chunks


def chunk_document(doc: DocumentRecord, pages: list[PageRecord], cfg: dict) -> list[ChunkRecord]:
    chunker_name = resolve_chunker_name(doc, cfg)

    if chunker_name == "manual_chunker":
        return chunk_manual(doc, pages, cfg)
    if chunker_name == "step_chunker":
        return chunk_steps(doc, pages, cfg)
    if chunker_name == "code_chunker":
        return chunk_codes(doc, pages, cfg)
    if chunker_name == "topic_chunker":
        return chunk_topics(doc, pages, cfg)
    if chunker_name == "case_chunker":
        return chunk_cases(doc, pages, cfg)

    raise ValueError(f"Unsupported chunker: {chunker_name}")
