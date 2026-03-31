from __future__ import annotations

import argparse
from pathlib import Path

from src.core.config_loader import load_configs
from src.core.models import ChunkRecord, DocumentRecord, PageRecord
from src.core.utils import ensure_dir, logger, write_jsonl
from src.ingest.cleaner import clean_pages
from src.ingest.chunker import chunk_document
from src.ingest.deduplicator import deduplicate_chunks
from src.ingest.extract_mhtml import extract_mhtml_document
from src.ingest.extract_pdf import extract_pdf_document
from src.ingest.validator import validate_chunks


def _save_pages(pages: list[PageRecord], output_path: str | Path) -> None:
    write_jsonl([p.to_dict() for p in pages], output_path)


def _save_chunks(chunks: list[ChunkRecord], output_path: str | Path) -> None:
    write_jsonl([c.to_dict() for c in chunks], output_path)


def _process_one_document(doc: DocumentRecord, cfg: dict) -> list[ChunkRecord]:
    source_kind = doc.source_kind.lower()

    if source_kind == "pdf":
        raw_pages = extract_pdf_document(doc, cfg)
    elif source_kind == "mhtml":
        raw_pages = extract_mhtml_document(doc, cfg)
    else:
        logger.warning(f"Unsupported source_kind={source_kind} for {doc.doc_id}")
        return []

    raw_text_dir = Path(cfg["paths"]["raw_text_dir"])
    clean_text_dir = Path(cfg["paths"]["clean_text_dir"])

    _save_pages(raw_pages, raw_text_dir / f"{doc.doc_id}.jsonl")

    cleaned_pages = clean_pages(doc, raw_pages, cfg)
    _save_pages(cleaned_pages, clean_text_dir / f"{doc.doc_id}.jsonl")

    chunks = chunk_document(doc, cleaned_pages, cfg)
    return chunks


def build_pipeline(config_path: str) -> list[ChunkRecord]:
    cfg, manifest_cfg, documents = load_configs(config_path)

    enabled_doc_types = set(cfg["mvp_scope"]["enabled_doc_types"])
    chunks_dir = Path(cfg["paths"]["chunks_dir"])
    chunks_file = Path(cfg["paths"]["chunks_file"])
    logs_dir = Path(cfg["paths"]["logs_dir"])

    ensure_dir(chunks_dir)
    ensure_dir(logs_dir)

    all_chunks: list[ChunkRecord] = []

    for doc in documents:
        if doc.status != "active":
            continue
        if doc.doc_type not in enabled_doc_types:
            continue

        logger.info(f"Processing document: {doc.doc_id}")
        doc_chunks = _process_one_document(doc, cfg)
        _save_chunks(doc_chunks, chunks_dir / f"{doc.doc_id}.jsonl")
        all_chunks.extend(doc_chunks)

    all_chunks = deduplicate_chunks(all_chunks, cfg)
    validate_chunks(all_chunks, cfg)
    _save_chunks(all_chunks, chunks_file)

    logger.info(f"Done. Total chunks: {len(all_chunks)}")
    return all_chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Build RAG chunks from V20 documents")
    parser.add_argument("--config", required=True, help="Path to chunking_config.yaml")
    args = parser.parse_args()

    build_pipeline(args.config)


if __name__ == "__main__":
    main()
