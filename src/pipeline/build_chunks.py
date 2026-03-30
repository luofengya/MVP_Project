from pathlib import Path
from src.core.config_loader import load_configs
from src.ingest.extract_pdf import extract_pdf_document
from src.ingest.extract_mhtml import extract_mhtml_document
from src.ingest.cleaner import clean_pages
from src.ingest.chunker import chunk_document
from src.ingest.deduplicator import deduplicate_chunks
from src.ingest.validator import validate_chunks

def build_pipeline(config_path: str):
    chunk_cfg, manifest_cfg = load_configs(config_path)

    docs = manifest_cfg["documents"]
    enabled_doc_types = set(chunk_cfg["mvp_scope"]["enabled_doc_types"])
    pipeline_cfg = chunk_cfg["pipeline"]

    all_chunks = []

    for doc in docs:
        if doc["status"] != "active":
            continue
        if doc["doc_type"] not in enabled_doc_types:
            continue

        source_kind = doc["source_kind"].lower()

        if pipeline_cfg["extract"]:
            if source_kind == "pdf":
                pages = extract_pdf_document(doc, chunk_cfg)
            elif source_kind == "mhtml":
                pages = extract_mhtml_document(doc, chunk_cfg)
            else:
                continue
        else:
            # 后面你可以从 raw_text 直接读取
            pages = []

        if pipeline_cfg["clean"]:
            pages = clean_pages(doc, pages, chunk_cfg)

        if pipeline_cfg["chunk"]:
            chunks = chunk_document(doc, pages, chunk_cfg)
            all_chunks.extend(chunks)

    if pipeline_cfg["deduplicate"]:
        all_chunks = deduplicate_chunks(all_chunks, chunk_cfg)

    if pipeline_cfg["validate"]:
        validate_chunks(all_chunks, chunk_cfg)

    return all_chunks
