from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF

from src.core.models import DocumentRecord, PageRecord
from src.core.utils import logger


def extract_pdf_document(doc: DocumentRecord, cfg: dict) -> list[PageRecord]:
    pdf_cfg = cfg["extractors"]["pdf"]
    path = Path(doc.path)

    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")

    logger.info(f"Extracting PDF: {doc.doc_id}")
    pages: list[PageRecord] = []

    engine = pdf_cfg.get("engine", "pymupdf").lower()
    if engine != "pymupdf":
        logger.warning("Only pymupdf is implemented in this version. Falling back to pymupdf.")

    pdf = fitz.open(path)
    for i, page in enumerate(pdf, start=1):
        text = page.get_text("text")
        pages.append(
            PageRecord(
                doc_id=doc.doc_id,
                page_num=i,
                text=text or "",
                source_kind="pdf",
                extra={"file_name": doc.file_name},
            )
        )
    return pages
