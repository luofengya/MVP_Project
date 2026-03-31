from __future__ import annotations

from email import policy
from email.parser import BytesParser
from pathlib import Path

import trafilatura
from bs4 import BeautifulSoup

from src.core.models import DocumentRecord, PageRecord
from src.core.utils import logger


def _read_mhtml_html(path: str | Path) -> str:
    path = Path(path)
    with path.open("rb") as f:
        msg = BytesParser(policy=policy.default).parse(f)

    for part in msg.walk():
        if part.get_content_type() == "text/html":
            payload = part.get_payload(decode=True)
            charset = part.get_content_charset() or "utf-8"
            return payload.decode(charset, errors="ignore")
    return ""


def extract_mhtml_document(doc: DocumentRecord, cfg: dict) -> list[PageRecord]:
    mhtml_cfg = cfg["extractors"]["mhtml"]
    path = Path(doc.path)

    if not path.exists():
        raise FileNotFoundError(f"MHTML file not found: {path}")

    logger.info(f"Extracting MHTML: {doc.doc_id}")
    html = _read_mhtml_html(path)
    if not html:
        return [PageRecord(doc_id=doc.doc_id, page_num=1, text="", source_kind="mhtml")]

    engine = mhtml_cfg.get("engine", "trafilatura").lower()
    text = ""

    if engine == "trafilatura":
        text = trafilatura.extract(
            html,
            include_links=False,
            include_tables=True,
            include_comments=False,
            include_formatting=False,
        ) or ""

    if not text:
        soup = BeautifulSoup(html, "lxml")
        text = soup.get_text("\n", strip=True)

    return [
        PageRecord(
            doc_id=doc.doc_id,
            page_num=1,
            text=text,
            source_kind="mhtml",
            extra={"file_name": doc.file_name},
        )
    ]
