from __future__ import annotations

import re
import unicodedata

from src.core.models import DocumentRecord, PageRecord
from src.core.utils import logger


def normalize_text(text: str, cfg: dict) -> str:
    norm_cfg = cfg["global"]["text_normalization"]

    normalize_mode = norm_cfg.get("unicode_normalize")
    if normalize_mode:
        text = unicodedata.normalize(normalize_mode, text)

    if norm_cfg.get("fullwidth_to_halfwidth", False):
        text = unicodedata.normalize("NFKC", text)

    if norm_cfg.get("normalize_colons", False):
        text = text.replace("：", ":")

    if norm_cfg.get("normalize_parentheses", False):
        text = text.replace("（", "(").replace("）", ")")

    if norm_cfg.get("normalize_dashes", False):
        text = text.replace("—", "-").replace("–", "-")

    if norm_cfg.get("collapse_multiple_spaces", False):
        text = re.sub(r"[ \t]+", " ", text)

    if norm_cfg.get("collapse_blank_lines", False):
        text = re.sub(r"\n\s*\n+", "\n\n", text)

    if norm_cfg.get("trim_line_edges", False):
        text = "\n".join(line.strip() for line in text.splitlines())

    return text.strip()


def remove_boilerplate(text: str, cfg: dict) -> str:
    cleaning_cfg = cfg["global"]["cleaning"]
    lines = text.splitlines()
    cleaned_lines: list[str] = []

    block_patterns: list[str] = []
    if cleaning_cfg.get("remove_navigation_text", False):
        block_patterns.extend(
            [
                r"^首页$",
                r".*Cookie.*",
                r".*联系我们.*",
                r".*推荐.*",
                r".*相关链接.*",
                r".*技术支持.*",
            ]
        )
    if cleaning_cfg.get("remove_legal_disclaimer", False):
        block_patterns.extend(
            [
                r".*法律声明.*",
                r".*责任免除.*",
                r".*免责.*",
            ]
        )
    if cleaning_cfg.get("remove_trademark_notices", False):
        block_patterns.extend(
            [
                r".*商标.*",
                r".*All rights reserved.*",
                r".*保留所有权利.*",
            ]
        )

    for line in lines:
        line_strip = line.strip()
        if not line_strip:
            cleaned_lines.append("")
            continue

        skip = False
        for pattern in block_patterns:
            if re.search(pattern, line_strip, re.IGNORECASE):
                skip = True
                break

        if not skip:
            cleaned_lines.append(line_strip)

    return "\n".join(cleaned_lines).strip()


def should_skip_page(text: str, cfg: dict) -> bool:
    filters = cfg["global"]["filters"]

    if len(text.strip()) < int(filters.get("min_clean_text_chars_per_page", 30)):
        return True

    for kw in filters.get("skip_pages_matching", []):
        if kw and kw in text:
            return True

    return False


def clean_pages(doc: DocumentRecord, pages: list[PageRecord], cfg: dict) -> list[PageRecord]:
    logger.info(f"Cleaning pages: {doc.doc_id}")
    cleaned: list[PageRecord] = []

    for page in pages:
        text = normalize_text(page.text, cfg)
        text = remove_boilerplate(text, cfg)

        if should_skip_page(text, cfg):
            continue

        cleaned.append(
            PageRecord(
                doc_id=page.doc_id,
                page_num=page.page_num,
                text=text,
                source_kind=page.source_kind,
                extra=page.extra,
            )
        )

    return cleaned
