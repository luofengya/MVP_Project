from __future__ import annotations

from difflib import SequenceMatcher

from src.core.models import ChunkRecord
from src.core.utils import normalize_for_hash, sha1_text


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def deduplicate_chunks(chunks: list[ChunkRecord], cfg: dict) -> list[ChunkRecord]:
    dedup_cfg = cfg.get("deduplication", {})
    if not dedup_cfg.get("enabled", False):
        return chunks

    threshold = float(dedup_cfg.get("similarity_threshold", 0.96))
    seen_hashes: set[str] = set()
    deduped: list[ChunkRecord] = []

    for chunk in chunks:
        normalized = normalize_for_hash(chunk.text)
        text_hash = sha1_text(normalized)

        if text_hash in seen_hashes:
            continue

        duplicate_found = False
        for existing in deduped:
            sim = _similarity(normalized, normalize_for_hash(existing.text))
            if sim >= threshold:
                duplicate_found = True
                break

        if not duplicate_found:
            seen_hashes.add(text_hash)
            deduped.append(chunk)

    return deduped
