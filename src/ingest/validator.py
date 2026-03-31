from __future__ import annotations

from src.core.models import ChunkRecord, ValidationIssue
from src.core.utils import logger


def validate_chunks(chunks: list[ChunkRecord], cfg: dict) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    required_fields = set(cfg["global"]["metadata"]["required_fields"])
    warn_short = int(cfg["validation"].get("warn_on_short_chunk_under_chars", 100))
    warn_long = int(cfg["validation"].get("warn_on_long_chunk_over_chars", 1400))

    for chunk in chunks:
        missing = [f for f in required_fields if f not in chunk.metadata]
        if missing:
            issues.append(
                ValidationIssue(
                    level="WARN",
                    chunk_id=chunk.chunk_id,
                    message=f"Missing metadata fields: {missing}",
                )
            )

        text_len = len(chunk.text)
        if text_len < warn_short:
            issues.append(
                ValidationIssue(
                    level="WARN",
                    chunk_id=chunk.chunk_id,
                    message=f"Chunk too short: {text_len}",
                )
            )

        if text_len > warn_long:
            issues.append(
                ValidationIssue(
                    level="WARN",
                    chunk_id=chunk.chunk_id,
                    message=f"Chunk too long: {text_len}",
                )
            )

    for issue in issues:
        logger.warning(f"{issue.chunk_id} - {issue.message}")

    return issues
