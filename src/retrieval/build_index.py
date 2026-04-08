from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

from src.core.config_loader import load_configs
from src.core.utils import ensure_dir, logger, read_jsonl, write_jsonl


def resolve_chunks_file(cfg: dict[str, Any], explicit_chunks_file: str | None) -> Path:
    if explicit_chunks_file:
        return Path(explicit_chunks_file)
    return Path(cfg["paths"]["chunks_file"])


def resolve_index_dir(chunks_file: Path, explicit_index_dir: str | None) -> Path:
    if explicit_index_dir:
        return Path(explicit_index_dir)
    # 默认放到 workspace/index
    return chunks_file.parent.parent / "index"


def normalize_text(text: str) -> str:
    text = text or ""
    text = text.replace("\u3000", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def metadata_to_search_hints(metadata: dict[str, Any]) -> str:
    parts: list[str] = []

    simple_fields = [
        "doc_title",
        "doc_type",
        "doc_role",
        "section_h1",
        "section_h2",
        "section_h3",
        "event_type",
        "event_code",
        "event_name_zh",
        "param_id",
        "param_name_zh",
        "feature_group",
        "macro_type",
        "macro_id",
        "protocol",
        "controller",
        "function_code",
        "comm_topic",
    ]

    for field in simple_fields:
        value = metadata.get(field)
        if value:
            parts.append(str(value))

    keywords = metadata.get("keywords_zh", [])
    if isinstance(keywords, list):
        parts.extend(str(x) for x in keywords if x)

    return " ".join(parts).strip()


def build_search_text(record: dict[str, Any]) -> str:
    text = normalize_text(record.get("text", ""))
    metadata = record.get("metadata", {}) or {}
    hints = metadata_to_search_hints(metadata)

    # 重复关键标签一遍，增强 event_code / param_id / 协议等召回
    boosted_hints = []
    for key in ("event_code", "param_id", "macro_id", "protocol", "comm_topic"):
        value = metadata.get(key)
        if value:
            boosted_hints.append(str(value))
            boosted_hints.append(str(value))

    merged = "\n".join(part for part in [text, hints, " ".join(boosted_hints)] if part).strip()
    return normalize_text(merged)


def validate_chunk_record(record: dict[str, Any]) -> bool:
    if not isinstance(record, dict):
        return False
    if "chunk_id" not in record:
        return False
    if "text" not in record:
        return False
    if "metadata" not in record or not isinstance(record["metadata"], dict):
        return False
    return True


def load_chunk_records(chunks_file: Path) -> list[dict[str, Any]]:
    if not chunks_file.exists():
        raise FileNotFoundError(f"chunks.jsonl not found: {chunks_file}")

    records = read_jsonl(chunks_file)
    valid_records = [r for r in records if validate_chunk_record(r)]

    if not valid_records:
        raise ValueError(f"No valid chunk records found in: {chunks_file}")

    return valid_records


def enrich_records_for_index(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for record in records:
        enriched_record = {
            "chunk_id": record["chunk_id"],
            "text": normalize_text(record.get("text", "")),
            "metadata": record.get("metadata", {}),
            "search_text": build_search_text(record),
        }
        enriched.append(enriched_record)
    return enriched


def build_tfidf_vectorizer(
    max_features: int = 80000,
    min_df: int = 1,
    ngram_min: int = 2,
    ngram_max: int = 4,
) -> TfidfVectorizer:
    # 中文+英文+代码混合场景，char ngram 最稳，不依赖分词
    return TfidfVectorizer(
        analyzer="char",
        ngram_range=(ngram_min, ngram_max),
        lowercase=True,
        min_df=min_df,
        max_features=max_features,
        sublinear_tf=True,
    )


def save_index_artifacts(
    index_dir: Path,
    vectorizer: TfidfVectorizer,
    matrix: Any,
    enriched_records: list[dict[str, Any]],
    build_info: dict[str, Any],
) -> None:
    ensure_dir(index_dir)

    vectorizer_path = index_dir / "tfidf_vectorizer.joblib"
    matrix_path = index_dir / "tfidf_matrix.joblib"
    records_path = index_dir / "records.jsonl"
    build_info_path = index_dir / "build_info.json"

    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(matrix, matrix_path)
    write_jsonl(enriched_records, records_path)

    with build_info_path.open("w", encoding="utf-8") as f:
        json.dump(build_info, f, ensure_ascii=False, indent=2)


def build_index(
    config_path: str,
    chunks_file: str | None = None,
    index_dir: str | None = None,
    max_features: int = 80000,
    min_df: int = 1,
    ngram_min: int = 2,
    ngram_max: int = 4,
) -> Path:
    chunk_cfg, _manifest_cfg, _documents = load_configs(config_path)

    chunks_path = resolve_chunks_file(chunk_cfg, chunks_file)
    index_path = resolve_index_dir(chunks_path, index_dir)

    logger.info(f"Loading chunks from: {chunks_path}")
    records = load_chunk_records(chunks_path)
    enriched_records = enrich_records_for_index(records)

    corpus = [r["search_text"] for r in enriched_records]
    logger.info(f"Building TF-IDF index for {len(corpus)} chunks")

    vectorizer = build_tfidf_vectorizer(
        max_features=max_features,
        min_df=min_df,
        ngram_min=ngram_min,
        ngram_max=ngram_max,
    )
    matrix = vectorizer.fit_transform(corpus)

    build_info = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "chunks_file": str(chunks_path),
        "index_dir": str(index_path),
        "num_chunks": len(enriched_records),
        "num_features": int(len(vectorizer.vocabulary_)),
        "vectorizer": {
            "analyzer": "char",
            "ngram_range": [ngram_min, ngram_max],
            "max_features": max_features,
            "min_df": min_df,
            "sublinear_tf": True,
        },
    }

    save_index_artifacts(index_path, vectorizer, matrix, enriched_records, build_info)
    logger.info(f"Index build complete: {index_path}")
    logger.info(f"Chunks: {build_info['num_chunks']}, Features: {build_info['num_features']}")
    return index_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build local TF-IDF index from chunks.jsonl")
    parser.add_argument("--config", required=True, help="Path to chunking_config.yaml")
    parser.add_argument("--chunks-file", default=None, help="Optional override for chunks.jsonl")
    parser.add_argument("--index-dir", default=None, help="Optional override for output index dir")
    parser.add_argument("--max-features", type=int, default=80000, help="Max TF-IDF features")
    parser.add_argument("--min-df", type=int, default=1, help="Min document frequency")
    parser.add_argument("--ngram-min", type=int, default=2, help="Min char ngram")
    parser.add_argument("--ngram-max", type=int, default=4, help="Max char ngram")
    args = parser.parse_args()

    build_index(
        config_path=args.config,
        chunks_file=args.chunks_file,
        index_dir=args.index_dir,
        max_features=args.max_features,
        min_df=args.min_df,
        ngram_min=args.ngram_min,
        ngram_max=args.ngram_max,
    )


if __name__ == "__main__":
    main()
