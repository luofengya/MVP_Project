from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.core.config_loader import load_configs
from src.core.utils import logger, read_jsonl


def resolve_index_dir(config_path: str | None, explicit_index_dir: str | None) -> Path:
    if explicit_index_dir:
        return Path(explicit_index_dir)

    if not config_path:
        raise ValueError("Either --config or --index-dir must be provided")

    chunk_cfg, _manifest_cfg, _documents = load_configs(config_path)
    chunks_file = Path(chunk_cfg["paths"]["chunks_file"])
    return chunks_file.parent.parent / "index"


def load_index(index_dir: Path) -> tuple[Any, Any, list[dict[str, Any]], dict[str, Any]]:
    vectorizer_path = index_dir / "tfidf_vectorizer.joblib"
    matrix_path = index_dir / "tfidf_matrix.joblib"
    records_path = index_dir / "records.jsonl"
    build_info_path = index_dir / "build_info.json"

    if not vectorizer_path.exists():
        raise FileNotFoundError(f"Vectorizer not found: {vectorizer_path}")
    if not matrix_path.exists():
        raise FileNotFoundError(f"Matrix not found: {matrix_path}")
    if not records_path.exists():
        raise FileNotFoundError(f"records.jsonl not found: {records_path}")

    vectorizer = joblib.load(vectorizer_path)
    matrix = joblib.load(matrix_path)
    records = read_jsonl(records_path)

    build_info: dict[str, Any] = {}
    if build_info_path.exists():
        with build_info_path.open("r", encoding="utf-8") as f:
            build_info = json.load(f)

    return vectorizer, matrix, records, build_info


def normalize_text(text: str) -> str:
    text = text or ""
    text = text.replace("\u3000", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_query_entities(query: str) -> dict[str, str | None]:
    query = query or ""
    result: dict[str, str | None] = {
        "event_code": None,
        "param_id": None,
        "macro_id": None,
        "protocol": None,
        "controller": None,
    }

    event_match = re.search(r"\b([FA][0-9]{1,4})\b", query, re.IGNORECASE)
    if event_match:
        result["event_code"] = event_match.group(1).upper()

    param_match = re.search(r"\b([Ppr][0-9]{4})\b", query, re.IGNORECASE)
    if param_match:
        result["param_id"] = param_match.group(1).upper()

    macro_match = re.search(r"\b((?:Cn|Ap)[0-9]{3})\b", query, re.IGNORECASE)
    if macro_match:
        result["macro_id"] = macro_match.group(1)

    query_lower = query.lower()
    if "modbus" in query_lower:
        result["protocol"] = "MODBUS"
    elif "uss" in query_lower:
        result["protocol"] = "USS"

    if "s7-1200" in query_lower:
        result["controller"] = "S7-1200"
    elif "plc" in query_lower:
        result["controller"] = "PLC"

    return result


def build_query_text(query: str, entities: dict[str, str | None]) -> str:
    parts = [normalize_text(query)]

    # 给 TF-IDF 一个轻量增强，不是过滤，只是增加相关 ngram
    for key in ("event_code", "param_id", "macro_id", "protocol", "controller"):
        value = entities.get(key)
        if value:
            parts.append(value)
            parts.append(value)

    return " ".join(x for x in parts if x).strip()


def record_matches_filters(record: dict[str, Any], filters: dict[str, Any]) -> bool:
    metadata = record.get("metadata", {})

    for key, expected in filters.items():
        if expected in (None, ""):
            continue

        actual = metadata.get(key)

        if key == "protocol" and isinstance(actual, str) and isinstance(expected, str):
            if expected.lower() not in actual.lower():
                return False
        else:
            if actual != expected:
                return False

    return True


def score_with_boosts(
    base_score: float,
    record: dict[str, Any],
    query: str,
    entities: dict[str, str | None],
) -> float:
    metadata = record.get("metadata", {})
    boosted = float(base_score)
    query_lower = query.lower()

    # 精确实体匹配加分
    if entities.get("event_code") and metadata.get("event_code") == entities["event_code"]:
        boosted += 0.30

    if entities.get("param_id") and metadata.get("param_id") == entities["param_id"]:
        boosted += 0.30

    if entities.get("macro_id") and metadata.get("macro_id") == entities["macro_id"]:
        boosted += 0.20

    if entities.get("protocol") and isinstance(metadata.get("protocol"), str):
        if entities["protocol"].lower() in metadata["protocol"].lower():
            boosted += 0.15

    if entities.get("controller") and isinstance(metadata.get("controller"), str):
        if entities["controller"].lower() in metadata["controller"].lower():
            boosted += 0.10

    # 文档角色轻量加权
    doc_type = metadata.get("doc_type", "")
    if entities.get("event_code") and doc_type == "fault_page":
        boosted += 0.08
    if entities.get("param_id") and doc_type in {"operating_manual", "compact_manual"}:
        boosted += 0.05
    if "modbus" in query_lower and doc_type == "comm_page":
        boosted += 0.08

    # retrieval_boost
    try:
        boosted += float(metadata.get("retrieval_boost", 1.0)) * 0.02
    except Exception:
        pass

    return boosted


def format_result(record: dict[str, Any], score: float, rank: int) -> dict[str, Any]:
    metadata = record.get("metadata", {})
    section_parts = [
        metadata.get("section_h1", ""),
        metadata.get("section_h2", ""),
        metadata.get("section_h3", ""),
    ]
    section_parts = [x for x in section_parts if x]

    return {
        "rank": rank,
        "score": round(float(score), 6),
        "chunk_id": record.get("chunk_id", ""),
        "text": record.get("text", ""),
        "metadata": metadata,
        "citation": {
            "doc_title": metadata.get("doc_title", ""),
            "doc_type": metadata.get("doc_type", ""),
            "page_start": metadata.get("page_start"),
            "page_end": metadata.get("page_end"),
            "section": " > ".join(section_parts),
        },
    }


def search_index(
    query: str,
    index_dir: Path,
    top_k: int = 5,
    filters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not query.strip():
        raise ValueError("Query cannot be empty")

    vectorizer, matrix, records, build_info = load_index(index_dir)
    filters = filters or {}

    entities = extract_query_entities(query)
    query_text = build_query_text(query, entities)
    query_vec = vectorizer.transform([query_text])

    candidate_indices = [
        idx for idx, record in enumerate(records)
        if record_matches_filters(record, filters)
    ]

    if not candidate_indices:
        return {
            "query": query,
            "index_dir": str(index_dir),
            "build_info": build_info,
            "results": [],
            "message": "No candidates matched the filters.",
        }

    candidate_matrix = matrix[candidate_indices]
    scores = cosine_similarity(query_vec, candidate_matrix).ravel()

    ranked: list[tuple[int, float]] = []
    for local_idx, base_score in enumerate(scores):
        global_idx = candidate_indices[local_idx]
        record = records[global_idx]
        final_score = score_with_boosts(base_score, record, query, entities)
        ranked.append((global_idx, final_score))

    ranked.sort(key=lambda x: x[1], reverse=True)
    ranked = ranked[:top_k]

    results = [
        format_result(records[global_idx], score=score, rank=rank)
        for rank, (global_idx, score) in enumerate(ranked, start=1)
    ]

    return {
        "query": query,
        "entities": entities,
        "filters": filters,
        "index_dir": str(index_dir),
        "build_info": build_info,
        "results": results,
    }


def print_human_readable(search_output: dict[str, Any]) -> None:
    print("=" * 80)
    print(f"Query: {search_output['query']}")
    if search_output.get("entities"):
        print(f"Parsed entities: {search_output['entities']}")
    if search_output.get("filters"):
        print(f"Filters: {search_output['filters']}")
    print("-" * 80)

    results = search_output.get("results", [])
    if not results:
        print(search_output.get("message", "No results."))
        print("=" * 80)
        return

    for item in results:
        citation = item["citation"]
        text_preview = item["text"][:260].replace("\n", " ")
        print(f"[{item['rank']}] score={item['score']}")
        print(f"doc: {citation['doc_title']}")
        print(f"type: {citation['doc_type']}")
        print(f"page: {citation['page_start']} - {citation['page_end']}")
        if citation["section"]:
            print(f"section: {citation['section']}")
        print(f"chunk_id: {item['chunk_id']}")
        print(f"text: {text_preview}")
        print("-" * 80)

    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description="Search local TF-IDF index")
    parser.add_argument("--query", required=True, help="Search query")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results")
    parser.add_argument("--config", default=None, help="Path to chunking_config.yaml")
    parser.add_argument("--index-dir", default=None, help="Path to built index directory")
    parser.add_argument("--doc-type", default=None, help="Exact metadata filter: doc_type")
    parser.add_argument("--doc-role", default=None, help="Exact metadata filter: doc_role")
    parser.add_argument("--event-code", default=None, help="Exact metadata filter: event_code")
    parser.add_argument("--param-id", default=None, help="Exact metadata filter: param_id")
    parser.add_argument("--protocol", default=None, help="Loose metadata filter: protocol")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of readable text")
    args = parser.parse_args()

    index_dir = resolve_index_dir(config_path=args.config, explicit_index_dir=args.index_dir)

    filters = {
        "doc_type": args.doc_type,
        "doc_role": args.doc_role,
        "event_code": args.event_code,
        "param_id": args.param_id,
        "protocol": args.protocol,
    }

    logger.info(f"Searching index: {index_dir}")
    output = search_index(
        query=args.query,
        index_dir=index_dir,
        top_k=args.top_k,
        filters=filters,
    )

    if args.json:
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        print_human_readable(output)


if __name__ == "__main__":
    main()
