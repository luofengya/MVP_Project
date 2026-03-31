from __future__ import annotations

import re
from typing import Any


def _extract_first_match(text: str, patterns: list[str]) -> str | None:
    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return m.group(1) if m.groups() else m.group(0)
    return None


def extract_event_code(text: str, cfg: dict) -> str | None:
    patterns = cfg["field_extractors"]["event_code"]["patterns"]
    return _extract_first_match(text, patterns)


def extract_param_id(text: str, cfg: dict) -> str | None:
    patterns = cfg["field_extractors"]["param_id"]["patterns"]
    value = _extract_first_match(text, patterns)
    return value.upper() if value else None


def extract_macro_id(text: str, cfg: dict) -> str | None:
    patterns = cfg["field_extractors"]["macro_id"]["patterns"]
    return _extract_first_match(text, patterns)


def extract_protocol(text: str, cfg: dict) -> str | None:
    patterns = cfg["field_extractors"]["protocol"]["patterns"]
    return _extract_first_match(text, patterns)


def extract_controller(text: str, cfg: dict) -> str | None:
    patterns = cfg["field_extractors"]["controller"]["patterns"]
    return _extract_first_match(text, patterns)


def extract_function_code(text: str, cfg: dict) -> str | None:
    patterns = cfg["field_extractors"]["function_code"]["patterns"]
    return _extract_first_match(text, patterns)


def infer_event_type(event_code: str | None, cfg: dict) -> str | None:
    if not event_code:
        return None
    rules = cfg["field_extractors"]["event_type_rules"]
    if event_code.startswith(rules.get("fault_prefix", "F")):
        return "fault"
    if event_code.startswith(rules.get("alarm_prefix", "A")):
        return "alarm"
    return None


def infer_feature_group(text: str, cfg: dict) -> str | None:
    rules = cfg["field_extractors"].get("feature_group_rules", {})
    text_lower = text.lower()
    for group_name, keywords in rules.items():
        for kw in keywords:
            if kw.lower() in text_lower:
                return group_name
    return None


def enrich_keywords(text: str, cfg: dict) -> list[str]:
    if not cfg.get("keyword_enrichment", {}).get("enabled", False):
        return []

    dictionaries = cfg["keyword_enrichment"].get("dictionaries", {})
    found: list[str] = []
    text_lower = text.lower()

    for _, keywords in dictionaries.items():
        for kw in keywords:
            if kw.lower() in text_lower and kw not in found:
                found.append(kw)

    max_keywords = int(cfg["keyword_enrichment"].get("max_keywords", 8))
    return found[:max_keywords]


def enrich_structured_fields(text: str, metadata: dict[str, Any], cfg: dict) -> dict[str, Any]:
    event_code = extract_event_code(text, cfg)
    if event_code:
        metadata["event_code"] = event_code
        metadata["event_type"] = infer_event_type(event_code, cfg)

    param_id = extract_param_id(text, cfg)
    if param_id:
        metadata["param_id"] = param_id

    macro_id = extract_macro_id(text, cfg)
    if macro_id:
        metadata["macro_id"] = macro_id
        metadata["macro_type"] = "connection" if macro_id.startswith("Cn") else "application"

    protocol = extract_protocol(text, cfg)
    if protocol:
        metadata["protocol"] = protocol

    controller = extract_controller(text, cfg)
    if controller:
        metadata["controller"] = controller

    function_code = extract_function_code(text, cfg)
    if function_code:
        metadata["function_code"] = function_code

    feature_group = infer_feature_group(text, cfg)
    if feature_group:
        metadata["feature_group"] = feature_group

    keywords = enrich_keywords(text, cfg)
    if keywords:
        existing = metadata.get("keywords_zh", [])
        metadata["keywords_zh"] = list(dict.fromkeys(existing + keywords))

    return metadata
