from __future__ import annotations

import re

from src.core.models import SectionBlock


def detect_heading_level(line: str, cfg: dict) -> str | None:
    if not cfg.get("heading_detection", {}).get("enabled", False):
        return None

    patterns = cfg["heading_detection"]["heading_patterns"]
    line = line.strip()

    for level in ("h1", "h2", "h3"):
        for pattern in patterns.get(level, []):
            if re.match(pattern, line):
                return level
    return None


def split_by_headings(text: str, cfg: dict, page_num: int = 0) -> list[SectionBlock]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return []

    blocks: list[SectionBlock] = []
    current_lines: list[str] = []
    current_h1 = ""
    current_h2 = ""
    current_h3 = ""

    def flush() -> None:
        if current_lines:
            blocks.append(
                SectionBlock(
                    text="\n".join(current_lines).strip(),
                    h1=current_h1,
                    h2=current_h2,
                    h3=current_h3,
                    page_num=page_num,
                )
            )

    for line in lines:
        level = detect_heading_level(line, cfg)
        if level:
            flush()
            current_lines = [line]
            if level == "h1":
                current_h1 = line
                current_h2 = ""
                current_h3 = ""
            elif level == "h2":
                current_h2 = line
                current_h3 = ""
            elif level == "h3":
                current_h3 = line
        else:
            current_lines.append(line)

    flush()
    return blocks
