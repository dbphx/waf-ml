"""Parse data/attack.txt and data/normal.txt lines (shared by training scripts)."""
import re


def parse_category_lines(filepath: str):
    categories = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            match = re.match(r"^\d+\.\s+(.*?):\s+(.*)$", line.strip())
            if match:
                categories.append(
                    {"category": match.group(1).strip(), "payload": match.group(2).strip()}
                )
    return categories
