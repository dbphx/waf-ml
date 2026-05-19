"""Parse category files and rewrite them as field-level component specs."""

import json
import os

from preprocessing import expand_request_to_field_rows, parse_category_lines


CATEGORY_FIELD_FILE_MAP = {
    "attack.txt": "attack_fields.txt",
    "normal.txt": "normal_fields.txt",
}

def write_field_level_category_file(source_path: str, dest_path: str):
    rows = []
    for sample in parse_category_lines(source_path):
        request = dict(sample["request"])
        request["category"] = sample["category"]
        request["source_payload"] = sample.get("payload", "")
        field_rows = expand_request_to_field_rows(request, include_combined=False)
        for field_row in field_rows:
            rows.append({
                "category": f"{sample['category']} [{field_row['field']}]",
                "field": field_row.get("field", ""),
                "method": field_row.get("method", "GET"),
                "path": field_row.get("path", ""),
                "query": field_row.get("query", ""),
                "headers": field_row.get("headers", ""),
                "body": field_row.get("body", ""),
                "source_payload": sample.get("payload", ""),
            })

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with open(dest_path, "w", encoding="utf-8") as f:
        for index, row in enumerate(rows, start=1):
            payload = json.dumps(row, ensure_ascii=False, separators=(",", ":"))
            f.write(f"{index}. {row['category']}: {payload}\n")

    return len(rows)


def generate_field_level_category_files(data_dir: str):
    generated_counts = {}
    for source_name, dest_name in CATEGORY_FIELD_FILE_MAP.items():
        generated_counts[dest_name] = write_field_level_category_file(
            os.path.join(data_dir, source_name),
            os.path.join(data_dir, dest_name),
        )
    return generated_counts


if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(project_root, "data")
    for dest_name, count in generate_field_level_category_files(data_dir).items():
        print(f"Generated {count} field-level samples in {dest_name}")
