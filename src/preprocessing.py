import json
import pandas as pd
import numpy as np
import re
import urllib.parse
from sklearn.model_selection import train_test_split
import os


REQUEST_FIELDS = ("path", "query", "headers", "body")


def split_request_components(http_data):
    """Normalize raw input into request fields used by the models."""
    request = {
        "method": "GET",
        "url": "",
        "path": "",
        "query": "",
        "headers": "",
        "body": "",
        "user_agent": "",
    }

    if isinstance(http_data, dict):
        request["method"] = str(http_data.get("method", "GET")).upper()
        request["url"] = str(http_data.get("url", ""))
        request["path"] = str(http_data.get("path", "") or "")
        request["query"] = str(http_data.get("query", ""))
        request["headers"] = str(http_data.get("headers", ""))
        request["body"] = str(http_data.get("body", ""))
        request["user_agent"] = str(http_data.get("user_agent", http_data.get("ua", "")))

        if request["url"] and not (http_data.get("path") or http_data.get("query")):
            try:
                parts = urllib.parse.urlparse(request["url"])
                path = parts.path
                if parts.params:
                    path = f"{path};{parts.params}" if path else parts.params
                query = parts.query
                if parts.fragment:
                    query = f"{query}#{parts.fragment}" if query else parts.fragment
                request["path"] = path or ""
                request["query"] = query
            except Exception:
                request["path"] = request["url"] or ""

        if not request["url"]:
            request["url"] = request["path"]
            if request["query"]:
                request["url"] = f"{request['path']}?{request['query']}"

        if not request["user_agent"] and request["headers"]:
            ua_match = re.search(r"(?im)^User-Agent:\s*(.+)$", request["headers"])
            if ua_match:
                request["user_agent"] = ua_match.group(1).strip()

        return request

    payload = str(http_data)
    request["url"] = payload
    first_line = payload.strip().splitlines()[0] if payload.strip() else ""
    method_match = re.match(r"^(GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS)\s+(\S+)", first_line, re.IGNORECASE)

    if method_match:
        request["method"] = method_match.group(1).upper()
        request["url"] = method_match.group(2)
        remainder = payload[method_match.end():].lstrip()
        if remainder:
            parts = re.split(r"\r\n\r\n|\n\n", remainder, maxsplit=1)
            request["headers"] = parts[0].strip()
            request["body"] = parts[1].strip() if len(parts) > 1 else ""
            ua_match = re.search(r"(?im)^User-Agent:\s*(.+)$", request["headers"])
            if ua_match:
                request["user_agent"] = ua_match.group(1).strip()

        try:
            parts = urllib.parse.urlparse(request["url"])
            request["path"] = parts.path or "/"
            request["query"] = parts.query
        except Exception:
            request["path"] = request["url"] or "/"

        return request

    parsed = parse_http_string(payload)
    request["method"] = str(parsed.get("method", "GET")).upper()
    request["path"] = str(parsed.get("path", "") or "")
    request["query"] = str(parsed.get("query", ""))
    request["headers"] = str(parsed.get("headers", ""))
    request["body"] = str(parsed.get("body", ""))
    request["url"] = request["path"]
    if request["query"]:
        request["url"] = f"{request['path']}?{request['query']}"
    request["user_agent"] = str(parsed.get("ua", parsed.get("user_agent", "")))
    return request


def request_row_from_components(data):
    request = split_request_components(data)
    return {
        "method": request["method"],
        "path": request["path"],
        "query": request["query"],
        "headers": request["headers"],
        "body": request["body"],
        "ua": request["user_agent"],
    }


def parse_category_lines(filepath):
    categories = []
    with open(filepath, "r", encoding="utf-8") as file_obj:
        for raw_line in file_obj:
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split(":", 1)
            if len(parts) != 2:
                continue

            cat_match = re.match(r"^\d+\.\s+(.*)$", parts[0].strip())
            if not cat_match:
                continue

            payload = parts[1].strip()
            categories.append(
                {
                    "category": cat_match.group(1).strip(),
                    "payload": payload,
                    "request": parse_http_string(payload),
                }
            )
    return categories


def encode_request_components(row):
    encoded = dict(row)
    for field in REQUEST_FIELDS:
        value = str(encoded.get(field, ""))
        if value:
            encoded[field] = urllib.parse.quote(value, safe="")
    return encoded


def expand_request_to_field_rows(row, include_combined=False):
    rows = []
    if include_combined:
        rows.append(dict(row))

    for field in REQUEST_FIELDS:
        value = str(row.get(field, "")).strip()
        if not value or value.lower() == "nan":
            continue
        field_row = {
            "method": row.get("method", "GET"),
            "path": "",
            "query": "",
            "headers": "",
            "body": "",
            "ua": row.get("ua", row.get("user_agent", "")),
            "field": field,
        }
        for key in ("label", "weight", "category", "source_payload"):
            if key in row:
                field_row[key] = row[key]
        field_row[field] = value
        rows.append(field_row)

    if not rows:
        fallback = {
            "method": row.get("method", "GET"),
            "path": row.get("path", "/") or "/",
            "query": "",
            "headers": "",
            "body": "",
            "ua": row.get("ua", row.get("user_agent", "")),
            "field": "path",
        }
        for key in ("label", "weight", "category", "source_payload"):
            if key in row:
                fallback[key] = row[key]
        rows.append(fallback)

    return rows

def parse_http_string(payload):
    """Shared logic to decompose raw HTTP snippets into standard fields."""
    row = {"method": "GET", "path": "", "query": "", "headers": "", "body": "", "ua": ""}
    if not payload: return row
    payload = str(payload).strip()

    if payload.startswith('{'):
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            data = None

        if isinstance(data, dict) and any(key in data for key in ("method", "path", "query", "headers", "body", "url")):
            return request_row_from_components(data)
    
    # 1. Handle Method/Path pattern (e.g., 'GET /path?q=v {"body": 1}')
    if payload.startswith(('GET ', 'POST ', 'PUT ', 'DELETE ', 'PATCH ', 'HEAD ', 'OPTIONS ')):
        parts = payload.split(' ', 2)
        row['method'] = parts[0]
        if len(parts) > 1:
            url_part = parts[1]
            try:
                p = urllib.parse.urlparse(url_part)
                row['path'] = p.path or ""
                row['query'] = p.query
            except:
                row['path'] = url_part
        if len(parts) > 2:
            row['body'] = parts[2]
            if ':' in row['body'] and not row['body'].startswith(('{', '[')):
                row['headers'] = row['body']
                row['body'] = ""
        if row['headers']:
            ua_match = re.search(r'(?im)^User-Agent:\s*(.+)$', row['headers'])
            if ua_match:
                row['ua'] = ua_match.group(1).strip()

    # 2. Handle Body pattern (e.g., '{"key": "val"}')
    elif payload.startswith(('{', '[')):
        row['method'] = "POST"
        row['body'] = payload

    # 3. Handle header-only pattern
    elif re.match(r'^[A-Za-z0-9-]+:\s*.+$', payload):
        row['headers'] = payload
        ua_match = re.search(r'(?im)^User-Agent:\s*(.+)$', row['headers'])
        if ua_match:
            row['ua'] = ua_match.group(1).strip()

    # 4. Path (+ optional query), e.g. '/api/foo/?all=true' — must run before bare query heuristic
    elif payload.startswith('/'):
        parsed = urllib.parse.urlparse(payload)
        row['path'] = parsed.path or '/'
        row['query'] = parsed.query

    # 5. Bare query string (e.g., 'id=1&name=test') — no leading slash
    elif any(sep in payload for sep in ['=', '&']) and ' ' not in payload:
        row['query'] = payload

    # 6. Fallback: Entire string as payload-carrying path/body
    else:
        if payload.startswith(('<', '(', ';', '|', '$', '`')) or '://' in payload:
            row['method'] = 'POST'
            row['body'] = payload
        else:
            row['path'] = payload
        
    return row

def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    # 1. Lowercase all text
    text = text.lower()
    
    # 2. URL decode (2 passes to handle double encoding)
    try:
        text = urllib.parse.unquote(text)
        text = urllib.parse.unquote(text)
    except:
        pass
    
    # 3. Normalize whitespace but keep everything else
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def preprocess_data(input_path, output_dir):
    """Legacy preprocessing function - replaced by standardize_data.py but kept for compatibility."""
    df = pd.read_csv(input_path)
    df['cleaned_text'] = df.apply(lambda x: clean_text(str(x)), axis=1)
    # This is a stub for the old pipeline
    pass

if __name__ == "__main__":
    pass
