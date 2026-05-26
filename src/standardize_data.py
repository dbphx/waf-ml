import glob
import os
import random
import re

import pandas as pd
from sklearn.model_selection import train_test_split

from parse_category_files import generate_field_level_category_files
from preprocessing import expand_request_to_field_rows, parse_category_lines, parse_http_string

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def clean_val(v):
    if pd.isna(v) or str(v).lower() == 'nan': return ""
    return str(v).strip()


def load_csv_with_detected_delimiter(path):
    with open(path, "r", encoding="utf-8", newline="") as file_obj:
        header_line = file_obj.readline()

    delimiter = ";" if header_line.count(";") > header_line.count(",") else ","
    return pd.read_csv(path, sep=delimiter, on_bad_lines='skip', low_memory=False)


def extract_http_rows(df):
    return pd.DataFrame([
        {
            'method': clean_val(r.get('http_method', r.get('Method', 'GET'))),
            'path': clean_val(r.get('http_path', r.get('Path', '/'))),
            'query': clean_val(r.get('http_query', r.get('Query', ''))),
            'headers': clean_val(r.get('http_headers', r.get('Headers', ''))),
            'body': clean_val(r.get('body', r.get('Body', ''))),
            'ua': clean_val(r.get('http_user_agent', r.get('User-Agent', ''))),
        }
        for _, r in df.iterrows()
    ])

def load_txt_categories(filename, label, data_dir):
    path = os.path.join(data_dir, filename)
    cats = []
    if os.path.exists(path):
        for sample in parse_category_lines(path):
            row = dict(sample['request'])
            row['label'] = label
            row['category'] = sample['category']
            cats.append(row)
    return pd.DataFrame(cats)


def expand_dataset_to_field_rows(df):
    rows = []
    for row in df.to_dict(orient='records'):
        rows.extend(expand_request_to_field_rows(row, include_combined=False))
    return pd.DataFrame(rows)

def process_all_data():
    data_dir = f"{PROJECT_ROOT}/data"
    generated = generate_field_level_category_files(data_dir)
    for dest_name, count in generated.items():
        print(f"Generated {count} field-level samples in {dest_name}.")
    
    # Metadata pools for realistic normalcy
    ua_pool = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Mobile/15E148 Safari/604.1"
    ]
    host_pool = ["example.com", "api.internal", "app.local", "www.mysite.com"]

    def inject_metadata(row):
        h = row.get('headers', '')
        if not h or h.lower() == 'nan':
            row['headers'] = f"User-Agent: {random.choice(ua_pool)}\r\nHost: {random.choice(host_pool)}\r\nAccept: */*"
        return row

    # 1. Load Data
    attack_paths = sorted(glob.glob(os.path.join(data_dir, "attack*.csv")))
    attack_frames = [expand_dataset_to_field_rows(extract_http_rows(load_csv_with_detected_delimiter(path))) for path in attack_paths]
    all_attacks_logs = pd.concat(attack_frames, ignore_index=True).drop_duplicates().reset_index(drop=True)
    print(f"Loaded {len(all_attacks_logs)} unique attack rows from {len(attack_paths)} file(s).")

    nm2 = load_csv_with_detected_delimiter(os.path.join(data_dir, "nm2.xlsx.csv"))
    all_normals_logs = expand_dataset_to_field_rows(extract_http_rows(nm2)).drop_duplicates().reset_index(drop=True)
    print(f"Loaded {len(all_normals_logs)} unique normal rows.")

    # 2. Golden Regression Injection
    print("Injecting golden regression samples...")
    regression_attacks = []
    failed_patterns = [
        "id=1' OR '1'='1",
        "<script>alert('XSS')</script>",
        "cat /etc/passwd",
        "../../../../etc/passwd",
        '{"$gt": ""}',
        "onerror=alert(1)",
        "union select",
        "select from",
        "GET /assets/octagon-alert-7zrgUous.js?q=%27or%201=1;-- HTTP/1.1",
        "GET /assets/octagon-alert-7zrgUous.js?q=%3Cscript%3E%3C/script%3E HTTP/1.1",
        "GET /static/js/select.min.js?id=1%20UNION%20SELECT%201,2,3 HTTP/1.1",
        "GET /assets/script-tag-2befZuvx.js?file=../../../../etc/passwd HTTP/1.1"
    ]
    for p in failed_patterns:
        row = parse_http_string(p)
        regression_attacks.extend(expand_request_to_field_rows(row, include_combined=False))
    
    attack_cats = load_txt_categories("attack_fields.txt", 1, data_dir)
    normal_cats = load_txt_categories("normal_fields.txt", 0, data_dir)

    # 3. Normal Regression Injection (Fixing JWT False Positives)
    print("Injecting normal regression samples...")
    normal_regression = [
        "GET /id?token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjE3NzIzNTEwMzcsImlhdCI6MTc3MjA5MTgzNywidXNlciI6eyJzZXNzaW9uX2lkIjoiMDE5Yzk4ZTctOGFjMi03Mjc0LWFkM2EtMmRiNzFhZTYzZThlIiwic291cmNlIjoiQk8iLCJzdGFmZl9jb2RlIjoibGluaGxoMiIsInVzZXJfZW1haWwiOiJsaW5obGgyQHZucGF5LnZuIiwidXNlcl9mdWxsX25hbWUiOiJMaW5oIEzDqiBIw",
        "GET /?token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjE3NzIxODE2MDMsImlhdCI6MTc3MTkyMjQwMywidXNlciI6eyJncm91cF9pZCI6IjAxOWMwZTZlLWI0YTctNzRjYS05MGNlLTUyY2M0Njg5MzZjZSIsImxlYWRlcl9pZCI6IjAxOWMwOWU0LTNmY2QtNzc2ZS05ZGUwLWFiZGYyZjI3ZTYwMyIsInNlc3Npb25faWQiOiIwMTljOGVjZS0zMTkyLTc4MjQtYTI1OS0zZDZmM2JhM2NkM",
        "GET /admin/users?limit=20&offset=40",
        'POST /user/password-update {"old": "pass1", "new": "pass2"}',
        "GET /products?category=electronics&brand=apple",
        'POST /logs/client {"error": "Uncaught TypeError", "stack": "..."}',
        "GET / User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
                "POST /login Content-Type: application/x-www-form-urlencoded user=john&pass=doe",
        "GET /assets/octagon-alert-7zrgUous.js HTTP/1.1",
        "GET /assets/octagon-alert-7zrgUous.css HTTP/1.1",
        "GET /assets/script-tag-2befZuvx.js HTTP/1.1",
        "GET /static/js/select.min.js HTTP/1.1",
        "GET /static/js/alert.min.js HTTP/1.1",
                "GET /js/components/AlertDialog.js HTTP/1.1",
        "GET /search?q=I%20select%20like%20you%20union HTTP/1.1",
        "GET /api/chat?message=can%20you%20select%20the%20best%20union%20for%20me HTTP/1.1",
        "GET /forum/post?title=why%20I%20drop%20out%20of%20the%20student%20union HTTP/1.1",
        "GET /query?text=please%20insert%20the%20coin%20and%20select%20your%20drink HTTP/1.1",
        "GET /help?q=how%20to%20update%20my%20profile%20and%20delete%20old%20photos HTTP/1.1",
        "GET /api/workspaces/insky/projects/bce0a79c-90d2-4558-9084-945ad6acbdae/issues/ HTTP/1.1",
        "PATCH /api/workspaces/insky/projects/bce0a79c-90d2-4558-9084-945ad6acbdae/issues/9906eeae-3678-40e2-9869-64bc8b84c7c5/ HTTP/1.1",
        "PATCH /api/workspaces/demo/projects/11111111-2222-3333-4444-555555555555/issues/aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee/ HTTP/1.1"
    ]
    norm_reg_rows = []
    for payload in normal_regression:
        row = parse_http_string(payload)
        row['label'] = 0
        norm_reg_rows.extend(expand_request_to_field_rows(row, include_combined=False))

    # 4. Mirror Construction
    print("Constructing reduced production mirror pool...")
    
    # Normals: Mix of headers and no-headers
    # Keep the mirror mix, but cap the pool so retraining stays practical.
    normal_pool_logs = all_normals_logs.sample(min(len(all_normals_logs), 100000), random_state=42)
    # 50% keep headers, 50% empty headers (to be robust)
    half = len(normal_pool_logs) // 2
    normal_pool_logs.loc[normal_pool_logs.index[half:], 'headers'] = ""
    
    # Diverse Root/Short Path Padding (Mix of headers/no-headers)
    short_paths = ["/", "/favicon.ico", "/index.html", "/robots.txt", "/api/health"]
    diverse_short = pd.DataFrame([{"path": random.choice(short_paths), "headers": ""} for _ in range(20000)])
    # half of short paths get headers
    h_short = len(diverse_short) // 2
    diverse_short_with_headers = diverse_short.iloc[:h_short].apply(inject_metadata, axis=1)
    diverse_short.loc[diverse_short.index[:h_short], 'headers'] = diverse_short_with_headers['headers'].astype(str).to_list()

    normal_pool = pd.concat([
        normal_pool_logs, 
        pd.concat([normal_cats] * 100), # Keep regression normals prominent without dominating the pool
        pd.concat([pd.DataFrame(norm_reg_rows)] * 1000), # Preserve false-positive protection at a smaller scale
        diverse_short
    ], ignore_index=True)
    normal_pool['label'] = 0
    
    # Attacks: Inject metadata to some attacks so they don't look purely "headerless"
    attack_pool = pd.concat([
        pd.concat([all_attacks_logs] * 2), # Keep broad attack coverage with a manageable retrain size
        pd.concat([attack_cats] * 50),
        pd.concat([pd.DataFrame(regression_attacks)] * 200)
    ], ignore_index=True)
    h_att = len(attack_pool) // 2
    attack_pool_with_headers = attack_pool.iloc[:h_att].apply(inject_metadata, axis=1)
    attack_pool.loc[attack_pool.index[:h_att], 'headers'] = attack_pool_with_headers['headers'].astype(str).to_list()
    attack_pool['label'] = 1
    
    n_samples = len(attack_pool) 
    print(f"Sampling {n_samples} for mirror training.")
    final_attacks = attack_pool
    final_normals = normal_pool.sample(min(len(normal_pool), n_samples), random_state=42)
    combined = pd.concat([final_attacks, final_normals], ignore_index=True)
    
    # 4. Save
    train_df, val_df = train_test_split(combined, test_size=0.1, random_state=42, stratify=combined['label'])
    processed_dir = os.path.join(data_dir, "processed")
    os.makedirs(processed_dir, exist_ok=True)
    train_df.to_csv(os.path.join(processed_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(processed_dir, "val.csv"), index=False)
    print("Standardized processed data (Mirror Split) saved.")

if __name__ == "__main__":
    process_all_data()
