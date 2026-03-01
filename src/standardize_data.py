import pandas as pd
import numpy as np
import os
import glob
import re
import urllib.parse
import random
from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def clean_val(v):
    if pd.isna(v) or str(v).lower() == 'nan': return ""
    return str(v).strip()

def load_txt_categories(filename, label, data_dir):
    from preprocessing import parse_http_string
    path = os.path.join(data_dir, filename)
    cats = []
    if os.path.exists(path):
        with open(path, 'r') as f:
            for line in f:
                match = re.match(r'^\d+\.\s+(.*?):\s+(.*)$', line.strip())
                if match:
                    row = parse_http_string(match.group(2))
                    row['label'] = label
                    cats.append(row)
    return pd.DataFrame(cats)

def process_all_data():
    data_dir = f"{PROJECT_ROOT}/data"
    
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
    attack_waf = pd.read_csv(os.path.join(data_dir, "attack.csv"), on_bad_lines='skip', low_memory=False)
    all_attacks_logs = pd.DataFrame([{'method': clean_val(r.get('http_method', 'GET')), 'path': clean_val(r.get('http_path', '/')), 'query': clean_val(r.get('http_query', '')), 'headers': clean_val(r.get('http_headers', '')), 'body': "", 'ua': ""} for idx, r in attack_waf.iterrows()])
    
    nm2 = pd.read_csv(os.path.join(data_dir, "nm2.xlsx.csv"), on_bad_lines='skip', low_memory=False)
    all_normals_logs = pd.DataFrame([{'method': clean_val(r.get('Method', 'GET')), 'path': clean_val(r.get('Path', '/')), 'query': clean_val(r.get('Query', '')), 'headers': clean_val(r.get('Headers', '')), 'body': clean_val(r.get('Body', '')), 'ua': ""} for idx, r in nm2.iterrows()])

    # 2. Golden Regression Injection
    print("Injecting golden regression samples...")
    from preprocessing import parse_http_string
    
    regression_attacks = []
    failed_patterns = [
        "id=1' OR '1'='1",
        "<script>alert('XSS')</script>",
        "cat /etc/passwd",
        "../../../../etc/passwd",
        '{"$gt": ""}',
        "onerror=alert(1)",
        "union select",
        "select from"
    ]
    for p in failed_patterns:
        row = parse_http_string(p)
        regression_attacks.append(row)
    
    attack_cats = load_txt_categories("attack.txt", 1, data_dir)
    normal_cats = load_txt_categories("normal.txt", 0, data_dir)

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
        "POST /login Content-Type: application/x-www-form-urlencoded user=john&pass=doe"
    ]
    norm_reg_rows = [parse_http_string(p) for p in normal_regression]
    for r in norm_reg_rows: r['label'] = 0

    # 4. Mirror Construction
    print("Constructing 500k-sample Production Pool...")
    
    # Normals: Mix of headers and no-headers
    normal_pool_logs = all_normals_logs.sample(min(len(all_normals_logs), 500000), random_state=42)
    # 50% keep headers, 50% empty headers (to be robust)
    half = len(normal_pool_logs) // 2
    normal_pool_logs.iloc[half:, normal_pool_logs.columns.get_loc('headers')] = ""
    
    # Diverse Root/Short Path Padding (Mix of headers/no-headers)
    short_paths = ["/", "/favicon.ico", "/index.html", "/robots.txt", "/api/health"]
    diverse_short = pd.DataFrame([{"path": random.choice(short_paths), "headers": ""} for _ in range(100000)])
    # half of short paths get headers
    h_short = len(diverse_short) // 2
    diverse_short.iloc[:h_short] = diverse_short.iloc[:h_short].apply(inject_metadata, axis=1)

    normal_pool = pd.concat([
        normal_pool_logs, 
        pd.concat([normal_cats] * 500), # Increased normal cat weighting
        pd.concat([pd.DataFrame(norm_reg_rows)] * 10000), # Extremely high frequency for regression
        diverse_short
    ], ignore_index=True)
    normal_pool['label'] = 0
    
    # Attacks: Inject metadata to some attacks so they don't look purely "headerless"
    attack_pool = pd.concat([
        pd.concat([all_attacks_logs] * 20), # Increased base pool
        pd.concat([attack_cats] * 200),
        pd.concat([pd.DataFrame(regression_attacks)] * 1000)
    ], ignore_index=True)
    h_att = len(attack_pool) // 2
    attack_pool.iloc[:h_att] = attack_pool.iloc[:h_att].apply(inject_metadata, axis=1)
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
