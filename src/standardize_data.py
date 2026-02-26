import pandas as pd
import os
import urllib.parse
import re
from preprocessing import clean_text
from sklearn.model_selection import train_test_split

import json

def map_waf_to_standard(row):
    """Maps WAF log columns to unified schema and sanitizes JSON headers."""
    raw_headers = str(row.get('http_headers', ''))
    headers_str = ""
    try:
        # WAF headers are JSON strings
        h_dict = json.loads(raw_headers)
        if isinstance(h_dict, dict):
            headers_str = "; ".join([f"{k}: {v}" for k, v in h_dict.items()])
        else:
            headers_str = raw_headers
    except:
        # Fallback: strip JSON structures if parsing fails
        headers_str = raw_headers.strip('{}').replace('""', '"')
        
    def clean_val(v):
        if pd.isna(v): return ""
        return str(v)

    return {
        'method': clean_val(row.get('http_method', '')),
        'path': clean_val(row.get('http_path', '')),
        'query': clean_val(row.get('http_query', '')),
        'headers': headers_str,
        'body': "",
        'ua': clean_val(row.get('http_user_agent', ''))
    }

def map_diverse_to_standard(row):
    """Maps diverse dataset columns to unified schema."""
    def clean_val(v):
        if pd.isna(v): return ""
        return str(v)

    url = clean_val(row.get('url', ''))
    headers = clean_val(row.get('headers', ''))
    
    try:
        parts = urllib.parse.urlparse(url)
        path = parts.path
        query = parts.query
    except:
        path = url
        query = ""
    
    match = re.search(r'User-Agent:\s*(.*)', headers, re.IGNORECASE)
    ua = match.group(1) if match else clean_val(row.get('user_agent', ''))
    
    return {
        'method': clean_val(row.get('method', '')),
        'path': path,
        'query': query,
        'headers': headers,
        'body': clean_val(row.get('body', '')),
        'ua': ua
    }

def process_all_data():
    data_dir = "/Users/dmac/Desktop/ml/data"
    processed_dir = os.path.join(data_dir, "processed")
    os.makedirs(processed_dir, exist_ok=True)

    print("Loading datasets...")
    # 1. WAF Attacks
    attack_waf_raw = pd.read_csv(os.path.join(data_dir, "attack.csv"), sep=';')
    attack_waf = pd.DataFrame([map_waf_to_standard(r) for _, r in attack_waf_raw.iterrows()])
    attack_waf['label'] = 1

    # 2. Diverse Set
    diverse_df_raw = pd.read_csv(os.path.join(data_dir, "http_dataset_20k_balanced.csv"))
    diverse_mapped = pd.DataFrame([map_diverse_to_standard(r) for _, r in diverse_df_raw.iterrows()])
    diverse_mapped['label'] = diverse_df_raw['label']
    
    diverse_attack = diverse_mapped[diverse_mapped['label'] == 1]
    diverse_normal = diverse_mapped[diverse_mapped['label'] == 0]
    
    # 3. Collect and Balance
    all_attacks = pd.concat([attack_waf, diverse_attack], ignore_index=True)
    num_attacks = len(all_attacks)
    print(f"Total attacks: {num_attacks}")

    # 4. Normals (Balanced Diversity to avoid Path Leakage)
    # We want normals to be DIVERSE, not just millions of ping logs.
    
    # Start with the full diverse normal set (approx 10k)
    all_normals = diverse_normal.copy()
    
    # Supplement with a SMALL, DIVERSE sample from WAF logs
    normal_waf_raw = pd.read_csv(os.path.join(data_dir, "nm2.xlsx.csv"), sep=';')
    # Filter for unique paths to avoid the 'ping' domination
    unique_waf_paths = normal_waf_raw.drop_duplicates(subset=['http_path'])
    waf_sample_size = min(2000, len(unique_waf_paths))
    normal_waf_sample = unique_waf_paths.sample(waf_sample_size, random_state=42)
    
    normal_waf_mapped = pd.DataFrame([map_waf_to_standard(r) for _, r in normal_waf_sample.iterrows()])
    normal_waf_mapped['label'] = 0
    
    all_normals = pd.concat([all_normals, normal_waf_mapped], ignore_index=True)
    all_normals['label'] = 0
    print(f"Total normals (diverse): {len(all_normals)}")

    # 5. Categorical Anchor Injection
    print("Injecting categorical anchors...")
    def load_txt_categories(filename, label):
        path = os.path.join(data_dir, filename)
        cats = []
        if os.path.exists(path):
            with open(path, 'r') as f:
                for line in f:
                    match = re.match(r'^\d+\.\s+(.*?):\s+(.*)$', line.strip())
                    if match:
                        payload = match.group(2)
                        # Minimal mapping to standard schema
                        cats.append({
                            "method": "GET", "path": "/", "query": "", 
                            "headers": payload, "body": "", "ua": "", "label": label
                        })
        return pd.DataFrame(cats)

    attack_cats = load_txt_categories("attack.txt", 1)
    normal_cats = load_txt_categories("normal.txt", 0)
    
    # 6. Massive Synthetic Normal Anchor (Homepage/Mozilla)
    synthetic_normals = pd.DataFrame([
        {"method": "GET", "path": "/", "query": "", "headers": "User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36", "body": "", "ua": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36", "label": 0},
        {"method": "GET", "path": "/favicon.ico", "query": "", "headers": "", "body": "", "ua": "", "label": 0}
    ] * 10000)
    
    # Merge all
    print("Merging datasets...")
    combined = pd.concat([
        all_attacks, 
        all_normals,
        synthetic_normals,
        pd.concat([attack_cats] * 1000), # Heavy anchor for attacks
        pd.concat([normal_cats] * 1000)   # Heavy anchor for normals
    ], ignore_index=True)
    
    print(f"Final dataset shape: {combined.shape}")
    print(f"Distribution: {combined['label'].value_counts().to_dict()}")

    train_df, temp_df = train_test_split(combined, test_size=0.2, stratify=combined['label'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)
    
    train_df.to_csv(os.path.join(processed_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(processed_dir, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(processed_dir, 'test.csv'), index=False)
    print("Standardized processed data saved.")

if __name__ == "__main__":
    process_all_data()
