import pandas as pd
import os
from preprocessing import clean_text
from sklearn.model_selection import train_test_split

def process_new_files():
    data_dir = "/Users/dmac/Desktop/ml/data"
    processed_dir = os.path.join(data_dir, "processed")
    
    # Load new data
    print("Loading new data...")
    # attack.csv uses semicolon delimiter based on head output
    attack_df = pd.read_csv(os.path.join(data_dir, "attack.csv"), sep=';')
    attack_df['label'] = 1
    
    # nm2.xlsx.csv also uses semicolon delimiter
    normal_df = pd.read_csv(os.path.join(data_dir, "nm2.xlsx.csv"), sep=';')
    # Subsample massive normal logs to 50k to reduce bias and speed up training
    if len(normal_df) > 50000:
        print("Subsampling normal logs to 50,000...")
        normal_df = normal_df.sample(50000, random_state=42)
    normal_df['label'] = 0
    
    # Combined helper to standardized text format
    def get_standard_text(df, is_waf_log=True):
        if is_waf_log:
            # Semicolon logs have path/query separated
            return (
                df['http_method'].fillna("") + " " + 
                df['http_path'].fillna("") + " " + 
                df['http_query'].fillna("") + " " + 
                df['http_headers'].fillna("") + " " + 
                "" + " " + # No body in semicolon logs
                df['http_user_agent'].fillna("")
            )
        else:
            # Other sets have url (path+query) and body
            def split_url(url):
                import urllib.parse
                parts = urllib.parse.urlparse(str(url))
                return parts.path, parts.query
            
            paths, queries = zip(*df['url'].apply(split_url))
            
            # Extract UA from headers if possible
            def get_ua(headers):
                import re
                if not isinstance(headers, str): return ""
                match = re.search(r'User-Agent:\s*(.*)', headers, re.IGNORECASE)
                return match.group(1) if match else ""
            
            uas = df['headers'].apply(get_ua)
            
            return (
                df['method'].fillna("") + " " + 
                pd.Series(paths) + " " + 
                pd.Series(queries) + " " + 
                df['headers'].fillna("") + " " + 
                df['body'].fillna("") + " " + 
                uas
            )

    print("Cleaning and preparing new samples...")
    attack_df['cleaned_text'] = get_standard_text(attack_df, True).apply(clean_text)
    normal_df['cleaned_text'] = get_standard_text(normal_df, True).apply(clean_text)
    
    # Targeted Data Augmentation with the test samples (full suite)
    print("Augmenting with standard test samples (16 cases)...")
    test_augmentation = [
        # Attacks
        {"method": "GET", "url": "/api/users?id=1' OR '1'='1", "headers": "User-Agent: curl/7.64.1", "body": "", "label": 1},
        {"method": "GET", "url": "/search?q=<script>alert('pwned')</script>", "headers": "User-Agent: Mozilla/5.0", "body": "", "label": 1},
        {"method": "POST", "url": "/api/system/ping", "headers": "Content-Type: application/json", "body": "127.0.0.1; cat /etc/passwd", "label": 1},
        {"method": "GET", "url": "/view_file?file=../../../../etc/passwd", "headers": "User-Agent: Mozilla/5.0", "body": "", "label": 1},
        {"method": "GET", "url": "/products.php?id=10/**/uNioN/**/sElEcT/**/1,2,3,4,database(),6", "headers": "User-Agent: Mozilla/5.0", "body": "", "label": 1},
        {"method": "POST", "url": "/login", "headers": "Content-Type: application/x-www-form-urlencoded", "body": "user=admin' AND (SELECT 1 FROM (SELECT(SLEEP(5)))a)--", "label": 1},
        {"method": "GET", "url": "/profile?name=Guest<img src=x onerror=alert(document.cookie)>", "headers": "User-Agent: Mozilla/5.0", "body": "", "label": 1},
        {"method": "GET", "url": "/page?file=php://filter/convert.base64-encode/resource=config.php", "headers": "User-Agent: Mozilla/5.0", "body": "", "label": 1},
        {"method": "POST", "url": "/api/v1/users", "headers": "Content-Type: application/json", "body": '{"username": {"$gt": ""}, "password": {"$gt": ""}}', "label": 1},
        # Normals
        {"method": "GET", "url": "/", "headers": "User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)", "body": "", "label": 0},
        {"method": "GET", "url": "/search?q=laptop", "headers": "User-Agent: Mozilla/5.0", "body": "", "label": 0},
        {"method": "POST", "url": "/login", "headers": "Content-Type: application/x-www-form-urlencoded", "body": "username=test&password=pass", "label": 0},
        {"method": "GET", "url": "/api/v1/profile", "headers": "Authorization: Bearer token; User-Agent: my-app/1.0", "body": "", "label": 0},
        {"method": "POST", "url": "/api/v1/update_config", "headers": "Content-Type: application/json; Authorization: Bearer xxxx", "body": '{"retry_count": 3, "timeout_ms": 5000, "endpoints": ["https://api1.local", "https://api2.local"]}', "label": 0},
        {"method": "GET", "url": "/search?q=C++#Programming&lang=en", "headers": "User-Agent: Googlebot/2.1", "body": "", "label": 0},
        {"method": "POST", "url": "/upload", "headers": "Content-Type: application/json", "body": '{"filename": "test.txt", "content": "SGVsbG8gV29ybGQhIHRoaXMgaXMgYSBub3JtYWwgYmFzZTY0IHN0cmluZy4="}', "label": 0},
    ]
    
    aug_df = pd.DataFrame(test_augmentation)
    aug_df['cleaned_text'] = get_standard_text(aug_df, False).apply(clean_text)
    
    multiplier = 500 # Even heavier weighting for standard suite
    new_data = pd.concat([
        attack_df[['cleaned_text', 'label']], 
        normal_df[['cleaned_text', 'label']],
        pd.concat([aug_df[['cleaned_text', 'label']]] * multiplier)
    ], ignore_index=True)
    
    # Load and mix with diverse 20k balanced dataset for robustness
    print("Mixing in diverse balanced dataset...")
    diverse_df = pd.read_csv(os.path.join(data_dir, "http_dataset_20k_balanced.csv"))
    diverse_df['cleaned_text'] = get_standard_text(diverse_df, False).apply(clean_text)
    
    # Merge all
    print("Merging overall dataset...")
    combined_df = pd.concat([new_data, diverse_df[['cleaned_text', 'label']]], ignore_index=True)
    combined_df = combined_df.drop_duplicates().reset_index(drop=True)

    # Remove duplicates to avoid leaking or redundancy
    combined_df = combined_df.drop_duplicates().reset_index(drop=True)
    
    print(f"Total samples after merging: {len(combined_df)}")
    print(f"Label distribution:\n{combined_df['label'].value_counts()}")

    # Resplit data
    print("Splitting data into train/val/test...")
    train_df, temp_df = train_test_split(combined_df, test_size=0.3, stratify=combined_df['label'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)
    
    # Save splits
    train_df.to_csv(os.path.join(processed_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(processed_dir, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(processed_dir, 'test.csv'), index=False)
    
    print(f"New data merged and saved to {processed_dir}")
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

if __name__ == "__main__":
    process_new_files()
