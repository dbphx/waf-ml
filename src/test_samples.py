import sys
import os
import json
import argparse

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='logistic_regression',
                        choices=['logistic_regression', 'random_forest', 'bert_uncased'],
                        help='Model folder to use for predictions')
    args = parser.parse_args()
    
    model_name = args.model
    models_dir = f"{PROJECT_ROOT}/models/{model_name}"
    
    # Add model source dir to path and import
    model_src_dir = os.path.join(os.path.dirname(__file__), model_name)
    if model_src_dir not in sys.path:
        sys.path.insert(0, model_src_dir)
        
    try:
        from predict import HTTPAttackPredictor
    except ImportError as e:
        print(f"Error importing predictor for {model_name}: {e}")
        return

    if not os.path.exists(os.path.join(models_dir, 'model.joblib')):
        print(f"Error: Model not found at {models_dir}. Please train the model first.")
        return

    predictor = HTTPAttackPredictor(models_dir)

    # Mixed Sample: Normal and Attack Request Examples
    samples = [
        {
            "description": "Normal: Homepage Access",
            "data": {
                "method": "GET",
                "url": "/",
                "headers": "User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
                "body": ""
            },
            "expected": "NORMAL"
        },
        {
            "description": "Attack: Simple SQL Injection in URL",
            "data": {
                "method": "GET",
                "url": "/api/users?id=1' OR '1'='1",
                "headers": "User-Agent: curl/7.64.1",
                "body": ""
            },
            "expected": "ATTACK"
        },
        {
            "description": "Normal: Product Search",
            "data": {
                "method": "GET",
                "url": "/products?search=laptop&category=electronics",
                "headers": "User-Agent: Mozilla/5.0",
                "body": ""
            },
            "expected": "NORMAL"
        },
        {
            "description": "Attack: Cross-Site Scripting (XSS) in URL",
            "data": {
                "method": "GET",
                "url": "/search?q=<script>alert('XSS')</script>",
                "headers": "User-Agent: Mozilla/5.0",
                "body": ""
            },
            "expected": "ATTACK"
        },
        {
            "description": "Normal: Login Attempt",
            "data": {
                "method": "POST",
                "url": "/login",
                "headers": "Content-Type: application/x-www-form-urlencoded",
                "body": "user=john&pass=doe"
            },
            "expected": "NORMAL"
        },
        {
            "description": "Attack: Command Injection in Body",
            "data": {
                "method": "POST",
                "url": "/api/ping",
                "headers": "Content-Type: application/json",
                "body": '{"ip": "127.0.0.1; cat /etc/passwd"}'
            },
            "expected": "ATTACK"
        },
        {
            "description": "Attack: Path Traversal Attempt",
            "data": {
                "method": "GET",
                "url": "/view_file?file=../../../../etc/passwd",
                "headers": "",
                "body": ""
            },
            "expected": "ATTACK"
        },
        {
            "description": "Normal: API Request with Auth Header",
            "data": {
                "method": "GET",
                "url": "/api/v1/profile",
                "headers": "Authorization: Bearer valid_token_123",
                "body": ""
            },
            "expected": "NORMAL"
        },
        {
            "description": "Attack: SQLi with Comments and Unconventional Case",
            "data": {
                "method": "GET",
                "url": "/posts?id=1/*UNION*/sELecT/**/password/**/fRoM/**/users",
                "headers": "",
                "body": ""
            },
            "expected": "ATTACK"
        },
        {
            "description": "Attack: Blind SQLi (Time-based)",
            "data": {
                "method": "GET",
                "url": "/vulnerable.php?id=1-SLEEP(5)",
                "headers": "",
                "body": ""
            },
            "expected": "ATTACK"
        },
        {
            "description": "Attack: Reflected XSS (Event Handler)",
            "data": {
                "method": "GET",
                "url": "/search?q=<img src=x onerror=alert(1)>",
                "headers": "",
                "body": ""
            },
            "expected": "ATTACK"
        },
        {
            "description": "Attack: Local File Inclusion (LFI) via wrapper",
            "data": {
                "method": "GET",
                "url": "/index.php?page=php://filter/convert.base64-encode/resource=config.php",
                "headers": "",
                "body": ""
            },
            "expected": "ATTACK"
        },
        {
            "description": "Attack: NoSQL Injection",
            "data": {
                "method": "POST",
                "url": "/api/login",
                "headers": "Content-Type: application/json",
                "body": '{"username": {"$gt": ""}, "password": {"$gt": ""}}'
            },
            "expected": "ATTACK"
        },
        {
            "description": "Normal: Complex JSON Payload",
            "data": {
                "method": "POST",
                "url": "/api/data",
                "headers": "Content-Type: application/json",
                "body": '{"metadata": {"version": "1.0", "encoding": "UTF-8"}, "items": [{"id": 1, "tags": ["tag1", "tag2"]}]}'
            },
            "expected": "NORMAL"
        },
        {
            "description": "Normal: Search with Special Characters (Non-malicious)",
            "data": {
                "method": "GET",
                "url": "/products?filter=price > 100 AND category IN ('books', 'games')",
                "headers": "",
                "body": ""
            },
            "expected": "NORMAL"
        },
        {
            "description": "Normal: Large Encoded Data (Base64)",
            "data": {
                "method": "POST",
                "url": "/api/upload",
                "headers": "Content-Type: application/json",
                "body": '{"filename": "image.png", "data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="}'
            },
            "expected": "NORMAL"
        }
    ]

    print(f"{'Description':<50} | {'Result':<10} | {'Confidence':<10} | Status")
    print("-" * 85)

    passed = 0
    for sample in samples:
        # returns (pred, conf)
        pred, conf = predictor.predict(sample['data'])
        is_correct = pred == sample['expected']
        if is_correct:
            passed += 1
        
        status = "✅" if is_correct else "❌"
        print(f"{sample['description']:<50} | {pred:<10} | {conf:<10.4f} | {status}")

    print("-" * 85)
    print(f"Passed {passed}/{len(samples)} samples.")

if __name__ == "__main__":
    main()
