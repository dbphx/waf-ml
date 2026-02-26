from predict import HTTPAttackPredictor
import os
import json

def main():
    model_path = "/Users/dmac/Desktop/ml/models/model.joblib"
    vectorizer_path = "/Users/dmac/Desktop/ml/models/vectorizer.joblib"
    
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        print("Error: Model or vectorizer not found. Please train the model first.")
        return

    predictor = HTTPAttackPredictor(model_path, vectorizer_path)

    # Mixed Sample: Normal and Attack Request Examples
    samples = [
        {
            "description": "Normal: Homepage Access",
            "data": {
                "method": "GET",
                "url": "/",
                "headers": "User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
                "body": ""
            }
        },
        {
            "description": "Attack: Simple SQL Injection in URL",
            "data": {
                "method": "GET",
                "url": "/api/users?id=1' OR '1'='1",
                "headers": "User-Agent: curl/7.64.1",
                "body": ""
            }
        },
        {
            "description": "Normal: Product Search",
            "data": {
                "method": "GET",
                "url": "/search?q=laptop&category=electronics",
                "headers": "User-Agent: Mozilla/5.0",
                "body": ""
            }
        },
        {
            "description": "Attack: Cross-Site Scripting (XSS) in URL",
            "data": {
                "method": "GET",
                "url": "/search?q=<script>alert('pwned')</script>",
                "headers": "User-Agent: Mozilla/5.0",
                "body": ""
            }
        },
        {
            "description": "Normal: Login Attempt",
            "data": {
                "method": "POST",
                "url": "/login",
                "headers": "Content-Type: application/x-www-form-urlencoded",
                "body": "username=john_doe&password=secure_password123"
            }
        },
        {
            "description": "Attack: Command Injection in Body",
            "data": {
                "method": "POST",
                "url": "/api/system/ping",
                "headers": "Content-Type: application/json",
                "body": '{"ip": "127.0.0.1; cat /etc/passwd"}'
            }
        },
        {
            "description": "Attack: Path Traversal Attempt",
            "data": {
                "method": "GET",
                "url": "/view_file?file=../../../../etc/passwd",
                "headers": "User-Agent: Mozilla/5.0",
                "body": ""
            }
        },
        {
            "description": "Normal: API Request with Auth Header",
            "data": {
                "method": "GET",
                "url": "/api/v1/profile",
                "headers": "Authorization: Bearer my_secret_token_123; User-Agent: my-app/1.0",
                "body": ""
            }
        },
        {
            "description": "Attack: SQLi with Comments and Unconventional Case",
            "data": {
                "method": "GET",
                "url": "/products.php?id=10/**/uNioN/**/sElEcT/**/1,2,3,4,database(),6",
                "headers": "User-Agent: Mozilla/5.0",
                "body": ""
            }
        },
        {
            "description": "Attack: Blind SQLi (Time-based)",
            "data": {
                "method": "POST",
                "url": "/login",
                "headers": "Content-Type: application/x-www-form-urlencoded",
                "body": "user=admin' AND (SELECT 1 FROM (SELECT(SLEEP(5)))a)--"
            }
        },
        {
            "description": "Attack: Reflected XSS (Event Handler)",
            "data": {
                "method": "GET",
                "url": "/profile?name=Guest<img src=x onerror=alert(document.cookie)>",
                "headers": "User-Agent: Mozilla/5.0",
                "body": ""
            }
        },
        {
            "description": "Attack: Local File Inclusion (LFI) via wrapper",
            "data": {
                "method": "GET",
                "url": "/page?file=php://filter/convert.base64-encode/resource=config.php",
                "headers": "User-Agent: Mozilla/5.0",
                "body": ""
            }
        },
        {
            "description": "Attack: NoSQL Injection",
            "data": {
                "method": "POST",
                "url": "/api/v1/users",
                "headers": "Content-Type: application/json",
                "body": '{"username": {"$gt": ""}, "password": {"$gt": ""}}'
            }
        },
        {
            "description": "Normal: Complex JSON Payload",
            "data": {
                "method": "POST",
                "url": "/api/v1/update_config",
                "headers": "Content-Type: application/json; Authorization: Bearer xxxx",
                "body": '{"retry_count": 3, "timeout_ms": 5000, "endpoints": ["https://api1.local", "https://api2.local"]}'
            }
        },
        {
            "description": "Normal: Search with Special Characters (Non-malicious)",
            "data": {
                "method": "GET",
                "url": "/search?q=C++#Programming&lang=en",
                "headers": "User-Agent: Googlebot/2.1",
                "body": ""
            }
        },
        {
            "description": "Normal: Large Encoded Data (Base64)",
            "data": {
                "method": "POST",
                "url": "/upload",
                "headers": "Content-Type: application/json",
                "body": '{"filename": "test.txt", "content": "SGVsbG8gV29ybGQhIHRoaXMgaXMgYSBub3JtYWwgYmFzZTY0IHN0cmluZy4="}'
            }
        }
    ]

    print(f"{'Description':<40} | {'Prediction':<10} | {'Confidence'}")
    print("-" * 70)

    for sample in samples:
        result = predictor.predict(sample['data'])
        prediction = result['prediction']
        confidence = result['confidence']
        print(f"{sample['description']:<40} | {prediction:<10} | {confidence:.4f}")

if __name__ == "__main__":
    main()
