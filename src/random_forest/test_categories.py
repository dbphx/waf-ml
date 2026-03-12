import os
import sys
import re
import urllib.parse

# Allow importing from parent src/ directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from random_forest.predict import HTTPAttackPredictor

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

def parse_file(filepath):
    """Parses categories and samples from the generated txt files."""
    categories = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        try:
            parts = line.split(":", 1)
            if len(parts) == 2:
                cat_match = re.match(r'^\d+\.\s+(.*)$', parts[0].strip())
                if cat_match:
                    categories.append({
                        "category": cat_match.group(1).strip(),
                        "payload": parts[1].strip()
                    })
        except:
            pass
    return categories

def run_categorical_test(use_onnx=False):
    models_dir = f"{PROJECT_ROOT}/models/random_forest"
    data_dir = f"{PROJECT_ROOT}/data"
    
    if use_onnx:
        onnx_path = f"{PROJECT_ROOT}/application/go/random_forest/assets/model.onnx"
        if not os.path.exists(onnx_path) and not os.path.exists(os.path.join(models_dir, 'model.onnx')):
             print("Error: ONNX model file not found.")
             return
    else:
        if not (os.path.exists(os.path.join(models_dir, 'model.joblib'))):
            print("Error: Model files not found. Run train.py first.")
            return

    predictor = HTTPAttackPredictor(models_dir, use_onnx=use_onnx)
    
    test_files = [
        {"file": "attack.txt", "expected": "ATTACK"},
        {"file": "normal.txt", "expected": "NORMAL"}
    ]
    
    results = []
    print(f"{'Category':<50} | {'Type':<8} | {'Expected':<8} | {'Pred':<8} | {'Conf':<6} | {'Time (ms)':<9} | {'Status'}")
    print("-" * 115)
    
    total = 0
    passed = 0
    
    import time

    def predict_as_runtime_request(payload):
        method = "GET"
        url = str(payload)
        headers = ""
        body = ""
        user_agent = ""

        first_line = url.strip().splitlines()[0] if url.strip() else ""
        m = re.match(r'^(GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS)\s+(\S+)', first_line, re.IGNORECASE)
        if m:
            method = m.group(1).upper()
            url = m.group(2)
            remainder = str(payload)[m.end():].lstrip()
            if remainder:
                parts = re.split(r'\r\n\r\n|\n\n', remainder, maxsplit=1)
                headers = parts[0].strip()
                body = parts[1].strip() if len(parts) > 1 else ""
                ua_match = re.search(r'(?im)^User-Agent:\s*(.+)$', headers)
                if ua_match:
                    user_agent = ua_match.group(1).strip()

        return predictor.predict({
            "method": method,
            "url": url,
            "headers": headers,
            "body": body,
            "user_agent": user_agent
        })
    
    for tf in test_files:
        path = os.path.join(data_dir, tf['file'])
        if not os.path.exists(path):
            print(f"Warning: {tf['file']} not found.")
            continue
            
        categories = parse_file(path)
        for cat in categories:
            # 1. Test Original
            total += 1
            t0 = time.time()
            pred, conf = predict_as_runtime_request(cat['payload'])
            elapsed = (time.time() - t0) * 1000
            
            is_correct = pred == tf['expected']
            if is_correct: passed += 1
            
            status = "✅" if is_correct else "❌"
            print(f"{cat['category'][:50]:<50} | {'RAW':<8} | {tf['expected']:<8} | {pred:<8} | {conf:.4f} | {elapsed:>8.2f} | {status}")
            
            results.append({
                "category": cat['category'],
                "payload": cat['payload'],
                "type": "RAW",
                "expected": tf['expected'],
                "predicted": pred,
                "confidence": conf,
                "time_ms": elapsed,
                "correct": is_correct
            })

            # 2. Test Encoded
            encoded_payload = urllib.parse.quote(cat['payload'])
            total += 1
            t0 = time.time()
            pred_enc, conf_enc = predict_as_runtime_request(encoded_payload)
            elapsed_enc = (time.time() - t0) * 1000
            
            is_correct_enc = pred_enc == tf['expected']
            if is_correct_enc: passed += 1
            
            status_enc = "✅" if is_correct_enc else "❌"
            print(f"{cat['category'][:50]:<50} | {'ENC':<8} | {tf['expected']:<8} | {pred_enc:<8} | {conf_enc:.4f} | {elapsed_enc:>8.2f} | {status_enc}")
            
            results.append({
                "category": cat['category'],
                "payload": encoded_payload,
                "type": "ENCODED",
                "expected": tf['expected'],
                "predicted": pred_enc,
                "confidence": conf_enc,
                "time_ms": elapsed_enc,
                "correct": is_correct_enc
            })

    print("-" * 115)
    accuracy = (passed / total) * 100 if total > 0 else 0
    print(f"SUMMARY: {passed}/{total} Passed ({accuracy:.2f}%)")
    
    # Save detailed report
    report_path = f"{PROJECT_ROOT}/reports/random_forest/categorical_results.json"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    import json
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Detailed categorical report saved to {report_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test classification of various payloads")
    parser.add_argument('--onnx', action='store_true', help="Use ONNX model instead of Joblib")
    args = parser.parse_args()
    
    run_categorical_test(use_onnx=args.onnx)
