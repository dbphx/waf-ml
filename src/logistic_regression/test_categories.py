import os
import sys
import re
import pandas as pd
import urllib.parse

# Allow importing from parent src/ directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from logistic_regression.predict import HTTPAttackPredictor

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

def parse_file(filepath):
    """Parses categories and samples from the generated txt files."""
    categories = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        # Match "Number. Category Name: Sample"
        match = re.match(r'^\d+\.\s+(.*?):\s+(.*)$', line.strip())
        if match:
            categories.append({
                "category": match.group(1),
                "payload": match.group(2)
            })
    return categories

def run_categorical_test():
    models_dir = f"{PROJECT_ROOT}/models/logistic_regression"
    data_dir = f"{PROJECT_ROOT}/data"
    
    if not (os.path.exists(os.path.join(models_dir, 'model.joblib'))):
        print("Error: Model files not found. Run train.py first.")
        return

    predictor = HTTPAttackPredictor(models_dir)
    
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
            pred, conf = predictor.predict(cat['payload'])
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

            # 2. Test Encoded (Only for attacks usually, but we do both)
            encoded_payload = urllib.parse.quote(cat['payload'])
            total += 1
            t0 = time.time()
            pred_enc, conf_enc = predictor.predict(encoded_payload)
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
    report_path = f"{PROJECT_ROOT}/reports/logistic_regression/categorical_results.json"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    import json
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Detailed categorical report saved to {report_path}")

if __name__ == "__main__":
    run_categorical_test()
