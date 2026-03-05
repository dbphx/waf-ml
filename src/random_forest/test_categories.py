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

def run_categorical_test():
    models_dir = f"{PROJECT_ROOT}/models/random_forest"
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
    print(f"{'Category':<50} | {'Type':<8} | {'Expected':<8} | {'Pred':<8} | {'Conf':<6} | {'Status'}")
    print("-" * 100)
    
    total = 0
    passed = 0
    
    for tf in test_files:
        path = os.path.join(data_dir, tf['file'])
        if not os.path.exists(path):
            print(f"Warning: {tf['file']} not found.")
            continue
            
        categories = parse_file(path)
        for cat in categories:
            # 1. Test Original
            total += 1
            pred, conf = predictor.predict(cat['payload'])
            
            is_correct = pred == tf['expected']
            if is_correct: passed += 1
            
            status = "✅" if is_correct else "❌"
            print(f"{cat['category'][:50]:<50} | {'RAW':<8} | {tf['expected']:<8} | {pred:<8} | {conf:.4f} | {status}")
            
            results.append({
                "category": cat['category'],
                "payload": cat['payload'],
                "type": "RAW",
                "expected": tf['expected'],
                "predicted": pred,
                "confidence": conf,
                "correct": is_correct
            })

            # 2. Test Encoded
            encoded_payload = urllib.parse.quote(cat['payload'])
            total += 1
            pred_enc, conf_enc = predictor.predict(encoded_payload)
            
            is_correct_enc = pred_enc == tf['expected']
            if is_correct_enc: passed += 1
            
            status_enc = "✅" if is_correct_enc else "❌"
            print(f"{cat['category'][:50]:<50} | {'ENC':<8} | {tf['expected']:<8} | {pred_enc:<8} | {conf_enc:.4f} | {status_enc}")
            
            results.append({
                "category": cat['category'],
                "payload": encoded_payload,
                "type": "ENCODED",
                "expected": tf['expected'],
                "predicted": pred_enc,
                "confidence": conf_enc,
                "correct": is_correct_enc
            })

    print("-" * 100)
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
    run_categorical_test()
