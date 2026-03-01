import os
import sys
import re
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from bert_uncased.predict import HTTPAttackPredictor

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

def parse_file(filepath):
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
    models_dir = f"{PROJECT_ROOT}/models/bert_uncased"
    data_dir = f"{PROJECT_ROOT}/data"
    
    if not os.path.exists(models_dir):
        print("Error: Model files not found. Run train.py first.")
        return

    predictor = HTTPAttackPredictor(models_dir)
    
    test_files = [
        {"file": "attack.txt", "expected": "ATTACK"},
        {"file": "normal.txt", "expected": "NORMAL"}
    ]
    
    results = []
    print(f"{'Category':<50} | {'Expected':<8} | {'Pred':<8} | {'Conf':<6} | {'Status'}")
    print("-" * 90)
    
    total = 0
    passed = 0
    
    for tf in test_files:
        path = os.path.join(data_dir, tf['file'])
        if not os.path.exists(path):
            continue
            
        categories = parse_file(path)
        for cat in categories:
            total += 1
            pred, conf = predictor.predict(cat['payload'])
            
            is_correct = pred == tf['expected']
            if is_correct: passed += 1
            
            status = "✅" if is_correct else "❌"
            print(f"{cat['category'][:50]:<50} | {tf['expected']:<8} | {pred:<8} | {conf:.4f} | {status}")
            
            results.append({
                "category": cat['category'],
                "payload": cat['payload'],
                "expected": tf['expected'],
                "predicted": pred,
                "confidence": conf,
                "correct": is_correct
            })

    print("-" * 90)
    accuracy = (passed / total) * 100 if total > 0 else 0
    print(f"SUMMARY: {passed}/{total} Passed ({accuracy:.2f}%)")
    
    report_path = f"{PROJECT_ROOT}/reports/bert_uncased/categorical_results.json"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Detailed categorical report saved to {report_path}")

if __name__ == "__main__":
    run_categorical_test()
