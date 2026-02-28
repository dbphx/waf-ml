import os
import sys
import re

# Allow importing from parent src/ directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from random_forest.predict import HTTPAttackPredictor

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
    models_dir = "/Users/dmac/Desktop/ml/models/random_forest"
    data_dir = "/Users/dmac/Desktop/ml/data"
    
    if not (os.path.exists(os.path.join(models_dir, 'model.joblib'))):
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
            print(f"Warning: {tf['file']} not found.")
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
    
    # Save detailed report
    report_path = "/Users/dmac/Desktop/ml/reports/random_forest/categorical_results.json"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    import json
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Detailed categorical report saved to {report_path}")

if __name__ == "__main__":
    run_categorical_test()
