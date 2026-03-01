import pandas as pd
import numpy as np
import joblib
import os
import sys

# Allow importing from parent src/ directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from feature_engineering import FeatureEngineer
from preprocessing import clean_text

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

def run_robustness_tests(model, fe):
    print("\nRunning Robustness Tests...")
    test_cases = [
        # Case mutation
        {"text": "/LoGiN", "label": 0, "desc": "Case mutation (Normal)"},
        {"text": "SELECT * FROM users", "label": 1, "desc": "SQL Injection (Attack)"},
        {"text": "SeLeCt * FrOm users", "label": 1, "desc": "Case mutation (Attack)"},
        # Encoding
        {"text": "%2e%2e%2fetc%2fpasswd", "label": 1, "desc": "URL Encoded Path Traversal"},
        {"text": "%3cscript%3ealert(1)%3c/script%3e", "label": 1, "desc": "URL Encoded XSS"},
        # Obfuscation
        {"text": "/**/union/**/select", "label": 1, "desc": "SQLi Obfuscation"},
        # Benign lookalikes
        {"text": "/homeSS", "label": 0, "desc": "Benign lookalike (SS)"},
        {"text": "/selecta", "label": 0, "desc": "Benign lookalike (selecta)"},
        {"text": "/scripture", "label": 0, "desc": "Benign lookalike (scripture)"},
    ]
    
    results = []
    for case in test_cases:
        cleaned = clean_text(case['text'])
        features = fe.transform([cleaned])
        pred = model.predict(features)[0]
        conf = model.predict_proba(features)[0].max()
        
        status = "PASS" if pred == case['label'] else "FAIL"
        results.append({
            "test": case['desc'],
            "input": case['text'],
            "expected": "ATTACK" if case['label'] == 1 else "NORMAL",
            "prediction": "ATTACK" if pred == 1 else "NORMAL",
            "confidence": f"{conf:.4f}",
            "status": status
        })
        print(f"[{status}] {case['desc']}: Input='{case['text']}' -> Pred='{results[-1]['prediction']}' (Conf: {conf:.4f})")
    
    return results

def evaluate():
    processed_dir = f"{PROJECT_ROOT}/data/processed"
    models_dir = f"{PROJECT_ROOT}/models/logistic_regression"
    reports_dir = f"{PROJECT_ROOT}/reports/logistic_regression"
    os.makedirs(reports_dir, exist_ok=True)
    
    print("Loading test data...")
    test_df = pd.read_csv(os.path.join(processed_dir, 'test.csv'))
    
    print("Loading model and vectorizer...")
    model = joblib.load(os.path.join(models_dir, 'model.joblib'))
    fe = FeatureEngineer(os.path.join(models_dir, 'vectorizer.joblib'))
    
    print("Transforming test data...")
    X_test = fe.transform(test_df['cleaned_text'].fillna(""))
    y_test = test_df['label']
    
    print("Making predictions...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate Metrics
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1_score": float(f1_score(y_test, y_pred)),
    }
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fpr = float(fp / (fp + tn))
    metrics["false_positive_rate"] = fpr
    
    print("\nEvaluation Metrics:")
    for m, v in metrics.items():
        print(f"{m.capitalize()}: {v:.4f}")
    
    # Check "Definition of Done"
    dod_status = "PASSED" if metrics['recall'] > 0.95 and metrics['false_positive_rate'] < 0.03 else "FAILED"
    print(f"\nDefinition of Done Status: {dod_status}")
    
    # Save metrics
    with open(os.path.join(reports_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
        
    # Confusion Matrix Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(reports_dir, 'confusion_matrix.png'))
    print(f"Confusion matrix saved to {reports_dir}")
    
    # Robustness Testing
    robustness_results = run_robustness_tests(model, fe)
    with open(os.path.join(reports_dir, 'robustness_tests.json'), 'w') as f:
        json.dump(robustness_results, f, indent=4)

if __name__ == "__main__":
    evaluate()
