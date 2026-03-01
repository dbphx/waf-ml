import onnxruntime as ort
import joblib
import pandas as pd
import numpy as np
import sys
import os

# Allow importing from parent src/ directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from feature_engineering import FeatureEngineer

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

def check_parity():
    models_dir = f"{PROJECT_ROOT}/models/random_forest"
    onnx_path = f"{PROJECT_ROOT}/go/internal/assets/random_forest/model.onnx"
    
    # 1. Python Scikit-Learn Prediction
    model = joblib.load(os.path.join(models_dir, 'model.joblib'))
    fe = FeatureEngineer(os.path.join(models_dir, 'vectorizer.joblib'))
    
    sample_data = {
        "path": "/search",
        "query": "q=apple' OR '1'='1",
        "headers": ""
    }
    df = pd.DataFrame([sample_data])
    X_sparse = fe.transform(df)
    X_dense = X_sparse.toarray().astype(np.float32)
    
    prob_sklearn = model.predict_proba(X_sparse)[0][1]
    
    # 2. ONNX Prediction
    sess = ort.InferenceSession(onnx_path)
    input_name = sess.get_inputs()[0].name
    outputs = sess.run(None, {input_name: X_dense})
    
    # In skl2onnx, outputs are typically [label, probabilities_dict]
    # or probabilities array depending on zipmap. We turned off zipmap.
    # So outputs[1] should be a tensor of shape (N, C).
    prob_onnx = outputs[1][0][1]
    
    print(f"Scikit-Learn Probability: {prob_sklearn:.6f}")
    print(f"ONNX Probability:         {prob_onnx:.6f}")
    if np.isclose(prob_sklearn, prob_onnx, atol=1e-5):
        print("Status: PARITY VERIFIED ✅")
    else:
        print("Status: PARITY FAILED ❌")

if __name__ == "__main__":
    check_parity()
