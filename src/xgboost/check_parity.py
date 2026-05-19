import os
import sys

import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from feature_engineering import FeatureEngineer

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


def check_parity():
    models_dir = f"{PROJECT_ROOT}/models/xgboost"
    onnx_path = f"{PROJECT_ROOT}/application/go/xgboost/assets/model.onnx"

    model = joblib.load(os.path.join(models_dir, "model.joblib"))
    fe = FeatureEngineer(os.path.join(models_dir, "vectorizer.joblib"))

    sample_data = {
        "path": "/search",
        "query": "q=apple' OR '1'='1",
        "headers": "",
    }
    df = pd.DataFrame([sample_data])
    X_sparse = fe.transform(df)
    X_dense = X_sparse.toarray().astype(np.float32)

    prob_sklearn = model.predict_proba(X_sparse)[0][1]

    X_onnx = X_dense.copy()
    X_onnx[X_onnx == 0] = np.nan

    sess = ort.InferenceSession(onnx_path)
    input_name = sess.get_inputs()[0].name
    outputs = sess.run(None, {input_name: X_onnx})
    prob_onnx = outputs[1][0][1]

    print(f"XGBoost Probability:      {prob_sklearn:.6f}")
    print(f"ONNX Probability:         {prob_onnx:.6f}")
    if np.isclose(prob_sklearn, prob_onnx, atol=1e-4):
        print("Status: PARITY VERIFIED ✅")
    else:
        print("Status: PARITY FAILED ❌")


if __name__ == "__main__":
    check_parity()
