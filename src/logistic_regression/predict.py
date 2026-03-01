import pandas as pd
import numpy as np
import joblib
import os
import sys

# Allow importing from parent src/ directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from feature_engineering import FeatureEngineer
from preprocessing import parse_http_string

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

class HTTPAttackPredictor:
    def __init__(self, model_dir=f"{PROJECT_ROOT}/models/logistic_regression"):
        self.model = joblib.load(os.path.join(model_dir, 'model.joblib'))
        self.fe = FeatureEngineer(os.path.join(model_dir, 'vectorizer.joblib'))

    def predict(self, http_data):
        if isinstance(http_data, dict):
            url = str(http_data.get('url', ''))
            import urllib.parse
            try:
                parts = urllib.parse.urlparse(url)
                path = parts.path
                query = parts.query
            except:
                path = url
                query = ""
            df = pd.DataFrame([{
                'method': str(http_data.get('method', '')),
                'path': path,
                'query': query,
                'headers': str(http_data.get('headers', '')),
                'body': str(http_data.get('body', '')),
                'ua': str(http_data.get('user_agent', ''))
            }])
        else:
            parsed = parse_http_string(str(http_data))
            parsed['ua'] = ""
            df = pd.DataFrame([parsed])

        X = self.fe.transform(df)
        prob = self.model.predict_proba(X)[0][1]
        
        threshold = 0.7
        prediction = "ATTACK" if prob >= threshold else "NORMAL"
        confidence = round(float(prob if prob >= threshold else 1 - prob), 4)
        
        return prediction, confidence

if __name__ == "__main__":
    predictor = HTTPAttackPredictor()
    res = predictor.predict("id=1' OR '1'='1")
    print(res)
