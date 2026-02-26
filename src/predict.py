import joblib
import os
import json
import sys
import pandas as pd
# Add src to path to import components
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing import clean_text
from feature_engineering import FeatureEngineer

class HTTPAttackPredictor:
    def __init__(self, model_path, vectorizer_path):
        self.model = joblib.load(model_path)
        self.fe = FeatureEngineer(vectorizer_path)

    def predict(self, http_data):
        """
        http_data can be a string or a dict with method, url, headers, body etc.
        """
        # Map input to standard schema for extraction
        if isinstance(http_data, dict):
            url = str(http_data.get('url', ''))
            import urllib.parse
            import re
            try:
                parts = urllib.parse.urlparse(url)
                path = parts.path
                query = parts.query
            except:
                path = url
                query = ""
            
            headers = str(http_data.get('headers', ''))
            match = re.search(r'User-Agent:\s*(.*)', headers, re.IGNORECASE)
            ua = match.group(1) if match else str(http_data.get('user_agent', ''))

            std_row = {
                'method': str(http_data.get('method', '')),
                'path': path,
                'query': query,
                'headers': headers,
                'body': str(http_data.get('body', '')),
                'ua': ua
            }
            df = pd.DataFrame([std_row])
        else:
            # Handle generic strings
            df = pd.DataFrame([{
                'method': "", 'path': "", 'query': "", 'headers': str(http_data), 'body': "", 'ua': ""
            }])
        features = self.fe.transform(df)
        
        prob = self.model.predict_proba(features)[0][1]
        
        # Calibrated threshold (0.6) for anchor-trained model
        prediction = "ATTACK" if prob >= 0.6 else "NORMAL"
        
        return {
            "prediction": prediction,
            "confidence": round(float(prob if prob >= 0.5 else 1 - prob), 4)
        }

if __name__ == "__main__":
    model_file = "/Users/dmac/Desktop/ml/models/model.joblib"
    vectorizer_file = "/Users/dmac/Desktop/ml/models/vectorizer.joblib"
    
    if not (os.path.exists(model_file) and os.path.exists(vectorizer_file)):
        print("Error: Model files not found. Run train.py first.")
        sys.exit(1)
        
    predictor = HTTPAttackPredictor(model_file, vectorizer_file)
    
    # Check if input is provided via CLI
    if len(sys.argv) > 1:
        input_text = " ".join(sys.argv[1:])
        result = predictor.predict(input_text)
        print(json.dumps(result, indent=2))
    else:
        # Default test cases
        test_inputs = [
            "/login",
            "SELECT * FROM users",
            "<script>alert(1)</script>",
            "/LoGiN",
            "/homeSS"
        ]
        print("Running sample predictions:")
        for ti in test_inputs:
            res = predictor.predict(ti)
            print(f"Input: {ti} -> {res}")
