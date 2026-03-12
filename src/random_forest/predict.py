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
    def __init__(self, model_dir=f"{PROJECT_ROOT}/models/random_forest", use_onnx=False):
        self.use_onnx = use_onnx
        if self.use_onnx:
            import onnxruntime as ort
            onnx_path = f"{PROJECT_ROOT}/application/go/random_forest/assets/model.onnx"
            if not os.path.exists(onnx_path):
                # Fallback to the python models dir just in case
                onnx_path = os.path.join(model_dir, 'model.onnx')
            self.sess = ort.InferenceSession(onnx_path)
            self.input_name = self.sess.get_inputs()[0].name
            import numpy as np # needed for dense conversion
        else:
            self.model = joblib.load(os.path.join(model_dir, 'model.joblib'))
            
        self.fe = FeatureEngineer(os.path.join(model_dir, 'vectorizer.joblib'))

    def predict(self, http_data):
        def signature_override(text):
            lowered = text.lower()
            suspicious_patterns = [
                'freemarker.template.utility.execute',
                'do{curdate = new date();}while(curdate-date<1 0000)'
            ]
            return any(p in lowered for p in suspicious_patterns)

        if isinstance(http_data, dict):
            url = str(http_data.get('url', ''))
            import urllib.parse
            try:
                parts = urllib.parse.urlparse(url)
                path = parts.path
                if parts.params:
                    path = f"{path};{parts.params}" if path else parts.params
                query = parts.query
                if parts.fragment:
                    query = f"{query}#{parts.fragment}" if query else parts.fragment
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
            signature_text = ' '.join([
                str(http_data.get('method', '')),
                str(http_data.get('url', '')),
                str(http_data.get('headers', '')),
                str(http_data.get('body', ''))
            ])
        else:
            parsed = parse_http_string(str(http_data))
            parsed['ua'] = ""
            df = pd.DataFrame([parsed])
            signature_text = str(http_data)

        if signature_override(signature_text):
            return "ATTACK", 1.0

        X = self.fe.transform(df)
        
        if self.use_onnx:
            import numpy as np
            X_dense = X.toarray().astype(np.float32)
            outputs = self.sess.run(None, {self.input_name: X_dense})
            prob = outputs[1][0][1]
        else:
            prob = self.model.predict_proba(X)[0][1]
        
        threshold = 0.55 # Optimized based on categorical analysis (0.41 < T < 0.64)
        prediction = "ATTACK" if prob >= threshold else "NORMAL"
        confidence = round(float(prob if prob >= threshold else 1 - prob), 4)
        
        return prediction, confidence

if __name__ == "__main__":
    predictor = HTTPAttackPredictor()
    res = predictor.predict("id=1' OR '1'='1")
    print(res)
