import pandas as pd
import numpy as np
import joblib
import os
import sys

# Allow importing from parent src/ directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from feature_engineering import FeatureEngineer
from preprocessing import split_request_components

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

class HTTPAttackPredictor:
    def __init__(self, model_dir=f"{PROJECT_ROOT}/models/logistic_regression", use_onnx=False):
        self.use_onnx = bool(use_onnx)
        self.fe = FeatureEngineer(os.path.join(model_dir, "vectorizer.joblib"))
        if self.use_onnx:
            import onnxruntime as ort

            onnx_path = f"{PROJECT_ROOT}/application/go/logistic_regression/assets/model.onnx"
            if not os.path.exists(onnx_path):
                onnx_path = os.path.join(model_dir, "model.onnx")
            self.sess = ort.InferenceSession(onnx_path)
            self.input_name = self.sess.get_inputs()[0].name
            self.model = None
        else:
            self.model = joblib.load(os.path.join(model_dir, "model.joblib"))
            self.sess = None

    def _build_dataframe(self, method='', path='', query='', headers='', body='', ua=''):
        return pd.DataFrame([{
            'method': str(method),
            'path': str(path),
            'query': str(query),
            'headers': str(headers),
            'body': str(body),
            'ua': str(ua),
        }])

    def _predict_probability(self, df):
        X = self.fe.transform(df)
        if self.use_onnx:
            X_dense = X.toarray().astype(np.float32)
            return float(self.sess.run(None, {self.input_name: X_dense})[1][0][1])
        return float(self.model.predict_proba(X)[0][1])

    def _classify_probability(self, prob):
        threshold = 0.77 # Keeps a margin above current benign FP max (0.7621) while below attack min (0.9475)
        prediction = "ATTACK" if prob >= threshold else "NORMAL"
        confidence = round(float(prob if prob >= threshold else 1 - prob), 4)
        return prediction, confidence

    def _predict_component(self, component_name, component_value, method='', ua=''):
        df = self._build_dataframe(
            method=method,
            path=component_value if component_name == 'path' else '',
            query=component_value if component_name == 'query' else '',
            headers=component_value if component_name == 'headers' else '',
            body=component_value if component_name == 'body' else '',
            ua=ua,
        )
        prob = self._predict_probability(df)
        prediction, confidence = self._classify_probability(prob)
        return {
            'component': component_name,
            'value': component_value,
            'prediction': prediction,
            'confidence': confidence,
            'probability': round(prob, 4),
        }

    def predict_components(self, http_data):
        request = split_request_components(http_data)
        component_predictions = {}

        for component_name in ('path', 'query', 'headers', 'body'):
            component_value = str(request.get(component_name, '')).strip()
            if not component_value:
                continue
            component_predictions[component_name] = self._predict_component(
                component_name,
                component_value,
                method=request.get('method', ''),
                ua=request.get('user_agent', ''),
            )

        if component_predictions:
            decisive = max(component_predictions.values(), key=lambda item: item['probability'])
        else:
            decisive = self._predict_component('path', '/', method=request.get('method', ''), ua=request.get('user_agent', ''))
            component_predictions['path'] = decisive

        overall_prediction, overall_confidence = self._classify_probability(decisive['probability'])
        return {
            'request': request,
            'components': component_predictions,
            'prediction': overall_prediction,
            'confidence': overall_confidence,
            'decisive_component': decisive['component'],
        }

    def predict(self, http_data):
        result = self.predict_components(http_data)
        return result['prediction'], result['confidence']

if __name__ == "__main__":
    predictor = HTTPAttackPredictor()
    res = predictor.predict("id=1' OR '1'='1")
    print(res)
