import sys
import os
import urllib.parse
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), 'random_forest'))
from predict import HTTPAttackPredictor

predictor = HTTPAttackPredictor(os.path.join(PROJECT_ROOT, "models/random_forest"))

payload = """GET /uploads/628668cae9db4d51b4edf55214b73fca-Facebook Image.jpg?response-content-disposition=inline%3B%20filename%2A%3DUTF-8%27%276b67c704b68a4517aa6664c1df2f7afe&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=9wnmzzHYVGwlnTscRdzx%2F20260310%2F%2Fs3%2Faws4_request&X-Amz-Date=20260310T023341Z&X-Amz-Expires=3600&X-Amz-SignedHeaders=host&X-Amz-Signature=5bfffbe1cc40d1ef12ee72b2126381daf34408143c87a1c0bf11e66c162e101b&q=%3Cscript%3E%3C/script%3E"""
enc_payload = urllib.parse.quote(payload)

print("--- RAW ---")
pred, conf = predictor.predict(payload)
print(f"Prediction: {pred}, Confidence: {conf}")

print("\n--- ENCODED ---")
pred2, conf2 = predictor.predict(enc_payload)
print(f"Prediction: {pred2}, Confidence: {conf2}")
