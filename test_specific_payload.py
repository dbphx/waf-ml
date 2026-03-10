import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src/random_forest')))
from predict import HTTPAttackPredictor

predictor = HTTPAttackPredictor("models/random_forest")

payload = "/uploads/628668cae9db4d51b4edf55214b73fca-Facebook Image.jpg?response-content-disposition=inline%3B%20filename%2A%3DUTF-8%27%276b67c704b68a4517aa6664c1df2f7afe&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=9wnmzzHYVGwlnTscRdzx%2F20260310%2F%2Fs3%2Faws4_request&X-Amz-Date=20260310T023341Z&X-Amz-Expires=3600&X-Amz-SignedHeaders=host&X-Amz-Signature=5bfffbe1cc40d1ef12ee72b2126381daf34408143c87a1c0bf11e66c162e101b&q=admin%20%27or%201=1;--"

print("--- Testing Full Payload ---")
pred, conf = predictor.predict(payload)
print(f"Prediction: {pred}, Confidence: {conf}")

print("\n--- Testing Attack Part Only ---")
attack_only = "q=admin%20%27or%201=1;--"
pred2, conf2 = predictor.predict(attack_only)
print(f"Prediction: {pred2}, Confidence: {conf2}")
