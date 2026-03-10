import sys
import os
import urllib.parse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src/random_forest')))
from predict import HTTPAttackPredictor

predictor = HTTPAttackPredictor("models/random_forest")

payload = "GET /"
enc_payload = urllib.parse.quote(payload)

print("--- RAW ---")
pred, conf = predictor.predict(payload)
print(f"Prediction: {pred}, Confidence: {conf}")

print("\n--- ENCODED ---")
pred2, conf2 = predictor.predict(enc_payload)
print(f"Prediction: {pred2}, Confidence: {conf2}")
