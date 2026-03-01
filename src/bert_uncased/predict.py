import os
import sys
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preprocessing import clean_text

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

class HTTPAttackPredictor:
    def __init__(self, model_dir=f"{PROJECT_ROOT}/models/bert_uncased"):
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()
        
    def predict(self, raw_payload):
        cleaned = clean_text(raw_payload)
        inputs = self.tokenizer(cleaned, return_tensors="pt", max_length=128, truncation=True, padding=True)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1)
        
        normal_prob = probs[0][0].item()
        attack_prob = probs[0][1].item()
        
        if attack_prob > 0.5:
            return "ATTACK", attack_prob
        else:
            return "NORMAL", normal_prob

if __name__ == '__main__':
    predictor = HTTPAttackPredictor()
    pred, conf = predictor.predict("SELECT * FROM users")
    print(f"Predicted: {pred}, Confidence: {conf}")
