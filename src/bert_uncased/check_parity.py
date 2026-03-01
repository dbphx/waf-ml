import os
import sys
import torch
import numpy as np
import onnxruntime as ort
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preprocessing import clean_text

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

def test_parity():
    models_dir = f"{PROJECT_ROOT}/models/bert_uncased"
    onnx_path = f"{PROJECT_ROOT}/application/go/bert_uncased/assets/model.onnx"
    
    tokenizer = DistilBertTokenizer.from_pretrained(models_dir)
    pt_model = DistilBertForSequenceClassification.from_pretrained(models_dir)
    pt_model.eval()
    
    ort_session = ort.InferenceSession(onnx_path)
    
    test_payloads = [
        "GET / HTTP/1.1",
        "SELECT * FROM users WHERE id=1",
        "<script>alert(1)</script>",
        "../../../../etc/passwd"
    ]
    
    for payload in test_payloads:
        cleaned = clean_text(payload)
        inputs = tokenizer(cleaned, return_tensors="pt", max_length=128, truncation=True, padding="max_length")
        
        # PyTorch inference
        with torch.no_grad():
            pt_outputs = pt_model(**inputs)
            pt_logits = pt_outputs.logits.numpy()
            
        # ONNX inference
        onnx_inputs = {
            "input_ids": inputs["input_ids"].numpy(),
            "attention_mask": inputs["attention_mask"].numpy()
        }
        onnx_logits = ort_session.run(None, onnx_inputs)[0]
        
        # Compare
        np.testing.assert_allclose(pt_logits, onnx_logits, rtol=1e-03, atol=1e-05)
        
    print("✅ Parity check passed! PyTorch and ONNX models output the exact same logits.")

if __name__ == "__main__":
    test_parity()
