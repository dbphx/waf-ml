import os
import sys
import torch
import json
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def export():
    models_dir = f"{PROJECT_ROOT}/models/bert_uncased"
    go_dir = f"{PROJECT_ROOT}/application/go/bert_uncased/assets"
    os.makedirs(go_dir, exist_ok=True)
    
    print("Loading model...")
    model = DistilBertForSequenceClassification.from_pretrained(models_dir)
    tokenizer = DistilBertTokenizer.from_pretrained(models_dir)
    
    model.eval()
    
    # Dummy input for tracing
    dummy_input = "SELECT * FROM users"
    inputs = tokenizer(dummy_input, return_tensors="pt", padding="max_length", max_length=128, truncation=True)
    
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    print("Exporting to ONNX...")
    onnx_path = os.path.join(go_dir, "model.onnx")
    
    torch.onnx.export(
        model, 
        (input_ids, attention_mask),
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input_ids', 'attention_mask'],
        output_names=['logits'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence_length'},
            'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
            'logits': {0: 'batch_size'}
        }
    )
    print(f"Exported ONNX model to {onnx_path}")
    # Explicitly dump the vocab.txt file from the tokenizer
    print("Exporting vocab.txt")
    vocab = tokenizer.get_vocab()
    # Sort by ID so they are written in sequential order
    sorted_vocab = sorted(vocab.items(), key=lambda item: item[1])
    vocab_path = os.path.join(go_dir, "vocab.txt")
    with open(vocab_path, "w", encoding="utf-8") as f:
        for word, _ in sorted_vocab:
            f.write(f"{word}\n")
        
    metadata = {
        "model_type": "distilbert",
        "max_length": 128
    }
    with open(os.path.join(go_dir, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

if __name__ == "__main__":
    export()
