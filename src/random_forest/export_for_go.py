import os
import sys
import joblib
import json

# Allow importing from parent src/ directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from feature_engineering import FeatureEngineer

sys.setrecursionlimit(50000)

def export():
    models_dir = "/Users/dmac/Desktop/ml/models/random_forest"
    go_dir = "/Users/dmac/Desktop/ml/go/internal/assets/random_forest"
    os.makedirs(go_dir, exist_ok=True)
    
    # 1. Load Model and Vectorizer
    model = joblib.load(os.path.join(models_dir, 'model.joblib'))
    fe = FeatureEngineer(os.path.join(models_dir, 'vectorizer.joblib'))
    
    # 2. Export TF-IDF Parameters
    vectorizer = fe.vectorizer
    vocab = vectorizer.vocabulary_
    # Sort vocab by index to match feature vector
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
    # Extract just the terms
    terms = [item[0] for item in sorted_vocab]
    
    idf = vectorizer.idf_.tolist()
    
    metadata = {
        "ngram_range": vectorizer.ngram_range,
        "max_features": vectorizer.max_features,
        "vocabulary": terms,
        "idf": idf,
        "keywords": [
            'select', 'union', 'insert', 'update', 'delete', 'drop', 'from', 'where',
            'script', 'alert', 'onerror', 'eval', '../', './',
            '${', '{{', '}}', '() {', ';', '|', '&',
            '$gt', '$ne', '$in', 'cat ', 'whoami'
        ]
    }
    
    with open(os.path.join(go_dir, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Exported metadata to {go_dir}/model_metadata.json")
    
    # 3. Export to ONNX
    print("Generating ONNX model...")
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    # Define input type: dynamic batch size, fixed feature size
    n_features = len(vocab) + 2 + len(metadata["keywords"])
    initial_type = [('float_input', FloatTensorType([None, n_features]))]
    
    # Convert Scikit-Learn Random Forest to ONNX
    onnx_model = convert_sklearn(
        model, 
        initial_types=initial_type, 
        target_opset=12,
        options={type(model): {'zipmap': False}}
    )
    
    model_go_dir = "/Users/dmac/Desktop/ml/go/internal/assets/random_forest"
    os.makedirs(model_go_dir, exist_ok=True)
    
    onnx_path = os.path.join(model_go_dir, "model.onnx")
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"Exported ONNX model to {onnx_path}")

if __name__ == "__main__":
    export()
