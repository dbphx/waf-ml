import hashlib
import json
import os
import subprocess
import sys

import joblib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from feature_engineering import FeatureEngineer, REQUEST_FIELDS, SUSPICIOUS_KEYWORDS

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

sys.setrecursionlimit(50000)


def get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def file_sha256(path):
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def export():
    models_dir = f"{PROJECT_ROOT}/models/xgboost"
    go_dir = f"{PROJECT_ROOT}/application/go/xgboost/assets"
    os.makedirs(go_dir, exist_ok=True)

    model = joblib.load(os.path.join(models_dir, "model.joblib"))
    fe = FeatureEngineer(os.path.join(models_dir, "vectorizer.joblib"))

    field_vectorizers = {}
    total_tfidf_features = 0
    for field in REQUEST_FIELDS:
        vectorizer = fe.vectorizers[field]
        vocab = vectorizer.vocabulary_
        sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
        terms = [item[0] for item in sorted_vocab]
        field_vectorizers[field] = {
            "ngram_range": vectorizer.ngram_range,
            "max_features": vectorizer.max_features,
            "vocabulary": terms,
            "idf": vectorizer.idf_.tolist(),
        }
        total_tfidf_features += len(terms)

    metadata = {
        "model_name": "xgboost",
        "field_order": list(REQUEST_FIELDS),
        "field_vectorizers": field_vectorizers,
        "keywords": SUSPICIOUS_KEYWORDS,
        "exported_from_commit": get_git_commit(),
    }

    print("Generating ONNX model...")
    import onnx
    from onnxmltools.convert import convert_xgboost
    from onnxmltools.convert.common.data_types import FloatTensorType

    stats_per_field = 2 + len(metadata["keywords"])
    n_features = total_tfidf_features + (len(REQUEST_FIELDS) * stats_per_field)
    initial_type = [("float_input", FloatTensorType([None, n_features]))]

    onnx_model = convert_xgboost(
        model,
        initial_types=initial_type,
        target_opset=12,
    )

    model_go_dir = f"{PROJECT_ROOT}/application/go/xgboost/assets"
    os.makedirs(model_go_dir, exist_ok=True)

    onnx_path = os.path.join(model_go_dir, "model.onnx")
    onnx.save_model(onnx_model, onnx_path)
    print(f"Exported ONNX model to {onnx_path}")

    onnx_python_path = os.path.join(models_dir, "model.onnx")
    onnx.save_model(onnx_model, onnx_python_path)
    print(f"Exported ONNX model to {onnx_python_path}")

    metadata["model_sha256"] = file_sha256(onnx_path)
    metadata_path = os.path.join(go_dir, "model_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Exported metadata to {metadata_path}")


if __name__ == "__main__":
    export()
