"""
Copy SecureBERT2 checkpoint + tokenizer into application/go/securebert2/assets for packaging.
Also exports end-to-end model.onnx (encoder + mean-pool + LR head) for onnxruntime.
"""
import json
import os
import shutil
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


def export():
    models_dir = os.path.join(PROJECT_ROOT, "models", "securebert2")
    go_dir = os.path.join(PROJECT_ROOT, "application", "go", "securebert2", "assets")
    os.makedirs(go_dir, exist_ok=True)

    if not os.path.isfile(os.path.join(models_dir, "head.joblib")):
        print(f"Nothing to export: train first (missing {models_dir}/head.joblib)")
        sys.exit(1)

    for name in os.listdir(models_dir):
        src = os.path.join(models_dir, name)
        dst = os.path.join(go_dir, name)
        if os.path.isfile(src):
            shutil.copy2(src, dst)
        elif os.path.isdir(src):
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)

    meta = {
        "backend": "transformers",
        "hf_model_id": "cisco-ai/SecureBERT2.0-base",
        "note": "Inference: Python (src/securebert2/predict.py). Go waf.BaseDetector is not compatible with this ONNX schema.",
    }
    thresh = os.path.join(models_dir, "threshold.json")
    if os.path.isfile(thresh):
        with open(thresh) as f:
            meta["threshold"] = json.load(f)
    inf = os.path.join(models_dir, "inference_config.json")
    if os.path.isfile(inf):
        with open(inf) as f:
            meta["inference"] = json.load(f)

    with open(os.path.join(go_dir, "model_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Copied SecureBERT2 assets to {go_dir}")

    # ONNX (Python onnxruntime; schema differs from sklearn TF-IDF Go detector)
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
    from securebert2.onnx_model import export_end_to_end_onnx, verify_onnx_vs_sklearn

    inf_path = os.path.join(models_dir, "inference_config.json")
    max_len = 256
    if os.path.isfile(inf_path):
        with open(inf_path) as f:
            max_len = int(json.load(f).get("max_length", 256))

    onnx_models = os.path.join(models_dir, "model.onnx")
    onnx_go = os.path.join(go_dir, "model.onnx")
    try:
        export_end_to_end_onnx(models_dir, onnx_models, max_len)
        shutil.copy2(onnx_models, onnx_go)
        verify_onnx_vs_sklearn(models_dir, onnx_models, max_len)
        meta["onnx"] = {"path": "model.onnx", "inputs": ["input_ids", "attention_mask"], "output": "logit"}
        with open(os.path.join(go_dir, "model_metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"ONNX export OK: {onnx_models}")
    except Exception as e:
        print(f"ONNX export failed (PyTorch path still works): {e}", file=sys.stderr)


if __name__ == "__main__":
    export()
