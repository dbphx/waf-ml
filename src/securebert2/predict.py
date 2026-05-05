import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from preprocessing import parse_http_string

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


def mean_pool(last_hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden)
    summed = (last_hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    return summed / denom


class HTTPAttackPredictor:
    """Frozen SecureBERT 2.0 embeddings + sklearn logistic head (same HTTP API as logistic_regression)."""

    def __init__(self, model_dir=f"{PROJECT_ROOT}/models/securebert2", use_onnx=False):
        self.use_onnx = bool(use_onnx)
        self.model_dir = model_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

        thresh_path = os.path.join(model_dir, "threshold.json")
        if os.path.isfile(thresh_path):
            with open(thresh_path) as f:
                self.attack_threshold = float(json.load(f)["attack_threshold"])
        else:
            self.attack_threshold = 0.5
        cfg = os.path.join(model_dir, "inference_config.json")
        if os.path.isfile(cfg):
            with open(cfg) as f:
                self.max_length = int(json.load(f).get("max_length", 256))
        else:
            self.max_length = int(os.environ.get("SECUREBERT2_MAX_LEN", "256"))

        if self.use_onnx:
            import onnxruntime as ort

            onnx_path = os.path.join(model_dir, "model.onnx")
            if not os.path.isfile(onnx_path):
                onnx_path = os.path.join(
                    PROJECT_ROOT, "application", "go", "securebert2", "assets", "model.onnx"
                )
            if not os.path.isfile(onnx_path):
                raise FileNotFoundError(
                    f"model.onnx not found under {model_dir} or application/go/securebert2/assets. Run export."
                )
            self.sess = ort.InferenceSession(
                onnx_path, providers=["CPUExecutionProvider"]
            )
            self.base = None
            self.lr = None
        else:
            enc_dir = os.path.join(model_dir, "encoder")
            self.base = AutoModel.from_pretrained(enc_dir, trust_remote_code=True)
            self.base.to(self.device)
            self.base.eval()
            self.lr = joblib.load(os.path.join(model_dir, "head.joblib"))
            self.sess = None

    def _row_to_text(self, http_data: dict) -> str:
        url = str(http_data.get("url", ""))
        import urllib.parse

        try:
            parts = urllib.parse.urlparse(url)
            path = parts.path
            if parts.params:
                path = f"{path};{parts.params}" if path else parts.params
            query = parts.query
            if parts.fragment:
                query = f"{query}#{parts.fragment}" if query else parts.fragment
        except Exception:
            path = url
            query = ""

        from feature_engineering import FeatureEngineer
        from preprocessing import clean_text

        fe = FeatureEngineer()
        row = {
            "method": str(http_data.get("method", "")),
            "path": path,
            "query": query,
            "headers": str(http_data.get("headers", "")),
            "body": str(http_data.get("body", "")),
            "ua": str(http_data.get("user_agent", "")),
        }
        return clean_text(fe.extract_text(pd.Series(row)))

    def predict(self, http_data):
        if isinstance(http_data, dict):
            text = self._row_to_text(http_data)
        else:
            parsed = parse_http_string(str(http_data))
            parsed["ua"] = ""
            from feature_engineering import FeatureEngineer
            from preprocessing import clean_text

            fe = FeatureEngineer()
            text = clean_text(fe.extract_text(pd.Series(parsed)))

        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="np",
        )
        input_ids = enc["input_ids"].astype(np.int64)
        attention_mask = enc["attention_mask"].astype(np.int64)

        if self.use_onnx:
            logit = float(
                np.array(
                    self.sess.run(
                        None,
                        {"input_ids": input_ids, "attention_mask": attention_mask},
                    )[0]
                ).ravel()[0]
            )
            prob_attack = float(1.0 / (1.0 + np.exp(-logit)))
        else:
            enc_pt = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                h = self.base(**enc_pt).last_hidden_state
                pooled = mean_pool(h, enc_pt["attention_mask"])
            prob_attack = float(self.lr.predict_proba(pooled.cpu().numpy())[0, 1])

        prediction = "ATTACK" if prob_attack >= self.attack_threshold else "NORMAL"
        confidence = round(
            float(prob_attack if prob_attack >= self.attack_threshold else 1.0 - prob_attack), 4
        )
        return prediction, confidence


if __name__ == "__main__":
    p = HTTPAttackPredictor()
    print(p.predict("GET /?id=1' OR '1'='1"))
