"""Build end-to-end ONNX: SecureBERT encoder + mean pool + sklearn LR as Linear (logit for positive class)."""
import json
import os
import sys

import joblib
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden)
    summed = (last_hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    return summed / denom


class SecureBERT2ClassifierOnnx(nn.Module):
    def __init__(self, base: nn.Module, head: nn.Module):
        super().__init__()
        self.base = base
        self.head = head

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        h = self.base(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        pooled = mean_pool(h, attention_mask)
        return self.head(pooled)


def linear_from_sklearn(lr, hidden_size: int) -> nn.Linear:
    lin = nn.Linear(hidden_size, 1)
    w = lr.coef_.astype(np.float32)
    b = lr.intercept_.astype(np.float32)
    if w.shape != (1, hidden_size):
        raise ValueError(f"LR coef_ shape {w.shape} != (1, {hidden_size})")
    lin.weight.data = torch.from_numpy(w)
    lin.bias.data = torch.from_numpy(b)
    return lin


def export_end_to_end_onnx(
    models_dir: str,
    out_path: str,
    max_len: int,
    opset: int = 14,
) -> None:
    enc_dir = os.path.join(models_dir, "encoder")
    with open(os.path.join(enc_dir, "config.json")) as f:
        hidden = int(json.load(f)["hidden_size"])

    lr = joblib.load(os.path.join(models_dir, "head.joblib"))
    head = linear_from_sklearn(lr, hidden)

    base = AutoModel.from_pretrained(enc_dir, trust_remote_code=True)
    full = SecureBERT2ClassifierOnnx(base, head)
    full.eval()
    full.cpu()

    dummy_ids = torch.zeros(1, max_len, dtype=torch.long)
    dummy_mask = torch.ones(1, max_len, dtype=torch.long)

    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    torch.onnx.export(
        full,
        (dummy_ids, dummy_mask),
        out_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logit"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "logit": {0: "batch"},
        },
        opset_version=opset,
        do_constant_folding=True,
    )
    print(f"Wrote ONNX: {out_path}")


def verify_onnx_vs_sklearn(models_dir: str, onnx_path: str, max_len: int) -> None:
    import onnxruntime as ort
    from transformers import AutoTokenizer

    lr = joblib.load(os.path.join(models_dir, "head.joblib"))
    tokenizer = AutoTokenizer.from_pretrained(models_dir, trust_remote_code=True)
    base = AutoModel.from_pretrained(os.path.join(models_dir, "encoder"), trust_remote_code=True)
    base.eval()
    text = "GET /search?q=1' OR '1'='1"
    enc = tokenizer(
        text, truncation=True, max_length=max_len, padding="max_length", return_tensors="pt"
    )
    with torch.no_grad():
        h = base(**enc).last_hidden_state
        pooled = mean_pool(h, enc["attention_mask"])
        logit_sk = float(lr.decision_function(pooled.numpy())[0])

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    enp = tokenizer(
        text, truncation=True, max_length=max_len, padding="max_length", return_tensors="np"
    )
    logit_ort = float(
        np.array(
            sess.run(
                None,
                {
                    "input_ids": enp["input_ids"].astype(np.int64),
                    "attention_mask": enp["attention_mask"].astype(np.int64),
                },
            )[0]
        ).ravel()[0]
    )
    diff = abs(logit_sk - logit_ort)
    ok = diff < 0.05
    print(f"ONNX check: sklearn_logit={logit_sk:.6f} onnx_logit={logit_ort:.6f} |diff|={diff:.6f} OK={ok}")
