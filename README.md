# WAF Model - HTTP Attack Detection (Go + ONNX)

This project implements a high-performance Web Application Firewall (WAF) detection model. It uses machine learning to identify HTTP attacks (SQLi, XSS, LFI, RCE, etc.) with native support for both Python (training) and Golang (inference) for the **TF-IDF + sklearn** models.

We support two Python model bundles side-by-side:

| Bundle | Inference (Python) | Go `pkg/waf` ONNX |
| ------ | ------------------ | ----------------- |
| **Random Forest** | joblib or ONNX | Yes — TF-IDF `float_input` |
| **Logistic Regression** | joblib or ONNX | Yes — TF-IDF `float_input` |

## Key features

- **Multiple models**: Random Forest and Logistic Regression (TF-IDF + stats).
- **Hybrid features (TF-IDF models)**: Character n-grams (2–5), TF-IDF, entropy, keyword ratios.
- **Regression suite**: Payloads from `data/attack.txt` and `data/normal.txt` (RAW + URL-encoded) — target **100%** pass on categorical tests for production parity.
- **Go runtime**: TF-IDF ONNX models load via `onnxruntime` with shared `model_metadata.json` (vocabulary + IDF + keywords).
- **Stateful reputation** (Go): `ReputationManager` scores clients over time.

## Project structure

```
.
├── application/go/
│   ├── pkg/waf/                  # BaseDetector (TF-IDF ONNX), reputation
│   ├── random_forest/assets/     # model.onnx, model_metadata.json
│   ├── logistic_regression/assets/
│   └── example.go
├── data/
│   ├── processed/              # train.csv, val.csv (from standardize_data.py)
│   ├── attack.txt              # Labeled attack lines for regression tests + train merge
│   └── normal.txt
├── models/
│   ├── random_forest/
│   ├── logistic_regression/
├── reports/                    # JSON + optional test logs
└── src/
    ├── feature_engineering.py
    ├── standardize_data.py
    ├── parse_category_files.py
    ├── random_forest/
    ├── logistic_regression/
    ├── test_samples.py
    └── test_holdout_regression.py
```

---

## Quick start (new contributors)

### 1. Virtual environment (venv)

Luôn làm việc **trong venv** để dependencies không lẫn với Python hệ thống.

**Tạo venv** (một lần, tại thư mục gốc repo):

```bash
cd /path/to/ml
python3 -m venv .venv
```

**Kích hoạt venv** — chọn đúng shell của bạn:

| Môi trường | Lệnh |
| ---------- | ---- |
| **macOS / Linux** (bash, zsh) | `source .venv/bin/activate` |
| **Windows CMD** | `.venv\Scripts\activate.bat` |
| **Windows PowerShell** | `.venv\Scripts\Activate.ps1` |

Sau khi activate, prompt thường có tiền tố `(.venv)`. Kiểm tra:

```bash
which python    # macOS/Linux — phải trỏ vào .../ml/.venv/bin/python
python -V
pip -V
```

**Cài dependency** (chỉ khi venv đang bật):

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Chạy script trong venv** — dùng `python` (đã trỏ vào `.venv`), ví dụ:

```bash
python src/random_forest/test_categories.py --onnx
```

**Thoát venv** khi xong:

```bash
deactivate
```

**Lưu ý:** không commit thư mục `.venv`; mỗi máy tự tạo và `pip install -r requirements.txt`. Dùng **Python 3.9+** (repo ghim `scikit-learn>=1.6.1,<1.8` cho joblib).

### 2. Train data pipeline (when you change `attack.txt` / `normal.txt`)

```bash
python src/standardize_data.py
python src/random_forest/train.py
python src/logistic_regression/train.py
```

### 3. Export ONNX for Go (TF-IDF models)

```bash
python src/random_forest/export_for_go.py
python src/logistic_regression/export_for_go.py
```


---

## Running tests (modes)

All commands assume repository root and **`source .venv/bin/activate`** (venv đang bật). Targets are typically **882/882** lines (category × RAW + ENC) when models are trained and thresholds aligned.

### Categorical regression (`test_categories.py`)

| Mode | Flags | What it uses |
| ---- | ----- | ------------- |
| **joblib (default)** | *(none)* | `models/<name>/model.joblib` + vectorizer |
| **ONNX** | `--onnx` | `application/go/<name>/assets/model.onnx` (fallback: `models/<name>/model.onnx`) |
| **Save log** | `--log PATH` | Mirror stdout to a file (table + SUMMARY) |

```bash
# Random Forest — ONNX + log
python src/random_forest/test_categories.py --onnx --log reports/random_forest/categorical_test_onnx.log

# Logistic Regression — ONNX + log
python src/logistic_regression/test_categories.py --onnx --log reports/logistic_regression/categorical_test_onnx.log

# Same without ONNX (joblib)
python src/random_forest/test_categories.py --log reports/random_forest/categorical_test_joblib.log
python src/logistic_regression/test_categories.py --log reports/logistic_regression/categorical_test_joblib.log
```

JSON reports are written under `reports/<model>/categorical_results.json` or `categorical_results_onnx.json` when `--onnx` is used.

### Quick interactive samples

```bash
python src/test_samples.py --model random_forest
python src/test_samples.py --model logistic_regression
```

### Hold-out set (`data/holdout_*.txt` — not merged into training)

```bash
python src/test_holdout_regression.py --model both          # RF + LR only
python src/test_holdout_regression.py --model all
```

---

## Go application usage

Located in `application/go`. Random Forest and Logistic Regression use `waf.NewBaseDetector` (TF-IDF ONNX).

### Prerequisites

- Go 1.22+
- ONNX Runtime shared library (`libonnxruntime.dylib` / `.so`)

### Run example

```bash
cd application/go

go run example.go -model random_forest -lib /path/to/libonnxruntime.dylib
go run example.go -model logistic_regression -lib /path/to/libonnxruntime.dylib
```

### Integration (TF-IDF ONNX)

```go
import (
    "waf-detector-lib/pkg/waf"
)

detector, err := waf.NewBaseDetector("path/to/model.onnx", "path/to/metadata.json", "path/to/libonnxruntime.so")
if err != nil {
    // Handle error
}
defer detector.Destroy()

isAttack := detector.Predict(requestMap)
blocked, score, reason := detector.PredictSemantic("192.168.1.10", requestMap)
```

---

## Model performance (reference)

| Model | Categorical regression | Notes |
| ----- | ---------------------- | ----- |
| **Random Forest** | 100% target on suite | TF-IDF + RF; thresholds in predict / Go |
| **Logistic Regression** | 100% target on suite | TF-IDF + LR; threshold ~0.72 (Python) |

Hold-out files (`data/holdout_attack.txt`, `holdout_normal.txt`) are intentionally **excluded** from `train.py` merges — use them to sanity-check generalization beyond the main regression lists.

Repository agents should also read **`AGENTS.md`** (build commands, beads, and constraints).
