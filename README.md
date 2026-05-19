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
- **Field-aware inference**: Requests are split into `path`, `query`, `headers`, and `body`; each field is scored independently and the highest-risk field becomes the final decision (Python `predict_components()`; Go `PredictScore()` uses the same multipart layout internally).
- **Regression suite**: Field-level categories in `data/attack_fields.txt` and `data/normal_fields.txt` (generated from `data/attack.txt` / `data/normal.txt`), tested as RAW + URL-encoded (ENC) pairs.
- **Go runtime**: TF-IDF ONNX models load via `onnxruntime` with shared `model_metadata.json` (`field_order`, per-field vocabularies + IDF, keywords).
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
│   ├── attack.txt              # Category list (attacks) — source for training + field expansion
│   ├── normal.txt              # Category list (benign)
│   ├── attack_fields.txt       # Field-level attack cases (generated; used by test_categories.py)
│   └── normal_fields.txt       # Field-level benign cases (generated)
├── models/
│   ├── random_forest/
│   └── logistic_regression/
├── reports/                    # JSON + optional test logs
└── src/
    ├── feature_engineering.py
    ├── standardize_data.py
    ├── parse_category_files.py
    ├── preprocessing.py
    ├── random_forest/
    ├── logistic_regression/
    ├── test_samples.py
    ├── test_specific_payload.py
    └── test_holdout_regression.py
```

---

## Quick start (new contributors)

### 1. Virtual environment (venv)

Luôn làm việc **trong venv** để dependencies không lẫn với Python hệ thống.

**Tạo venv** (một lần, tại thư mục gốc repo):

```bash
cd /path/to/ml
python3 -m venv venv
```

**Kích hoạt venv** — chọn đúng shell của bạn:

| Môi trường | Lệnh |
| ---------- | ---- |
| **macOS / Linux** (bash, zsh) | `source venv/bin/activate` |
| **Windows CMD** | `venv\Scripts\activate.bat` |
| **Windows PowerShell** | `venv\Scripts\Activate.ps1` |

Sau khi activate, prompt thường có tiền tố `(venv)`. Kiểm tra:

```bash
which python    # macOS/Linux — phải trỏ vào .../ml/venv/bin/python
python -V
pip -V
```

**Cài dependency** (chỉ khi venv đang bật):

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Chạy script trong venv** — dùng `python` (đã trỏ vào `venv`), ví dụ:

```bash
python src/random_forest/test_categories.py --onnx
```

**Thoát venv** khi xong:

```bash
deactivate
```

**Lưu ý:** không commit thư mục `venv/`; mỗi máy tự tạo và `pip install -r requirements.txt`. Dùng **Python 3.9+** (repo ghim `scikit-learn>=1.6.1,<1.8` cho joblib).

### 2. Train data pipeline (when you change `attack.txt` / `normal.txt`)

`standardize_data.py` regenerates field-level regression files and processed CSVs:

```bash
python src/standardize_data.py
python src/random_forest/train.py
python src/logistic_regression/train.py
```

To regenerate only the field-level category files:

```bash
python src/parse_category_files.py
```

### 3. Export ONNX for Go (TF-IDF models)

```bash
python src/random_forest/export_for_go.py
python src/logistic_regression/export_for_go.py
```

Each export refreshes:

- `application/go/<model>/assets/model.onnx`
- `application/go/<model>/assets/model_metadata.json`
- `models/<model>/model.onnx`

Current ONNX layout for both TF-IDF bundles:

- `field_order = ["path", "query", "headers", "body"]`
- one TF-IDF vectorizer per field (`max_features = 5000` each)
- per-field stats = `length + entropy + 26 keyword ratios`
- total input width = `20112` features (`20000` TF-IDF + `112` stats)

---

## Running tests (modes)

All commands assume repository root and **`source venv/bin/activate`** (venv đang bật).

### Categorical regression (`test_categories.py`)

Tests read **`data/attack_fields.txt`** and **`data/normal_fields.txt`** (not the legacy whole-request lines in `attack.txt` / `normal.txt` directly). Each category is exercised as **RAW** and **ENC** (URL-encoded components), for **1292** cases total.

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

### Python prediction behavior

`src/*/predict.py` normalizes input with `split_request_components()`, then scores `path`, `query`, `headers`, and `body` separately via `predict_components()`. The component with the highest attack probability is the final verdict (`decisive_component`).

Debug a single payload:

```bash
python src/test_specific_payload.py
```

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

Optional payload from a line in a `.txt` file:

```bash
go run example.go -model random_forest -lib /path/to/libonnxruntime.dylib \
  -payload-file ../../data/attack.txt -payload-contains "SQL Injection"
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

score, _ := detector.PredictScore(requestMap) // max over per-field scores
isAttack := detector.Predict(requestMap)
blocked, score, reason := detector.PredictSemantic("192.168.1.10", requestMap)
```

`pkg/waf` matches Python multipart inference: for each non-empty field in `field_order`, it builds a feature row with only that field set, runs ONNX, and uses the **maximum** attack probability as `PredictScore`. There is no exported `PredictComponents` API yet—only the aggregated score/threshold helpers above.

Current decision thresholds:

- `random_forest`: stateless attack threshold `0.55`
- `logistic_regression`: stateless attack threshold `0.77`
- Go `PredictSemantic()` example defaults: RF block/suspicion = `0.55 / 0.35`, LR block/suspicion = `0.77 / 0.50`

---

## Model performance (reference)

Latest **ONNX** categorical regression (`attack_fields.txt` + `normal_fields.txt`, RAW + ENC, **1292** cases):

| Model | Categorical regression | Known misses (ONNX) |
| ----- | ---------------------- | ------------------- |
| **Random Forest** | `1288 / 1292` (`99.69%`) | `Slowloris Header Pattern [path]` (RAW/ENC), `PADDED_XSS [path]` (RAW/ENC) |
| **Logistic Regression** | `1283 / 1292` (`99.30%`) | Same Slowloris misses; `Attack_PDF_34 [path]` (ENC); false positives on `FP_USER_57 [query]`, `FP_USER_60 [path]`, `Benign Issue Collection Path [path]` (RAW/ENC) |

Re-run after export to refresh numbers:

```bash
python src/random_forest/test_categories.py --onnx
python src/logistic_regression/test_categories.py --onnx
```

Hold-out files (`data/holdout_attack.txt`, `holdout_normal.txt`) are intentionally **excluded** from `train.py` merges — use them to sanity-check generalization beyond the main regression lists.

Repository agents should also read **`AGENTS.md`** (build commands, beads, and constraints).
