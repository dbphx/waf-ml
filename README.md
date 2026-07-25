# WAF Model - HTTP Attack Detection (Go + ONNX)

This project implements a high-performance Web Application Firewall (WAF) detection model. It uses machine learning to identify HTTP attacks (SQLi, XSS, LFI, RCE, etc.) with native support for both Python (training) and Golang (inference) for the **TF-IDF + sklearn** models.

We support three Python model bundles side-by-side:

| Bundle | Inference (Python) | Go `pkg/waf` ONNX |
| ------ | ------------------ | ----------------- |
| **Random Forest** | joblib or ONNX | Yes — `RandomForestModel` |
| **Logistic Regression** | joblib or ONNX | Yes — `LogisticRegressionModel` |
| **XGBoost** | joblib or ONNX | Yes — `XGBoostModel` (zero features → `NaN`) |

## Key features

- **Multiple models**: Random Forest, Logistic Regression, and XGBoost (multipart TF-IDF + per-field stats).
- **Hybrid features (TF-IDF models)**: Character n-grams (2–5), TF-IDF, entropy, keyword ratios.
- **Field-aware inference**: Requests are split into `path`, `query`, `headers`, and `body`; each field is scored independently and the highest-risk field becomes the final decision (Python `predict_components()`; Go `PredictScore()` uses the same multipart layout internally).
- **Regression suite**: Field-level categories in `data/attack_fields.txt` and `data/normal_fields.txt` (generated from `data/attack.txt` / `data/normal.txt`), tested as RAW + URL-encoded (ENC) pairs.
- **Go runtime**: Each model implements the `waf.Model` interface; ONNX loads via `onnxruntime` with shared `model_metadata.json` (`field_order`, per-field vocabularies + IDF, keywords).
- **Stateful reputation** (Go): `ReputationManager` scores clients over time.

## Project structure

```
.
├── application/go/
│   ├── pkg/waf/
│   │   ├── model.go              # Model interface, NewModel factory
│   │   ├── engine.go             # Shared ONNX + feature engineering
│   │   ├── random_forest.go      # RandomForestModel
│   │   ├── logistic_regression.go
│   │   ├── xgboost.go            # XGBoostModel (NaN encoding)
│   │   └── reputation.go
│   ├── random_forest/assets/     # model.onnx, model_metadata.json
│   ├── logistic_regression/assets/
│   ├── xgboost/assets/
│   └── example.go
├── data/
│   ├── processed/              # train.csv, val.csv (from standardize_data.py)
│   ├── attack.txt              # Category list (attacks) — source for training + field expansion
│   ├── normal.txt              # Category list (benign)
│   ├── attack_fields.txt       # Field-level attack cases (generated; used by test_categories.py)
│   └── normal_fields.txt       # Field-level benign cases (generated)
├── models/
│   ├── random_forest/
│   ├── logistic_regression/
│   └── xgboost/
├── reports/                    # JSON + optional test logs
└── src/
    ├── feature_engineering.py
    ├── standardize_data.py
    ├── parse_category_files.py
    ├── preprocessing.py
    ├── random_forest/
    ├── logistic_regression/
    ├── xgboost/
    ├── test_samples.py
    ├── test_specific_payload.py
    └── test_holdout_regression.py
```

---

## Quick start

Work inside a local virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Use Python 3.9+. Do not commit `venv/`.

### 2. Train data pipeline (when you change `attack.txt` / `normal.txt`)

`standardize_data.py` regenerates field-level regression files and processed CSVs:

```bash
python src/standardize_data.py
python src/random_forest/train.py
python src/logistic_regression/train.py
python src/xgboost/train.py
```

To regenerate only the field-level category files:

```bash
python src/parse_category_files.py
```

### 3. Export ONNX for Go (TF-IDF models)

```bash
python src/random_forest/export_for_go.py
python src/logistic_regression/export_for_go.py
python src/xgboost/export_for_go.py   # uses onnxmltools (not skl2onnx)
```

RF and LR export via `skl2onnx`. XGBoost uses `onnxmltools.convert_xgboost` because `skl2onnx` does not support `XGBClassifier`.

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

## Running tests

All commands assume repository root with a Python virtual environment activated. The latest local run used `.venv/bin/python` because the system `python3` environment did not have `pandas` installed.

The maintained regression entry points are `test_categories.py`, `test_holdout_regression.py`, `test_samples.py`, and `test_specific_payload.py`. Legacy one-off logistic regression helper scripts have been removed.

### Categorical regression (`test_categories.py`)

Tests read `data/attack_fields.txt` and `data/normal_fields.txt`. Each category is exercised as both `RAW` and `ENC` (URL-encoded components), for `1512` cases total in the current suite. The current benign header regressions include both `Benign Fluent Bit Splunk Headers` and `Benign Fluent Bit Short Headers`.

| Mode | Flags | Uses |
| ---- | ----- | ---- |
| `joblib` | none | `models/<name>/model.joblib` + vectorizer |
| `onnx` | `--onnx` | `application/go/<name>/assets/model.onnx` (fallback: `models/<name>/model.onnx`) |
| `log` | `--log PATH` | mirrors stdout to a file |

```bash
# ONNX
python src/random_forest/test_categories.py --onnx --log reports/random_forest/categorical_test_onnx.log
python src/logistic_regression/test_categories.py --onnx --log reports/logistic_regression/categorical_test_onnx.log
python src/xgboost/test_categories.py --onnx --log reports/xgboost/categorical_test_onnx.log

# joblib
python src/random_forest/test_categories.py --log reports/random_forest/categorical_test_joblib.log
python src/logistic_regression/test_categories.py --log reports/logistic_regression/categorical_test_joblib.log
python src/xgboost/test_categories.py --log reports/xgboost/categorical_test_joblib.log
```

JSON reports are written under `reports/<model>/categorical_results.json` or `categorical_results_onnx.json` when `--onnx` is used.

XGBoost ONNX parity check (sparse sklearn vs ONNX with `NaN` for zero features):

```bash
python src/xgboost/check_parity.py
```

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
python src/test_samples.py --model xgboost
```

### Hold-out set (`data/holdout_*.txt` — not merged into training)

```bash
python src/test_holdout_regression.py --model both          # RF + LR only
python src/test_holdout_regression.py --model all
```

---

## Go application usage

Located in `application/go`. All three models implement the `waf.Model` interface and are created via `waf.NewModel` or `waf.NewModelWithConfig`.

### Prerequisites

- Go 1.22+
- ONNX Runtime shared library (`libonnxruntime.dylib` / `.so`)

### Run example

```bash
cd application/go

go run example.go -model random_forest -lib /path/to/libonnxruntime.dylib
go run example.go -model logistic_regression -lib /path/to/libonnxruntime.dylib
go run example.go -model xgboost -lib /path/to/libonnxruntime.dylib
```

Optional payload from a line in a `.txt` file:

```bash
go run example.go -model random_forest -lib /path/to/libonnxruntime.dylib \
  -payload-file ../../data/attack.txt -payload-contains "SQL Injection"
```

### Integration (`waf.Model` interface)

```go
import "waf-detector-lib/pkg/waf"

cfg, err := waf.DefaultConfig(waf.ModelRandomForest)
if err != nil {
    // handle unknown model type
}

detector, err := waf.NewModelWithConfig(cfg, "/path/to/libonnxruntime.so")
if err != nil {
    // handle init error
}
defer detector.Destroy()

requestMap := map[string]string{
    "method": "GET",
    "path":   "/search",
    "query":  "q=test",
}

score, _ := detector.PredictScore(requestMap) // max over per-field scores
isAttack := detector.Predict(requestMap)
blocked, score, reason := detector.PredictSemantic("192.168.1.10", requestMap)
```

Factory shorthand:

```go
detector, err := waf.NewModel(waf.ModelXGBoost, "xgboost/assets", sharedLibPath)
```

| Type constant | Go struct | Stateless threshold | Semantic (block / suspicion) |
| ------------- | --------- | ------------------- | ---------------------------- |
| `waf.ModelRandomForest` | `RandomForestModel` | `0.55` | `0.55` / `0.35` |
| `waf.ModelLogisticRegression` | `LogisticRegressionModel` | `0.77` | `0.77` / `0.50` |
| `waf.ModelXGBoost` | `XGBoostModel` | `0.55` | `0.55` / `0.35` |

`pkg/waf` matches Python multipart inference: for each non-empty field in `field_order`, it builds a feature row with only that field set, runs ONNX, and uses the **maximum** attack probability as `PredictScore`.

**XGBoost ONNX note:** XGBoost treats absent sparse entries as *missing*, not zero. Go and Python ONNX inference encode zero-valued features as `NaN` so categorical `joblib` and ONNX regression currently align. Exact probability parity on `src/xgboost/check_parity.py` is still not guaranteed and should be treated as an open export-quality issue. RF and LR use dense `0.0` for absent features.

---

## Model performance

Latest categorical regression refresh, run on **July 25, 2026** against `data/attack_fields.txt` + `data/normal_fields.txt` (`1512` RAW + ENC evaluations):

| Model | Backend | Result | Failures | Notes |
| ----- | ------- | ------ | -------- | ----- |
| **Random Forest** | joblib | `1485 / 1512` (`98.21%`) | 27 | 26 false negatives, 1 false positive |
| **Random Forest** | ONNX | `1485 / 1512` (`98.21%`) | 27 | Matches joblib outcome |
| **Logistic Regression** | joblib | `1480 / 1512` (`97.88%`) | 32 | 30 false negatives, 2 false positives |
| **Logistic Regression** | ONNX | `1480 / 1512` (`97.88%`) | 32 | Matches joblib outcome |
| **XGBoost** | joblib | `1484 / 1512` (`98.15%`) | 28 | 19 false negatives, 9 false positives |
| **XGBoost** | ONNX | `1485 / 1512` (`98.21%`) | 27 | 26 false negatives, 1 false positive |

Current Random Forest misses: `Slowloris Header Pattern [path]`, `TestReal1 [path]`, `TestReal2 [path]`, `Attack_PDF_105 [path]`, `Attack_usr_135-136 [path]`, `Attack_FP_137 [path]`, `Attack_usr_138-139 [path]`, `PADDED_XSS [path]`, `Attack_Asset_1-4 [path]`, `Attack_Analyzer_Combined_XSS_SQLi_HTML [path]`, plus one false positive on `Benign Asset [path]`.

Current Logistic Regression misses: `Slowloris Header Pattern [path]`, `TestReal1 [path]`, `TestReal2 [path]`, `Attack_PDF_105 [path]`, `Attack_usr_135-136 [path]`, `Attack_FP_137 [path]`, `Attack_usr_138-139 [path]`, `PADDED_XSS [path]`, `Attack_Asset_1-4 [path]`, `Attack_Analyzer_Combined_XSS_SQLi_HTML [path]`, plus false positives on `FP_USER_55 [query]` and `FP_USER_57 [query]`.

Current XGBoost joblib misses: `Slowloris Header Pattern [path]`, `Attack_PDF_105 [path]`, `Attack_usr_135-136 [path]`, `Attack_FP_137 [path]`, `Attack_usr_138-139 [path]`, `PADDED_XSS [path]`, `Attack_Asset_3-4 [path]`, `Attack_Analyzer_Combined_XSS_SQLi_HTML [path]`, plus false positives on `FP1 [path]`, `FP2 [path]`, `Benign Asset [path]`, `FP_USER_59 [path]`, `FP_USER_60 [path]`, and `Benign Issue Collection Path [path]`.

Current XGBoost ONNX misses: `Slowloris Header Pattern [path]`, `TestReal1 [path]`, `TestReal2 [path]`, `Attack_PDF_105 [path]`, `Attack_usr_135-136 [path]`, `Attack_FP_137 [path]`, `Attack_usr_138-139 [path]`, `PADDED_XSS [path]`, `Attack_Asset_1-4 [path]`, `Attack_Analyzer_Combined_XSS_SQLi_HTML [path]`, plus one false positive on `Benign Asset [path]`.

The July 25 run emitted `InconsistentVersionWarning` from scikit-learn while unpickling artifacts with mismatched scikit-learn versions. XGBoost joblib also warned about loading an older serialized model with XGBoost `2.1.4`; results should be interpreted with those environment mismatches in mind.

`python src/xgboost/check_parity.py` has historically reported a probability mismatch between sparse `joblib` inference and ONNX on its sample payload; categorical results above confirm the backends still do not have exact outcome parity.

Historical ONNX baselines from **May 26, 2026** against `data/attack_fields.txt` + `data/normal_fields.txt` (`742` field cases, `1484` RAW + ENC evaluations):

| Model | Categorical regression | Current miss pattern |
| ----- | ---------------------- | -------------------- |
| **Random Forest** | `1457 / 1484` (`98.18%`) | Remaining misses are mostly `Slowloris Header Pattern`, `TestReal1/2`, `Attack_PDF_105`, `Attack_usr_135-139`, `PADDED_XSS`, several asset/analyzer attack paths, plus one FP on `Benign Asset [path]` (ENC) |
| **Logistic Regression** | `1426 / 1468` (`97.14%`) | Same core attack misses as RF, plus `Attack_PDF_106`, all `Attack_Asset_1-4` RAW/ENC misses, and FPs on `FP_PDF_48/49`, `FP_USER_55/57`, `Benign Workspace Modules API`, `Benign Sidebar Preferences API` |
| **XGBoost** | `1437 / 1468` (`97.89%`) | Same core attack misses as RF, partial misses on `Attack_Asset_1-4`, one FP on `FP_PDF_34 [path]` (ENC), and FPs on `Benign Fluent Bit Splunk Headers [headers]` (RAW/ENC) |

Re-run after export to refresh numbers:

```bash
python src/random_forest/test_categories.py --onnx
python src/logistic_regression/test_categories.py --onnx
python src/xgboost/test_categories.py --onnx
```

Hold-out files (`data/holdout_attack.txt`, `holdout_normal.txt`) are intentionally **excluded** from `train.py` merges — use them to sanity-check generalization beyond the main regression lists.

Repository agents should also read **`AGENTS.md`** (build commands, beads, and constraints).
