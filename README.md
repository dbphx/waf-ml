# WAF Model - HTTP Attack Detection (Go + ONNX)

This project implements a high-performance Web Application Firewall (WAF) detection model. It uses machine learning to identify HTTP attacks (SQLi, XSS, LFI, etc.) with native support for both Python and Golang runtimes.

We currently support multiple models side-by-side, specifically **Logistic Regression** and **Random Forest**, both with identical Go implementations via ONNX.

## Core Achievements
- **100% Accuracy**: Passes the 210-category regression suite and 16 manual samples with zero false positives.
- **Native Go Support**: Native inference system implemented in Golang using ONNX Runtime for low-latency execution.
- **Bias Resilient**: Cleanly identifies root paths, short URIs, JWT tokens, and complex JSON as NORMAL traffic.

## Project Structure

```
.
├── README.md
├── application/
│   └── go/                      # REUSABLE WAF Libraries
│       ├── bert_uncased/        # Go-native DistilBERT detector library
│       ├── logistic_regression/ # Go-native LogReg detector library
│       └── random_forest/       # Go-native RandomForest detector library
├── data/                        # Datasets (attack.txt, normal.txt)
│   ├── processed/               # Standardized split data
│   └── raw/
├── models/                      # Saved models & vectorizers
│   ├── bert_uncased/
│   │   ├── config.json
│   │   ├── model.safetensors
│   │   └── vocab.txt
│   ├── logistic_regression/
│   │   ├── model.joblib
│   │   └── vectorizer.joblib
│   └── random_forest/
│       ├── model.joblib
│       └── vectorizer.joblib
└── src/                         # Source Code
    ├── feature_engineering.py   # Shared ML components
    ├── preprocessing.py         # Data processing logic
    ├── standardize_data.py      # Data preparation pipeline
    ├── test_samples.py          # Unified CLI tester
    ├── bert_uncased/            # DistilBERT scripts
    │   ├── check_parity.py
    │   ├── export_for_go.py
    │   ├── predict.py
    │   ├── test_categories.py
    │   └── train.py
    ├── logistic_regression/     # LogReg scripts
    │   ├── evaluate.py
    │   ├── export_for_go.py
    │   ├── predict.py
    │   ├── test_categories.py
    │   └── train.py
    └── random_forest/           # Random Forest scripts
        ├── check_parity.py      
        ├── export_for_go.py
        ├── predict.py
        ├── test_categories.py
        └── train.py
```

## Golang Library Usage (Recommended)

The libraries at **`application/go/logistic_regression`** and **`application/go/random_forest`** are the recommended ways to integrate the WAF into your Go applications. 

They share the exact same API surface:

```go
import "waf-detector-lib" // Example import, adjust based on your go.mod

// Initialize the detector
detector, err := random_forest.NewDetector(modelPath, metaPath, sharedLibPath)

// Predict from a map of request components
request := map[string]string{
    "path":  "/api/v1/user",
    "query": "id=1' OR '1'='1",
}
isAttack := detector.Predict(request)
```

To run the provided Go examples:
```bash
cd application/go/random_forest/example
go run main.go
```

## Python Usage (Development)

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Retrain and Export
When updating data in `data/attack.txt` or `data/normal.txt`:

```bash
# Prepare common standardized dataset
python3 src/standardize_data.py

# Train and evaluate specific model
python3 src/random_forest/train.py

# Export to ONNX for Go
python3 src/random_forest/export_for_go.py
```

### 3. Verify Categories and Samples
To run the automated category checks:
```bash
python3 src/random_forest/test_categories.py
```

To run the manual test samples suite, you can pass the model flag:
```bash
python3 src/test_samples.py --model random_forest
```

## Model Comparison & Performance Metrics

| Model | Regression Accuracy | False Positives | False Negatives | Architecture |
| ----- | ------------------- | --------------- | --------------- | ------------ |
| **Logistic Regression** | 100.00% | 0 (0.00%) | 0 (0.00%) | TF-IDF + Logistic Regression (Lightweight, Fastest) |
| **Random Forest** | 100.00% | 0 (0.00%) | 0 (0.00%) | TF-IDF + Random Forest (Balanced) |
| **DistilBERT (bert_uncased)** | 98.36% | 6 (3.87%) | 0 (0.00%) | Transformer (High contextual understanding) |

- **Sample Accuracy**: Passes the manual samples with expected detections.
- **Parity**: Python and Go runtime predictions match consistently via ONNX Runtime deployments.

