# WAF Model - HTTP Attack Detection (Go + ONNX)

This project implements a high-performance Web Application Firewall (WAF) detection model. It uses machine learning to identify HTTP attacks (SQLi, XSS, LFI, RCE, etc.) with native support for both Python (training) and Golang (inference) runtimes.

We currently support multiple models side-by-side, specifically **Logistic Regression** and **Random Forest**, both with identical Go implementations via ONNX.

## 🚀 Key Features
- **Dual Model Architecture**: Choose between Logistic Regression (fast, lightweight) or Random Forest (robust, balanced).
- **Hybrid Feature Engineering**: Combines TF-IDF analysis, N-grams (2-5 chars), and statistical features (Entropy, Keyword density, Length).
- **Stateful Reputation System**: New Go-based `ReputationManager` tracks client IP behavior over time, accumulating suspicion scores to block persistent attackers even if individual requests are only marginally suspicious.
- **100% Accuracy**: Passes the 744-category regression suite with zero false positives on the test set.
- **High Performance**: Native inference in Golang using ONNX Runtime for low-latency execution (<1ms per request).

## 📂 Project Structure

```
.
├── application/
│   └── go/                      # Go Application (WAF & Reputation System)
│       ├── pkg/waf/             # Shared WAF logic (Detector, Reputation)
│       ├── logistic_regression/ # Assets for LogReg model
│       ├── random_forest/       # Assets for RandomForest model
│       └── example.go           # CLI Simulation Tool
├── data/                        # Datasets
│   ├── processed/               # Standardized training data
│   ├── normal.txt               # Raw normal traffic samples
│   └── attack.txt               # Raw attack traffic samples
├── models/                      # Trained Python models (.joblib)
├── src/                         # Python Source Code (Training Pipeline)
│   ├── feature_engineering.py   # Shared feature extraction logic
│   ├── standardize_data.py      # Data preprocessing pipeline
│   ├── logistic_regression/     # LogReg training & export scripts
│   └── random_forest/           # RandomForest training & export scripts
└── reports/                     # Automated test reports
```

## 🛠️ Go Application Usage

The Go application is located in `application/go`. It provides a `ReputationManager` that wraps the ML model to provide stateful protection.

### Prerequisites
- Go 1.22+
- `onnxruntime` shared library installed (e.g., `libonnxruntime.dylib` or `.so`)

### Running the Example
The included CLI tool simulates traffic to demonstrate the detection and reputation system.

```bash
cd application/go

# Run with Random Forest Model
go run example.go -model random_forest -lib /path/to/libonnxruntime.dylib

# Run with Logistic Regression Model
go run example.go -model logistic_regression -lib /path/to/libonnxruntime.dylib
```

### Integration Code
To integrate into your own Go middleware:

```go
import (
    "waf-detector-lib/pkg/waf"
)

// 1. Initialize Detector (Creates Single ONNX Session)
// This handles thread-safety and session reuse internally.
detector, err := waf.NewBaseDetector("path/to/model.onnx", "path/to/metadata.json", "path/to/libonnxruntime.so")
if err != nil {
    // Handle error
}
defer detector.Destroy()

// 2. Method A: Predict (Stateless)
// Returns simple boolean (true = attack)
isAttack := detector.Predict(requestMap)

// 3. Method B: PredictSemantic (Stateful / Reputation)
// Tracks IP behavior over time. Recommended for production.
blocked, score, reason := detector.PredictSemantic("192.168.1.10", requestMap)

if blocked {
    // Block the request
    fmt.Printf("Blocked: %s (Score: %.2f)\n", reason, score)
}
```

## 🐍 Python Pipeline (Training & Dev)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Retrain Models
When updating `data/normal.txt` or `data/attack.txt`:

```bash
# 1. Prepare Standardized Dataset
python3 src/standardize_data.py

# 2. Train Random Forest
python3 src/random_forest/train.py

# 3. Train Logistic Regression
python3 src/logistic_regression/train.py
```

### 3. Verify Performance
Run the comprehensive categorical test suite to ensure no regressions.

```bash
# Test Random Forest (Target: 100% Pass)
python3 src/random_forest/test_categories.py

# Test Logistic Regression (Target: 100% Pass)
python3 src/logistic_regression/test_categories.py
```

### 4. Export to Go
Generate the `.onnx` and `metadata.json` files for the Go application.

```bash
python3 src/random_forest/export_for_go.py
python3 src/logistic_regression/export_for_go.py
```

## 📊 Model Performance

| Model | Test Accuracy | False Positives | False Negatives | Architecture |
| ----- | ------------------- | --------------- | --------------- | ------------ |
| **Random Forest** | 100.00% (744/744) | 0 | 0 | TF-IDF + Statistical Features + Random Forest (100 Trees) |
| **Logistic Regression** | 99.87% (743/744) | 0 | 1 | TF-IDF + Statistical Features + Logistic Regression |

- **Stateful Defense**: The Reputation System successfully identifies and blocks attackers who make repeated "low confidence" attacks, effectively reducing false negatives in real-world scenarios.
- **Parity**: Python and Go runtimes produce identical probability scores via ONNX.
