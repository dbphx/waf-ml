# HTTP Attack Detection ML Model

This project implements a machine learning model to detect HTTP attacks like SQL Injection, Cross-Site Scripting (XSS), Command Injection, and Path Traversal.

## Project Structure

```
.
├── data/
│   ├── raw/             # Raw input data
│   └── processed/       # Preprocessed train/val/test splits
├── src/
│   ├── preprocessing.py      # Mandatory cleaning and data splitting
│   ├── feature_engineering.py # TF-IDF + statistical features
│   ├── train.py              # Model training (Random Forest)
│   ├── evaluate.py           # Robustness and metrics reporting
│   ├── predict.py            # Inference script
│   └── test_samples.py       # Mixed sample testing script
├── models/                   # Exported model and vectorizer
├── reports/                  # Confusion matrix and metrics.json
├── requirements.txt
└── README.md
```

## Setup

1. **Create and activate a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Preprocess Data
Initializes the splits and applies cleaning rules.
```bash
python3 src/preprocessing.py
```

### 2. Train Model
Fits the model and saves it to `models/`.
```bash
python3 src/train.py
```

### 3. Evaluate Model
Generates reports and runs robustness tests.
```bash
python3 src/evaluate.py
```

### 4. Run Inference
Use the inference script to predict attacks from raw strings.
```bash
python3 src/predict.py "GET /admin?id=1' OR '1'='1"
```

### 5. Run Mixed Samples
Test a predefined set of mixed normal and attack samples.
```bash
python3 src/test_samples.py
```

## Mandatory Preprocessing Rules
The model applies the following rules before inference:
- Lowercase all text
- Double URL decoding
- HTML entity decoding
- Number normalization (`<NUM>`)
- Hex normalization (`<HEX>`)
- Whitespace normalization

## Performance Metrics
- **Accuracy**: ~99.6%
- **Recall**: 100%
- **False Positive Rate**: < 1% (on test set)

> [!IMPORTANT]
> Always ensure the virtual environment is activated (`source venv/bin/activate`) before running any scripts, otherwise you may encounter `ModuleNotFoundError`.
