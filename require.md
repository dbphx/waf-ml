# HTTP Attack Detection – ML Agent Guide

This repository defines rules, structure, and requirements for an AI agent to build a machine learning model that detects HTTP attacks such as SQL Injection, XSS, Command Injection, Path Traversal, and obfuscation-based attacks.

The agent MUST follow this document strictly.

---

## 1. Goal

Build a binary classification model:

- Input: Raw HTTP request data
- Output:
  - 0 → NORMAL
  - 1 → ATTACK

The model must generalize beyond known rules and signatures.

---

## 2. Project Structure

.
├── data/
│   ├── raw/
│   │   ├── attack.csv
│   │   └── normal.csv
│   └── processed/
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
│
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── models/
│   └── model.bin
│
├── reports/
│   ├── metrics.json
│   └── confusion_matrix.png
│
├── requirements.txt
└── README.md

---

## 3. Data Format

If multiple HTTP fields exist, they must be merged into one text input.

Required columns:

- http_method
- http_path
- http_query
- http_headers
- http_body
- user_agent
- label (0 = normal, 1 = attack)

---

## 4. Preprocessing Rules (MANDATORY)

The agent MUST apply:

1. Lowercase all text
2. URL decode (1–2 passes)
3. HTML entity decoding
4. Replace numbers with <NUM>
5. Replace hexadecimal patterns with <HEX>
6. Normalize whitespace
7. Preserve special characters:
   ' " ; < > ( ) { } [ ] / \ %

Do NOT remove symbols. Attacks depend on them.

---

## 5. Feature Strategy

The agent may use one or combine multiple strategies.

Option A – Semantic Embeddings (Preferred)
- Transformer-based embeddings
- Robust to obfuscation and encoding tricks

Option B – Classical ML
- Character-level TF-IDF (3–5 grams)
- Logistic Regression or Linear SVM

Option C – Hybrid (Recommended)
- Embeddings + statistical features:
  - Input length
  - Special character counts
  - Entropy
  - Keyword frequency (select, union, script, ../)

---

## 6. Training Requirements

- Train / Validation / Test split: 70 / 15 / 15
- Loss: Binary Cross Entropy
- Optimizer: Adam or AdamW
- Early stopping REQUIRED
- Handle class imbalance using weights or resampling

---

## 7. Evaluation Metrics (ALL REQUIRED)

The agent MUST report:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- False Positive Rate (critical)

High recall with uncontrolled false positives is NOT acceptable.

---

## 8. Robustness Testing (REQUIRED)

The agent must test against:

- Case mutation
  /login → /LoGiN
- Encoding
  %2e%2e%2fetc%2fpasswd
- Obfuscation
  /**/union/**/select
- Benign lookalikes
  /homeSS, /selecta, /scripture

Failures must be documented.

---

## 9. Model Export

The agent must export:

- Trained model file
- Tokenizer or vectorizer
- Label mapping
- Inference script (predict.py)

Inference output format:

{
  "prediction": "ATTACK",
  "confidence": 0.97
}

---

## 10. Forbidden Practices

The agent MUST NOT:

- Rely only on keyword matching
- Memorize fixed paths or payloads
- Leak labels via rule IDs or metadata
- Overfit to training signatures

---

## 11. Deployment Assumptions

- Model is used in a WAF or API Gateway
- Latency target: under 10ms per request
- Supports batch and streaming inference

---

## 12. Definition of Done

The task is complete when:

- Model generalizes to unseen attacks
- False Positive Rate < 3%
- Recall > 95%
- All reports are generated
- Model loads and infers independently

---

## 13. Agent Guidance

HTTP input is adversarial, not natural language.

Robustness > Accuracy  
Semantics > Keywords  
Generalization > Memorization