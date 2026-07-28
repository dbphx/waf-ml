# WAF Model - Phat Hien Tan Cong HTTP (Go + ONNX)

Ngon ngu: [English](README.md) | Tieng Viet

Du an nay trien khai mo hinh phat hien Web Application Firewall (WAF) hieu nang cao. He thong dung machine learning de nhan dien cac tan cong HTTP (SQLi, XSS, LFI, RCE, v.v.) va ho tro ca Python (training) lan Golang (inference) cho cac mo hinh **TF-IDF + sklearn**.

Du an ho tro song song ba goi mo hinh Python:

| Goi | Inference (Python) | Go `pkg/waf` ONNX |
| --- | ------------------ | ----------------- |
| **Random Forest** | joblib hoac ONNX | Co - `RandomForestModel` |
| **Logistic Regression** | joblib hoac ONNX | Co - `LogisticRegressionModel` |
| **XGBoost** | joblib hoac ONNX | Co - `XGBoostModel` (feature bang 0 -> `NaN`) |

## Tinh nang chinh

- **Nhieu mo hinh**: Random Forest, Logistic Regression va XGBoost (multipart TF-IDF + thong ke theo tung field).
- **Feature ket hop (cac mo hinh TF-IDF)**: Character n-grams (2-5), TF-IDF, entropy, ty le keyword.
- **Inference theo field**: Request duoc tach thanh `path`, `query`, `headers`, va `body`; moi field duoc cham diem doc lap, field co rui ro cao nhat tro thanh quyet dinh cuoi cung (Python `predict_components()`; Go `PredictScore()` dung cung bo cuc multipart noi bo).
- **Bo regression**: Cac category cap field nam trong `data/attack_fields.txt` va `data/normal_fields.txt` (sinh tu `data/attack.txt` / `data/normal.txt`), duoc test theo cap RAW + URL-encoded (ENC).
- **Go runtime**: Moi model trien khai interface `waf.Model`; ONNX duoc load qua `onnxruntime` voi `model_metadata.json` dung chung (`field_order`, vocabulary + IDF theo tung field, keywords).
- **Reputation co trang thai** (Go): `ReputationManager` cham diem client theo thoi gian.

## Bat dau nhanh

Lam viec trong virtual environment cuc bo:

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Dung Python 3.9+. Khong commit `venv/`.

### 2. Train pipeline du lieu (khi thay doi `attack.txt` / `normal.txt`)

`standardize_data.py` tao lai cac file regression cap field va cac CSV da xu ly:

```bash
python src/standardize_data.py
python src/random_forest/train.py
python src/logistic_regression/train.py
python src/xgboost/train.py
```

Chi tao lai cac file category cap field:

```bash
python src/parse_category_files.py
```

### 3. Export ONNX cho Go (cac mo hinh TF-IDF)

```bash
python src/random_forest/export_for_go.py
python src/logistic_regression/export_for_go.py
python src/xgboost/export_for_go.py   # uses onnxmltools (not skl2onnx)
```

RF va LR export bang `skl2onnx`. XGBoost dung `onnxmltools.convert_xgboost` vi `skl2onnx` khong ho tro `XGBClassifier`.

Moi lan export se cap nhat:

- `application/go/<model>/assets/model.onnx`
- `application/go/<model>/assets/model_metadata.json`
- `models/<model>/model.onnx`

Bo cuc ONNX hien tai cho ca hai goi TF-IDF:

- `field_order = ["path", "query", "headers", "body"]`
- mot TF-IDF vectorizer cho moi field (`max_features = 5000` moi field)
- thong ke theo field = `length + entropy + 26 keyword ratios`
- tong chieu input = `20112` features (`20000` TF-IDF + `112` stats)

---

## Quy trinh benchmark

Tat ca lenh gia dinh dang o root repository va da kich hoat Python virtual environment. Lan chay local gan nhat dung `.venv/bin/python` vi moi truong `python3` cua he thong khong co `pandas`.

Dung sequence nay khi can so benchmark co the lap lai:

```bash
# 1. Tuy chon: rebuild processed dataset sau khi sua data/attack.txt hoac data/normal.txt
python src/standardize_data.py

# 2. Tuy chon: retrain models sau khi du lieu thay doi
python src/random_forest/train.py
python src/logistic_regression/train.py
python src/xgboost/train.py

# 3. Tuy chon: refresh ONNX artifacts dung cho Go/runtime benchmark
python src/random_forest/export_for_go.py
python src/logistic_regression/export_for_go.py
python src/xgboost/export_for_go.py

# 4. Chay categorical benchmark voi joblib artifacts
python src/random_forest/test_categories.py --log reports/random_forest/categorical_test_joblib.log
python src/logistic_regression/test_categories.py --log reports/logistic_regression/categorical_test_joblib.log
python src/xgboost/test_categories.py --log reports/xgboost/categorical_test_joblib.log

# 5. Chay categorical benchmark voi ONNX artifacts
python src/random_forest/test_categories.py --onnx --log reports/random_forest/categorical_test_onnx.log
python src/logistic_regression/test_categories.py --onnx --log reports/logistic_regression/categorical_test_onnx.log
python src/xgboost/test_categories.py --onnx --log reports/xgboost/categorical_test_onnx.log
```

Cac entry point benchmark dang duoc duy tri la `test_categories.py`, `test_holdout_regression.py`, `test_samples.py`, va `test_specific_payload.py`. Cac script helper logistic regression cu kieu one-off da bi go bo.

### Regression theo category (`test_categories.py`)

Test doc `data/attack_fields.txt` va `data/normal_fields.txt`. Moi category duoc chay ca `RAW` va `ENC` (cac component da URL-encode), tong cong `1512` case trong bo hien tai. Cac regression header benign hien tai gom ca `Benign Fluent Bit Splunk Headers` va `Benign Fluent Bit Short Headers`.

| Mode | Flags | Su dung |
| ---- | ----- | ------- |
| `joblib` | none | `models/<name>/model.joblib` + vectorizer |
| `onnx` | `--onnx` | `application/go/<name>/assets/model.onnx` (fallback: `models/<name>/model.onnx`) |
| `log` | `--log PATH` | ghi stdout dong thoi ra file |

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

Bao cao JSON duoc ghi vao `reports/<model>/categorical_results.json` hoac `categorical_results_onnx.json` khi dung `--onnx`.

Kiem tra parity ONNX cua XGBoost (sparse sklearn so voi ONNX voi `NaN` cho feature bang 0):

```bash
python src/xgboost/check_parity.py
```

### Hanh vi prediction cua Python

`src/*/predict.py` chuan hoa input bang `split_request_components()`, sau do cham diem rieng `path`, `query`, `headers`, va `body` thong qua `predict_components()`. Component co xac suat attack cao nhat la verdict cuoi cung (`decisive_component`).

Debug mot payload rieng le:

```bash
python src/test_specific_payload.py
```

### Mau interactive nhanh

```bash
python src/test_samples.py --model random_forest
python src/test_samples.py --model logistic_regression
python src/test_samples.py --model xgboost
```

### Hold-out set (`data/holdout_*.txt` - khong merge vao training)

```bash
python src/test_holdout_regression.py --model both          # chi RF + LR
python src/test_holdout_regression.py --model all
```

---

## Hieu nang model

Lan refresh categorical regression gan nhat, chay vao **July 25, 2026** tren `data/attack_fields.txt` + `data/normal_fields.txt` (`1512` danh gia RAW + ENC):

| Model | Backend | Ket qua | Loi | Ghi chu |
| ----- | ------- | ------- | --- | ------- |
| **Random Forest** | joblib | `1485 / 1512` (`98.21%`) | 27 | 26 false negatives, 1 false positive |
| **Random Forest** | ONNX | `1485 / 1512` (`98.21%`) | 27 | Khop ket qua joblib |
| **Logistic Regression** | joblib | `1480 / 1512` (`97.88%`) | 32 | 30 false negatives, 2 false positives |
| **Logistic Regression** | ONNX | `1480 / 1512` (`97.88%`) | 32 | Khop ket qua joblib |
| **XGBoost** | joblib | `1484 / 1512` (`98.15%`) | 28 | 19 false negatives, 9 false positives |
| **XGBoost** | ONNX | `1485 / 1512` (`98.21%`) | 27 | 26 false negatives, 1 false positive |

Cac miss hien tai cua Random Forest: `Slowloris Header Pattern [path]`, `TestReal1 [path]`, `TestReal2 [path]`, `Attack_PDF_105 [path]`, `Attack_usr_135-136 [path]`, `Attack_FP_137 [path]`, `Attack_usr_138-139 [path]`, `PADDED_XSS [path]`, `Attack_Asset_1-4 [path]`, `Attack_Analyzer_Combined_XSS_SQLi_HTML [path]`, cong mot false positive tren `Benign Asset [path]`.

Cac miss hien tai cua Logistic Regression: `Slowloris Header Pattern [path]`, `TestReal1 [path]`, `TestReal2 [path]`, `Attack_PDF_105 [path]`, `Attack_usr_135-136 [path]`, `Attack_FP_137 [path]`, `Attack_usr_138-139 [path]`, `PADDED_XSS [path]`, `Attack_Asset_1-4 [path]`, `Attack_Analyzer_Combined_XSS_SQLi_HTML [path]`, cong false positive tren `FP_USER_55 [query]` va `FP_USER_57 [query]`.

Cac miss hien tai cua XGBoost joblib: `Slowloris Header Pattern [path]`, `Attack_PDF_105 [path]`, `Attack_usr_135-136 [path]`, `Attack_FP_137 [path]`, `Attack_usr_138-139 [path]`, `PADDED_XSS [path]`, `Attack_Asset_3-4 [path]`, `Attack_Analyzer_Combined_XSS_SQLi_HTML [path]`, cong false positive tren `FP1 [path]`, `FP2 [path]`, `Benign Asset [path]`, `FP_USER_59 [path]`, `FP_USER_60 [path]`, va `Benign Issue Collection Path [path]`.

Cac miss hien tai cua XGBoost ONNX: `Slowloris Header Pattern [path]`, `TestReal1 [path]`, `TestReal2 [path]`, `Attack_PDF_105 [path]`, `Attack_usr_135-136 [path]`, `Attack_FP_137 [path]`, `Attack_usr_138-139 [path]`, `PADDED_XSS [path]`, `Attack_Asset_1-4 [path]`, `Attack_Analyzer_Combined_XSS_SQLi_HTML [path]`, cong mot false positive tren `Benign Asset [path]`.

Lan chay July 25 phat sinh `InconsistentVersionWarning` tu scikit-learn khi unpickle artifact voi phien ban scikit-learn khong khop. XGBoost joblib cung canh bao ve viec load model serialized cu bang XGBoost `2.1.4`; nen dien giai ket qua voi cac khac biet moi truong do.

`python src/xgboost/check_parity.py` truoc day bao cao mismatch xac suat giua sparse `joblib` inference va ONNX tren payload mau; cac ket qua categorical o tren xac nhan cac backend van chua co parity ket qua tuyet doi.

Baseline ONNX lich su tu **May 26, 2026** tren `data/attack_fields.txt` + `data/normal_fields.txt` (`742` field cases, `1484` danh gia RAW + ENC):

| Model | Categorical regression | Mau miss hien tai |
| ----- | ---------------------- | ----------------- |
| **Random Forest** | `1457 / 1484` (`98.18%`) | Cac miss con lai chu yeu la `Slowloris Header Pattern`, `TestReal1/2`, `Attack_PDF_105`, `Attack_usr_135-139`, `PADDED_XSS`, vai path attack asset/analyzer, cong mot FP tren `Benign Asset [path]` (ENC) |
| **Logistic Regression** | `1426 / 1468` (`97.14%`) | Cung nhom miss attack loi nhu RF, them `Attack_PDF_106`, tat ca miss RAW/ENC cua `Attack_Asset_1-4`, va FP tren `FP_PDF_48/49`, `FP_USER_55/57`, `Benign Workspace Modules API`, `Benign Sidebar Preferences API` |
| **XGBoost** | `1437 / 1468` (`97.89%`) | Cung nhom miss attack loi nhu RF, miss mot phan tren `Attack_Asset_1-4`, mot FP tren `FP_PDF_34 [path]` (ENC), va FP tren `Benign Fluent Bit Splunk Headers [headers]` (RAW/ENC) |

Chay lai sau khi export de cap nhat so lieu:

```bash
python src/random_forest/test_categories.py --onnx
python src/logistic_regression/test_categories.py --onnx
python src/xgboost/test_categories.py --onnx
```

File hold-out (`data/holdout_attack.txt`, `holdout_normal.txt`) duoc co y **loai khoi** cac lan merge trong `train.py` - hay dung chung de sanity-check kha nang generalization ben ngoai cac danh sach regression chinh.

Agent lam viec trong repository cung nen doc **`AGENTS.md`** (build commands, beads, va constraints).
