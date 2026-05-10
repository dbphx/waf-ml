WAF-ML — gói model + chạy test
================================

Sau khi giải nén thư mục waf-ml/:

1) Tạo môi trường ảo (một lần)
   python3 -m venv .venv

2) Kích hoạt
   macOS / Linux:  source .venv/bin/activate
   Windows CMD:     .venv\Scripts\activate.bat
   Windows PowerShell:  .venv\Scripts\Activate.ps1

3) Cài dependency
   pip install --upgrade pip
   pip install -r requirements.txt

4) Chạy test hồi quy theo category (mặc định joblib; cần Python 3.9+)
   python src/random_forest/test_categories.py
   python src/logistic_regression/test_categories.py

   Tùy chọn ONNX (nếu đã có model.onnx trong models/...):
   python src/random_forest/test_categories.py --onnx
   python src/logistic_regression/test_categories.py --onnx

5) Mẫu nhanh / hold-out
   python src/test_samples.py --model random_forest
   python src/test_holdout_regression.py --model all
