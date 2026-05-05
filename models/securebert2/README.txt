SecureBERT 2.0 bundle (not fully in git: *.safetensors is ignored by repo .gitignore).

Train from repo root (downloads ~600MB weights on first run):
  python src/securebert2/train.py

Optional env:
  SECUREBERT2_MAX_LEN=256
  SECUREBERT2_EMBED_BATCH=8        (increase on GPU)
  SECUREBERT2_MAX_TRAIN_ROWS=      (omit for full data; small values are smoke-test only and will not match 100% categorical tests)

Export copies + end-to-end ONNX (encoder + mean-pool + LR → model.onnx):
  python src/securebert2/export_for_go.py

Test categorical suite with ONNXRuntime:
  python src/securebert2/test_categories.py --onnx --log reports/securebert2/onnx_test.log

Go waf.BaseDetector does not load this model; use Python (src/securebert2/predict.py).
