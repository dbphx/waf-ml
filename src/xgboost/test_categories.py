import argparse
import contextlib
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(__file__))

from preprocessing import encode_request_components, parse_category_lines
from predict import HTTPAttackPredictor

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def run_categorical_test(use_onnx=False):
    models_dir = f"{PROJECT_ROOT}/models/xgboost"
    data_dir = f"{PROJECT_ROOT}/data"

    if use_onnx:
        onnx_path = f"{PROJECT_ROOT}/application/go/xgboost/assets/model.onnx"
        if not os.path.exists(onnx_path) and not os.path.exists(os.path.join(models_dir, "model.onnx")):
            print("Error: ONNX model file not found.")
            return False
    elif not os.path.exists(os.path.join(models_dir, "model.joblib")):
        print("Error: Model files not found. Run train.py first.")
        return False

    predictor = HTTPAttackPredictor(models_dir, use_onnx=use_onnx)
    mode = "ONNXRuntime" if use_onnx else "joblib+xgboost"
    print(f"=== XGBoost categorical test | backend={mode} ===\n")

    test_files = [
        {"file": "attack_fields.txt", "expected": "ATTACK"},
        {"file": "normal_fields.txt", "expected": "NORMAL"},
    ]

    results = []
    print(
        f"{'Category':<50} | {'Type':<8} | {'Expected':<8} | {'Pred':<8} | "
        f"{'Conf':<6} | {'Time (ms)':<9} | {'Status'}"
    )
    print("-" * 115)

    total = 0
    passed = 0

    import time

    for test_file in test_files:
        path = os.path.join(data_dir, test_file["file"])
        if not os.path.exists(path):
            print(f"Warning: {test_file['file']} not found.")
            continue

        categories = parse_category_lines(path)
        for cat in categories:
            total += 1
            start = time.time()
            pred, conf = predictor.predict(cat["request"])
            elapsed = (time.time() - start) * 1000

            is_correct = pred == test_file["expected"]
            if is_correct:
                passed += 1

            status = "✅" if is_correct else "❌"
            print(
                f"{cat['category'][:50]:<50} | {'RAW':<8} | {test_file['expected']:<8} | "
                f"{pred:<8} | {conf:.4f} | {elapsed:>8.2f} | {status}"
            )

            results.append(
                {
                    "category": cat["category"],
                    "payload": cat["payload"],
                    "type": "RAW",
                    "expected": test_file["expected"],
                    "predicted": pred,
                    "confidence": conf,
                    "time_ms": elapsed,
                    "correct": is_correct,
                }
            )

            encoded_request = encode_request_components(cat["request"])
            total += 1
            start = time.time()
            pred_enc, conf_enc = predictor.predict(encoded_request)
            elapsed_enc = (time.time() - start) * 1000

            is_correct_enc = pred_enc == test_file["expected"]
            if is_correct_enc:
                passed += 1

            status_enc = "✅" if is_correct_enc else "❌"
            print(
                f"{cat['category'][:50]:<50} | {'ENC':<8} | {test_file['expected']:<8} | "
                f"{pred_enc:<8} | {conf_enc:.4f} | {elapsed_enc:>8.2f} | {status_enc}"
            )

            results.append(
                {
                    "category": cat["category"],
                    "payload": encoded_request,
                    "type": "ENCODED",
                    "expected": test_file["expected"],
                    "predicted": pred_enc,
                    "confidence": conf_enc,
                    "time_ms": elapsed_enc,
                    "correct": is_correct_enc,
                }
            )

    print("-" * 115)
    accuracy = (passed / total) * 100 if total > 0 else 0
    print(f"SUMMARY: {passed}/{total} Passed ({accuracy:.2f}%)")

    suffix = "_onnx" if use_onnx else ""
    report_path = f"{PROJECT_ROOT}/reports/xgboost/categorical_results{suffix}.json"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    import json

    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"Detailed categorical report saved to {report_path}")
    return passed == total and total > 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test XGBoost categorical classification")
    parser.add_argument("--onnx", action="store_true", help="Use ONNX model instead of Joblib")
    parser.add_argument("--log", type=str, default=None, help="Also write stdout to this file")
    args = parser.parse_args()

    if args.log:
        os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)
        with open(args.log, "w", encoding="utf-8") as log_file:
            with contextlib.redirect_stdout(_Tee(sys.stdout, log_file)):
                ok = run_categorical_test(use_onnx=args.onnx)
    else:
        ok = run_categorical_test(use_onnx=args.onnx)
    sys.exit(0 if ok else 1)
