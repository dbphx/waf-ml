import argparse
import contextlib
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from preprocessing import encode_request_components, parse_category_lines
from logistic_regression.predict import HTTPAttackPredictor

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()

def run_categorical_test(use_onnx=False):
    models_dir = f"{PROJECT_ROOT}/models/logistic_regression"
    data_dir = f"{PROJECT_ROOT}/data"

    if use_onnx:
        onnx_p = os.path.join(PROJECT_ROOT, "application/go/logistic_regression/assets/model.onnx")
        if not os.path.isfile(onnx_p):
            onnx_p = os.path.join(models_dir, "model.onnx")
        if not os.path.isfile(onnx_p):
            print(f"Error: ONNX not found at application/go/logistic_regression/assets/model.onnx")
            return False
    elif not (os.path.exists(os.path.join(models_dir, "model.joblib"))):
        print("Error: Model files not found. Run train.py first.")
        return False

    predictor = HTTPAttackPredictor(models_dir, use_onnx=use_onnx)
    mode = "ONNXRuntime" if use_onnx else "joblib+sklearn"
    print(f"=== Logistic regression categorical test | backend={mode} ===\n")

    test_files = [
        {"file": "attack_fields.txt", "expected": "ATTACK"},
        {"file": "normal_fields.txt", "expected": "NORMAL"},
    ]

    results = []
    print(f"{'Category':<50} | {'Type':<8} | {'Expected':<8} | {'Pred':<8} | {'Conf':<6} | {'Time (ms)':<9} | {'Status'}")
    print("-" * 115)

    total = 0
    passed = 0
    import time

    for tf in test_files:
        path = os.path.join(data_dir, tf["file"])
        if not os.path.exists(path):
            print(f"Warning: {tf['file']} not found.")
            continue

        categories = parse_category_lines(path)
        for cat in categories:
            total += 1
            t0 = time.time()
            pred, conf = predictor.predict(cat["request"])
            elapsed = (time.time() - t0) * 1000

            is_correct = pred == tf["expected"]
            if is_correct:
                passed += 1

            status = "✅" if is_correct else "❌"
            print(f"{cat['category'][:50]:<50} | {'RAW':<8} | {tf['expected']:<8} | {pred:<8} | {conf:.4f} | {elapsed:>8.2f} | {status}")

            results.append(
                {
                    "category": cat["category"],
                    "payload": cat["payload"],
                    "type": "RAW",
                    "expected": tf["expected"],
                    "predicted": pred,
                    "confidence": conf,
                    "time_ms": elapsed,
                    "correct": is_correct,
                }
            )

            encoded_request = encode_request_components(cat["request"])
            total += 1
            t0 = time.time()
            pred_enc, conf_enc = predictor.predict(encoded_request)
            elapsed_enc = (time.time() - t0) * 1000

            is_correct_enc = pred_enc == tf["expected"]
            if is_correct_enc:
                passed += 1

            status_enc = "✅" if is_correct_enc else "❌"
            print(
                f"{cat['category'][:50]:<50} | {'ENC':<8} | {tf['expected']:<8} | {pred_enc:<8} | {conf_enc:.4f} | {elapsed_enc:>8.2f} | {status_enc}"
            )

            results.append(
                {
                    "category": cat["category"],
                    "payload": encoded_request,
                    "type": "ENCODED",
                    "expected": tf["expected"],
                    "predicted": pred_enc,
                    "confidence": conf_enc,
                    "time_ms": elapsed_enc,
                    "correct": is_correct_enc,
                }
            )

    print("-" * 115)
    accuracy = (passed / total) * 100 if total > 0 else 0
    print(f"SUMMARY: {passed}/{total} Passed ({accuracy:.2f}%)")

    sfx = "_onnx" if use_onnx else ""
    report_path = f"{PROJECT_ROOT}/reports/logistic_regression/categorical_results{sfx}.json"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Detailed categorical report saved to {report_path}")
    return passed == total and total > 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", action="store_true", help="Use ONNX from application/go/.../assets")
    parser.add_argument("--log", type=str, default=None, help="Also write stdout to this file")
    args = parser.parse_args()

    if args.log:
        os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)
        with open(args.log, "w", encoding="utf-8") as logf:
            with contextlib.redirect_stdout(_Tee(sys.stdout, logf)):
                ok = run_categorical_test(use_onnx=args.onnx)
    else:
        ok = run_categorical_test(use_onnx=args.onnx)
    sys.exit(0 if ok else 1)
