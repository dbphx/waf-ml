"""
Held-out regression: payloads in data/holdout_*.txt are never merged in train.py
(train only injects data/attack.txt + data/normal.txt). Same request parsing as categorical tests.
"""
import argparse
import importlib.util
import os
import re
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))


def parse_holdout_file(filepath: str):
    """Lines like: 1. Label text: GET /path — category must not contain unpaired ':' before payload."""
    rows = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            m = re.match(r"^\d+\.\s+(.*?):\s+(.*)$", line.strip())
            if m:
                rows.append({"category": m.group(1).strip(), "payload": m.group(2).strip()})
    return rows


def payload_to_request_dict(payload: str) -> dict:
    """Match random_forest/test_categories predict_as_runtime_request."""
    method = "GET"
    url = str(payload)
    headers = ""
    body = ""
    user_agent = ""

    first_line = url.strip().splitlines()[0] if url.strip() else ""
    m = re.match(r"^(GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS)\s+(\S+)", first_line, re.IGNORECASE)
    if m:
        method = m.group(1).upper()
        url = m.group(2)
        remainder = str(payload)[m.end() :].lstrip()
        if remainder:
            parts = re.split(r"\r\n\r\n|\n\n", remainder, maxsplit=1)
            headers = parts[0].strip()
            body = parts[1].strip() if len(parts) > 1 else ""
            ua_match = re.search(r"(?im)^User-Agent:\s*(.+)$", headers)
            if ua_match:
                user_agent = ua_match.group(1).strip()

    return {
        "method": method,
        "url": url,
        "headers": headers,
        "body": body,
        "user_agent": user_agent,
    }


def load_predictor(model_name: str):
    # Load each src/<model>/predict.py as its own module — `from predict import ...` would
    # reuse the first cached `predict` (e.g. random_forest) when --model all.
    predict_path = os.path.join(os.path.dirname(__file__), model_name, "predict.py")
    if not os.path.isfile(predict_path):
        print(f"No predict.py at {predict_path}", file=sys.stderr)
        return None
    spec = importlib.util.spec_from_file_location(
        f"waf_holdout_{model_name}_predict",
        predict_path,
    )
    if spec is None or spec.loader is None:
        print(f"Could not load {predict_path}", file=sys.stderr)
        return None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    HTTPAttackPredictor = mod.HTTPAttackPredictor

    models_dir = os.path.join(PROJECT_ROOT, "models", model_name)
    if not os.path.exists(os.path.join(models_dir, "model.joblib")):
        print(f"Missing model.joblib under {models_dir}", file=sys.stderr)
        return None
    return HTTPAttackPredictor(models_dir)


def run_holdout(model_name: str) -> bool:
    predictor = load_predictor(model_name)
    if predictor is None:
        return False

    attack_path = os.path.join(PROJECT_ROOT, "data", "holdout_attack.txt")
    normal_path = os.path.join(PROJECT_ROOT, "data", "holdout_normal.txt")
    cases = []
    for cat in parse_holdout_file(attack_path):
        cases.append({**cat, "expected": "ATTACK"})
    for cat in parse_holdout_file(normal_path):
        cases.append({**cat, "expected": "NORMAL"})

    if not cases:
        print("No holdout cases parsed; check data/holdout_*.txt format.", file=sys.stderr)
        return False

    print(f"\n=== hold-out regression | model={model_name} | n={len(cases)} ===\n")
    ok = 0
    for c in cases:
        req = payload_to_request_dict(c["payload"])
        pred, conf = predictor.predict(req)
        good = pred == c["expected"]
        if good:
            ok += 1
        mark = "OK" if good else "FAIL"
        print(f"[{mark}] {c['category'][:52]:<52} exp={c['expected']:<6} pred={pred:<6} conf={conf}")

    print(f"\nSUMMARY: {ok}/{len(cases)} passed\n")
    return ok == len(cases)


def main():
    parser = argparse.ArgumentParser(description="Held-out tests (not in training merge)")
    parser.add_argument(
        "--model",
        choices=["random_forest", "logistic_regression", "both", "all"],
        default="both",
        help="both/all = RF+LR",
    )
    args = parser.parse_args()

    if args.model == "both":
        models = ["random_forest", "logistic_regression"]
    elif args.model == "all":
        models = ["random_forest", "logistic_regression"]
    else:
        models = [args.model]
    all_ok = True
    for m in models:
        if not run_holdout(m):
            all_ok = False
    return all_ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
