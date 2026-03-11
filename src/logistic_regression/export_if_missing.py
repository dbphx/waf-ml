import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# Allow importing from parent src/ directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def check_and_export():
    model_dir = f"{PROJECT_ROOT}/models/logistic_regression"
    onnx_path = f"{PROJECT_ROOT}/go/internal/assets/logistic_regression/model.onnx"
    
    if not os.path.exists(os.path.join(model_dir, 'model.joblib')):
        print(f"Error: model.joblib not found in {model_dir}. Please run train.py first.")
        return

    if not os.path.exists(onnx_path):
        print(f"ONNX model not found at {onnx_path}. Exporting now...")
        from export_for_go import export
        export()
    else:
        print(f"ONNX model already exists at {onnx_path}. Skipping export.")

if __name__ == "__main__":
    check_and_export()
