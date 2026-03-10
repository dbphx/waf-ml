import joblib
import os
import sys

try:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
except ImportError:
    pass # we already installed it

def export_model():
    models_dir = "models/logistic_regression"
    model_path = os.path.join(models_dir, 'model.joblib')
    onnx_path = os.path.join(models_dir, 'model.onnx')
    
    if not os.path.exists(model_path):
        print("LR model not found")
        return

    print("Loading LR model...")
    model = joblib.load(model_path)
    
    n_features = model.n_features_in_
    print(f"Model has {n_features} features")
    
    initial_type = [('float_input', FloatTensorType([None, n_features]))]
    
    print("Converting LR to ONNX...")
    options = {id(model): {'zipmap': False}}
    onnx_model = convert_sklearn(model, initial_types=initial_type, options=options)
    
    print(f"Saving to {onnx_path}...")
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
        
    print("ONNX export successful!")

if __name__ == "__main__":
    export_model()
