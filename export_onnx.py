import joblib
import os
import sys

try:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "skl2onnx", "onnxruntime"])
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

def export_model():
    models_dir = "models/random_forest"
    model_path = os.path.join(models_dir, 'model.joblib')
    onnx_path = os.path.join(models_dir, 'model.onnx')
    
    print("Loading model...")
    model = joblib.load(model_path)
    
    n_features = model.n_features_in_
    print(f"Model has {n_features} features")
    
    # Random Forest expects floats
    initial_type = [('float_input', FloatTensorType([None, n_features]))]
    
    print("Converting to ONNX...")
    # The default target_opset depends on installed onnx, usually it is fine.
    # Convert probability output too
    options = {id(model): {'zipmap': False}}
    onnx_model = convert_sklearn(model, initial_types=initial_type, options=options)
    
    print(f"Saving to {onnx_path}...")
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
        
    print("ONNX export successful!")
    
    # Quick sanity check if onnxruntime is available
    try:
        import onnxruntime as rt
        import numpy as np
        
        sess = rt.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        input_name = sess.get_inputs()[0].name
        
        # Create dummy data with 1 sample
        dummy_data = np.random.rand(1, n_features).astype(np.float32)
        
        # Predict
        res = sess.run(None, {input_name: dummy_data})
        print(f"Sanity check prediction output shape: {res[0].shape}, proba shape: {res[1].shape}")
        print("Model works in ONNX Runtime!")
    except Exception as e:
        print(f"Sanity check skipped or failed: {e}")

if __name__ == "__main__":
    export_model()
