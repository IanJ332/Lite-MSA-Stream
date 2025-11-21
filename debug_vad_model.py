import onnxruntime as ort
import numpy as np
import os

model_path = os.path.join("models", "onnx", "model.onnx")

print(f"Loading model from: {model_path}")

try:
    sess = ort.InferenceSession(model_path)
    
    with open("vad_debug_info.txt", "w") as f:
        f.write("\n--- Model Inputs ---\n")
        for i, input_meta in enumerate(sess.get_inputs()):
            f.write(f"Input {i}:\n")
            f.write(f"  Name: {input_meta.name}\n")
            f.write(f"  Shape: {input_meta.shape}\n")
            f.write(f"  Type: {input_meta.type}\n")
            
        f.write("\n--- Model Outputs ---\n")
        for i, output_meta in enumerate(sess.get_outputs()):
            f.write(f"Output {i}:\n")
            f.write(f"  Name: {output_meta.name}\n")
            f.write(f"  Shape: {output_meta.shape}\n")
            f.write(f"  Type: {output_meta.type}\n")
            
    print("Debug info written to vad_debug_info.txt")

except Exception as e:
    with open("vad_debug_info.txt", "w") as f:
        f.write(f"Error loading model: {e}")
    print(f"Error: {e}")
