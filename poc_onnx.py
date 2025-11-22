import os
from pathlib import Path
from funasr_onnx import SenseVoiceSmall
from funasr_onnx.utils.postprocess_utils import rich_transcription_postprocess
import time

def main():
    print("=== SenseVoice ONNX PoC ===")
    
    # 1. Locate Model
    # Assuming model is in default modelscope cache
    home = Path.home()
    model_dir = home / ".cache/modelscope/hub/iic/SenseVoiceSmall"
    
    print(f"Looking for model in: {model_dir}")
    if not model_dir.exists():
        print(f"❌ Model directory not found: {model_dir}")
        # Try to find it elsewhere or ask user
        return

    model_file = model_dir / "model_quant.onnx"
    if not model_file.exists():
        print(f"❌ Quantized model file not found: {model_file}")
        print("Checking for non-quantized model...")
        model_file = model_dir / "model.onnx"
        if not model_file.exists():
             print(f"❌ No ONNX model found in {model_dir}")
             return
    
    print(f"Found model: {model_file}")

    # 2. Load Model
    try:
        # funasr_onnx.SenseVoiceSmall expects model_dir, and looks for model.onnx or model_quant.onnx inside
        # It also needs config.yaml, am.mvn, tokens.json/bpe.model
        model = SenseVoiceSmall(str(model_dir), batch_size=1, quantize=True)
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # 3. Test Inference
    wav_files = []
    for root, dirs, files in os.walk("."):
        for file in files:
            if file.endswith(".wav"):
                wav_files.append(os.path.join(root, file))
                if len(wav_files) >= 1: break
        if wav_files: break
    
    if not wav_files:
        print("No wav files found for testing.")
        return
        
    test_file = wav_files[0]
    print(f"Testing with file: {test_file}")
    
    start_time = time.time()
    
    try:
        # inference
        # language="auto", textnorm="withitn"
        res = model([test_file], language="auto", textnorm="withitn")
        
        end_time = time.time()
        print(f"Inference time: {end_time - start_time:.4f}s")
        
        print("Raw Result:")
        print(res)
        
        print("Post-processed Result:")
        print([rich_transcription_postprocess(i) for i in res])
        
    except Exception as e:
        print(f"❌ Inference failed: {e}")

if __name__ == "__main__":
    main()
