import os
import time
from funasr import AutoModel
# from funasr.utils.postprocess_utils import rich_transcription_postprocess

def main():
    print("=== SenseVoice PoC ===")
    
    # 1. Load Model
    # This will download the model from ModelScope/HuggingFace
    model_dir = "iic/SenseVoiceSmall"
    print(f"Loading model: {model_dir}...")
    
    try:
        model = AutoModel(
            model=model_dir,
            trust_remote_code=True,
            device="cpu", # Force CPU for this test
            disable_update=True
        )
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # 2. Test Inference
    # Use a local file if available, otherwise use a dummy or download one
    # Let's try to find a wav file in the current directory or subdirectories
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
        
    test_file = os.path.abspath(wav_files[0])
    print(f"Testing with file: {test_file}")
    
    start_time = time.time()
    
    # SenseVoice inference
    # language="auto", use_itn=True
    try:
        res = model.generate(
            input=test_file,
            cache={},
            language="auto",  # "zn", "en", "yue", "ja", "ko"
            use_itn=True,
            batch_size_s=60,
            merge_vad=True,  # merge VAD segments
            merge_thr=1.0,
        )
        
        end_time = time.time()
        print(f"Inference time: {end_time - start_time:.4f}s")
        print("Result:")
        print(res)
        
        # Check for emotion tags if possible (SenseVoice output usually contains them if enabled)
        # Note: Standard generate might need specific config to output emotion tags
        
    except Exception as e:
        print(f"Inference failed: {str(e).encode('ascii', 'replace').decode('ascii')}")

if __name__ == "__main__":
    main()
