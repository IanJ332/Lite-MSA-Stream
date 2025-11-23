import torch
import soundfile as sf
import numpy as np
from funasr import AutoModel

def debug_token_logits():
    print("Loading model...")
    model = AutoModel(
        model="iic/SenseVoiceSmall",
        trust_remote_code=True,
        device="cpu",
        disable_update=True
    )
    
    test_file = "ravdess_data/03-01-05-01-01-01-01.wav" # Angry
    print(f"Loading {test_file}...")
    waveform, sr = sf.read(test_file, dtype='float32')
    if len(waveform.shape) > 1:
        waveform = np.mean(waveform, axis=1)
    
    # Resample
    if sr != 16000:
        from scipy import signal
        num_samples = int(len(waveform) * 16000 / sr)
        waveform = signal.resample(waveform, num_samples)

    print("Running inference with detailed output options...")
    
    # Try to force return of scores/logits
    # SenseVoice often uses `kwargs` passed to the underlying model.
    # We'll try a few common flags.
    
    try:
        res = model.generate(
            input=waveform,
            cache={},
            language="auto",
            use_itn=True,
            return_logits=True,  # We tried this, let's look closer at the output
            output_scores=True,  # HuggingFace style
            return_token_ids=True # FunASR style?
        )
        
        print("\n--- Result Inspection ---")
        print(f"Type: {type(res)}")
        if isinstance(res, list) and len(res) > 0:
            item = res[0]
            print(f"Keys: {item.keys()}")
            
            if 'logits' in item:
                logits = item['logits']
                print(f"Logits shape: {logits.shape if hasattr(logits, 'shape') else 'N/A'}")
                # If we have logits, we can find the emotion token position
                
            if 'scores' in item:
                print(f"Scores found: {type(item['scores'])}")
                
            if 'token_ids' in item:
                print(f"Token IDs: {item['token_ids']}")
                
    except Exception as e:
        print(f"Inference failed: {e}")

if __name__ == "__main__":
    debug_token_logits()
