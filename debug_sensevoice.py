import os
import soundfile as sf
import numpy as np
import torch
from app.services.sensevoice_service import SenseVoiceService

# Path to an Angry file (05 = Angry)
# 03-01-05-01-01-01-01.wav
TEST_FILE = "ravdess_data/03-01-05-01-01-01-01.wav"

from funasr import AutoModel

def debug_inference():
    test_file = TEST_FILE
    if not os.path.exists(test_file):
        # ... (same file finding logic) ...
        import glob
        files = glob.glob("ravdess_data/*-05-*.wav")
        if files:
            test_file = files[0]
        else:
            print("No angry files found.")
            return

    print("Loading Model directly...")
    model = AutoModel(
        model="iic/SenseVoiceSmall",
        trust_remote_code=True,
        device="cpu",
        disable_update=True
    )
    
    print(f"Loading {test_file}...")
    waveform, sr = sf.read(test_file, dtype='float32')
    if len(waveform.shape) > 1:
        waveform = np.mean(waveform, axis=1)
    # Experiment 10: Use SenseVoiceService Class
    print("\n--- Exp 10: SenseVoiceService Class ---")
    
    # Prepare audio bytes (Exp 8 style)
    pcm_data = (waveform * 32767).clip(-32768, 32767).astype(np.int16)
    audio_bytes = pcm_data.tobytes()
    
    from app.services.sensevoice_service import SenseVoiceService
    service = SenseVoiceService(device="cpu")
    # We need to pass audio_bytes (PCM16)
    res10 = service.predict(audio_bytes)
    print(res10)

if __name__ == "__main__":
    debug_inference()
