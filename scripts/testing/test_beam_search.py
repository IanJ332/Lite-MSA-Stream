import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import soundfile as sf
import numpy as np
from funasr import AutoModel

def test_beam():
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
        
    if sr != 16000:
        from scipy import signal
        num_samples = int(len(waveform) * 16000 / sr)
        waveform = signal.resample(waveform, num_samples)
        
    with open("beam_summary.txt", "w") as f:
        print("\n--- Testing Beam Size = 1 (Default) ---")
        start = time.time()
        try:
            res1 = model.generate(
                input=waveform,
                cache={},
                language="auto",
                use_itn=True,
                beam_size=1
            )
            t1 = time.time() - start
            print(f"Time: {t1:.4f}s")
            f.write(f"Beam 1 Time: {t1:.4f}s\n")
            f.write(f"Beam 1 Result: {res1}\n")
        except Exception as e:
            print(f"Failed: {e}")
            f.write(f"Beam 1 Failed: {e}\n")

        print("\n--- Testing Beam Size = 10 ---")
        start = time.time()
        try:
            res10 = model.generate(
                input=waveform,
                cache={},
                language="auto",
                use_itn=True,
                beam_size=10
            )
            t10 = time.time() - start
            print(f"Time: {t10:.4f}s")
            f.write(f"Beam 10 Time: {t10:.4f}s\n")
            f.write(f"Beam 10 Result: {res10}\n")
        except Exception as e:
            print(f"Failed: {e}")
            f.write(f"Beam 10 Failed: {e}\n")

if __name__ == "__main__":
    test_beam()
