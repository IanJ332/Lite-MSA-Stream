import numpy as np
from funasr import AutoModel
import logging

logging.basicConfig(level=logging.INFO)

def test_numpy():
    print("Loading model...")
    model = AutoModel(
        model="iic/SenseVoiceSmall",
        trust_remote_code=True,
        device="cpu",
        disable_update=True
    )
    
    print("Generating dummy audio...")
    # 1 second of silence
    audio_np = np.zeros(16000, dtype=np.float32)
    
    print("Running inference with numpy array...")
    try:
        res = model.generate(
            input=audio_np,
            cache={},
            language="auto",
            use_itn=True
        )
        print("Success!")
        print(res)
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    test_numpy()
