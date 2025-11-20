import sys
import os
import numpy as np
import logging

# Add project root to path
sys.path.append(os.getcwd())

from app.services.vad_iterator import VADIterator
from app.services.acoustic_analyzer import AcousticAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_pipeline():
    print("--- Testing Pipeline Logic ---")
    
    # 1. Test VAD
    print("\n1. Initializing VAD...")
    try:
        vad = VADIterator()
        print("VAD Initialized.")
        
        # Generate dummy audio (1 chunk of 512 samples, int16)
        # Silero expects float32, but our service handles conversion
        dummy_audio = np.zeros(512, dtype=np.int16).tobytes()
        
        print("   Running VAD Process...")
        vad.process(dummy_audio)
        print("VAD Process Success.")
    except Exception as e:
        print(f"VAD Failed: {e}")
        import traceback
        traceback.print_exc()

    # 2. Test Acoustic
    print("\n2. Initializing Acoustic Analyzer...")
    try:
        acoustic = AcousticAnalyzer()
        print("Acoustic Initialized.")
        
        # Generate dummy audio segment (e.g. 2 seconds)
        # 16000 Hz * 2s = 32000 samples
        dummy_segment = np.random.randint(-32768, 32767, 32000, dtype=np.int16).tobytes()
        
        print("   Running Acoustic Predict...")
        result = acoustic.predict(dummy_segment)
        print(f"Acoustic Predict Result: {result}")
        
        if "error" in result:
             print(f"Acoustic Logic Error: {result['error']}")
             
    except Exception as e:
        print(f"Acoustic Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pipeline()
