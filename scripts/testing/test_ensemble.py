import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import soundfile as sf
from app.services.ensemble_service import EnsembleService

def test_ensemble():
    print("Initializing Ensemble Service...")
    service = EnsembleService(device="cpu")
    
    test_file = "ravdess_data/03-01-05-01-01-01-01.wav" # Angry
    print(f"Loading {test_file}...")
    audio, sr = sf.read(test_file)
    
    # Resample if needed
    if sr != 16000:
        from scipy import signal
        num_samples = int(len(audio) * 16000 / sr)
        audio = signal.resample(audio, num_samples)
        
    print("Extracting features...")
    features = service.extract_features(audio)
    
    print(f"Feature Shape: {features.shape}")
    expected_dim = 1024 + 1024 # HuBERT Large + Wav2Vec2 Large
    
    if features.shape[1] == expected_dim:
        print("SUCCESS: Feature dimension matches expected (2048).")
    else:
        print(f"FAILURE: Expected {expected_dim}, got {features.shape[1]}")

if __name__ == "__main__":
    test_ensemble()
