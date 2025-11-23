import torch
import torch.nn.functional as F
import soundfile as sf
import numpy as np
import time
from transformers import AutoConfig, AutoModelForAudioClassification, Wav2Vec2FeatureExtractor

def test_ser_model():
    # Candidate Model: A Wav2Vec2 model fine-tuned for Emotion Recognition
    # This specific model is trained on multiple datasets and usually supports ~7-8 emotions
    model_id = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
    
    print(f"Loading model: {model_id}...")
    start_load = time.time()
    try:
        config = AutoConfig.from_pretrained(model_id)
        model = AutoModelForAudioClassification.from_pretrained(model_id)
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
        print(f"Model loaded in {time.time() - start_load:.2f}s")
        
        print("Labels:", config.id2label)
        
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Load Test Audio (Angry)
    test_file = "ravdess_data/03-01-05-01-01-01-01.wav" 
    print(f"\nProcessing {test_file}...")
    
    audio, sr = sf.read(test_file)
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
        
    # Resample to 16k if needed (Wav2Vec2 usually expects 16k)
    if sr != 16000:
        from scipy import signal
        num_samples = int(len(audio) * 16000 / sr)
        audio = signal.resample(audio, num_samples)
        sr = 16000

    # Inference
    print("Running Inference...")
    start_inf = time.time()
    
    inputs = feature_extractor(audio, sampling_rate=sr, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        logits = model(**inputs).logits
        
    probabilities = F.softmax(logits, dim=-1)
    scores = probabilities[0].detach().numpy()
    
    inf_time = time.time() - start_inf
    print(f"Inference Time: {inf_time:.4f}s")
    
    # Top 3
    print("\n--- Top 3 Emotions ---")
    ranked = np.argsort(scores)[::-1]
    for i in range(3):
        idx = ranked[i]
        label = config.id2label[idx]
        score = scores[idx]
        print(f"{i+1}. {label}: {score:.2%}")

if __name__ == "__main__":
    test_ser_model()
