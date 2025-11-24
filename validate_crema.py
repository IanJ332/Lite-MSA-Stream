import os
import requests
import random
import json
import numpy as np
import librosa
from tqdm import tqdm
from app.services.ensemble_service import EnsembleService

# Config
DATA_DIR = "crema_validation"
LOG_FILE = "crema_validation_log.jsonl"
NUM_SAMPLES = 30 # Number of files to test

# CREMA-D Mapping
# Filename: 1001_DFA_ANG_XX.wav
EMOTION_MAP = {
    "ANG": "angry",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad",
    "FEA": "fearful",
    "DIS": "disgusted"
}

# Speakers (1001-1091)
SPEAKERS = range(1001, 1050) # Use first 50 speakers
SENTENCES = ["DFA", "IEO", "TIE", "IOM", "IWW", "TAI"]
EMOTIONS = ["ANG", "HAP", "NEU", "SAD", "FEA", "DIS"]

def download_file(url, filepath):
    try:
        r = requests.get(url, allow_redirects=True)
        if r.status_code == 200:
            with open(filepath, 'wb') as f:
                f.write(r.content)
            return True
    except:
        pass
    return False

def main():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    print("--- 1. Downloading CREMA-D Samples ---")
    files_to_test = []
    
    # Generate random targets
    attempts = 0
    while len(files_to_test) < NUM_SAMPLES and attempts < 100:
        attempts += 1
        spk = random.choice(SPEAKERS)
        sen = random.choice(SENTENCES)
        emo = random.choice(EMOTIONS)
        filename = f"{spk}_{sen}_{emo}_XX.wav"
        url = f"https://github.com/CheyneyComputerScience/CREMA-D/raw/master/AudioWAV/{filename}"
        filepath = os.path.join(DATA_DIR, filename)
        
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...", end="\r")
            if download_file(url, filepath):
                files_to_test.append((filepath, EMOTION_MAP[emo]))
        else:
            files_to_test.append((filepath, EMOTION_MAP[emo]))
            
    print(f"\nDownloaded {len(files_to_test)} files.")

    print("\n--- 2. Running Validation ---")
    service = EnsembleService(device="cpu")
    if os.path.exists("fusion_weights.pth"):
        service.load_fusion_model("fusion_weights.pth")
    else:
        print("Error: fusion_weights.pth not found!")
        return
    
    results = []
    correct_count = 0
    
    with open(LOG_FILE, "w") as f:
        for filepath, true_emotion in tqdm(files_to_test):
            try:
                # Load Audio
                y, sr = librosa.load(filepath, sr=16000)
                
                # Predict
                probs = service.predict_emotions(y)
                
                # Get Top Prediction
                pred_emotion = max(probs, key=probs.get)
                confidence = probs[pred_emotion]
                
                is_correct = (pred_emotion == true_emotion)
                if is_correct:
                    correct_count += 1
                    
                # Log Entry
                entry = {
                    "file": os.path.basename(filepath),
                    "true_emotion": true_emotion,
                    "predicted_emotion": pred_emotion,
                    "confidence": float(confidence),
                    "result": "CORRECT" if is_correct else "WRONG",
                    "all_probs": probs
                }
                
                f.write(json.dumps(entry) + "\n")
                results.append(entry)
                
            except Exception as e:
                print(f"Error processing {filepath}: {e}")

    # Summary
    accuracy = (correct_count / len(files_to_test)) * 100
    print(f"\n--- Validation Complete ---")
    print(f"Accuracy: {accuracy:.2f}% ({correct_count}/{len(files_to_test)})")
    print(f"Detailed logs saved to {LOG_FILE}")
    
    # Print Confusion Matrix-style summary
    print("\nError Analysis:")
    for r in results:
        if r["result"] == "WRONG":
            print(f"File: {r['file']} | True: {r['true_emotion']} -> Pred: {r['predicted_emotion']} ({r['confidence']:.2f})")

if __name__ == "__main__":
    main()
