import os
import sys
import shutil
import subprocess
import json
import csv
import glob
import librosa
import numpy as np
from tqdm import tqdm
import kagglehub # Added kagglehub

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from app.services.ensemble_service import EnsembleService

# Config
DATASET_NAME = "ejlok1/cremad" # Kaggle dataset
TEMP_DIR = "temp_crema_d"
LOG_FILE_JSON = "outputs/crema_full_validation_log.jsonl"
LOG_FILE_CSV = "outputs/crema_full_validation_log.csv"

# CREMA-D Mapping
EMOTION_MAP = {
    "ANG": "angry",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad",
    "FEA": "fearful",
    "DIS": "disgusted"
}

def download_dataset():
    print(f"Downloading CREMA-D from Kaggle ({DATASET_NAME})...")
    try:
        # Download to Kaggle cache
        path = kagglehub.dataset_download(DATASET_NAME)
        print(f"Dataset downloaded to: {path}")
        return path
    except Exception as e:
        print(f"Failed to download dataset: {e}")
        return None

def get_label_from_filename(filename):
    # Filename format: 1001_DFA_ANG_XX.wav
    parts = filename.split("_")
    if len(parts) >= 3:
        emo_code = parts[2]
        return EMOTION_MAP.get(emo_code)
    return None

def cleanup(path):
    # User requested to delete the dataset. 
    # Since kagglehub downloads to a global cache, deleting it might be aggressive,
    # but the user said "script删除所有的dataset" (script delete all dataset).
    # So we will delete the specific dataset folder from the cache.
    if os.path.exists(path):
        print(f"Cleaning up: Removing {path}...")
        try:
            shutil.rmtree(path)
            print("Cleanup complete.")
        except Exception as e:
            print(f"Warning: Could not fully delete {path}: {e}")

def main():
    # Ensure outputs dir exists
    if not os.path.exists("outputs"):
        os.makedirs("outputs")

    # 1. Download
    dataset_path = download_dataset()
    if not dataset_path:
        return

    # 2. Find Files
    print("Scanning for audio files...")
    # Recursive search as Kaggle datasets structure varies
    wav_files = glob.glob(os.path.join(dataset_path, "**/*.wav"), recursive=True)
    print(f"Found {len(wav_files)} audio files.")

    if len(wav_files) == 0:
        print("No audio files found. Check the dataset structure.")
        cleanup(dataset_path)
        return

    # 3. Initialize Service
    print("Initializing Ensemble Service...")
    service = EnsembleService(device="cpu")
    
    # Find project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    weights_path = os.path.join(project_root, "fusion_weights.pth")
    
    if os.path.exists(weights_path):
        service.load_fusion_model(weights_path)
    else:
        print("Error: fusion_weights.pth not found! Running without fusion weights (low accuracy expected).")

    # 4. Run Validation
    print(f"Starting validation on {len(wav_files)} files...")
    
    results = []
    correct_count = 0
    
    f_json = open(LOG_FILE_JSON, "w")
    f_csv = open(LOG_FILE_CSV, "w", newline='')
    csv_writer = csv.writer(f_csv)
    
    # CSV Header
    header = ["File", "True Emotion", "Predicted Emotion", "Confidence", "Result", 
              "Happy", "Sad", "Angry", "Fearful", "Disgusted", "Neutral"]
    csv_writer.writerow(header)
    
    for filepath in tqdm(wav_files):
        try:
            filename = os.path.basename(filepath)
            true_emotion = get_label_from_filename(filename)
            
            if not true_emotion:
                continue # Skip files we can't label
                
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
                "file": filename,
                "true_emotion": true_emotion,
                "predicted_emotion": pred_emotion,
                "confidence": float(confidence),
                "result": "CORRECT" if is_correct else "WRONG",
                "all_probs": probs
            }
            
            # Write JSON
            f_json.write(json.dumps(entry) + "\n")
            
            # Write CSV
            row = [
                filename,
                true_emotion,
                pred_emotion,
                f"{confidence:.4f}",
                "CORRECT" if is_correct else "WRONG",
                f"{probs.get('happy', 0):.4f}",
                f"{probs.get('sad', 0):.4f}",
                f"{probs.get('angry', 0):.4f}",
                f"{probs.get('fearful', 0):.4f}",
                f"{probs.get('disgusted', 0):.4f}",
                f"{probs.get('neutral', 0):.4f}"
            ]
            csv_writer.writerow(row)
            
            results.append(entry)
            
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            
    f_json.close()
    f_csv.close()

    # 5. Summary & Cleanup
    if len(results) > 0:
        accuracy = (correct_count / len(results)) * 100
    else:
        accuracy = 0
        
    print(f"\n--- Full Validation Complete ---")
    print(f"Total Files Tested: {len(results)}")
    print(f"Accuracy: {accuracy:.2f}% ({correct_count}/{len(results)})")
    print(f"Detailed JSON logs: {os.path.abspath(LOG_FILE_JSON)}")
    print(f"Detailed CSV logs:  {os.path.abspath(LOG_FILE_CSV)}")
    
    cleanup(dataset_path)

if __name__ == "__main__":
    main()
