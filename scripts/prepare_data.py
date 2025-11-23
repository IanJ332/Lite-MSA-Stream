import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import glob
import h5py
import numpy as np
import soundfile as sf
from tqdm import tqdm
from app.services.ensemble_service import EnsembleService

# RAVDESS Emotion Mapping
# 01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised
# We map them to 0-7 indices
EMOTION_MAP = {
    '01': 0, # neutral
    '02': 1, # calm
    '03': 2, # happy
    '04': 3, # sad
    '05': 4, # angry
    '06': 5, # fearful
    '07': 6, # disgust
    '08': 7  # surprised
}

def prepare_data():
    print("Initializing Ensemble Service...")
    service = EnsembleService(device="cpu") # Use GPU if available, but user said CPU
    
    data_dir = "ravdess_data"
    output_file = "features.h5"
    
    # Find all wav files
    wav_files = glob.glob(os.path.join(data_dir, "**/*.wav"), recursive=True)
    print(f"Found {len(wav_files)} audio files.")
    
    if len(wav_files) == 0:
        print("No files found. Please ensure 'ravdess_data' contains the dataset.")
        return

    features_list = []
    labels_list = []
    
    # Limit for quick testing (Set to None for full dataset)
    LIMIT = None
    print(f"Extracting features (Full Dataset)...")
    
    count = 0
    for file_path in tqdm(wav_files):
        if count >= LIMIT:
            break
        count += 1
        try:
            # Parse Label
            filename = os.path.basename(file_path)
            parts = filename.split("-")
            if len(parts) < 3:
                continue
            
            emotion_code = parts[2]
            if emotion_code not in EMOTION_MAP:
                continue
                
            label = EMOTION_MAP[emotion_code]
            
            # Load Audio
            audio, sr = sf.read(file_path)
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            # Resample
            if sr != 16000:
                from scipy import signal
                num_samples = int(len(audio) * 16000 / sr)
                audio = signal.resample(audio, num_samples)
            
            # Extract Features
            feat = service.extract_features(audio) # (1, 2048)
            
            features_list.append(feat.squeeze())
            labels_list.append(label)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            
    # Save to HDF5
    print(f"Saving {len(features_list)} samples to {output_file}...")
    with h5py.File(output_file, "w") as f:
        f.create_dataset("features", data=np.array(features_list))
        f.create_dataset("labels", data=np.array(labels_list))
        
    print("Data preparation complete.")

if __name__ == "__main__":
    prepare_data()
