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
    
    import librosa
    
    def augment_audio(y, sr):
        """Generate augmented versions of audio."""
        augmented = []
        
        # 1. Add Noise (SNR ~30dB)
        noise = np.random.randn(len(y))
        y_noise = y + 0.005 * noise
        augmented.append(y_noise)
        
        # 2. Pitch Shift (Lower)
        try:
            y_pitch_down = librosa.effects.pitch_shift(y, sr=sr, n_steps=-2)
            augmented.append(y_pitch_down)
        except:
            pass
            
        # 3. Pitch Shift (Higher)
        try:
            y_pitch_up = librosa.effects.pitch_shift(y, sr=sr, n_steps=2)
            augmented.append(y_pitch_up)
        except:
            pass
            
        return augmented

    count = 0
    for file_path in tqdm(wav_files):
        if LIMIT is not None and count >= LIMIT:
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
            
            # Load Audio (Using librosa for consistency with augmentation)
            # Librosa loads as float32, normalized, mono by default
            y, sr = librosa.load(file_path, sr=16000)
            
            # Process Original
            feat = service.extract_features(y)
            features_list.append(feat.squeeze())
            labels_list.append(label)
            
            # Process Augmented
            augmented_versions = augment_audio(y, sr)
            for y_aug in augmented_versions:
                feat_aug = service.extract_features(y_aug)
                features_list.append(feat_aug.squeeze())
                labels_list.append(label)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
            
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
