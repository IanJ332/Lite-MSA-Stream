import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import glob
import h5py
import numpy as np
import librosa
from tqdm import tqdm
from app.services.ensemble_service import EnsembleService

# Unified Emotion Map (6 Classes)
# 0: neutral, 1: happy, 2: sad, 3: angry, 4: fearful, 5: disgusted
EMOTION_MAP = {
    "neutral": 0,
    "happy": 1,
    "sad": 2,
    "angry": 3,
    "fearful": 4,
    "disgusted": 5
}

def get_ravdess_info(filename):
    # 03-01-06-01-02-01-01.wav
    parts = filename.split("-")
    if len(parts) < 7: return None, None
    
    emotion_code = parts[2]
    speaker_id = parts[6].split(".")[0]
    
    # Map RAVDESS codes
    # 01=neutral, 03=happy, 04=sad, 05=angry, 06=fearful, 07=disgust
    # Skip: 02 (calm), 08 (surprised)
    mapping = {
        '01': 'neutral',
        '03': 'happy',
        '04': 'sad',
        '05': 'angry',
        '06': 'fearful',
        '07': 'disgusted'
    }
    
    if emotion_code in mapping:
        return mapping[emotion_code], f"RAVDESS_{speaker_id}"
    return None, None

def get_tess_info(filepath):
    # tess_data/OAF_angry/OAF_back_angry.wav
    # Folder name contains emotion: OAF_angry
    folder = os.path.basename(os.path.dirname(filepath))
    filename = os.path.basename(filepath)
    
    # Speaker is OAF or YAF (first part of folder or filename)
    speaker_id = "TESS_" + filename.split("_")[0]
    
    # Emotion is in folder name (e.g., OAF_angry -> angry)
    # TESS folders: OAF_angry, OAF_disgust, OAF_Fear, OAF_happy, OAF_neutral, OAF_Pleasant_surprise, OAF_Sad
    emotion_part = folder.split("_")[1].lower()
    
    mapping = {
        'neutral': 'neutral',
        'happy': 'happy',
        'sad': 'sad',
        'angry': 'angry',
        'fear': 'fearful',
        'disgust': 'disgusted'
    }
    # Skip 'pleasant_surprise' (ps)
    
    if emotion_part in mapping:
        return mapping[emotion_part], speaker_id
    return None, None

def get_crema_info(filename):
    # 1001_DFA_ANG_XX.wav
    parts = filename.split("_")
    if len(parts) < 3: return None, None
    
    speaker_id = "CREMA_" + parts[0]
    emotion_code = parts[2]
    
    # Map CREMA codes
    # ANG, DIS, FEA, HAP, NEU, SAD
    mapping = {
        'NEU': 'neutral',
        'HAP': 'happy',
        'SAD': 'sad',
        'ANG': 'angry',
        'FEA': 'fearful',
        'DIS': 'disgusted'
    }
    
    if emotion_code in mapping:
        return mapping[emotion_code], speaker_id
    return None, None

def augment_audio(y, sr):
    """Generate augmented versions of audio."""
    augmented = []
    
    # 1. Add Noise (Random SNR)
    noise = np.random.randn(len(y))
    noise_factor = np.random.uniform(0.001, 0.015)
    y_noise = y + noise_factor * noise
    augmented.append(y_noise)
    
    # 2. Pitch Shift (Lower)
    try:
        y_pitch_down = librosa.effects.pitch_shift(y, sr=sr, n_steps=-2)
        augmented.append(y_pitch_down)
    except:
        pass
        
    return augmented

def prepare_data():
    print("Initializing Ensemble Service...")
    service = EnsembleService(device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu")
    
    datasets = [
        ("ravdess_data", get_ravdess_info),
        ("tess_data", get_tess_info),
        ("crema_data", get_crema_info)
    ]
    
    features_list = []
    labels_list = []
    speakers_list = [] # For split
    
    output_file = "all_features.h5"
    
    for data_dir, info_func in datasets:
        print(f"Scanning {data_dir}...")
        files = glob.glob(os.path.join(data_dir, "**/*.wav"), recursive=True)
        print(f"Found {len(files)} files in {data_dir}")
        
        for file_path in tqdm(files):
            try:
                filename = os.path.basename(file_path)
                if info_func == get_tess_info:
                    emotion, speaker = info_func(file_path)
                else:
                    emotion, speaker = info_func(filename)
                
                if emotion is None: continue
                
                label = EMOTION_MAP[emotion]
                
                # Load Audio
                y, sr = librosa.load(file_path, sr=16000)
                
                # Extract Features (Original)
                feat = service.extract_features(y)
                features_list.append(feat.squeeze())
                labels_list.append(label)
                speakers_list.append(speaker)
                
                # Augmentation (Only for Training Speakers? No, do all, split later)
                # Actually, augmentation triples the size. 
                # TESS+CREMA+RAVDESS is big. Let's ONLY augment RAVDESS and TESS (Clean datasets).
                # CREMA-D is already noisy/diverse enough? 
                # User said: "Since RAVDESS is too theatrical... add noise".
                # Let's augment everything to be safe, but maybe lighter on CREMA?
                # For simplicity and robustness, augment all.
                
                # SKIP Augmentation for now to save time/space? 
                # User specifically asked for "Add Noise".
                # Let's add noise ONLY. Pitch shift is slow.
                
                # 1. Add Noise
                noise = np.random.randn(len(y))
                y_noise = y + 0.005 * noise
                feat_noise = service.extract_features(y_noise)
                features_list.append(feat_noise.squeeze())
                labels_list.append(label)
                speakers_list.append(speaker)
                
            except Exception as e:
                # print(f"Error: {e}")
                pass

    print(f"Saving {len(features_list)} samples to {output_file}...")
    
    # Save string list as fixed-length numpy bytes
    dt = h5py.special_dtype(vlen=str)
    
    with h5py.File(output_file, "w") as f:
        f.create_dataset("features", data=np.array(features_list))
        f.create_dataset("labels", data=np.array(labels_list))
        
        # Save speakers as strings
        ds = f.create_dataset("speakers", (len(speakers_list),), dtype=dt)
        ds[:] = speakers_list
        
    print("Data preparation complete.")

if __name__ == "__main__":
    prepare_data()
