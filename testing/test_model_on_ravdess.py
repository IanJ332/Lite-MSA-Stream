import sys
import os

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
os.chdir(project_root)

import glob
import random
import kagglehub
import logging
import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
import soundfile as sf
import string
from difflib import SequenceMatcher
from app.services.acoustic_analyzer import AcousticAnalyzer
from app.services.transcription_service import TranscriptionService

def text_similarity(a, b):
    # Remove punctuation and lower case
    a_clean = a.lower().translate(str.maketrans('', '', string.punctuation))
    b_clean = b.lower().translate(str.maketrans('', '', string.punctuation))
    return SequenceMatcher(None, a_clean, b_clean).ratio()

def main():
    print("=== RAVDESS Full Benchmark (Acoustic + ASR) ===")
    
    # 1. Download Data
    try:
        path = kagglehub.dataset_download("uwrfkaggler/ravdess-emotional-speech-audio")
    except Exception as e:
        print(f"Download failed: {e}")
        return

    # 2. Find Files
    wav_files = glob.glob(os.path.join(path, "**/*.wav"), recursive=True)
    if not wav_files:
        print("No wav files found!")
        return

    # 3. Use ALL Samples (No sampling)
    test_files = wav_files
    # test_files = random.sample(wav_files, 10) # Debug: Uncomment for quick test

    # 4. Init Analyzers
    try:
        acoustic_analyzer = AcousticAnalyzer()
        asr_service = TranscriptionService()
        print("✅ Analyzers initialized.")
    except Exception as e:
        print(f"❌ Analyzer init failed: {e}")
        return
    
    # 5. Test Loop
    emotion_map = {
        '01': 'neutral', '02': 'neutral',
        '03': 'positive', '08': 'positive',
        '04': 'negative', '05': 'negative', '06': 'negative', '07': 'negative'
    }
    
    statement_map = {
        '01': "kids are talking by the door",
        '02': "dogs are sitting by the door"
    }

    correct_emotion = 0
    total_asr_similarity = 0.0
    total = 0
    
    print(f"Testing {len(test_files)} samples...")
    
    for i, file_path in enumerate(test_files):
        filename = os.path.basename(file_path)
        parts = filename.split('-')
        if len(parts) != 7: continue
        
        # Parse Ground Truth
        emotion_code = parts[2]
        statement_code = parts[4]
        
        if emotion_code not in emotion_map: continue
        truth_emotion = emotion_map[emotion_code]
        
        truth_text = statement_map.get(statement_code, "")
        
        try:
            # Load with SoundFile
            data, sr = sf.read(file_path)
            
            # --- Acoustic Prediction ---
            waveform = torch.from_numpy(data).float()
            if waveform.ndim == 1: waveform = waveform.unsqueeze(0)
            else: waveform = waveform.t()
            
            # Mono
            if waveform.shape[0] > 1: waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample for Acoustic (16k)
            if sr != 16000:
                resampler = T.Resample(sr, 16000)
                waveform_16k = resampler(waveform)
            else:
                waveform_16k = waveform
                
            # To Bytes for Acoustic
            waveform_int16 = (waveform_16k * 32767).clamp(-32768, 32767).to(torch.int16)
            pcm_bytes = waveform_int16.numpy().tobytes()
            
            # Predict Emotion
            res = acoustic_analyzer.predict(pcm_bytes)
            if "error" in res: continue
            
            pred_emotion = res['sentiment']
            
            # --- ASR Prediction ---
            # TranscriptionService also expects bytes, usually 16k mono int16 is fine
            # We can reuse pcm_bytes
            pred_text = asr_service.transcribe(pcm_bytes)
            
            # --- Metrics ---
            # Emotion
            is_emotion_correct = (pred_emotion == truth_emotion)
            if is_emotion_correct: correct_emotion += 1
            
            # ASR
            similarity = text_similarity(truth_text, pred_text)
            total_asr_similarity += similarity
            
            total += 1
            
            # Log every 10 files or if error
            if i % 10 == 0:
                print(f"[{i}/{len(test_files)}] {filename}")
                print(f"  Emotion: Truth={truth_emotion} | Pred={pred_emotion} | {'✅' if is_emotion_correct else '❌'}")
                print(f"  Text:    Truth='{truth_text}'")
                print(f"           Pred ='{pred_text}' | Sim={similarity:.2f}")
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            
    if total > 0:
        avg_asr_acc = (total_asr_similarity / total) * 100
        emotion_acc = (correct_emotion / total) * 100
        
        print(f"\n" + "="*40)
        print(f"       FULL BENCHMARK RESULTS ({total} files)")
        print(f"="*40)
        print(f"Text Recognition Accuracy (Avg Sim): {avg_asr_acc:.2f}%")
        print(f"Emotion Prediction Accuracy:         {emotion_acc:.2f}%")
        print(f"="*40)
    else:
        print("No valid samples processed.")

if __name__ == "__main__":
    main()
