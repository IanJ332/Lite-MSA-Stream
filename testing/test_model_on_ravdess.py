import os
import glob
import random
import kagglehub
import logging
import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
import soundfile as sf
from app.services.acoustic_analyzer import AcousticAnalyzer

def main():
    print("=== RAVDESS Model Verification (Restored) ===")
    
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

    # 3. Select 50 Random Samples
    test_files = random.sample(wav_files, 50) if len(wav_files) > 50 else wav_files

    # 4. Init Analyzer
    try:
        analyzer = AcousticAnalyzer()
        print("✅ Analyzer initialized.")
    except Exception as e:
        print(f"❌ Analyzer init failed: {e}")
        return
    
    # 5. Test Loop
    emotion_map = {
        '01': 'neutral', '02': 'neutral',
        '03': 'positive', '08': 'positive',
        '04': 'negative', '05': 'negative', '06': 'negative', '07': 'negative'
    }

    correct = 0
    total = 0
    
    print(f"Testing {len(test_files)} samples...")
    
    for file_path in test_files:
        filename = os.path.basename(file_path)
        parts = filename.split('-')
        if len(parts) != 7: continue
        
        code = parts[2]
        if code not in emotion_map: continue
        truth = emotion_map[code]
        
        try:
            # Load with SoundFile
            data, sr = sf.read(file_path)
            waveform = torch.from_numpy(data).float()
            if waveform.ndim == 1: waveform = waveform.unsqueeze(0)
            else: waveform = waveform.t()
            
            # Mono
            if waveform.shape[0] > 1: waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample
            if sr != 16000:
                resampler = T.Resample(sr, 16000)
                waveform = resampler(waveform)
                
            # To Bytes
            waveform_int16 = (waveform * 32767).clamp(-32768, 32767).to(torch.int16)
            pcm_bytes = waveform_int16.numpy().tobytes()
            
            # Predict
            res = analyzer.predict(pcm_bytes)
            if "error" in res: continue
            
            pred = res['sentiment']
            if pred == truth: correct += 1
            total += 1
            
            print(f"File: {filename} | Truth: {truth} | Pred: {pred} ({res['confidence']:.2f}) | {'✅' if pred==truth else '❌'}")
            
        except Exception as e:
            print(f"Error: {e}")
            
    if total > 0:
        print(f"\n=== Final Accuracy: {correct/total*100:.2f}% ({correct}/{total}) ===")
    else:
        print("No valid samples.")

if __name__ == "__main__":
    main()
