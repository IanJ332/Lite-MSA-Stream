import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import glob
import asyncio
import numpy as np
import torch
import soundfile as sf
from scipy import signal
from difflib import SequenceMatcher
import string
from app.services.sensevoice_service import SenseVoiceService

# Constants
DATA_DIR = "ravdess_data"
TARGET_SR = 16000

# Mappings
EMOTION_MAP = {
    '01': 'neutral',  # neutral
    '02': 'neutral',  # calm
    '03': 'positive', # happy
    '04': 'negative', # sad
    '05': 'negative', # angry
    '06': 'negative', # fearful
    '07': 'negative', # disgust
    '08': 'positive'  # surprised
}

STATEMENT_MAP = {
    '01': "kids are talking by the door",
    '02': "dogs are sitting by the door"
}

def get_ground_truth(filename):
    # Filename format: 03-01-06-01-02-01-12.wav
    parts = filename.split('.')[0].split('-')
    if len(parts) != 7:
        return None, None
    
    emotion_code = parts[2]
    statement_code = parts[4]
    
    emotion = EMOTION_MAP.get(emotion_code)
    statement = STATEMENT_MAP.get(statement_code)
    
    return emotion, statement

def text_similarity(a, b):
    # Remove punctuation and lower case
    a_clean = a.lower().translate(str.maketrans('', '', string.punctuation))
    b_clean = b.lower().translate(str.maketrans('', '', string.punctuation))
    return SequenceMatcher(None, a_clean, b_clean).ratio()

    print("Benchmark complete. Results saved to benchmark_results_direct.txt")

if __name__ == "__main__":
    # asyncio.run(run_benchmark())
    # Run synchronously to avoid potential asyncio/funasr conflicts
    import asyncio
    # Hack to run async function synchronously if needed, but better to make it sync.
    # Since SenseVoiceService is sync, we can just run the body.
    # But run_benchmark is async def.
    # Let's just run it with asyncio.run but ensure no other async stuff is happening?
    # Wait, I want to REMOVE asyncio.
    
    # Redefine run_benchmark as sync
    pass

def run_benchmark_sync():
    print("--- Initializing SenseVoice Service ---")
    sensevoice_service = SenseVoiceService(device="cpu")
    print("--- Service Ready ---\n")
    
    wav_files = glob.glob(os.path.join(DATA_DIR, "*.wav"))
    if not wav_files:
        print(f"No wav files found in {DATA_DIR}. Please run download_dataset.py first.")
        return

    # Metrics
    total_files = 0
    correct_sentiment = 0
    total_asr_score = 0.0
    
    # Confusion Matrix: {Truth: {Pred: Count}}
    confusion = {
        'positive': {'positive': 0, 'neutral': 0, 'negative': 0},
        'neutral': {'positive': 0, 'neutral': 0, 'negative': 0},
        'negative': {'positive': 0, 'neutral': 0, 'negative': 0}
    }

    # Filter for specific debug file
    target_file = "03-01-05-01-01-01-01.wav"
    wav_files = [f for f in wav_files if os.path.basename(f) == target_file]
    
    if not wav_files:
        print(f"Target file {target_file} not found!")
        return 
    
    for file_path in wav_files:
        filename = os.path.basename(file_path)
        truth_emotion, truth_text = get_ground_truth(filename)
        
        if not truth_emotion or not truth_text:
            continue
            
        total_files += 1
        
        # 1. Load Audio using soundfile (avoids torchcodec issues)
        try:
            # Load audio with soundfile
            waveform, sr = sf.read(file_path, dtype='float32')
            
            # Convert to mono if stereo
            if len(waveform.shape) > 1:
                waveform = np.mean(waveform, axis=1)
            
            # Resample to 16kHz if needed
            if sr != TARGET_SR:
                num_samples = int(len(waveform) * TARGET_SR / sr)
                waveform = signal.resample(waveform, num_samples)
                
            # Convert to Int16 PCM bytes (what the services expect)
            # waveform is already float32 in range [-1, 1]
            # pcm_data = (waveform * 32767).clip(-32768, 32767).astype(np.int16)
            # audio_bytes = pcm_data.tobytes()
            
            # 2. Run Pipeline (Unified)
            # Pass waveform directly to avoid Int16 quantization loss on CPU
            result = sensevoice_service.predict(waveform)
            
            transcript = result["text"]
            pred_emotion = result["sentiment"]
            
            # Map 'error' or unknown sentiments to neutral to avoid crash
            if pred_emotion not in confusion:
                pred_emotion = 'neutral'
            
        except Exception as e:
            print(f"Error processing {filename}: {e}", flush=True)
            continue
            
        # 3. Calculate Metrics
        # ASR
        sim_score = text_similarity(truth_text, transcript)
        total_asr_score += sim_score
        
        # Sentiment
        if pred_emotion == truth_emotion:
            correct_sentiment += 1
        
        confusion[truth_emotion][pred_emotion] += 1
        
        print(f"[{total_files}] {filename} | Truth: {truth_emotion} | Pred: {pred_emotion} | ASR Sim: {sim_score:.2f}", flush=True)
        print(f"    Transcript: {transcript}", flush=True)

    # Final Report
    avg_asr = total_asr_score / total_files if total_files > 0 else 0
    acc_sentiment = correct_sentiment / total_files if total_files > 0 else 0
    
    with open("benchmark_results_direct.txt", "w", encoding="utf-8") as f:
        f.write("\n" + "="*40 + "\n")
        f.write("       BENCHMARK RESULTS       \n")
        f.write("="*40 + "\n")
        f.write(f"Total Files Processed: {total_files}\n")
        f.write(f"Speech Recognition Accuracy (Similarity): {avg_asr*100:.2f}%\n")
        f.write(f"Emotion Recognition Accuracy: {acc_sentiment*100:.2f}%\n")
        f.write("-" * 40 + "\n")
        f.write("Confusion Matrix (Truth \\ Pred):\n")
        f.write(f"{'':<10} {'Pos':<8} {'Neu':<8} {'Neg':<8}\n")
        for truth in ['positive', 'neutral', 'negative']:
            row = confusion[truth]
            f.write(f"{truth:<10} {row['positive']:<8} {row['neutral']:<8} {row['negative']:<8}\n")
        f.write("="*40 + "\n")
    
    print("Benchmark complete. Results saved to benchmark_results_direct.txt")

if __name__ == "__main__":
    run_benchmark_sync()
