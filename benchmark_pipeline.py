import os
import glob
import asyncio
import numpy as np
import torch
import soundfile as sf
from scipy import signal
from difflib import SequenceMatcher
import string
from app.services.transcription_service import TranscriptionService
from app.services.acoustic_analyzer import AcousticAnalyzer
from app.services.text_sentiment_analyzer import TextSentimentAnalyzer

# Constants
DATA_DIR = "ravdess_data"
ALPHA = 0.7 # Fusion Weight for Text
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

async def run_benchmark():
    print("--- Initializing Services ---")
    asr_service = TranscriptionService()
    acoustic_service = AcousticAnalyzer()
    text_service = TextSentimentAnalyzer()
    print("--- Services Ready ---\n")
    
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

    print(f"Starting Benchmark on {len(wav_files)} files...")
    
    # Limit to subset for speed if needed, e.g., first 100
    wav_files = wav_files[:100] 
    
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
            pcm_data = (waveform * 32767).clip(-32768, 32767).astype(np.int16)
            audio_bytes = pcm_data.tobytes()
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue
            
        # 2. Run Pipeline
        # ASR
        transcript = asr_service.transcribe(audio_bytes)
        
        # Acoustic
        acoustic_result = acoustic_service.predict(audio_bytes)
        
        # Text Sentiment
        text_result = text_service.predict(transcript)
        
        # Fusion
        fused_scores = {}
        for label in ["positive", "negative", "neutral"]:
            text_p = text_result["scores"].get(label, 0.0)
            audio_p = acoustic_result["scores"].get(label, 0.0)
            fused_scores[label] = (ALPHA * text_p) + ((1 - ALPHA) * audio_p)
        
        pred_emotion = max(fused_scores, key=fused_scores.get)
        
        # 3. Calculate Metrics
        # ASR
        sim_score = text_similarity(truth_text, transcript)
        total_asr_score += sim_score
        
        # Sentiment
        if pred_emotion == truth_emotion:
            correct_sentiment += 1
        
        confusion[truth_emotion][pred_emotion] += 1
        
        print(f"[{total_files}] {filename} | Truth: {truth_emotion} | Pred: {pred_emotion} | ASR Sim: {sim_score:.2f}")
        print(f"    Transcript: {transcript}")

    # Final Report
    avg_asr = total_asr_score / total_files if total_files > 0 else 0
    acc_sentiment = correct_sentiment / total_files if total_files > 0 else 0
    
    print("\n" + "="*40)
    print("       BENCHMARK RESULTS       ")
    print("="*40)
    print(f"Total Files Processed: {total_files}")
    print(f"Speech Recognition Accuracy (Similarity): {avg_asr*100:.2f}%")
    print(f"Emotion Recognition Accuracy: {acc_sentiment*100:.2f}%")
    print("-" * 40)
    print("Confusion Matrix (Truth \\ Pred):")
    print(f"{'':<10} {'Pos':<8} {'Neu':<8} {'Neg':<8}")
    for truth in ['positive', 'neutral', 'negative']:
        row = confusion[truth]
        print(f"{truth:<10} {row['positive']:<8} {row['neutral']:<8} {row['negative']:<8}")
    print("="*40)

if __name__ == "__main__":
    asyncio.run(run_benchmark())
