import logging
import os
import numpy as np
import whisper
import torch

logger = logging.getLogger(__name__)

class TranscriptionService:
    def __init__(self, model_size="tiny.en", device="cpu", compute_type="int8"):
        """
        Initialize OpenAI Whisper ASR model.
        Args:
            model_size (str): Model size (tiny, base, small, medium, large). 
                              "tiny.en" is recommended for CPU real-time.
            device (str): "cpu" or "cuda".
            compute_type (str): Ignored for standard Whisper on CPU (uses FP32 usually).
        """
        self.model_size = model_size
        self.device = device
        
        logger.info(f"Loading OpenAI Whisper model: {model_size} ({device})...")
        try:
            # Load model
            self.model = whisper.load_model(model_size, device=device)
            logger.info("OpenAI Whisper model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load OpenAI Whisper model: {e}")
            raise

    def transcribe(self, audio_bytes: bytes) -> str:
        """
        Transcribe raw audio bytes to text.
        Args:
            audio_bytes (bytes): Raw PCM audio data (16kHz, Mono, Int16).
        Returns:
            str: Transcribed text.
        """
        try:
            # Convert bytes to float32 numpy array (normalized to [-1, 1])
            audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
            audio_float32 = audio_int16.astype(np.float32) / 32768.0
            
            # Run transcription
            # fp16=False is important for CPU
            result = self.model.transcribe(audio_float32, fp16=False, language="en")
            
            text = result["text"].strip()
            
            if text:
                logger.info(f"ASR Output: '{text}'")
            
            return text
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return ""
