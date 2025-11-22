import os
import re
import time
import torch
import logging
import asyncio
from funasr import AutoModel

logger = logging.getLogger("vox-pathos")

class SenseVoiceService:
    def __init__(self, model_dir="iic/SenseVoiceSmall", device="cpu"):
        """
        Initializes the SenseVoice service.
        Args:
            model_dir (str): Path or ModelScope ID for SenseVoiceSmall.
            device (str): 'cpu' or 'cuda'.
        """
        self.device = device
        self.model_dir = model_dir
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            logger.info(f"Loading SenseVoice model from {self.model_dir} on {self.device}...")
            start_time = time.time()
            
            # Load model using FunASR's AutoModel
            self.model = AutoModel(
                model=self.model_dir,
                trust_remote_code=True,
                device=self.device,
                disable_update=True,
                disable_pbar=True
            )
            
            duration = time.time() - start_time
            logger.info(f"SenseVoice model loaded successfully in {duration:.2f}s.")
            
        except Exception as e:
            logger.error(f"Failed to load SenseVoice model: {e}")
            raise e

    def predict(self, audio_bytes: bytes) -> dict:
        """
        Runs inference on the provided audio bytes.
        Returns a dictionary with text, sentiment, and confidence.
        """
        if not self.model:
            logger.error("Model not initialized.")
            return {"text": "", "sentiment": "neutral", "confidence": 0.0}

        logger.info(f"SenseVoice Predict called with {len(audio_bytes)} bytes")

        try:
            # FunASR expects a file path or numpy array. 
            # For streaming/bytes, we might need to save to a temp file or use a specific input handler.
            # To keep it robust for this implementation, we'll save to a temp file.
            # Optimization: In the future, pass numpy array directly if supported.
            
            import tempfile
            import soundfile as sf
            import numpy as np
            
            # Convert bytes to numpy array (assuming 16kHz mono PCM 16-bit)
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            
            start_time = time.time()
            
            # Run Inference directly on numpy array
            # language="auto", use_itn=True for inverse text normalization (numbers, etc.)
            res = self.model.generate(
                input=audio_np,
                cache={},
                language="auto",
                use_itn=True,
                batch_size_s=60,
                merge_vad=False,
                merge_thr=1.0,
            )
            
            inference_time = time.time() - start_time
            
            # Parse Result
            # Result format is typically: [{'key': '...', 'text': '<|en|><|NEUTRAL|>... text ...'}]
            if not res or not isinstance(res, list):
                return {"text": "", "sentiment": "neutral", "confidence": 0.0}
            
            raw_text = res[0].get("text", "")
            
            # Extract Sentiment and Text
            # SenseVoice outputs tags like <|en|><|HAPPY|> or <|zh|><|SAD|>
            # We need to parse these.
            
            sentiment, clean_text = self._parse_output(raw_text)
            
            return {
                "text": clean_text,
                "sentiment": sentiment,
                "confidence": 0.95, # Placeholder, SenseVoice doesn't always return confidence per tag easily
                "processing_time": inference_time,
                "raw_output": raw_text
            }

        except Exception as e:
            logger.error(f"Inference error: {e}")
            return {"text": "", "sentiment": "error", "confidence": 0.0}

    def _parse_output(self, raw_text: str):
        """
        Parses SenseVoice output to extract sentiment and clean text.
        Example raw: "<|en|><|HAPPY|>Hello world"
        """
        # 1. Define Emotion Tags Mapping
        # SenseVoice typically uses: <|HAPPY|>, <|SAD|>, <|ANGRY|>, <|NEUTRAL|>, <|FEARFUL|>, <|DISGUSTED|>, <|SURPRISED|>
        # We map them to our 3-class system: positive, negative, neutral
        
        tag_map = {
            "<|HAPPY|>": "positive",
            "<|SAD|>": "negative",
            "<|ANGRY|>": "negative",
            "<|NEUTRAL|>": "neutral",
            "<|FEARFUL|>": "negative",
            "<|DISGUSTED|>": "negative",
            "<|SURPRISED|>": "positive", # Context dependent, but often positive in casual speech
        }
        
        detected_sentiment = "neutral"
        
        # Find tags
        for tag, sentiment in tag_map.items():
            if tag in raw_text:
                detected_sentiment = sentiment
                break # Prioritize the first tag found? Or specific hierarchy?
        
        # Clean Text
        # Remove all tags like <|en|>, <|HAPPY|>, etc.
        clean_text = re.sub(r"<\|.*?\|>", "", raw_text).strip()
        
        return detected_sentiment, clean_text
