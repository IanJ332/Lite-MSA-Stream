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
            
            # Load model using FunASR's AutoModel
            self.model = AutoModel(
                model=self.model_dir,
                trust_remote_code=True,
                device=self.device,
                disable_update=True,
            )
            
            start_time = time.time()
            duration = time.time() - start_time
            logger.info(f"SenseVoice model loaded successfully in {duration:.2f}s.")

            try:
                import soundfile as sf
                import numpy as np
                from scipy import signal
                import glob
                import os

                # Find a warmup file (use a RAVDESS file if available, or a bundled warmup.wav)
                warmup_file = "warmup.wav"
                if not os.path.exists(warmup_file):
                    # Fallback to finding a RAVDESS file
                    ravdess_files = glob.glob("ravdess_data/**/*.wav", recursive=True)
                    if ravdess_files:
                        warmup_file = ravdess_files[0]
                
                if os.path.exists(warmup_file):
                    print(f"DEBUG: Using warmup file: {warmup_file}")
                    audio_warmup, sr_warmup = sf.read(warmup_file, dtype='float32')
                    
                    # Handle multi-channel
                    if len(audio_warmup.shape) > 1:
                        audio_warmup = np.mean(audio_warmup, axis=1)
                        
                    # Resample to 16000 Hz
                    if sr_warmup != 16000:
                        num_samples = int(len(audio_warmup) * 16000 / sr_warmup)
                        audio_warmup = signal.resample(audio_warmup, num_samples)
                    
                    # Run inference
                    self.model.generate(
                        input=audio_warmup,
                        cache={},
                        language="auto",
                        use_itn=True
                    )
                    print("DEBUG: Warmup completed successfully.")
                else:
                    print("DEBUG: No warmup file found. Emotion recognition may be unstable.")
                    
            except Exception as w_e:
                print(f"DEBUG: Warmup failed: {w_e}")
            
        except Exception as e:
            logger.error(f"Failed to load SenseVoice model: {e}")
            raise e


    def predict(self, audio_input) -> dict:
        """
        Runs inference on the provided audio input.
        Args:
            audio_input: bytes (PCM 16-bit) or np.ndarray (float32)
        Returns a dictionary with text, sentiment, and confidence.
        """
        if not self.model:
            logger.error("Model not initialized.")
            return {"text": "", "sentiment": "neutral", "confidence": 0.0}

        input_len = len(audio_input) if isinstance(audio_input, (bytes, str)) else audio_input.shape[0]
        logger.info(f"SenseVoice Predict called with input length {input_len}")

        try:
            import numpy as np
            
            start_time = time.time()
            
            # Handle input type
            if isinstance(audio_input, bytes):
                # Assume Float32 bytes from frontend or VADIterator
                audio_np = np.frombuffer(audio_input, dtype=np.float32)
            elif isinstance(audio_input, np.ndarray):
                # Assume already float32 and normalized
                audio_np = audio_input.astype(np.float32)
            else:
                logger.error(f"Unsupported input type: {type(audio_input)}")
                return {"text": "", "sentiment": "error", "confidence": 0.0}

            # Run Inference directly on numpy array
            # language="auto", use_itn=True for inverse text normalization (numbers, etc.)
            # DEBUG: Check audio stats
            print(f"DEBUG: audio_np stats: min={audio_np.min()}, max={audio_np.max()}, mean={audio_np.mean()}, shape={audio_np.shape}", flush=True)
            
            res = self.model.generate(
                input=audio_np,
                cache={},
                language="auto",
                use_itn=True,
            )
            
            print(f"DEBUG: Raw model result: {res}", flush=True)
            
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
        # SenseVoice uses: <|HAPPY|>, <|SAD|>, <|ANGRY|>, <|NEUTRAL|>, <|FEARFUL|>, <|DISGUSTED|>, <|SURPRISED|>
        # We map them to specific emotions as requested by the user.
        
        tag_map = {
            "<|HAPPY|>": "happy",
            "<|SAD|>": "sad",
            "<|ANGRY|>": "angry",
            "<|NEUTRAL|>": "neutral",
            "<|FEARFUL|>": "fearful",
            "<|DISGUSTED|>": "disgusted",
            "<|SURPRISED|>": "surprised",
        }
        
        detected_sentiment = "neutral"
        
        # Find tags
        for tag, sentiment in tag_map.items():
            if tag in raw_text:
                detected_sentiment = sentiment
                break 
        
        # Clean Text
        # Remove all tags like <|en|>, <|HAPPY|>, etc.
        clean_text = re.sub(r"<\|.*?\|>", "", raw_text).strip()
        
        return detected_sentiment, clean_text
