import onnxruntime as ort
import numpy as np
import logging
from app.utils.model_utils import download_model

logger = logging.getLogger(__name__)

class VADIterator:
    def __init__(self, threshold: float = 0.002, sampling_rate: int = 16000, min_silence_duration_ms: int = 500):
        self.threshold = threshold
        self.sampling_rate = sampling_rate
        self.min_silence_samples = min_silence_duration_ms * sampling_rate / 1000
        
        # State
        self.triggered = False
        self.current_speech = bytearray()
        self.temp_end = 0
        self.h = np.zeros((2, 1, 64), dtype=np.float32)
        self.c = np.zeros((2, 1, 64), dtype=np.float32)
        
        # Load Model
        # Using onnx-community/silero-vad as the reliable source
        model_path = download_model(
            repo_id="onnx-community/silero-vad",
            filename="onnx/model.onnx", 
            local_dir="models"
        )
        
        # Configure Session Options for Concurrency
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        # Disable graph optimizations to prevent potential fusion errors
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        
        self.session = ort.InferenceSession(model_path, sess_options)
        logger.info("VAD Model loaded successfully with single-thread config.")

    def reset_states(self):
        self.h = np.zeros((2, 1, 64), dtype=np.float32)
        self.c = np.zeros((2, 1, 64), dtype=np.float32)
        self.triggered = False
        self.current_speech = bytearray()
        self.temp_end = 0

    def process(self, audio_chunk: bytes):
        """
        Process an audio chunk and return a speech segment if a sentence is completed.
        
        Args:
            audio_chunk (bytes): Raw PCM audio data (16kHz, Mono, Int16).
            
        Returns:
            bytes: Complete speech segment if detected, else None.
        """
        # Convert bytes to float32 numpy array
        audio_int16 = np.frombuffer(audio_chunk, dtype=np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        
        # Add batch dimension: (1, N)
        input_tensor = audio_float32[np.newaxis, :]
        
        # Run Inference
        # Silero VAD v5 (onnx-community) uses 'state' (2, 1, 128) instead of h/c
        # We need to check if the model expects 'state' or 'h'/'c'.
        # Based on the error "Required inputs (['state']) are missing", it expects 'state'.
        
        # Initialize state if not compatible or first run
        if not hasattr(self, 'state') or self.state.shape != (2, 1, 128):
             self.state = np.zeros((2, 1, 128), dtype=np.float32)

        ort_inputs = {
            "input": input_tensor,
            "state": self.state,
            "sr": np.array(self.sampling_rate, dtype=np.int64)
        }
        
        # Output is (output, state)
        out, self.state = self.session.run(None, ort_inputs)
        speech_prob = out[0][0]
        
        # Debug logging to see if VAD is detecting anything
        logger.info(f"VAD Prob: {speech_prob:.4f}") 
        
        # State Machine Logic
        if speech_prob >= self.threshold:
            # Speech detected
            if not self.triggered:
                logger.debug("VAD: Speech started.")
                self.triggered = True
            self.current_speech.extend(audio_chunk)
            self.temp_end = 0
        elif self.triggered:
            # Silence during speech
            self.current_speech.extend(audio_chunk)
            self.temp_end += len(audio_int16)
            
            if self.temp_end >= self.min_silence_samples:
                logger.debug("VAD: Speech ended (silence limit reached).")
                # Return the full segment (trimming trailing silence could be added here)
                segment = bytes(self.current_speech)
                self.reset_states()
                return segment
                
        return None
