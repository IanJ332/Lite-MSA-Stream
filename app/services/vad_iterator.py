import onnxruntime as ort
import numpy as np
import logging
from app.utils.model_utils import download_model

logger = logging.getLogger(__name__)

class VADIterator:
    def __init__(self, threshold: float = 0.5, sampling_rate: int = 16000, min_silence_duration_ms: int = 500, min_speech_duration_ms: int = 250, max_speech_duration_s: float = 10.0):
        self.threshold = threshold
        self.sampling_rate = sampling_rate
        self.min_silence_samples = min_silence_duration_ms * sampling_rate / 1000
        self.min_speech_samples = min_speech_duration_ms * sampling_rate / 1000
        self.max_speech_samples = max_speech_duration_s * sampling_rate
        
        # State
        self.triggered = False
        self.current_speech = bytearray()
        self.temp_end = 0
        
        # Initialize model state (2, 1, 128) for Silero V5
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        
        # Load Model
        model_path = download_model(
            repo_id="onnx-community/silero-vad",
            filename="onnx/model.onnx", 
            local_dir="models"
        )
        
        # Configure Session Options
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session = ort.InferenceSession(model_path, sess_options)
        logger.info("VAD Model loaded successfully.")

    def reset_states(self):
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self.triggered = False
        self.current_speech = bytearray()
        self.temp_end = 0

    def set_threshold(self, threshold: float):
        """
        Dynamically update the VAD threshold.
        Args:
            threshold (float): New threshold (0.0 to 1.0). Lower = More Sensitive (High CPU).
        """
        self.threshold = max(0.01, min(0.99, threshold))
        logger.info(f"VAD Threshold updated to {self.threshold}")

    def process(self, audio_chunk: bytes):
        """
        Process an audio chunk and return a speech segment if a sentence is completed.
        
        Args:
            audio_chunk (bytes): Raw PCM audio data (16kHz, Mono, Float32).
            
        Returns:
            tuple: (bytes or None, float probability)
        """
        # Convert bytes to float32 numpy array
        # Input is now raw Float32 bytes from frontend
        audio_float32 = np.frombuffer(audio_chunk, dtype=np.float32)
        
        # Add batch dimension: (1, N)
        input_tensor = audio_float32[np.newaxis, :]
        
        # Ensure state shape is correct
        if self.state.shape != (2, 1, 128):
             self.state = np.zeros((2, 1, 128), dtype=np.float32)

        # 'sr' must be a scalar (0-D tensor) for this model version
        sr_tensor = np.array(self.sampling_rate, dtype=np.int64)

        ort_inputs = {
            "input": input_tensor,
            "state": self.state,
            "sr": sr_tensor
        }
        
        speech_prob = 0.0
        try:
            # Output is (output, state)
            out, self.state = self.session.run(None, ort_inputs)
            speech_prob = out[0][0]
            
            # Debug logging for VAD probability (every ~1 second)
            if np.random.rand() < 0.05: 
                logger.debug(f"VAD Prob: {speech_prob:.4f} | Triggered: {self.triggered}")
        except Exception as e:
            logger.error(f"VAD Inference Error: {e}")
            self.reset_states()
            return None, 0.0
        
        # State Machine Logic
        segment = None
        
        # 1. Check Max Duration (Force Flush)
        # len(current_speech) is in bytes (float32 = 4 bytes)
        current_samples = len(self.current_speech) / 4
        if current_samples >= self.max_speech_samples:
            logger.info(f"VAD: Max duration reached ({current_samples/self.sampling_rate:.2f}s). Forcing flush.")
            segment = bytes(self.current_speech)
            self.reset_states()
            return segment, float(speech_prob)

        # 2. VAD Trigger Logic
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
            self.temp_end += len(audio_float32)
            
            if self.temp_end >= self.min_silence_samples:
                # Silence limit reached, check if speech was long enough
                total_samples = len(self.current_speech) / 4
                
                # Actual speech duration = Total - Silence
                speech_duration_samples = total_samples - self.temp_end
                
                if speech_duration_samples >= self.min_speech_samples:
                    logger.info(f"VAD: Speech ended. Duration: {speech_duration_samples/self.sampling_rate:.2f}s")
                    # Remove the trailing silence from the segment for cleaner ASR
                    # silence_bytes = int(self.temp_end * 4)
                    # segment = bytes(self.current_speech[:-silence_bytes])
                    # Actually, keeping a bit of silence is okay, but let's trim most of it
                    segment = bytes(self.current_speech)
                else:
                    logger.debug(f"VAD: Discarding short noise ({speech_duration_samples/self.sampling_rate:.2f}s)")
                
                self.reset_states()
                
        return segment, float(speech_prob)
