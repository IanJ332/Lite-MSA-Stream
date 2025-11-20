import torch
import torchaudio
import onnxruntime as ort
import numpy as np
import logging
import torch
import torchaudio
import onnxruntime as ort
import numpy as np
import logging
from app.utils.model_utils import download_model

logger = logging.getLogger(__name__)

class AcousticAnalyzer:
    def __init__(self):
        # MelSpectrogram Parameters for AST
        self.sample_rate = 16000
        self.n_fft = 1024
        self.hop_length = 160
        self.n_mels = 128
        
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            center=True,
            power=2.0
        )
        
        # Load Model: Xenova/ast-finetuned-speech-commands-v2
        # This is a lightweight (49MB) quantized ONNX model.
        model_path = download_model(
            repo_id="Xenova/ast-finetuned-speech-commands-v2", 
            filename="onnx/model_q4f16.onnx",
            local_dir="models"
        )
        
        # Configure Session Options
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        
        # Disable graph optimizations to prevent fusion errors with quantized models
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        
        self.session = ort.InferenceSession(model_path, sess_options)
        self.input_name = self.session.get_inputs()[0].name
        logger.info(f"Acoustic Model loaded: {self.input_name}")

    def predict(self, audio_bytes: bytes) -> dict:
        """
        Predict emotion (mapped from keywords) from audio segment.
        """
        try:
            # Preprocess
            input_tensor = self._preprocess(audio_bytes)
            
            # Inference
            ort_inputs = {self.input_name: input_tensor.numpy()}
            output = self.session.run(None, ort_inputs)
            logits = output[0][0] # Shape: (num_classes,)
            
            # Softmax
            probs = np.exp(logits) / np.sum(np.exp(logits))
            
            # For demo purposes, we map the top class index to a sentiment
            # This model predicts speech commands, so this is just a placeholder mapping
            top_class_idx = np.argmax(probs)
            
            # Placeholder mapping logic (Arbitrary for demo)
            # Real implementation would use a model trained on emotions
            sentiment_map = {
                0: "neutral", 1: "positive", 2: "negative"
            }
            sentiment = sentiment_map.get(top_class_idx % 3, "neutral")
            
            return {
                "sentiment": sentiment,
                "confidence": float(probs[top_class_idx]),
                "raw_class_id": int(top_class_idx)
            }
        except Exception as e:
            logger.error(f"Acoustic inference failed: {e}")
            return {"error": str(e)}

    def _preprocess(self, audio_bytes: bytes) -> torch.Tensor:
        # Convert bytes to float32 tensor
        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        waveform = torch.from_numpy(audio_float32).unsqueeze(0) # (1, N)
        
        # Extract MelSpectrogram
        # Shape: (1, n_mels, time) -> (1, 128, T)
        mel_spec = self.mel_transform(waveform) 
        
        # The model expects a fixed shape of [1, 128, 128] (Batch, Mels, Time)
        # We need to pad or crop the time dimension to 128.
        target_time = 128
        current_time = mel_spec.shape[2]
        
        if current_time > target_time:
            # Crop
            mel_spec = mel_spec[:, :, :target_time]
        elif current_time < target_time:
            # Pad with zeros
            pad_amount = target_time - current_time
            # Pad last dimension (time) on the right
            mel_spec = torch.nn.functional.pad(mel_spec, (0, pad_amount))
            
        # Now shape is (1, 128, 128)
        return mel_spec
