import torch
import torch.nn as nn
import numpy as np
import logging
from transformers import Wav2Vec2Model, HubertModel, Wav2Vec2Processor, Wav2Vec2FeatureExtractor

logger = logging.getLogger("vox-pathos")

class EnsembleService:
    def __init__(self, device="cpu"):
        self.device = device
        logger.info(f"Initializing EnsembleService on {self.device}...")

        # 1. HuBERT (Large, Fine-tuned on LibriSpeech)
        # We use the base Model to get hidden states, not the ForCTC version
        self.hubert_id = "facebook/hubert-large-ls960-ft"
        logger.info(f"Loading HuBERT: {self.hubert_id}")
        self.hubert_processor = Wav2Vec2FeatureExtractor.from_pretrained(self.hubert_id)
        self.hubert_model = HubertModel.from_pretrained(self.hubert_id).to(self.device)
        self.hubert_model.eval()

        # 2. Wav2Vec2 (Large, Robust)
        self.wav2vec_id = "facebook/wav2vec2-large-robust"
        logger.info(f"Loading Wav2Vec2: {self.wav2vec_id}")
        self.wav2vec_processor = Wav2Vec2FeatureExtractor.from_pretrained(self.wav2vec_id)
        self.wav2vec_model = Wav2Vec2Model.from_pretrained(self.wav2vec_id).to(self.device)
        self.wav2vec_model.eval()
        
        logger.info("Ensemble Models Loaded Successfully.")

    def extract_features(self, audio_input):
        """
        Extracts and concatenates features from all backbone models.
        Args:
            audio_input: np.ndarray (float32), 16kHz mono
        Returns:
            np.ndarray: Concatenated feature vector (Dim: 1024 + 1024 = 2048)
        """
        # Ensure input is Tensor
        if isinstance(audio_input, np.ndarray):
            inputs = torch.tensor(audio_input).float().to(self.device)
        elif isinstance(audio_input, bytes):
            inputs = torch.from_numpy(np.frombuffer(audio_input, dtype=np.float32)).to(self.device)
        else:
            inputs = audio_input.to(self.device)
            
        # Add batch dim if needed
        if inputs.ndim == 1:
            inputs = inputs.unsqueeze(0)

        with torch.no_grad():
            # 1. HuBERT Features
            # Processor handles normalization/padding if we used it, but for raw audio we can pass directly if normalized
            # Ideally use processor to be safe
            processed_hubert = self.hubert_processor(inputs.squeeze().cpu().numpy(), sampling_rate=16000, return_tensors="pt").input_values.to(self.device)
            hubert_out = self.hubert_model(processed_hubert).last_hidden_state
            # Mean Pooling: (Batch, Time, Dim) -> (Batch, Dim)
            hubert_feat = torch.mean(hubert_out, dim=1)

            # 2. Wav2Vec2 Features
            processed_wav2vec = self.wav2vec_processor(inputs.squeeze().cpu().numpy(), sampling_rate=16000, return_tensors="pt").input_values.to(self.device)
            wav2vec_out = self.wav2vec_model(processed_wav2vec).last_hidden_state
            wav2vec_feat = torch.mean(wav2vec_out, dim=1)

            # Concatenate
            combined_feat = torch.cat((hubert_feat, wav2vec_feat), dim=1)
            
        return combined_feat.cpu().numpy()

    def load_fusion_model(self, weights_path):
        """
        Loads the trained FusionNet weights.
        """
        from app.models.fusion_layer import FusionNet
        self.fusion_model = FusionNet().to(self.device)
        try:
            self.fusion_model.load_state_dict(torch.load(weights_path, map_location=self.device))
            self.fusion_model.eval()
            logger.info(f"Fusion Model loaded from {weights_path}")
        except Exception as e:
            logger.error(f"Failed to load Fusion Model: {e}")
            self.fusion_model = None

    def predict_emotions(self, audio_input):
        """
        Predicts emotion probabilities using the Ensemble.
        Returns:
            dict: {emotion_label: probability}
        """
        if not hasattr(self, 'fusion_model') or self.fusion_model is None:
            return {}

        # Extract Features
        features = self.extract_features(audio_input) # (1, 2048)
        features_tensor = torch.tensor(features).float().to(self.device)
        
        # Inference
        with torch.no_grad():
            logits = self.fusion_model(features_tensor)
            probs = torch.nn.functional.softmax(logits, dim=1)[0].cpu().numpy()
            
        # Map to labels
        EMOTION_LABELS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgusted', 'surprised']
        
        result = {}
        for i, label in enumerate(EMOTION_LABELS):
            result[label] = float(probs[i])
            
        return result
