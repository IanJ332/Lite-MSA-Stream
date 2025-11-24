import os
import logging
import torch
import numpy as np
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification

logger = logging.getLogger("vox-pathos")

class TextService:
    def __init__(self):
        self.model_path = os.path.join("app", "models", "text_emotion_model")
        self.model = None
        self.tokenizer = None
        self.labels = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
        
        # Mapping to our 6 classes
        # Our classes: neutral, happy, sad, angry, fearful, disgusted
        self.emotion_map = {
            'anger': 'angry',
            'disgust': 'disgusted',
            'fear': 'fearful',
            'joy': 'happy',
            'neutral': 'neutral',
            'sadness': 'sad',
            'surprise': 'happy' # Map surprise to happy (or neutral?) User said "joy + surprise -> happy" in plan
        }
        
        self._load_model()

    def _load_model(self):
        if not os.path.exists(self.model_path):
            logger.warning(f"Text Emotion Model not found at {self.model_path}. Run scripts/setup_text_model.py")
            return

        try:
            logger.info(f"Loading Text Emotion Model from {self.model_path}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = ORTModelForSequenceClassification.from_pretrained(self.model_path)
            logger.info("Text Emotion Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Text Emotion Model: {e}")

    def analyze_sentiment(self, text):
        """
        Analyze text sentiment using DistilRoberta ONNX.
        Returns:
            dict: {emotion: probability} (Normalized to our 6 classes)
        """
        if not self.model or not text:
            return {}

        try:
            inputs = self.tokenizer(text, return_tensors="pt")
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)[0].detach().numpy()
            
            # Map to our 6 classes
            result = {
                'neutral': 0.0,
                'happy': 0.0,
                'sad': 0.0,
                'angry': 0.0,
                'fearful': 0.0,
                'disgusted': 0.0
            }
            
            for i, label in enumerate(self.labels):
                mapped_label = self.emotion_map.get(label)
                if mapped_label:
                    result[mapped_label] += float(probs[i])
            
            # Normalize (just in case, though sum should be close to 1)
            total = sum(result.values())
            if total > 0:
                for k in result:
                    result[k] /= total
                    
            return result
            
        except Exception as e:
            logger.error(f"Text Analysis Error: {e}")
            return {}
