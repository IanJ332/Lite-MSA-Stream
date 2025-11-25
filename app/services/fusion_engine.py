import numpy as np

class FusionEngine:
    def __init__(self):
        self.smoothed_emotions = {}
        self.alpha = 0.6  # Smoothing factor (0.6 = responsive but smooth)
        self.hallucinations = ["The.", "You.", "I.", "It.", "A.", "And.", "But.", "So.", "He.", "She.", "They.", "We."]

    def fuse(self, asr_result, emotion_probs, text_emotions):
        text_content = asr_result.get("text", "")
        
        # 1. Aggressive Noise Filtering
        # Discard if empty, < 2 chars, or in hallucinations list
        # Also discard if it's just punctuation
        clean_text = text_content.strip()
        if not clean_text or len(clean_text) < 2 or clean_text in [".", ",", "?", "!", "。", "，", "？", "！"] or clean_text in self.hallucinations:
            # Return None to signal "discard this result"
            return None

        # 2. Semantic Bias (Text -> Acoustic)
        if emotion_probs and text_emotions:
            for emotion, text_prob in text_emotions.items():
                if emotion in emotion_probs:
                    # Boost acoustic probability if text agrees
                    emotion_probs[emotion] = emotion_probs.get(emotion, 0) * (1.0 + text_prob * 2.0)
            
            # Acoustic Calibration (Boost Neutral/Calm)
            if text_emotions.get('neutral', 0) > 0.5:
                emotion_probs['neutral'] = emotion_probs.get('neutral', 0) * 2.0
                emotion_probs['calm'] = emotion_probs.get('calm', 0) * 1.5

            # Normalize
            total = sum(emotion_probs.values())
            if total > 0:
                for k in emotion_probs:
                    emotion_probs[k] /= total

        # 3. EMA Smoothing
        if not self.smoothed_emotions:
            self.smoothed_emotions = emotion_probs.copy()
        else:
            for k in emotion_probs:
                if k not in self.smoothed_emotions:
                    self.smoothed_emotions[k] = emotion_probs[k]
                else:
                    self.smoothed_emotions[k] = self.alpha * emotion_probs[k] + (1 - self.alpha) * self.smoothed_emotions[k]
        
        return self.smoothed_emotions
