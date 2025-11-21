import logging
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

logger = logging.getLogger(__name__)

class TextSentimentAnalyzer:
    def __init__(self, model_id="distilbert-base-uncased-finetuned-sst-2-english"):
        """
        Initialize DistilBERT Sentiment Analyzer (PyTorch).
        Args:
            model_id (str): Hugging Face model ID.
        """
        self.model_id = model_id
        logger.info(f"Loading Text Sentiment Model (PyTorch): {model_id}...")
        
        try:
            # Load model and tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_id)
            
            # Create pipeline
            self.classifier = pipeline(
                "text-classification", 
                model=self.model, 
                tokenizer=self.tokenizer,
                device=-1, # CPU
                top_k=None # Return all scores
            )
            logger.info("Text Sentiment Model loaded successfully.")
            
        except Exception as e:
            logger.error(f"Failed to load Text Sentiment Model: {e}")
            raise

    def predict(self, text: str) -> dict:
        """
        Analyze sentiment of text.
        Args:
            text (str): Input text.
        Returns:
            dict: {
                "sentiment": "positive" | "negative" | "neutral",
                "confidence": float,
                "scores": {"positive": float, "negative": float, "neutral": float} 
            }
        """
        if not text or not text.strip():
            return {"sentiment": "neutral", "confidence": 0.0, "scores": {"positive": 0.0, "negative": 0.0, "neutral": 1.0}}

        try:
            # Run inference
            results = self.classifier(text)
            # results is a list of lists of dicts: [[{'label': 'POSITIVE', 'score': 0.9}, ...]]
            scores_list = results[0]
            
            # Parse scores
            score_map = {item['label'].lower(): item['score'] for item in scores_list}
            
            # SST-2: POSITIVE, NEGATIVE
            pos = score_map.get('positive', 0.0)
            neg = score_map.get('negative', 0.0)
            neu = 0.0 
            
            # Determine winner
            winner = max(score_map, key=score_map.get)
            confidence = score_map[winner]
            
            return {
                "sentiment": winner,
                "confidence": confidence,
                "scores": {
                    "positive": pos,
                    "negative": neg,
                    "neutral": neu
                }
            }
            
        except Exception as e:
            logger.error(f"Text Sentiment Error: {e}")
            return {"error": str(e)}
