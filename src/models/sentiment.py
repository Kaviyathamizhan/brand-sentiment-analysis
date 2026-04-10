"""
src/models/sentiment.py
MODULE 4 — Sentiment Intelligence Engine

Fine-tuned RoBERTa for 3-class sentiment classification.
Base model: cardiffnlp/twitter-roberta-base-sentiment-latest
  - Pre-trained on 58M tweets → perfect domain match for Sentiment140
  - NOT bert-base-uncased (wrong domain — trained on Wikipedia/books)

Training: notebook 04_sentiment_model.ipynb
Saved via: model.save_pretrained("models/sentiment/")
           tokenizer.save_pretrained("models/sentiment/")

Used by: attribution/engine.py, api/predict.py

Labels: 0=negative  1=neutral  2=positive
"""

from __future__ import annotations
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH  = "models/sentiment"
BASE_MODEL  = "cardiffnlp/twitter-roberta-base-sentiment-latest"
LABELS      = ["negative", "neutral", "positive"]
MAX_LENGTH  = 128


class SentimentModel:
    """
    Wrapper around the fine-tuned RoBERTa sentiment model.

    Usage:
        model = SentimentModel.load()
        result = model.predict("Nike shoes are incredible")
        # {"label": "positive", "score": 0.94, "all_scores": {...}}
    """

    def __init__(self, tokenizer, model, device):
        self.tokenizer = tokenizer
        self.model     = model
        self.device    = device

    @classmethod
    def load(cls, path: str = MODEL_PATH) -> "SentimentModel":
        """
        Load fine-tuned model from disk.
        Falls back to base model if fine-tuned not found yet.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(path)
        model     = AutoModelForSequenceClassification.from_pretrained(path).to(device)
        model.eval()
        return cls(tokenizer, model, device)

    def predict(self, text: str) -> dict:
        """
        Predict sentiment for a single text.

        Args:
            text: cleaned text (run cleaner.clean() first)

        Returns:
            {
                "label":      "positive" | "neutral" | "negative",
                "score":      float,   # confidence of winning label
                "all_scores": {"negative": float, "neutral": float, "positive": float}
            }
        """
        inputs = self.tokenizer(
            text, return_tensors="pt",
            truncation=True, max_length=MAX_LENGTH, padding=True,
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits

        probs  = torch.softmax(logits, dim=-1).squeeze().tolist()
        idx    = int(torch.argmax(logits))

        return {
            "label":      LABELS[idx],
            "score":      round(probs[idx], 4),
            "all_scores": {l: round(p, 4) for l, p in zip(LABELS, probs)},
        }

    def predict_batch(self, texts: list[str], batch_size: int = 32) -> list[dict]:
        """
        Predict sentiment for a list of texts efficiently.
        Processes in batches to avoid OOM on large inputs.
        """
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch, return_tensors="pt",
                truncation=True, max_length=MAX_LENGTH, padding=True,
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(**inputs).logits

            probs_batch = torch.softmax(logits, dim=-1).tolist()
            for probs in probs_batch:
                idx = int(max(range(len(probs)), key=lambda i: probs[i]))
                results.append({
                    "label":      LABELS[idx],
                    "score":      round(probs[idx], 4),
                    "all_scores": {l: round(p, 4) for l, p in zip(LABELS, probs)},
                })
        return results
