"""
src/models/emotion.py
MODULE 6 — Emotion Analytics

Multi-label emotion classifier fine-tuned on GoEmotions.
28 emotion classes. Multiple emotions can be active per text.

Base model: monologg/bert-base-cased-goemotions-original
Training:   notebook 06_emotion_model.ipynb
Saved via:  model.save_pretrained("models/emotion/")

Used by: api/predict.py, aggregation/aggregator.py
"""

from __future__ import annotations
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "models/emotion"
MAX_LENGTH = 128
THRESHOLD  = 0.3   # Min score to include emotion in output (from EDA Finding 4)

EMOTION_LABELS = [
    "admiration","amusement","anger","annoyance","approval","caring","confusion",
    "curiosity","desire","disappointment","disapproval","disgust","embarrassment",
    "excitement","fear","gratitude","grief","joy","love","nervousness","optimism",
    "pride","realization","relief","remorse","sadness","surprise","neutral",
]


class EmotionModel:
    """
    Wrapper around fine-tuned GoEmotions multi-label classifier.

    Usage:
        model = EmotionModel.load()
        result = model.predict("I am so frustrated with this product")
        # {"emotions": {"anger": 0.82, "annoyance": 0.61}, "top_emotion": "anger"}
    """

    def __init__(self, tokenizer, model, device):
        self.tokenizer = tokenizer
        self.model     = model
        self.device    = device

    @classmethod
    def load(cls, path: str = MODEL_PATH) -> "EmotionModel":
        device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(path)
        model     = AutoModelForSequenceClassification.from_pretrained(path).to(device)
        model.eval()
        return cls(tokenizer, model, device)

    def predict(self, text: str, threshold: float = THRESHOLD) -> dict:
        """
        Predict emotions for a single text.

        Returns:
            {
                "emotions":    {"anger": 0.82, "annoyance": 0.61},  # above threshold only
                "top_emotion": "anger",
                "all_scores":  {"admiration": 0.01, "anger": 0.82, ...}  # all 28
            }
        """
        inputs = self.tokenizer(
            text, return_tensors="pt",
            truncation=True, max_length=MAX_LENGTH, padding=True,
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits

        scores     = torch.sigmoid(logits).squeeze().tolist()
        all_scores = {l: round(s, 4) for l, s in zip(EMOTION_LABELS, scores)}
        active     = {l: s for l, s in all_scores.items() if s >= threshold}
        top        = max(all_scores, key=all_scores.get) if all_scores else "neutral"

        return {
            "emotions":    active,
            "top_emotion": top,
            "all_scores":  all_scores,
        }

    def predict_batch(self, texts: list[str], batch_size: int = 32) -> list[dict]:
        results = []
        for i in range(0, len(texts), batch_size):
            batch  = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch, return_tensors="pt",
                truncation=True, max_length=MAX_LENGTH, padding=True,
            ).to(self.device)
            with torch.no_grad():
                logits = self.model(**inputs).logits
            for scores_t in torch.sigmoid(logits).tolist():
                all_scores = {l: round(s, 4) for l, s in zip(EMOTION_LABELS, scores_t)}
                active     = {l: s for l, s in all_scores.items() if s >= THRESHOLD}
                top        = max(all_scores, key=all_scores.get)
                results.append({"emotions": active, "top_emotion": top, "all_scores": all_scores})
        return results
