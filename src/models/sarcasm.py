"""
src/models/sarcasm.py
MODULE 4 (sub) — Sarcasm / Irony Detector

Fine-tuned RoBERTa for binary irony detection.
Used to flip or adjust sentiment when sarcasm is detected.

Base model:  roberta-base
Training:    notebook 05_sarcasm_model.ipynb  (SemEval 2018 Task 3)
Saved via:   model.save_pretrained("models/sarcasm/")

Key insight (EDA Finding 7):
    Irony = positive surface words + negative intent
    e.g. "Oh great, my Nike shoes fell apart after 2 days. Love it."
    → sentiment model says POSITIVE (surface words)
    → sarcasm model says IRONIC
    → attribution engine flips → NEGATIVE toward Nike

Used by: attribution/engine.py
"""

from __future__ import annotations
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH   = "models/sarcasm"
MAX_LENGTH   = 128
LABELS       = ["not_ironic", "ironic"]
IRONY_THRESHOLD = 0.6   # Min score to flag as sarcastic


class SarcasmModel:
    """
    Wrapper around fine-tuned irony detector.

    Usage:
        model = SarcasmModel.load()
        result = model.predict("Oh great, another Nike product that broke in a week")
        # {"is_sarcastic": True, "score": 0.87}
    """

    def __init__(self, tokenizer, model, device):
        self.tokenizer = tokenizer
        self.model     = model
        self.device    = device

    @classmethod
    def load(cls, path: str = MODEL_PATH) -> "SarcasmModel":
        device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(path)
        model     = AutoModelForSequenceClassification.from_pretrained(path).to(device)
        model.eval()
        return cls(tokenizer, model, device)

    def predict(self, text: str) -> dict:
        """
        Predict whether text is sarcastic/ironic.

        Returns:
            {
                "is_sarcastic": bool,
                "score":        float,   # confidence of ironic class
                "label":        "ironic" | "not_ironic"
            }
        """
        inputs = self.tokenizer(
            text, return_tensors="pt",
            truncation=True, max_length=MAX_LENGTH, padding=True,
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits

        probs        = torch.softmax(logits, dim=-1).squeeze().tolist()
        irony_score  = probs[1]   # index 1 = "ironic"
        is_sarcastic = irony_score >= IRONY_THRESHOLD

        return {
            "is_sarcastic": is_sarcastic,
            "score":        round(irony_score, 4),
            "label":        "ironic" if is_sarcastic else "not_ironic",
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
            for probs in torch.softmax(logits, dim=-1).tolist():
                irony_score = probs[1]
                results.append({
                    "is_sarcastic": irony_score >= IRONY_THRESHOLD,
                    "score":        round(irony_score, 4),
                    "label":        "ironic" if irony_score >= IRONY_THRESHOLD else "not_ironic",
                })
        return results
