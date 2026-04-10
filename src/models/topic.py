"""
src/models/topic.py
MODULE 7 — Topic Modeling Engine

Discovers what customers are talking about per brand.
Algorithm: BERTopic (embedding-based, no bag-of-words assumptions)

Training:  notebook 08_topic_model.ipynb
Saved via: topic_model.save("models/topic/", serialization="safetensors")

Used by: api/predict.py, aggregation/aggregator.py
"""

from __future__ import annotations
import numpy as np

MODEL_PATH = "models/topic"


class TopicModel:
    """
    Wrapper around BERTopic model.

    Usage:
        model = TopicModel.load("/path/to/models/topic")
        result = model.predict("The sole peeled off after 3 weeks")
        # {"topic_id": 3, "label": "sole_durability", "keywords": [...], "probability": 0.82}
    """

    def __init__(self, model):
        self.model = model

    @classmethod
    def load(cls, path: str = MODEL_PATH) -> "TopicModel":
        from bertopic import BERTopic
        from sentence_transformers import SentenceTransformer
        # REQUIRED: pass embedding_model= when loading a safetensors-serialized
        # BERTopic model. Without this, transform() raises ValueError because
        # BERTopic cannot encode new texts at inference time.
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        model = BERTopic.load(path, embedding_model=embedding_model)
        return cls(model)

    @staticmethod
    def _extract_prob(probs_row, topic_id: int) -> float:
        """
        Safely extract the probability scalar for a single document.

        BERTopic.transform() returns probs with shape (n_docs, n_topics).
        probs[doc_idx] is therefore a 1-D array of all topic probabilities.
        We index into it with the assigned topic_id to get one float.

        Edge cases:
          - topic_id == -1  (outlier): use max probability in the row
          - probs is None   (some BERTopic configs skip probability calc)
          - probs_row is already a scalar (older BERTopic versions)
        """
        if probs_row is None:
            return 0.0
        arr = np.asarray(probs_row)
        if arr.ndim == 0:                   # already a scalar
            return round(float(arr), 4)
        if topic_id == -1 or topic_id >= len(arr):
            return round(float(arr.max()), 4)
        return round(float(arr[topic_id]), 4)

    def predict(self, text: str) -> dict:
        """
        Assign a topic to a single text.

        Returns:
            {
                "topic_id":    int,        # -1 = outlier / no clear topic
                "label":       str,        # e.g. "sole_durability_quality"
                "keywords":    list[str],  # top keywords for this topic
                "probability": float,      # 0.0 - 1.0
            }
        """
        topics, probs = self.model.transform([text])
        topic_id = int(topics[0])
        keywords = [w for w, _ in self.model.get_topic(topic_id)] if topic_id != -1 else []
        label    = "_".join(keywords[:3]) if keywords else "other"
        prob     = self._extract_prob(
            probs[0] if probs is not None else None,
            topic_id,
        )
        return {
            "topic_id":    topic_id,
            "label":       label,
            "keywords":    keywords[:10],
            "probability": prob,
        }

    def predict_batch(self, texts: list[str]) -> list[dict]:
        topics, probs = self.model.transform(texts)
        results = []
        for i, (topic_id, prob_row) in enumerate(zip(topics, probs)):
            topic_id = int(topic_id)
            keywords = [w for w, _ in self.model.get_topic(topic_id)] if topic_id != -1 else []
            label    = "_".join(keywords[:3]) if keywords else "other"
            prob     = self._extract_prob(prob_row, topic_id)
            results.append({
                "topic_id":    topic_id,
                "label":       label,
                "keywords":    keywords[:10],
                "probability": prob,
            })
        return results
