"""
src/api/predict.py
MODULE 10 — Serving Interface  (FROZEN — do not change signature after backend handoff)

Single entry point for the entire ML pipeline.
Backend calls predict_post() only — never individual model files.

Full pipeline per call:
    text
      → cleaner.clean()              Module 2
      → brand.detect()               Module 3
      → sentiment_model.predict()    Module 4
      → sarcasm_model.predict()      Module 4 (sub)
      → attribution.attribute()      Module 5  ← key differentiator
      → emotion_model.predict()      Module 6
      → topic_model.predict()        Module 7
      → crisis (flag only here,      Module 8  (full crisis engine in aggregator)
                full check in aggregation)
      → PredictionResponse           Module 10

STUB MODE (Week 1):
    Models not trained yet. Returns realistic fake outputs.
    Backend can build entire app against this.
    Swap stub → real by setting STUB_MODE = False after notebook 09.

Usage:
    from src.api.predict import predict_post
    result = predict_post("Nike shoes are incredible", platform="reddit")
"""

from __future__ import annotations
from datetime import datetime, timezone
from src.api.schemas import (
    PredictionRequest, PredictionResponse,
    SentimentResult, BrandSentimentResult,
    EmotionResult, TopicResult, CrisisResult,
    BatchPredictionRequest, BatchPredictionResponse,
)

# ── STUB MODE ─────────────────────────────────────────────────────────────────
# True  = returns realistic fake data (Week 1 — models not trained yet)
# False = runs real pipeline (set this after notebook 10 is complete)
#
# HOW TO KNOW WHEN TO FLIP THIS:
#   The ML person will tell you when all four model folders are ready:
#     models/sentiment/   models/sarcasm/   models/emotion/   models/topic/
#   Run the smoke test in BACKEND_HANDOFF.md before going live.
#
# WHO FLIPS IT: ML person sets this to False and confirms models are on Drive.
#   Backend person changes NOTHING — same function, same return shape.
# ─────────────────────────────────────────────────────────────────────────────
STUB_MODE = False


# ── Lazy model loading — loaded once on first call, not at import ─────────────
_sentiment_model = None
_sarcasm_model   = None
_emotion_model   = None
_topic_model     = None


def _load_models():
    global _sentiment_model, _sarcasm_model, _emotion_model, _topic_model
    import os
    from src.models.sentiment import SentimentModel
    from src.models.sarcasm   import SarcasmModel
    from src.models.emotion   import EmotionModel
    from src.models.topic     import TopicModel

    # Resolve model root: use DRIVE_ROOT env var if set (Colab),
    # otherwise resolve relative to this file's location (deployment server).
    _drive_root = os.environ.get(
        "DRIVE_ROOT",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
    )
    _model_root = os.path.join(_drive_root, "models")

    # ── Auto-download from HuggingFace if weights are missing ─────────────────
    # This is a safety net for Render: if the build-time download failed
    # (e.g. env vars were added after the build), the server downloads
    # the weights on first startup instead of crashing.
    _required_dirs = ["sentiment", "sarcasm", "emotion", "topic"]
    _missing = [d for d in _required_dirs
                if not os.path.isdir(os.path.join(_model_root, d))]

    if _missing:
        hf_repo_id = os.environ.get("HF_REPO_ID")
        hf_token   = os.environ.get("HF_TOKEN")
        if hf_repo_id:
            import logging
            logging.getLogger("uvicorn").info(
                f"Model folders missing: {_missing}. "
                f"Downloading from HuggingFace repo '{hf_repo_id}'..."
            )
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=hf_repo_id,
                repo_type="model",
                local_dir=_model_root,
                local_dir_use_symlinks=False,
                token=hf_token,
            )
            logging.getLogger("uvicorn").info("HuggingFace download complete.")
        else:
            raise RuntimeError(
                f"Model folders {_missing} not found locally and "
                "HF_REPO_ID env var is not set. Cannot load models."
            )

    _sentiment_model = SentimentModel.load(os.path.join(_model_root, "sentiment"))
    _sarcasm_model   = SarcasmModel.load(os.path.join(_model_root, "sarcasm"))
    _emotion_model   = EmotionModel.load(os.path.join(_model_root, "emotion"))
    _topic_model     = TopicModel.load(os.path.join(_model_root, "topic"))


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def predict_post(
    text:          str,
    brand:         str | None = None,
    platform:      str        = "reddit",
    context_posts: list[str]  = [],
) -> PredictionResponse:
    """
    Run the full 10-module pipeline on a single text.

    Args:
        text:          Raw post text — cleaning happens inside this function
        brand:         Target brand (None = detect all brands automatically)
        platform:      "reddit" | "news" — affects preprocessing rules
        context_posts: Up to 3 prior posts for sarcasm context window

    Returns:
        PredictionResponse — full structured output

    Signature is FROZEN after backend handoff.
    Internal implementation will change as models are trained.
    """
    if STUB_MODE:
        return _stub_response(text, brand, platform)
    return _real_pipeline(text, brand, platform, context_posts)


def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """
    Batch prediction for multiple posts.
    Used by collector.py after each collection cycle.
    """
    results = [
        predict_post(
            text=p.text,
            brand=p.brand or request.brand,
            platform=p.platform,
            context_posts=p.context_posts,
        )
        for p in request.posts
    ]
    return BatchPredictionResponse(
        results=results,
        total=len(results),
        processed_at=datetime.now(timezone.utc),
    )


# ══════════════════════════════════════════════════════════════════════════════
# STUB — Week 1, returns realistic fake output
# ══════════════════════════════════════════════════════════════════════════════

def _stub_response(text: str, brand: str | None, platform: str) -> PredictionResponse:
    """
    Realistic fake output for backend development.
    Shape is identical to real pipeline output.
    """
    detected_brand = brand or "Nike"
    return PredictionResponse(
        text=text[:500],
        platform=platform,
        overall=SentimentResult(
            label="positive",
            score=0.91,
            is_sarcastic=False,
            sarcasm_score=0.03,
        ),
        brands=[
            BrandSentimentResult(
                brand=detected_brand,
                sentiment="positive",
                score=0.91,
                method="default",
            )
        ],
        emotions=EmotionResult(
            emotions={"joy": 0.72, "admiration": 0.41},
            top_emotion="joy",
        ),
        topic=TopicResult(
            topic_id=0,
            label="placeholder_topic",
            keywords=["placeholder"],
        ),
        crisis=CrisisResult(
            crisis_flag=False,
            severity=None,
        ),
        processed_at=datetime.now(timezone.utc),
    )


# ══════════════════════════════════════════════════════════════════════════════
# REAL PIPELINE — activated after all notebooks complete
# ══════════════════════════════════════════════════════════════════════════════

def _real_pipeline(
    text:          str,
    brand:         str | None,
    platform:      str,
    context_posts: list[str],
) -> PredictionResponse:
    """
    Full pipeline. Each step corresponds to an architecture module.
    Uncommented and activated by setting STUB_MODE = False.
    """
    global _sentiment_model, _sarcasm_model, _emotion_model, _topic_model

    if _sentiment_model is None:
        _load_models()

    from src.preprocessing.cleaner   import clean
    from src.brand.detector          import detect, detect_with_positions
    from src.attribution.engine      import attribute, resolve_conflicts

    # Module 2 — Preprocess
    clean_text = clean(text, platform=platform)
    if not clean_text:
        return _stub_response(text, brand, platform)

    # Module 3 — Brand Detection
    detected_brands  = detect(clean_text)
    brand_positions  = detect_with_positions(clean_text)
    if brand and brand not in detected_brands:
        detected_brands.append(brand)

    # Module 4 — Sentiment + Sarcasm
    sentiment_result = _sentiment_model.predict(clean_text)
    sarcasm_result   = _sarcasm_model.predict(
        " ".join(context_posts[-3:] + [clean_text])   # context window
    )

    # Module 5 — Attribution (key differentiator)
    brand_results = attribute(
        text=clean_text,
        detected_brands=detected_brands,
        sentiment_result=sentiment_result,
        sarcasm_result=sarcasm_result,
        brand_positions=brand_positions,
    )
    brand_results = resolve_conflicts(brand_results)

    # Module 6 — Emotions
    emotion_result = _emotion_model.predict(clean_text)

    # Module 7 — Topic
    topic_result = _topic_model.predict(clean_text)

    # Module 8 — Crisis flag (simple per-post flag; full engine runs in aggregator)
    crisis_flag = sentiment_result["label"] == "negative" and sentiment_result["score"] > 0.85

    return PredictionResponse(
        text=clean_text,
        platform=platform,
        overall=SentimentResult(
            label=sentiment_result["label"],
            score=sentiment_result["score"],
            is_sarcastic=sarcasm_result["is_sarcastic"],
            sarcasm_score=sarcasm_result["score"],
        ),
        brands=[
            BrandSentimentResult(
                brand=b.brand,
                sentiment=b.sentiment,
                score=b.score,
                method=b.method,
            )
            for b in brand_results
        ],
        emotions=EmotionResult(
            emotions=emotion_result["emotions"],
            top_emotion=emotion_result["top_emotion"],
        ),
        topic=TopicResult(
            topic_id=topic_result["topic_id"],
            label=topic_result["label"],
            keywords=topic_result["keywords"],
        ),
        crisis=CrisisResult(
            crisis_flag=crisis_flag,
            severity="high" if crisis_flag else None,
        ),
        processed_at=datetime.now(timezone.utc),
    )
