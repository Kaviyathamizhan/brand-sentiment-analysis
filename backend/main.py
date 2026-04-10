from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import csv, os, glob, json, time
from io import StringIO
from datetime import datetime, timezone

import pandas as pd

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    import logging
    logger = logging.getLogger("uvicorn")
    logger.info("Warming up ML pipelines into RAM...")
    try:
        from src.api.predict import _load_models
        _load_models()
        logger.info("Successfully loaded multi-gigabyte models.")
    except Exception as e:
        logger.error(f"CRITICAL: Failed to load models from disk/HuggingFace! {e}")

    try:
        from src.data.collector import Collector, start_scheduler
        collector = Collector()
        start_scheduler(collector)
    except Exception as e:
        logger.warning(f"[Scheduler] Could not start live data collection: {e}")

    yield
    logger.info("Shutting down API...")

# ─────────────────────────────────────────
# Initialize FastAPI
# ─────────────────────────────────────────

app = FastAPI(title="Brand Sentiment API", lifespan=lifespan)

# ─────────────────────────────────────────
# CORS
# ─────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────
# ML Pipeline
# ─────────────────────────────────────────

from src.api.predict import predict_post

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def _find_csv(brand: str):
    """Find latest raw CSV for a brand."""
    pattern = os.path.join(ROOT, "data", "live", "combined",
                           f"live_combined_{brand.lower()}_latest.csv")
    matches = glob.glob(pattern)
    return matches[0] if matches else None

def _find_scored_csv(brand: str):
    """Find pre-scored CSV. Falls back to raw CSV if scored doesn't exist."""
    scored = os.path.join(ROOT, "data", "live", "combined",
                          f"live_combined_{brand.lower()}_scored.csv")
    if os.path.exists(scored):
        return scored, True
    raw = _find_csv(brand)
    return raw, False

def _run_predict(text: str, platform: str = "news"):
    """Run ML prediction and return a dict with all fields."""
    try:
        r = predict_post(str(text), platform=platform)
        return {
            "text": r.text,
            "overall": {
                "label": r.overall.label.lower(),
                "score": round(r.overall.score, 3),
                "is_sarcastic": getattr(r.overall, "is_sarcastic", False),
                "sarcasm_score": round(getattr(r.overall, "sarcasm_score", 0.0), 3),
            },
            "brands": [
                {"brand": b.brand, "sentiment": b.sentiment, "score": round(b.score, 3),
                 "method": getattr(b, "method", "default")}
                for b in (r.brands or [])
            ],
            "emotions": {
                "emotions": {k: round(v, 3) for k, v in (r.emotions.emotions or {}).items()}
                            if r.emotions and hasattr(r.emotions, "emotions") and r.emotions.emotions else {},
                "top_emotion": r.emotions.top_emotion if r.emotions else None,
            },
            "topic": {
                "topic_id": getattr(r.topic, "topic_id", None) if r.topic else None,
                "label": getattr(r.topic, "label", None) if r.topic else None,
                "keywords": getattr(r.topic, "keywords", []) if r.topic else [],
            },
            "crisis": {
                "crisis_flag": getattr(r.crisis, "crisis_flag", False) if hasattr(r, "crisis") and r.crisis else False,
                "severity": getattr(r.crisis, "severity", None) if hasattr(r, "crisis") and r.crisis else None,
            },
            "processed_at": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        return {
            "text": text[:200], "overall": {"label": "neutral", "score": 0, "is_sarcastic": False, "sarcasm_score": 0},
            "brands": [], "emotions": {"emotions": {}, "top_emotion": None},
            "topic": {"topic_id": None, "label": None, "keywords": []},
            "crisis": {"crisis_flag": False, "severity": None},
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "error": str(e),
        }

# ─────────────────────────────────────────
# 3.6 — GET /brands
# ─────────────────────────────────────────

MONITORED_BRANDS = ["Nike", "Adidas", "Puma", "Reebok"]

@app.get("/")
def home():
    return {"message": "Backend is running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/brands")
def get_brands():
    return {"brands": MONITORED_BRANDS}

# ─────────────────────────────────────────
# 3.1 — POST /predict  (single text)
# ─────────────────────────────────────────

class TextInput(BaseModel):
    text: str
    platform: str = "reddit"

@app.post("/predict")
def predict(data: TextInput):
    return _run_predict(data.text, data.platform)

# ─────────────────────────────────────────
# 3.2 — POST /predict/batch  (CSV / batch)
# ─────────────────────────────────────────

class BatchInput(BaseModel):
    texts: List[str]
    platform: str = "reddit"

@app.post("/predict/batch")
def predict_batch(data: BatchInput):
    results = []
    processed = skipped = 0
    for text in data.texts:
        if not text or len(text.strip()) < 5:
            results.append(None)
            skipped += 1
            continue
        r = _run_predict(text, data.platform)
        if not r["brands"]:
            results.append(None)
            skipped += 1
        else:
            results.append(r)
            processed += 1
    return {"results": results, "processed": processed, "skipped": skipped}

# ─────────────────────────────────────────
# 3.3 — GET /metrics/{brand}
# Reads from pre-scored CSV (instant)
# ─────────────────────────────────────────

@app.get("/metrics/{brand}")
def get_metrics(brand: str, hours: int = 24):
    empty = {
        "brand": brand, "window_hours": hours, "total_posts": 0,
        "sentiment": {"positive": 0, "neutral": 0, "negative": 0},
        "counts": {"positive": 0, "neutral": 0, "negative": 0},
        "top_emotions": [], "top_topics": [], "trend": [],
    }
    csv_path, is_scored = _find_scored_csv(brand)
    if not csv_path:
        return empty

    df = pd.read_csv(csv_path).dropna(subset=["full_text"])
    total = len(df)
    if total == 0:
        return empty

    if is_scored and "sentiment" in df.columns:
        # ── Fast path: read pre-computed scores ──
        pos = int((df["sentiment"] == "positive").sum())
        neg = int((df["sentiment"] == "negative").sum())
        neu = total - pos - neg

        emotion_agg = {}
        for emo in df["top_emotion"].dropna():
            emotion_agg[emo] = emotion_agg.get(emo, 0) + 1
        top_emotions = sorted(emotion_agg.items(), key=lambda x: -x[1])[:5]

        topic_agg = {}
        for topic in df["topic_label"].dropna():
            topic_agg[topic] = topic_agg.get(topic, 0) + 1
        top_topics = [t[0] for t in sorted(topic_agg.items(), key=lambda x: -x[1])[:3]]
    else:
        # ── Slow fallback: run ML (capped at 50) ──
        cap = min(total, 50)
        pos = neg = neu = 0
        emotion_agg = {}
        topic_agg = {}
        for text in df["full_text"].head(cap):
            r = _run_predict(str(text), "news")
            lab = r["overall"]["label"]
            if lab == "positive": pos += 1
            elif lab == "negative": neg += 1
            else: neu += 1
            te = r["emotions"].get("top_emotion")
            if te: emotion_agg[te] = emotion_agg.get(te, 0) + 1
            tl = r["topic"].get("label")
            if tl: topic_agg[tl] = topic_agg.get(tl, 0) + 1
        top_emotions = sorted(emotion_agg.items(), key=lambda x: -x[1])[:5]
        top_topics = [t[0] for t in sorted(topic_agg.items(), key=lambda x: -x[1])[:3]]

    denom = total if (is_scored and "sentiment" in df.columns) else min(total, 50)
    return {
        "brand": brand, "window_hours": hours, "total_posts": total,
        "sentiment": {
            "positive": round(pos / denom, 3) if denom else 0,
            "neutral": round(neu / denom, 3) if denom else 0,
            "negative": round(neg / denom, 3) if denom else 0,
        },
        "counts": {"positive": pos, "neutral": neu, "negative": neg},
        "top_emotions": top_emotions,
        "top_topics": top_topics,
        "trend": [],
    }

# ─────────────────────────────────────────
# 3.4 — GET /competitive
# ─────────────────────────────────────────

@app.get("/competitive")
def get_competitive(brands: str = "Nike,Adidas,Puma", hours: int = 48):
    brand_list = [b.strip() for b in brands.split(",")]
    result = {}
    for brand in brand_list:
        result[brand] = get_metrics(brand, hours)
    return {"brands": result}

# ─────────────────────────────────────────
# 3.5 — GET /alerts + PATCH resolve
# ─────────────────────────────────────────

# In-memory alert store (would be DB in production)
_alerts = []

def _scan_for_alerts():
    """Check if any brand has high negative sentiment using pre-scored data."""
    for brand in MONITORED_BRANDS:
        csv_path, is_scored = _find_scored_csv(brand)
        if not csv_path:
            continue
        df = pd.read_csv(csv_path).dropna(subset=["full_text"])
        if len(df) == 0:
            continue

        if is_scored and "sentiment" in df.columns:
            # Fast path: count from pre-scored data
            sample = df.head(20)
            neg = int((sample["sentiment"] == "negative").sum())
            neg_pct = neg / len(sample) * 100
        else:
            # Slow fallback
            sample = df["full_text"].head(20)
            neg = 0
            for text in sample:
                r = _run_predict(str(text), "news")
                if r["overall"]["label"] == "negative":
                    neg += 1
            neg_pct = neg / len(sample) * 100

        if neg_pct > 25:
            severity = "HIGH" if neg_pct > 50 else "MEDIUM" if neg_pct > 35 else "LOW"
            existing = [a for a in _alerts if a["brand"] == brand and not a["resolved"]]
            if not existing:
                _alerts.append({
                    "id": len(_alerts) + 1,
                    "brand": brand,
                    "severity": severity,
                    "reason": f"{neg_pct:.0f}% negative sentiment detected in recent posts",
                    "triggered_at": datetime.now(timezone.utc).isoformat(),
                    "resolved": False,
                })

@app.get("/alerts")
def get_alerts():
    # Run a quick scan if no alerts exist
    if not _alerts:
        try:
            _scan_for_alerts()
        except Exception:
            pass
    active = [a for a in _alerts if not a["resolved"]]
    return {"alerts": active}

@app.patch("/alerts/{alert_id}/resolve")
def resolve_alert(alert_id: int):
    for a in _alerts:
        if a["id"] == alert_id:
            a["resolved"] = True
            return {"status": "ok", "alert_id": alert_id}
    return {"status": "not_found"}

# ─────────────────────────────────────────
# Legacy CSV Upload (kept for compat)
# ─────────────────────────────────────────

@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    content = await file.read()
    csv_reader = csv.DictReader(StringIO(content.decode("utf-8")))
    results = []
    for row in csv_reader:
        text = row.get("text", "")
        r = _run_predict(text, "reddit")
        results.append(r)
    return {"results": results}

# ─────────────────────────────────────────
# Live News Feed API
# Reads from pre-scored CSV (instant)
# ─────────────────────────────────────────

@app.get("/live-news")
def get_live_news(brand: str = "Nike", limit: int = 50, offset: int = 0):
    raw_path = _find_csv(brand)
    if not raw_path:
        return {"articles": [], "total": 0, "brand": brand, "message": f"No data found for '{brand}'."}

    raw_df = pd.read_csv(raw_path).dropna(subset=["full_text"])
    
    # Sort newest first
    if "created_utc" in raw_df.columns:
        raw_df = raw_df.sort_values(by="created_utc", ascending=False)
        
    total = len(raw_df)
    
    # Try to merge pre-scored results
    scored_path = os.path.join(ROOT, "data", "live", "combined", f"live_combined_{brand.lower()}_scored.csv")
    is_joined = False
    if os.path.exists(scored_path):
        try:
            scored_df = pd.read_csv(scored_path).dropna(subset=["full_text"])
            # Join by full_text to attach ML scores to the raw live rows
            raw_df = raw_df.merge(
                scored_df[["full_text", "sentiment", "sentiment_score", "top_emotion"]].drop_duplicates("full_text"),
                on="full_text", how="left"
            )
            is_joined = True
        except Exception:
            pass

    page_df = raw_df.iloc[offset : offset + limit]

    articles = []
    for _, row in page_df.iterrows():
        text = str(row.get("full_text", ""))

        if is_joined and pd.notna(row.get("sentiment")):
            articles.append({
                "title": str(row.get("title", text[:100])),
                "source": str(row.get("source_name", "NewsAPI")),
                "url": str(row.get("url", "")),
                "published": str(row.get("created_utc", "")),
                "full_text": text[:300],
                "sentiment": str(row.get("sentiment", "unknown")).capitalize(),
                "emotion": str(row.get("top_emotion", "—") or "—").capitalize(),
                "score": round(float(row.get("sentiment_score", 0)) * 100),
            })
        else:
            # Post is brand new and ML is currently processing it in background
            articles.append({
                "title": str(row.get("title", text[:100])),
                "source": str(row.get("source_name", "NewsAPI")),
                "url": str(row.get("url", "")),
                "published": str(row.get("created_utc", "")),
                "full_text": text[:300],
                "sentiment": "Unknown",
                "emotion": "Processing...",
                "score": 0,
            })

    return {"articles": articles, "total": total, "brand": brand}

@app.post("/live-news/collect")
def trigger_collection(background_tasks: BackgroundTasks, brand: str = "Nike"):
    try:
        from src.data.collector import Collector
        collector = Collector()
        summary = collector.run(brand=brand)
        
        # Run the heavy re-scoring asynchronously in the background
        # so the frontend doesn't time out waiting for 400+ posts to be analyzed.
        def _bg_score(b):
            try:
                from backend.precompute import score_brand
                score_brand(b)
            except Exception as e:
                print(f"Background scoring failed for {b}: {e}")

        background_tasks.add_task(_bg_score, brand)
        
        return {"status": "ok", "summary": summary, "message": "Collection complete. Re-scoring running in background."}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# ─────────────────────────────────────────
# Trigger pre-compute from API
# ─────────────────────────────────────────

@app.post("/precompute")
def trigger_precompute(brand: str = None):
    try:
        from backend.precompute import score_brand, BRANDS
        brands = [brand] if brand else BRANDS
        total = 0
        for b in brands:
            total += score_brand(b)
        return {"status": "ok", "scored": total, "brands": brands}
    except Exception as e:
        return {"status": "error", "message": str(e)}
