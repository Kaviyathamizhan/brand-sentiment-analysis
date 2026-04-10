"""
src/data/collector.py
─────────────────────────────────────────────────────────────────
Live data collector — DEPLOYMENT ONLY.

NOT imported by any training notebook (00-09).
Called by the FastAPI app on a schedule (APScheduler) or manually.

Data flow:
    Reddit PRAW  ──┐
                   ├──► combine ──► preprocess ──► Supabase DB
    NewsAPI     ──┘

Schedule (from config.yaml):
    Reddit  : every 30 min
    NewsAPI : every 60 min

Rate limits handled automatically:
    Reddit  : PRAW enforces 100 req/min internally
    NewsAPI : 100 req/day — we use 3 keywords × 1 request = 3 req/pull

Usage:
    # Run once (manual / testing):
    python -m src.data.collector --brand Nike --once

    # Run on schedule (called by FastAPI startup):
    from src.data.collector import start_scheduler
    start_scheduler()
"""

from __future__ import annotations

import os
import re
import time
import json
import glob
import logging
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml
from dotenv import load_dotenv
from loguru import logger

# ── Optional imports — fail gracefully so notebooks don't break ──
try:
    import praw
    PRAW_AVAILABLE = True
except ImportError:
    PRAW_AVAILABLE = False
    logger.warning("praw not installed — Reddit collection disabled")

try:
    from newsapi import NewsApiClient
    NEWSAPI_AVAILABLE = True
except ImportError:
    NEWSAPI_AVAILABLE = False
    logger.warning("newsapi-python not installed — NewsAPI collection disabled")

try:
    import feedparser
    FEEDPARSER_AVAILABLE = True
except ImportError:
    FEEDPARSER_AVAILABLE = False

try:
    from sqlalchemy import create_engine, text
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False


# ── Config & Environment ─────────────────────────────────────────

def load_config(config_path: str = "src/config/config.yaml") -> dict:
    """Load config.yaml — resolves path from project root."""
    root = Path(__file__).resolve().parents[2]
    full_path = root / config_path
    with open(full_path) as f:
        return yaml.safe_load(f)


def load_env():
    """Load .env from project root."""
    root = Path(__file__).resolve().parents[2]
    env_path = root / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()  # fallback: look in cwd


load_env()
CONFIG = load_config()
ONLINE = CONFIG["online"]
BRANDS_CFG = ONLINE["brands"]


# ── Helpers ──────────────────────────────────────────────────────

def resolve_path(relative: str) -> Path:
    """Resolve a relative path from config against the project root."""
    root = Path(__file__).resolve().parents[2]
    return root / relative


def ensure_dirs():
    """Create live data directories if they don't exist."""
    for key in ["live_reddit", "live_news", "live_combined"]:
        path = resolve_path(ONLINE["paths"][key])
        path.mkdir(parents=True, exist_ok=True)


def timestamp_str() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean_text(text: str) -> str:
    """
    Lightweight cleaning for storage — NOT the full preprocessing pipeline.
    Full preprocessing happens in src/data/preprocess.py before model inference.
    This just ensures stored text is UTF-8 safe and not empty.
    """
    if not text:
        return ""
    text = text.encode("utf-8", errors="ignore").decode("utf-8")
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── Reddit Collector ─────────────────────────────────────────────

class RedditCollector:
    """
    Collects brand-related posts and comments from Reddit via PRAW.

    Free tier:
        - 100 requests/min (PRAW enforces this automatically)
        - No approval needed — just register a 'script' app at
          https://www.reddit.com/prefs/apps

    Collection strategy:
        1. r/all keyword search  → broad reach across all subreddits
        2. Brand subreddits .new() → deep community posts

    Both strategies run per pull, results are deduplicated by post ID.
    """

    def __init__(self):
        if not PRAW_AVAILABLE:
            raise ImportError("Install praw: pip install praw==7.7.1")

        client_id     = os.getenv("REDDIT_CLIENT_ID", "")
        client_secret = os.getenv("REDDIT_CLIENT_SECRET", "")
        user_agent    = os.getenv("REDDIT_USER_AGENT", "BrandSentimentMonitor/1.0")

        if not client_id or "your_" in client_id:
            raise ValueError(
                "REDDIT_CLIENT_ID not set in .env\n"
                "Get free credentials at: https://www.reddit.com/prefs/apps"
            )

        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent,
            read_only=True,          # We never post
        )
        self.cfg = ONLINE["reddit"]
        logger.info("RedditCollector initialised (read-only)")

    def _extract_post(self, submission, brand: str, source_label: str) -> dict:
        """Normalise a PRAW Submission into a flat dict."""
        return {
            "id"          : submission.id,
            "type"        : "post",
            "platform"    : "reddit",
            "brand"       : brand,
            "subreddit"   : str(submission.subreddit),
            "source"      : source_label,
            "title"       : clean_text(submission.title),
            "body"        : clean_text(submission.selftext),
            "full_text"   : clean_text(f"{submission.title} {submission.selftext}"),
            "author"      : str(submission.author) if submission.author else "[deleted]",
            "score"       : submission.score,
            "upvote_ratio": submission.upvote_ratio,
            "num_comments": submission.num_comments,
            "url"         : f"https://reddit.com{submission.permalink}",
            "created_utc" : datetime.fromtimestamp(
                                submission.created_utc, tz=timezone.utc
                            ).isoformat(),
            "collected_at": utc_now(),
            "flair"       : submission.link_flair_text,
        }

    def _extract_comment(self, comment, brand: str, parent_post_id: str,
                         subreddit: str) -> dict:
        """Normalise a PRAW Comment into a flat dict."""
        body = clean_text(str(comment.body))
        return {
            "id"            : comment.id,
            "type"          : "comment",
            "platform"      : "reddit",
            "brand"         : brand,
            "subreddit"     : subreddit,
            "source"        : "comment",
            "parent_post_id": parent_post_id,
            "full_text"     : body,
            "author"        : str(comment.author) if comment.author else "[deleted]",
            "score"         : comment.score,
            "url"           : f"https://reddit.com{comment.permalink}",
            "created_utc"   : datetime.fromtimestamp(
                                  comment.created_utc, tz=timezone.utc
                              ).isoformat(),
            "collected_at"  : utc_now(),
        }

    def collect(self, brand: str) -> list[dict]:
        """
        Full collection run for one brand.
        Returns list of post + comment dicts, deduplicated by ID.
        """
        keywords   = BRANDS_CFG["keywords"].get(brand, [brand])
        subreddits = self.cfg["subreddits"].get(brand, ["all"])
        post_limit = self.cfg["post_limit_per_subreddit"]
        min_len    = self.cfg["min_text_length"]
        filter_nsfw = self.cfg["filter_nsfw"]

        records: list[dict] = []
        seen_ids: set[str]  = set()

        # ── Strategy 1: r/all keyword search ─────────────────────
        logger.info(f"[{brand}] Strategy 1: r/all keyword search")
        n_keywords = min(
            ONLINE["newsapi"]["keywords_per_brand"],
            len(keywords)
        )
        for kw in keywords[:n_keywords]:
            try:
                for sub in self.reddit.subreddit("all").search(
                    kw,
                    sort="new",
                    time_filter=self.cfg["search_time_filter"],
                    limit=post_limit,
                ):
                    if sub.id in seen_ids:
                        continue
                    if filter_nsfw and sub.over_18:
                        continue
                    record = self._extract_post(sub, brand, "r/all_search")
                    if len(record["full_text"]) >= min_len:
                        records.append(record)
                        seen_ids.add(sub.id)
                time.sleep(0.6)        # stay well under 100 req/min
            except Exception as e:
                logger.warning(f"[{brand}] r/all search '{kw}': {e}")

        logger.info(f"[{brand}] r/all search → {len(records)} posts so far")

        # ── Strategy 2: Brand subreddits .new() ──────────────────
        logger.info(f"[{brand}] Strategy 2: subreddit .new()")
        for sr_name in subreddits:
            try:
                sr = self.reddit.subreddit(sr_name)
                _ = sr.display_name          # verify subreddit exists
                for sub in sr.new(limit=post_limit):
                    if sub.id in seen_ids:
                        continue
                    if filter_nsfw and sub.over_18:
                        continue
                    combined = f"{sub.title} {sub.selftext}".lower()
                    if not any(k.lower() in combined for k in keywords):
                        continue    # not brand-relevant
                    record = self._extract_post(sub, brand, f"r/{sr_name}")
                    if len(record["full_text"]) >= min_len:
                        records.append(record)
                        seen_ids.add(sub.id)
                logger.info(f"[{brand}] r/{sr_name} ✓")
                time.sleep(0.6)
            except Exception as e:
                logger.warning(f"[{brand}] r/{sr_name}: {e}")

        posts_found = len(records)
        logger.info(f"[{brand}] {posts_found} posts total after dedup")

        # ── Comments from top-scoring posts ──────────────────────
        comment_limit  = self.cfg["comment_limit_per_post"]
        top_n          = self.cfg["top_posts_for_comments"]

        if records:
            df_posts   = pd.DataFrame(records)
            top_posts  = df_posts.nlargest(top_n, "score")
            logger.info(f"[{brand}] Fetching comments from {len(top_posts)} top posts")

            for _, row in top_posts.iterrows():
                try:
                    submission = self.reddit.submission(id=row["id"])
                    submission.comments.replace_more(limit=0)
                    for c in submission.comments.list()[:comment_limit]:
                        body = str(c.body)
                        if (len(body) < min_len
                                or body in ("[deleted]", "[removed]")):
                            continue
                        comment = self._extract_comment(
                            c, brand, row["id"], row["subreddit"]
                        )
                        records.append(comment)
                    time.sleep(0.6)
                except Exception as e:
                    logger.debug(f"Comment fetch failed for {row['id']}: {e}")

        comments_found = len(records) - posts_found
        logger.info(f"[{brand}] +{comments_found} comments → "
                    f"{len(records)} total records")
        return records

    def save(self, records: list[dict], brand: str) -> Path:
        """Save collected records to timestamped CSV in data/live/reddit/."""
        if not records:
            logger.warning(f"[{brand}] No Reddit records to save")
            return None

        df   = pd.DataFrame(records)
        path = resolve_path(ONLINE["paths"]["live_reddit"])
        out  = path / f"reddit_{brand.lower()}_{timestamp_str()}.csv"
        df.to_csv(out, index=False)
        logger.info(f"[{brand}] Saved {len(df)} Reddit records → {out}")
        return out


# ── NewsAPI Collector ────────────────────────────────────────────

class NewsCollector:
    """
    Collects brand-related news articles from NewsAPI.

    Free tier:
        - 100 requests/day
        - 1 month lookback
        - 100 articles per request
        - No credit card required

    Budget management:
        We use 3 keywords per brand = 3 API calls per pull.
        At 1 pull/hour that's 72 calls/day — safely under the 100 limit.
        config.yaml: newsapi.keywords_per_brand = 3

    Why this replaces Twitter:
        Twitter free tier only allows POSTING — reading costs $200/month.
        NewsAPI covers 150,000+ sources at no cost.
    """

    def __init__(self):
        if not NEWSAPI_AVAILABLE:
            raise ImportError(
                "Install newsapi-python: pip install newsapi-python==0.2.7"
            )

        api_key = os.getenv("NEWSAPI_KEY", "")
        if not api_key or "your_" in api_key:
            raise ValueError(
                "NEWSAPI_KEY not set in .env\n"
                "Register free at: https://newsapi.org/register"
            )

        self.client = NewsApiClient(api_key=api_key)
        self.cfg    = ONLINE["newsapi"]
        logger.info("NewsCollector initialised")

    def _build_full_text(self, article: dict) -> str:
        """Combine title + description + content into one searchable field."""
        parts = [
            article.get(field, "") or ""
            for field in self.cfg["text_fields"]
        ]
        combined = " ".join(filter(None, parts))
        return clean_text(combined[: self.cfg["max_full_text_chars"]])

    def collect(self, brand: str) -> list[dict]:
        """
        Collect articles for one brand using top N keywords.
        Deduplicates by URL.
        """
        keywords   = BRANDS_CFG["keywords"].get(brand, [brand])
        n_keywords = self.cfg["keywords_per_brand"]
        from_date  = (
            datetime.now() - timedelta(days=self.cfg["lookback_days"])
        ).strftime("%Y-%m-%d")
        to_date    = datetime.now().strftime("%Y-%m-%d")

        articles:  list[dict] = []
        seen_urls: set[str]   = set()

        for kw in keywords[:n_keywords]:
            try:
                resp = self.client.get_everything(
                    q=kw,
                    language=self.cfg["language"],
                    from_param=from_date,
                    to=to_date,
                    sort_by=self.cfg["sort_by"],
                    page_size=self.cfg["page_size"],
                )

                if resp.get("status") != "ok":
                    logger.warning(f"[{brand}] NewsAPI non-ok response for '{kw}': "
                                   f"{resp.get('message')}")
                    continue

                for a in resp.get("articles", []):
                    url = a.get("url", "")
                    if url in seen_urls:
                        continue
                    seen_urls.add(url)

                    full_text = self._build_full_text(a)
                    if not full_text:
                        continue

                    articles.append({
                        "id"          : url[-80:],          # URL tail as ID
                        "type"        : "news_article",
                        "platform"    : "news",
                        "brand"       : brand,
                        "keyword_used": kw,
                        "source_name" : a.get("source", {}).get("name", ""),
                        "source_id"   : a.get("source", {}).get("id", ""),
                        "title"       : clean_text(a.get("title", "")),
                        "description" : clean_text(a.get("description", "")),
                        "full_text"   : full_text,
                        "author"      : a.get("author", ""),
                        "url"         : url,
                        "created_utc" : a.get("publishedAt", ""),
                        "collected_at": utc_now(),
                    })

                logger.info(
                    f"[{brand}] NewsAPI '{kw}' → "
                    f"{resp['totalResults']} total, "
                    f"{len(resp['articles'])} fetched"
                )
                time.sleep(0.5)

            except Exception as e:
                logger.warning(f"[{brand}] NewsAPI '{kw}': {e}")

        logger.info(f"[{brand}] {len(articles)} news articles after dedup")
        return articles

    def save(self, articles: list[dict], brand: str) -> Path:
        """Save articles to timestamped CSV in data/live/news/."""
        if not articles:
            logger.warning(f"[{brand}] No news articles to save")
            return None

        df   = pd.DataFrame(articles)
        path = resolve_path(ONLINE["paths"]["live_news"])
        out  = path / f"news_{brand.lower()}_{timestamp_str()}.csv"
        df.to_csv(out, index=False)
        logger.info(f"[{brand}] Saved {len(df)} news articles → {out}")
        return out


# ── RSS Fallback Collector ───────────────────────────────────────

class RSSCollector:
    """
    Fallback collector using RSS feeds.
    Used when NewsAPI is rate-limited or unavailable.
    No API key required.
    """

    def __init__(self):
        if not FEEDPARSER_AVAILABLE:
            raise ImportError("Install feedparser: pip install feedparser==6.0.11")
        self.cfg = ONLINE.get("rss", {})

    def collect(self, brand: str) -> list[dict]:
        feeds = self.cfg.get("feeds", {}).get(brand, [])
        if not feeds:
            logger.info(f"[{brand}] No RSS feeds configured")
            return []

        articles: list[dict] = []
        for feed_url in feeds:
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries[:30]:
                    title   = clean_text(entry.get("title", ""))
                    summary = clean_text(entry.get("summary", ""))
                    url     = entry.get("link", "")
                    if not title:
                        continue
                    articles.append({
                        "id"          : url[-80:] if url else title[:80],
                        "type"        : "rss_article",
                        "platform"    : "news",
                        "brand"       : brand,
                        "source_name" : feed.feed.get("title", feed_url),
                        "title"       : title,
                        "full_text"   : f"{title} {summary}",
                        "url"         : url,
                        "created_utc" : entry.get("published", ""),
                        "collected_at": utc_now(),
                    })
                logger.info(f"[{brand}] RSS {feed_url} → {len(feed.entries)} entries")
                time.sleep(0.3)
            except Exception as e:
                logger.warning(f"[{brand}] RSS {feed_url}: {e}")

        return articles


# ── Combiner ─────────────────────────────────────────────────────

UNIFIED_COLS = [
    "id", "type", "platform", "brand",
    "full_text", "created_utc", "collected_at", "url",
]


def combine_and_save(brand: str) -> Optional[Path]:
    """
    Merge all recent Reddit + News CSVs for a brand into one
    deduplicated combined file.

    The combined file is what the preprocessing pipeline and
    model inference reads.
    """
    reddit_dir   = resolve_path(ONLINE["paths"]["live_reddit"])
    news_dir     = resolve_path(ONLINE["paths"]["live_news"])
    combined_dir = resolve_path(ONLINE["paths"]["live_combined"])
    combined_dir.mkdir(parents=True, exist_ok=True)

    dfs: list[pd.DataFrame] = []

    for csv_path in sorted(reddit_dir.glob(f"*{brand.lower()}*.csv")):
        try:
            df = pd.read_csv(csv_path)
            if "full_text" in df.columns:
                for col in UNIFIED_COLS:
                    if col not in df.columns:
                        df[col] = ""
                dfs.append(df[UNIFIED_COLS])
        except Exception as e:
            logger.warning(f"Could not read {csv_path}: {e}")

    for csv_path in sorted(news_dir.glob(f"*{brand.lower()}*.csv")):
        try:
            df = pd.read_csv(csv_path)
            if "full_text" in df.columns:
                for col in UNIFIED_COLS:
                    if col not in df.columns:
                        df[col] = ""
                dfs.append(df[UNIFIED_COLS])
        except Exception as e:
            logger.warning(f"Could not read {csv_path}: {e}")

    if not dfs:
        logger.warning(f"[{brand}] No files to combine")
        return None

    combined = (
        pd.concat(dfs, ignore_index=True)
        .drop_duplicates(subset=["id"])
        .dropna(subset=["full_text"])
        .query("full_text.str.len() >= 15")
    )

    out = combined_dir / f"live_combined_{brand.lower()}_latest.csv"
    combined.to_csv(out, index=False)
    logger.info(
        f"[{brand}] Combined → {len(combined)} records "
        f"({combined['platform'].value_counts().to_dict()}) → {out}"
    )
    return out


# ── Database Writer ───────────────────────────────────────────────

class DBWriter:
    """
    Writes collected records to Supabase (PostgreSQL).
    Uses INSERT ... ON CONFLICT DO NOTHING so re-runs are safe.
    Falls back to CSV-only mode if DB is not configured.
    """

    def __init__(self):
        db_url = os.getenv("DATABASE_URL", "")
        if not db_url or "your-project-ref" in db_url or not DB_AVAILABLE:
            logger.info("DBWriter: no DATABASE_URL — running in CSV-only mode")
            self.engine = None
            return

        self.engine = create_engine(db_url, pool_pre_ping=True)
        logger.info("DBWriter: connected to Supabase")

    def write(self, records: list[dict]) -> int:
        """
        Insert records into live_posts table.
        Returns number of rows actually inserted (not already existing).
        """
        if not self.engine or not records:
            return 0

        inserted = 0
        with self.engine.connect() as conn:
            for r in records:
                try:
                    conn.execute(
                        text("""
                            INSERT INTO live_posts
                                (id, platform, type, brand, full_text,
                                 url, author, score, created_utc, collected_at, metadata)
                            VALUES
                                (:id, :platform, :type, :brand, :full_text,
                                 :url, :author, :score, :created_utc::timestamptz,
                                 :collected_at::timestamptz, :metadata::jsonb)
                            ON CONFLICT (id) DO NOTHING
                        """),
                        {
                            "id"          : str(r.get("id", ""))[:200],
                            "platform"    : r.get("platform", ""),
                            "type"        : r.get("type", ""),
                            "brand"       : r.get("brand", ""),
                            "full_text"   : r.get("full_text", ""),
                            "url"         : r.get("url", ""),
                            "author"      : r.get("author", ""),
                            "score"       : r.get("score", None),
                            "created_utc" : r.get("created_utc", None),
                            "collected_at": r.get("collected_at", utc_now()),
                            "metadata"    : json.dumps({
                                k: v for k, v in r.items()
                                if k not in {
                                    "id","platform","type","brand","full_text",
                                    "url","author","score","created_utc","collected_at"
                                }
                            }),
                        },
                    )
                    inserted += 1
                except Exception as e:
                    logger.debug(f"DB insert skipped ({r.get('id')}): {e}")

            conn.commit()

        logger.info(f"DB: inserted {inserted}/{len(records)} records")
        return inserted


# ── Orchestrator ─────────────────────────────────────────────────

class Collector:
    """
    Top-level orchestrator. Runs a full collection cycle:
        1. Reddit posts + comments
        2. NewsAPI articles  (+ RSS fallback if NewsAPI fails)
        3. Combine & save unified CSV
        4. Write to Supabase DB

    Called by:
        - APScheduler in FastAPI app (scheduled pulls)
        - CLI for manual testing (--once flag)
    """

    def __init__(self):
        ensure_dirs()
        self.db = DBWriter()

        # Initialise collectors — each one logs a clear error if keys missing
        self.reddit = None
        self.news   = None
        self.rss    = None

        if ONLINE["reddit"]["enabled"] and PRAW_AVAILABLE:
            try:
                self.reddit = RedditCollector()
            except (ValueError, ImportError) as e:
                logger.warning(f"RedditCollector not available: {e}")

        if ONLINE["newsapi"]["enabled"] and NEWSAPI_AVAILABLE:
            try:
                self.news = NewsCollector()
            except (ValueError, ImportError) as e:
                logger.warning(f"NewsCollector not available: {e}")

        if ONLINE["rss"]["enabled"] and FEEDPARSER_AVAILABLE:
            try:
                self.rss = RSSCollector()
            except ImportError as e:
                logger.warning(f"RSSCollector not available: {e}")

    def run(self, brand: Optional[str] = None) -> dict:
        """
        Run a full collection cycle.
        brand: specific brand to collect, or None to collect all brands.

        Returns summary dict with counts per brand and source.
        """
        brands = (
            [brand] if brand
            else [BRANDS_CFG["primary"]] + BRANDS_CFG["competitors"]
        )

        summary: dict = {}

        for b in brands:
            logger.info(f"{'='*50}")
            logger.info(f"Collecting: {b}")
            logger.info(f"{'='*50}")

            all_records: list[dict] = []
            brand_summary: dict = {"reddit": 0, "news": 0, "rss": 0, "db_inserted": 0}

            # ── Reddit ───────────────────────────────────────────
            if self.reddit:
                try:
                    reddit_records = self.reddit.collect(b)
                    self.reddit.save(reddit_records, b)
                    all_records.extend(reddit_records)
                    brand_summary["reddit"] = len(reddit_records)
                except Exception as e:
                    logger.error(f"[{b}] Reddit collection failed: {e}")

            # ── NewsAPI ──────────────────────────────────────────
            news_records: list[dict] = []
            if self.news:
                try:
                    news_records = self.news.collect(b)
                    self.news.save(news_records, b)
                    all_records.extend(news_records)
                    brand_summary["news"] = len(news_records)
                except Exception as e:
                    logger.warning(f"[{b}] NewsAPI failed: {e} — trying RSS fallback")

            # ── RSS fallback ─────────────────────────────────────
            if self.rss and not news_records:
                try:
                    rss_records = self.rss.collect(b)
                    if rss_records:
                        # Save into news dir so combine_and_save picks it up
                        df_rss = pd.DataFrame(rss_records)
                        rss_path = (
                            resolve_path(ONLINE["paths"]["live_news"])
                            / f"rss_{b.lower()}_{timestamp_str()}.csv"
                        )
                        df_rss.to_csv(rss_path, index=False)
                        all_records.extend(rss_records)
                        brand_summary["rss"] = len(rss_records)
                        logger.info(f"[{b}] RSS fallback → {len(rss_records)} articles")
                except Exception as e:
                    logger.error(f"[{b}] RSS fallback failed: {e}")

            # ── Combine ──────────────────────────────────────────
            combine_and_save(b)

            # ── Write to DB ──────────────────────────────────────
            if all_records:
                brand_summary["db_inserted"] = self.db.write(all_records)

            summary[b] = brand_summary
            total = sum(v for k,v in brand_summary.items() if k != "db_inserted")
            logger.info(
                f"[{b}] Done — {total} records "
                f"(Reddit={brand_summary['reddit']}, "
                f"News={brand_summary['news']}, "
                f"RSS={brand_summary['rss']}, "
                f"DB={brand_summary['db_inserted']})"
            )

        return summary


# ── Scheduler ────────────────────────────────────────────────────

def start_scheduler(collector: Optional[Collector] = None):
    """
    Start APScheduler with Reddit (30 min) and NewsAPI (60 min) jobs.
    Called by FastAPI app startup event.

    APScheduler is not in requirements.txt — install separately:
        pip install apscheduler==3.10.4
    """
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
    except ImportError:
        logger.error(
            "APScheduler not installed. "
            "pip install apscheduler==3.10.4\n"
            "Or call Collector().run() manually."
        )
        return None

    c = collector or Collector()

    scheduler = BackgroundScheduler()

    # Reddit every 30 min
    scheduler.add_job(
        func=lambda: c.run(),
        trigger="interval",
        minutes=ONLINE["reddit"]["schedule_interval_minutes"],
        id="reddit_collector",
        name="Reddit collection",
        replace_existing=True,
    )

    # NewsAPI every 60 min (separate job so timing is independent)
    scheduler.add_job(
        func=lambda: c.run(),
        trigger="interval",
        minutes=ONLINE["newsapi"]["schedule_interval_minutes"],
        id="news_collector",
        name="NewsAPI collection",
        replace_existing=True,
    )

    scheduler.start()
    logger.info(
        f"Scheduler started — "
        f"Reddit every {ONLINE['reddit']['schedule_interval_minutes']} min, "
        f"NewsAPI every {ONLINE['newsapi']['schedule_interval_minutes']} min"
    )
    return scheduler


# ── CLI ───────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Brand Sentiment Monitor — live data collector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.data.collector --once
  python -m src.data.collector --brand Nike --once
  python -m src.data.collector --brand Nike --scheduled
  python -m src.data.collector --combine-only --brand Nike
        """,
    )
    p.add_argument("--brand", type=str, default=None,
                   help="Brand to collect (default: all brands from config)")
    p.add_argument("--once", action="store_true",
                   help="Run one collection cycle and exit")
    p.add_argument("--scheduled", action="store_true",
                   help="Run continuously on schedule (blocks)")
    p.add_argument("--combine-only", action="store_true",
                   help="Only combine existing CSVs — no API calls")
    p.add_argument("--dry-run", action="store_true",
                   help="Initialise collectors and validate credentials, then exit")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.combine_only:
        brands = [args.brand] if args.brand else (
            [BRANDS_CFG["primary"]] + BRANDS_CFG["competitors"]
        )
        for b in brands:
            combine_and_save(b)

    elif args.dry_run:
        logger.info("Dry run — validating credentials only")
        c = Collector()
        logger.info("Collector initialised successfully — credentials valid")
        logger.info(f"Reddit  : {'✅' if c.reddit else '❌ not configured'}")
        logger.info(f"NewsAPI : {'✅' if c.news else '❌ not configured'}")
        logger.info(f"RSS     : {'✅' if c.rss else '❌ not configured'}")
        logger.info(f"Database: {'✅' if c.db.engine else '⚠️  CSV-only mode'}")

    elif args.once:
        c = Collector()
        summary = c.run(brand=args.brand)
        logger.info(f"Collection complete: {json.dumps(summary, indent=2)}")

    elif args.scheduled:
        c = Collector()
        scheduler = start_scheduler(c)
        if scheduler:
            logger.info("Running on schedule — press Ctrl+C to stop")
            try:
                import time as _time
                while True:
                    _time.sleep(60)
            except KeyboardInterrupt:
                scheduler.shutdown()
                logger.info("Scheduler stopped")
    else:
        print("Specify --once, --scheduled, --combine-only, or --dry-run")
        print("Run with --help for full usage.")
