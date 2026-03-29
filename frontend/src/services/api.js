/**
 * services/api.js
 *
 * Central API layer with built-in in-memory cache.
 * This is the ONLY file that talks to the backend.
 * Every page imports from here — never use axios directly in pages.
 *
 * HOW THE CACHE WORKS:
 *   - Each endpoint result is stored with a timestamp.
 *   - If you request the same data within TTL (default 5 min), you get
 *     the cached copy instantly — no network call, no spinner.
 *   - Switching Nike → Puma → Nike shows Nike instantly from cache.
 *   - Navigating away and back shows cached data while background refresh runs.
 */

import axios from "axios";

export const API_BASE = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

const http = axios.create({ baseURL: API_BASE, timeout: 120_000 });

// ─── In-memory cache ──────────────────────────────────────────────────────────
const _cache   = {};   // { [key]: { data, ts } }
const TTL_MS   = 5 * 60 * 1000;   // 5 minutes default TTL
const NEWS_TTL = 2 * 60 * 1000;   // 2 minutes for live news (fresher)

function _get(key) {
  const entry = _cache[key];
  if (!entry) return null;
  if (Date.now() - entry.ts > entry.ttl) { delete _cache[key]; return null; }
  return entry.data;
}

function _set(key, data, ttl = TTL_MS) {
  _cache[key] = { data, ts: Date.now(), ttl };
}

/** Force-invalidate one or all cache entries (call after Collect Now) */
export function invalidateCache(keyPrefix = null) {
  if (!keyPrefix) { Object.keys(_cache).forEach(k => delete _cache[k]); return; }
  Object.keys(_cache).filter(k => k.startsWith(keyPrefix)).forEach(k => delete _cache[k]);
}

// ─── API functions ────────────────────────────────────────────────────────────

/** GET /brands → string[] */
export async function getBrands() {
  const key = "brands";
  const hit = _get(key);
  if (hit) return hit;
  const { data } = await http.get("/brands");
  const result = data.brands || [];
  _set(key, result, TTL_MS * 6);   // brands rarely change — cache 30 min
  return result;
}

/** GET /metrics/:brand → BrandMetrics */
export async function getMetrics(brand, hours = 24, { force = false } = {}) {
  const key = `metrics:${brand}:${hours}`;
  if (!force) { const hit = _get(key); if (hit) return hit; }
  const { data } = await http.get(`/metrics/${brand}`, { params: { hours } });
  _set(key, data);
  return data;
}

/** GET /metrics for all 4 brands in parallel */
export async function getAllMetrics(hours = 24, { force = false } = {}) {
  const brands = ["Nike", "Adidas", "Puma", "Reebok"];
  return Promise.all(brands.map(b => getMetrics(b, hours, { force })));
}

/** GET /competitive → CompetitiveSnapshot */
export async function getCompetitive(brands = ["Nike", "Adidas", "Puma", "Reebok"], hours = 48, { force = false } = {}) {
  const key = `competitive:${brands.join(",")}:${hours}`;
  if (!force) { const hit = _get(key); if (hit) return hit; }
  const { data } = await http.get("/competitive", { params: { brands: brands.join(","), hours } });
  const result = data.brands || data;
  _set(key, result);
  return result;
}

/** GET /alerts → CrisisAlert[] */
export async function getAlerts({ force = false } = {}) {
  const key = "alerts";
  // Alerts always need to be fresh — short TTL
  if (!force) { const hit = _get(key); if (hit) return hit; }
  const { data } = await http.get("/alerts");
  const result = data.alerts || [];
  _set(key, result, 60_000);   // 1 minute TTL for alerts
  return result;
}

/** PATCH /alerts/:id/resolve */
export async function resolveAlert(id) {
  await http.patch(`/alerts/${id}/resolve`);
  invalidateCache("alerts");   // clear alerts cache so next fetch is fresh
}

/** GET /live-news → { articles, total, message } */
export async function getLiveNews(brand, limit = 20, offset = 0, { force = false } = {}) {
  const key = `news:${brand}:${limit}:${offset}`;
  if (!force) { const hit = _get(key); if (hit) return hit; }
  const { data } = await http.get("/live-news", { params: { brand, limit, offset } });
  _set(key, data, NEWS_TTL);
  return data;
}

/** POST /live-news/collect — fire and forget, returns immediately */
export async function collectNews(brand) {
  const { data } = await http.post("/live-news/collect", null, { params: { brand } });
  // After collecting, invalidate news cache for this brand so next read is fresh
  invalidateCache(`news:${brand}`);
  return data;
}

/** POST /predict → PredictionResponse */
export async function predict(text, platform = "reddit") {
  const { data } = await http.post("/predict", { text, platform });
  return data;
}

/** POST /predict/batch → { results, processed, skipped } */
export async function predictBatch(texts, platform = "reddit") {
  const { data } = await http.post("/predict/batch", { texts, platform });
  return data;
}

/** GET /health → { status } */
export async function checkHealth() {
  try {
    const { data } = await http.get("/health");
    return data.status === "ok";
  } catch {
    return false;
  }
}

/**
 * Lightweight cache warm — only fetches fast endpoints.
 * Metrics and live-news are fetched on-demand when the user visits a page.
 */
export function warmCache() {
  getBrands().catch(() => {});
  getAlerts().catch(() => {});
}