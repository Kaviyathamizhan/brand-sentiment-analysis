import { useEffect, useState, useContext } from "react";
import { useNavigate } from "react-router-dom";
import { getMetrics, getBrands, getAlerts } from "../services/api";
import SentimentChart from "../components/SentimentChart";
import { C, card, badge, mono } from "../theme";
import { BrandContext } from "../context/BrandContext";

const PAGE = { background: C.void, minHeight: "100vh", padding: "44px 56px", animation: "fadeUp .35s ease both" };
const WINDOWS = [{ label: "1h", val: 1 }, { label: "6h", val: 6 }, { label: "24h", val: 24 }, { label: "7d", val: 168 }];

export default function Dashboard() {
  const navigate = useNavigate();

  const [brands, setBrands]         = useState([]);
  const [selectedBrand, setSelectedBrand] = useState("Nike");
  const [hours, setHours]           = useState(24);
  const [metrics, setMetrics]       = useState(null);
  const [alertCount, setAlertCount] = useState(0);
  const [loading, setLoading]       = useState(true);
  const [bgRefreshing, setBgRefreshing] = useState(false);

  // Hook into our global data Context so network payloads survive React unmounts
  const { metricsCache, globalAlerts, setGlobalAlerts } = useContext(BrandContext);

  // ── Fetch brands once ──────────────────────────────────────────────────────
  useEffect(() => {
    getBrands().then(setBrands).catch(() => {});
  }, []);

  // ── Poll alerts count (low frequency) ─────────────────────────────────────
  useEffect(() => {
    const poll = () => getAlerts().then(a => {
        setAlertCount(a.length);
        setGlobalAlerts(a);
    }).catch(() => {});
    poll();
    const id = setInterval(poll, 60_000);
    return () => clearInterval(id);
  }, []);

  // ── Fetch metrics when brand / time window changes ─────────────────────────
  useEffect(() => {
    const key = `${selectedBrand}:${hours}`;
    const cached = metricsCache.current[key];

    if (cached) {
      // Instant render from cache
      setMetrics(cached);
      setLoading(false);
      // Background refresh — silent, no spinner
      setBgRefreshing(true);
      getMetrics(selectedBrand, hours, { force: true })
        .then(data => {
          metricsCache.current[key] = data;
          setMetrics(data);
        })
        .catch(() => {})
        .finally(() => setBgRefreshing(false));
      return;
    }

    // First load for this brand/window
    setLoading(true);
    getMetrics(selectedBrand, hours)
      .then(data => {
        metricsCache.current[key] = data;
        setMetrics(data);
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [selectedBrand, hours]);

  const total = metrics?.total_posts || 0;
  const pPos  = (metrics?.sentiment?.positive || 0) * 100;
  const pNeu  = (metrics?.sentiment?.neutral  || 0) * 100;
  const pNeg  = (metrics?.sentiment?.negative || 0) * 100;

  const STATS = [
    { label: "Total Posts", value: total,                          color: C.cyan,   meta: `Last ${hours}h`, icon: "◈", glow: "rgba(56,189,248,0.15)"  },
    { label: "Positive",    value: metrics?.counts?.positive || 0, color: C.pos,    meta: `${pPos.toFixed(1)}%`,  icon: "▲", glow: "rgba(14,245,160,0.15)"  },
    { label: "Neutral",     value: metrics?.counts?.neutral  || 0, color: C.neu,    meta: `${pNeu.toFixed(1)}%`,  icon: "◆", glow: "rgba(245,200,66,0.15)"  },
    { label: "Negative",    value: metrics?.counts?.negative || 0, color: C.neg,    meta: `${pNeg.toFixed(1)}%`,  icon: "▼", glow: "rgba(255,77,112,0.15)"  },
  ];

  return (
    <div style={PAGE}>

      {/* Header */}
      <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", marginBottom: 38, paddingBottom: 28, borderBottom: `1px solid ${C.bDim}` }}>
        <div>
          <h1 style={{ fontFamily: C.mono, fontSize: 28, fontWeight: 700, color: C.t1, display: "flex", alignItems: "center", gap: 14 }}>
            <span style={{ display: "block", width: 5, height: 32, background: "linear-gradient(180deg,#38bdf8,#4f8ef7)", borderRadius: 3, boxShadow: "0 0 18px rgba(56,189,248,.8)" }} />
            Brand Sentiment Dashboard
          </h1>
          <p style={mono(11, C.t3, { letterSpacing: ".12em", textTransform: "uppercase", marginTop: 10, marginLeft: 19 })}>
            Real-time sentiment intelligence
            {bgRefreshing && <span style={{ marginLeft: 12, color: C.t4 }}>↻ refreshing…</span>}
          </p>
        </div>

        <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
          {/* Brand selector */}
          <select value={selectedBrand} onChange={e => setSelectedBrand(e.target.value)}
            style={{
              padding: "8px 14px", borderRadius: 8, border: `1px solid ${C.bDim}`,
              background: C.card, color: C.t1, fontFamily: C.mono, fontSize: 12,
              cursor: "pointer", outline: "none",
            }}>
            {(brands.length ? brands : ["Nike", "Adidas", "Puma", "Reebok"]).map(b =>
              <option key={b} value={b}>{b}</option>
            )}
          </select>

          {/* Time window */}
          <div style={{ display: "flex", gap: 4, background: C.card, padding: 3, borderRadius: 8, border: `1px solid ${C.bDim}` }}>
            {WINDOWS.map(w => (
              <button key={w.label} onClick={() => setHours(w.val)} style={{
                padding: "6px 12px", borderRadius: 6, border: "none", cursor: "pointer",
                fontFamily: C.mono, fontSize: 11, fontWeight: 700,
                background: hours === w.val ? "rgba(56,189,248,0.12)" : "transparent",
                color: hours === w.val ? C.cyan : C.t3,
              }}>{w.label}</button>
            ))}
          </div>

          {/* Alert bell */}
          <button onClick={() => navigate("/alerts")} style={{
            position: "relative", padding: "8px 12px", borderRadius: 8,
            border: `1px solid ${alertCount > 0 ? "rgba(255,77,112,0.3)" : C.bDim}`,
            background: alertCount > 0 ? "rgba(255,77,112,0.06)" : "transparent",
            cursor: "pointer", fontSize: 18, lineHeight: 1,
          }}>
            🔔
            {alertCount > 0 && (
              <span style={{
                position: "absolute", top: -4, right: -4,
                width: 18, height: 18, borderRadius: "50%",
                background: C.neg, display: "flex", alignItems: "center", justifyContent: "center",
                fontFamily: C.mono, fontSize: 9, fontWeight: 700, color: "#fff",
              }}>{alertCount}</span>
            )}
          </button>

          <span style={badge(C.pos)}>● LIVE</span>
        </div>
      </div>

      {loading ? (
        // Skeleton grid — never blank
        <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 20, marginBottom: 28 }}>
          {[1,2,3,4].map(i => (
            <div key={i} style={{ background: C.card, border: `1px solid ${C.bDim}`, borderRadius: 16, padding: "28px 26px", height: 120, animation: "pulse 1.5s ease-in-out infinite" }} />
          ))}
        </div>
      ) : (
        <>
          {/* Stat Cards */}
          <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 20, marginBottom: 28 }}>
            {STATS.map(({ label, value, color, meta, icon, glow }) => (
              <div key={label} onClick={() => navigate(`/brand/${selectedBrand}`)} style={{
                background: C.card, border: `1px solid ${C.bDim}`, borderRadius: 16, padding: "28px 26px",
                position: "relative", overflow: "hidden", cursor: "pointer",
                borderTop: `2px solid ${color}`, boxShadow: "0 4px 24px rgba(0,0,0,0.4)", transition: "all .2s",
              }}>
                <div style={{ position: "absolute", top: 0, right: 0, width: 110, height: 110, background: glow, borderRadius: "50%", filter: "blur(30px)", pointerEvents: "none" }} />
                <div style={mono(10, C.t3, { letterSpacing: ".14em", textTransform: "uppercase", marginBottom: 18 })}>{label}</div>
                <div style={{ fontFamily: C.mono, fontSize: 42, fontWeight: 700, color, lineHeight: 1, textShadow: `0 0 28px ${color}55` }}>{value}</div>
                <div style={{ display: "flex", alignItems: "center", gap: 7, marginTop: 14 }}>
                  <div style={{ width: 6, height: 6, borderRadius: "50%", background: color, boxShadow: `0 0 8px ${color}` }} />
                  <span style={mono(11, C.t3)}>{meta}</span>
                </div>
                <div style={{ position: "absolute", bottom: 10, right: 18, fontFamily: C.mono, fontSize: 64, color, opacity: .05, lineHeight: 1, pointerEvents: "none" }}>{icon}</div>
              </div>
            ))}
          </div>

          {/* Top Emotions + Topics */}
          {metrics && (
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20, marginBottom: 28 }}>
              <div style={card({ padding: "22px 26px" })}>
                <div style={mono(10, C.t3, { letterSpacing: ".12em", textTransform: "uppercase", marginBottom: 14 })}>Top Emotions</div>
                {(metrics.top_emotions || []).slice(0, 5).map(([emo, count]) => (
                  <div key={emo} style={{ marginBottom: 10 }}>
                    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                      <span style={{ fontFamily: C.sans, fontSize: 12, color: C.t2, textTransform: "capitalize" }}>{emo}</span>
                      <span style={mono(10, C.t3)}>{count}</span>
                    </div>
                    <div style={{ height: 6, background: C.void, borderRadius: 3, overflow: "hidden" }}>
                      <div style={{ height: "100%", width: `${Math.min(count / (metrics.total_posts || 1) * 100 * 5, 100)}%`, background: `linear-gradient(90deg,${C.violet}88,${C.violet})`, borderRadius: 3 }} />
                    </div>
                  </div>
                ))}
                {(!metrics.top_emotions || metrics.top_emotions.length === 0) && (
                  <div style={mono(11, C.t4)}>No emotion data yet</div>
                )}
              </div>

              <div style={card({ padding: "22px 26px" })}>
                <div style={mono(10, C.t3, { letterSpacing: ".12em", textTransform: "uppercase", marginBottom: 14 })}>Top Topics</div>
                <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
                  {(metrics.top_topics || []).map(t => (
                    <span key={t} style={{ ...badge(C.cyan), fontSize: 11, padding: "7px 16px" }}>{t}</span>
                  ))}
                  {(!metrics.top_topics || metrics.top_topics.length === 0) && (
                    <div style={mono(11, C.t4)}>No topic data yet</div>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* Chart */}
          <div style={{ ...card(), boxShadow: "0 8px 48px rgba(0,0,0,0.5)" }}>
            <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: 1, background: "linear-gradient(90deg,transparent,rgba(56,189,248,0.4),transparent)" }} />
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 8 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                <div style={{ width: 8, height: 8, borderRadius: "50%", background: C.pos, boxShadow: `0 0 12px ${C.pos}`, animation: "glowPulse 2.2s ease-in-out infinite" }} />
                <span style={mono(12, C.t2, { letterSpacing: ".1em", textTransform: "uppercase", fontWeight: 700 })}>
                  Sentiment Distribution — {selectedBrand}
                </span>
              </div>
              <span style={badge(C.pos)}>LIVE DATA</span>
            </div>
            <SentimentChart sentimentData={metrics?.sentiment} />
          </div>
        </>
      )}
    </div>
  );
}