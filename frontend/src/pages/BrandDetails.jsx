import { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import axios from "axios";
import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer } from "recharts";
import { C, card, badge, mono } from "../theme";

const API = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";
const PAGE = { background: C.void, minHeight: "100vh", padding: "44px 56px", animation: "fadeUp .35s ease both" };
const COLORS = { positive: C.pos, neutral: C.neu, negative: C.neg };

export default function BrandDetails() {
    const { brand } = useParams();
    const navigate = useNavigate();
    const [metrics, setMetrics] = useState(null);
    const [articles, setArticles] = useState([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        setLoading(true);
        Promise.all([
            axios.get(`${API}/metrics/${brand}`, { params: { hours: 24 } }),
            axios.get(`${API}/live-news`, { params: { brand, limit: 10, offset: 0 } }),
        ])
            .then(([mRes, aRes]) => {
                setMetrics(mRes.data);
                setArticles(aRes.data.articles || []);
            })
            .catch(() => { })
            .finally(() => setLoading(false));
    }, [brand]);

    if (loading) return (
        <div style={PAGE}>
            <div style={{ textAlign: "center", padding: "120px 0", color: C.cyan, fontFamily: C.mono, fontSize: 14 }}>
                Loading {brand} deep dive...
            </div>
        </div>
    );

    const m = metrics || {};
    const donutData = [
        { name: "Positive", value: (m.sentiment?.positive || 0) * 100, fill: C.pos },
        { name: "Neutral", value: (m.sentiment?.neutral || 0) * 100, fill: C.neu },
        { name: "Negative", value: (m.sentiment?.negative || 0) * 100, fill: C.neg },
    ];

    return (
        <div style={PAGE}>
            {/* Header */}
            <div style={{ display: "flex", alignItems: "center", gap: 16, marginBottom: 38, paddingBottom: 28, borderBottom: `1px solid ${C.bDim}` }}>
                <button onClick={() => navigate("/")} style={{
                    padding: "6px 14px", borderRadius: 8, border: `1px solid ${C.bDim}`,
                    background: "transparent", color: C.t3, fontFamily: C.mono, fontSize: 12, cursor: "pointer",
                }}>← Back</button>
                <h1 style={{ fontFamily: C.mono, fontSize: 28, fontWeight: 700, color: C.t1, margin: 0 }}>{brand}</h1>
                <span style={badge(C.cyan)}>DEEP DIVE</span>
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "2fr 3fr", gap: 24 }}>
                {/* Left panel (40%) */}
                <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>

                    {/* Donut chart */}
                    <div style={card({ padding: "24px", textAlign: "center" })}>
                        <div style={mono(10, C.t3, { letterSpacing: ".12em", textTransform: "uppercase", marginBottom: 10 })}>Sentiment Breakdown</div>
                        <ResponsiveContainer width="100%" height={220}>
                            <PieChart>
                                <Pie data={donutData} dataKey="value" cx="50%" cy="50%" innerRadius={55} outerRadius={90} paddingAngle={3} strokeWidth={0}>
                                    {donutData.map((d, i) => <Cell key={i} fill={d.fill} />)}
                                </Pie>
                                <Tooltip contentStyle={{ background: C.card, border: `1px solid ${C.bDim}`, borderRadius: 10, fontFamily: "'Space Mono',monospace", fontSize: 12 }} />
                            </PieChart>
                        </ResponsiveContainer>
                        <div style={mono(12, C.t2, { marginTop: 8 })}>{m.total_posts || 0} total posts</div>
                    </div>

                    {/* Top Emotions — progress bars */}
                    <div style={card({ padding: "20px 24px" })}>
                        <div style={mono(10, C.t3, { letterSpacing: ".12em", textTransform: "uppercase", marginBottom: 14 })}>Top Emotions</div>
                        {(m.top_emotions || []).slice(0, 5).map(([emo, count]) => {
                            const pct = Math.min((count / (m.counts?.positive + m.counts?.neutral + m.counts?.negative || 1)) * 100, 100);
                            return (
                                <div key={emo} style={{ marginBottom: 12 }}>
                                    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                                        <span style={{ fontFamily: C.sans, fontSize: 12, color: C.t2, textTransform: "capitalize" }}>{emo}</span>
                                        <span style={mono(10, C.t3)}>{(pct).toFixed(0)}%</span>
                                    </div>
                                    <div style={{ height: 8, background: C.void, borderRadius: 4, overflow: "hidden" }}>
                                        <div style={{ height: "100%", width: `${pct}%`, background: `linear-gradient(90deg,${C.violet}88,${C.violet})`, borderRadius: 4, transition: "width .5s" }} />
                                    </div>
                                </div>
                            );
                        })}
                        {(!m.top_emotions || m.top_emotions.length === 0) && <div style={mono(11, C.t4)}>No data yet</div>}
                    </div>

                    {/* Top Topics — pill tags */}
                    <div style={card({ padding: "20px 24px" })}>
                        <div style={mono(10, C.t3, { letterSpacing: ".12em", textTransform: "uppercase", marginBottom: 14 })}>Top Topics</div>
                        <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                            {(m.top_topics || []).map(t => (
                                <span key={t} style={{ ...badge(C.cyan), fontSize: 11, padding: "7px 16px", cursor: "pointer" }}>{t}</span>
                            ))}
                            {(!m.top_topics || m.top_topics.length === 0) && <div style={mono(11, C.t4)}>No topics yet</div>}
                        </div>
                    </div>
                </div>

                {/* Right panel (60%) */}
                <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>

                    {/* Recent post feed */}
                    <div style={card({ padding: "22px 26px" })}>
                        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 16 }}>
                            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                                <div style={{ width: 8, height: 8, borderRadius: "50%", background: C.pos, boxShadow: `0 0 12px ${C.pos}`, animation: "glowPulse 2.2s ease-in-out infinite" }} />
                                <span style={mono(11, C.t2, { letterSpacing: ".1em", textTransform: "uppercase", fontWeight: 700 })}>Recent Posts</span>
                            </div>
                            <span style={mono(10, C.t4)}>{articles.length} shown</span>
                        </div>

                        {articles.length === 0 ? (
                            <div style={{ padding: "40px 0", textAlign: "center" }}>
                                <div style={mono(12, C.t3)}>No posts collected yet for {brand}</div>
                            </div>
                        ) : (
                            <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                                {articles.map((a, i) => {
                                    const sc = COLORS[a.sentiment?.toLowerCase()] || C.t3;
                                    return (
                                        <div key={i} style={{
                                            padding: "14px 18px", borderRadius: 10, background: C.raised,
                                            border: `1px solid ${C.bDim}`, borderLeft: `3px solid ${sc}`,
                                        }}>
                                            <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 6 }}>
                                                <span style={{ ...badge(sc), fontSize: 9 }}>{a.sentiment}</span>
                                                {a.emotion !== "—" && <span style={mono(10, C.violet)}>✦ {a.emotion}</span>}
                                                <span style={{ marginLeft: "auto", ...mono(9, C.t4) }}>{a.source}</span>
                                            </div>
                                            <p style={{ fontFamily: C.sans, fontSize: 12, color: C.t2, lineHeight: 1.5, margin: 0 }}>
                                                {a.full_text?.slice(0, 180)}{a.full_text?.length > 180 ? "…" : ""}
                                            </p>
                                        </div>
                                    );
                                })}
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
}
