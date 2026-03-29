import { useState } from "react";
import axios from "axios";
import Papa from "papaparse";
import { C, card, badge, mono } from "../theme";

const API = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";
const PAGE = { background: C.void, minHeight: "100vh", padding: "44px 56px", animation: "fadeUp .35s ease both" };

const sentColor = (l) => {
  if (!l) return C.t3;
  const ll = l.toLowerCase();
  return ll === "positive" ? C.pos : ll === "negative" ? C.neg : C.neu;
};

export default function CsvUpload() {
  const [file, setFile] = useState(null);
  const [rows, setRows] = useState([]);
  const [platform, setPlatform] = useState("reddit");
  const [processing, setProcessing] = useState(false);
  const [results, setResults] = useState(null);
  const [summary, setSummary] = useState(null);

  function handleFile(selectedFile) {
    setFile(selectedFile);
    setResults(null);
    setSummary(null);
    Papa.parse(selectedFile, {
      header: true,
      complete: ({ data }) => {
        const valid = data.filter(r => r.text && r.text.trim());
        setRows(valid);
      },
    });
  }

  async function analyze() {
    if (rows.length === 0) return;
    setProcessing(true);
    setResults(null);
    try {
      const texts = rows.map(r => r.text);
      const { data } = await axios.post(`${API}/predict/batch`, { texts, platform });
      setResults(data.results);
      setSummary({ processed: data.processed, skipped: data.skipped });
    } catch (e) {
      console.error(e);
    } finally {
      setProcessing(false);
    }
  }

  function exportCsv() {
    if (!results) return;
    const csvRows = results.map((r, i) => ({
      "#": i + 1,
      text: r ? r.text : rows[i]?.text || "",
      brand: r ? (r.brands?.[0]?.brand || "No brand") : "Skipped",
      sentiment: r ? r.overall.label : "skipped",
      sarcasm: r ? (r.overall.is_sarcastic ? "Yes" : "No") : "",
      top_emotion: r ? (r.emotions?.top_emotion || "") : "",
      topic: r ? (r.topic?.label || "") : "",
    }));
    const csv = Papa.unparse(csvRows);
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "sentiment_results.csv";
    a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <div style={PAGE}>
      {/* Header */}
      <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", marginBottom: 38, paddingBottom: 28, borderBottom: `1px solid ${C.bDim}` }}>
        <div>
          <h1 style={{ fontFamily: C.mono, fontSize: 28, fontWeight: 700, color: C.t1, display: "flex", alignItems: "center", gap: 14 }}>
            <span style={{ display: "block", width: 5, height: 32, background: "linear-gradient(180deg,#f5c842,#f59e0b)", borderRadius: 3, boxShadow: "0 0 18px rgba(245,200,66,.8)" }} />
            CSV Batch Upload
          </h1>
          <p style={mono(11, C.t3, { letterSpacing: ".12em", textTransform: "uppercase", marginTop: 10, marginLeft: 19 })}>
            Upload CSV · parse client-side · batch analyse with AI
          </p>
        </div>
        <span style={badge(C.neu)}>POST /predict/batch</span>
      </div>

      {/* Step 1 — Upload */}
      {!results && (
        <div style={{ display: "grid", gridTemplateColumns: "1.5fr 1fr", gap: 24 }}>
          <div style={card()}>
            <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: 1, background: "linear-gradient(90deg,transparent,rgba(245,200,66,0.4),transparent)" }} />

            {/* Platform selector */}
            <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 18 }}>
              <span style={mono(10, C.t3, { letterSpacing: ".1em", textTransform: "uppercase" })}>Platform:</span>
              {["reddit", "news"].map(p => (
                <button key={p} onClick={() => setPlatform(p)} style={{
                  padding: "6px 16px", borderRadius: 6, border: `1px solid ${platform === p ? C.bGlow : C.bDim}`,
                  background: platform === p ? "rgba(56,189,248,0.1)" : "transparent",
                  color: platform === p ? C.cyan : C.t3, fontFamily: C.mono, fontSize: 11,
                  fontWeight: 600, cursor: "pointer", textTransform: "capitalize",
                }}>{p}</button>
              ))}
            </div>

            {/* Drop zone */}
            <label htmlFor="csv-file" style={{
              display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center",
              gap: 10, padding: "52px 32px", borderRadius: 14, cursor: "pointer",
              border: `1px dashed rgba(245,200,66,0.35)`, background: "rgba(245,200,66,0.03)", textAlign: "center",
            }}>
              <div style={{ fontSize: 44, opacity: file ? .9 : .4, lineHeight: 1, color: file ? C.neu : C.t3 }}>
                {file ? "✓" : "⬆"}
              </div>
              {file ? (
                <>
                  <div style={{ fontFamily: C.mono, fontSize: 15, fontWeight: 700, color: C.neu }}>{file.name}</div>
                  <div style={mono(11, C.t3)}>{rows.length} rows loaded · {(file.size / 1024).toFixed(1)} KB</div>
                </>
              ) : (
                <>
                  <div style={{ fontFamily: C.mono, fontSize: 14, fontWeight: 700, color: C.t2 }}>Drop your CSV file here</div>
                  <div style={mono(11, C.t3)}>or click to browse — must have a "text" column</div>
                </>
              )}
            </label>
            <input id="csv-file" type="file" accept=".csv" style={{ display: "none" }}
              onChange={e => e.target.files[0] && handleFile(e.target.files[0])} />

            {/* Analyse button */}
            <button onClick={analyze} disabled={processing || rows.length === 0} style={{
              marginTop: 20, padding: "13px 28px", borderRadius: 10,
              background: "rgba(56,189,248,0.12)", border: `1px solid rgba(56,189,248,0.45)`,
              color: C.cyan, fontFamily: C.mono, fontSize: 12, fontWeight: 700,
              letterSpacing: ".09em", textTransform: "uppercase",
              cursor: processing || rows.length === 0 ? "not-allowed" : "pointer",
              opacity: processing || rows.length === 0 ? 0.5 : 1,
            }}>
              {processing ? `Analysing ${rows.length} posts...` : `▶  Analyse ${rows.length} posts`}
            </button>
          </div>

          {/* Right: info cards */}
          <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
            {[
              { label: "Parsing", value: "Client-side (PapaParse)", icon: "◈", color: C.cyan },
              { label: "API", value: "POST /predict/batch", icon: "▶", color: C.violet },
              { label: "Output", value: "Labeled + export CSV", icon: "✓", color: C.pos },
              { label: "No-brand rows", value: "Shown as 'Skipped'", icon: "◆", color: C.neu },
            ].map(({ label, value, icon, color }) => (
              <div key={label} style={{
                background: C.card, border: `1px solid ${C.bDim}`, borderLeft: `3px solid ${color}`,
                borderRadius: 12, padding: "20px 22px", display: "flex", alignItems: "center", gap: 16,
              }}>
                <span style={{ fontSize: 20, color, opacity: .85, width: 24, textAlign: "center" }}>{icon}</span>
                <div>
                  <div style={mono(9, C.t3, { letterSpacing: ".1em", textTransform: "uppercase", marginBottom: 4 })}>{label}</div>
                  <div style={{ fontFamily: C.sans, fontSize: 14, fontWeight: 600, color: C.t2 }}>{value}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Step 2+3 — Results */}
      {results && (
        <>
          {/* Summary banner */}
          <div style={{ display: "flex", alignItems: "center", gap: 16, marginBottom: 24, padding: "16px 24px", borderRadius: 12, background: "rgba(14,245,160,0.06)", border: `1px solid rgba(14,245,160,0.25)` }}>
            <span style={{ fontSize: 20 }}>✓</span>
            <span style={{ fontFamily: C.sans, fontSize: 14, color: C.pos, fontWeight: 600 }}>
              {summary?.processed} processed, {summary?.skipped} skipped (no brand detected)
            </span>
            <button onClick={exportCsv} style={{
              marginLeft: "auto", padding: "8px 18px", borderRadius: 8,
              border: `1px solid ${C.bGlow}`, background: "rgba(56,189,248,0.08)",
              color: C.cyan, fontFamily: C.mono, fontSize: 11, fontWeight: 700, cursor: "pointer",
            }}>⬇ Export CSV</button>
            <button onClick={() => { setResults(null); setSummary(null); setFile(null); setRows([]); }} style={{
              padding: "8px 18px", borderRadius: 8, border: `1px solid ${C.bDim}`,
              background: "transparent", color: C.t3, fontFamily: C.mono, fontSize: 11, cursor: "pointer",
            }}>✕ New Upload</button>
          </div>

          {/* Results table */}
          <div style={{ ...card({ padding: "0", overflow: "auto" }) }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontFamily: C.sans, fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: `1px solid ${C.bDim}` }}>
                  {["#", "Text", "Brand(s)", "Sentiment", "Sarcasm", "Top Emotion", "Topic"].map(h => (
                    <th key={h} style={{ padding: "14px 16px", textAlign: "left", fontFamily: C.mono, fontSize: 10, color: C.t3, letterSpacing: ".1em", textTransform: "uppercase", fontWeight: 700 }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {results.map((r, i) => (
                  <tr key={i} style={{ borderBottom: `1px solid ${C.bDim}22` }}>
                    <td style={{ padding: "12px 16px", color: C.t4, fontFamily: C.mono, fontSize: 11 }}>{i + 1}</td>
                    <td style={{ padding: "12px 16px", color: r ? C.t2 : C.t4, maxWidth: 300, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", fontStyle: r ? "normal" : "italic" }}>
                      {r ? r.text.slice(0, 80) : rows[i]?.text?.slice(0, 80) || "—"}
                    </td>
                    <td style={{ padding: "12px 16px" }}>
                      {r && r.brands?.length > 0
                        ? r.brands.map(b => b.brand).join(", ")
                        : <span style={{ color: C.t4, fontStyle: "italic" }}>No brand</span>
                      }
                    </td>
                    <td style={{ padding: "12px 16px" }}>
                      {r ? <span style={{ ...badge(sentColor(r.overall.label)), fontSize: 9 }}>{r.overall.label}</span>
                        : <span style={{ color: C.t4, fontStyle: "italic" }}>—</span>}
                    </td>
                    <td style={{ padding: "12px 16px", color: C.t3, fontFamily: C.mono, fontSize: 11 }}>
                      {r ? (r.overall.is_sarcastic ? "⚠ Yes" : "No") : "—"}
                    </td>
                    <td style={{ padding: "12px 16px", color: C.violet, fontFamily: C.mono, fontSize: 11, textTransform: "capitalize" }}>
                      {r ? (r.emotions?.top_emotion || "—") : "—"}
                    </td>
                    <td style={{ padding: "12px 16px", color: C.cyan, fontFamily: C.mono, fontSize: 11 }}>
                      {r ? (r.topic?.label || "—") : "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}