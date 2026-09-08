import { useEffect, useState } from "react";
import { useParams, Link } from "react-router-dom";
import ScoreCard from "../components/ScoreCard";
import ReportViewer from "../components/ReportViewer";
import ChatBox from "../components/ChatBox";
import ErrorRetry from "../components/ErrorRetry";
import { getDocument, generateReport, getReport, deepDiveReport, DocDetail, ReportData } from "../services/api";
import "./DocumentReport.css";

type Tab = "report" | "original" | "chat";

export default function DocumentReport() {
  const { id } = useParams<{ id: string }>();
  const [doc, setDoc] = useState<DocDetail | null>(null);
  const [report, setReport] = useState<ReportData | null>(null);
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState("");
  const [loadError, setLoadError] = useState(false);
  const [selectedSections, setSelectedSections] = useState<string[]>([]);
  const [diveGenerating, setDiveGenerating] = useState(false);
  const [tab, setTab] = useState<Tab>("report");

  const load = () => {
    if (!id) return;
    setLoadError(false);
    getDocument(id).then(setDoc).catch(() => setLoadError(true));
    getReport(id).then(setReport).catch(() => {});
  };

  useEffect(() => { load(); }, [id]);

  async function handleGenerate() {
    if (!id) return;
    setGenerating(true);
    setError("");
    try {
      const result = await generateReport(id);
      setReport(result);
      setTab("report");
    } catch (e: any) {
      setError(e.message || "生成失败");
    } finally {
      setGenerating(false);
    }
  }

  function parseSections(content: string): { name: string; range: string; summary: string; stars: number }[] {
    const sections: { name: string; range: string; summary: string; stars: number }[] = [];
    const lines = content.split("\n");
    let inTable = false;
    for (const line of lines) {
      if (line.includes("|") && (line.includes("章节") || line.includes("页码") || line.includes("---"))) {
        inTable = true;
        continue;
      }
      if (inTable && line.startsWith("|") && !line.includes("---")) {
        const cells = line.split("|").map(c => c.trim()).filter(Boolean);
        if (cells.length >= 2) {
          const stars = (cells[3] || "").split("★").length - 1;
          sections.push({
            name: cells[0].replace(/[*_]/g, "").trim(),
            range: cells[1] || "",
            summary: cells[2] || "",
            stars: stars > 0 ? stars : 3,
          });
        }
      } else if (inTable && !line.startsWith("|")) {
        inTable = false;
      }
    }
    return sections;
  }

  async function handleDeepDive() {
    if (!id || selectedSections.length === 0) return;
    setDiveGenerating(true);
    setError("");
    try {
      const result = await deepDiveReport(id, selectedSections);
      setReport((prev) =>
        prev ? { ...prev, content: prev.content + "\n\n---\n\n## 深度分析\n\n" + result.content } : result
      );
    } catch (e: any) {
      setError(e.message || "深度分析失败");
    } finally {
      setDiveGenerating(false);
    }
  }

  if (!id) return <p>无效的文档 ID</p>;

  const sections = report ? parseSections(report.content) : [];
  const isLongDoc = sections.length >= 4;

  const structuredSections = doc?.structured?.sections ?? [];

  return (
    <div className="report-page">
      <div className="page-header">
        <Link to="/kb" className="back-link">← 返回知识库</Link>
        <h1 className="page-title">{doc?.title || "阅读报告"}</h1>
      </div>

      {loadError ? (
        <ErrorRetry message="文档加载失败" onRetry={load} />
      ) : error && <p className="error-msg">{error}</p>}

      {loadError ? null : !report ? (
        <div className="no-report">
          <p>尚未生成报告</p>
          <button className="btn" onClick={handleGenerate} disabled={generating}>
            {generating ? "生成中..." : "生成阅读报告"}
          </button>
          {structuredSections.length > 0 && (
            <div className="original-fallback">
              <h3>文档原文</h3>
              <div className="doc-content">
                {structuredSections.map((sec, i) => (
                  <div key={i} className="doc-block">
                    {sec.heading && (
                      <div className="doc-heading" style={{ marginLeft: (sec.level - 1) * 16 }}>
                        {sec.level === 1 ? <h2>{sec.heading}</h2> : sec.level === 2 ? <h3>{sec.heading}</h3> : <h4>{sec.heading}</h4>}
                        {sec.page > 0 && <span className="page-tag">第 {sec.page} 页</span>}
                      </div>
                    )}
                    {sec.content && <p className="doc-text">{sec.content}</p>}
                    {sec.tables && sec.tables.length > 0 && sec.tables.map((t, ti) => (
                      <table key={ti} className="doc-table">
                        <thead>
                          <tr>{t.headers.map((h, hi) => <th key={hi}>{h}</th>)}</tr>
                        </thead>
                        <tbody>
                          {t.rows.map((row, ri) => (
                            <tr key={ri}>{row.map((cell, ci) => <td key={ci}>{cell}</td>)}</tr>
                          ))}
                        </tbody>
                      </table>
                    ))}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      ) : (
        <>
          <div className="tab-bar">
            <button
              className={`tab-btn ${tab === "report" ? "active" : ""}`}
              onClick={() => setTab("report")}
            >
              AI 分析报告
            </button>
            <button
              className={`tab-btn ${tab === "original" ? "active" : ""}`}
              onClick={() => setTab("original")}
            >
              文档原文
            </button>
            <button
              className={`tab-btn ${tab === "chat" ? "active" : ""}`}
              onClick={() => setTab("chat")}
            >
              智能问答
            </button>
          </div>

          {tab === "report" && (
            <div className="report-layout">
              <div className="report-sidebar">
                <ScoreCard content={report.content} />
                <button
                  className="btn btn-outline btn-full"
                  onClick={handleGenerate}
                  disabled={generating}
                >
                  {generating ? "生成中..." : "重新生成"}
                </button>

                {isLongDoc && (
                  <div className="section-picker">
                    <h4>选择深入分析的章节</h4>
                    {sections.map((sec) => (
                      <label key={sec.name} className="section-checkbox">
                        <input
                          type="checkbox"
                          checked={selectedSections.includes(sec.name)}
                          onChange={(e) => {
                            if (e.target.checked) {
                              setSelectedSections((prev) => [...prev, sec.name]);
                            } else {
                              setSelectedSections((prev) => prev.filter((s) => s !== sec.name));
                            }
                          }}
                        />
                        <span>
                          {sec.name}
                          <span className="section-stars">{"★".repeat(sec.stars)}</span>
                        </span>
                      </label>
                    ))}
                    <button
                      className="btn btn-sm"
                      style={{ marginTop: 8, width: "100%" }}
                      onClick={handleDeepDive}
                      disabled={diveGenerating || selectedSections.length === 0}
                    >
                      {diveGenerating ? "分析中..." : "对选中章节深度分析"}
                    </button>
                  </div>
                )}
              </div>
              <div className="report-main">
                <ReportViewer content={report.content} />
              </div>
            </div>
          )}

          {tab === "original" && (
            <div className="report-main full-width">
              {structuredSections.length > 0 ? (
                <div className="doc-content">
                  {structuredSections.map((sec, i) => (
                    <div key={i} className="doc-block">
                      {sec.heading && (
                        <div className="doc-heading" style={{ marginLeft: (sec.level - 1) * 16 }}>
                          {sec.level === 1 ? <h2>{sec.heading}</h2> : sec.level === 2 ? <h3>{sec.heading}</h3> : <h4>{sec.heading}</h4>}
                          {sec.page > 0 && <span className="page-tag">第 {sec.page} 页</span>}
                        </div>
                      )}
                      {sec.content && <p className="doc-text">{sec.content}</p>}
                      {sec.tables && sec.tables.length > 0 && sec.tables.map((t, ti) => (
                        <table key={ti} className="doc-table">
                          <thead>
                            <tr>{t.headers.map((h, hi) => <th key={hi}>{h}</th>)}</tr>
                          </thead>
                          <tbody>
                            {t.rows.map((row, ri) => (
                              <tr key={ri}>{row.map((cell, ci) => <td key={ci}>{cell}</td>)}</tr>
                            ))}
                          </tbody>
                        </table>
                      ))}
                    </div>
                  ))}
                </div>
              ) : (
                <p className="no-content">无法加载文档原文</p>
              )}
            </div>
          )}

          {tab === "chat" && (
            <div className="report-main full-width">
              <ChatBox docId={id} />
            </div>
          )}

        </>
      )}
    </div>
  );
}
