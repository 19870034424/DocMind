import { useEffect, useRef, useState } from "react";
import { useParams, Link } from "react-router-dom";
import { getDocument, getChunks, DocDetail, ChunkItem } from "../services/api";
import MarkdownRenderer from "../components/MarkdownRenderer";
import MathText from "../components/MathText";
import "./DocumentView.css";

type Tab = "raw" | "chunks";

export default function DocumentView() {
  const { id } = useParams<{ id: string }>();
  const [doc, setDoc] = useState<DocDetail | null>(null);
  const [error, setError] = useState("");
  const [tab, setTab] = useState<Tab>("raw");

  // 分块状态
  const [chunks, setChunks] = useState<ChunkItem[]>([]);
  const [total, setTotal] = useState(0);
  const [offset, setOffset] = useState(0);
  const [chunksLoading, setChunksLoading] = useState(false);
  const PAGE = 20;

  useEffect(() => {
    if (!id) return;
    getDocument(id).then(setDoc).catch(() => setError("文档加载失败"));
  }, [id]);

  useEffect(() => {
    if (!id || tab !== "chunks") return;
    setChunksLoading(true);
    getChunks(id, offset, PAGE)
      .then((p) => {
        setChunks(p.items);
        setTotal(p.total);
      })
      .catch(() => setError("分块加载失败"))
      .finally(() => setChunksLoading(false));
  }, [id, tab, offset]);

  if (!id) return <p>无效的文档 ID</p>;

  const sections = doc?.structured?.sections ?? [];

  return (
    <div className="report-page">
      <div className="page-header">
        <Link to="/kb" className="back-link">← 返回知识库</Link>
        <h1 className="page-title">{doc?.title || "文档原文"}</h1>
      </div>

      {error && <p className="error-msg">{error}</p>}

      <div className="view-tabs">
        <button className={tab === "raw" ? "active" : ""} onClick={() => setTab("raw")}>
          AI 解析视图
        </button>
        <button className={tab === "chunks" ? "active" : ""} onClick={() => { setTab("chunks"); setOffset(0); }}>
          分块预览 {total > 0 ? `(${total})` : ""}
        </button>
      </div>

      {tab === "raw" && (
        <div className="doc-viewer">
          {sections.map((sec, i) => (
            <div key={i} className={`doc-section level-${sec.level}`}>
              {sec.heading && (
                <div className="section-heading" style={{ marginLeft: (sec.level - 1) * 20 }}>
                  {sec.level === 1 ? (
                    <h2>{sec.heading}</h2>
                  ) : sec.level === 2 ? (
                    <h3>{sec.heading}</h3>
                  ) : (
                    <h4>{sec.heading}</h4>
                  )}
                  {sec.page > 0 && <span className="page-tag">第 {sec.page} 页</span>}
                </div>
              )}
              {sec.content && <MarkdownRenderer content={sec.content} className="section-content" />}
              {sec.tables.length > 0 && sec.tables.map((t, ti) => (
                <table key={ti} className="doc-table">
                  <thead>
                    <tr>
                      {t.headers.map((h, hi) => <th key={hi}><MathText text={h} /></th>)}
                    </tr>
                  </thead>
                  <tbody>
                    {t.rows.map((row, ri) => (
                      <tr key={ri}>
                        {row.map((cell, ci) => <td key={ci}><MathText text={cell} /></td>)}
                      </tr>
                    ))}
                  </tbody>
                </table>
              ))}
            </div>
          ))}
        </div>
      )}

      {tab === "chunks" && (
        <div className="doc-viewer">
          {chunksLoading && <p>加载中…</p>}
          {!chunksLoading && chunks.length === 0 && <p>没有分块数据(文档可能尚未处理完成)</p>}
          {chunks.map((c) => (
            <div key={c.index} className="chunk-card">
              <div className="chunk-meta">
                <span className="chunk-index">#{c.index}</span>
                <span className="chunk-chars">{c.chars} 字</span>
                {c.heading_path && <span className="chunk-path">{c.heading_path}</span>}
              </div>
              <div className="chunk-text"><MarkdownRenderer content={c.text} /></div>
            </div>
          ))}
          {total > PAGE && (
            <div className="chunk-pager">
              <button disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - PAGE))}>
                上一页
              </button>
              <span>{offset + 1}–{Math.min(offset + PAGE, total)} / {total}</span>
              <button disabled={offset + PAGE >= total} onClick={() => setOffset(offset + PAGE)}>
                下一页
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
