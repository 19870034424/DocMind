import { useEffect, useState, useRef } from "react";
import { Link } from "react-router-dom";
import { listDocuments, uploadDocument, deleteDocument, DocListItem } from "../services/api";
import ErrorRetry from "../components/ErrorRetry";
import "./KnowledgeBase.css";

const ALLOWED = [".pdf", ".docx", ".md", ".markdown"];
const STATUS_LABEL: Record<string, string> = {
  uploaded: "已上传", queued: "排队中", parsing: "解析中", chunking: "分块中", ready: "就绪", error: "错误",
};
const STATUS_CLASS: Record<string, string> = {
  ready: "green", parsing: "yellow", chunking: "yellow", uploaded: "gray", queued: "gray", error: "red",
};
const FORMAT_ICONS: Record<string, string> = {
  pdf: "PDF",
  docx: "DOC",
  doc: "DOC",
  md: "MD",
};

export default function KnowledgeBase() {
  const [docs, setDocs] = useState<DocListItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [error, setError] = useState("");
  const [loadError, setLoadError] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  async function fetchDocs() {
    setLoading(true);
    setLoadError(false);
    try {
      setDocs(await listDocuments());
    } catch {
      setLoadError(true);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => { fetchDocs(); }, []);

  // 上传为异步管线: 有文档处理中时轮询刷新状态
  useEffect(() => {
    const processing = docs.some((d) => ["uploaded", "queued", "parsing", "chunking"].includes(d.status));
    if (!processing) return;
    const t = setTimeout(async () => {
      try {
        setDocs(await listDocuments());
      } catch { /* 下一轮重试 */ }
    }, 2000);
    return () => clearTimeout(t);
  }, [docs]);

  async function handleFile(file: File) {
    const ext = "." + file.name.split(".").pop()?.toLowerCase();
    if (!ALLOWED.includes(ext)) {
      setError(`不支持的文件格式: ${ext}`);
      return;
    }
    setError("");
    setUploading(true);
    try {
      await uploadDocument(file);
      await fetchDocs();
    } catch (e: any) {
      setError(e.message || "上传失败");
    } finally {
      setUploading(false);
    }
  }

  async function handleDelete(id: string) {
    if (!confirm("确定删除此文档？")) return;
    try {
      await deleteDocument(id);
      setDocs((prev) => prev.filter((d) => d.id !== id));
    } catch {
      setError("删除失败");
    }
  }

  return (
    <div className="kb-page">
      <h1 className="page-title">本地知识库</h1>
      <p className="page-subtitle">上传文档，AI 自动解析并生成分析报告、知识图谱和智能问答</p>

      {/* Upload zone */}
      <div
        className={`kb-upload ${dragOver ? "dragover" : ""} ${uploading ? "uploading" : ""}`}
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault(); setDragOver(false);
          const file = e.dataTransfer.files[0];
          if (file) handleFile(file);
        }}
      >
        <input
          ref={inputRef} type="file" accept={ALLOWED.join(",")}
          style={{ display: "none" }}
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) handleFile(file);
            e.target.value = "";
          }}
        />
        {uploading ? (
          <div className="upload-state">
            <div className="spinner" />
            <span>正在解析文档并生成向量索引...</span>
          </div>
        ) : (
          <div className="upload-state">
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="var(--primary)" strokeWidth="2" strokeLinecap="round">
              <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M17 8l-5-5-5 5M12 3v12"/>
            </svg>
            <span className="upload-text">拖拽文档到此处上传，或点击选择文件</span>
            <span className="upload-hint">支持 PDF、Word (.docx)、Markdown — 最大 50MB（不支持旧版 .doc）</span>
          </div>
        )}
      </div>

      {error && <p className="error-msg">{error}</p>}

      {/* Document grid */}
      {loadError ? (
        <ErrorRetry message="文档列表加载失败" onRetry={fetchDocs} />
      ) : loading ? (
        <div className="kb-grid">
          {[1,2,3].map(i => <div key={i} className="kb-card skeleton"><div className="skeleton-pulse" style={{height:120}} /></div>)}
        </div>
      ) : docs.length === 0 ? (
        <div className="kb-empty">
          <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="var(--text-muted)" strokeWidth="1.5" strokeLinecap="round">
            <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/>
            <polyline points="14 2 14 8 20 8"/>
          </svg>
          <p>知识库为空</p>
          <span>上传第一份文档开始探索</span>
        </div>
      ) : (
        <div className="kb-grid">
          {docs.map((doc) => (
            <div key={doc.id} className="kb-card">
              <div className="card-header">
                <span className={`format-badge ${doc.format || "md"}`}>
                  {FORMAT_ICONS[doc.format || "md"] || doc.format}
                </span>
                <span className={`status-dot ${STATUS_CLASS[doc.status]}`}>
                  {STATUS_LABEL[doc.status]}
                </span>
              </div>
              <h3 className="card-title">
                {doc.title || doc.id}
              </h3>
              <div className="card-meta">
                {doc.pages > 0 && <span>{doc.pages} 页</span>}
                <span>{new Date(doc.created_at).toLocaleDateString("zh-CN")}</span>
              </div>
              <div className="card-actions">
                <Link to={`/documents/${doc.id}/view`} className="btn btn-sm btn-outline">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>
                  查看原文
                </Link>
                {doc.status === "ready" && (
                  <>
                    <Link to={`/documents/${doc.id}/report`} className="btn btn-sm">
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
                      分析报告
                    </Link>
                    <Link to={`/documents/${doc.id}/kg`} className="btn btn-sm btn-outline">
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><circle cx="12" cy="12" r="3"/><circle cx="5" cy="5" r="2"/><circle cx="19" cy="5" r="2"/><circle cx="5" cy="19" r="2"/><circle cx="19" cy="19" r="2"/><line x1="7" y1="6" x2="10.5" y3="10.5"/></svg>
                      知识图谱
                    </Link>
                  </>
                )}
                <button className="btn btn-sm btn-danger" onClick={() => handleDelete(doc.id)}>
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2"/></svg>
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
