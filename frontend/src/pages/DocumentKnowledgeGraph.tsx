import { useEffect, useState } from "react";
import { useParams, Link } from "react-router-dom";
import { getKnowledgeGraph, buildKnowledgeGraph, getDocument, type KnowledgeGraphData, type DocDetail } from "../services/api";
import KnowledgeGraphViewer from "../components/KnowledgeGraphViewer";
import ErrorRetry from "../components/ErrorRetry";

export default function DocumentKnowledgeGraph() {
  const { id } = useParams<{ id: string }>();
  const docId = id!;
  const [kg, setKg] = useState<KnowledgeGraphData | null>(null);
  const [doc, setDoc] = useState<DocDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [building, setBuilding] = useState(false);
  const [error, setError] = useState("");
  const [loadError, setLoadError] = useState(false);

  const load = () => {
    setLoading(true);
    setLoadError(false);
    getDocument(docId).then(setDoc).catch(() => {});
    getKnowledgeGraph(docId).then(data => {
      if (data) setKg(data);
    }).catch(() => setLoadError(true))
      .finally(() => setLoading(false));
  };

  useEffect(() => { load(); }, [docId]);

  const handleBuild = async () => {
    setBuilding(true);
    setError("");
    try {
      const data = await buildKnowledgeGraph(docId);
      setKg(data);
    } catch (e: any) {
      setError(e.message || "构建失败");
    } finally {
      setBuilding(false);
    }
  };

  return (
    <div className="report-page">
      <div className="page-header">
        <Link to="/kb" className="back-link">← 返回知识库</Link>
        <h1 className="page-title">{doc?.title || "知识图谱"}</h1>
      </div>

      {loading ? (
        <div className="loading-state">加载中...</div>
      ) : loadError ? (
        <ErrorRetry message="加载失败" onRetry={load} />
      ) : building ? (
        <div className="loading-state">正在构建知识图谱，请稍候...</div>
      ) : kg ? (
        <KnowledgeGraphViewer kg={kg} docId={docId} />
      ) : (
        <div className="empty-state">
          <p>尚未构建知识图谱</p>
          {error && <p style={{ color: "var(--coral)", fontSize: 13 }}>{error}</p>}
          <button className="btn" onClick={handleBuild} disabled={building}>
            {building ? "构建中..." : "生成知识图谱"}
          </button>
        </div>
      )}
    </div>
  );
}
