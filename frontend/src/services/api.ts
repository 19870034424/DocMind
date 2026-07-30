const BASE = "/api";

export interface DocListItem {
  id: string;
  title: string;
  status: string;
  created_at: string;
  format: string | null;
  pages: number;
}

export interface DocDetail extends DocListItem {
  metadata: {
    filename: string;
    format: string;
    pages: number;
    uploaded_at: string;
    file_size: number;
  } | null;
  structured: StructuredDoc | null;
}

export interface StructuredDoc {
  title: string;
  sections: Section[];
}

export interface TableData {
  headers: string[];
  rows: string[][];
}

export interface Section {
  heading: string;
  level: number;
  content: string;
  page: number;
  tables: TableData[];
}

export interface ReportData {
  content: string;
  generated_at: string;
}

export interface MindmapNode {
  name: string;
  meta?: string | null;
  children: MindmapNode[];
}

export interface MindmapData {
  root: MindmapNode;
  generated_at: string;
}

export async function uploadDocument(file: File): Promise<DocListItem> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${BASE}/documents/upload`, { method: "POST", body: form });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || "Upload failed");
  }
  return res.json();
}

export async function listDocuments(): Promise<DocListItem[]> {
  const res = await fetch(`${BASE}/documents/`);
  if (!res.ok) throw new Error("Failed to fetch documents");
  return res.json();
}

export async function getDocument(id: string): Promise<DocDetail> {
  const res = await fetch(`${BASE}/documents/${id}`);
  if (!res.ok) throw new Error("Document not found");
  return res.json();
}

export async function deleteDocument(id: string): Promise<void> {
  const res = await fetch(`${BASE}/documents/${id}`, { method: "DELETE" });
  if (!res.ok) throw new Error("Failed to delete");
}

export async function generateReport(id: string): Promise<ReportData> {
  const res = await fetch(`${BASE}/documents/${id}/report`, { method: "POST" });
  if (!res.ok) throw new Error("Failed to generate report");
  return res.json();
}

export async function getReport(id: string): Promise<ReportData> {
  const res = await fetch(`${BASE}/documents/${id}/report`);
  if (!res.ok) throw new Error("Report not found");
  return res.json();
}

export async function deepDiveReport(id: string, sections: string[]): Promise<ReportData> {
  const res = await fetch(`${BASE}/documents/${id}/report/deep-dive`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sections }),
  });
  if (!res.ok) throw new Error("Failed to generate deep-dive");
  return res.json();
}

export async function generateMindmap(id: string): Promise<MindmapData> {
  const res = await fetch(`${BASE}/documents/${id}/mindmap`, { method: "POST" });
  if (!res.ok) throw new Error("Failed to generate mindmap");
  return res.json();
}

export async function getMindmap(id: string): Promise<MindmapData> {
  const res = await fetch(`${BASE}/documents/${id}/mindmap`);
  if (!res.ok) throw new Error("Mindmap not found");
  return res.json();
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export interface ChatResponse {
  answer: string;
  history: ChatMessage[];
}

export async function askQuestion(id: string, question: string): Promise<ChatResponse> {
  const res = await fetch(`${BASE}/documents/${id}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
  if (!res.ok) throw new Error("Failed to get answer");
  return res.json();
}

export async function getChatHistory(id: string): Promise<ChatMessage[]> {
  const res = await fetch(`${BASE}/documents/${id}/chat`);
  if (!res.ok) return [];
  const data = await res.json();
  return data.history || [];
}

export async function clearChatHistory(id: string): Promise<void> {
  await fetch(`${BASE}/documents/${id}/chat`, { method: "DELETE" });
}

// ── 知识图谱类型 ──────────────────────────────────────────

export type NodeType = "concept" | "definition" | "algorithm" | "property" | "example";

export type RelationType = "prerequisite" | "leads_to" | "contrasts" | "extends" | "part_of";

export interface GraphNode {
  id: string;
  name: string;
  type: NodeType;
  importance: number;
  summary: string;
  source_sections: string[];
  related_chunk_ids: string[];
}

export interface GraphEdge {
  source_id: string;
  target_id: string;
  relation: RelationType;
  description: string;
  source_sections: string[];
}

export type BuildingStatus = "idle" | "building" | "ready" | "error";

export interface KnowledgeGraphData {
  doc_id: string;
  document_title: string;
  nodes: GraphNode[];
  edges: GraphEdge[];
  status: BuildingStatus;
  generated_at: string;
  node_count: number;
  edge_count: number;
}

export interface NodeDetail {
  node: GraphNode;
  neighbors: {
    node: GraphNode;
    relation: string;
    relation_desc: string;
    direction: "incoming" | "outgoing";
  }[];
}

export interface SearchResults {
  results: GraphNode[];
}

// ── 知识图谱 API 函数 ──────────────────────────────────────

export async function buildKnowledgeGraph(docId: string): Promise<KnowledgeGraphData> {
  const res = await fetch(`/api/documents/${docId}/kg/build`, { method: "POST" });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "构建知识图谱失败");
  }
  return res.json();
}

export async function getKnowledgeGraph(docId: string): Promise<KnowledgeGraphData | null> {
  const res = await fetch(`/api/documents/${docId}/kg`);
  if (res.status === 404) return null;
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "获取知识图谱失败");
  }
  return res.json();
}

export async function getNodeDetail(docId: string, nodeId: string): Promise<NodeDetail> {
  const res = await fetch(`/api/documents/${docId}/kg/node/${nodeId}`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "获取节点详情失败");
  }
  return res.json();
}

export async function searchKGNodes(docId: string, query: string): Promise<SearchResults> {
  const res = await fetch(`/api/documents/${docId}/kg/search?q=${encodeURIComponent(query)}`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "搜索节点失败");
  }
  return res.json();
}
