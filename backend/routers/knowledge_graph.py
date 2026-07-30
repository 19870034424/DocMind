"""知识图谱 API 路由"""
from fastapi import APIRouter, HTTPException
from models.knowledge_graph import KnowledgeGraph
from services.knowledge_graph import kg_builder
from storage.kg_store import kg_store

router = APIRouter(prefix="/api/documents/{doc_id}/kg", tags=["knowledge-graph"])


def _get_document(doc_id: str):
    """从文档路由的存储中获取文档记录"""
    from routers.documents import _documents
    doc = _documents.get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="文档不存在")
    return doc


@router.post("/build", response_model=KnowledgeGraph)
async def build_knowledge_graph(doc_id: str):
    """构建知识图谱"""
    doc = _get_document(doc_id)
    if doc.structured is None:
        raise HTTPException(status_code=400, detail="文档尚未解析完成")

    if doc.status.value not in ("ready",):
        raise HTTPException(status_code=400, detail="文档状态不允许构建图谱")

    kg = kg_builder.build(
        doc_id=doc_id,
        structured_doc=doc.structured,
        document_title=doc.title,
    )
    return kg


@router.get("/", response_model=KnowledgeGraph)
async def get_knowledge_graph(doc_id: str):
    """获取知识图谱"""
    kg = kg_store.load(doc_id)
    if kg is None:
        raise HTTPException(status_code=404, detail="知识图谱尚未构建")
    return kg


@router.get("/node/{node_id}")
async def get_node_detail(doc_id: str, node_id: str):
    """获取节点详情（含邻居节点）"""
    kg = kg_store.load(doc_id)
    if kg is None:
        raise HTTPException(status_code=404, detail="知识图谱尚未构建")

    target_node = None
    for node in kg.nodes:
        if node.id == node_id:
            target_node = node
            break
    if target_node is None:
        raise HTTPException(status_code=404, detail="节点不存在")

    neighbors: list[dict] = []
    for edge in kg.edges:
        if edge.source_id == node_id:
            neighbor_node = next((n for n in kg.nodes if n.id == edge.target_id), None)
            if neighbor_node:
                neighbors.append({
                    "node": neighbor_node.model_dump(),
                    "relation": edge.relation.value,
                    "relation_desc": edge.description,
                    "direction": "outgoing",
                })
        elif edge.target_id == node_id:
            neighbor_node = next((n for n in kg.nodes if n.id == edge.source_id), None)
            if neighbor_node:
                neighbors.append({
                    "node": neighbor_node.model_dump(),
                    "relation": edge.relation.value,
                    "relation_desc": edge.description,
                    "direction": "incoming",
                })

    return {
        "node": target_node.model_dump(),
        "neighbors": neighbors,
    }


@router.get("/search")
async def search_nodes(doc_id: str, q: str = ""):
    """搜索图谱节点"""
    kg = kg_store.load(doc_id)
    if kg is None:
        raise HTTPException(status_code=404, detail="知识图谱尚未构建")

    if not q:
        return {"results": []}

    q_lower = q.lower()
    results = [
        n.model_dump()
        for n in kg.nodes
        if q_lower in n.name.lower() or q_lower in n.summary.lower()
    ]
    results.sort(key=lambda n: 0 if n["name"] == q else 1)
    return {"results": results[:20]}
