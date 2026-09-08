"""知识图谱 API 路由"""
import asyncio
import logging

from fastapi import APIRouter, HTTPException

from models.knowledge_graph import KnowledgeGraph
from services.knowledge_graph import kg_builder
from storage.kg_store import kg_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/documents/{doc_id}/kg", tags=["knowledge-graph"])

# 正在构建中的文档 ID 集合，防止重复构建
_building_locks: set[str] = set()


def _get_document(doc_id: str) -> dict:
    """从 SQLite 获取文档记录"""
    import json

    from storage.db import get_document as db_get_document
    doc = db_get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="文档不存在")
    # 还原 structured JSON 为对象供 kg_builder 使用
    if doc.get("structured_json"):
        from models.document import StructuredDocument
        data = json.loads(doc["structured_json"])
        doc["structured"] = StructuredDocument(**data)
    return doc


@router.post("/build", response_model=KnowledgeGraph, summary="构建知识图谱(LLM 抽取概念与关系)")
async def build_knowledge_graph(doc_id: str):
    """构建知识图谱（后台线程执行，不阻塞其他请求）"""
    doc = _get_document(doc_id)
    structured = doc.get("structured")
    if structured is None:
        raise HTTPException(status_code=400, detail="文档尚未解析完成")

    if doc.get("status") not in ("ready",):
        raise HTTPException(status_code=400, detail="文档状态不允许构建图谱")

    if doc_id in _building_locks:
        raise HTTPException(status_code=409, detail="知识图谱正在构建中，请稍后再试")

    _building_locks.add(doc_id)
    try:
        # 在线程池中执行，不阻塞事件循环
        loop = asyncio.get_event_loop()
        kg = await loop.run_in_executor(
            None,
            lambda: kg_builder.build(
                doc_id=doc_id,
                structured_doc=structured,
                document_title=doc.get("title", ""),
            ),
        )
        return kg
    except Exception as e:
        logger.error(f"Knowledge graph build failed for {doc_id}: {e}")
        raise HTTPException(status_code=500, detail=f"知识图谱构建失败: {e!s}") from e
    finally:
        _building_locks.discard(doc_id)


@router.get("/", response_model=KnowledgeGraph, summary="获取已构建的知识图谱")
async def get_knowledge_graph(doc_id: str):
    """获取知识图谱"""
    kg = kg_store.load(doc_id)
    if kg is None:
        raise HTTPException(status_code=404, detail="知识图谱尚未构建")
    return kg


@router.get("/node/{node_id}", summary="获取节点详情及关联概念")
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


@router.get("/search", summary="按名称模糊搜索图谱节点")
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
