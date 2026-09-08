"""Agentic RAG — 工具函数实现

6 个工具: search_document / get_section / query_knowledge_graph / search_keyword / search_web / reflect_on_answer
工具调度: execute_tool() 统一入口 + format_tool_result() 人类可读格式化
"""
import json
import logging

from config import settings
from services.chunker import chunker
from storage.db import get_document
from storage.kg_store import kg_store
from storage.milvus_client import milvus_client

logger = logging.getLogger(__name__)

MAX_TOP_K = 20
MIN_HEADING_LEN = 2


def tool_search_document(doc_id: str, query: str, top_k: int = 10) -> dict:
    """混合检索: BM25 + 向量 RRF 融合"""
    if not query or not query.strip():
        return {"error": "查询不能为空", "tool": "search_document"}
    top_k = max(1, min(top_k, MAX_TOP_K))
    try:
        query_embedding = chunker.embed([query.strip()])[0]
        chunks = milvus_client.hybrid_query(
            doc_id, query_embedding, query.strip(), top_k=top_k
        )
        hits = []
        for c in chunks:
            meta = c.get("metadata", {})
            hits.append({
                "text": c.get("text", ""),
                "heading": meta.get("heading_path", ""),
                "page": meta.get("page", 0),
            })
        return {"hits": hits, "total": len(hits), "tool": "search_document"}
    except Exception as e:
        logger.error(f"search_document failed: {e}")
        return {"error": str(e), "tool": "search_document"}


def tool_get_section(doc_id: str, heading: str) -> dict:
    """获取指定章节的完整内容"""
    if not heading or len(heading.strip()) < MIN_HEADING_LEN:
        return {"error": "章节标题至少需要 2 个字符", "tool": "get_section"}
    heading = heading.strip()
    try:
        row = get_document(doc_id)
        if not row:
            return {"error": "文档不存在", "tool": "get_section"}
        structured_json = row.get("structured_json")
        if not structured_json:
            return {"error": "文档尚未解析", "tool": "get_section"}
        data = json.loads(structured_json) if isinstance(structured_json, str) else structured_json
        sections = data.get("sections", [])
        heading_lower = heading.lower()
        matches = []
        for s in sections:
            s_heading = s.get("heading", "")
            if heading_lower in s_heading.lower():
                matches.append({
                    "heading": s_heading,
                    "level": s.get("level", 1),
                    "content": s.get("content", ""),
                    "page": s.get("page", 0),
                })
        if not matches:
            return {
                "hits": [],
                "total": 0,
                "available_headings": [s.get("heading", "") for s in sections[:20]],
                "tool": "get_section",
            }
        return {"hits": matches, "total": len(matches), "tool": "get_section"}
    except Exception as e:
        logger.error(f"get_section failed: {e}")
        return {"error": str(e), "tool": "get_section"}


def tool_query_knowledge_graph(doc_id: str, concept: str) -> dict:
    """查询知识图谱中的概念及其关联"""
    if not concept or not concept.strip():
        return {"error": "概念名不能为空", "tool": "query_knowledge_graph"}
    concept = concept.strip()
    try:
        kg = kg_store.load(doc_id)
        if kg is None:
            return {"available": False, "message": "知识图谱尚未构建", "tool": "query_knowledge_graph"}
        concept_lower = concept.lower()
        matched_node = None
        for node in kg.nodes:
            if concept_lower in node.name.lower():
                matched_node = node
                break
        if matched_node is None:
            return {
                "found": False,
                "message": f"未找到概念 '{concept}'",
                "similar": [n.name for n in kg.nodes if concept_lower[:2] in n.name.lower()][:5],
                "tool": "query_knowledge_graph",
            }
        neighbors = []
        for edge in kg.edges:
            if edge.source_id == matched_node.id:
                tgt = next((n for n in kg.nodes if n.id == edge.target_id), None)
                if tgt:
                    neighbors.append({
                        "concept": tgt.name,
                        "relation": edge.relation.value,
                        "description": edge.description,
                    })
            elif edge.target_id == matched_node.id:
                src = next((n for n in kg.nodes if n.id == edge.source_id), None)
                if src:
                    neighbors.append({
                        "concept": src.name,
                        "relation": f"inverse_{edge.relation.value}",
                        "description": edge.description,
                    })
        return {
            "found": True,
            "concept": {
                "name": matched_node.name,
                "type": matched_node.type.value,
                "summary": matched_node.summary,
                "importance": matched_node.importance,
                "source_sections": matched_node.source_sections,
            },
            "neighbors": neighbors,
            "tool": "query_knowledge_graph",
        }
    except Exception as e:
        logger.error(f"query_knowledge_graph failed: {e}")
        return {"error": str(e), "tool": "query_knowledge_graph"}


def tool_search_keyword(doc_id: str, keyword: str, top_k: int = 10) -> dict:
    """纯 BM25 关键词检索"""
    if not keyword or len(keyword.strip()) < MIN_HEADING_LEN:
        return {"error": "关键词至少需要 2 个字符", "tool": "search_keyword"}
    keyword = keyword.strip()
    top_k = max(1, min(top_k, MAX_TOP_K))
    try:
        bm25_idx = milvus_client._bm25_indices.get(doc_id)
        if bm25_idx is None:
            return {"error": "文档 BM25 索引不存在", "tool": "search_keyword"}
        import jieba
        tokens = jieba.lcut(keyword)
        scores = bm25_idx.get_scores(tokens)
        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        corpus = milvus_client._bm25_corpora.get(doc_id, [])
        hits = []
        for idx, score in indexed[:top_k]:
            if score <= 0:
                break
            hits.append({
                "text": " ".join(corpus[idx]) if idx < len(corpus) else "",
                "heading": "",
                "page": 0,
                "score": float(score),
            })
        return {"hits": hits, "total": len(hits), "tool": "search_keyword"}
    except Exception as e:
        logger.error(f"search_keyword failed: {e}")
        return {"error": str(e), "tool": "search_keyword"}


def tool_search_web(query: str, max_results: int = 5) -> dict:
    """联网搜索，在文档知识不足时补充外部信息"""
    if not query or not query.strip():
        return {"error": "搜索关键词不能为空", "tool": "search_web"}
    if not settings.web_search_enabled:
        return {"error": "Web 搜索功能已被管理员禁用", "tool": "search_web"}
    max_results = max(1, min(max_results, settings.web_search_max_results))
    try:
        from ddgs import DDGS
        results = DDGS().text(query.strip(), max_results=max_results)
        hits = []
        for r in results:
            hits.append({
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "snippet": r.get("body", ""),
            })
        return {"hits": hits, "total": len(hits), "tool": "search_web"}
    except Exception as e:
        logger.error(f"search_web failed: {e}")
        return {"error": str(e), "tool": "search_web"}


def tool_reflect_on_answer(draft_answer: str, cited_sources: list[str]) -> dict:
    """自检答案质量：检查是否基于文档、有无编造、引用是否准确"""
    if not draft_answer or not draft_answer.strip():
        return {"passed": False, "issues": ["答案为空"], "tool": "reflect_on_answer"}
    if not cited_sources:
        return {"passed": False, "issues": ["答案未提供任何引用来源，可能存在编造"],
                "tool": "reflect_on_answer"}
    issues = []
    if len(draft_answer.strip()) < 10:
        issues.append("答案过短，可能不够完整")
    if len(cited_sources) < 1:
        issues.append("引用来源不足")
    return {
        "passed": len(issues) == 0,
        "issues": issues,
        "source_count": len(cited_sources),
        "tool": "reflect_on_answer",
    }


def format_tool_result(tool_name: str, result: dict) -> str:
    """将工具返回的结构化结果格式化为人类可读的文本，注入到 LLM 消息中"""
    if "error" in result:
        return f"[工具 {tool_name} 执行失败] {result['error']}"

    if tool_name == "search_document":
        total = result.get("total", 0)
        if total == 0:
            return "[search_document] 未找到匹配的文档片段。请尝试更换查询词或使用其他工具。"
        lines = [f"[search_document] 找到 {total} 个相关片段:"]
        for i, hit in enumerate(result.get("hits", []), 1):
            heading = hit.get("heading", "")
            page = hit.get("page", 0)
            loc = f" [{heading}]" if heading else ""
            if page:
                loc += f" (第{page}页)"
            lines.append(f"{i}.{loc}\n   {hit.get('text', '')[:300]}")
        return "\n".join(lines)

    elif tool_name == "get_section":
        total = result.get("total", 0)
        if total == 0:
            avail = result.get("available_headings", [])
            hint = f" 可用的章节有: {', '.join(avail)}" if avail else ""
            return f"[get_section] 未找到匹配章节。{hint}"
        lines = [f"[get_section] 找到 {total} 个匹配章节:"]
        for hit in result.get("hits", []):
            lines.append(f"## {hit['heading']} (第{hit.get('page', '?')}页)\n{hit.get('content', '')[:2000]}")
        return "\n\n".join(lines)

    elif tool_name == "query_knowledge_graph":
        if not result.get("available", True):
            return f"[query_knowledge_graph] {result.get('message', '知识图谱不可用')}"
        if not result.get("found"):
            similar = result.get("similar", [])
            hint = f" 相关概念: {', '.join(similar)}" if similar else ""
            return f"[query_knowledge_graph] {result.get('message', '概念未找到')}{hint}"
        c = result["concept"]
        lines = [
            f"[query_knowledge_graph] 概念: {c['name']} (类型: {c['type']}, 重要度: {c['importance']})",
            f"概述: {c['summary']}",
            f"来源章节: {', '.join(c.get('source_sections', []))}",
        ]
        if result.get("neighbors"):
            lines.append("关联概念:")
            for n in result["neighbors"][:10]:
                lines.append(f"  - {n['concept']} ({n['relation']}): {n.get('description', '')}")
        return "\n".join(lines)

    elif tool_name == "search_keyword":
        total = result.get("total", 0)
        if total == 0:
            return "[search_keyword] 未找到匹配的关键词。"
        lines = [f"[search_keyword] 找到 {total} 个结果:"]
        for i, hit in enumerate(result.get("hits", []), 1):
            lines.append(f"{i}. [{hit.get('heading', '')}] {hit.get('text', '')[:200]}")
        return "\n".join(lines)

    elif tool_name == "search_web":
        total = result.get("total", 0)
        if total == 0:
            return "[search_web] 未找到相关网页结果。请尝试更换搜索词。"
        lines = [f"[search_web] 找到 {total} 个网页结果:"]
        for i, hit in enumerate(result.get("hits", []), 1):
            lines.append(
                f"{i}. **{hit.get('title', '无标题')}**\n"
                f"   URL: {hit.get('url', '')}\n"
                f"   {hit.get('snippet', '')[:300]}"
            )
        return "\n".join(lines)

    elif tool_name == "reflect_on_answer":
        if result.get("passed"):
            return "[reflect_on_answer] 自检通过：答案基于文档，引用来源充足。"
        issues = result.get("issues", [])
        return f"[reflect_on_answer] 自检未通过: {'; '.join(issues)}。请继续检索补充信息或修正答案。"

    elif tool_name == "ask_user":
        return f"[ask_user] {result.get('question', '')}"

    return f"[{tool_name}] 执行完成"


# 工具调度表
TOOL_MAP = {
    "search_document": tool_search_document,
    "get_section": tool_get_section,
    "query_knowledge_graph": tool_query_knowledge_graph,
    "search_keyword": tool_search_keyword,
    "search_web": tool_search_web,
    "reflect_on_answer": tool_reflect_on_answer,
}


def execute_tool(doc_id: str, tool_name: str, arguments: dict) -> dict:
    """统一工具调度入口，捕获异常并返回结构化错误"""
    if tool_name == "ask_user":
        return arguments  # ask_user 由 Agent 层直接处理, 不需要执行
    func = TOOL_MAP.get(tool_name)
    if func is None:
        return {"error": f"未知工具: {tool_name}", "tool": tool_name}
    try:
        if tool_name == "reflect_on_answer":
            return func(
                draft_answer=arguments.get("draft_answer", ""),
                cited_sources=arguments.get("cited_sources", []),
            )
        if tool_name == "search_web":
            return func(
                query=arguments.get("query", ""),
                max_results=arguments.get("max_results", 5),
            )
        return func(doc_id=doc_id, **arguments)
    except TypeError as e:
        return {"error": f"参数错误: {e}", "tool": tool_name}
    except Exception as e:
        logger.error(f"Tool {tool_name} execution failed: {e}")
        return {"error": str(e), "tool": tool_name}
