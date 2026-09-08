import json
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from models.document import (
    DocumentStatus,
    ReportResult,
)
from services.chunker import chunker
from services.llm_service import llm_service
from services.parser import document_parser
from storage.db import (
    add_chat_message,
    append_report,
    clear_chat_history,
    delete_document,
    get_chat_history,
    get_document,
    get_report,
    insert_document,
    list_documents,
    save_report,
    update_document,
)
from storage.file_store import file_store
from storage.milvus_client import milvus_client

router = APIRouter(prefix="/api/documents", tags=["documents"])


def _build_doc_metadata(doc_id: str, filename: str, file_path: Path) -> None:

    fmt = document_parser._detect_format(filename)
    file_size = file_path.stat().st_size
    insert_document(doc_id, filename, fmt.value, file_size)


MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50MB


@router.post("/upload", summary="上传文档(立即返回, 后台异步执行解析→分块→嵌入→入库)")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(400, "Filename is required")
    if not file.content_type:
        raise HTTPException(400, "Content-Type is required")

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(400, "File is empty")
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(400, f"File exceeds {MAX_UPLOAD_SIZE // 1024 // 1024}MB limit")

    doc_id, file_path = await file_store.save(file.filename, content)
    _build_doc_metadata(doc_id, file.filename, file_path)

    # 入队: 由 worker(内嵌线程或独立进程) 认领执行, 带重试/超时/并发限流
    update_document(doc_id, status=DocumentStatus.QUEUED)
    from services.metrics import log_event
    log_event("pipeline_queued", doc_id=doc_id, filename=file.filename)

    return {"id": doc_id, "title": "", "status": DocumentStatus.QUEUED.value, "metadata": None}


@router.post("/{doc_id}/reprocess", summary="重新处理文档(重新入队, 失败自动重试; 已解析的结果会被复用)")
async def api_reprocess(doc_id: str):
    row = get_document(doc_id)
    if not row:
        raise HTTPException(404, "Document not found")
    file_path = file_store.get_path(doc_id, row["filename"])
    if not Path(file_path).exists():
        raise HTTPException(400, "Original file missing, please re-upload")
    update_document(doc_id, status=DocumentStatus.QUEUED, retry_count=0,
                    error_stage="", error_msg="")
    from services.metrics import log_event
    log_event("pipeline_queued", doc_id=doc_id, reason="reprocess")
    return {"id": doc_id, "status": DocumentStatus.QUEUED.value}


@router.get("/", summary="获取文档列表")
async def api_list_documents():
    docs = list_documents()
    return [
        {
            "id": r["id"],
            "title": r["title"],
            "status": r["status"],
            "created_at": r["created_at"],
            "format": r.get("format"),
            "pages": r.get("pages", 0),
        }
        for r in docs
    ]


@router.get("/{doc_id}", summary="获取单个文档详情")
async def api_get_document(doc_id: str):
    row = get_document(doc_id)
    if not row:
        raise HTTPException(404, "Document not found")
    import json
    metadata = json.loads(row.get("metadata_json", "{}")) if row.get("metadata_json") else None
    structured = json.loads(row["structured_json"]) if row.get("structured_json") else None
    return {
        "id": row["id"],
        "title": row["title"],
        "status": row["status"],
        "created_at": row["created_at"],
        "metadata": metadata,
        "structured": structured,
        "error_stage": row.get("error_stage", ""),
        "error_msg": row.get("error_msg", ""),
    }


@router.get("/{doc_id}/chunks", summary="查看文档分块结果(入库的原文片段, 支持分页)")
async def api_get_chunks(doc_id: str, offset: int = 0, limit: int = 20):
    row = get_document(doc_id)
    if not row:
        raise HTTPException(404, "Document not found")
    raw = row.get("chunks_json")
    if not raw:
        raise HTTPException(400, "Document has no chunks (status: " + row["status"] + ")")
    chunks = json.loads(raw)
    total = len(chunks)
    items = [
        {
            "index": c.get("metadata", {}).get("chunk_index", i),
            "text": c.get("text", ""),
            "chars": len(c.get("text", "")),
            "heading_path": c.get("metadata", {}).get("heading_path", ""),
            "section_heading": c.get("metadata", {}).get("section_heading", ""),
        }
        for i, c in enumerate(chunks)
    ][offset : offset + limit]
    return {"total": total, "offset": offset, "limit": limit, "items": items}


@router.delete("/{doc_id}", summary="删除文档及其关联数据")
async def api_delete_document(doc_id: str):
    row = get_document(doc_id)
    if not row:
        raise HTTPException(404, "Document not found")
    milvus_client.delete_collection(doc_id)
    if row.get("filename"):
        file_store.delete(doc_id, row["filename"])
    # 清理管线解析缓存
    parsed_cache = Path(__file__).resolve().parent.parent.parent / "data" / "parsed" / f"{doc_id}.json"
    parsed_cache.unlink(missing_ok=True)
    delete_document(doc_id)
    return {"deleted": doc_id}


@router.post("/{doc_id}/report", summary="生成四步阅读报告(评估→拆解→批判→读者指南)")
async def api_generate_report(doc_id: str):
    import json
    row = get_document(doc_id)
    if not row:
        raise HTTPException(404, "Document not found")
    if row["status"] != "ready":
        raise HTTPException(400, "Document not ready, current status: " + row["status"])

    structured = json.loads(row["structured_json"]) if row.get("structured_json") else None
    if not structured:
        raise HTTPException(400, "Document has no parsed content")

    title = row["title"]
    sections = structured.get("sections", []) if structured else []
    summary_text = title
    for section in sections[:3]:
        summary_text += " " + section.get("heading", "") + " " + section.get("content", "")[:200]
    query_embedding = chunker.embed([summary_text[:1000]])[0]

    try:
        content = llm_service.generate_report(
            doc_id=doc_id,
            title=title,
            fmt=row.get("format", "pdf"),
            pages=row.get("pages", 0),
            query_embedding=query_embedding,
        )
    except Exception as e:
        raise HTTPException(500, "Report generation failed") from e

    result = ReportResult(content=content)
    save_report(doc_id, content, result.generated_at.isoformat())
    return {"content": result.content, "generated_at": result.generated_at.isoformat()}


@router.get("/{doc_id}/report", summary="获取已生成的阅读报告")
async def api_get_report(doc_id: str):
    result = get_report(doc_id)
    if not result:
        raise HTTPException(404, "Report not generated yet")
    return {"content": result["content"], "generated_at": result["generated_at"]}


@router.post("/{doc_id}/report/deep-dive", summary="章节深度分析(定向深读选定章节)")
async def api_deep_dive_report(doc_id: str, body: dict):
    import json
    row = get_document(doc_id)
    if not row:
        raise HTTPException(404, "Document not found")
    sections = body.get("sections", [])
    if not sections:
        raise HTTPException(400, "sections is required")

    structured = json.loads(row["structured_json"]) if row.get("structured_json") else None
    sections_list = structured.get("sections", []) if structured else []

    summary_text = row["title"]
    for section in sections_list[:3]:
        summary_text += " " + section.get("heading", "")
    query_embedding = chunker.embed([summary_text[:1000]])[0]

    # Build section texts from stored structured data
    section_texts: list[str] = []
    sections_lower = [s.strip().lower() for s in sections]
    # Reconstruct sections for _build_heading_paths
    from models.document import Section as SecModel
    sec_objects = [
        SecModel(
            heading=s.get("heading", ""),
            level=s.get("level", 1),
            content=s.get("content", ""),
            page=s.get("page", 0),
            tables=s.get("tables", []),
        )
        for s in sections_list
    ]
    for sec in sec_objects:
        if sec.heading.lower() in sections_lower:
            section_texts.append(sec.heading + "\n" + sec.content)
    sections_content = "\n\n=====\n\n".join(section_texts) if section_texts else ""

    try:
        content = llm_service.generate_deep_dive(
            doc_id=doc_id,
            title=row["title"],
            selected_sections=sections,
            sections_content=sections_content,
            query_embedding=query_embedding,
        )
    except Exception as e:
        raise HTTPException(500, "Deep-dive analysis failed") from e

    append_report(doc_id, content)
    return {"content": content}




@router.post("/{doc_id}/chat", summary="普通 RAG 问答(非流式)")
async def api_ask_question(doc_id: str, body: dict):
    row = get_document(doc_id)
    if not row:
        raise HTTPException(404, "Document not found")
    if row["status"] != "ready":
        raise HTTPException(400, "Document not ready")

    question = body.get("question", "").strip()
    if not question:
        raise HTTPException(400, "question is required")
    if len(question) > 2000:
        raise HTTPException(400, "question too long")

    query_embedding = chunker.embed([question])[0]
    history = get_chat_history(doc_id)

    try:
        answer = llm_service.ask(
            doc_id=doc_id,
            question=question,
            history=history,
            query_embedding=query_embedding,
        )
    except Exception as e:
        raise HTTPException(500, "Chat failed") from e

    add_chat_message(doc_id, "user", question)
    add_chat_message(doc_id, "assistant", answer)
    history = get_chat_history(doc_id)

    return {"answer": answer, "history": history}


@router.get("/{doc_id}/chat", summary="获取对话历史")
async def api_get_chat_history(doc_id: str):
    return {"history": get_chat_history(doc_id)}


@router.delete("/{doc_id}/chat", summary="清空对话历史")
async def api_clear_chat_history(doc_id: str):
    clear_chat_history(doc_id)
    return {"cleared": doc_id}
