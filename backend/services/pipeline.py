"""文档处理管线 — 分阶段执行

阶段:
  parse  — MinerU 解析, 结构化结果持久化到 data/parsed/{doc_id}.json
  chunk  — 递归分块
  embed  — BGE 嵌入
  index  — 写入 Milvus + BM25 索引

每个阶段是独立纯函数, 可被 pytest 单独测试;
process_document 串联全部阶段并维护文档状态与错误记录。
"""
import contextlib
from pathlib import Path

from models.document import DocumentStatus, StructuredDocument
from services.chunker import chunker
from services.parser import document_parser
from storage.db import update_document
from storage.milvus_client import milvus_client

PARSED_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "parsed"
PARSED_DIR.mkdir(parents=True, exist_ok=True)


def parse_stage(file_path: Path, filename: str, doc_id: str = "") -> StructuredDocument:
    """阶段1: 解析文档 → StructuredDocument (doc_id 用于图片资产归档)"""
    return document_parser.parse(file_path, filename, doc_id=doc_id)


def load_parsed_cache(doc_id: str) -> StructuredDocument | None:
    """读取已持久化的解析结果(供重跑时跳过最慢的解析阶段), 失败返回 None"""
    p = PARSED_DIR / f"{doc_id}.json"
    if not p.exists():
        return None
    try:
        return StructuredDocument.model_validate_json(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def chunk_stage(structured: StructuredDocument) -> list[dict]:
    """阶段2: 分块"""
    return chunker.chunk(structured)


def embed_stage(chunks: list[dict]) -> list[list[float]]:
    """阶段3: 嵌入"""
    return chunker.embed([c["text"] for c in chunks])


def index_stage(doc_id: str, chunks: list[dict], embeddings: list[list[float]]) -> None:
    """阶段4: 写入 Milvus(先删旧集合保证幂等)"""
    with contextlib.suppress(Exception):
        milvus_client.delete_collection(doc_id)
    collection_name = milvus_client.create_collection(doc_id)
    milvus_client.add_chunks(collection_name, chunks, embeddings, doc_id=doc_id)


def process_document(doc_id: str, file_path: Path, filename: str) -> None:
    """完整管线: 在后台任务中运行, 逐步更新状态; 失败记录 error_stage/error_msg"""
    global _stage_now
    _set_running(doc_id)
    try:
        # 解析: 有持久化缓存(上次失败遗留)则跳过最慢的解析
        structured = load_parsed_cache(doc_id)
        if structured is None:
            _stage_now = "parse"
            structured = parse_stage(file_path, filename, doc_id)
            (PARSED_DIR / f"{doc_id}.json").write_text(
                structured.model_dump_json(), encoding="utf-8"
            )
        update_document(
            doc_id,
            status=DocumentStatus.PARSING,
            title=structured.title,
            pages=structured.metadata.pages,
            structured=structured,
        )

        update_document(doc_id, status=DocumentStatus.CHUNKING)
        _stage_now = "chunk"
        chunks = chunk_stage(structured)
        if not chunks:
            raise ValueError("文档未提取到任何文本内容")
        update_document(doc_id, chunks=chunks)

        _stage_now = "embed"
        embeddings = embed_stage(chunks)
        if not embeddings:
            raise ValueError("嵌入生成结果为空")

        _stage_now = "index"
        index_stage(doc_id, chunks, embeddings)
        update_document(doc_id, status=DocumentStatus.READY)
    except Exception as e:
        update_document(
            doc_id,
            status=DocumentStatus.ERROR,
            error_stage=_stage_now,
            error_msg=str(e)[:1000],
        )


def _set_running(doc_id: str) -> None:
    global _stage_now
    _stage_now = "parse"
    update_document(doc_id, status=DocumentStatus.PARSING, error_stage="", error_msg="")
