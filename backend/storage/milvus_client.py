"""Milvus 向量存储

每个文档一个 Collection（doc_{doc_id}），字段: id, text, embedding(1024d),
heading_path, section_heading, page, chunk_index, block_type
BM25 索引: 语料持久化到 data/bm25/{doc_id}.pkl, 启动后惰性重建,
避免重启后静默退化为纯向量检索。
"""
import contextlib
import pickle
from pathlib import Path

import jieba
from pymilvus import (
    CollectionSchema,
    DataType,
    FieldSchema,
)
from pymilvus import (
    MilvusClient as MC,
)
from pymilvus.milvus_client import IndexParams
from rank_bm25 import BM25Okapi

from config import settings

MILVUS_HOST = getattr(settings, "milvus_host", "localhost")
MILVUS_PORT = getattr(settings, "milvus_port", 19530)
EMBEDDING_DIM = 1024  # BGE-m3
BM25_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "bm25"
BM25_DIR.mkdir(parents=True, exist_ok=True)

# 每个 Collection 的 Schema 字段
FIELDS = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=4096),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
    FieldSchema(name="heading_path", dtype=DataType.VARCHAR, max_length=512),
    FieldSchema(name="section_heading", dtype=DataType.VARCHAR, max_length=256),
    FieldSchema(name="page", dtype=DataType.INT64),
    FieldSchema(name="chunk_index", dtype=DataType.INT64),
    FieldSchema(name="block_type", dtype=DataType.VARCHAR, max_length=16, default_value="text"),
]
SCHEMA = CollectionSchema(fields=FIELDS, description="Document chunk")

# 向量索引参数
INDEX_PARAMS = {
    "metric_type": "IP",  # inner product for normalized embeddings
    "index_type": "IVF_FLAT",
    "params": {"nlist": 128},
}


class MilvusStore:
    def __init__(self, host: str = MILVUS_HOST, port: int = MILVUS_PORT):
        self._host = host
        self._port = port
        self._bm25_indices: dict[str, BM25Okapi] = {}
        self._bm25_corpora: dict[str, list[list[str]]] = {}
        self._client = MC(uri=f"http://{host}:{port}")

    def _col_name(self, doc_id: str) -> str:
        """Milvus collection name: doc_{doc_id}"""
        return f"doc_{doc_id}"

    def create_collection(self, doc_id: str) -> str:
        """创建新 Collection，返回名称"""
        name = self._col_name(doc_id)
        # 删除已存在的同名 Collection
        with contextlib.suppress(Exception):
            self._client.drop_collection(name)
        index_params = IndexParams()
        index_params.add_index(
            field_name="embedding",
            metric_type=INDEX_PARAMS["metric_type"],
            index_type=INDEX_PARAMS["index_type"],
            params=INDEX_PARAMS["params"],
        )
        self._client.create_collection(
            collection_name=name,
            schema=SCHEMA,
            index_params=index_params,
        )
        self._client.load_collection(name)
        return name

    def get_collection(self, doc_id: str) -> bool:
        """检查 Collection 是否存在"""
        name = self._col_name(doc_id)
        return self._client.has_collection(name)

    def delete_collection(self, doc_id: str):
        """删除 Collection"""
        name = self._col_name(doc_id)
        try:
            self._client.release_collection(name)
            self._client.drop_collection(name)
        except Exception:
            pass
        self._bm25_indices.pop(doc_id, None)
        self._bm25_corpora.pop(doc_id, None)
        (BM25_DIR / f"{doc_id}.pkl").unlink(missing_ok=True)

    def add_chunks(
        self,
        collection_name: str,
        chunks: list[dict],
        embeddings: list[list[float]],
        doc_id: str = "",
    ):
        """批量插入 chunk 向量到 Milvus，同时构建 BM25 索引"""
        data = []
        for i, chunk in enumerate(chunks):
            meta = chunk.get("metadata", {})
            data.append({
                "text": chunk["text"],
                "embedding": embeddings[i],
                "heading_path": meta.get("heading_path", ""),
                "section_heading": meta.get("section_heading", ""),
                "page": meta.get("page", 0),
                "chunk_index": meta.get("chunk_index", i),
                "block_type": meta.get("block_type", "text"),
            })

        self._client.insert(collection_name=collection_name, data=data)

        # Build BM25 index
        if doc_id:
            corpus = [jieba.lcut(c["text"]) for c in chunks]
            self._bm25_corpora[doc_id] = corpus
            self._bm25_indices[doc_id] = BM25Okapi(corpus)
            # 持久化词料, 重启后可重建(索引对象本身不 pickle, 避免版本兼容问题)
            try:
                with open(BM25_DIR / f"{doc_id}.pkl", "wb") as f:
                    pickle.dump(corpus, f)
            except OSError:
                pass

    def _get_bm25(self, doc_id: str) -> BM25Okapi | None:
        """取 BM25 索引; 内存没有则尝试从磁盘恢复(惰性重建)"""
        idx = self._bm25_indices.get(doc_id)
        if idx is not None:
            return idx
        pkl = BM25_DIR / f"{doc_id}.pkl"
        if not pkl.exists():
            return None
        try:
            with open(pkl, "rb") as f:
                corpus = pickle.load(f)
            self._bm25_corpora[doc_id] = corpus
            self._bm25_indices[doc_id] = BM25Okapi(corpus)
            return self._bm25_indices[doc_id]
        except Exception:
            return None

    def bm25_scores(self, doc_id: str, query_text: str) -> list[float] | None:
        """BM25 对全部 chunk 打分(集合无 BM25 索引时返回 None)"""
        idx = self._get_bm25(doc_id)
        if idx is None:
            return None
        return idx.get_scores(jieba.lcut(query_text)).tolist()

    def query(
        self,
        doc_id: str,
        query_embedding: list[float],
        top_k: int = 20,
    ) -> list[dict]:
        """纯向量检索"""
        name = self._col_name(doc_id)
        if not self._client.has_collection(name):
            return []

        results = self._client.search(
            collection_name=name,
            data=[query_embedding],
            anns_field="embedding",
            limit=top_k,
            search_params={"params": {"nprobe": settings.retrieval_nprobe}},
            output_fields=["text", "heading_path", "section_heading", "page", "chunk_index", "block_type"],
        )
        if not results or not results[0]:
            return []

        chunks = []
        for hit in results[0]:
            entity = hit.get("entity", {})
            chunks.append({
                "text": entity.get("text", ""),
                "metadata": {
                    "heading_path": entity.get("heading_path", ""),
                    "section_heading": entity.get("section_heading", ""),
                    "page": entity.get("page", 0),
                    "chunk_index": entity.get("chunk_index", 0),
                    "block_type": entity.get("block_type", "text"),
                },
            })
        return chunks

    def hybrid_query(
        self,
        doc_id: str,
        query_embedding: list[float],
        query_text: str,
        top_k: int = 20,
    ) -> list[dict]:
        """兼容入口: 检索策略(双路召回→RRF→精排)已迁移到 services.retrieval.hybrid_search"""
        from services.retrieval import hybrid_search
        return hybrid_search(doc_id, query_text, top_k, rerank=True)


# 模块级单例
milvus_client = MilvusStore()
