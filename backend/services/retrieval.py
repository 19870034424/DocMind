"""检索流水线 — 双路召回 → RRF 融合 → CrossEncoder 精排 → 编号上下文

统一的检索入口, 供问答/报告/Agent 工具复用。
错误记录日志而不是静默吞掉(此前精排静默失效导致评估结果失真)。
"""
import logging

from config import settings
from services import embedder, reranker
from storage.milvus_client import milvus_client

logger = logging.getLogger(__name__)


def vector_search(doc_id: str, query: str, top_k: int) -> list[dict]:
    emb = embedder.embed([query])[0]
    return milvus_client.query(doc_id, emb, top_k=top_k)


def hybrid_search(doc_id: str, query: str, top_k: int, rerank: bool = True) -> list[dict]:
    """BM25 + 向量双路召回 → RRF 融合 → 可选精排, 返回 top_k 块"""
    pool_size = max(top_k * 2, top_k)
    emb = embedder.embed([query])[0]
    vec_results = milvus_client.query(doc_id, emb, top_k=pool_size)

    bm25_scores = milvus_client.bm25_scores(doc_id, query)
    if bm25_scores is None:
        logger.warning("doc %s 无 BM25 索引(未入库或语料缺失), 退化为纯向量检索", doc_id)
        fused = vec_results
    else:
        fused = rrf_fusion(vec_results, bm25_scores,
                           k=settings.retrieval_rrf_k, top_k=pool_size)

    if rerank and reranker.enabled() and len(fused) > top_k:
        try:
            return reranker.rerank(query, fused, top_k)
        except Exception:
            logger.exception("rerank 失败, 退回 RRF 融合排序")
    return fused[:top_k]


def rrf_fusion(
    vec_results: list[dict],
    bm25_scores: list[float],
    k: int = 60,
    top_k: int = 20,
) -> list[dict]:
    """Reciprocal Rank Fusion: 只用名次融合, 避免两路分数量纲不可比。

    BM25 分数按语料顺序排列, 通过 chunk 元数据里的 chunk_index 对齐;
    旧实现假设 Milvus 返回顺序==语料顺序, 导致 BM25 名次错位。
    """
    order = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)
    rank_bm25 = {ci: r for r, ci in enumerate(order)}

    scored: list[tuple[float, int, dict]] = []
    for rank, chunk in enumerate(vec_results):
        s = 1.0 / (k + rank + 1)  # 向量通道名次
        ci = chunk.get("metadata", {}).get("chunk_index")
        if ci is not None and ci in rank_bm25:
            s += 1.0 / (k + rank_bm25[ci] + 1)  # BM25 通道名次
        scored.append((s, rank, chunk))

    scored.sort(key=lambda t: (-t[0], t[1]))
    return [c for _, _, c in scored[:top_k]]


def build_context(
    chunks: list[dict],
    max_tokens: int = 6000,
    numbered: bool = True,
) -> tuple[str, int]:
    """把检索块拼装为带编号和章节头的上下文。

    返回 (上下文文本, 使用的块数)。每块格式:
        [片段N | 章节: heading_path]
        块正文
    中文按 1 字≈1 token、英文按 4 字符≈1 token 粗略估算。
    """
    def _est_len(s: str) -> int:
        cjk = sum(1 for ch in s if ord(ch) > 0x2E80)
        return cjk + (len(s) - cjk) // 4

    parts: list[str] = []
    total = 0
    used = 0
    for c in chunks:
        meta = c.get("metadata", {})
        n = used + 1
        head = f"[片段{n} | 章节: {meta.get('heading_path', '') or '正文'}]" if numbered else ""
        body = c.get("text", "")
        block = f"{head}\n{body}" if head else body
        cost = _est_len(block)
        if total + cost > max_tokens:
            break
        parts.append(block)
        total += cost
        used = n
    return "\n\n".join(parts), used
