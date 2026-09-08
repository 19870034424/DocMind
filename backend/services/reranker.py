"""CrossEncoder 精排(重排)— 对混合检索召回的候选块按"问题-块"相关性重新排序

模型: BAAI/bge-reranker-v2-m3 (多语言, 与 bge-m3 配套)
懒加载单例, 首次调用才下载/加载模型; GPU 可用时自动使用。
"""
import threading

from config import settings

_lock = threading.Lock()
_model = None


def _get_model():
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                from sentence_transformers import CrossEncoder
                _model = CrossEncoder(
                    settings.rerank_model,
                    max_length=1024,
                )
    return _model


def rerank(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    """按相关性重排候选, 返回前 top_k 个。candidates: [{text, metadata}]"""
    if not candidates:
        return []
    model = _get_model()
    pairs = [(query, c["text"]) for c in candidates]
    scores = model.predict(pairs, show_progress_bar=False)
    for c, s in zip(candidates, scores, strict=False):
        c["rerank_score"] = float(s)
    ranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
    return ranked[:top_k]


def enabled() -> bool:
    return settings.rerank_enabled
