"""嵌入服务 — 文本 → 向量

支持两种模型路径:
- bge-m3 系列: transformers 原生加载, CLS 池化 + L2 归一化(官方 dense 用法)
- 其他 sentence-transformers 模型: SentenceTransformer.encode
模型懒加载(GPU 可用时自动使用), 全局单例。
"""
import threading

from config import settings

_lock = threading.Lock()
_state = {"st_model": None, "m3_tokenizer": None, "m3_model": None}


def _get_st_model():
    if _state["st_model"] is None:
        with _lock:
            if _state["st_model"] is None:
                from sentence_transformers import SentenceTransformer
                _state["st_model"] = SentenceTransformer(
                    settings.embedding_model, local_files_only=True
                )
    return _state["st_model"]


def _get_m3():
    if _state["m3_model"] is None:
        with _lock:
            if _state["m3_model"] is None:
                from transformers import AutoModel, AutoTokenizer
                _state["m3_tokenizer"] = AutoTokenizer.from_pretrained(
                    settings.embedding_model, local_files_only=True
                )
                model = AutoModel.from_pretrained(
                    settings.embedding_model, local_files_only=True
                )
                model.eval()
                _state["m3_model"] = model
    return _state["m3_tokenizer"], _state["m3_model"]


def _is_m3() -> bool:
    return "m3" in settings.embedding_model.lower()


def embed(texts: list[str]) -> list[list[float]]:
    """批量嵌入, 返回归一化向量列表"""
    if not texts:
        return []
    if _is_m3():
        return _embed_m3(texts)
    result = _get_st_model().encode(texts, normalize_embeddings=True)
    return result.tolist()


def _embed_m3(texts: list[str]) -> list[list[float]]:
    """bge-m3 官方 dense 用法: CLS 池化 + L2 归一化。
    不走 sentence-transformers(其对 m3 的 AutoProcessor 处理在离线环境会失败)"""
    import torch

    tokenizer, model = _get_m3()
    embs: list[list[float]] = []
    batch_size = 16
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tokenizer(
                batch, padding=True, truncation=True, max_length=1024,
                return_tensors="pt",
            )
            out = model(**enc)
            cls = out.last_hidden_state[:, 0]
            cls = torch.nn.functional.normalize(cls, p=2, dim=-1)
            embs.extend(cls.tolist())
    return embs
