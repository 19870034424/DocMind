import math
import re
from difflib import SequenceMatcher


def _norm(t: str) -> str:
    return re.sub(r"\s+", "", t)


def _is_relevant(chunk_text: str, relevant: set[str]) -> bool:
    t = _norm(chunk_text)
    for r in relevant:
        r = _norm(r)
        if not r:
            continue
        if t == r:
            return True
        if len(r) >= 20 and (r in t or t in r):
            return True
    # 兜底: 切分策略变化后块边界不同, 用相似度匹配同一内容
    for r in relevant:
        r = _norm(r)
        if len(r) >= 30:
            ratio = SequenceMatcher(None, t[:600], r[:600]).ratio()
            if ratio >= 0.55:
                return True
    return False


def recall_at_k(
    retrieved_texts: list[str], relevant_texts: set[str], k: int
) -> float:
    if not relevant_texts:
        return 0.0
    hits = sum(1 for t in retrieved_texts[:k] if _is_relevant(t, relevant_texts))
    return hits / len(relevant_texts)


def precision_at_k(
    retrieved_texts: list[str], relevant_texts: set[str], k: int
) -> float:
    if k <= 0:
        return 0.0
    hits = sum(1 for t in retrieved_texts[:k] if _is_relevant(t, relevant_texts))
    return hits / k


def reciprocal_rank(
    retrieved_texts: list[str], relevant_texts: set[str]
) -> float:
    for i, text in enumerate(retrieved_texts):
        if _is_relevant(text, relevant_texts):
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(
    retrieved_texts: list[str], relevant_texts: set[str], k: int
) -> float:
    dcg = 0.0
    for i, text in enumerate(retrieved_texts[:k]):
        if _is_relevant(text, relevant_texts):
            dcg += 1.0 / math.log2(i + 2)

    ideal_hits = min(len(relevant_texts), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))

    return dcg / idcg if idcg > 0 else 0.0


def average_precision_at_k(
    retrieved_texts: list[str], relevant_texts: set[str], k: int
) -> float:
    if not relevant_texts:
        return 0.0
    hits = 0
    sum_prec = 0.0
    for i, text in enumerate(retrieved_texts[:k]):
        if _is_relevant(text, relevant_texts):
            hits += 1
            sum_prec += hits / (i + 1)
    return sum_prec / len(relevant_texts)


def compute_query_metrics(
    retrieved_texts: list[str],
    relevant_texts: set[str],
    k_values: list[int],
) -> dict:
    return {
        **{f"recall@{k}": recall_at_k(retrieved_texts, relevant_texts, k)
           for k in k_values},
        **{f"precision@{k}": precision_at_k(retrieved_texts, relevant_texts, k)
           for k in k_values},
        "mrr": reciprocal_rank(retrieved_texts, relevant_texts),
        **{f"ndcg@{k}": ndcg_at_k(retrieved_texts, relevant_texts, k)
           for k in k_values},
        **{f"map@{k}": average_precision_at_k(retrieved_texts, relevant_texts, k)
           for k in k_values},
    }
