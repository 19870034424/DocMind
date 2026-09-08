"""查询改写 — 多轮对话中把带指代的问题改写为自包含检索查询

例如: 历史"Transformer 的注意力机制是什么?" + 新问题"它的复杂度呢?"
→ "Transformer 的 scaled dot-product attention 的计算复杂度是什么?"

改写失败/超时一律回退原问题, 不阻塞检索。
"""
import logging

from config import settings

logger = logging.getLogger(__name__)

_REWRITE_SYSTEM = """你是检索查询改写器。根据对话历史把用户的最新问题改写为一个**自包含**的检索查询。

规则:
1. 消解代词和省略("它""这个方法"等), 补全为历史中指代的具体对象
2. 保留用户问题的原始语言(中文问题输出中文, 英文输出英文)
3. 只输出改写后的查询文本本身, 不要任何解释、引号或前缀
4. 如果问题已经自包含, 原样输出"""

_REWRITE_USER = """对话历史:
{history}

最新问题: {question}

改写后的检索查询:"""


def rewrite_query(question: str, history: list[dict] | None) -> str:
    """结合对话历史改写查询; 无历史或改写失败时返回原问题"""
    if not history:
        return question
    try:
        from services import llm_client

        recent = history[-4:]  # 最近两轮
        history_text = "\n".join(
            f"{'用户' if m.get('role') == 'user' else '助手'}: {m.get('content', '')[:300]}"
            for m in recent
        )
        resp = llm_client.chat_completion(
            [
                {"role": "system", "content": _REWRITE_SYSTEM},
                {"role": "user", "content": _REWRITE_USER.format(
                    history=history_text, question=question)},
            ],
            temperature=0,
            max_tokens=200,
            timeout=settings.llm_rewrite_timeout_seconds,
            caller="query_rewrite",
        )
        rewritten = (resp.choices[0].message.content or "").strip().strip('"').strip("`")
        if rewritten and len(rewritten) <= 500:
            return rewritten
        return question
    except Exception:
        logger.exception("query rewrite failed, 使用原始问题")
        return question
