"""Chat 路由 — SSE 流式快速问答 + 深度 Agent 问答"""
import asyncio
import json
import logging
import re
import time

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from config import settings
from models.agent import AgentSession, ClarifyRequest
from services.agent import AgentLoop
from services.prompt_templates import QA_SYSTEM_PROMPT, QA_USER_PROMPT
from storage.db import add_chat_message, get_chat_history, get_document

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/documents/{doc_id}/chat", tags=["chat"])

_agent_sessions: dict[str, AgentSession] = {}
SESSION_TIMEOUT_MINUTES = 30


class ChatRequest(BaseModel):
    question: str


class DeepChatRequest(BaseModel):
    question: str
    max_iterations: int = 8


def _cleanup_expired_sessions():
    expired = [
        sid for sid, s in _agent_sessions.items()
        if s.is_expired(SESSION_TIMEOUT_MINUTES)
    ]
    for sid in expired:
        del _agent_sessions[sid]


def _get_doc_check(doc_id: str) -> dict:
    row = get_document(doc_id)
    if not row:
        raise HTTPException(status_code=404, detail="文档不存在")
    if row["status"] != "ready":
        raise HTTPException(status_code=400, detail="文档尚未就绪")
    return row


def _sanitize_content(text: str) -> str:
    text = re.sub(
        r'(?i)(忽略|无视|忘记|ignore)\s*(上述|以上|所有|all|previous)\s*(指令|指示|说明|规则|instructions?)',
        '[过滤]', text,
    )
    return text


@router.post("/stream", summary="快速模式问答(SSE 流式,单轮 RAG)")
async def chat_stream(doc_id: str, body: ChatRequest, request: Request):
    """SSE 流式快速问答（单次检索 + 流式输出）"""
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question is required")
    if len(question) > 2000:
        raise HTTPException(status_code=400, detail="question too long")

    _get_doc_check(doc_id)

    async def generate():
        t_total = time.perf_counter()
        try:
            history = get_chat_history(doc_id)

            from services import llm_client
            from services.metrics import log_event
            from services.query_rewriter import rewrite_query
            from services.retrieval import build_context, hybrid_search

            t0 = time.perf_counter()
            search_query = rewrite_query(question, history)
            rewrite_ms = int((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            chunks = hybrid_search(doc_id, search_query,
                                   top_k=settings.retrieval_top_k_qa)
            retrieval_ms = int((time.perf_counter() - t0) * 1000)

            if not chunks:
                yield f"data: {json.dumps({'event': 'chunk', 'data': {'text': '无法检索到相关文档内容。'}})}\n\n"
                yield f"data: {json.dumps({'event': 'done', 'data': {}})}\n\n"
                log_event("qa_request", doc_id=doc_id, ok=False,
                          reason="no_chunks", rewrite_ms=rewrite_ms, retrieval_ms=retrieval_ms)
                return

            context, used = build_context(
                chunks, max_tokens=settings.context_max_tokens_qa, numbered=True,
            )
            context = _sanitize_content(context)

            # 引用片段信息: 供前端右栏展示(编号与上下文中的 片段N 对应)
            fragments = [
                {
                    "n": i + 1,
                    "heading_path": c.get("metadata", {}).get("heading_path", ""),
                    "block_type": c.get("metadata", {}).get("block_type", "text"),
                    "text": c.get("text", "")[:600],
                }
                for i, c in enumerate(chunks[:used])
            ]
            yield f"data: {json.dumps({'event': 'sources', 'data': {'fragments': fragments}})}\n\n"

            history_text = ""
            if history:
                lines = ["## 对话历史"]
                for turn in history[-6:]:
                    role = "用户" if turn["role"] == "user" else "助手"
                    lines.append(f"{role}：{turn['content']}")
                history_text = "\n".join(lines)

            user_prompt = QA_USER_PROMPT.format(
                history=history_text, content=context, question=question,
            )

            from openai import OpenAI  # noqa: F401  (类型兼容保留)

            stream = llm_client.stream_completion(
                [
                    {"role": "system", "content": QA_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=2048,
                caller="qa_stream",
            )

            full_answer = ""
            first_token_ms = None
            usage = None
            for chunk in stream:
                if getattr(chunk, "usage", None):
                    usage = chunk.usage
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.content:
                    if first_token_ms is None:
                        first_token_ms = int((time.perf_counter() - t_total) * 1000)
                    text = delta.content
                    text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL)
                    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
                    if text:
                        full_answer += text
                        yield f"data: {json.dumps({'event': 'chunk', 'data': {'text': text}})}\n\n"

            llm_client.record_usage(usage)
            add_chat_message(doc_id, "user", question)
            add_chat_message(doc_id, "assistant", full_answer)

            log_event(
                "qa_request", doc_id=doc_id, ok=True,
                query=search_query, rewritten=search_query != question,
                chunks=len(chunks), used=used,
                rewrite_ms=rewrite_ms, retrieval_ms=retrieval_ms,
                first_token_ms=first_token_ms,
                total_ms=int((time.perf_counter() - t_total) * 1000),
                answer_chars=len(full_answer),
            )

            yield f"data: {json.dumps({'event': 'done', 'data': {}})}\n\n"

        except asyncio.CancelledError:
            logger.info("SSE connection cancelled by client")
        except Exception as e:
            logger.error(f"Chat stream error: {e}")
            log_event("qa_request", doc_id=doc_id, ok=False, reason=type(e).__name__,
                      total_ms=int((time.perf_counter() - t_total) * 1000))
            yield f"data: {json.dumps({'event': 'done', 'data': {'error': str(e)}})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/deep", summary="深度模式问答(SSE 流式,ReAct Agent 多步推理)")
async def chat_deep(doc_id: str, body: DeepChatRequest, request: Request):
    """SSE 流式深度 Agent 问答（ReAct 循环）"""
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question is required")
    if len(question) > 2000:
        raise HTTPException(status_code=400, detail="question too long")

    _get_doc_check(doc_id)
    _cleanup_expired_sessions()

    session = AgentSession(doc_id=doc_id, question=question)
    _agent_sessions[session.session_id] = session

    async def generate():
        try:
            loop = AgentLoop(
                doc_id=doc_id,
                question=question,
                max_iterations=body.max_iterations,
            )
            async for event in loop.run():
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                if event["event"] == "step":
                    session.messages = loop.messages
            # Attach session_id in done event
            yield f"data: {json.dumps({'event': 'session', 'data': {'session_id': session.session_id}})}\n\n"

        except asyncio.CancelledError:
            logger.info("Deep chat SSE cancelled by client")
        except Exception as e:
            logger.error(f"Deep chat error: {e}")
            yield f"data: {json.dumps({'event': 'done', 'data': {'error': str(e)}})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/deep/resume", summary="继续被暂停的 Agent 会话(补充澄清信息后)")
async def chat_deep_resume(doc_id: str, body: ClarifyRequest, request: Request):
    """恢复被追问暂停的 Agent 循环"""
    _cleanup_expired_sessions()
    session = _agent_sessions.get(body.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="会话不存在或已过期")

    async def generate():
        try:
            loop = AgentLoop(
                doc_id=session.doc_id,
                question=session.question,
            )
            loop.messages = session.messages
            loop.resume(body.user_response)

            async for event in loop.run():
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                if event["event"] == "step":
                    session.messages = loop.messages

        except asyncio.CancelledError:
            logger.info("Resume SSE cancelled by client")
        except Exception as e:
            logger.error(f"Resume error: {e}")
            yield f"data: {json.dumps({'event': 'done', 'data': {'error': str(e)}})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
