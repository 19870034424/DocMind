"""Chat 路由 — SSE 流式快速问答 + 深度 Agent 问答"""
import asyncio
import json
import logging
import re
from datetime import datetime

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from models.agent import ClarifyRequest, AgentSession
from services.agent import AgentLoop
from services.chunker import chunker
from storage.db import get_document, get_chat_history, add_chat_message

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
    now = datetime.now()
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


@router.post("/stream")
async def chat_stream(doc_id: str, body: ChatRequest, request: Request):
    """SSE 流式快速问答（单次检索 + 流式输出）"""
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question is required")
    if len(question) > 2000:
        raise HTTPException(status_code=400, detail="question too long")

    _get_doc_check(doc_id)

    async def generate():
        try:
            query_embedding = chunker.embed([question])[0]
            history = get_chat_history(doc_id)

            from storage.milvus_client import milvus_client
            chunks = milvus_client.hybrid_query(doc_id, query_embedding, question, top_k=15)

            if not chunks:
                yield f"data: {json.dumps({'event': 'chunk', 'data': {'text': '无法检索到相关文档内容。'}})}\n\n"
                yield f"data: {json.dumps({'event': 'done', 'data': {}})}\n\n"
                return

            from services.llm_service import _CONTENT_GUARD_PREFIX, _CONTENT_GUARD_SUFFIX

            parts = []
            total_len = 0
            for chunk in chunks:
                meta = chunk.get("metadata", {})
                heading = meta.get("heading_path", "")
                header = f"[{heading}]" if heading else ""
                block = f"{header}\n{chunk['text']}"
                if total_len + len(block) // 1.5 > 6000:
                    break
                parts.append(block)
                total_len += len(block) // 1.5

            context = f"{_CONTENT_GUARD_PREFIX}\n{_sanitize_content('\n\n---\n\n'.join(parts))}\n{_CONTENT_GUARD_SUFFIX}"

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

            from openai import OpenAI
            from config import settings

            if settings.llm_provider == "qwen":
                client = OpenAI(api_key=settings.qwen_api_key, base_url=settings.qwen_base_url)
                model = settings.qwen_model
            else:
                client = OpenAI(api_key=settings.deepseek_api_key, base_url=settings.deepseek_base_url)
                model = settings.deepseek_model

            stream = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": QA_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                stream=True,
                temperature=0.3,
                max_tokens=2048,
            )

            full_answer = ""
            for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.content:
                    text = delta.content
                    text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL)
                    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
                    if text:
                        full_answer += text
                        yield f"data: {json.dumps({'event': 'chunk', 'data': {'text': text}})}\n\n"

            add_chat_message(doc_id, "user", question)
            add_chat_message(doc_id, "assistant", full_answer)

            yield f"data: {json.dumps({'event': 'done', 'data': {}})}\n\n"

        except asyncio.CancelledError:
            logger.info("SSE connection cancelled by client")
        except Exception as e:
            logger.error(f"Chat stream error: {e}")
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


@router.post("/deep")
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


@router.post("/deep/resume")
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
