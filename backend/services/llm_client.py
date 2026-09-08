"""LLM 客户端统一入口 — 超时 / 自动重试 / 用量打点 / 每日 token 预算

所有 OpenAI 兼容调用(问答、报告、改写、KG 抽取)都应通过 get_llm() 拿客户端,
通过 chat_completion()/stream_completion() 发起请求, 由本模块统一容错:
- 超时: settings.llm_timeout_seconds
- 重试: OpenAI SDK max_retries(仅对超时/限流/连接错误生效, 已开始流式输出的不重试)
- 用量: 每次调用打 llm_usage 事件到 metrics
- 预算: settings.llm_daily_token_budget > 0 时, 超预算抛 LLMBudgetExceeded
"""
import json
import logging
import threading
from datetime import date
from pathlib import Path

import httpx
from openai import OpenAI

from config import settings
from services.metrics import log_event

logger = logging.getLogger(__name__)

_client: OpenAI | None = None

USAGE_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "logs" / "token_usage.json"
_usage_lock = threading.Lock()


class LLMBudgetExceeded(Exception):
    """每日 token 预算耗尽"""


def get_llm(timeout: int | None = None) -> OpenAI:
    """按配置的 provider 返回带超时/重试的 OpenAI 客户端(单例)"""
    global _client
    if _client is None:
        if settings.llm_provider == "qwen":
            api_key, base_url = settings.qwen_api_key, settings.qwen_base_url
        else:
            api_key, base_url = settings.deepseek_api_key, settings.deepseek_base_url
        _client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )
    # 允许个别调用用更短超时(如查询改写): 通过 with_options 派生, 不污染单例
    if timeout is not None and timeout != settings.llm_timeout_seconds:
        return _client.with_options(timeout=httpx.Timeout(timeout))
    return _client


def current_model() -> str:
    return settings.qwen_model if settings.llm_provider == "qwen" else settings.deepseek_model


def _check_budget(estimated_tokens: int = 0) -> None:
    if settings.llm_daily_token_budget <= 0:
        return
    used = _today_usage()
    if used + estimated_tokens > settings.llm_daily_token_budget:
        raise LLMBudgetExceeded(
            f"今日 token 预算已用完 ({used}/{settings.llm_daily_token_budget})"
        )


def _today_usage() -> int:
    try:
        data = json.loads(USAGE_PATH.read_text(encoding="utf-8"))
        return data.get(str(date.today()), 0)
    except Exception:
        return 0


def record_usage(usage) -> None:
    """记录一次调用的 token 用量(metrics 事件 + 每日累计文件)"""
    if usage is None:
        return
    prompt = getattr(usage, "prompt_tokens", 0) or 0
    completion = getattr(usage, "completion_tokens", 0) or 0
    total = prompt + completion
    log_event("llm_usage", model=current_model(),
              prompt_tokens=prompt, completion_tokens=completion, total_tokens=total)
    try:
        with _usage_lock:
            today = str(date.today())
            data = {}
            if USAGE_PATH.exists():
                data = json.loads(USAGE_PATH.read_text(encoding="utf-8"))
            data[today] = data.get(today, 0) + total
            USAGE_PATH.write_text(json.dumps(data), encoding="utf-8")
    except Exception:
        logger.exception("record_usage failed")


def chat_completion(messages: list[dict], *, max_tokens: int | None = None,
                    temperature: float = 0.3, timeout: int | None = None,
                    caller: str = "unknown"):
    """非流式调用, 自动重试 + 用量记录"""
    _check_budget()
    resp = get_llm(timeout).chat.completions.create(
        model=current_model(),
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    record_usage(getattr(resp, "usage", None))
    return resp


def stream_completion(messages: list[dict], *, max_tokens: int | None = None,
                      temperature: float = 0.3, caller: str = "unknown"):
    """流式调用。连接建立阶段有重试; 一旦开始出字, 网络错误直接抛给上层。
    DeepSeek 兼容 stream_options.include_usage, 流末尾会带一次带 usage 的 chunk。"""
    _check_budget()
    try:
        return get_llm().chat.completions.create(
            model=current_model(),
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            stream_options={"include_usage": True},
        )
    except Exception:
        logger.exception("LLM stream_connection failed (caller=%s)", caller)
        raise
