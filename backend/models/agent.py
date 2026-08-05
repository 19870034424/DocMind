"""Agentic RAG 数据模型"""
from __future__ import annotations

import enum
import uuid
from datetime import datetime

from pydantic import BaseModel, Field


class AgentStepType(str, enum.Enum):
    PLAN = "plan"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    REFLECT = "reflect"
    CLARIFY = "clarify"
    ANSWER = "answer"
    ERROR = "error"


class AgentStep(BaseModel):
    """Agent 执行过程中的单个步骤"""
    step_index: int
    type: AgentStepType
    content: str
    detail: dict | None = None
    timestamp: datetime = Field(default_factory=datetime.now)


class Source(BaseModel):
    """答案引用来源"""
    heading: str = ""
    page: int = 0
    excerpt: str = ""
    score: float = 0.0


class AgentResult(BaseModel):
    """一次 Agent 深度问答的完整结果"""
    question: str
    steps: list[AgentStep] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)
    final_answer: str = ""
    total_steps: int = 0
    iterations: int = 0
    tokens_used: int = 0
    follow_up_suggestions: list[str] = Field(default_factory=list)
    session_id: str | None = None


class ClarifyRequest(BaseModel):
    """用户回复追问的请求"""
    session_id: str
    user_response: str


class AgentSession:
    """内存中的 Agent 会话（30 分钟 TTL）"""

    def __init__(self, doc_id: str, question: str):
        self.session_id = str(uuid.uuid4())
        self.doc_id = doc_id
        self.question = question
        self.messages: list[dict] = []
        self.created_at = datetime.now()
        self.iteration_count = 0

    def is_expired(self, timeout_minutes: int = 30) -> bool:
        from datetime import timedelta
        return datetime.now() - self.created_at > timedelta(minutes=timeout_minutes)
