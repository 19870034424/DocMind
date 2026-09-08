"""Agent 数据模型测试 — AgentStep / AgentResult / Source 序列化与验证"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json

from models.agent import (
    AgentResult,
    AgentSession,
    AgentStep,
    AgentStepType,
    ClarifyRequest,
    Source,
)


class TestAgentStep:
    """AgentStep — 单个步骤的序列化"""

    def test_create_plan_step(self):
        step = AgentStep(step_index=0, type=AgentStepType.PLAN, content="制定检索计划")
        d = step.model_dump()
        assert d["step_index"] == 0
        assert d["type"] == "plan"
        assert d["content"] == "制定检索计划"
        assert d["detail"] is None
        assert "timestamp" in d

    def test_create_tool_call_step_with_detail(self):
        step = AgentStep(
            step_index=1,
            type=AgentStepType.TOOL_CALL,
            content="调用 search_document",
            detail={"tool_name": "search_document", "arguments": {"query": "动态规划"}},
        )
        d = step.model_dump()
        assert d["detail"]["tool_name"] == "search_document"
        assert d["detail"]["arguments"]["query"] == "动态规划"

    def test_create_error_step(self):
        step = AgentStep(
            step_index=3,
            type=AgentStepType.ERROR,
            content="工具调用超时",
            detail={"error": "timeout", "tool": "search_document"},
        )
        assert step.type == AgentStepType.ERROR

    def test_json_serialization(self):
        """验证完整序列化为 JSON 字符串"""
        step = AgentStep(
            step_index=0,
            type=AgentStepType.PLAN,
            content="test",
            detail={"key": "value"},
        )
        json_str = step.model_dump_json()
        data = json.loads(json_str)
        assert data["step_index"] == 0
        assert data["type"] == "plan"
        assert data["detail"]["key"] == "value"


class TestAgentResult:
    """AgentResult — 完整 Agent 执行结果"""

    def test_empty_result(self):
        result = AgentResult(question="测试问题")
        assert result.question == "测试问题"
        assert result.steps == []
        assert result.sources == []
        assert result.final_answer == ""
        assert result.total_steps == 0

    def test_full_result(self):
        steps = [
            AgentStep(step_index=0, type=AgentStepType.PLAN, content="plan"),
            AgentStep(step_index=1, type=AgentStepType.ANSWER, content="answer"),
        ]
        sources = [
            Source(heading="第1章", page=5, excerpt="原文...", score=0.9),
        ]
        result = AgentResult(
            question="test?",
            steps=steps,
            sources=sources,
            final_answer="答案",
            total_steps=2,
            iterations=1,
            tokens_used=500,
            follow_up_suggestions=["需要详细解释吗？"],
            session_id="abc-123",
        )
        d = result.model_dump()
        assert len(d["steps"]) == 2
        assert len(d["sources"]) == 1
        assert d["session_id"] == "abc-123"


class TestSource:
    """Source — 引用来源"""

    def test_source_defaults(self):
        s = Source(excerpt="原文片段")
        assert s.heading == ""
        assert s.page == 0
        assert s.score == 0.0

    def test_source_full(self):
        s = Source(heading="第3章 动态规划", page=42, excerpt="原文...", score=0.92)
        d = s.model_dump()
        assert d["heading"] == "第3章 动态规划"
        assert d["page"] == 42


class TestClarifyRequest:
    """ClarifyRequest — 追问回复请求"""

    def test_valid_request(self):
        req = ClarifyRequest(session_id="abc-123", user_response="重点看第3章")
        assert req.session_id == "abc-123"
        assert req.user_response == "重点看第3章"


class TestAgentSession:
    """AgentSession — 会话管理（非持久化，仅内存）"""

    def test_session_creation(self):
        session = AgentSession(doc_id="doc-1", question="测试问题")
        assert session.doc_id == "doc-1"
        assert session.session_id is not None
        assert len(session.session_id) == 36  # UUID4
        assert session.messages == []

    def test_session_is_expired(self):
        from datetime import datetime, timedelta
        session = AgentSession(doc_id="doc-1", question="test")
        # 新创建的会话不应该过期
        assert not session.is_expired(timeout_minutes=30)
        # 模拟过期
        session.created_at = datetime.now() - timedelta(minutes=31)
        assert session.is_expired(timeout_minutes=30)
