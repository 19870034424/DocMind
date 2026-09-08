"""Agent 循环测试 — ReAct 决策 / clarify 检测 / 最大轮数截断 / 工具解析"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import MagicMock, patch

from services.agent import (
    MAX_ITERATIONS,
    AgentLoop,
    _detect_clarify_intent,
    _parse_tool_calls,
)


class TestDetectClarifyIntent:
    """_detect_clarify_intent — 检测 LLM 是否在追问"""

    def test_explicit_clarify_question(self):
        assert _detect_clarify_intent("请问你想了解哪个章节？")

    def test_non_clarify_text(self):
        assert not _detect_clarify_intent("动态规划是一种算法设计技术。")

    def test_clarify_with_options(self):
        assert _detect_clarify_intent("文档涵盖了8个章节，你想总结全部还是选几个？")

    def test_short_answer_not_clarify(self):
        assert not _detect_clarify_intent("可以")

    def test_clarify_multiple_keywords(self):
        assert _detect_clarify_intent("你能告诉我具体想看哪一部分吗？")

    def test_factual_answer_not_clarify(self):
        assert not _detect_clarify_intent("根据第3章，动态规划的核心思想是将问题分解为子问题。")


class TestParseToolCalls:
    """_parse_tool_calls — 解析 LLM 返回的 tool_calls"""

    def test_none_choice(self):
        result = _parse_tool_calls(None)
        assert result == []

    def test_no_tool_calls(self):
        msg = MagicMock()
        msg.tool_calls = None
        result = _parse_tool_calls(msg)
        assert result == []

    def test_valid_tool_calls(self):
        msg = MagicMock()
        tc = MagicMock()
        tc.id = "call_1"
        tc.function.name = "search_document"
        tc.function.arguments = '{"query": "动态规划", "top_k": 5}'
        msg.tool_calls = [tc]
        result = _parse_tool_calls(msg)
        assert len(result) == 1
        assert result[0]["id"] == "call_1"
        assert result[0]["name"] == "search_document"
        assert result[0]["arguments"] == {"query": "动态规划", "top_k": 5}

    def test_invalid_json_arguments(self):
        msg = MagicMock()
        tc = MagicMock()
        tc.id = "call_2"
        tc.function.name = "bad_tool"
        tc.function.arguments = "not valid json"
        msg.tool_calls = [tc]
        result = _parse_tool_calls(msg)
        assert result == []

    def test_empty_arguments(self):
        msg = MagicMock()
        tc = MagicMock()
        tc.id = "call_3"
        tc.function.name = "reflect_on_answer"
        tc.function.arguments = ""
        msg.tool_calls = [tc]
        result = _parse_tool_calls(msg)
        assert len(result) == 1
        assert result[0]["arguments"] == {}

    def test_multiple_tool_calls(self):
        msg = MagicMock()
        tc1 = MagicMock()
        tc1.id = "call_1"
        tc1.function.name = "search_document"
        tc1.function.arguments = '{"query": "a"}'
        tc2 = MagicMock()
        tc2.id = "call_2"
        tc2.function.name = "search_keyword"
        tc2.function.arguments = '{"keyword": "b"}'
        msg.tool_calls = [tc1, tc2]
        result = _parse_tool_calls(msg)
        assert len(result) == 2


class TestAgentLoopInit:
    """AgentLoop 初始化"""

    @patch("services.agent._build_llm_client")
    def test_creates_with_doc_id_and_question(self, mock_build):
        loop = AgentLoop(doc_id="doc-1", question="测试问题")
        assert loop.doc_id == "doc-1"
        assert loop.question == "测试问题"
        assert loop.iteration_count == 0
        assert len(loop.messages) >= 2
        assert loop.messages[0]["role"] == "system"
        assert loop.messages[1]["role"] == "user"

    @patch("services.agent._build_llm_client")
    def test_user_message_contains_question(self, mock_build):
        loop = AgentLoop(doc_id="doc-1", question="什么是动态规划？")
        user_msgs = [m for m in loop.messages if m["role"] == "user"]
        assert any("什么是动态规划" in m.get("content", "") for m in user_msgs)

    @patch("services.agent._build_llm_client")
    def test_default_max_iterations(self, mock_build):
        loop = AgentLoop(doc_id="doc-1", question="test")
        assert loop.max_iterations == MAX_ITERATIONS

    @patch("services.agent._build_llm_client")
    def test_custom_max_iterations(self, mock_build):
        loop = AgentLoop(doc_id="doc-1", question="test", max_iterations=3)
        assert loop.max_iterations == 3


class TestAgentLoopIntegrity:
    """AgentLoop 完整性和边界条件"""

    @patch("services.agent._build_llm_client")
    def test_should_force_answer_when_at_max(self, mock_build):
        loop = AgentLoop(doc_id="doc-1", question="test", max_iterations=2)
        assert not loop._should_force_answer()
        loop.iteration_count = 2
        assert loop._should_force_answer()

    @patch("services.agent._build_llm_client")
    def test_resume_appends_message(self, mock_build):
        loop = AgentLoop(doc_id="doc-1", question="test")
        original_count = len(loop.messages)
        loop.resume("重点看第3章")
        assert len(loop.messages) == original_count + 1
        assert "重点看第3章" in loop.messages[-1]["content"]

    @patch("services.agent._build_llm_client")
    def test_generate_follow_ups_for_comparison(self, mock_build):
        loop = AgentLoop(doc_id="doc-1", question="A和B有什么区别？")
        loop.used_sources = []
        loop.iteration_count = 1
        suggestions = loop._generate_follow_ups("答案")
        assert any("详细" in s for s in suggestions)

    @patch("services.agent._build_llm_client")
    def test_generate_follow_ups_near_limit(self, mock_build):
        loop = AgentLoop(doc_id="doc-1", question="test", max_iterations=8)
        loop.used_sources = []
        loop.iteration_count = 7
        suggestions = loop._generate_follow_ups("答案")
        assert any("重新" in s or "不够全面" in s for s in suggestions)


class TestMaxIterations:
    """MAX_ITERATIONS 常量"""

    def test_max_iterations_value(self):
        assert MAX_ITERATIONS == 8
        assert isinstance(MAX_ITERATIONS, int)
