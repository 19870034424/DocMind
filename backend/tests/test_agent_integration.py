"""Agent 集成测试 — 工具执行 + 格式化 + AgentLoop 端到端模拟"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
from unittest.mock import MagicMock, patch

from services.agent import (
    AgentLoop,
    _detect_clarify_intent,
    _parse_tool_calls,
)
from services.agent_tools import TOOL_MAP, execute_tool, format_tool_result


class TestIntegrationToolExecution:
    """工具执行集成测试"""

    def test_all_tools_in_map(self):
        assert "search_document" in TOOL_MAP
        assert "get_section" in TOOL_MAP
        assert "query_knowledge_graph" in TOOL_MAP
        assert "search_keyword" in TOOL_MAP
        assert "reflect_on_answer" in TOOL_MAP

    def test_execute_unknown_tool(self):
        result = execute_tool("doc-1", "nonexistent_tool", {})
        assert "error" in result

    def test_execute_reflect(self):
        result = execute_tool("", "reflect_on_answer", {
            "draft_answer": "test answer long enough for check",
            "cited_sources": ["source 1"],
        })
        assert "tool" in result

    def test_ask_user_passthrough(self):
        args = {"question": "你想了解哪个章节？", "options": ["第3章", "第5章"]}
        result = execute_tool("doc-1", "ask_user", args)
        assert result == args


class TestIntegrationFormatToolResult:
    """format_tool_result 格式化集成"""

    def test_search_document_with_hits(self):
        result = {"hits": [{"text": "test", "heading": "Ch1", "page": 1}], "total": 1}
        formatted = format_tool_result("search_document", result)
        assert "找到 1 个" in formatted
        assert "Ch1" in formatted

    def test_search_document_empty(self):
        formatted = format_tool_result("search_document", {"hits": [], "total": 0})
        assert "未找到" in formatted

    def test_get_section_with_hits(self):
        result = {"hits": [{"heading": "Ch1", "content": "test content", "page": 5}], "total": 1}
        formatted = format_tool_result("get_section", result)
        assert "Ch1" in formatted

    def test_get_section_empty(self):
        result = {"hits": [], "total": 0, "available_headings": ["Ch1", "Ch2"]}
        formatted = format_tool_result("get_section", result)
        assert "未找到" in formatted or "匹配章节" in formatted

    def test_kg_format_found(self):
        result = {
            "found": True,
            "concept": {"name": "DP", "type": "algorithm", "summary": "desc", "importance": 0.9, "source_sections": ["Ch3"]},
            "neighbors": [{"concept": "递归", "relation": "prerequisite", "description": "前置知识"}],
        }
        formatted = format_tool_result("query_knowledge_graph", result)
        assert "DP" in formatted
        assert "递归" in formatted

    def test_kg_format_not_found(self):
        result = {"found": False, "message": "未找到概念", "similar": ["DP"], "tool": "query_knowledge_graph"}
        formatted = format_tool_result("query_knowledge_graph", result)
        assert "未找到" in formatted or "DP" in formatted

    def test_reflect_format_passed(self):
        formatted = format_tool_result("reflect_on_answer", {"passed": True, "issues": []})
        assert "通过" in formatted

    def test_reflect_format_failed(self):
        formatted = format_tool_result("reflect_on_answer", {"passed": False, "issues": ["缺少引用"]})
        assert "未通过" in formatted or "缺少引用" in formatted

    def test_keyword_format(self):
        result = {"hits": [{"text": "hello", "heading": "Ch1", "page": 1}], "total": 1}
        formatted = format_tool_result("search_keyword", result)
        assert "找到 1 个" in formatted

    def test_error_format(self):
        formatted = format_tool_result("search_document", {"error": "timeout"})
        assert "失败" in formatted or "timeout" in formatted


class TestIntegrationAgentLoopMocks:
    """AgentLoop 端到端 — 模拟 LLM 返回"""

    @patch("services.agent._build_llm_client")
    def test_full_loop_tool_call_then_answer(self, mock_build):
        mock_client = MagicMock()
        mock_build.return_value = (mock_client, "test-model")

        # 第一次调用: tool_call (search_document)
        tc_choice = MagicMock()
        tc_msg = MagicMock()
        tc_msg.content = None
        tc = MagicMock()
        tc.id = "call_1"
        tc.function.name = "search_document"
        tc.function.arguments = '{"query": "动态规划", "top_k": 5}'
        tc_msg.tool_calls = [tc]
        tc_choice.message = tc_msg

        # 第二次调用: 无 tool_calls (进入回答阶段)
        answer_choice = MagicMock()
        answer_msg = MagicMock()
        answer_msg.content = "根据检索结果..."
        answer_msg.tool_calls = None
        answer_choice.message = answer_msg

        # 第三次调用: 流式答案
        stream_chunk = MagicMock()
        stream_chunk.choices = [MagicMock()]
        stream_chunk.choices[0].delta = MagicMock()
        stream_chunk.choices[0].delta.content = "动态规划是一种算法。"

        # Mock chunker for tool execution
        with patch("services.agent_tools.chunker") as mock_chunker:
            mock_chunker.embed.return_value = [[0.1] * 512]
            with patch("services.agent_tools.milvus_client") as mock_milvus:
                mock_milvus.hybrid_query.return_value = [
                    {"text": "动态规划通过存储子问题解来优化", "metadata": {"heading_path": "第3章", "page": 42}},
                ]

                # Mock 顺序: tool_call → answer → stream(1 chunk then StopIteration)
                mock_client.chat.completions.create.side_effect = [
                    tc_choice,
                    answer_choice,
                    iter([stream_chunk]),
                ]

                loop = AgentLoop(doc_id="doc-1", question="动态规划是什么？", max_iterations=3)

                async def collect():
                    events = []
                    async for event in loop.run():
                        events.append(event)
                    return events

                events = asyncio.run(collect())
                assert len(events) > 0

    @patch("services.agent._build_llm_client")
    def test_force_answer_at_max_iterations(self, mock_build):
        loop = AgentLoop(doc_id="doc-1", question="test", max_iterations=1)
        loop.iteration_count = 1
        assert loop._should_force_answer()

    @patch("services.agent._build_llm_client")
    def test_resume_appends_user_response(self, mock_build):
        loop = AgentLoop(doc_id="doc-1", question="test")
        orig = len(loop.messages)
        loop.resume("重点看第3章")
        assert len(loop.messages) == orig + 1
        assert "重点看第3章" in loop.messages[-1]["content"]

    @patch("services.agent._build_llm_client")
    def test_generate_follow_ups_comparison(self, mock_build):
        loop = AgentLoop(doc_id="doc-1", question="A和B有什么区别？")
        loop.used_sources = []
        loop.iteration_count = 1
        result = loop._generate_follow_ups("答案")
        assert any("详细" in s for s in result)

    @patch("services.agent._build_llm_client")
    def test_generate_follow_ups_near_limit(self, mock_build):
        loop = AgentLoop(doc_id="doc-1", question="test", max_iterations=8)
        loop.used_sources = []
        loop.iteration_count = 7
        result = loop._generate_follow_ups("答案")
        assert any("重新" in s or "不够全面" in s for s in result)


class TestIntegrationClarify:
    """clarify 检测集成"""

    def test_clarify_many_patterns(self):
        assert _detect_clarify_intent("请问你想了解哪个部分？")
        assert _detect_clarify_intent("你能告诉我具体需要哪一章吗？")
        assert _detect_clarify_intent("你希望我重点分析哪个章节？")

    def test_not_clarify(self):
        assert not _detect_clarify_intent("根据文档内容，答案是...")
        assert not _detect_clarify_intent("hi")
        assert not _detect_clarify_intent("")


class TestIntegrationParseToolCalls:
    """工具调用解析集成"""

    def test_parse_valid_search_document(self):
        msg = MagicMock()
        tc = MagicMock()
        tc.id = "call_1"
        tc.function.name = "search_document"
        tc.function.arguments = '{"query": "test", "top_k": 5}'
        msg.tool_calls = [tc]
        result = _parse_tool_calls(msg)
        assert len(result) == 1
        assert result[0]["name"] == "search_document"
        assert result[0]["arguments"]["top_k"] == 5

    def test_parse_invalid_tool_skipped(self):
        msg = MagicMock()
        tc1 = MagicMock()
        tc1.id = "c1"
        tc1.function.name = "good_tool"
        tc1.function.arguments = '{"q": "test"}'
        tc2 = MagicMock()
        tc2.id = "c2"
        tc2.function.name = "bad_tool"
        tc2.function.arguments = '{invalid json}'
        msg.tool_calls = [tc1, tc2]
        result = _parse_tool_calls(msg)
        # Only the valid one should be parsed
        assert len(result) == 1
        assert result[0]["name"] == "good_tool"
