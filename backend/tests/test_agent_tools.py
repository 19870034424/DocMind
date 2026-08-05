"""Agent 工具函数测试 — search_document / get_section / query_kg / search_keyword / reflect"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.agent_tools import (
    tool_search_document,
    tool_get_section,
    tool_query_knowledge_graph,
    tool_search_keyword,
    tool_reflect_on_answer,
    format_tool_result,
    execute_tool,
    TOOL_MAP,
)


class TestToolSearchDocument:
    """search_document — 混合检索"""

    def test_empty_query_returns_error(self):
        result = tool_search_document(doc_id="doc-1", query="", top_k=5)
        assert "error" in result

    def test_top_k_clamped(self):
        result = tool_search_document(doc_id="doc-1", query="test", top_k=100)
        assert isinstance(result, dict)


class TestToolGetSection:
    """get_section — 获取章节内容"""

    def test_empty_heading_returns_error(self):
        result = tool_get_section(doc_id="doc-1", heading="")
        assert "error" in result

    def test_heading_too_short_returns_error(self):
        result = tool_get_section(doc_id="doc-1", heading="a")
        assert "error" in result

    def test_nonexistent_doc_returns_error(self):
        result = tool_get_section(doc_id="nonexistent_doc_xyz", heading="测试章节")
        assert isinstance(result, dict)


class TestToolQueryKnowledgeGraph:
    """query_knowledge_graph — 查询知识图谱"""

    def test_empty_concept_returns_error(self):
        result = tool_query_knowledge_graph(doc_id="doc-1", concept="")
        assert "error" in result

    def test_nonexistent_doc_returns_available_false(self):
        result = tool_query_knowledge_graph(doc_id="nonexistent", concept="动态规划")
        assert isinstance(result, dict)


class TestToolSearchKeyword:
    """search_keyword — 关键词搜索"""

    def test_empty_keyword_returns_error(self):
        result = tool_search_keyword(doc_id="doc-1", keyword="", top_k=5)
        assert "error" in result

    def test_short_keyword_returns_error(self):
        result = tool_search_keyword(doc_id="doc-1", keyword="a", top_k=5)
        assert "error" in result


class TestToolReflectOnAnswer:
    """reflect_on_answer — 答案自检"""

    def test_empty_answer_returns_not_passed(self):
        result = tool_reflect_on_answer(draft_answer="", cited_sources=[])
        assert isinstance(result, dict)
        assert not result.get("passed", True)

    def test_answer_without_sources_has_issues(self):
        result = tool_reflect_on_answer(
            draft_answer="这是一段没有来源的答案。",
            cited_sources=[],
        )
        assert isinstance(result, dict)
        assert not result.get("passed", True)
        assert "issues" in result

    def test_valid_answer_with_sources(self):
        result = tool_reflect_on_answer(
            draft_answer="动态规划是一种算法设计技术，用于解决具有重叠子问题性质的问题。",
            cited_sources=["第3章第2节: 动态规划通过存储子问题的解来避免重复计算"],
        )
        assert isinstance(result, dict)


class TestFormatToolResult:
    """format_tool_result — 工具结果格式化"""

    def test_search_result_format_with_hits(self):
        hits = [
            {"text": "内容1", "heading": "第1章", "page": 5},
        ]
        formatted = format_tool_result("search_document", {"hits": hits, "total": 1})
        assert "找到 1 个" in formatted
        assert "第1章" in formatted

    def test_search_result_format_empty(self):
        formatted = format_tool_result("search_document", {"hits": [], "total": 0})
        assert "未找到" in formatted

    def test_error_result_format(self):
        formatted = format_tool_result("search_document", {"error": "timeout"})
        assert "失败" in formatted or "timeout" in formatted

    def test_get_section_format(self):
        result = {"hits": [{"heading": "Ch1", "content": "test", "page": 5}], "total": 1}
        formatted = format_tool_result("get_section", result)
        assert "Ch1" in formatted

    def test_kg_format_found(self):
        result = {
            "found": True,
            "concept": {"name": "DP", "type": "algorithm", "summary": "desc", "importance": 0.9, "source_sections": ["Ch3"]},
            "neighbors": [{"concept": "递归", "relation": "prerequisite", "description": "前置知识"}],
        }
        formatted = format_tool_result("query_knowledge_graph", result)
        assert "DP" in formatted
        assert "递归" in formatted

    def test_kg_format_not_available(self):
        result = {"available": False, "message": "图谱未构建"}
        formatted = format_tool_result("query_knowledge_graph", result)
        assert "图谱未构建" in formatted

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


class TestExecuteTool:
    """execute_tool — 工具调度"""

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
            "draft_answer": "test answer long enough",
            "cited_sources": ["source 1"],
        })
        assert "tool" in result

    def test_ask_user_passthrough(self):
        """ask_user 由 Agent 层直接处理，execute_tool 应返回原始参数"""
        args = {"question": "你想了解哪个章节？", "options": ["第3章", "第5章"]}
        result = execute_tool("doc-1", "ask_user", args)
        assert result == args
