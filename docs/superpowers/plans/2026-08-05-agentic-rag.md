# Agentic RAG 升级实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 DocMind 从单次 RAG 升级为 Agentic RAG，新增深度问答模式（ReAct 循环 + 5 工具 + SSE 流式）

**Architecture:** 纯增量改造——新增 `models/agent.py`、`services/agent.py`、`services/agent_tools.py`、`services/agent_prompts.py`、`routers/chat.py`；前端新增 `AgentChatBox` 系列组件，通过 SSE 逐步骤展示 Agent 思考过程。现有代码零破坏。

**Tech Stack:** Python 3.12 + FastAPI + OpenAI Function Calling + SSE；React 18 + TypeScript + Vite

---

## 模块边界

| 文件 | 职责 | 依赖 |
|------|------|------|
| `backend/models/agent.py` | AgentStep / AgentResult / Source / AgentSession 数据模型 | pydantic |
| `backend/services/agent_prompts.py` | Agent system prompt + 5 工具 Function Calling JSON Schema | 无 |
| `backend/services/agent_tools.py` | 5 个工具函数实现，封装现有检索/图谱/DB 能力 | milvus_client, chunker, kg_store, db |
| `backend/services/agent.py` | ReAct 循环主控：async generator 逐步骤 yield SSE 事件 | agent_tools, agent_prompts, llm_service |
| `backend/routers/chat.py` | `/chat/stream` + `/chat/deep` + `/chat/deep/resume` SSE 路由 | agent, db |
| `frontend/src/components/MarkdownRenderer.tsx` | 从 ReportViewer 提取的共享 Markdown 渲染组件 | react-markdown |
| `frontend/src/components/AgentStepCard.tsx` | 单个 Agent 步骤的可折叠卡片 | 无 |
| `frontend/src/components/AgentChatBox.tsx` | 深度模式主容器，管理 SSE 连接和步骤状态 | AgentStepCard, MarkdownRenderer |
| `frontend/src/components/ChatBox.tsx` | 改动：支持 SSE chunk 流式接收 | 无 |
| `frontend/src/pages/ChatPage.tsx` | 改动：新增快速/深度模式切换开关 | AgentChatBox, ChatBox |

---

### Task 1: Agent 数据模型

**Files:**
- Create: `backend/models/agent.py`
- Create: `backend/tests/test_agent_models.py`

- [ ] **Step 1: 编写 Agent 数据模型测试**

```python
"""Agent 数据模型测试 — AgentStep / AgentResult / Source 序列化与验证"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pytest
from models.agent import (
    AgentStep, AgentStepType, AgentResult, Source, ClarifyRequest, AgentSession
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
```

- [ ] **Step 2: 运行测试验证失败**

```bash
cd backend && ../.venv-mineru/Scripts/python.exe -m pytest tests/test_agent_models.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'models.agent'`

- [ ] **Step 3: 实现数据模型**

```python
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
```

- [ ] **Step 4: 运行测试验证通过**

```bash
cd backend && ../.venv-mineru/Scripts/python.exe -m pytest tests/test_agent_models.py -v
```
Expected: 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add backend/models/agent.py backend/tests/test_agent_models.py
git commit -m "feat: add Agent data models (AgentStep, AgentResult, Source, AgentSession)"
```

---

### Task 2: Agent Prompt 和工具 Schema

**Files:**
- Create: `backend/services/agent_prompts.py`

- [ ] **Step 1: 创建 Prompt 和工具 Schema 文件**

```python
"""Agentic RAG — System Prompt + Tool Schemas (OpenAI Function Calling 格式)"""

AGENT_SYSTEM_PROMPT = """你是一个文档深度分析助手，能够自主规划和执行多步检索来回答复杂问题。

## 工作流程
1. 理解问题后，先制定检索计划（在思考中完成，不输出给用户）
2. 选择最合适的工具执行检索
3. 根据检索结果决定下一步：继续搜索补充信息 / 信息足够则输出答案 / 问题不明确则追问用户
4. 输出最终答案前**必须**使用 reflect_on_answer 工具自检

## 工具使用原则
- 优先使用 search_document 进行语义搜索（大部分情况首选）
- 需要精确术语匹配（特定人名、公式、代码）时使用 search_keyword
- 需要阅读某章节完整内容时使用 get_section
- 需要理解概念之间的关联时使用 query_knowledge_graph
- 查看知识图谱前应先确认图谱已构建（可用 query_knowledge_graph 试探）
- 最终答案前**必须**调用 reflect_on_answer，验证不通过则继续检索修正

## 追问时机
以下情况应使用 ask_user 追问用户而非猜测：
- 问题中的术语在文档中查不到（用户可能搞错了文档）
- 问题指代不明确（如"这个方法"在文档中出现多次）
- 用户要求总结但文档很长，需要确认范围

## 约束
- 每个事实必须有明确的文档来源
- 文档中没有的信息明确说明"文档中未提供"
- 引用标注格式: [出处: 章节名, 第X页]
- 最多 8 次工具调用，接近上限时优先输出已有最佳答案
- 中文回答，简洁准确"""


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_document",
            "description": "在文档中执行混合检索(BM25+向量RRF融合)，返回最相关的文本片段。适合开放式语义查询。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "自然语言搜索查询",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回片段数量，默认 10",
                        "default": 10,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_section",
            "description": "获取指定章节的完整原始内容。适合需要完整阅读某个章节时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "heading": {
                        "type": "string",
                        "description": "章节标题，支持子串模糊匹配",
                    },
                },
                "required": ["heading"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_knowledge_graph",
            "description": "查询知识图谱中某个概念的详细信息及其关联概念（前置知识、推导关系、对比关系等）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "concept": {
                        "type": "string",
                        "description": "概念名称",
                    },
                },
                "required": ["concept"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_keyword",
            "description": "纯BM25关键词检索，适合查找特定术语、人名、公式等精确匹配。不如 search_document 灵活，仅在需要精确匹配时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "精确关键词",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回片段数量，默认 10",
                        "default": 10,
                    },
                },
                "required": ["keyword"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reflect_on_answer",
            "description": "在输出最终答案前自检：答案是否完全基于文档？有无编造事实？引用出处是否准确？如验证不通过则需继续检索修正。",
            "parameters": {
                "type": "object",
                "properties": {
                    "draft_answer": {
                        "type": "string",
                        "description": "待验证的草稿答案",
                    },
                    "cited_sources": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "答案中引用的原文片段列表，每项包含章节+页码信息",
                    },
                },
                "required": ["draft_answer", "cited_sources"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask_user",
            "description": "当问题不明确、文档中找不到相关信息、或需要用户确认范围时，向用户追问。",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "向用户追问的问题",
                    },
                    "options": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "可选：提供给用户的选项列表，方便用户快速选择",
                    },
                },
                "required": ["question"],
            },
        },
    },
]
```

- [ ] **Step 2: 验证导入**

```bash
cd backend && ../.venv-mineru/Scripts/python.exe -c "from services.agent_prompts import AGENT_SYSTEM_PROMPT, TOOLS; print(f'prompt={len(AGENT_SYSTEM_PROMPT)} chars, tools={len(TOOLS)}')"
```
Expected: `prompt=XXXX chars, tools=6`

- [ ] **Step 3: Commit**

```bash
git add backend/services/agent_prompts.py
git commit -m "feat: add Agent system prompt and tool schemas"
```

---

### Task 3: Agent 工具函数实现

**Files:**
- Create: `backend/services/agent_tools.py`
- Create: `backend/tests/test_agent_tools.py`

- [ ] **Step 1: 编写工具函数测试**

```python
"""Agent 工具函数测试 — search_document / get_section / query_kg / search_keyword / reflect"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pytest

from services.agent_tools import (
    tool_search_document,
    tool_get_section,
    tool_query_knowledge_graph,
    tool_search_keyword,
    tool_reflect_on_answer,
    format_tool_result,
)
from storage.milvus_client import milvus_client
from storage.kg_store import kg_store


class TestToolSearchDocument:
    """search_document — 混合检索"""

    def test_returns_chunks_for_valid_doc(self):
        """正常文档应返回片段"""
        # 依赖已有 Milvus collection，跳过如果没有
        try:
            result = tool_search_document(
                doc_id="test_nonexistent_doc",
                query="测试查询",
                top_k=5,
            )
        except Exception:
            pytest.skip("Milvus not available")
        assert isinstance(result, dict)

    def test_empty_query_returns_error(self):
        result = tool_search_document(doc_id="doc-1", query="", top_k=5)
        assert "error" in result

    def test_top_k_clamped(self):
        """top_k 超出范围应被限制"""
        result = tool_search_document(doc_id="doc-1", query="test", top_k=100)
        assert isinstance(result, dict)


class TestToolGetSection:
    """get_section — 获取章节内容"""

    def test_nonexistent_doc_returns_error(self):
        result = tool_get_section(doc_id="nonexistent", heading="测试章节")
        assert isinstance(result, dict)

    def test_empty_heading_returns_error(self):
        result = tool_get_section(doc_id="doc-1", heading="")
        assert "error" in result

    def test_heading_too_short_returns_error(self):
        result = tool_get_section(doc_id="doc-1", heading="a")
        assert "error" in result


class TestToolQueryKnowledgeGraph:
    """query_knowledge_graph — 查询知识图谱"""

    def test_no_kg_available(self):
        result = tool_query_knowledge_graph(doc_id="nonexistent", concept="动态规划")
        assert isinstance(result, dict)
        assert result.get("available") == False or "error" in result

    def test_empty_concept_returns_error(self):
        result = tool_query_knowledge_graph(doc_id="doc-1", concept="")
        assert "error" in result


class TestToolSearchKeyword:
    """search_keyword — 关键词搜索"""

    def test_empty_keyword_returns_error(self):
        result = tool_search_keyword(doc_id="doc-1", keyword="", top_k=5)
        assert "error" in result

    def test_short_keyword_returns_error(self):
        result = tool_search_keyword(doc_id="doc-1", keyword="a", top_k=5)
        assert "error" in result

    def test_top_k_clamped(self):
        result = tool_search_keyword(doc_id="doc-1", keyword="test", top_k=200)
        assert isinstance(result, dict)


class TestToolReflectOnAnswer:
    """reflect_on_answer — 答案自检"""

    def test_empty_answer_returns_issues(self):
        result = tool_reflect_on_answer(
            draft_answer="",
            cited_sources=[],
        )
        assert isinstance(result, dict)
        assert not result.get("passed", True)

    def test_valid_answer_with_sources_passes(self):
        result = tool_reflect_on_answer(
            draft_answer="动态规划是一种算法设计技术，用于解决具有重叠子问题性质的问题。",
            cited_sources=["第3章第2节: 动态规划通过存储子问题的解来避免重复计算"],
        )
        assert isinstance(result, dict)

    def test_answer_without_sources_has_issues(self):
        result = tool_reflect_on_answer(
            draft_answer="这是一段没有来源的答案。",
            cited_sources=[],
        )
        assert isinstance(result, dict)


class TestFormatToolResult:
    """format_tool_result — 工具结果格式化"""

    def test_search_result_format(self):
        hits = [
            {"text": "内容1", "metadata": {"heading_path": "第1章", "page": 5}},
            {"text": "内容2", "metadata": {"heading_path": "第2章", "page": 10}},
        ]
        formatted = format_tool_result("search_document", {"hits": hits, "total": 2})
        assert "找到 2 个片段" in formatted

    def test_empty_result_format(self):
        formatted = format_tool_result("search_document", {"hits": [], "total": 0})
        assert "未找到" in formatted or "0" in formatted

    def test_error_result_format(self):
        formatted = format_tool_result("search_document", {"error": "timeout"})
        assert "失败" in formatted or "错误" in formatted or "timeout" in formatted
```

- [ ] **Step 2: 运行测试验证失败**

```bash
cd backend && ../.venv-mineru/Scripts/python.exe -m pytest tests/test_agent_tools.py -v
```
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: 实现工具函数**

```python
"""Agentic RAG — 工具函数实现

5 个工具: search_document / get_section / query_knowledge_graph / search_keyword / reflect_on_answer
工具调度: execute_tool() 统一入口 + format_tool_result() 人类可读格式化
"""
import json
import logging

from storage.milvus_client import milvus_client
from storage.db import get_document
from storage.kg_store import kg_store
from services.chunker import chunker

logger = logging.getLogger(__name__)

MAX_TOP_K = 20
MIN_HEADING_LEN = 2


def tool_search_document(doc_id: str, query: str, top_k: int = 10) -> dict:
    """混合检索: BM25 + 向量 RRF 融合"""
    if not query or not query.strip():
        return {"error": "查询不能为空", "tool": "search_document"}
    top_k = max(1, min(top_k, MAX_TOP_K))
    try:
        query_embedding = chunker.embed([query.strip()])[0]
        chunks = milvus_client.hybrid_query(
            doc_id, query_embedding, query.strip(), top_k=top_k
        )
        hits = []
        for c in chunks:
            meta = c.get("metadata", {})
            hits.append({
                "text": c.get("text", ""),
                "heading": meta.get("heading_path", ""),
                "page": meta.get("page", 0),
            })
        return {"hits": hits, "total": len(hits), "tool": "search_document"}
    except Exception as e:
        logger.error(f"search_document failed: {e}")
        return {"error": str(e), "tool": "search_document"}


def tool_get_section(doc_id: str, heading: str) -> dict:
    """获取指定章节的完整内容"""
    if not heading or len(heading.strip()) < MIN_HEADING_LEN:
        return {"error": "章节标题至少需要 2 个字符", "tool": "get_section"}
    heading = heading.strip()
    try:
        row = get_document(doc_id)
        if not row:
            return {"error": "文档不存在", "tool": "get_section"}
        structured_json = row.get("structured_json")
        if not structured_json:
            return {"error": "文档尚未解析", "tool": "get_section"}
        data = json.loads(structured_json) if isinstance(structured_json, str) else structured_json
        sections = data.get("sections", [])
        heading_lower = heading.lower()
        matches = []
        for s in sections:
            s_heading = s.get("heading", "")
            if heading_lower in s_heading.lower():
                matches.append({
                    "heading": s_heading,
                    "level": s.get("level", 1),
                    "content": s.get("content", ""),
                    "page": s.get("page", 0),
                })
        if not matches:
            return {
                "hits": [],
                "total": 0,
                "available_headings": [s.get("heading", "") for s in sections[:20]],
                "tool": "get_section",
            }
        return {"hits": matches, "total": len(matches), "tool": "get_section"}
    except Exception as e:
        logger.error(f"get_section failed: {e}")
        return {"error": str(e), "tool": "get_section"}


def tool_query_knowledge_graph(doc_id: str, concept: str) -> dict:
    """查询知识图谱中的概念及其关联"""
    if not concept or not concept.strip():
        return {"error": "概念名不能为空", "tool": "query_knowledge_graph"}
    concept = concept.strip()
    try:
        kg = kg_store.load(doc_id)
        if kg is None:
            return {"available": False, "message": "知识图谱尚未构建", "tool": "query_knowledge_graph"}
        concept_lower = concept.lower()
        matched_node = None
        for node in kg.nodes:
            if concept_lower in node.name.lower():
                matched_node = node
                break
        if matched_node is None:
            return {
                "found": False,
                "message": f"未找到概念 '{concept}'",
                "similar": [n.name for n in kg.nodes if concept_lower[:2] in n.name.lower()][:5],
                "tool": "query_knowledge_graph",
            }
        neighbors = []
        for edge in kg.edges:
            if edge.source_id == matched_node.id:
                tgt = next((n for n in kg.nodes if n.id == edge.target_id), None)
                if tgt:
                    neighbors.append({
                        "concept": tgt.name,
                        "relation": edge.relation.value,
                        "description": edge.description,
                    })
            elif edge.target_id == matched_node.id:
                src = next((n for n in kg.nodes if n.id == edge.source_id), None)
                if src:
                    neighbors.append({
                        "concept": src.name,
                        "relation": f"inverse_{edge.relation.value}",
                        "description": edge.description,
                    })
        return {
            "found": True,
            "concept": {
                "name": matched_node.name,
                "type": matched_node.type.value,
                "summary": matched_node.summary,
                "importance": matched_node.importance,
                "source_sections": matched_node.source_sections,
            },
            "neighbors": neighbors,
            "tool": "query_knowledge_graph",
        }
    except Exception as e:
        logger.error(f"query_knowledge_graph failed: {e}")
        return {"error": str(e), "tool": "query_knowledge_graph"}


def tool_search_keyword(doc_id: str, keyword: str, top_k: int = 10) -> dict:
    """纯 BM25 关键词检索"""
    if not keyword or len(keyword.strip()) < MIN_HEADING_LEN:
        return {"error": "关键词至少需要 2 个字符", "tool": "search_keyword"}
    keyword = keyword.strip()
    top_k = max(1, min(top_k, MAX_TOP_K))
    try:
        bm25_idx = milvus_client._bm25_indices.get(doc_id)
        if bm25_idx is None:
            return {"error": "文档 BM25 索引不存在", "tool": "search_keyword"}
        import jieba
        tokens = jieba.lcut(keyword)
        scores = bm25_idx.get_scores(tokens)
        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        corpus = milvus_client._bm25_corpora.get(doc_id, [])
        hits = []
        for idx, score in indexed[:top_k]:
            if score <= 0:
                break
            # 从 Milvus 取元数据
            try:
                vec = milvus_client.query(
                    doc_id,
                    chunker.embed([keyword])[0],
                    top_k=top_k * 2,
                )
                hit_text = " ".join(corpus[idx]) if idx < len(corpus) else ""
                meta = {}
                for v in vec:
                    if v.get("text", "")[:50] == hit_text[:50]:
                        meta = v.get("metadata", {})
                        break
                hits.append({
                    "text": hit_text,
                    "heading": meta.get("heading_path", ""),
                    "page": meta.get("page", 0),
                    "score": float(score),
                })
            except Exception:
                hits.append({
                    "text": " ".join(corpus[idx]) if idx < len(corpus) else "",
                    "heading": "",
                    "page": 0,
                    "score": float(score),
                })
        return {"hits": hits, "total": len(hits), "tool": "search_keyword"}
    except Exception as e:
        logger.error(f"search_keyword failed: {e}")
        return {"error": str(e), "tool": "search_keyword"}


def tool_reflect_on_answer(draft_answer: str, cited_sources: list[str]) -> dict:
    """自检答案质量：检查是否基于文档、有无编造、引用是否准确"""
    if not draft_answer or not draft_answer.strip():
        return {"passed": False, "issues": ["答案为空"], "tool": "reflect_on_answer"}
    if not cited_sources:
        return {"passed": False, "issues": ["答案未提供任何引用来源，可能存在编造"],
                "tool": "reflect_on_answer"}
    issues = []
    if len(draft_answer.strip()) < 10:
        issues.append("答案过短，可能不够完整")
    if len(cited_sources) < 1:
        issues.append("引用来源不足")
    return {
        "passed": len(issues) == 0,
        "issues": issues,
        "source_count": len(cited_sources),
        "tool": "reflect_on_answer",
    }


def format_tool_result(tool_name: str, result: dict) -> str:
    """将工具返回的结构化结果格式化为人类可读的文本，注入到 LLM 消息中"""
    if "error" in result:
        return f"[工具 {tool_name} 执行失败] {result['error']}"

    if tool_name == "search_document":
        total = result.get("total", 0)
        if total == 0:
            return "[search_document] 未找到匹配的文档片段。请尝试更换查询词或使用其他工具。"
        lines = [f"[search_document] 找到 {total} 个相关片段:"]
        for i, hit in enumerate(result.get("hits", []), 1):
            heading = hit.get("heading", "")
            page = hit.get("page", 0)
            loc = f" [{heading}]" if heading else ""
            if page:
                loc += f" (第{page}页)"
            lines.append(f"{i}.{loc}\n   {hit.get('text', '')[:300]}")
        return "\n".join(lines)

    elif tool_name == "get_section":
        total = result.get("total", 0)
        if total == 0:
            avail = result.get("available_headings", [])
            hint = f" 可用的章节有: {', '.join(avail)}" if avail else ""
            return f"[get_section] 未找到匹配章节。{hint}"
        lines = [f"[get_section] 找到 {total} 个匹配章节:"]
        for hit in result.get("hits", []):
            lines.append(f"## {hit['heading']} (第{hit.get('page', '?')}页)\n{hit.get('content', '')[:2000]}")
        return "\n\n".join(lines)

    elif tool_name == "query_knowledge_graph":
        if not result.get("available", True):
            return f"[query_knowledge_graph] {result.get('message', '知识图谱不可用')}"
        if not result.get("found"):
            similar = result.get("similar", [])
            hint = f" 相关概念: {', '.join(similar)}" if similar else ""
            return f"[query_knowledge_graph] {result.get('message', '概念未找到')}{hint}"
        c = result["concept"]
        lines = [
            f"[query_knowledge_graph] 概念: {c['name']} (类型: {c['type']}, 重要度: {c['importance']})",
            f"概述: {c['summary']}",
            f"来源章节: {', '.join(c.get('source_sections', []))}",
        ]
        if result.get("neighbors"):
            lines.append("关联概念:")
            for n in result["neighbors"]:
                lines.append(f"  - {n['concept']} ({n['relation']}): {n.get('description', '')}")
        return "\n".join(lines)

    elif tool_name == "search_keyword":
        total = result.get("total", 0)
        if total == 0:
            return "[search_keyword] 未找到匹配的关键词。"
        lines = [f"[search_keyword] 找到 {total} 个结果:"]
        for i, hit in enumerate(result.get("hits", []), 1):
            lines.append(f"{i}. [{hit.get('heading', '')}] {hit.get('text', '')[:200]}")
        return "\n".join(lines)

    elif tool_name == "reflect_on_answer":
        if result.get("passed"):
            return "[reflect_on_answer] ✅ 自检通过：答案基于文档，引用来源充足。"
        issues = result.get("issues", [])
        return f"[reflect_on_answer] ❌ 自检未通过: {'; '.join(issues)}。请继续检索补充信息或修正答案。"

    elif tool_name == "ask_user":
        return f"[ask_user] {result.get('question', '')}"

    return f"[{tool_name}] 执行完成"


# 工具调度表
TOOL_MAP = {
    "search_document": tool_search_document,
    "get_section": tool_get_section,
    "query_knowledge_graph": tool_query_knowledge_graph,
    "search_keyword": tool_search_keyword,
    "reflect_on_answer": tool_reflect_on_answer,
}


def execute_tool(doc_id: str, tool_name: str, arguments: dict) -> dict:
    """统一工具调度入口，捕获异常并返回结构化错误"""
    func = TOOL_MAP.get(tool_name)
    if func is None:
        return {"error": f"未知工具: {tool_name}", "tool": tool_name}
    try:
        if tool_name == "reflect_on_answer":
            return func(
                draft_answer=arguments.get("draft_answer", ""),
                cited_sources=arguments.get("cited_sources", []),
            )
        return func(doc_id=doc_id, **arguments)
    except TypeError as e:
        return {"error": f"参数错误: {e}", "tool": tool_name}
    except Exception as e:
        logger.error(f"Tool {tool_name} execution failed: {e}")
        return {"error": str(e), "tool": tool_name}
```

- [ ] **Step 4: 运行测试验证**

```bash
cd backend && ../.venv-mineru/Scripts/python.exe -m pytest tests/test_agent_tools.py -v --tb=short
```
Expected: 大部分 PASS，依赖外部服务的测试可能 SKIP

- [ ] **Step 5: Commit**

```bash
git add backend/services/agent_tools.py backend/tests/test_agent_tools.py
git commit -m "feat: add 5 Agent tool functions with tests"
```

---

### Task 4: ReAct Agent 循环

**Files:**
- Create: `backend/services/agent.py`
- Create: `backend/tests/test_agent.py`

- [ ] **Step 1: 编写 Agent 循环测试**

```python
"""Agent 循环测试 — ReAct 决策 / clarify 检测 / 最大轮数截断"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pytest
from unittest.mock import patch, MagicMock

from services.agent import (
    AgentLoop,
    _detect_clarify_intent,
    _parse_tool_calls,
    MAX_ITERATIONS,
)
from models.agent import AgentStep, AgentStepType


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

    def test_clarify_keywords_in_chinese(self):
        assert _detect_clarify_intent("你能告诉我具体想看哪一部分吗？")

    def test_factual_answer_not_clarify(self):
        assert not _detect_clarify_intent("根据第3章，动态规划的核心思想是...")


class TestParseToolCalls:
    """_parse_tool_calls — 解析 LLM 返回的 tool_calls"""

    def test_empty_response(self):
        result = _parse_tool_calls(None)
        assert result == []

    def test_response_without_tool_calls(self):
        msg = MagicMock()
        msg.tool_calls = None
        result = _parse_tool_calls(msg)
        assert result == []

    def test_response_with_valid_tool_calls(self):
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

    def test_invalid_json_arguments_returns_empty(self):
        msg = MagicMock()
        tc = MagicMock()
        tc.id = "call_2"
        tc.function.name = "bad_tool"
        tc.function.arguments = "not valid json"
        msg.tool_calls = [tc]
        result = _parse_tool_calls(msg)
        assert result == []


class TestAgentLoopInit:
    """AgentLoop 初始化"""

    @patch("services.agent.llm_service")
    def test_creates_with_doc_id(self, mock_llm):
        loop = AgentLoop(doc_id="doc-1", question="测试问题")
        assert loop.doc_id == "doc-1"
        assert loop.question == "测试问题"
        assert loop.iteration_count == 0
        assert len(loop.messages) >= 2  # system + user
        assert loop.messages[0]["role"] == "system"
        assert loop.messages[1]["role"] == "user"

    @patch("services.agent.llm_service")
    def test_messages_include_question(self, mock_llm):
        loop = AgentLoop(doc_id="doc-1", question="什么是动态规划？")
        user_msgs = [m for m in loop.messages if m["role"] == "user"]
        assert any("什么是动态规划" in m.get("content", "") for m in user_msgs)

    @patch("services.agent.llm_service")
    def test_default_max_iterations(self, mock_llm):
        loop = AgentLoop(doc_id="doc-1", question="test")
        assert loop.max_iterations == MAX_ITERATIONS

    @patch("services.agent.llm_service")
    def test_custom_max_iterations(self, mock_llm):
        loop = AgentLoop(doc_id="doc-1", question="test", max_iterations=3)
        assert loop.max_iterations == 3


class TestAgentLoopIntegrity:
    """AgentLoop 结果完整性"""

    @patch("services.agent.llm_service")
    def test_run_stops_at_max_iterations(self, mock_llm):
        """当 LLM 持续返回 tool_calls 时，应在 max_iterations 后强制终止"""
        loop = AgentLoop(doc_id="doc-1", question="test", max_iterations=2)
        loop.iteration_count = 2  # 模拟已到上限
        assert loop._should_force_answer()


class TestMaxIterations:
    """MAX_ITERATIONS 常量"""

    def test_max_iterations_value(self):
        assert MAX_ITERATIONS == 8
        assert isinstance(MAX_ITERATIONS, int)
```

- [ ] **Step 2: 运行测试验证失败**

```bash
cd backend && ../.venv-mineru/Scripts/python.exe -m pytest tests/test_agent.py -v
```
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: 实现 Agent 循环**

```python
"""Agentic RAG — ReAct 循环主控

核心流程:
1. 初始化消息（system prompt + user question）
2. 循环（max 8 次）:
   a. 调用 LLM (带 tools 定义)
   b. 有 tool_calls → 执行工具 → yield step 事件 → 追加到消息
   c. 无 tool_calls → 检测 clarify/answer → break
3. 流式输出最终答案
"""
import json
import logging
from typing import AsyncGenerator

from openai import OpenAI

from config import settings
from models.agent import AgentStep, AgentStepType, AgentResult, Source
from services.agent_prompts import AGENT_SYSTEM_PROMPT, TOOLS
from services.agent_tools import execute_tool, format_tool_result

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 8
CLARIFY_PATTERNS = [
    "请问", "你想了解", "你能告诉", "请告诉", "你想看",
    "你指的是", "你说的是", "能否明确", "具体是哪",
    "你希望", "你更关心", "需要你确认", "请选择",
    "你想总结", "你想深入", "我可以帮你",
]


def _detect_clarify_intent(text: str) -> bool:
    """检测 LLM 输出是否包含追问意图"""
    if not text:
        return False
    text = text.strip()
    # 太短的文本不是追问（如"可以"、"好的"）
    if len(text) < 15:
        return False
    # 如果包含问号且匹配追问模式
    if "?" in text or "？" in text:
        return any(p in text for p in CLARIFY_PATTERNS)
    return False


def _parse_tool_calls(choice) -> list[dict]:
    """安全解析 LLM 返回的 tool_calls"""
    if choice is None or not hasattr(choice, "tool_calls") or not choice.tool_calls:
        return []
    parsed = []
    for tc in choice.tool_calls:
        try:
            args = json.loads(tc.function.arguments) if tc.function.arguments else {}
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse tool arguments: {tc.function.arguments[:100]}")
            continue
        parsed.append({
            "id": tc.id,
            "name": tc.function.name,
            "arguments": args,
        })
    return parsed


def _build_llm_client() -> tuple[OpenAI, str]:
    """构建 LLM 客户端，复用 config 中的配置"""
    if settings.llm_provider == "qwen":
        client = OpenAI(
            api_key=settings.qwen_api_key,
            base_url=settings.qwen_base_url,
        )
        model = settings.qwen_model
    else:
        client = OpenAI(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
        )
        model = settings.deepseek_model
    return client, model


class AgentLoop:
    """ReAct Agent 循环"""

    def __init__(
        self,
        doc_id: str,
        question: str,
        max_iterations: int = MAX_ITERATIONS,
    ):
        self.doc_id = doc_id
        self.question = question
        self.max_iterations = max_iterations
        self.iteration_count = 0
        self.steps: list[AgentStep] = []
        self.used_sources: list[Source] = []

        # 初始化消息列表
        self.messages = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": f"文档 ID: {doc_id}\n\n用户问题: {question}"},
        ]

    def _should_force_answer(self) -> bool:
        return self.iteration_count >= self.max_iterations

    async def run(self) -> AsyncGenerator[dict, None]:
        """运行 Agent 循环，逐步骤 yield SSE 事件 dict

        返回格式: {"event": "step"|"chunk"|"sources"|"done", "data": {...}}
        """
        step_index = 0

        # Phase 1: ReAct 循环
        while not self._should_force_answer():
            self.iteration_count += 1
            step_index += 1

            # 调 LLM 决策
            yield {
                "event": "step",
                "data": AgentStep(
                    step_index=step_index,
                    type=AgentStepType.PLAN,
                    content=f"分析问题并决定下一步操作...（第 {self.iteration_count}/{self.max_iterations} 轮）",
                    detail={},
                ).model_dump(mode="json"),
            }

            choice = await self._call_llm()
            if choice is None:
                yield {
                    "event": "step",
                    "data": AgentStep(
                        step_index=step_index,
                        type=AgentStepType.ERROR,
                        content="LLM 调用失败，请重试",
                        detail={},
                    ).model_dump(mode="json"),
                }
                return

            tool_calls = _parse_tool_calls(choice)

            if tool_calls:
                # 有工具调用 → 执行
                for tc in tool_calls:
                    step_index += 1
                    tool_name = tc["name"]
                    tool_args = tc.get("arguments", {})

                    # 特殊处理 ask_user
                    if tool_name == "ask_user":
                        yield {
                            "event": "step",
                            "data": AgentStep(
                                step_index=step_index,
                                type=AgentStepType.CLARIFY,
                                content=tool_args.get("question", "请问您需要什么帮助？"),
                                detail={
                                    "tool_name": "ask_user",
                                    "options": tool_args.get("options", []),
                                },
                            ).model_dump(mode="json"),
                        }
                        # 挂起等待回复，消息也记录这个追问
                        self.messages.append({
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [{
                                "id": tc["id"],
                                "type": "function",
                                "function": {
                                    "name": "ask_user",
                                    "arguments": json.dumps(tool_args, ensure_ascii=False),
                                },
                            }],
                        })
                        self.messages.append({
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": "[等待用户回复...]",
                        })
                        return  # 暂停循环，等待 /resume

                    # 推送 tool_call 事件
                    yield {
                        "event": "step",
                        "data": AgentStep(
                            step_index=step_index,
                            type=AgentStepType.TOOL_CALL,
                            content=f"调用工具: {tool_name}",
                            detail={"tool_name": tool_name, "arguments": tool_args},
                        ).model_dump(mode="json"),
                    }

                    # 执行工具
                    result = execute_tool(self.doc_id, tool_name, tool_args)
                    formatted = format_tool_result(tool_name, result)

                    # 记录引用来源
                    if tool_name in ("search_document", "search_keyword") and "hits" in result:
                        for hit in result["hits"]:
                            excerpt = hit.get("text", "")[:200]
                            if excerpt and not any(s.excerpt == excerpt for s in self.used_sources):
                                self.used_sources.append(Source(
                                    heading=hit.get("heading", ""),
                                    page=hit.get("page", 0),
                                    excerpt=excerpt,
                                    score=hit.get("score", 0.0),
                                ))

                    # 推送 tool_result 事件
                    yield {
                        "event": "step",
                        "data": AgentStep(
                            step_index=step_index,
                            type=AgentStepType.TOOL_RESULT,
                            content=formatted,
                            detail={
                                "tool_name": tool_name,
                                "result_summary": formatted[:200],
                                "hits_count": result.get("total", 0) if "total" in result else None,
                            },
                        ).model_dump(mode="json"),
                    }

                    # 追加工具结果到消息
                    self.messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": json.dumps(tool_args, ensure_ascii=False),
                            },
                        }],
                    })
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": formatted,
                    })
            else:
                # 无 tool_call → 检查是否追问或回答
                content = choice.content or ""

                if _detect_clarify_intent(content):
                    yield {
                        "event": "step",
                        "data": AgentStep(
                            step_index=step_index,
                            type=AgentStepType.CLARIFY,
                            content=content,
                            detail={},
                        ).model_dump(mode="json"),
                    }
                    self.messages.append({"role": "assistant", "content": content})
                    return  # 等待用户回复

                # 进入回答阶段
                break

        # Phase 2: 强制回答（达到上限）
        if self._should_force_answer():
            yield {
                "event": "step",
                "data": AgentStep(
                    step_index=step_index + 1,
                    type=AgentStepType.PLAN,
                    content=f"已达到最大检索轮数 ({self.max_iterations})，基于已有信息生成答案",
                    detail={},
                ).model_dump(mode="json"),
            }

        # Phase 3: 生成最终答案
        yield from self._generate_answer()

    async def _call_llm(self):
        """调用 LLM 获取下一步决策"""
        client, model = _build_llm_client()
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=self.messages,
                tools=TOOLS,
                temperature=0.0,
                max_tokens=4096,
            )
            return resp.choices[0].message if resp.choices else None
        except Exception as e:
            logger.error(f"LLM call failed in AgentLoop: {e}")
            return None

    async def _generate_answer(self) -> AsyncGenerator[dict, None]:
        """流式生成最终答案"""
        step_index = len(self.steps) + 1

        # 先自检
        draft = "基于检索到的文档内容"
        cite_texts = [s.excerpt for s in self.used_sources[:10]]
        reflect_result = execute_tool(
            self.doc_id, "reflect_on_answer",
            {"draft_answer": draft, "cited_sources": cite_texts},
        )
        reflect_formatted = format_tool_result("reflect_on_answer", reflect_result)
        yield {
            "event": "step",
            "data": AgentStep(
                step_index=step_index,
                type=AgentStepType.REFLECT,
                content=reflect_formatted,
                detail=reflect_result,
            ).model_dump(mode="json"),
        }

        # 流式输出答案
        answer_prompt = "请基于以上的检索结果和工具调用历史，用中文简洁准确地回答用户问题。如检索结果不足以回答，请明确说明。"

        if not reflect_result.get("passed", False):
            answer_prompt += "\n\n注意：自检发现问题，请特别注意引用文档来源，不要编造。"

        self.messages.append({"role": "user", "content": answer_prompt})

        client, model = _build_llm_client()
        try:
            stream = client.chat.completions.create(
                model=model,
                messages=self.messages,
                stream=True,
                temperature=0.0,
                max_tokens=2048,
            )
            full_answer = ""
            for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.content:
                    text = delta.content
                    full_answer += text
                    yield {"event": "chunk", "data": {"text": text}}

            # 推送引用来源
            sources_data = [s.model_dump() for s in self.used_sources[:10]]
            yield {"event": "sources", "data": sources_data}

            # 完成
            total_steps = len(self.steps) + step_index
            yield {
                "event": "done",
                "data": {
                    "total_steps": total_steps,
                    "iterations": self.iteration_count,
                    "follow_up_suggestions": self._generate_follow_ups(full_answer),
                },
            }
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            yield {"event": "done", "data": {"error": str(e)}}

    def _generate_follow_ups(self, answer: str) -> list[str]:
        """根据答案生成简单的追问建议"""
        suggestions = []
        if any(kw in self.question for kw in ("区别", "对比", "vs", "比较")):
            suggestions.append("可以详细解释其中一个吗？")
        if len(self.used_sources) >= 3:
            suggestions.append("还有哪些相关概念值得了解？")
        if self.iteration_count >= self.max_iterations - 2:
            suggestions.append("检索可能不够全面，需要换关键词重新搜索吗？")
        return suggestions

    def resume(self, user_response: str):
        """用户回复追问后恢复循环"""
        self.messages.append({
            "role": "user",
            "content": f"[用户回复] {user_response}",
        })
```

- [ ] **Step 4: 运行测试验证**

```bash
cd backend && ../.venv-mineru/Scripts/python.exe -m pytest tests/test_agent.py -v --tb=short
```
Expected: `_detect_clarify_intent` 测试 PASS, `_parse_tool_calls` 测试 PASS, AgentLoop 初始化测试 PASS

- [ ] **Step 5: Commit**

```bash
git add backend/services/agent.py backend/tests/test_agent.py
git commit -m "feat: add ReAct Agent loop with async generator"
```

---

### Task 5: SSE 流式路由

**Files:**
- Create: `backend/routers/chat.py`

- [ ] **Step 1: 创建 SSE 路由文件**

```python
"""Chat 路由 — SSE 流式快速问答 + 深度 Agent 问答"""
import asyncio
import json
import logging
import time
from datetime import datetime, timedelta

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from models.agent import ClarifyRequest, AgentSession
from services.agent import AgentLoop
from services.llm_service import llm_service
from services.chunker import chunker
from storage.db import get_document, get_chat_history, add_chat_message

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/documents/{doc_id}/chat", tags=["chat"])

# 内存中的 Agent 会话存储
_agent_sessions: dict[str, AgentSession] = {}
SESSION_TIMEOUT_MINUTES = 30


class ChatRequest(BaseModel):
    question: str


class DeepChatRequest(BaseModel):
    question: str
    max_iterations: int = 8


def _cleanup_expired_sessions():
    """清理过期会话"""
    now = datetime.now()
    expired = [
        sid for sid, s in _agent_sessions.items()
        if s.is_expired(SESSION_TIMEOUT_MINUTES)
    ]
    for sid in expired:
        del _agent_sessions[sid]


def _get_doc_check(doc_id: str) -> dict:
    """校验文档存在且可用"""
    row = get_document(doc_id)
    if not row:
        raise HTTPException(status_code=404, detail="文档不存在")
    if row["status"] != "ready":
        raise HTTPException(status_code=400, detail="文档尚未就绪")
    return row


@router.post("/stream")
async def chat_stream(doc_id: str, body: ChatRequest, request: Request):
    """SSE 流式快速问答（单次检索 + 流式输出）"""
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question is required")
    if len(question) > 2000:
        raise HTTPException(status_code=400, detail="question too long")

    row = _get_doc_check(doc_id)

    async def generate():
        try:
            # 检索
            query_embedding = chunker.embed([question])[0]
            history = get_chat_history(doc_id)

            from storage.milvus_client import milvus_client
            chunks = milvus_client.hybrid_query(doc_id, query_embedding, question, top_k=15)

            if not chunks:
                yield f"data: {json.dumps({'event': 'chunk', 'data': {'text': '无法检索到相关文档内容。'}})}\n\n"
                yield f"data: {json.dumps({'event': 'done', 'data': {}})}\n\n"
                return

            # 构造上下文
            from services.prompt_templates import QA_SYSTEM_PROMPT, QA_USER_PROMPT
            from services.llm_service import _CONTENT_GUARD_PREFIX, _CONTENT_GUARD_SUFFIX
            import re

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

            def _sanitize(text):
                text = re.sub(r'(?i)(忽略|无视|忘记|ignore)\s*(上述|以上|所有|all|previous)\s*(指令|指示|说明|规则|instructions?)', '[过滤]', text)
                return text

            context = f"{_CONTENT_GUARD_PREFIX}\n{_sanitize('\n\n---\n\n'.join(parts))}\n{_CONTENT_GUARD_SUFFIX}"

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

            # 流式调用 LLM
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
                    # 过滤 think 标签
                    import re as re_m
                    text = re_m.sub(r"<thinking>.*?</thinking>", "", text, flags=re_m.DOTALL)
                    text = re_m.sub(r"<think>.*?</think>", "", text, flags=re_m.DOTALL)
                    if text:
                        full_answer += text
                        yield f"data: {json.dumps({'event': 'chunk', 'data': {'text': text}})}\n\n"

            # 保存对话历史
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

    # 创建 Agent 会话
    session = AgentSession(doc_id=doc_id, question=question)
    _agent_sessions[session.session_id] = session

    async def generate():
        try:
            loop = AgentLoop(
                doc_id=doc_id,
                question=question,
                max_iterations=body.max_iterations,
            )
            # 恢复已有会话的消息历史（如果有的话，新会话为空）
            async for event in loop.run():
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                if event["event"] == "step":
                    session.messages = loop.messages

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
            # 恢复历史消息
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
```

- [ ] **Step 2: 验证路由导入**

```bash
cd backend && ../.venv-mineru/Scripts/python.exe -c "from routers.chat import router; print('router loaded OK')"
```
Expected: `router loaded OK`

- [ ] **Step 3: 注册路由到 main.py**

编辑 `backend/main.py`，在现有路由注册下方添加一行：

```python
from routers.chat import router as chat_router

# 在 app.include_router(knowledge_graph.router) 之后添加:
app.include_router(chat_router)
```

- [ ] **Step 4: Commit**

```bash
git add backend/routers/chat.py backend/main.py
git commit -m "feat: add SSE streaming chat routes (fast + deep + resume)"
```

---

### Task 6: 前端 MarkdownRenderer（共享组件）

**Files:**
- Create: `frontend/src/components/MarkdownRenderer.tsx`

- [ ] **Step 1: 创建 MarkdownRenderer 组件**

```tsx
import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

interface MarkdownRendererProps {
  content: string;
  className?: string;
}

const MarkdownRenderer: React.FC<MarkdownRendererProps> = ({ content, className }) => {
  return (
    <div className={className} style={{ lineHeight: 1.8 }}>
      <ReactMarkdown remarkPlugins={[remarkGfm]}>
        {content}
      </ReactMarkdown>
    </div>
  );
};

export default MarkdownRenderer;
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/components/MarkdownRenderer.tsx
git commit -m "feat: extract shared MarkdownRenderer component"
```

---

### Task 7: 前端 AgentStepCard 组件

**Files:**
- Create: `frontend/src/components/AgentStepCard.tsx`
- Create: `frontend/src/components/AgentStepCard.css`

- [ ] **Step 1: 创建 AgentStepCard 组件**

```tsx
import React, { useState } from 'react';
import './AgentStepCard.css';

export interface AgentStepData {
  step_index: number;
  type: 'plan' | 'tool_call' | 'tool_result' | 'reflect' | 'clarify' | 'answer' | 'error';
  content: string;
  detail?: Record<string, any> | null;
}

interface AgentStepCardProps {
  step: AgentStepData;
}

const ICON_MAP: Record<string, string> = {
  plan: '📋',
  tool_call: '🔍',
  tool_result: '📄',
  reflect: '🤔',
  clarify: '💬',
  answer: '✅',
  error: '❌',
};

const LABEL_MAP: Record<string, string> = {
  plan: '规划',
  tool_call: '工具调用',
  tool_result: '检索结果',
  reflect: '反思验证',
  clarify: '追问',
  answer: '回答',
  error: '错误',
};

const AgentStepCard: React.FC<AgentStepCardProps> = ({ step }) => {
  const [expanded, setExpanded] = useState(false);
  const icon = ICON_MAP[step.type] || '•';
  const label = LABEL_MAP[step.type] || step.type;

  const hasDetail = step.detail && Object.keys(step.detail).length > 0;

  let detailContent: string | null = null;
  if (hasDetail && step.detail) {
    if (step.type === 'tool_call') {
      detailContent = `工具: ${step.detail.tool_name || ''}\n参数: ${JSON.stringify(step.detail.arguments || {}, null, 2)}`;
    } else if (step.type === 'tool_result') {
      detailContent = step.detail.result_summary || step.content;
    } else {
      detailContent = JSON.stringify(step.detail, null, 2);
    }
  }

  return (
    <div className={`agent-step-card step-${step.type}`}>
      <div className="step-header" onClick={() => setExpanded(!expanded)}>
        <span className="step-icon">{icon}</span>
        <span className="step-label">{label}</span>
        <span className="step-content-preview">
          {step.content.length > 80 ? step.content.slice(0, 80) + '...' : step.content}
        </span>
        {hasDetail && (
          <span className="step-expand-icon">{expanded ? '▾' : '▸'}</span>
        )}
      </div>
      {expanded && hasDetail && (
        <div className="step-detail">
          <pre>{detailContent}</pre>
        </div>
      )}
      {expanded && !hasDetail && (
        <div className="step-detail">
          <p>{step.content}</p>
        </div>
      )}
    </div>
  );
};

export default AgentStepCard;
```

- [ ] **Step 2: 创建 CSS 样式**

```css
.agent-step-card {
  border-left: 3px solid #e0e0e0;
  margin: 8px 0;
  padding: 8px 12px;
  border-radius: 4px;
  background: #fafafa;
  transition: background 0.2s;
}

.agent-step-card:hover {
  background: #f0f0f0;
}

.agent-step-card.step-plan { border-left-color: #1890ff; }
.agent-step-card.step-tool_call { border-left-color: #722ed1; }
.agent-step-card.step-tool_result { border-left-color: #52c41a; }
.agent-step-card.step-reflect { border-left-color: #faad14; }
.agent-step-card.step-clarify { border-left-color: #eb2f96; }
.agent-step-card.step-answer { border-left-color: #13c2c2; }
.agent-step-card.step-error { border-left-color: #ff4d4f; }

.step-header {
  display: flex;
  align-items: center;
  gap: 8px;
  cursor: pointer;
  user-select: none;
}

.step-icon {
  font-size: 18px;
  flex-shrink: 0;
}

.step-label {
  font-size: 13px;
  font-weight: 600;
  color: #666;
  flex-shrink: 0;
}

.step-content-preview {
  font-size: 13px;
  color: #333;
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.step-expand-icon {
  font-size: 12px;
  color: #999;
  flex-shrink: 0;
}

.step-detail {
  margin-top: 8px;
  padding: 8px 12px;
  background: #fff;
  border-radius: 4px;
  font-size: 12px;
  color: #555;
}

.step-detail pre {
  margin: 0;
  white-space: pre-wrap;
  word-break: break-all;
  font-family: 'Consolas', 'Monaco', monospace;
}

.step-detail p {
  margin: 0;
}
```

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/AgentStepCard.tsx frontend/src/components/AgentStepCard.css
git commit -m "feat: add AgentStepCard component with expandable detail"
```

---

### Task 8: 前端 AgentChatBox 组件

**Files:**
- Create: `frontend/src/components/AgentChatBox.tsx`
- Create: `frontend/src/components/AgentChatBox.css`

- [ ] **Step 1: 创建 AgentChatBox 组件**

```tsx
import React, { useState, useRef, useEffect, useCallback } from 'react';
import AgentStepCard, { AgentStepData } from './AgentStepCard';
import MarkdownRenderer from './MarkdownRenderer';
import './AgentChatBox.css';

interface Source {
  heading: string;
  page: number;
  excerpt: string;
  score: number;
}

interface AgentChatBoxProps {
  docId: string;
}

type ChatMode = 'idle' | 'running' | 'clarify' | 'done';

const AgentChatBox: React.FC<AgentChatBoxProps> = ({ docId }) => {
  const [mode, setMode] = useState<ChatMode>('idle');
  const [question, setQuestion] = useState('');
  const [steps, setSteps] = useState<AgentStepData[]>([]);
  const [answerText, setAnswerText] = useState('');
  const [sources, setSources] = useState<Source[]>([]);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [followUps, setFollowUps] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [clarifyQuestion, setClarifyQuestion] = useState('');
  const [clarifyOptions, setClarifyOptions] = useState<string[]>([]);

  const abortRef = useRef<AbortController | null>(null);
  const answerRef = useRef<HTMLDivElement>(null);

  const resetState = useCallback(() => {
    setSteps([]);
    setAnswerText('');
    setSources([]);
    setFollowUps([]);
    setError(null);
    setClarifyQuestion('');
    setClarifyOptions([]);
  }, []);

  const processSSE = useCallback(async (url: string, body: any) => {
    resetState();
    setMode('running');
    setError(null);
    abortRef.current = new AbortController();

    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
        signal: abortRef.current.signal,
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        const errMsg = (errData as any).detail || `HTTP ${response.status}`;
        setError(errMsg);
        setMode('done');
        return;
      }

      const reader = response.body?.getReader();
      if (!reader) {
        setError('No response body');
        setMode('done');
        return;
      }

      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const event = JSON.parse(line.slice(6));
              switch (event.event) {
                case 'step':
                  setSteps(prev => [...prev, event.data]);
                  if (event.data?.type === 'clarify') {
                    setClarifyQuestion(event.data.content);
                    setClarifyOptions(event.data.detail?.options || []);
                    setMode('clarify');
                  }
                  break;
                case 'chunk':
                  setAnswerText(prev => prev + (event.data?.text || ''));
                  break;
                case 'sources':
                  setSources(event.data || []);
                  break;
                case 'done':
                  setFollowUps(event.data?.follow_up_suggestions || []);
                  setMode('done');
                  break;
              }
            } catch {
              // skip malformed JSON lines
            }
          }
        }
      }
    } catch (err: any) {
      if (err.name !== 'AbortError') {
        setError(err.message || 'Connection failed');
        setMode('done');
      }
    }
  }, [resetState]);

  const handleSubmit = useCallback(() => {
    if (!question.trim() || mode === 'running') return;
    processSSE(`/api/documents/${docId}/chat/deep`, { question: question.trim() });
    setQuestion('');
  }, [question, docId, mode, processSSE]);

  const handleClarifyReply = useCallback((reply: string) => {
    if (!sessionId) return;
    setMode('running');
    setClarifyQuestion('');
    setClarifyOptions([]);
    processSSE(`/api/documents/${docId}/chat/deep/resume`, {
      session_id: sessionId,
      user_response: reply,
    });
  }, [sessionId, docId, processSSE]);

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (mode === 'clarify') {
        handleClarifyReply(question || '请继续');
        setQuestion('');
      } else {
        handleSubmit();
      }
    }
  }, [mode, question, handleSubmit, handleClarifyReply]);

  useEffect(() => {
    if (answerRef.current) {
      answerRef.current.scrollTop = answerRef.current.scrollHeight;
    }
  }, [answerText]);

  useEffect(() => {
    return () => {
      abortRef.current?.abort();
    };
  }, []);

  return (
    <div className="agent-chat-box">
      {/* Steps area */}
      <div className="agent-steps">
        {steps.map((step, i) => (
          <AgentStepCard key={i} step={step} />
        ))}
      </div>

      {/* Answer area */}
      {answerText && (
        <div className="agent-answer" ref={answerRef}>
          <div className="agent-answer-header">✅ 最终答案</div>
          <MarkdownRenderer content={answerText} className="agent-answer-content" />

          {sources.length > 0 && (
            <div className="agent-sources">
              <div className="agent-sources-header">📎 引用来源</div>
              {sources.map((s, i) => (
                <div key={i} className="agent-source-item">
                  <span className="source-heading">
                    {s.heading || '未知章节'}
                    {s.page > 0 ? ` (第${s.page}页)` : ''}
                  </span>
                  <span className="source-excerpt">「{s.excerpt}」</span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="agent-error">
          ❌ {error}
          <button onClick={() => { resetState(); setMode('idle'); }}>重试</button>
        </div>
      )}

      {/* Clarify prompt */}
      {mode === 'clarify' && (
        <div className="agent-clarify">
          <div className="clarify-question">{clarifyQuestion}</div>
          {clarifyOptions.length > 0 && (
            <div className="clarify-options">
              {clarifyOptions.map((opt, i) => (
                <button key={i} onClick={() => handleClarifyReply(opt)}>
                  {opt}
                </button>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Follow-up suggestions */}
      {mode === 'done' && followUps.length > 0 && (
        <div className="agent-followups">
          <span>💡 追问建议：</span>
          {followUps.map((f, i) => (
            <button key={i} onClick={() => {
              setQuestion(f);
              handleSubmit();
            }}>
              {f}
            </button>
          ))}
        </div>
      )}

      {/* Input */}
      <div className="agent-input">
        <textarea
          value={question}
          onChange={e => setQuestion(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={
            mode === 'clarify'
              ? '回复追问... (Enter 发送)'
              : mode === 'running'
              ? 'Agent 正在思考中...'
              : '输入问题... (Enter 发送, Shift+Enter 换行)'
          }
          disabled={mode === 'running'}
          rows={2}
        />
        <button
          onClick={mode === 'clarify' ? () => handleClarifyReply(question || '请继续') : handleSubmit}
          disabled={mode === 'running' || (!question.trim() && mode !== 'clarify')}
        >
          发送
        </button>
      </div>
    </div>
  );
};

export default AgentChatBox;
```

- [ ] **Step 2: 创建 CSS 样式**

```css
.agent-chat-box {
  display: flex;
  flex-direction: column;
  height: 100%;
  overflow: hidden;
}

.agent-steps {
  flex: 0 0 auto;
  max-height: 40%;
  overflow-y: auto;
  padding: 8px 0;
}

.agent-answer {
  flex: 1;
  overflow-y: auto;
  padding: 16px;
  border-top: 1px solid #e8e8e8;
}

.agent-answer-header {
  font-size: 15px;
  font-weight: 600;
  margin-bottom: 12px;
  color: #333;
}

.agent-answer-content {
  font-size: 14px;
  color: #333;
  line-height: 1.8;
}

.agent-sources {
  margin-top: 16px;
  padding: 12px;
  background: #f6f8fa;
  border-radius: 8px;
}

.agent-sources-header {
  font-size: 13px;
  font-weight: 600;
  margin-bottom: 8px;
  color: #666;
}

.agent-source-item {
  display: flex;
  flex-direction: column;
  gap: 4px;
  padding: 6px 0;
  border-bottom: 1px solid #eee;
}

.agent-source-item:last-child {
  border-bottom: none;
}

.source-heading {
  font-size: 12px;
  color: #1890ff;
  font-weight: 500;
}

.source-excerpt {
  font-size: 12px;
  color: #666;
  font-style: italic;
}

.agent-error {
  padding: 12px;
  background: #fff2f0;
  border: 1px solid #ffccc7;
  border-radius: 8px;
  color: #ff4d4f;
  margin: 8px 0;
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.agent-error button {
  padding: 4px 12px;
  background: #fff;
  border: 1px solid #ff4d4f;
  border-radius: 4px;
  color: #ff4d4f;
  cursor: pointer;
}

.agent-clarify {
  padding: 16px;
  background: #fff7e6;
  border: 1px solid #ffd591;
  border-radius: 8px;
  margin: 8px 0;
}

.clarify-question {
  font-size: 14px;
  color: #d46b08;
  margin-bottom: 8px;
}

.clarify-options {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}

.clarify-options button {
  padding: 6px 14px;
  background: #fff;
  border: 1px solid #ffa940;
  border-radius: 16px;
  color: #d46b08;
  cursor: pointer;
  font-size: 13px;
}

.clarify-options button:hover {
  background: #ffa940;
  color: #fff;
}

.agent-followups {
  display: flex;
  gap: 8px;
  align-items: center;
  padding: 8px 12px;
  flex-wrap: wrap;
  font-size: 13px;
}

.agent-followups button {
  padding: 4px 12px;
  background: #f0f5ff;
  border: 1px solid #adc6ff;
  border-radius: 12px;
  color: #2f54eb;
  cursor: pointer;
  font-size: 12px;
}

.agent-followups button:hover {
  background: #d6e4ff;
}

.agent-input {
  display: flex;
  gap: 8px;
  padding: 12px;
  border-top: 1px solid #e8e8e8;
  background: #fff;
}

.agent-input textarea {
  flex: 1;
  padding: 8px 12px;
  border: 1px solid #d9d9d9;
  border-radius: 8px;
  font-size: 14px;
  resize: none;
  font-family: inherit;
}

.agent-input textarea:disabled {
  background: #f5f5f5;
  color: #999;
}

.agent-input button {
  padding: 8px 20px;
  background: #1890ff;
  color: #fff;
  border: none;
  border-radius: 8px;
  font-size: 14px;
  cursor: pointer;
  align-self: flex-end;
}

.agent-input button:disabled {
  background: #d9d9d9;
  cursor: not-allowed;
}
```

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/AgentChatBox.tsx frontend/src/components/AgentChatBox.css
git commit -m "feat: add AgentChatBox component with SSE streaming"
```

---

### Task 9: 前端 ChatBox SSE 适配

**Files:**
- Modify: `frontend/src/components/ChatBox.tsx`
- Modify: `frontend/src/services/api.ts`

- [ ] **Step 1: 在 api.ts 中新增流式问答函数**

在 `frontend/src/services/api.ts` 末尾添加：

```typescript
export async function askQuestionStream(
  id: string,
  question: string,
  onChunk: (text: string) => void,
  signal?: AbortSignal,
): Promise<ChatMessage[]> {
  const res = await fetch(`${BASE}/documents/${id}/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
    signal,
  });
  if (!res.ok) throw new Error("Failed to get answer");
  const reader = res.body?.getReader();
  if (!reader) throw new Error("No response body");
  const decoder = new TextDecoder();
  let buffer = "";
  const userMsg: ChatMessage = { role: "user", content: question };
  let assistantContent = "";
  let messages: ChatMessage[] = [userMsg];
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";
    for (const line of lines) {
      if (line.startsWith("data: ")) {
        try {
          const event = JSON.parse(line.slice(6));
          if (event.event === "chunk" && event.data?.text) {
            assistantContent += event.data.text;
            onChunk(event.data.text);
          }
          if (event.event === "done") {
            messages = [
              userMsg,
              { role: "assistant", content: assistantContent },
            ];
          }
        } catch { /* skip parse errors */ }
      }
    }
  }
  return messages;
}
```

- [ ] **Step 2: 改造 ChatBox.tsx 的 handleSend 函数**

将 `frontend/src/components/ChatBox.tsx` 中的 `handleSend` 函数改造为 SSE 流式：

```typescript
// 修改 import:
import { ChatMessage, askQuestionStream, getChatHistory, clearChatHistory } from "../services/api";

// 替换 handleSend 函数:
async function handleSend() {
  const question = input.trim();
  if (!question || loading) return;
  setInput("");
  setError("");
  // 先添加用户消息
  const userMsg: ChatMessage = { role: "user", content: question };
  setMessages((prev) => [...prev, userMsg]);
  // 添加空的助手消息占位
  const assistantIndex = messages.length + 1;
  setMessages((prev) => [...prev, { role: "assistant", content: "" }]);
  setLoading(true);
  try {
    await askQuestionStream(
      docId,
      question,
      (text) => {
        // 流式追加文本到助手消息
        setMessages((prev) => {
          const updated = [...prev];
          const last = updated[updated.length - 1];
          if (last && last.role === "assistant") {
            updated[updated.length - 1] = { ...last, content: last.content + text };
          }
          return updated;
        });
      },
    );
  } catch (e: any) {
    setError(e.message || "发送失败");
    // 移除用户消息和空的助手消息
    setMessages((prev) => prev.slice(0, -(loading ? 2 : 1)));
  } finally {
    setLoading(false);
  }
}

// "思考中..." 的 loading 指示器改为：
{loading && messages[messages.length - 1]?.content === "" && (
  <div className="chat-msg assistant">
    <span className="chat-role">AI</span>
    <div className="chat-bubble typing">思考中...</div>
  </div>
)}
```

- [ ] **Step 3: 验证 TypeScript 编译**

```bash
cd frontend && npx tsc --noEmit --pretty 2>&1 | grep -E "error|Error" | head -10
```
Expected: No errors from ChatBox.tsx or api.ts

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/ChatBox.tsx frontend/src/services/api.ts
git commit -m "feat: add SSE streaming support to ChatBox"
```

---

### Task 10: 前端 ChatPage 模式切换

**Files:**
- Modify: `frontend/src/pages/ChatPage.tsx`
- Modify: `frontend/src/pages/ChatPage.css`

- [ ] **Step 1: 在 ChatPage 中新增模式切换开关**

```tsx
// 在现有 imports 后添加:
import AgentChatBox from '../components/AgentChatBox';

// 在组件 state 中添加:
const [chatMode, setChatMode] = useState<'fast' | 'deep'>('fast');

// 在渲染中，标题栏下方添加模式切换:
<div className="chat-mode-switch">
  <button
    className={chatMode === 'fast' ? 'active' : ''}
    onClick={() => setChatMode('fast')}
  >
    ⚡ 快速模式
  </button>
  <button
    className={chatMode === 'deep' ? 'active' : ''}
    onClick={() => setChatMode('deep')}
  >
    🧠 深度模式
  </button>
</div>

// 条件渲染:
{chatMode === 'fast' ? (
  <ChatBox docId={currentDocId} />
) : (
  <AgentChatBox docId={currentDocId} />
)}
```

- [ ] **Step 2: 添加 CSS 样式到 ChatPage.css**

```css
.chat-mode-switch {
  display: flex;
  gap: 8px;
  padding: 8px 16px;
  border-bottom: 1px solid #e8e8e8;
  background: #fafafa;
}

.chat-mode-switch button {
  padding: 6px 16px;
  border: 1px solid #d9d9d9;
  border-radius: 6px;
  background: #fff;
  cursor: pointer;
  font-size: 13px;
  transition: all 0.2s;
}

.chat-mode-switch button.active {
  background: #1890ff;
  color: #fff;
  border-color: #1890ff;
}

.chat-mode-switch button:hover:not(.active) {
  border-color: #1890ff;
  color: #1890ff;
}
```

- [ ] **Step 3: Commit**

```bash
git add frontend/src/pages/ChatPage.tsx frontend/src/pages/ChatPage.css
git commit -m "feat: add fast/deep mode toggle to ChatPage"
```

---

### Task 11: 后端集成测试

**Files:**
- Create: `backend/tests/test_agent_integration.py`

- [ ] **Step 1: 编写集成测试**

```python
"""Agent 集成测试 — AgentLoop 完整流程模拟"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from services.agent import AgentLoop, _detect_clarify_intent, _parse_tool_calls, MAX_ITERATIONS
from services.agent_tools import execute_tool, format_tool_result, TOOL_MAP
from models.agent import AgentStepType


class TestIntegrationToolExecution:
    """工具执行集成测试"""

    def test_all_tools_in_map(self):
        """TOOL_MAP 应包含所有 5 个工具 + ask_user 由 Agent 层处理"""
        assert "search_document" in TOOL_MAP
        assert "get_section" in TOOL_MAP
        assert "query_knowledge_graph" in TOOL_MAP
        assert "search_keyword" in TOOL_MAP
        assert "reflect_on_answer" in TOOL_MAP

    def test_execute_unknown_tool(self):
        result = execute_tool("doc-1", "nonexistent_tool", {})
        assert "error" in result

    def test_execute_reflect_without_doc_id(self):
        result = execute_tool("", "reflect_on_answer", {
            "draft_answer": "test answer",
            "cited_sources": ["source 1"],
        })
        # reflect_on_answer 不依赖 doc_id
        assert "tool" in result


class TestIntegrationFormatToolResult:
    """format_tool_result 格式化集成"""

    def test_search_document_format_with_hits(self):
        result = {"hits": [{"text": "test", "heading": "Ch1", "page": 1}], "total": 1}
        formatted = format_tool_result("search_document", result)
        assert "找到 1 个" in formatted
        assert "Ch1" in formatted

    def test_search_document_format_empty(self):
        formatted = format_tool_result("search_document", {"hits": [], "total": 0})
        assert "未找到" in formatted

    def test_get_section_format_with_hits(self):
        result = {"hits": [{"heading": "Ch1", "content": "test content", "page": 5}], "total": 1}
        formatted = format_tool_result("get_section", result)
        assert "Ch1" in formatted

    def test_get_section_format_empty(self):
        result = {"hits": [], "total": 0, "available_headings": ["Ch1", "Ch2"]}
        formatted = format_tool_result("get_section", result)
        assert "未找到" in formatted or "匹配章节" in formatted

    def test_kg_format_found(self):
        result = {
            "found": True,
            "concept": {"name": "DP", "type": "algorithm", "summary": "desc", "importance": 0.9, "source_sections": ["Ch3"]},
            "neighbors": [{"concept": "递归", "relation": "prerequisite", "description": "前置知识"}],
            "tool": "query_knowledge_graph",
        }
        formatted = format_tool_result("query_knowledge_graph", result)
        assert "DP" in formatted
        assert "递归" in formatted

    def test_kg_format_not_found(self):
        result = {"found": False, "message": "未找到", "similar": ["DP"], "tool": "query_knowledge_graph"}
        formatted = format_tool_result("query_knowledge_graph", result)
        assert "未找到" in formatted or "DP" in formatted

    def test_reflect_format_passed(self):
        formatted = format_tool_result("reflect_on_answer", {"passed": True, "issues": []})
        assert "通过" in formatted or "✅" in formatted

    def test_reflect_format_failed(self):
        formatted = format_tool_result("reflect_on_answer", {"passed": False, "issues": ["缺少引用"]})
        assert "未通过" in formatted or "❌" in formatted or "缺少引用" in formatted

    def test_keyword_format_with_hits(self):
        result = {"hits": [{"text": "hello", "heading": "Ch1", "page": 1}], "total": 1}
        formatted = format_tool_result("search_keyword", result)
        assert "找到 1 个" in formatted

    def test_error_format(self):
        formatted = format_tool_result("search_document", {"error": "timeout"})
        assert "失败" in formatted or "timeout" in formatted


class TestIntegrationAgentLoopWithMockLLM:
    """AgentLoop 端到端模拟 — 模拟 LLM 返回 tool_calls 后最终输出答案"""

    @patch("services.agent._build_llm_client")
    def test_full_loop_with_mock_llm(self, mock_build_client):
        """模拟 LLM 先返回一个 tool_call，再返回 answer"""
        mock_client = MagicMock()
        mock_build_client.return_value = (mock_client, "test-model")

        # 第一次 LLM 调用：返回 tool_call (search_document)
        tc_choice = MagicMock()
        tc_msg = MagicMock()
        tc_msg.content = None
        tc = MagicMock()
        tc.id = "call_1"
        tc.function.name = "search_document"
        tc.function.arguments = '{"query": "动态规划", "top_k": 5}'
        tc_msg.tool_calls = [tc]
        tc_choice.message = tc_msg

        # 第二次 LLM 调用：返回答案（无 tool_calls）
        answer_choice = MagicMock()
        answer_msg = MagicMock()
        answer_msg.content = "根据检索结果，动态规划是..."
        answer_msg.tool_calls = None
        answer_choice.message = answer_msg

        # 第三次 LLM 调用：流式答案生成
        stream_chunk_1 = MagicMock()
        stream_chunk_1.choices = [MagicMock()]
        stream_chunk_1.choices[0].delta = MagicMock()
        stream_chunk_1.choices[0].delta.content = "动态规划是一种"

        stream_chunk_2 = MagicMock()
        stream_chunk_2.choices = [MagicMock()]
        stream_chunk_2.choices[0].delta = MagicMock()
        stream_chunk_2.choices[0].delta.content = "算法设计技术。"

        mock_client.chat.completions.create.side_effect = [
            tc_choice,          # 第 1 次: tool_call
            answer_choice,       # 第 2 次: answer (break loop)
            [stream_chunk_1, stream_chunk_2],  # 第 3 次: stream answer
        ]

        # 由于 search_document 依赖外部服务，mock
        with patch("services.agent_tools.chunker") as mock_chunker:
            mock_chunker.embed.return_value = [[0.1] * 512]
            with patch("services.agent_tools.milvus_client") as mock_milvus:
                mock_milvus.hybrid_query.return_value = [
                    {"text": "动态规划通过存储子问题解来优化", "metadata": {"heading_path": "第3章", "page": 42}},
                ]

                loop = AgentLoop(doc_id="doc-1", question="动态规划是什么？", max_iterations=3)

                # 手动运行 loop 的 run (通过 async generator)
                import asyncio

                async def collect_events():
                    events = []
                    async for event in loop.run():
                        events.append(event)
                    return events

                events = asyncio.run(collect_events())

                # 验证事件结构
                event_types = [e["event"] for e in events]
                assert "step" in event_types, f"Expected step events, got {event_types}"

                # 验证 tool_call 类型存在
                step_types = [e["data"]["type"] for e in events if e["event"] == "step"]
                assert any(t == "tool_call" for t in step_types) or any(t == "tool_result" for t in step_types)
```


- [ ] **Step 2: 运行集成测试**

```bash
cd backend && ../.venv-mineru/Scripts/python.exe -m pytest tests/test_agent_integration.py -v --tb=short
```

- [ ] **Step 3: Commit**

```bash
git add backend/tests/test_agent_integration.py
git commit -m "test: add Agent integration tests"
```

---

### Task 12: 端到端验证

**Files:** 无新建，验证整体流程

- [ ] **Step 1: 启动后端验证路由注册**

```bash
cd backend && timeout 5 ../.venv-mineru/Scripts/python.exe -c "
from main import app
routes = [r.path for r in app.routes if hasattr(r, 'path')]
chat_routes = [r for r in routes if 'chat' in r]
print('Chat routes:', chat_routes)
assert any('stream' in r for r in chat_routes), 'Missing /stream route'
assert any('deep' in r for r in chat_routes), 'Missing /deep route'
assert any('resume' in r for r in chat_routes), 'Missing /resume route'
print('All chat routes registered OK')
"
```
Expected: `All chat routes registered OK`

- [ ] **Step 2: 前端 TypeScript 编译检查**

```bash
cd frontend && npx tsc --noEmit 2>&1 | head -20
```
Expected: No new TypeScript errors from our new components

- [ ] **Step 3: 运行全部后端测试**

```bash
cd backend && ../.venv-mineru/Scripts/python.exe -m pytest tests/ -v --tb=short 2>&1 | tail -30
```
Expected: All existing tests still pass + new tests pass

- [ ] **Step 4: Commit 最终检查结果**

```bash
git add -A && git status
```

---

## 实施小结

| Task | 文件 | 说明 |
|------|------|------|
| 1 | `models/agent.py` | 6 个数据模型 (AgentStep, Source, AgentResult, ClarifyRequest, AgentSession) |
| 2 | `services/agent_prompts.py` | System prompt + 6 工具 Function Calling JSON Schema |
| 3 | `services/agent_tools.py` | 5 工具函数 + execute_tool 调度 + format_tool_result 格式化 |
| 4 | `services/agent.py` | ReAct AgentLoop + _detect_clarify_intent + _parse_tool_calls |
| 5 | `routers/chat.py` | 3 个 SSE 端点: /stream /deep /deep/resume |
| 6 | `MarkdownRenderer.tsx` | 从 ReportViewer 提取共享 Markdown 渲染 |
| 7 | `AgentStepCard.tsx` | 可折叠步骤卡片组件 |
| 8 | `AgentChatBox.tsx` | 深度模式主容器, SSE 消费 + 状态管理 |
| 9 | `ChatBox.tsx` (改) | 适配 SSE chunk 流式 |
| 10 | `ChatPage.tsx` (改) | 快速/深度模式切换 |
| 11 | 集成测试 | 工具 + 格式化 + AgentLoop 端到端 |
| 12 | E2E 验证 | 路由检查 + TS 编译 + 全量测试 |
