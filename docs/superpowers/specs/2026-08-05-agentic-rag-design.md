# Agentic RAG 升级设计文档

**日期**: 2026-08-05
**状态**: 已确认

---

## 1. 概述

将 DocMind 从单次 RAG（检索→拼接→回答）升级为 Agentic RAG，引入规划、多步检索、验证、追问和自修正能力。保留现有快速问答，新增深度问答模式，两种模式共存。

## 2. 核心设计决策

| 决策 | 选择 |
|------|------|
| 部署模式 | 双模式并存：`/chat` 快速问答 + `/chat/deep` 深度 Agent 问答 |
| Agent 实现 | 自建 ReAct 循环 + OpenAI Function Calling，零框架依赖 |
| 工具数量 | 5 个：search_document / get_section / query_knowledge_graph / search_keyword / reflect_on_answer |
| 循环深度 | 固定最大 8 轮迭代 |
| 前端展示 | SSE 完整流式，逐步骤展示 Agent 思考过程 |
| 代码策略 | 纯增量，现有文件零破坏 |

## 3. 系统架构

```
用户提问
    │
    ├── 快速模式 (现有)
    │   question → embed → hybrid_search → build_context → LLM → answer
    │
    └── 深度模式 (新增)
        question
            │
            ▼
        ┌─────────────────────────────────────────────────┐
        │              Agent 循环 (ReAct Loop)              │
        │                                                   │
        │  Plan ──→ Act (工具调用) ──→ Observe ──→ Reflect │
        │    ↑                                      │       │
        │    └──────────── 修正/追问 ◀──────────────┘       │
        │                                                   │
        │  工具集: search_document / get_section /          │
        │          query_kg / search_keyword / reflect      │
        └──────────────┬──────────────────────────────────┘
                       │ SSE 流式推送每一步
                       ▼
                最终答案 + 引用来源 + 思考过程
```

## 4. 新增模块

### 4.1 文件清单

```
backend/
  models/agent.py           # Agent 数据模型 (AgentStep, AgentTrace, AgentResult)
  services/agent.py          # ReAct 循环主控
  services/agent_tools.py    # 5 个工具的实现函数
  services/agent_prompts.py  # Agent system prompt + tool JSON Schema
  routers/chat.py            # 新增流式/深度问答路由

frontend/src/
  components/AgentChatBox.tsx    # 深度模式对话组件
  components/AgentChatBox.css    # 样式
  components/AgentStepCard.tsx   # 步骤卡片
  components/MarkdownRenderer.tsx # 从 ReportViewer 提取的共享 Markdown 渲染

现有文件改动:
  backend/main.py            # 注册新路由
  frontend/src/pages/ChatPage.tsx  # 新增模式切换
  frontend/src/components/ChatBox.tsx  # 支持 SSE chunk 流式
```

### 4.2 模块职责

| 模块 | 职责 |
|------|------|
| `services/agent.py` | ReAct 循环主控：接收问题 → 循环（调 LLM 决策 → 执行工具 → 观察结果）→ 输出答案，通过 async generator 逐步骤 yield SSE 事件 |
| `services/agent_tools.py` | 5 个工具函数，每个接收参数、执行检索/查询、返回结构化 dict。输入验证 + 异常兜底 |
| `services/agent_prompts.py` | 两段 prompt：AGENT_SYSTEM_PROMPT（ReAct 角色设定 + 输出格式约束）和 TOOLS_SCHEMA（OpenAI Function Calling 格式的 5 个工具定义） |
| `models/agent.py` | AgentStep / AgentTrace / AgentResult / ClarifyRequest，全部 Pydantic BaseModel |

## 5. Agent 循环详细设计

### 5.1 状态机

```
┌─────────────────────────────────────────────────────┐
│                    Agent Loop                        │
│                                                     │
│  1. INIT                                            │
│     └─ → THINK                                      │
│                                                     │
│  2. THINK (LLM 决策, 含 tools 定义)                  │
│     ├─ tool_call → ACT                              │
│     ├─ CLARIFY: 问题不明确 → 返回追问内容              │
│     └─ ANSWER: 信息足够 → 进入 ANSWER 阶段             │
│                                                     │
│  3. ACT (执行工具)                                   │
│     └─ → OBSERVE                                    │
│                                                     │
│  4. OBSERVE                                         │
│     ├─ 注入结果到消息列表，轮数+1                       │
│     ├─ 轮数 < 8 → THINK                             │
│     └─ 轮数 = 8 → ANSWER (强制总结)                  │
│                                                     │
│  5. ANSWER                                          │
│     └─ 调用 reflect 自检 → 流式输出答案 + 引用 + 过程   │
│                                                     │
│  6. CLARIFY                                         │
│     └─ 返回追问内容，等待用户回复 → 通过 /resume 恢复    │
└─────────────────────────────────────────────────────┘
```

### 5.2 单轮消息结构

LLM 每轮决策看到的消息列表（以第 3 轮为例）：

```
System:  AGENT_SYSTEM_PROMPT
User:    原始问题: "动态规划和贪心算法的区别？"
Assistant(tool_call): search_document("动态规划 贪心算法")
Tool:    {hits: [{text:"动态规划是一种...", heading:"第3章", page:42}, ...]}
Assistant(tool_call): search_keyword("贪心算法")
Tool:    {hits: [{text:"贪心算法每次都选...", heading:"第5章", page:87}, ...]}
Assistant: — LLM 在此决策下一动作 (继续搜索 / 输出答案 / 追问用户)
```

### 5.3 交互示例

用户: "动态规划和贪心算法有什么区别？本文档提到过吗？"

```
Step 1  → 📋 规划: 搜索"动态规划"定义 → 搜索"贪心算法"定义 → 对比分析
Step 2  → 🔍 search_document("动态规划 贪心算法 区别") → 5 片段
Step 3  → 🔍 search_keyword("贪心算法") → 3 片段
Step 4  → 📖 get_section("动态规划") → 完整第3章内容
Step 5  → 🤔 reflect: 信息足够，引用可追溯
Step 6  → ✅ 答案 + 📎 来源: 第3章第2节, 第5章第1节
```

用户: "帮我总结一下这篇文章"（文档 > 30 页）

```
Step 1  → 📋 规划: 先获取文档结构 → 逐章摘要 → 合并总结
Step 2  → 🔍 search_document("目录 章节 结构") → 结构信息
Step 3  → 🤔 reflect: 文档过长，需要澄清
Step 4  → 💬 CLARIFY: "文档共 52 页，涵盖 8 个章节。你想总结全部还是重点了解某几个章节？"
         → 等待用户回复 → /resume
```

## 6. 工具定义

### 6.1 工具 Schema (Function Calling JSON)

```python
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_document",
            "description": "在文档中执行混合检索(BM25+向量RRF融合)，返回最相关的文本片段。适合开放式语义查询，如'动态规划的定义'",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "自然语言查询"},
                    "top_k": {"type": "integer", "description": "返回数量，默认10", "default": 10},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_section",
            "description": "获取指定章节的完整原始内容。适合需要完整阅读某个章节时使用",
            "parameters": {
                "type": "object",
                "properties": {
                    "heading": {"type": "string", "description": "章节标题（支持子串模糊匹配）"},
                },
                "required": ["heading"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_knowledge_graph",
            "description": "查询知识图谱中某个概念的详细信息及其关联概念(前置知识、推导关系、对比关系等)",
            "parameters": {
                "type": "object",
                "properties": {
                    "concept": {"type": "string", "description": "概念名称"},
                },
                "required": ["concept"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_keyword",
            "description": "纯BM25关键词检索，适合查找特定术语、人名、公式等精确匹配",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {"type": "string", "description": "精确关键词"},
                    "top_k": {"type": "integer", "description": "返回数量，默认10", "default": 10},
                },
                "required": ["keyword"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reflect_on_answer",
            "description": "在输出最终答案前自检：答案是否完全基于文档？有无编造事实？引用出处是否准确？如果验证不通过，应该继续检索修正",
            "parameters": {
                "type": "object",
                "properties": {
                    "draft_answer": {"type": "string", "description": "草稿答案"},
                    "cited_sources": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "引用的原文片段列表",
                    },
                },
                "required": ["draft_answer", "cited_sources"],
            },
        },
    },
]
```

### 6.2 工具实现函数签名

```python
# services/agent_tools.py

def tool_search_document(doc_id: str, query: str, top_k: int = 10) -> dict:
    """封装现有的 milvus_client.hybrid_query + chunker.embed"""
    ...

def tool_get_section(doc_id: str, heading: str) -> dict:
    """从 SQLite 读取 structured_json，匹配章节 heading，返回完整 content"""
    ...

def tool_query_knowledge_graph(doc_id: str, concept: str) -> dict:
    """从 kg_store.load 的图谱中搜索节点 + 关联邻居"""
    ...

def tool_search_keyword(doc_id: str, keyword: str, top_k: int = 10) -> dict:
    """纯 BM25 检索（不用向量），用于精确术语查找"""
    ...

def tool_reflect_on_answer(
    draft_answer: str, cited_sources: list[str], doc_id: str
) -> dict:
    """LLM 自检：调用轻量 LLM 检查答案是否基于文档，返回 {passed: bool, issues: list}"""
    ...
```

## 7. API 设计

### 7.1 接口清单

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/documents/{id}/chat/stream` | 新增：SSE 流式快速问答 |
| POST | `/api/documents/{id}/chat/deep` | 新增：SSE 流式深度 Agent 问答 |
| POST | `/api/documents/{id}/chat` | 保留：当前非流式接口 |
| POST | `/api/documents/{id}/chat/deep/resume` | 新增：回复追问恢复 Agent 循环 |

### 7.2 SSE 事件类型

```
event: step
data: {
  "type": "plan" | "tool_call" | "tool_result" | "reflect" | "clarify" | "error",
  "step_index": 2,
  "content": "人类可读步骤描述",
  "detail": { ... }           // type=tool_call: {tool_name, arguments}
                               // type=tool_result: {tool_name, result_summary, hits_count}
                               // type=reflect: {passed, issues}
                               // type=clarify: {clarify_question, options}
}

event: chunk
data: {"text": "流式答案文本片段"}

event: sources
data: [
  {"heading": "第3章 动态规划", "page": 42, "excerpt": "原文片段...", "score": 0.92},
  ...
]

event: done
data: {
  "total_steps": 7,
  "iterations": 3,
  "tokens_used": 1520,
  "follow_up_suggestions": ["需要详细解释XX吗？", ...]
}
```

### 7.3 请求体

```json
// POST /api/documents/{id}/chat/deep
{
  "question": "动态规划和贪心算法的区别？",
  "max_iterations": 8     // 可选，覆盖默认值
}

// POST /api/documents/{id}/chat/deep/resume
{
  "session_id": "abc-123",
  "user_response": "重点看第3章和第5章就行"
}
```

## 8. 前端设计

### 8.1 组件树

```
ChatPage.tsx (改动: 新增模式切换开关)
├── TopNav (不改)
├── 左侧: 文档列表 (不改)
└── 右侧:
    ├── [快速模式] ChatBox.tsx
    │   改动: 支持 SSE chunk 流式接收替代一次性返回
    │
    └── [深度模式] AgentChatBox.tsx (新建)
         ├── AgentStepCard.tsx (新建)
         │    └── 可折叠卡片: 状态图标 + 描述 + 展开详情
         ├── AgentAnswer.tsx (新建)
         │    ├── <MarkdownRenderer /> (从 ReportViewer 提取)
         │    ├── 引用标注列表 (📎 点击定位)
         │    └── 追问建议按钮组
         └── AgentInput.tsx (新建)
              └── 输入框 + 追问快捷入口
```

### 8.2 深度模式 UI 示意图

```
┌──────────────────────────────────────────────────────┐
│  [快速模式]  [深度模式 ●]                              │
│──────────────────────────────────────────────────────│
│                                                      │
│  👤 动态规划和贪心算法有什么区别？                      │
│                                                      │
│  📋 分析计划                               [展开 ▾]  │
│    搜索"动态规划"定义 → 搜索"贪心算法" → 对比分析      │
│                                                      │
│  🔍 步骤 1: 混合检索                    [展开 ▸]     │
│  ▸ 搜索: "动态规划 贪心算法 区别"                      │
│  ▸ 返回 5 个相关片段                                  │
│                                                      │
│  🔍 步骤 2: 关键词搜索                 [展开 ▸]     │
│  ▸ 搜索: "贪心算法"                                   │
│  ▸ 返回 3 个相关片段                                  │
│                                                      │
│  📖 步骤 3: 获取章节                   [展开 ▸]     │
│  ▸ 获取: "动态规划"完整章节                            │
│                                                      │
│  🤔 步骤 4: 反思验证                   [展开 ▸]     │
│  ✅ 答案基于文档，引用可追溯                           │
│                                                      │
│  ─────────────────────────────────────               │
│  ✅ 最终答案                                         │
│                                                      │
│  本文档在第3章和第5章中分别阐述了动态规划和             │
│  贪心算法。主要区别在于：                              │
│                                                      │
│  1. **动态规划**适用于具有重叠子问题和最优子结构        │
│     性质的问题...(第3章第2节)                          │
│                                                      │
│  2. **贪心算法**在每一步都选择当前最优解...             │
│     (第5章第1节)                                      │
│                                                      │
│  📎 引用来源                                         │
│  ┌─────────────────────────────────────────┐        │
│  │ 第3章第2节 (P42)  "...动态规划通过存储..." │        │
│  │ 第5章第1节 (P87)  "...贪心策略不能保证..." │        │
│  └─────────────────────────────────────────┘        │
│                                                      │
│  💡 追问建议                                         │
│  [需要我详细解释最优子结构吗?] [贪心算法的适用场景?]    │
│                                                      │
│  ─────────────────────────────────────               │
│  💬 输入新问题...                          [发送]     │
└──────────────────────────────────────────────────────┘
```

### 8.3 状态管理

AgentChatBox 内部状态：

```typescript
interface AgentChatState {
  mode: 'idle' | 'running' | 'clarify' | 'done';
  steps: AgentStep[];           // 累积的步骤列表
  answerText: string;           // 流式累积的答案文本
  sources: Source[];            // 引用来源
  followUpSuggestions: string[];
  sessionId: string | null;     // 用于 resume 的会话 ID
}
```

## 9. Agent System Prompt 设计

### 9.1 核心指令

```
你是一个文档深度分析助手，能够自主规划和执行多步检索来回答复杂问题。

## 工作流程
1. 理解问题后，先制定检索计划（在思考中，不输出给用户）
2. 选择合适的工具执行检索
3. 根据检索结果决定：继续搜索更多信息 / 信息足够则输出答案 / 问题不明确则追问用户
4. 输出答案前必须用 reflect_on_answer 工具自检

## 工具使用原则
- 优先使用 search_document 进行语义搜索
- 需要精确术语匹配时使用 search_keyword
- 需要完整上下文时使用 get_section
- 需要理解概念关联时使用 query_knowledge_graph
- 最终答案前必须调用 reflect_on_answer

## 追问时机
以下情况应追问用户而非猜测：
- 问题中的术语在文档中不存在（可能是用户搞错了文档）
- 问题指代不明确（"这个方法"在文档中出现多次）
- 需要用户指定范围（文档很长时"总结一下"需要明确范围）

## 约束
- 每个事实必须有文档来源
- 文档没有的信息明确说明"文档中未提供"
- 引用标注格式: [出处: 章节名, 第X页]
- 最多 8 次工具调用，达到上限后输出已有最佳答案
```

### 9.2 输出格式指令

LLM 通过 Function Calling 的 tool_calls 机制自然输出工具调用。当 LLM 决定输出最终答案时，finish_reason 为 `stop`（不调工具），此时流式输出答案文本。

追问场景：将 `ask_user` 作为第 6 个伪工具（仅 Agent 内部识别，不暴露给用户侧）。LLM 调用 `ask_user(question, options?)` 时，Agent 循环暂停，将追问内容推送给前端并等待 `/resume`。如果 LLM 在 content 中输出追问文本（非 tool_call 形式），Agent 循环中也做关键词检测（"请问""你想了解""你能告诉我"）兜底。

## 10. 数据流

### 10.1 深度问答完整数据流

```
1. 前端 POST /chat/deep {"question": "..."}
2. 后端 agent.py:
   a. 创建 async generator
   b. 初始化消息: [system_prompt, user_question]
   c. LOOP (max 8 iterations):
      i.   调用 LLM (带 tools 定义)
      ii.  检查响应:
           - 有 tool_calls → yield step:tool_call → 执行工具 → yield step:tool_result → 追加到消息
           - 无 tool_calls → break (进入回答阶段)
           - clarify 检测 → yield step:clarify → 挂起等待 resume
      iii. 工具调用计数 +1
   d. 回答阶段:
      i.   调用 reflect_on_answer 工具
      ii.  yield step:reflect
      iii. 如果 reflect 未通过 → 追加反馈到消息 → 回到 LOOP
      iv.  reflect 通过 → yield chunk... (流式输出答案)
      v.   yield sources
      vi.  yield done
3. 前端逐事件渲染
```

### 10.2 工具调用执行流程

```python
# services/agent_tools.py 中的工具调度

TOOL_MAP = {
    "search_document": tool_search_document,
    "get_section": tool_get_section,
    "query_knowledge_graph": tool_query_knowledge_graph,
    "search_keyword": tool_search_keyword,
    "reflect_on_answer": tool_reflect_on_answer,
}

async def execute_tool(doc_id: str, tool_name: str, arguments: dict) -> dict:
    """统一工具调度，捕获异常并返回结构化错误"""
    try:
        return TOOL_MAP[tool_name](doc_id=doc_id, **arguments)
    except Exception as e:
        return {"error": str(e), "tool": tool_name}
```

## 11. 会话管理

深度问答的追问/回复流程需要轻量会话管理：

- `session_id` 由后端在首次 `/chat/deep` 请求时生成（UUID4），通过 SSE `event:done` 返回
- 会话数据存储在内存 dict `{session_id: {messages, doc_id, expires_at}}`，带 30 分钟 TTL
- 用户回复追问时通过 `/chat/deep/resume` 携带 `session_id`，后端恢复消息历史继续循环
- 超时会话自动清理，回复时返回 404

## 12. 错误处理

| 场景 | 处理 |
|------|------|
| 工具调用超时 (30s) | 返回 `{"error": "timeout"}`，Agent 可以换策略或继续 |
| LLM 返回非标准 tool_call | 解析失败时当作普通文本，检查是否追问/答案 |
| Milvus 不可用 | 降级为纯 BM25 检索 |
| KG 未构建 | `query_knowledge_graph` 返回 `{"available": false}` |
| reflect 自检不通过 | 将问题反馈注入消息，Agent 继续检索修正 |
| 达到最大轮数 | 强制调用 LLM 输出已有最佳答案，标注"部分信息可能不完整" |
| 用户断开 SSE | 取消当前 Agent 循环，不阻塞后续请求 |

## 13. 测试策略

### 12.1 单元测试

| 模块 | 测试内容 |
|------|---------|
| `agent_tools.py` | 5 个工具各 3 个测试：正常返回 / 空结果 / 异常输入 |
| `agent.py` | 工具调用决策解析 / clarify 检测 / 最大轮数截断 / reflect 不通过回退 |
| `agent_prompts.py` | prompt 模板变量填充 / tools schema 格式校验 |

### 12.2 集成测试

| 场景 | 预期 |
|------|------|
| 简单事实查询 | 2-3 轮完成，验证引用准确 |
| 复杂对比问题 | 4-6 轮，验证多来源综合 |
| 文档未覆盖的问题 | Agent 明确说明"文档未提供" |
| 含糊提问 | Agent 输出 clarify 追问 |
| reflect 不通过 | Agent 继续检索修正后输出 |

## 14. 实施顺序

| 阶段 | 内容 | 依赖 |
|------|------|------|
| 1 | `models/agent.py` 数据模型 | 无 |
| 2 | `services/agent_prompts.py` | 无 |
| 3 | `services/agent_tools.py` 5 个工具 | 现有 milvus/chunker/kg_store |
| 4 | `services/agent.py` ReAct 循环 | 2, 3 |
| 5 | `routers/chat.py` SSE 路由 | 4 |
| 6 | 前端 `AgentChatBox` + 子组件 | 5 |
| 7 | 前端 `ChatPage` 模式切换 | 6 |
| 8 | 单元测试 + 集成测试 | 4 |
| 9 | 端到端手工验证 | 全部 |

---

## 附录: 与现有代码的关系

| 现有文件 | 改动情况 |
|---------|---------|
| `backend/main.py` | 注册 `chat_router` |
| `backend/services/llm_service.py` | 不改，Agent 直接调 `OpenAI client` |
| `backend/services/chunker.py` | 不改 |
| `backend/storage/milvus_client.py` | 不改 |
| `backend/storage/db.py` | 不改 |
| `backend/routers/documents.py` | 不改（保留现有 /chat 接口） |
| `frontend/src/pages/ChatPage.tsx` | 新增模式切换开关 |
| `frontend/src/components/ChatBox.tsx` | 适配 SSE chunk 流式 |
| 其他现有文件 | 均不改 |
