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
- **文档内检索无法回答问题时**，使用 search_web 搜索外部信息补充（如最新进展、跨领域知识、文档未覆盖的背景概念）
- 最终答案前**必须**调用 reflect_on_answer，验证不通过则继续检索修正

## 追问时机
以下情况应使用 ask_user 追问用户而非猜测：
- 问题中的术语在文档中查不到（用户可能搞错了文档）
- 问题指代不明确（如"这个方法"在文档中出现多次）
- 用户要求总结但文档很长，需要确认范围

## 约束
- 每个事实必须有明确的来源（文档来源或网页来源）
- 文档中没有的信息明确说明"文档中未提供"
- 文档引用标注格式: [出处: 章节名, 第X页]
- Web 搜索结果引用格式: [来源: 网页标题](URL)
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
            "name": "search_web",
            "description": "在互联网上搜索外部信息。仅在文档内检索无法回答问题时使用，用于补充最新进展、跨领域知识或文档未覆盖的背景概念。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词或问题",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "返回结果数量，默认 5",
                        "default": 5,
                    },
                },
                "required": ["query"],
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
