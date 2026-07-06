# 文档理解 RAG 平台 — 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建一个面向小团队的 AI 文档阅读助手，支持上传 PDF/Word/Markdown，生成批判性阅读报告和交互式思维导图。

**Architecture:** FastAPI 后端处理文档解析、分块嵌入、ChromaDB 存储和 LLM 编排；React + TypeScript 前端提供文档管理、报告渲染和思维导图交互。前后端通过 REST API 通信。

**Tech Stack:** Python 3.11+, FastAPI, PyMuPDF, python-docx, LangChain text-splitters, ChromaDB, 通义千问/DeepSeek API, React 18, TypeScript, Vite, react-markdown, markmap

---

## 文件结构

```
doc-rag-platform/
├── backend/
│   ├── requirements.txt
│   ├── main.py                    # FastAPI 入口，挂载路由 + CORS
│   ├── config.py                  # 环境变量 / 配置
│   ├── routers/
│   │   ├── __init__.py
│   │   └── documents.py           # 文档 CRUD + 报告/导图生成 API
│   ├── services/
│   │   ├── __init__.py
│   │   ├── parser.py              # PDF/Word/Markdown → 统一结构化 JSON
│   │   ├── chunker.py             # 分块 + 嵌入
│   │   ├── llm_service.py         # LLM API 调用封装
│   │   └── prompt_templates.py    # 报告 & 导图 System Prompt
│   ├── models/
│   │   ├── __init__.py
│   │   └── document.py            # Pydantic 数据模型
│   └── storage/
│       ├── __init__.py
│       ├── chroma_client.py       # ChromaDB 操作封装
│       └── file_store.py          # 本地文件存储
├── frontend/
│   ├── package.json
│   ├── tsconfig.json
│   ├── tsconfig.app.json
│   ├── tsconfig.node.json
│   ├── vite.config.ts
│   ├── index.html
│   └── src/
│       ├── main.tsx
│       ├── App.tsx
│       ├── App.css
│       ├── pages/
│       │   ├── DocumentList.tsx
│       │   ├── DocumentList.css
│       │   ├── DocumentReport.tsx
│       │   ├── DocumentReport.css
│       │   ├── DocumentMindmap.tsx
│       │   └── DocumentMindmap.css
│       ├── components/
│       │   ├── FileUpload.tsx
│       │   ├── FileUpload.css
│       │   ├── ReportViewer.tsx
│       │   ├── ScoreCard.tsx
│       │   ├── ScoreCard.css
│       │   ├── MindmapViewer.tsx
│       │   └── MindmapViewer.css
│       └── services/
│           └── api.ts
└── uploads/                        # 运行时创建，存储上传文件
```

---

### Task 1: 后端项目脚手架

**Files:**
- Create: `backend/requirements.txt`

- [ ] **Step 1: 创建后端目录结构**

```bash
mkdir -p backend/routers backend/services backend/models backend/storage
touch backend/routers/__init__.py backend/services/__init__.py backend/models/__init__.py backend/storage/__init__.py
```

- [ ] **Step 2: 编写 requirements.txt**

```txt
fastapi==0.115.0
uvicorn[standard]==0.30.6
python-multipart==0.0.12
pymupdf==1.24.10
python-docx==1.1.2
markdown==3.7
beautifulsoup4==4.12.3
langchain==0.3.4
langchain-text-splitters==0.3.0
chromadb==0.5.15
openai==1.51.0
pydantic==2.9.2
pydantic-settings==2.5.2
aiofiles==24.1.0
httpx==0.27.2
```

- [ ] **Step 3: 安装依赖**

```bash
cd backend && pip install -r requirements.txt
```

- [ ] **Step 4: Commit**

```bash
git add backend/
git commit -m "feat: scaffold backend project with dependencies"
```

---

### Task 2: 配置管理 + 数据模型

**Files:**
- Create: `backend/config.py`
- Create: `backend/models/document.py`

- [ ] **Step 1: 编写 config.py**

```python
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    # LLM
    llm_provider: str = "qwen"  # qwen | deepseek
    qwen_api_key: str = ""
    qwen_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    qwen_model: str = "qwen-plus"

    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    deepseek_model: str = "deepseek-chat"

    # Embedding
    embedding_provider: str = "qwen"
    embedding_model: str = "text-embedding-v3"

    # Storage
    upload_dir: str = "../uploads"
    chroma_persist_dir: str = "../chroma_data"

    # Chunking
    chunk_size: int = 500
    chunk_overlap: int = 50

    # Server
    host: str = "0.0.0.0"
    port: int = 8000


settings = Settings()
```

- [ ] **Step 2: 编写 models/document.py**

```python
from __future__ import annotations

import enum
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class DocumentFormat(str, enum.Enum):
    PDF = "pdf"
    DOCX = "docx"
    MARKDOWN = "md"


class DocumentStatus(str, enum.Enum):
    UPLOADED = "uploaded"
    PARSING = "parsing"
    CHUNKING = "chunking"
    READY = "ready"
    ERROR = "error"


class TableData(BaseModel):
    headers: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)


class Section(BaseModel):
    heading: str = ""
    level: int = 1
    content: str = ""
    tables: list[TableData] = Field(default_factory=list)
    page: int = 0


class DocumentMetadata(BaseModel):
    filename: str
    format: DocumentFormat
    pages: int = 0
    uploaded_at: datetime = Field(default_factory=datetime.now)
    file_size: int = 0


class StructuredDocument(BaseModel):
    title: str = ""
    sections: list[Section] = Field(default_factory=list)
    metadata: DocumentMetadata


class DocumentRecord(BaseModel):
    id: str
    status: DocumentStatus = DocumentStatus.UPLOADED
    title: str = ""
    created_at: datetime = Field(default_factory=datetime.now)
    metadata: Optional[DocumentMetadata] = None
    structured: Optional[StructuredDocument] = None


class ReportResult(BaseModel):
    content: str  # Markdown
    generated_at: datetime = Field(default_factory=datetime.now)


class MindmapNode(BaseModel):
    name: str
    meta: Optional[str] = None
    children: list["MindmapNode"] = Field(default_factory=list)


class MindmapResult(BaseModel):
    root: MindmapNode
    generated_at: datetime = Field(default_factory=datetime.now)
```

- [ ] **Step 3: Commit**

```bash
git add backend/config.py backend/models/
git commit -m "feat: add config management and data models"
```

---

### Task 3: 文件存储服务

**Files:**
- Create: `backend/storage/file_store.py`

- [ ] **Step 1: 编写 file_store.py**

```python
import os
import uuid
from pathlib import Path

import aiofiles

from config import settings


class FileStore:
    def __init__(self, base_dir: str | None = None):
        self.base_dir = Path(base_dir or settings.upload_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def get_path(self, doc_id: str, filename: str) -> Path:
        ext = filename.rsplit(".", 1)[-1] if "." in filename else ""
        return self.base_dir / f"{doc_id}.{ext}"

    async def save(self, filename: str, content: bytes) -> tuple[str, Path]:
        doc_id = uuid.uuid4().hex[:12]
        file_path = self.get_path(doc_id, filename)
        async with aiofiles.open(file_path, "wb") as f:
            await f.write(content)
        return doc_id, file_path

    def exists(self, doc_id: str, filename: str) -> bool:
        return self.get_path(doc_id, filename).exists()

    def delete(self, doc_id: str, filename: str) -> bool:
        path = self.get_path(doc_id, filename)
        if path.exists():
            path.unlink()
            return True
        return False

    def read_bytes(self, doc_id: str, filename: str) -> bytes:
        return self.get_path(doc_id, filename).read_bytes()


file_store = FileStore()
```

- [ ] **Step 2: Commit**

```bash
git add backend/storage/file_store.py
git commit -m "feat: add file storage service"
```

---

### Task 4: 文档解析器

**Files:**
- Create: `backend/services/parser.py`

- [ ] **Step 1: 编写 parser.py**

```python
from pathlib import Path

import fitz  # PyMuPDF
from docx import Document as DocxDocument
from markdown import markdown
from bs4 import BeautifulSoup

from models.document import (
    DocumentFormat,
    DocumentMetadata,
    Section,
    StructuredDocument,
    TableData,
)


class DocumentParser:
    def parse(self, file_path: Path, filename: str) -> StructuredDocument:
        fmt = self._detect_format(filename)
        metadata = DocumentMetadata(
            filename=filename,
            format=fmt,
            file_size=file_path.stat().st_size,
        )

        if fmt == DocumentFormat.PDF:
            return self._parse_pdf(file_path, metadata)
        elif fmt == DocumentFormat.DOCX:
            return self._parse_docx(file_path, metadata)
        elif fmt == DocumentFormat.MARKDOWN:
            return self._parse_markdown(file_path, metadata)
        else:
            raise ValueError(f"Unsupported format: {fmt}")

    def _detect_format(self, filename: str) -> DocumentFormat:
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        mapping = {
            "pdf": DocumentFormat.PDF,
            "docx": DocumentFormat.DOCX,
            "doc": DocumentFormat.DOCX,
            "md": DocumentFormat.MARKDOWN,
            "markdown": DocumentFormat.MARKDOWN,
        }
        if ext not in mapping:
            raise ValueError(f"Unsupported file extension: .{ext}")
        return mapping[ext]

    def _parse_pdf(self, file_path: Path, metadata: DocumentMetadata) -> StructuredDocument:
        doc = fitz.open(file_path)
        metadata.pages = len(doc)

        sections: list[Section] = []
        current_section = Section(heading="正文", level=1, page=1)
        current_text: list[str] = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            blocks = page.get_text("blocks")
            blocks.sort(key=lambda b: (b[1], b[0]))  # sort by y then x

            for block in blocks:
                text = block[4].strip()
                if not text:
                    continue
                # Detect headings: short lines that are bold or larger font
                block_lines = text.split("\n")
                for line in block_lines:
                    line = line.strip()
                    if not line:
                        continue
                    # Heuristic: short line (< 60 chars) ending without punctuation → heading
                    if len(line) < 60 and not line[-1] in "。，；：》、.!?;:,":
                        if current_text:
                            current_section.content = "\n".join(current_text)
                            sections.append(current_section)
                            current_text = []
                        level = 1 if len(line) < 30 else 2
                        current_section = Section(
                            heading=line, level=level, page=page_num + 1
                        )
                    else:
                        current_text.append(line)

        if current_text:
            current_section.content = "\n".join(current_text)
        sections.append(current_section)

        # Extract tables
        for page_num in range(len(doc)):
            page = doc[page_num]
            tables = page.find_tables()
            for table in tables:
                data = table.extract()
                if data:
                    headers = [str(h).strip() if h else "" for h in data[0]]
                    rows = [[str(c).strip() if c else "" for c in row] for row in data[1:]]
                    # Attach table to nearest section by page
                    for sec in reversed(sections):
                        if sec.page <= page_num + 1:
                            sec.tables.append(TableData(headers=headers, rows=rows))
                            break

        doc.close()

        title = sections[0].heading if sections else filename
        return StructuredDocument(title=title, sections=sections, metadata=metadata)

    def _parse_docx(self, file_path: Path, metadata: DocumentMetadata) -> StructuredDocument:
        doc = DocxDocument(file_path)
        sections: list[Section] = []
        current_section = Section(heading="正文", level=1, page=0)
        current_text: list[str] = []

        for para in doc.paragraphs:
            text = para.text.strip()
            if not text:
                continue

            style_name = para.style.name if para.style else ""
            if "Heading" in style_name or "heading" in style_name or "标题" in style_name:
                if current_text:
                    current_section.content = "\n".join(current_text)
                    sections.append(current_section)
                    current_text = []
                level = self._extract_heading_level(style_name)
                current_section = Section(heading=text, level=level)
            else:
                current_text.append(text)

        if current_text:
            current_section.content = "\n".join(current_text)
        sections.append(current_section)

        # Extract tables
        for table in doc.tables:
            headers = []
            rows = []
            for i, row in enumerate(table.rows):
                cells = [cell.text.strip() for cell in row.cells]
                if i == 0:
                    headers = cells
                else:
                    rows.append(cells)
            if rows:
                sections[-1].tables.append(TableData(headers=headers, rows=rows))

        title = sections[0].heading if sections else metadata.filename
        return StructuredDocument(title=title, sections=sections, metadata=metadata)

    def _parse_markdown(self, file_path: Path, metadata: DocumentMetadata) -> StructuredDocument:
        text = file_path.read_text(encoding="utf-8")
        html = markdown(text, extensions=["tables", "fenced_code", "codehilite"])
        soup = BeautifulSoup(html, "html.parser")

        sections: list[Section] = []
        current_section = Section(heading="正文", level=1, page=0)
        current_text: list[str] = []

        for tag in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "table", "pre", "ul", "ol"]):
            if tag.name.startswith("h"):
                if current_text:
                    current_section.content = "\n".join(current_text)
                    sections.append(current_section)
                    current_text = []
                level = int(tag.name[1])
                current_section = Section(heading=tag.get_text(strip=True), level=level)
            elif tag.name == "table":
                headers = [th.get_text(strip=True) for th in tag.find_all("th")]
                rows = [
                    [td.get_text(strip=True) for td in tr.find_all("td")]
                    for tr in tag.find_all("tr") if tr.find_all("td")
                ]
                if not headers and rows:
                    headers = rows[0]
                    rows = rows[1:]
                current_section.tables.append(TableData(headers=headers, rows=rows))
            elif tag.name == "pre":
                current_text.append(f"```\n{tag.get_text()}\n```")
            elif tag.name in ("ul", "ol"):
                items = [f"- {li.get_text(strip=True)}" for li in tag.find_all("li")]
                current_text.extend(items)
            else:
                t = tag.get_text(strip=True)
                if t:
                    current_text.append(t)

        if current_text:
            current_section.content = "\n".join(current_text)
        sections.append(current_section)

        title = sections[0].heading if sections else metadata.filename
        return StructuredDocument(title=title, sections=sections, metadata=metadata)

    def _extract_heading_level(self, style_name: str) -> int:
        for i in range(1, 7):
            if str(i) in style_name:
                return i
        return 1


document_parser = DocumentParser()
```

- [ ] **Step 2: Commit**

```bash
git add backend/services/parser.py
git commit -m "feat: add multi-format document parser"
```

---

### Task 5: 分块 + 嵌入服务

**Files:**
- Create: `backend/services/chunker.py`

- [ ] **Step 1: 编写 chunker.py**

```python
from typing import Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

from config import settings
from models.document import Section, StructuredDocument


class Chunker:
    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", "。", "，", " ", ""],
        )
        self._embedding_client: Optional[OpenAI] = None

    @property
    def embedding_client(self) -> OpenAI:
        if self._embedding_client is None:
            self._embedding_client = OpenAI(
                api_key=settings.qwen_api_key,
                base_url=settings.qwen_base_url,
            )
        return self._embedding_client

    def _section_to_text(self, section: Section, path: str = "") -> str:
        header = f"# {section.heading}" if section.heading else ""
        if path:
            header = f"{path} > {header}" if header else path

        parts = [header] if header else []
        if section.content:
            parts.append(section.content)
        for table in section.tables:
            table_md = self._table_to_markdown(table)
            parts.append(table_md)
        return "\n\n".join(parts)

    def _table_to_markdown(self, table) -> str:
        lines = []
        if table.headers:
            lines.append("| " + " | ".join(table.headers) + " |")
            lines.append("| " + " | ".join(["---"] * len(table.headers)) + " |")
        for row in table.rows:
            lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines)

    def _build_heading_path(self, sections: list[Section], idx: int) -> str:
        path_parts = []
        current_level = float("inf")
        for s in reversed(sections[: idx + 1]):
            if s.heading and s.level < current_level:
                path_parts.insert(0, s.heading)
                current_level = s.level
        return " > ".join(path_parts)

    def chunk(self, doc: StructuredDocument) -> list[dict]:
        """Return list of {text, metadata} dicts."""
        chunks = []
        for i, section in enumerate(doc.sections):
            heading_path = self._build_heading_path(doc.sections, i)
            section_text = self._section_to_text(section, heading_path)

            sub_chunks = self.splitter.split_text(section_text)
            for j, chunk_text in enumerate(sub_chunks):
                chunks.append({
                    "text": chunk_text,
                    "metadata": {
                        "heading_path": heading_path,
                        "section_heading": section.heading,
                        "page": section.page,
                        "chunk_index": j,
                    },
                })
        return chunks

    def embed(self, texts: list[str]) -> list[list[float]]:
        resp = self.embedding_client.embeddings.create(
            model=settings.embedding_model,
            input=texts,
        )
        return [d.embedding for d in resp.data]


chunker = Chunker()
```

- [ ] **Step 2: Commit**

```bash
git add backend/services/chunker.py
git commit -m "feat: add chunking and embedding service"
```

---

### Task 6: ChromaDB 存储客户端

**Files:**
- Create: `backend/storage/chroma_client.py`

- [ ] **Step 1: 编写 chroma_client.py**

```python
import uuid
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from config import settings


class ChromaClient:
    def __init__(self):
        self._client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

    def create_collection(self, doc_id: str) -> chromadb.Collection:
        name = f"doc_{doc_id}"
        # Delete existing collection if re-processing
        try:
            self._client.delete_collection(name)
        except Exception:
            pass
        return self._client.create_collection(name=name)

    def get_collection(self, doc_id: str) -> Optional[chromadb.Collection]:
        name = f"doc_{doc_id}"
        try:
            return self._client.get_collection(name=name)
        except Exception:
            return None

    def delete_collection(self, doc_id: str):
        try:
            self._client.delete_collection(f"doc_{doc_id}")
        except Exception:
            pass

    def add_chunks(
        self,
        collection: chromadb.Collection,
        chunks: list[dict],
        embeddings: list[list[float]],
    ):
        ids = [uuid.uuid4().hex[:16] for _ in chunks]
        documents = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)

    def query(
        self,
        doc_id: str,
        query_embedding: list[float],
        top_k: int = 20,
    ) -> list[dict]:
        collection = self.get_collection(doc_id)
        if collection is None:
            return []
        results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
        chunks = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                chunks.append({"text": doc, "metadata": meta})
        return chunks


chroma_client = ChromaClient()
```

- [ ] **Step 2: Commit**

```bash
git add backend/storage/chroma_client.py
git commit -m "feat: add ChromaDB storage client"
```

---

### Task 7: 提示词模板

**Files:**
- Create: `backend/services/prompt_templates.py`

- [ ] **Step 1: 编写 prompt_templates.py**

```python
REPORT_SYSTEM_PROMPT = """你是一位资深分析师，从业 15 年，以批判性思维和直言不讳著称。你不会美化任何内容——论证薄弱的你会直接指出，数据缺失的你会明确标注。

## 第零步：文档评估（每次必须先做）
在分析前，先评估文档：
- 文档类型：（学术论文 / 行业报告 / 技术文档 / 会议纪要 / 其他）
- 篇幅级别：（短篇 ≤5页 / 中篇 5-30页 / 长篇 >30页）
- 信息密度：（高 / 中 / 低）
- 阅读建议：一句话告诉读者应该花多少时间、重点看什么

## 分析策略（根据篇幅自动选择）

### 策略 A：短篇（≤5页）— 全量深度分析
执行完整四步，每一部分做到最详细。

### 策略 B：中篇（5-30页）— 标准分析
执行完整四步，内容拆解按章节归类而非逐段展开。

### 策略 C：长篇（>30页）— 两阶段分析，本次为第一阶段

**第一阶段输出：**
1. 文档评估结果
2. 结构骨架表格（|章节|页码范围|核心内容一句话|重要性★1-5|）
3. 全局概览：核心主题、论证主线、关键结论
4. 阅读路径建议：三条路径 ——"5分钟了解大意""掌握核心论证""全面深入理解"
5. 结尾必须输出引导语："请告诉我你最想深入了解的章节（可多选），我将为你生成这些章节的深度分析。"

## 四步分析内容（策略 A/B 使用，策略 C 第二阶段使用）
第一步 — 快速评估：值得读吗？目标读者？
第二步 — 内容拆解：核心主张、论证架构、关键证据[标注出处]、重要数据（表格）、核心概念
第三步 — 批判性分析：最强论据、最弱环节、缺失内容、潜在假设、反方观点
第四步 — 读者指南：精读建议（附页码）、可信度评分（论据/逻辑/数据各1-10）、延伸阅读

## 约束
- 每个观点标注原文出处 [参见段落X]
- 文档未提供的信息写明"文档中未提供"，不编造
- 如实评分，不美化
- Markdown 格式输出，中文为主"""

REPORT_USER_PROMPT = """请分析以下文档内容，生成阅读分析报告。

文档标题：{title}
文档格式：{format}，共 {pages} 页

{content}"""

DEEP_DIVE_SYSTEM_PROMPT = """你是一位资深分析师。用户已阅读文档的结构骨架，现在指定了想深入了解的章节。请对这些章节执行完整的深度分析。

## 分析要求
对每个选中章节独立输出完整四步分析：
1. **核心主张**：该章节最想让读者相信什么？
2. **内容拆解**：论证架构 + 关键证据[标注出处] + 重要数据（表格）
3. **批判性分析**：论据强弱、逻辑漏洞、缺失角度、潜在假设
4. **读者指南**：精读建议 + 可信度评分（论据/逻辑/数据各1-10）

## 约束
- 每个章节一个二级标题，其下按四步结构展开
- 每个观点标注出处
- 文档未提供的信息写明"文档中未提供"
- Markdown 格式输出，中文为主"""

DEEP_DIVE_USER_PROMPT = """文档标题：{title}

用户选中的章节：{selected_sections}

请对这些章节进行深度分析：

{content}"""

MINDMAP_SYSTEM_PROMPT = """你是一位信息架构师，擅长将复杂内容转化为清晰的知识图谱。

## 你的任务
基于提供的文档内容，生成一份交互式思维导图 JSON，帮助读者快速建立文档的知识框架。

## 生成规则
1. **根节点**：用一句话（≤20字）概括文档核心主题
2. **一层节点**：主要章节/论证模块（5-7个）
3. **二层节点**：每章的核心观点/关键概念
4. **三层节点**：支撑细节（关键数据、重要引用、案例名称）
5. **四层节点**（可选，最多5个）：值得追问的问题、潜在争议点

## 节点增强标注
在节点名称后附加元标注：
- {bold} — 标记最重要的 3 个节点，值得优先阅读
- {warning} — 标记论证薄弱或有争议的节点（最多3处）
- {question} — 标记文档未充分回答的开放问题（最多3个）
- {link:页码} — 标注关键内容在原文的位置
- {concept:简短解释} — 对专业术语给出通俗解释（最多5处）

## 输出格式（严格JSON，不要markdown代码块）
{
  "root": {"name": "核心主题"},
  "children": [
    {
      "name": "章节一 {bold}",
      "children": [
        {"name": "核心观点"},
        {"name": "关键数据: 增长30% {link:第3页}"},
        {"name": "术语解释 {concept:指某种现象}"}
      ]
    }
  ]
}

## 约束
- 总节点数 30-50 个
- 一层节点必须覆盖所有主要章节
- 若文档某部分内容不足以生成节点，不要为该部分编造节点
- 输出纯 JSON，不要包裹在 ```json 代码块中"""

MINDMAP_USER_PROMPT = """请基于以下文档内容，生成思维导图 JSON。

文档标题：{title}

{content}"""
```

- [ ] **Step 2: Commit**

```bash
git add backend/services/prompt_templates.py
git commit -m "feat: add report and mindmap prompt templates"
```

---

### Task 8: LLM 服务

**Files:**
- Create: `backend/services/llm_service.py`

- [ ] **Step 1: 编写 llm_service.py**

```python
import json
from openai import OpenAI

from config import settings
from models.document import MindmapNode, MindmapResult
from services.prompt_templates import (
    REPORT_SYSTEM_PROMPT,
    REPORT_USER_PROMPT,
    DEEP_DIVE_SYSTEM_PROMPT,
    DEEP_DIVE_USER_PROMPT,
    MINDMAP_SYSTEM_PROMPT,
    MINDMAP_USER_PROMPT,
)
from storage.chroma_client import chroma_client


class LLMService:
    def __init__(self):
        if settings.llm_provider == "qwen":
            self.client = OpenAI(
                api_key=settings.qwen_api_key,
                base_url=settings.qwen_base_url,
            )
            self.model = settings.qwen_model
        else:
            self.client = OpenAI(
                api_key=settings.deepseek_api_key,
                base_url=settings.deepseek_base_url,
            )
            self.model = settings.deepseek_model

    def _build_context(self, chunks: list[dict], max_tokens: int = 12000) -> str:
        """Assemble retrieval results into context, respecting token budget."""
        parts = []
        total_len = 0
        for chunk in chunks:
            meta = chunk["metadata"]
            heading = meta.get("heading_path", "")
            page = meta.get("page", "")
            header = f"[{heading}]" if heading else ""
            if page:
                header += f" (第{page}页)"
            block = f"{header}\n{chunk['text']}"
            # Rough estimate: 1 char ≈ 0.5 token for Chinese
            if total_len + len(block) // 2 > max_tokens:
                break
            parts.append(block)
            total_len += len(block) // 2
        return "\n\n---\n\n".join(parts)

    def generate_report(
        self, doc_id: str, title: str, fmt: str, pages: int, query_embedding: list[float]
    ) -> str:
        chunks = chroma_client.query(doc_id, query_embedding, top_k=30)
        if not chunks:
            return "无法检索到文档内容，请确认文档已正确解析。"

        context = self._build_context(chunks)
        user_prompt = REPORT_USER_PROMPT.format(
            title=title, format=fmt, pages=pages, content=context
        )

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": REPORT_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=4096,
        )
        return resp.choices[0].message.content or ""

    def generate_mindmap(
        self, doc_id: str, title: str, query_embedding: list[float]
    ) -> MindmapResult:
        chunks = chroma_client.query(doc_id, query_embedding, top_k=30)
        if not chunks:
            # Fallback: empty mindmap
            return MindmapResult(root=MindmapNode(name="文档内容为空"))

        context = self._build_context(chunks)
        user_prompt = MINDMAP_USER_PROMPT.format(title=title, content=context)

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": MINDMAP_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=4096,
        )
        raw = resp.choices[0].message.content or "{}"
        return self._parse_mindmap_json(raw)

    def _parse_mindmap_json(self, raw: str) -> MindmapResult:
        # Strip markdown code fences if present
        raw = raw.strip()
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        try:
            data = json.loads(raw)
            root = self._dict_to_node(data)
            return MindmapResult(root=root)
        except (json.JSONDecodeError, KeyError):
            return MindmapResult(root=MindmapNode(name="思维导图生成失败，请重试"))

    def _dict_to_node(self, data: dict) -> MindmapNode:
        children = [self._dict_to_node(c) for c in data.get("children", [])]
        return MindmapNode(
            name=data.get("name", ""),
            meta=data.get("meta"),
            children=children,
        )

    def generate_deep_dive(
        self,
        doc_id: str,
        title: str,
        selected_sections: list[str],
        query_embedding: list[float],
    ) -> str:
        """长文档第二阶段：对用户选中的章节做深度分析"""
        chunks = chroma_client.query(doc_id, query_embedding, top_k=50)
        if not chunks:
            return "无法检索到文档内容。"
        context = self._build_context(chunks, max_tokens=16000)
        user_prompt = DEEP_DIVE_USER_PROMPT.format(
            title=title,
            selected_sections=", ".join(selected_sections),
            content=context,
        )
        system_prompt = DEEP_DIVE_SYSTEM_PROMPT.replace(
            "{selected_sections}", ", ".join(selected_sections)
        )
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=4096,
        )
        return resp.choices[0].message.content or ""


llm_service = LLMService()
```

- [ ] **Step 2: Commit**

```bash
git add backend/services/llm_service.py
git commit -m "feat: add LLM service for report and mindmap generation"
```

---

### Task 9: API 路由

**Files:**
- Create: `backend/routers/documents.py`

- [ ] **Step 1: 编写 routers/documents.py**

```python
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from config import settings
from models.document import (
    DocumentFormat,
    DocumentRecord,
    DocumentStatus,
    MindmapResult,
    ReportResult,
)
from services.parser import document_parser
from services.chunker import chunker
from services.llm_service import llm_service
from storage.chroma_client import chroma_client
from storage.file_store import file_store

router = APIRouter(prefix="/api/documents", tags=["documents"])

# In-memory registry (MVP: no database)
_documents: dict[str, DocumentRecord] = {}
_reports: dict[str, ReportResult] = {}
_mindmaps: dict[str, MindmapResult] = {}


def _build_doc_metadata(doc_id: str, filename: str, file_path: Path) -> DocumentRecord:
    fmt = document_parser._detect_format(filename)
    file_size = file_path.stat().st_size

    from models.document import DocumentMetadata
    metadata = DocumentMetadata(filename=filename, format=fmt, file_size=file_size)

    record = DocumentRecord(id=doc_id, metadata=metadata)
    _documents[doc_id] = record
    return record


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(400, "Filename is required")

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(400, "File is empty")

    doc_id, file_path = await file_store.save(file.filename, content)

    record = _build_doc_metadata(doc_id, file.filename, file_path)

    # Parse asynchronously — update status as we go
    try:
        record.status = DocumentStatus.PARSING
        structured = document_parser.parse(file_path, file.filename)
        record.structured = structured
        record.title = structured.title
        record.metadata.pages = structured.metadata.pages

        record.status = DocumentStatus.CHUNKING
        chunks = chunker.chunk(structured)
        texts = [c["text"] for c in chunks]
        embeddings = chunker.embed(texts)

        collection = chroma_client.create_collection(doc_id)
        chroma_client.add_chunks(collection, chunks, embeddings)

        record.status = DocumentStatus.READY
    except Exception as e:
        record.status = DocumentStatus.ERROR
        raise HTTPException(500, f"Processing failed: {str(e)}")

    return {
        "id": record.id,
        "title": record.title,
        "status": record.status.value,
        "metadata": record.metadata.model_dump() if record.metadata else None,
    }


@router.get("/")
async def list_documents():
    return [
        {
            "id": r.id,
            "title": r.title,
            "status": r.status.value,
            "created_at": r.created_at.isoformat(),
            "format": r.metadata.format.value if r.metadata else None,
            "pages": r.metadata.pages if r.metadata else 0,
        }
        for r in sorted(_documents.values(), key=lambda r: r.created_at, reverse=True)
    ]


@router.get("/{doc_id}")
async def get_document(doc_id: str):
    record = _documents.get(doc_id)
    if not record:
        raise HTTPException(404, "Document not found")
    return {
        "id": record.id,
        "title": record.title,
        "status": record.status.value,
        "created_at": record.created_at.isoformat(),
        "metadata": record.metadata.model_dump() if record.metadata else None,
        "structured": record.structured.model_dump() if record.structured else None,
    }


@router.delete("/{doc_id}")
async def delete_document(doc_id: str):
    record = _documents.pop(doc_id, None)
    if not record:
        raise HTTPException(404, "Document not found")
    _reports.pop(doc_id, None)
    _mindmaps.pop(doc_id, None)
    chroma_client.delete_collection(doc_id)
    if record.metadata:
        file_store.delete(doc_id, record.metadata.filename)
    return {"deleted": doc_id}


@router.post("/{doc_id}/report")
async def generate_report(doc_id: str):
    record = _documents.get(doc_id)
    if not record:
        raise HTTPException(404, "Document not found")
    if record.status != DocumentStatus.READY:
        raise HTTPException(400, "Document not ready, current status: " + record.status.value)
    if not record.structured:
        raise HTTPException(400, "Document has no parsed content")

    # Use a summary query to retrieve relevant chunks
    summary_text = record.structured.title
    for section in record.structured.sections[:3]:
        summary_text += " " + section.heading + " " + section.content[:200]
    query_embedding = chunker.embed([summary_text[:1000]])[0]

    content = llm_service.generate_report(
        doc_id=doc_id,
        title=record.title,
        fmt=record.metadata.format.value if record.metadata else "pdf",
        pages=record.metadata.pages if record.metadata else 0,
        query_embedding=query_embedding,
    )

    result = ReportResult(content=content)
    _reports[doc_id] = result
    return {"content": result.content, "generated_at": result.generated_at.isoformat()}


@router.get("/{doc_id}/report")
async def get_report(doc_id: str):
    result = _reports.get(doc_id)
    if not result:
        raise HTTPException(404, "Report not generated yet")
    return {"content": result.content, "generated_at": result.generated_at.isoformat()}


@router.post("/{doc_id}/mindmap")
async def generate_mindmap(doc_id: str):
    record = _documents.get(doc_id)
    if not record:
        raise HTTPException(404, "Document not found")
    if record.status != DocumentStatus.READY:
        raise HTTPException(400, "Document not ready, current status: " + record.status.value)

    summary_text = record.title
    for section in (record.structured.sections if record.structured else [])[:3]:
        summary_text += " " + section.heading
    query_embedding = chunker.embed([summary_text[:1000]])[0]

    result = llm_service.generate_mindmap(
        doc_id=doc_id, title=record.title, query_embedding=query_embedding
    )
    _mindmaps[doc_id] = result
    return result.model_dump()


@router.get("/{doc_id}/mindmap")
async def get_mindmap(doc_id: str):
    result = _mindmaps.get(doc_id)
    if not result:
        raise HTTPException(404, "Mindmap not generated yet")
    return result.model_dump()


@router.post("/{doc_id}/report/deep-dive")
async def deep_dive_report(doc_id: str, body: dict):
    record = _documents.get(doc_id)
    if not record:
        raise HTTPException(404, "Document not found")
    sections = body.get("sections", [])
    if not sections:
        raise HTTPException(400, "sections is required")

    summary_text = record.title
    for section in (record.structured.sections if record.structured else [])[:3]:
        summary_text += " " + section.heading
    query_embedding = chunker.embed([summary_text[:1000]])[0]

    content = llm_service.generate_deep_dive(
        doc_id=doc_id,
        title=record.title,
        selected_sections=sections,
        query_embedding=query_embedding,
    )
    # Append to existing report
    existing = _reports.get(doc_id)
    if existing:
        existing.content += "\n\n---\n\n" + content
    else:
        _reports[doc_id] = ReportResult(content=content)
    return {"content": content}
```

- [ ] **Step 2: Commit**

```bash
git add backend/routers/documents.py
git commit -m "feat: add document API routes with adaptive deep-dive endpoint"
```

---

### Task 10: FastAPI 入口

**Files:**
- Create: `backend/main.py`

- [ ] **Step 1: 编写 main.py**

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from routers.documents import router as documents_router

app = FastAPI(title="文档理解 RAG 平台", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents_router)


@app.get("/api/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=settings.host, port=settings.port, reload=True)
```

- [ ] **Step 2: 验证后端启动**

```bash
cd backend && python -c "from main import app; print('FastAPI app loaded OK')"
```

- [ ] **Step 3: Commit**

```bash
git add backend/main.py
git commit -m "feat: add FastAPI entry point with CORS and router"
```

---

### Task 11: 前端项目脚手架

**Files:**
- Create: `frontend/package.json`, `frontend/tsconfig.json`, `frontend/tsconfig.app.json`, `frontend/tsconfig.node.json`, `frontend/vite.config.ts`, `frontend/index.html`, `frontend/src/main.tsx`, `frontend/src/App.css`

- [ ] **Step 1: 创建前端目录**

```bash
mkdir -p frontend/src/pages frontend/src/components frontend/src/services
```

- [ ] **Step 2: 编写 package.json**

```json
{
  "name": "doc-rag-frontend",
  "private": true,
  "version": "0.1.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc -b && vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "markmap-view": "^0.17.0",
    "markmap-toolbar": "^0.17.0",
    "react": "^18.3.1",
    "react-dom": "^18.3.1",
    "react-markdown": "^9.0.1",
    "react-router-dom": "^6.26.2",
    "rehype-highlight": "^7.0.0",
    "rehype-raw": "^7.0.0",
    "remark-gfm": "^4.0.0"
  },
  "devDependencies": {
    "@types/react": "^18.3.11",
    "@types/react-dom": "^18.3.1",
    "@vitejs/plugin-react": "^4.3.2",
    "typescript": "^5.6.3",
    "vite": "^5.4.9"
  }
}
```

- [ ] **Step 3: 编写 tsconfig.json**

```json
{
  "files": [],
  "references": [
    { "path": "./tsconfig.app.json" },
    { "path": "./tsconfig.node.json" }
  ]
}
```

- [ ] **Step 4: 编写 tsconfig.app.json**

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "isolatedModules": true,
    "moduleDetection": "force",
    "noEmit": true,
    "jsx": "react-jsx",
    "strict": true,
    "noUnusedLocals": false,
    "noUnusedParameters": false,
    "noFallthroughCasesInSwitch": true
  },
  "include": ["src"]
}
```

- [ ] **Step 5: 编写 tsconfig.node.json**

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "lib": ["ES2023"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "isolatedModules": true,
    "moduleDetection": "force",
    "noEmit": true,
    "strict": true
  },
  "include": ["vite.config.ts"]
}
```

- [ ] **Step 6: 编写 vite.config.ts**

```typescript
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": "http://localhost:8000",
    },
  },
});
```

- [ ] **Step 7: 编写 index.html**

```html
<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>文档理解 RAG 平台</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
```

- [ ] **Step 8: 编写 src/main.tsx**

```tsx
import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "./App.css";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
```

- [ ] **Step 9: 编写 src/App.css（全局样式）**

```css
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #f5f7fa; color: #1a1a2e; }
a { color: #4f46e5; text-decoration: none; }
a:hover { text-decoration: underline; }
.container { max-width: 1200px; margin: 0 auto; padding: 24px; }
```

- [ ] **Step 10: 安装前端依赖**

```bash
cd frontend && npm install
```

- [ ] **Step 11: Commit**

```bash
git add frontend/
git commit -m "feat: scaffold React + TypeScript + Vite frontend"
```

---

### Task 12: 前端 API 服务层

**Files:**
- Create: `frontend/src/services/api.ts`

- [ ] **Step 1: 编写 api.ts**

```typescript
const BASE = "/api";

export interface DocListItem {
  id: string;
  title: string;
  status: string;
  created_at: string;
  format: string | null;
  pages: number;
}

export interface DocDetail extends DocListItem {
  metadata: {
    filename: string;
    format: string;
    pages: number;
    uploaded_at: string;
    file_size: number;
  } | null;
  structured: StructuredDoc | null;
}

export interface StructuredDoc {
  title: string;
  sections: Section[];
}

export interface Section {
  heading: string;
  level: number;
  content: string;
  page: number;
}

export interface ReportData {
  content: string;
  generated_at: string;
}

export interface MindmapNode {
  name: string;
  meta?: string | null;
  children: MindmapNode[];
}

export interface MindmapData {
  root: MindmapNode;
  generated_at: string;
}

export async function uploadDocument(file: File): Promise<DocListItem> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${BASE}/documents/upload`, { method: "POST", body: form });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || "Upload failed");
  }
  return res.json();
}

export async function listDocuments(): Promise<DocListItem[]> {
  const res = await fetch(`${BASE}/documents/`);
  if (!res.ok) throw new Error("Failed to fetch documents");
  return res.json();
}

export async function getDocument(id: string): Promise<DocDetail> {
  const res = await fetch(`${BASE}/documents/${id}`);
  if (!res.ok) throw new Error("Document not found");
  return res.json();
}

export async function deleteDocument(id: string): Promise<void> {
  const res = await fetch(`${BASE}/documents/${id}`, { method: "DELETE" });
  if (!res.ok) throw new Error("Failed to delete");
}

export async function generateReport(id: string): Promise<ReportData> {
  const res = await fetch(`${BASE}/documents/${id}/report`, { method: "POST" });
  if (!res.ok) throw new Error("Failed to generate report");
  return res.json();
}

export async function getReport(id: string): Promise<ReportData> {
  const res = await fetch(`${BASE}/documents/${id}/report`);
  if (!res.ok) throw new Error("Report not found");
  return res.json();
}

export async function deepDiveReport(id: string, sections: string[]): Promise<ReportData> {
  const res = await fetch(`${BASE}/documents/${id}/report/deep-dive`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sections }),
  });
  if (!res.ok) throw new Error("Failed to generate deep-dive");
  return res.json();
}

export async function generateMindmap(id: string): Promise<MindmapData> {
  const res = await fetch(`${BASE}/documents/${id}/mindmap`, { method: "POST" });
  if (!res.ok) throw new Error("Failed to generate mindmap");
  return res.json();
}

export async function getMindmap(id: string): Promise<MindmapData> {
  const res = await fetch(`${BASE}/documents/${id}/mindmap`);
  if (!res.ok) throw new Error("Mindmap not found");
  return res.json();
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/services/api.ts
git commit -m "feat: add frontend API service layer"
```

---

### Task 13: 文档列表页 + 文件上传

**Files:**
- Create: `frontend/src/components/FileUpload.tsx`
- Create: `frontend/src/components/FileUpload.css`
- Create: `frontend/src/pages/DocumentList.tsx`
- Create: `frontend/src/pages/DocumentList.css`

- [ ] **Step 1: 编写 FileUpload.tsx**

```tsx
import { useRef, useState } from "react";
import "./FileUpload.css";

interface Props {
  onUploaded: () => void;
}

export default function FileUpload({ onUploaded }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState("");

  const ALLOWED = [".pdf", ".docx", ".doc", ".md", ".markdown"];

  async function handleFile(file: File) {
    const ext = "." + file.name.split(".").pop()?.toLowerCase();
    if (!ALLOWED.includes(ext)) {
      setError(`不支持的文件格式: ${ext}，仅支持 PDF、Word、Markdown`);
      return;
    }
    setError("");
    setUploading(true);
    try {
      const { uploadDocument } = await import("../services/api");
      await uploadDocument(file);
      onUploaded();
    } catch (e: any) {
      setError(e.message || "上传失败");
    } finally {
      setUploading(false);
    }
  }

  return (
    <div className="file-upload">
      <div
        className={`drop-zone ${uploading ? "uploading" : ""}`}
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => {
          e.preventDefault();
          const file = e.dataTransfer.files[0];
          if (file) handleFile(file);
        }}
      >
        {uploading ? (
          <p>正在上传并解析文档...</p>
        ) : (
          <>
            <p className="drop-text">拖拽文档到此处，或点击选择</p>
            <p className="drop-hint">支持 PDF、Word (.docx)、Markdown (.md)</p>
          </>
        )}
        <input
          ref={inputRef}
          type="file"
          accept=".pdf,.docx,.doc,.md,.markdown"
          style={{ display: "none" }}
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) handleFile(file);
            e.target.value = "";
          }}
        />
      </div>
      {error && <p className="upload-error">{error}</p>}
    </div>
  );
}
```

- [ ] **Step 2: 编写 FileUpload.css**

```css
.file-upload { margin-bottom: 24px; }
.drop-zone {
  border: 2px dashed #cbd5e1;
  border-radius: 12px;
  padding: 48px;
  text-align: center;
  cursor: pointer;
  transition: all 0.2s;
  background: #fff;
}
.drop-zone:hover { border-color: #4f46e5; background: #f0f0ff; }
.drop-zone.uploading { opacity: 0.6; pointer-events: none; }
.drop-text { font-size: 16px; color: #475569; margin-bottom: 4px; }
.drop-hint { font-size: 13px; color: #94a3b8; }
.upload-error { color: #dc2626; font-size: 13px; margin-top: 8px; }
```

- [ ] **Step 3: 编写 DocumentList.tsx**

```tsx
import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import FileUpload from "../components/FileUpload";
import { listDocuments, deleteDocument, DocListItem } from "../services/api";
import "./DocumentList.css";

export default function DocumentList() {
  const [docs, setDocs] = useState<DocListItem[]>([]);
  const [loading, setLoading] = useState(true);

  async function fetchDocs() {
    setLoading(true);
    try {
      const data = await listDocuments();
      setDocs(data);
    } catch {
      // silently fail, show empty
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => { fetchDocs(); }, []);

  async function handleDelete(id: string) {
    if (!confirm("确定删除此文档？")) return;
    try {
      await deleteDocument(id);
      setDocs((prev) => prev.filter((d) => d.id !== id));
    } catch {
      alert("删除失败");
    }
  }

  const statusLabel: Record<string, string> = {
    uploaded: "已上传",
    parsing: "解析中",
    chunking: "分块中",
    ready: "就绪",
    error: "错误",
  };

  return (
    <div className="container">
      <h1 className="page-title">文档理解 RAG 平台</h1>
      <FileUpload onUploaded={fetchDocs} />

      {loading ? (
        <p className="loading">加载中...</p>
      ) : docs.length === 0 ? (
        <p className="empty">暂无文档，上传一份开始吧。</p>
      ) : (
        <div className="doc-list">
          {docs.map((doc) => (
            <div key={doc.id} className="doc-card">
              <div className="doc-info">
                <h3 className="doc-title">
                  <Link to={`/documents/${doc.id}/report`}>
                    {doc.title || doc.id}
                  </Link>
                </h3>
                <p className="doc-meta">
                  {doc.format?.toUpperCase()} · {doc.pages}页 ·{" "}
                  {new Date(doc.created_at).toLocaleDateString("zh-CN")}
                </p>
              </div>
              <div className="doc-actions">
                <span className={`status-badge status-${doc.status}`}>
                  {statusLabel[doc.status] || doc.status}
                </span>
                {doc.status === "ready" && (
                  <>
                    <Link
                      to={`/documents/${doc.id}/report`}
                      className="btn btn-sm"
                    >
                      查看报告
                    </Link>
                    <Link
                      to={`/documents/${doc.id}/mindmap`}
                      className="btn btn-sm btn-outline"
                    >
                      思维导图
                    </Link>
                  </>
                )}
                <button
                  className="btn btn-sm btn-danger"
                  onClick={() => handleDelete(doc.id)}
                >
                  删除
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 4: 编写 DocumentList.css**

```css
.page-title { font-size: 28px; margin-bottom: 24px; color: #1e293b; }
.loading, .empty { text-align: center; color: #94a3b8; padding: 48px 0; }
.doc-list { display: flex; flex-direction: column; gap: 12px; }
.doc-card {
  display: flex; justify-content: space-between; align-items: center;
  background: #fff; border-radius: 10px; padding: 20px 24px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
.doc-title { font-size: 16px; margin-bottom: 4px; }
.doc-meta { font-size: 13px; color: #64748b; }
.doc-actions { display: flex; align-items: center; gap: 8px; }
.status-badge { font-size: 12px; padding: 3px 10px; border-radius: 12px; font-weight: 500; }
.status-ready { background: #dcfce7; color: #166534; }
.status-uploaded, .status-parsing, .status-chunking { background: #fef9c3; color: #854d0e; }
.status-error { background: #fee2e2; color: #991b1b; }
.btn { padding: 6px 14px; border-radius: 6px; border: none; cursor: pointer; font-size: 13px; background: #4f46e5; color: #fff; display: inline-block; }
.btn:hover { background: #4338ca; text-decoration: none; }
.btn-sm { padding: 4px 12px; font-size: 12px; }
.btn-outline { background: transparent; border: 1px solid #cbd5e1; color: #475569; }
.btn-outline:hover { background: #f1f5f9; }
.btn-danger { background: transparent; border: 1px solid #fecaca; color: #dc2626; }
.btn-danger:hover { background: #fef2f2; }
```

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/FileUpload.tsx frontend/src/components/FileUpload.css frontend/src/pages/DocumentList.tsx frontend/src/pages/DocumentList.css
git commit -m "feat: add document list page with file upload"
```

---

### Task 14: 报告页 + 评分卡片

**Files:**
- Create: `frontend/src/components/ScoreCard.tsx`
- Create: `frontend/src/components/ScoreCard.css`
- Create: `frontend/src/components/ReportViewer.tsx`
- Create: `frontend/src/pages/DocumentReport.tsx`
- Create: `frontend/src/pages/DocumentReport.css`

- [ ] **Step 1: 编写 ScoreCard.tsx**

```tsx
import "./ScoreCard.css";

interface Props {
  content: string;
}

interface Scores {
  evidence: number;
  logic: number;
  data: number;
}

function extractScores(markdown: string): Scores | null {
  const pattern = /论据质量[：:]\s*(\d+).*?逻辑严密性[：:]\s*(\d+).*?数据可靠性[：:]\s*(\d+)/s;
  const match = markdown.match(pattern);
  if (match) {
    return {
      evidence: parseInt(match[1]),
      logic: parseInt(match[2]),
      data: parseInt(match[3]),
    };
  }
  return null;
}

export default function ScoreCard({ content }: Props) {
  const scores = extractScores(content);
  if (!scores) return null;

  const items = [
    { label: "论据质量", value: scores.evidence },
    { label: "逻辑严密性", value: scores.logic },
    { label: "数据可靠性", value: scores.data },
  ];

  return (
    <div className="score-card">
      <h4>可信度评分</h4>
      {items.map((item) => (
        <div key={item.label} className="score-item">
          <span className="score-label">{item.label}</span>
          <div className="score-bar-bg">
            <div
              className="score-bar-fill"
              style={{ width: `${(item.value / 10) * 100}%` }}
            />
          </div>
          <span className="score-value">{item.value}/10</span>
        </div>
      ))}
    </div>
  );
}
```

- [ ] **Step 2: 编写 ScoreCard.css**

```css
.score-card {
  background: #fff; border-radius: 10px; padding: 20px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.06); margin-bottom: 20px;
}
.score-card h4 { font-size: 14px; color: #475569; margin-bottom: 12px; }
.score-item { display: flex; align-items: center; gap: 10px; margin-bottom: 8px; }
.score-label { width: 80px; font-size: 13px; color: #64748b; }
.score-bar-bg { flex: 1; height: 6px; background: #e2e8f0; border-radius: 3px; overflow: hidden; }
.score-bar-fill { height: 100%; background: #4f46e5; border-radius: 3px; transition: width 0.5s; }
.score-value { font-size: 13px; font-weight: 600; color: #1e293b; width: 40px; }
```

- [ ] **Step 3: 编写 ReportViewer.tsx**

```tsx
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeRaw from "rehype-raw";

interface Props {
  content: string;
}

export default function ReportViewer({ content }: Props) {
  return (
    <div className="report-viewer">
      <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeRaw]}>
        {content}
      </ReactMarkdown>
    </div>
  );
}
```

- [ ] **Step 4: 编写 DocumentReport.tsx**

```tsx
import { useEffect, useState } from "react";
import { useParams, Link } from "react-router-dom";
import ScoreCard from "../components/ScoreCard";
import ReportViewer from "../components/ReportViewer";
import { getDocument, generateReport, getReport, deepDiveReport, DocDetail, ReportData } from "../services/api";
import "./DocumentReport.css";

export default function DocumentReport() {
  const { id } = useParams<{ id: string }>();
  const [doc, setDoc] = useState<DocDetail | null>(null);
  const [report, setReport] = useState<ReportData | null>(null);
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState("");
  const [selectedSections, setSelectedSections] = useState<string[]>([]);
  const [diveGenerating, setDiveGenerating] = useState(false);

  useEffect(() => {
    if (!id) return;
    getDocument(id).then(setDoc).catch(() => setError("文档加载失败"));
    getReport(id)
      .then(setReport)
      .catch(() => { /* not generated yet */ });
  }, [id]);

  async function handleGenerate() {
    if (!id) return;
    setGenerating(true);
    setError("");
    try {
      const result = await generateReport(id);
      setReport(result);
    } catch (e: any) {
      setError(e.message || "生成失败");
    } finally {
      setGenerating(false);
    }
  }

  // Parse section table from long document report to extract selectable chapters
  function parseSections(content: string): { name: string; range: string; summary: string; stars: number }[] {
    const sections: { name: string; range: string; summary: string; stars: number }[] = [];
    const lines = content.split("\n");
    let inTable = false;
    for (const line of lines) {
      if (line.includes("|") && (line.includes("章节") || line.includes("页码") || line.includes("---"))) {
        inTable = true;
        continue;
      }
      if (inTable && line.startsWith("|") && !line.includes("---")) {
        const cells = line.split("|").map(c => c.trim()).filter(Boolean);
        if (cells.length >= 2) {
          const stars = (cells[3] || "").split("★").length - 1;
          sections.push({
            name: cells[0].replace(/[*_]/g, "").trim(),
            range: cells[1] || "",
            summary: cells[2] || "",
            stars: stars > 0 ? stars : 3,
          });
        }
      } else if (inTable && !line.startsWith("|")) {
        inTable = false;
      }
    }
    return sections;
  }

  async function handleDeepDive() {
    if (!id || selectedSections.length === 0) return;
    setDiveGenerating(true);
    setError("");
    try {
      const result = await deepDiveReport(id, selectedSections);
      setReport((prev) =>
        prev ? { ...prev, content: prev.content + "\n\n---\n\n## 深度分析\n\n" + result.content } : result
      );
    } catch (e: any) {
      setError(e.message || "深度分析失败");
    } finally {
      setDiveGenerating(false);
    }
  }

  if (!id) return <p>无效的文档 ID</p>;

  const sections = report ? parseSections(report.content) : [];
  const isLongDoc = sections.length >= 4; // long doc has structure skeleton table

  return (
    <div className="container">
      <div className="page-header">
        <Link to="/" className="back-link">← 返回列表</Link>
        <h1 className="page-title">{doc?.title || "阅读报告"}</h1>
      </div>

      {error && <p className="error-msg">{error}</p>}

      {!report ? (
        <div className="no-report">
          <p>尚未生成报告</p>
          <button className="btn" onClick={handleGenerate} disabled={generating}>
            {generating ? "生成中..." : "生成阅读报告"}
          </button>
        </div>
      ) : (
        <div className="report-layout">
          <div className="report-sidebar">
            <ScoreCard content={report.content} />
            <button
              className="btn btn-outline btn-full"
              onClick={handleGenerate}
              disabled={generating}
            >
              {generating ? "生成中..." : "重新生成"}
            </button>

            {isLongDoc && (
              <div className="section-picker">
                <h4>选择深入分析的章节</h4>
                {sections.map((sec) => (
                  <label key={sec.name} className="section-checkbox">
                    <input
                      type="checkbox"
                      checked={selectedSections.includes(sec.name)}
                      onChange={(e) => {
                        if (e.target.checked) {
                          setSelectedSections((prev) => [...prev, sec.name]);
                        } else {
                          setSelectedSections((prev) => prev.filter((s) => s !== sec.name));
                        }
                      }}
                    />
                    <span>
                      {sec.name}
                      <span className="section-stars">{"★".repeat(sec.stars)}</span>
                    </span>
                  </label>
                ))}
                <button
                  className="btn btn-sm btn-full"
                  style={{ marginTop: 8 }}
                  onClick={handleDeepDive}
                  disabled={diveGenerating || selectedSections.length === 0}
                >
                  {diveGenerating ? "分析中..." : "对选中章节深度分析"}
                </button>
              </div>
            )}
          </div>
          <div className="report-main">
            <ReportViewer content={report.content} />
          </div>
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 5: 编写 DocumentReport.css**

```css
.page-header { margin-bottom: 24px; }
.back-link { font-size: 14px; color: #64748b; }
.back-link:hover { color: #4f46e5; }
.no-report { text-align: center; padding: 64px 0; }
.no-report p { color: #94a3b8; margin-bottom: 16px; }
.error-msg { color: #dc2626; padding: 12px; background: #fef2f2; border-radius: 8px; margin-bottom: 16px; }
.report-layout { display: flex; gap: 24px; }
.report-sidebar { width: 260px; flex-shrink: 0; }
.report-main { flex: 1; min-width: 0; background: #fff; border-radius: 10px; padding: 32px; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }
.btn-full { width: 100%; margin-top: 12px; }

.report-viewer h1 { font-size: 22px; margin: 16px 0 12px; }
.report-viewer h2 { font-size: 18px; margin: 20px 0 10px; padding-bottom: 8px; border-bottom: 1px solid #e2e8f0; }
.report-viewer h3 { font-size: 16px; margin: 16px 0 8px; }
.report-viewer p { line-height: 1.8; margin-bottom: 12px; color: #334155; }
.report-viewer ul, .report-viewer ol { padding-left: 20px; margin-bottom: 12px; }
.report-viewer li { line-height: 1.7; margin-bottom: 4px; }
.report-viewer table { width: 100%; border-collapse: collapse; margin: 12px 0; font-size: 14px; }
.report-viewer th, .report-viewer td { border: 1px solid #e2e8f0; padding: 8px 12px; text-align: left; }
.report-viewer th { background: #f8fafc; font-weight: 600; }
.report-viewer strong { color: #1e293b; }
.report-viewer code { background: #f1f5f9; padding: 2px 6px; border-radius: 4px; font-size: 13px; }

.section-picker { margin-top: 16px; padding: 12px; background: #f8fafc; border-radius: 8px; }
.section-picker h4 { font-size: 13px; color: #475569; margin-bottom: 8px; }
.section-checkbox { display: flex; align-items: center; gap: 6px; padding: 4px 0; font-size: 12px; cursor: pointer; }
.section-checkbox input { cursor: pointer; }
.section-stars { color: #eab308; margin-left: 4px; font-size: 11px; }
```

- [ ] **Step 6: Commit**

```bash
git add frontend/src/components/ScoreCard.tsx frontend/src/components/ScoreCard.css frontend/src/components/ReportViewer.tsx frontend/src/pages/DocumentReport.tsx frontend/src/pages/DocumentReport.css
git commit -m "feat: add report viewer page with credibility score card"
```

---

### Task 15: 思维导图页

**Files:**
- Create: `frontend/src/components/MindmapViewer.tsx`
- Create: `frontend/src/components/MindmapViewer.css`
- Create: `frontend/src/pages/DocumentMindmap.tsx`
- Create: `frontend/src/pages/DocumentMindmap.css`

- [ ] **Step 1: 编写 MindmapViewer.tsx**

```tsx
import { useEffect, useRef } from "react";
import { Transformer } from "markmap-lib";
import { Markmap } from "markmap-view";
import { MindmapNode } from "../services/api";
import "./MindmapViewer.css";

const transformer = new Transformer();

interface Props {
  root: MindmapNode;
}

function toMarkmapNode(node: MindmapNode): any {
  const payload: any = {};
  if (node.meta) {
    const meta = node.meta;
    if (meta.includes("{bold}")) payload.className = "mm-bold";
    if (meta.includes("{warning}")) payload.className = "mm-warning";
    if (meta.includes("{question}")) payload.className = "mm-question";
    if (meta.includes("{link")) {
      const m = meta.match(/\{link:([^}]+)\}/);
      if (m) payload.note = `原文: ${m[1]}`;
    }
    if (meta.includes("{concept")) {
      const m = meta.match(/\{concept:([^}]+)\}/);
      if (m) payload.note = m[1];
    }
  }
  return {
    content: node.name,
    children: node.children.map(toMarkmapNode),
    payload,
  };
}

export default function MindmapViewer({ root }: Props) {
  const svgRef = useRef<SVGSVGElement>(null);
  const mmRef = useRef<Markmap | null>(null);

  useEffect(() => {
    if (!svgRef.current) return;

    const data = toMarkmapNode(root);
    const { root: markmapRoot } = transformer.transform(data);

    if (mmRef.current) {
      mmRef.current.setData(markmapRoot);
    } else {
      mmRef.current = Markmap.create(svgRef.current, {
        autoFit: true,
        fitRatio: 0.85,
        duration: 400,
      }, markmapRoot);
    }

    return () => {
      // cleanup handled by React
    };
  }, [root]);

  return (
    <div className="mindmap-container">
      <svg ref={svgRef} className="mindmap-svg" />
    </div>
  );
}
```

- [ ] **Step 2: 编写 MindmapViewer.css**

```css
.mindmap-container { width: 100%; height: 70vh; border-radius: 10px; overflow: hidden; background: #fff; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }
.mindmap-svg { width: 100%; height: 100%; }

/* Node styling via markmap CSS custom properties */
.mindmap-container svg g.markmap-node.mm-bold > rect { fill: #ede9fe; stroke: #4f46e5; stroke-width: 2; }
.mindmap-container svg g.markmap-node.mm-warning > rect { fill: #fef9c3; }
.mindmap-container svg g.markmap-node.mm-warning > circle { fill: #eab308; }
.mindmap-container svg g.markmap-node.mm-question > rect { fill: #fce7f3; }
```

- [ ] **Step 3: 编写 DocumentMindmap.tsx**

```tsx
import { useEffect, useState } from "react";
import { useParams, Link } from "react-router-dom";
import MindmapViewer from "../components/MindmapViewer";
import { getDocument, generateMindmap, getMindmap, DocDetail, MindmapData } from "../services/api";
import "./DocumentMindmap.css";

export default function DocumentMindmap() {
  const { id } = useParams<{ id: string }>();
  const [doc, setDoc] = useState<DocDetail | null>(null);
  const [mindmap, setMindmap] = useState<MindmapData | null>(null);
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!id) return;
    getDocument(id).then(setDoc).catch(() => setError("文档加载失败"));
    getMindmap(id)
      .then(setMindmap)
      .catch(() => { /* not generated yet */ });
  }, [id]);

  async function handleGenerate() {
    if (!id) return;
    setGenerating(true);
    setError("");
    try {
      const result = await generateMindmap(id);
      setMindmap(result);
    } catch (e: any) {
      setError(e.message || "生成失败");
    } finally {
      setGenerating(false);
    }
  }

  if (!id) return <p>无效的文档 ID</p>;

  return (
    <div className="container">
      <div className="page-header">
        <Link to="/" className="back-link">← 返回列表</Link>
        <h1 className="page-title">{doc?.title || "思维导图"}</h1>
      </div>

      {error && <p className="error-msg">{error}</p>}

      {!mindmap ? (
        <div className="no-report">
          <p>尚未生成思维导图</p>
          <button className="btn" onClick={handleGenerate} disabled={generating}>
            {generating ? "生成中..." : "生成思维导图"}
          </button>
        </div>
      ) : (
        <>
          <div style={{ marginBottom: 12, display: "flex", gap: 8 }}>
            <button
              className="btn btn-outline btn-sm"
              onClick={handleGenerate}
              disabled={generating}
            >
              {generating ? "生成中..." : "重新生成"}
            </button>
            <span className="hint-text">
              蓝色外框 = 重点关注 · 黄色 = 存疑 · 粉色 = 待解答
            </span>
          </div>
          <MindmapViewer root={mindmap.root} />
        </>
      )}
    </div>
  );
}
```

- [ ] **Step 4: 编写 DocumentMindmap.css**

```css
.hint-text { font-size: 12px; color: #94a3b8; align-self: center; }
```

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/MindmapViewer.tsx frontend/src/components/MindmapViewer.css frontend/src/pages/DocumentMindmap.tsx frontend/src/pages/DocumentMindmap.css
git commit -m "feat: add interactive mindmap viewer with markmap"
```

---

### Task 16: App 路由

**Files:**
- Modify: `frontend/src/App.tsx`

- [ ] **Step 1: 编写 App.tsx**

```tsx
import { BrowserRouter, Routes, Route } from "react-router-dom";
import DocumentList from "./pages/DocumentList";
import DocumentReport from "./pages/DocumentReport";
import DocumentMindmap from "./pages/DocumentMindmap";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<DocumentList />} />
        <Route path="/documents/:id/report" element={<DocumentReport />} />
        <Route path="/documents/:id/mindmap" element={<DocumentMindmap />} />
      </Routes>
    </BrowserRouter>
  );
}
```

- [ ] **Step 2: 验证前端构建**

```bash
cd frontend && npx tsc -b --noEmit 2>&1 | head -20
```

- [ ] **Step 3: Commit**

```bash
git add frontend/src/App.tsx
git commit -m "feat: add App routing with React Router"
```

---

## 验证计划

完成所有 Task 后，按以下步骤验证：

1. **启动后端：**
   ```bash
   cd backend && python main.py
   ```
   确认 `http://localhost:8000/api/health` 返回 `{"status": "ok"}`

2. **启动前端：**
   ```bash
   cd frontend && npm run dev
   ```
   确认 `http://localhost:5173` 能打开

3. **上传测试：** 上传一份 PDF/Word/Markdown 文件，确认状态变为"就绪"

4. **报告生成：** 点击"生成阅读报告"，确认返回结构化分析（含快速评估、内容拆解、批判性分析、读者指南）

5. **思维导图：** 点击"生成思维导图"，确认交互式导图可缩放和展开

6. **删除：** 删除文档后确认列表更新
