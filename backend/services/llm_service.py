import json
import re
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
    QA_SYSTEM_PROMPT,
    QA_USER_PROMPT,
)
from storage.chroma_client import chroma_client


def _sanitize_content(text: str) -> str:
    """Strip common prompt injection patterns from document content."""
    # Remove "ignore previous/above instructions" patterns
    text = re.sub(
        r'(?i)(忽略|无视|忘记|ignore|disregard|forget)\s*(上述|以上|之前|所有|all|previous|above)\s*(指令|指示|说明|规则|instructions?|directions?|rules?)',
        '[过滤]', text,
    )
    # Remove "you are now" / "your new role" patterns
    text = re.sub(
        r'(?i)(你现在是|你的新角色是|your new role is|you are now|act as|从现在开始你是)',
        '[过滤]', text,
    )
    # Remove "output only" / "only output" patterns
    text = re.sub(
        r'(?i)(只输出|仅输出|只返回|仅返回|output only|only output|just say|just output)',
        '[过滤]', text,
    )
    return text


_CONTENT_GUARD_PREFIX = "[文档内容开始]"
_CONTENT_GUARD_SUFFIX = "[文档内容结束]"


class LLMService:
    def __init__(self):
        self._client = None
        self._model = None

    @property
    def client(self):
        if self._client is None:
            if settings.llm_provider == "qwen":
                self._client = OpenAI(
                    api_key=settings.qwen_api_key,
                    base_url=settings.qwen_base_url,
                )
                self._model = settings.qwen_model
            else:
                self._client = OpenAI(
                    api_key=settings.deepseek_api_key,
                    base_url=settings.deepseek_base_url,
                )
                self._model = settings.deepseek_model
        return self._client

    @property
    def model(self):
        if self._model is None:
            _ = self.client  # trigger lazy init
        return self._model

    def _build_context(self, chunks: list[dict], max_tokens: int = 12000) -> str:
        # Apply diversity: group by top-level heading, then interleave
        groups: dict[str, list[dict]] = {}
        for c in chunks:
            heading = c.get("metadata", {}).get("heading_path", "")
            prefix = heading.split(" > ")[0] if heading else "_"
            groups.setdefault(prefix, []).append(c)

        # Interleave chunks from different groups for diversity
        reordered: list[dict] = []
        group_chunks = list(groups.values())
        max_grp = max(len(g) for g in group_chunks) if group_chunks else 0
        for i in range(max_grp):
            for g in group_chunks:
                if i < len(g):
                    reordered.append(g[i])

        parts = []
        total_len = 0
        for chunk in reordered:
            meta = chunk["metadata"]
            heading = meta.get("heading_path", "")
            page = meta.get("page", "")
            header = f"[{heading}]" if heading else ""
            if page:
                header += f" (第{page}页)"
            block = f"{header}\n{chunk['text']}"
            # Chinese: ~1.5 chars per token
            if total_len + len(block) // 1.5 > max_tokens:
                break
            parts.append(block)
            total_len += len(block) // 1.5
        content = "\n\n---\n\n".join(parts)
        return f"{_CONTENT_GUARD_PREFIX}\n{_sanitize_content(content)}\n{_CONTENT_GUARD_SUFFIX}"

    def _extract_text(self, resp) -> str:
        """Extract and clean response content, filtering reasoning/thinking blocks."""
        msg = resp.choices[0].message
        text = msg.content or ""
        # Strip <｜end▁of▁thinking｜> think blocks (DeepSeek-R1 style)
        text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        text = re.sub(r"<reasoning>.*?</reasoning>", "", text, flags=re.DOTALL)
        return text.strip()

    def generate_report(
        self, doc_id: str, title: str, fmt: str, pages: int, query_embedding: list[float]
    ) -> str:
        chunks = chroma_client.hybrid_query(doc_id, query_embedding, title, top_k=30)
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
            temperature=0,
            max_tokens=4096,
        )
        return self._extract_text(resp)

    def generate_mindmap(
        self, doc_id: str, title: str, query_embedding: list[float]
    ) -> MindmapResult:
        chunks = chroma_client.hybrid_query(doc_id, query_embedding, title, top_k=30)
        if not chunks:
            return MindmapResult(root=MindmapNode(name="文档内容为空"))

        context = self._build_context(chunks)
        user_prompt = MINDMAP_USER_PROMPT.format(title=title, content=context)

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": MINDMAP_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            max_tokens=4096,
        )
        raw = self._extract_text(resp) or "{}"
        return self._parse_mindmap_json(raw)

    def _parse_mindmap_json(self, raw: str) -> MindmapResult:
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

    def _dict_to_node(self, data: dict, _depth: int = 0) -> MindmapNode:
        if _depth > 10:
            return MindmapNode(name=data.get("name", "..."), meta=data.get("meta"))
        children = [self._dict_to_node(c, _depth + 1) for c in data.get("children", [])]
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
        sections_content: str,
        query_embedding: list[float],
    ) -> str:
        query_text = " ".join(selected_sections) if selected_sections else title
        chunks = chroma_client.hybrid_query(doc_id, query_embedding, query_text, top_k=30)
        if not chunks:
            chunks = []

        # Filter chunks to prioritize selected sections
        relevant_chunks = []
        other_chunks = []
        section_names_lower = [s.strip().lower() for s in selected_sections]
        for c in chunks:
            heading = c.get("metadata", {}).get("heading_path", "").lower()
            if any(s in heading for s in section_names_lower):
                relevant_chunks.append(c)
            else:
                other_chunks.append(c)

        # Build context: section content first, then relevant RAG chunks
        parts = [sections_content]
        total_len = len(sections_content) // 1.5
        for c in relevant_chunks + other_chunks:
            meta = c["metadata"]
            heading = meta.get("heading_path", "")
            header = f"[{heading}]" if heading else ""
            block = f"{header}\n{c['text']}"
            if total_len + len(block) // 1.5 > 16000:
                break
            parts.append(block)
            total_len += len(block) // 1.5

        context = f"{_CONTENT_GUARD_PREFIX}\n{_sanitize_content('\n\n---\n\n'.join(parts))}\n{_CONTENT_GUARD_SUFFIX}"
        user_prompt = DEEP_DIVE_USER_PROMPT.format(
            title=title,
            selected_sections=", ".join(selected_sections),
            content=context,
        )
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": DEEP_DIVE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            max_tokens=4096,
        )
        return self._extract_text(resp)

    def ask(
        self,
        doc_id: str,
        question: str,
        history: list[dict],
        query_embedding: list[float],
    ) -> str:
        chunks = chroma_client.hybrid_query(doc_id, query_embedding, question, top_k=15)
        if not chunks:
            return "无法检索到相关文档内容，请确认文档已正确解析。"

        context = self._build_context(chunks, max_tokens=6000)

        # Format conversation history
        history_text = ""
        if history:
            lines = ["## 对话历史"]
            for i, turn in enumerate(history[-6:]):  # keep last 6 turns
                role = "用户" if turn["role"] == "user" else "助手"
                lines.append(f"{role}：{turn['content']}")
            history_text = "\n".join(lines)

        user_prompt = QA_USER_PROMPT.format(
            history=history_text,
            content=context,
            question=question,
        )

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": QA_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=2048,
        )
        return self._extract_text(resp)


    def _parse_kg_json(self, raw: str) -> dict:
        """解析 LLM 返回的知识图谱 JSON，容错处理"""
        import json
        import re
        text = raw.strip()
        fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if fence_match:
            text = fence_match.group(1).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(text[start:end + 1])
                except json.JSONDecodeError:
                    pass
            return {"concepts": [], "relations": []}

    def extract_concepts_and_relations(
        self,
        section_headings: str,
        candidate_terms: str,
        section_summaries: str,
    ) -> dict:
        """调用 LLM 抽取概念和关系，返回解析后的 dict"""
        from .prompt_templates import KG_EXTRACT_SYSTEM_PROMPT, KG_EXTRACT_USER_PROMPT

        user_prompt = KG_EXTRACT_USER_PROMPT.format(
            section_headings=section_headings,
            candidate_terms=candidate_terms,
            section_summaries=section_summaries,
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": KG_EXTRACT_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=4096,
        )

        raw = response.choices[0].message.content or "{}"
        return self._parse_kg_json(raw)


llm_service = LLMService()
