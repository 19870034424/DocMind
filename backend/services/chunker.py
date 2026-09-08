# 模块级：一次性解决 SSL + 离线问题(HF_HUB_CACHE 由 config.py 统一指定到 D 盘)
import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 必须最先导入 config: 它设置 HF_HUB_CACHE, 且必须发生在
# langchain_text_splitters(会间接导入 transformers/huggingface_hub)之前
import ssl

from config import settings

ssl._create_default_https_context = ssl._create_unverified_context

import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

from models.document import Section, StructuredDocument

_IMG_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")


class Chunker:
    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", "。", "，", " ", ""],
        )

    def _section_to_text(self, section: Section, path: str = "") -> str:
        header = f"所属章节：{path}" if path else ""
        content = _IMG_RE.sub("", section.content or "")  # 图片引用不参与嵌入
        parts = [header] if header else []
        if content.strip():
            parts.append(content)
        for table in section.tables:
            table_md = self._table_to_markdown(table)
            parts.append(table_md)
        return "\n\n".join(parts)

    def _iter_content_segments(self, content: str) -> list[tuple[str, str]]:
        """把 section 内容拆成 ("text"|"code", 原文) 段, 代码围栏整体成段不被切开。"""
        segments: list[tuple[str, str]] = []
        buf: list[str] = []
        in_code = False
        for line in (content or "").split("\n"):
            if line.strip().startswith("```"):
                if not in_code:
                    if "\n".join(buf).strip():
                        segments.append(("text", "\n".join(buf)))
                    buf = []
                    in_code = True
                else:
                    in_code = False
                buf.append(line)
                continue
            buf.append(line)
        if "\n".join(buf).strip():
            segments.append(("code" if in_code else "text", "\n".join(buf)))
        return segments

    def chunk(self, doc: StructuredDocument) -> list[dict]:
        """切分策略:
        - 代码块、表格是原子块, 整体作为一个 chunk, 不被递归切分打碎
        - 普通文本按章节 + 递归字符切分
        - 每个块带 block_type(text/code/table), heading_path 只拼一次
        """
        chunks: list[dict] = []
        paths = self._build_heading_paths(doc.sections)
        index = 0
        for i, section in enumerate(doc.sections):
            heading_path = paths[i]
            header = f"所属章节：{heading_path}" if heading_path else ""

            # 普通内容: 文本段递归切, 代码段原子
            for seg_type, seg_text in self._iter_content_segments(_IMG_RE.sub("", section.content or "")):
                full_text = f"{header}\n\n{seg_text}" if header else seg_text
                pieces = [full_text] if seg_type == "code" else self.splitter.split_text(full_text)
                for piece in pieces:
                    chunks.append({
                        "text": piece,
                        "metadata": {
                            "heading_path": heading_path,
                            "section_heading": section.heading,
                            "page": section.page,
                            "chunk_index": index,
                            "block_type": seg_type,
                        },
                    })
                    index += 1

            # 表格: 每张表一个原子块
            for table in section.tables:
                table_md = self._table_to_markdown(table)
                if not table_md.strip():
                    continue
                full_text = f"{header}\n\n{table_md}" if header else table_md
                chunks.append({
                    "text": full_text,
                    "metadata": {
                        "heading_path": heading_path,
                        "section_heading": section.heading,
                        "page": section.page,
                        "chunk_index": index,
                        "block_type": "table",
                    },
                })
                index += 1
        return chunks

    def _table_to_markdown(self, table) -> str:
        lines = []
        if table.headers:
            lines.append("| " + " | ".join(table.headers) + " |")
            lines.append("| " + " | ".join(["---"] * len(table.headers)) + " |")
        for row in table.rows:
            lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines)

    def _build_heading_paths(self, sections: list[Section]) -> list[str]:
        paths: list[str] = []
        stack: list[tuple[int, str]] = []  # (level, heading)
        for s in sections:
            if s.heading:
                while stack and stack[-1][0] >= s.level:
                    stack.pop()
                stack.append((s.level, s.heading))
            paths.append(" > ".join(h for _, h in stack))
        return paths

    def embed(self, texts: list[str]) -> list[list[float]]:
        """委托给独立嵌入服务(embedder), 保持兼容"""
        from services.embedder import embed
        return embed(texts)


chunker = Chunker()
