import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from bs4 import BeautifulSoup

from models.document import (
    DocumentFormat,
    DocumentMetadata,
    Section,
    StructuredDocument,
    TableData,
)


class DocumentParser:
    def __init__(self, mineru_venv: str = ""):
        self._mineru_python = Path(mineru_venv) / "Scripts" / "python.exe" if mineru_venv else ""

    def _find_mineru_bin(self) -> str:
        """Locate the mineru CLI in its venv."""
        if self._mineru_python and Path(self._mineru_python).exists():
            bin_dir = Path(self._mineru_python).parent
            mineru_exe = bin_dir / "mineru.exe"
            if mineru_exe.exists():
                return str(mineru_exe)
            mineru_cmd = bin_dir / "mineru"
            if mineru_cmd.exists():
                return str(mineru_cmd)
        return shutil.which("mineru") or shutil.which("mineru.exe") or "mineru"

    def parse(self, file_path: Path, filename: str, doc_id: str = "") -> StructuredDocument:
        fmt = self._detect_format(filename)
        metadata = DocumentMetadata(
            filename=filename,
            format=fmt,
            file_size=file_path.stat().st_size,
        )
        if fmt == DocumentFormat.MARKDOWN:
            return self._parse_markdown_direct(file_path, filename, metadata)
        return self._parse_with_mineru(file_path, filename, metadata, doc_id)

    def _parse_markdown_direct(
        self, file_path: Path, filename: str, metadata: DocumentMetadata
    ) -> StructuredDocument:
        """Parse markdown/text files directly, no MinerU needed."""
        md_content = file_path.read_text(encoding="utf-8")
        return self._markdown_to_structured(md_content, filename, metadata)

    def _detect_format(self, filename: str) -> DocumentFormat:
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        mapping = {
            "pdf": DocumentFormat.PDF,
            "docx": DocumentFormat.DOCX,
            "pptx": DocumentFormat.PDF,
            "xlsx": DocumentFormat.PDF,
            "md": DocumentFormat.MARKDOWN,
            "markdown": DocumentFormat.MARKDOWN,
            "txt": DocumentFormat.MARKDOWN,
        }
        # .doc (旧版 Word 二进制格式) 不支持 — MinerU 无法解析
        if ext == "doc":
            raise ValueError("不支持 .doc 格式，请转换为 .docx 后上传")
        if ext not in mapping:
            raise ValueError(f"Unsupported file extension: .{ext}")
        return mapping[ext]

    def _parse_with_mineru(
        self, file_path: Path, filename: str, metadata: DocumentMetadata, doc_id: str = ""
    ) -> StructuredDocument:
        output_dir = tempfile.mkdtemp(prefix="mineru_")
        try:
            # mineru.exe 入口脚本在此环境下会静默 exit 1，
            # 改用 venv python -m mineru.cli.client 调用（行为等价）
            if self._mineru_python and Path(self._mineru_python).exists():
                cmd = [
                    self._mineru_python,
                    "-m", "mineru.cli.client",
                    "-p", str(file_path),
                    "-o", output_dir,
                    "-b", "pipeline",
                ]
            else:
                mineru_bin = self._find_mineru_bin()
                cmd = [
                    mineru_bin,
                    "-p", str(file_path),
                    "-o", output_dir,
                    "-b", "pipeline",
                ]

            # If we have a venv python, prepend it to PATH so mineru uses correct deps
            env = os.environ.copy()
            if self._mineru_python and Path(self._mineru_python).exists():
                bin_dir = str(Path(self._mineru_python).parent)
                env["PATH"] = bin_dir + os.pathsep + env.get("PATH", "")

            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
                env=env,
            )

            if proc.returncode != 0:
                raise RuntimeError(
                    f"MinerU parsing failed (exit {proc.returncode}): {proc.stderr[:500]}"
                )

            # MinerU output: <output_dir>/<basename>/<basename>.md
            md_file = self._find_markdown_output(output_dir, file_path)
            if not md_file:
                raise RuntimeError("MinerU did not produce a markdown output file")

            markdown_content = md_file.read_text(encoding="utf-8")

            # 把 MinerU 抽取的图片拷贝到 uploads/{doc_id}/images 并重写引用,
            # 否则清理输出目录后图片全部丢失
            if doc_id:
                markdown_content = self._relocate_images(
                    md_file.parent, markdown_content, doc_id
                )

            return self._markdown_to_structured(markdown_content, filename, metadata)

        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def _relocate_images(self, md_dir: Path, md_content: str, doc_id: str) -> str:
        """拷贝 MinerU 输出的 images/ 到 uploads/{doc_id}/images,
        并把 markdown 里的相对引用改写为 /api/assets/{doc_id}/images/..."""
        src = md_dir / "images"
        if not src.is_dir():
            return md_content
        dst = Path(__file__).resolve().parent.parent.parent / "uploads" / doc_id / "images"
        dst.mkdir(parents=True, exist_ok=True)
        for f in src.iterdir():
            if f.is_file():
                shutil.copy2(f, dst / f.name)

        def _rewrite(m: re.Match) -> str:
            return f"]( /api/assets/{doc_id}/images/{m.group(1)} )".replace(" ", "")

        return re.sub(r"\]\(\s*images/([^)\s]+)\s*\)", _rewrite, md_content)

    def _find_markdown_output(self, output_dir: str, file_path: Path) -> Path | None:
        """Find the .md file produced by MinerU."""
        out = Path(output_dir)
        for root, _dirs, files in os.walk(out):
            for f in files:
                if f.endswith(".md"):
                    return Path(root) / f
        return None

    def _markdown_to_structured(
        self, md_content: str, filename: str, metadata: DocumentMetadata
    ) -> StructuredDocument:
        """按 ATX 标题切分 markdown, 各 section 保留原文(含公式/代码块/表格/图片引用)。

        刻意不做 markdown→HTML 的全文转换, 避免代码缩进丢失、公式被改写等有损转换。
        表格仍单独抽取为 TableData 供前端结构化渲染。
        """
        md_content = self._normalize_math_blocks(md_content)

        sections: list[Section] = []
        current_section = Section(heading="", level=0, page=0)
        current_text: list[str] = []

        in_code = False
        pending_table: list[str] = []

        def flush_table():
            nonlocal pending_table
            if pending_table:
                html = "\n".join(pending_table)
                table_tag = BeautifulSoup(html, "html.parser").find("table")
                if table_tag is not None:
                    headers, rows = self._extract_table(table_tag)
                    if headers or rows:
                        current_section.tables.append(
                            TableData(headers=headers, rows=rows)
                        )
                pending_table = []

        for line in md_content.split("\n"):
            stripped = line.strip()
            if stripped.lower().startswith("<table"):
                flush_table()
                pending_table.append(line)
                continue
            if pending_table:
                pending_table.append(line)
                if "</table>" in stripped.lower():
                    flush_table()
                continue

            if stripped.startswith("```"):
                in_code = not in_code
                current_text.append(line)
                continue
            if in_code:
                current_text.append(line)  # 代码块内原样保留(缩进/空行)
                continue

            m = re.match(r"^(#{1,6})\s+(.*)$", stripped)
            if m:
                self._flush_section(current_section, current_text, sections)
                current_text = []
                level = len(m.group(1))
                heading_text = self._clean_math(m.group(2).strip())
                current_section = Section(heading=heading_text, level=level, page=0)
            else:
                current_text.append(line)

        flush_table()
        self._flush_section(current_section, current_text, sections)

        # Post-process: if no headings found, create single section
        if not sections:
            sections = [Section(heading="正文", level=1, page=0, content=md_content)]

        # Determine title: first non-empty heading, or filename
        title = filename
        for s in sections:
            if s.heading and s.heading.strip():
                title = s.heading.strip()
                break

        metadata.pages = max(1, len(sections) // 2)
        return StructuredDocument(title=title, sections=sections, metadata=metadata)

    def _normalize_math_blocks(self, md_content: str) -> str:
        """MinerU 的块级公式常写成 "$$" 独占一行且块内夹空行,
        会导致下游(remark-math/LLM)解析失败, 这里剔除块内空行。
        另外 MinerU 公式 token 之间塞满空格("\\mathrm { m o d e l }"),
        TeX 数学模式本就忽略空格, 压缩后 KaTeX 才能正确渲染;
        \text{...} 参数内的空格有语义, 先占位保护。"""
        # 先丢弃空公式块($$ 之间只有空白), 避免打乱开关状态
        md_content = re.sub(r"\$\$\s*\$\$", "", md_content)

        # 保护 \text{...} 中的空格
        md_content = re.sub(
            r"\\text\s*\{([^{}]*)\}",
            lambda m: "\\text{" + m.group(1).replace(" ", "@@SP@@") + "}",
            md_content,
        )

        out: list[str] = []
        in_math = False
        buf: list[str] = []
        for line in md_content.split("\n"):
            if line.strip() == "$$":
                if in_math:
                    out.append("$$" + " ".join(x for x in buf if x) + "$$")
                    buf = []
                else:
                    out.append("$$")
                in_math = not in_math
                continue
            if in_math:
                if line.strip():
                    buf.append(line.strip())
                continue
            # 行内公式: 压缩 $...$ 内部空格; 单行 $$...$$ 块同样处理
            if "$$" in line and line.strip().startswith("$$") and line.strip().endswith("$$") and len(line.strip()) > 4:
                line = re.sub(
                    r"\$\$(.+)\$\$",
                    lambda m: "$$" + re.sub(r"\s+", "", m.group(1)) + "$$",
                    line,
                )
            elif "$" in line:
                line = re.sub(
                    r"\$([^$]+)\$",
                    lambda m: "$" + re.sub(r"\s+", "", m.group(1)) + "$",
                    line,
                )
            out.append(line)
        if in_math and buf:  # 未闭合的 $$ 块, 原样归还
            out.append("$$" + " ".join(buf))
        result = "\n".join(out)
        result = result.replace("@@SP@@", " ")
        return result

    def _flush_section(
        self, section: Section, text_parts: list[str], sections: list[Section]
    ) -> None:
        content = "\n\n".join(p for p in text_parts if p.strip())
        if content or section.tables:
            section.content = content
            sections.append(section)
        elif section.heading:
            sections.append(section)

    def _extract_table(self, table_tag) -> tuple[list[str], list[list[str]]]:
        """Extract headers and rows from an HTML table tag."""
        headers: list[str] = []
        rows: list[list[str]] = []
        thead = table_tag.find("thead")
        if thead:
            for th in thead.find_all("th"):
                headers.append(self._clean_math(th.get_text(strip=True)))

        tbody = table_tag.find("tbody") or table_tag
        for tr in tbody.find_all("tr"):
            cells: list[str] = []
            for td in tr.find_all(["td", "th"]):
                cells.append(self._clean_math(td.get_text(strip=True)))
            if cells:
                rows.append(cells)

        if not headers and rows:
            headers = rows[0]
            rows = rows[1:]

        return headers, rows

    def _clean_math(self, text: str) -> str:
        """Normalize LaTeX math delimiters for display.
        同时压缩 MinerU 公式 token 间的多余空格(\text{} 内除外)。"""
        text = re.sub(r"\$\$", "$$", text)
        text = re.sub(
            r"\\text\s*\{([^{}]*)\}",
            lambda m: "\\text{" + m.group(1).replace(" ", "@@SP@@") + "}",
            text,
        )
        for m in re.finditer(r"\$([^$]+)\$", text):
            compact = re.sub(r"\s+", "", m.group(1))
            text = text.replace(m.group(0), "$" + compact + "$")
        return text.replace("@@SP@@", " ")


parser_venv = Path(__file__).resolve().parent.parent.parent / ".venv-mineru"
document_parser = DocumentParser(mineru_venv=str(parser_venv))
