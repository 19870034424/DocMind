"""Dataset generator: LLM-assisted creation of eval datasets.

Uses reverse QA generation: for each sampled chunk, asks the LLM to produce a
factoid question that the chunk answers. The chunk text becomes the ground truth.
"""

from __future__ import annotations

from .models import EvalDataset, EvalQuery

GEN_QUESTION_PROMPT = """基于以下文档片段，生成一个具体的事实性问题。

要求：
1. 问题应具体而非笼统（如"2024年Q3云计算营收是多少？"而非"营收怎么样？"）
2. 问题的答案必须明确包含在提供的文档片段中
3. 问题的措辞应与原文不完全相同（需要语义理解才能匹配到原文）
4. 只输出问题本身，不要答案、解释或其他内容

文档片段：
{chunk_text}"""


class DatasetGenerator:
    def __init__(self):
        from services.llm_service import llm_service

        self._llm = llm_service

    def generate(
        self,
        doc_id: str,
        doc_title: str,
        chunks: list[dict],
        num_questions: int = 20,
        sampling: str = "diverse",
    ) -> EvalDataset:
        if sampling == "diverse":
            selected = self._diverse_sample(chunks, num_questions)
        else:
            import random
            selected = random.sample(
                chunks, min(num_questions, len(chunks))
            )

        queries: list[EvalQuery] = []
        for i, chunk in enumerate(selected):
            question = self._generate_question(chunk["text"])
            if question:
                queries.append(EvalQuery(
                    id=f"q{i + 1}",
                    question=question,
                    relevant_chunk_texts=[chunk["text"]],
                    notes=f"From: {chunk.get('metadata', {}).get('heading_path', '')}",
                ))

        return EvalDataset(
            doc_id=doc_id,
            document_title=doc_title,
            num_chunks=len(chunks),
            queries=queries,
        )

    def _diverse_sample(
        self, chunks: list[dict], n: int
    ) -> list[dict]:
        from collections import defaultdict

        groups: dict[str, list[dict]] = defaultdict(list)
        for c in chunks:
            heading = c.get("metadata", {}).get("heading_path", "")
            prefix = heading.split(" > ")[0] if heading else "_"
            groups[prefix].append(c)

        selected: list[dict] = []
        group_items = list(groups.values())
        idx = 0
        while len(selected) < n and group_items:
            g = group_items[idx % len(group_items)]
            if g:
                selected.append(g.pop(0))
            idx += 1
            if all(len(g) == 0 for g in group_items):
                break
        return selected[:n]

    def _generate_question(self, chunk_text: str) -> str | None:
        try:
            prompt = GEN_QUESTION_PROMPT.format(chunk_text=chunk_text[:2000])
            resp = self._llm.client.chat.completions.create(
                model=self._llm.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=256,
            )
            question = resp.choices[0].message.content.strip()
            if 5 < len(question) < 500:
                return question
            return None
        except Exception:
            return None

    def add_manual_query(
        self,
        dataset_path: str,
        question: str,
        chunk_texts: list[str],
        notes: str = "",
    ) -> None:
        ds = EvalDataset.load(dataset_path)
        qid = f"q{len(ds.queries) + 1}"
        ds.queries.append(EvalQuery(
            id=qid,
            question=question,
            relevant_chunk_texts=chunk_texts,
            notes=notes or "Manual entry",
        ))
        ds.save(dataset_path)


if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) < 4:
        print("Usage: python -m evaluation.generator <doc_id> <output_path> <chunks_json_or_num> [title]")
        print("  <chunks_json_or_num>: path to chunks JSON file, OR number of questions")
        print("  If a file path ends with .json, it's loaded as chunks; otherwise treated as num_questions")
        print("Example:")
        print("  python -m evaluation.generator abc123 ds.json chunks.json 'My Doc'")
        print("  python -m evaluation.generator abc123 ds.json 20 'My Doc'")
        sys.exit(1)

    doc_id = sys.argv[1]
    output_path = sys.argv[2]
    chunks_arg = sys.argv[3]
    doc_title = sys.argv[4] if len(sys.argv) > 4 else ""

    if chunks_arg.endswith(".json"):
        with open(chunks_arg, encoding="utf-8") as f:
            chunks = json.load(f)
        num_questions = 20
    else:
        num_questions = int(chunks_arg)
        import json as _json

        from storage.db import get_document
        row = get_document(doc_id)
        if not row:
            print(f"Error: Document '{doc_id}' not found in SQLite.")
            sys.exit(1)
        if not row.get("chunks_json"):
            print("Error: Document record has no stored chunks. Reprocess the document.")
            sys.exit(1)
        chunks = _json.loads(row["chunks_json"])
        doc_title = row["title"]

    gen = DatasetGenerator()
    dataset = gen.generate(
        doc_id=doc_id,
        doc_title=doc_title,
        chunks=chunks,
        num_questions=num_questions,
    )
    dataset.save(output_path)
    print(f"Dataset saved: {output_path} ({len(dataset.queries)} queries)")
