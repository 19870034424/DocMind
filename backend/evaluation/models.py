from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class EvalQuery(BaseModel):
    id: str
    question: str
    relevant_chunk_texts: list[str] = Field(default_factory=list)
    notes: str = ""


class EvalDataset(BaseModel):
    version: str = "1.0"
    doc_id: str
    document_title: str = ""
    created_at: datetime = Field(default_factory=datetime.now)
    num_chunks: int = 0
    queries: list[EvalQuery] = Field(default_factory=list)

    @classmethod
    def load(cls, path: str) -> EvalDataset:
        import json

        with open(path, encoding="utf-8") as f:
            return cls.model_validate(json.load(f))

    def save(self, path: str) -> None:
        import json
        import os

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                self.model_dump(mode="json"), f, ensure_ascii=False, indent=2
            )
