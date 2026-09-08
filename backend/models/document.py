from __future__ import annotations

import enum
from datetime import datetime

from pydantic import BaseModel, Field


class DocumentFormat(str, enum.Enum):
    PDF = "pdf"
    DOCX = "docx"
    MARKDOWN = "md"


class DocumentStatus(str, enum.Enum):
    UPLOADED = "uploaded"
    QUEUED = "queued"
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
    metadata: DocumentMetadata | None = None
    structured: StructuredDocument | None = None
    chunks: list[dict] = Field(default_factory=list)


class ReportResult(BaseModel):
    content: str
    generated_at: datetime = Field(default_factory=datetime.now)


class MindmapNode(BaseModel):
    name: str
    meta: str | None = None
    children: list[MindmapNode] = Field(default_factory=list)


class MindmapResult(BaseModel):
    root: MindmapNode
    generated_at: datetime = Field(default_factory=datetime.now)
