"""知识图谱数据模型"""
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class NodeType(str, Enum):
    CONCEPT = "concept"
    DEFINITION = "definition"
    ALGORITHM = "algorithm"
    PROPERTY = "property"
    EXAMPLE = "example"


class RelationType(str, Enum):
    PREREQUISITE = "prerequisite"
    LEADS_TO = "leads_to"
    CONTRASTS = "contrasts"
    EXTENDS = "extends"
    PART_OF = "part_of"


class GraphNode(BaseModel):
    id: str = Field(description="唯一标识，如 node_001")
    name: str = Field(description="概念名，如'动态规划'")
    type: NodeType = Field(description="节点类型")
    importance: float = Field(default=0.5, ge=0, le=1, description="LLM 评估的重要度")
    summary: str = Field(default="", description="一句话概述")
    source_sections: list[str] = Field(default_factory=list, description="来源章节")
    related_chunk_ids: list[str] = Field(default_factory=list, description="关联原文片段ID")


class GraphEdge(BaseModel):
    source_id: str = Field(description="起始节点ID")
    target_id: str = Field(description="目标节点ID")
    relation: RelationType = Field(description="关系类型")
    description: str = Field(default="", description="关系简述")
    source_sections: list[str] = Field(default_factory=list, description="此关系出现的章节")


class BuildingStatus(str, Enum):
    IDLE = "idle"
    BUILDING = "building"
    READY = "ready"
    ERROR = "error"


class KnowledgeGraph(BaseModel):
    doc_id: str = Field(description="关联的文档ID")
    document_title: str = Field(default="", description="文档标题")
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)
    status: BuildingStatus = Field(default=BuildingStatus.IDLE)
    generated_at: str = Field(default="", description="生成时间 ISO 格式")
    node_count: int = Field(default=0)
    edge_count: int = Field(default=0)
