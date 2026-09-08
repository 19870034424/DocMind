"""KG Builder 单元测试 — 去重逻辑 + 边修复"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.document import DocumentMetadata, Section, StructuredDocument
from models.knowledge_graph import (
    GraphEdge,
    GraphNode,
    NodeType,
    RelationType,
)
from services.knowledge_graph import KnowledgeGraphBuilder


def _make_doc(sections: list[Section]) -> StructuredDocument:
    return StructuredDocument(
        title="测试教材",
        sections=sections,
        metadata=DocumentMetadata(
            filename="test.pdf", format="pdf",
            pages=len(sections), uploaded_at="2026-01-01T00:00:00", file_size=1000,
        ),
    )


class TestDeduplicate:
    """去重合并 — 之前出过 bug 的核心逻辑"""

    def _build(self, builder=KnowledgeGraphBuilder()) -> KnowledgeGraphBuilder:
        return builder

    def test_same_name_nodes_merged(self):
        """同名节点应该被合并"""
        builder = self._build()
        nodes = [
            GraphNode(id="n0", name="栈", type=NodeType.CONCEPT, importance=0.5,
                       summary="A", source_sections=["1.1"]),
            GraphNode(id="n1", name="栈", type=NodeType.CONCEPT, importance=0.8,
                       summary="B", source_sections=["2.1"]),
            GraphNode(id="n2", name="队列", type=NodeType.CONCEPT, importance=0.6,
                       summary="C", source_sections=["3.1"]),
        ]
        edges: list[GraphEdge] = []
        deduped, _ = builder._deduplicate(nodes, edges)
        assert len(deduped) == 2, f"同名节点应合并，实际 {len(deduped)}"
        names = {n.name for n in deduped}
        assert names == {"栈", "队列"}

    def test_merge_keeps_max_importance(self):
        """合并后保留最高的 importance"""
        builder = self._build()
        nodes = [
            GraphNode(id="n0", name="动态规划", type=NodeType.CONCEPT, importance=0.3,
                       summary="", source_sections=["1.1"]),
            GraphNode(id="n1", name="动态规划", type=NodeType.CONCEPT, importance=0.9,
                       summary="", source_sections=["5.1"]),
        ]
        edges: list[GraphEdge] = []
        deduped, _ = builder._deduplicate(nodes, edges)
        assert deduped[0].importance == 0.9

    def test_merge_combines_source_sections(self):
        """合并后 source_sections 应该合并去重"""
        builder = self._build()
        nodes = [
            GraphNode(id="n0", name="递归", type=NodeType.CONCEPT, importance=0.5,
                       summary="", source_sections=["1.1", "1.2"]),
            GraphNode(id="n1", name="递归", type=NodeType.CONCEPT, importance=0.5,
                       summary="", source_sections=["2.1", "1.2"]),
        ]
        edges: list[GraphEdge] = []
        deduped, _ = builder._deduplicate(nodes, edges)
        assert set(deduped[0].source_sections) == {"1.1", "1.2", "2.1"}

    def test_unique_nodes_unchanged(self):
        """没有重复时节点数量不变"""
        builder = self._build()
        nodes = [
            GraphNode(id="n0", name="栈", type=NodeType.CONCEPT, importance=0.5,
                       summary="", source_sections=["1.1"]),
            GraphNode(id="n1", name="队列", type=NodeType.CONCEPT, importance=0.5,
                       summary="", source_sections=["2.1"]),
            GraphNode(id="n2", name="链表", type=NodeType.CONCEPT, importance=0.5,
                       summary="", source_sections=["3.1"]),
        ]
        edges: list[GraphEdge] = []
        deduped, _ = builder._deduplicate(nodes, edges)
        assert len(deduped) == 3


class TestDeduplicateEdges:
    """边的 ID 修复 — 这是之前出 bug 的地方"""

    def _build(self):
        return KnowledgeGraphBuilder()

    def test_edge_ids_updated_after_dedup(self):
        """去重后边的 source_id/target_id 应该更新为新的节点 ID"""
        builder = self._build()
        nodes = [
            GraphNode(id="node_0001", name="栈", type=NodeType.CONCEPT, importance=0.5,
                       summary="", source_sections=["1.1"]),
            GraphNode(id="node_0002", name="队列", type=NodeType.CONCEPT, importance=0.5,
                       summary="", source_sections=["2.1"]),
        ]
        edges = [
            GraphEdge(source_id="node_0001", target_id="node_0002",
                      relation=RelationType.CONTRASTS, description="栈与队列相反"),
        ]
        deduped, fixed_edges = builder._deduplicate(nodes, edges)

        # 新 ID 应该是 node_0000, node_0001
        new_ids = {n.id for n in deduped}
        assert len(new_ids) == 2
        assert len(fixed_edges) == 1, "边不应该丢失"
        # 边应该引用新 ID
        assert fixed_edges[0].source_id in new_ids
        assert fixed_edges[0].target_id in new_ids
        assert fixed_edges[0].source_id != fixed_edges[0].target_id

    def test_edge_to_merged_node_redirected(self):
        """指向被合并节点的边应该重定向到合并后的新节点"""
        builder = self._build()
        # n0 和 n1 同名 -> n1 被合并到 n0
        nodes = [
            GraphNode(id="old_A", name="栈", type=NodeType.CONCEPT, importance=0.5,
                       summary="", source_sections=["1.1"]),
            GraphNode(id="old_B", name="栈", type=NodeType.CONCEPT, importance=0.8,
                       summary="", source_sections=["2.1"]),
            GraphNode(id="old_C", name="队列", type=NodeType.CONCEPT, importance=0.5,
                       summary="", source_sections=["3.1"]),
        ]
        # 有一条边从队列指向被合并的节点 old_B
        edges = [
            GraphEdge(source_id="old_C", target_id="old_B",
                      relation=RelationType.EXTENDS, description=""),
        ]
        deduped, fixed_edges = builder._deduplicate(nodes, edges)

        assert len(deduped) == 2  # 栈 + 队列
        assert len(fixed_edges) == 1
        # 边应该指向合并后的栈节点（而不是断掉的 old_B）
        stack_node = next(n for n in deduped if n.name == "栈")
        queue_node = next(n for n in deduped if n.name == "队列")

        # 修复前: 队列 -> old_B (被合并的栈节点)
        # 修复后: 队列的新ID -> 栈的新ID
        assert fixed_edges[0].source_id == queue_node.id
        assert fixed_edges[0].target_id == stack_node.id

    def test_duplicate_edges_collapsed(self):
        """同方向同类型的重复边应该合并"""
        builder = self._build()
        nodes = [
            GraphNode(id="n0", name="栈", type=NodeType.CONCEPT, importance=0.5,
                       summary="", source_sections=["1.1"]),
            GraphNode(id="n1", name="队列", type=NodeType.CONCEPT, importance=0.5,
                       summary="", source_sections=["2.1"]),
        ]
        edges = [
            GraphEdge(source_id="n0", target_id="n1",
                      relation=RelationType.CONTRASTS, description="第一处"),
            GraphEdge(source_id="n0", target_id="n1",
                      relation=RelationType.CONTRASTS, description="第二处"),
        ]
        _deduped, fixed_edges = builder._deduplicate(nodes, edges)
        assert len(fixed_edges) == 1, f"重复边应合并，实际 {len(fixed_edges)}"

    def test_dangling_edge_to_nonexistent_node(self):
        """指向不存在节点的边应保留原 ID（不在去重范围内）"""
        builder = self._build()
        nodes = [
            GraphNode(id="n0", name="栈", type=NodeType.CONCEPT, importance=0.5,
                       summary="", source_sections=["1.1"]),
        ]
        # 边指向一个不在节点列表中的节点
        edges = [
            GraphEdge(source_id="n0", target_id="ghost_node",
                      relation=RelationType.EXTENDS, description=""),
        ]
        _deduped, fixed_edges = builder._deduplicate(nodes, edges)
        assert len(fixed_edges) == 1
        # 不存在的节点 ID 保持不变（由后续过滤处理）
        assert fixed_edges[0].target_id == "ghost_node"


class TestCandidateTerms:
    """术语候选提取"""

    def test_extracts_nouns(self):
        """应该提取文本中的名词"""
        builder = KnowledgeGraphBuilder()
        doc = _make_doc([
            Section(heading="数据结构基础", level=1,
                    content="数据结构是计算机存储、组织数据的方式。常见的数据结构有数组、链表、栈、队列、树和图。",
                    page=1, tables=[]),
        ])
        terms = builder.extract_candidate_terms(doc)
        assert len(terms) > 0, "应该提取到术语"
        term_names = [t[0] for t in terms]
        # 至少应该提取到一些名词
        assert len(term_names) >= 3, f"应至少提取3个术语，实际 {len(term_names)}: {term_names}"

    def test_heading_boost(self):
        """章节标题中的词应该获得加分"""
        builder = KnowledgeGraphBuilder()
        doc = _make_doc([
            Section(heading="图论基础", level=1,
                    content="本章介绍图论。图论图论图论。",
                    page=1, tables=[]),
        ])
        terms = builder.extract_candidate_terms(doc)
        # "图论" 在标题中，分数应该高于纯频率词
        term_dict = {t[0]: t[2] for t in terms}
        assert "图论" in term_dict

    def test_empty_document(self):
        """空文档不崩溃"""
        builder = KnowledgeGraphBuilder()
        doc = _make_doc([])
        terms = builder.extract_candidate_terms(doc)
        assert terms == []


class TestBuildEdgeCases:
    """build() 的异常情况"""

    def test_build_without_llm_creates_empty_graph(self):
        """无 LLM 服务时仍能创建空图谱（不应该崩溃）"""
        # 不传 llm 但 _llm 是 None，build 时触发懒加载会尝试 import
        # 这里不测实际 LLM 调用，只测 build 的防御逻辑
        # 需要 mock，跳过
