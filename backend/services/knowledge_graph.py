"""知识图谱构建服务

核心流程：术语候选提取、章节分批送 LLM 抽取概念和关系、
从章节层级生成 PART_OF 边、去重合并后持久化为 JSON。
"""
import re
from collections import Counter

import jieba.posseg as pseg

from models.document import StructuredDocument
from models.knowledge_graph import (
    BuildingStatus,
    GraphEdge,
    GraphNode,
    KnowledgeGraph,
    NodeType,
    RelationType,
)
from storage.kg_store import kg_store

# 停用词（常见无意义术语）
STOP_WORDS = {
    "可以", "使用", "通过", "一个", "这个", "其中", "如果", "我们",
    "它们", "这些", "那些", "以及", "所以", "因此", "但是", "然而",
    "例如", "比如", "一种", "这种", "各种", "不同", "问题", "方法",
    "进行", "需要", "能够", "用来", "基本", "主要", "一般",
}


class KnowledgeGraphBuilder:
    """知识图谱构建器"""

    def __init__(self, llm_service=None):
        self._llm = llm_service

    @property
    def llm(self):
        if self._llm is None:
            from services.llm_service import llm_service
            self._llm = llm_service
        return self._llm

    def extract_candidate_terms(
        self, structured_doc: StructuredDocument, top_n: int = 200
    ) -> list[tuple[str, str, int]]:
        """使用 jieba 词性标注提取候选术语

        返回: [(term, pos_tag, score), ...] 按 score 降序
        """
        all_text = ""
        heading_terms = set()
        for section in structured_doc.sections:
            heading_terms.add(section.heading)
            bold_matches = re.findall(r"\*\*(.+?)\*\*", section.content)
            heading_terms.update(bold_matches)
            all_text += section.heading + " " + section.content + " "

        pos_filter = {"n", "nr", "ns", "nt", "nz", "vn", "eng"}
        term_freq: Counter = Counter()
        for word, flag in pseg.cut(all_text):
            word = word.strip()
            if (
                flag in pos_filter
                and len(word) >= 2
                and word not in STOP_WORDS
                and not re.match(r"^\d+$", word)
            ):
                term_freq[word] += 1

        scored: list[tuple[str, str, int]] = []
        for term, freq in term_freq.most_common(top_n * 3):
            score = freq
            if term in heading_terms:
                score += 5
            scored.append((term, flag, score))

        scored.sort(key=lambda x: x[2], reverse=True)
        return scored[:top_n]

    def _build_section_summaries(
        self, structured_doc: StructuredDocument
    ) -> list[dict[str, str]]:
        """为每个章节生成摘要信息（标题 + 前 500 字内容）"""
        summaries = []
        for section in structured_doc.sections:
            content_preview = section.content[:500]
            summaries.append({
                "heading": section.heading,
                "level": str(section.level),
                "preview": content_preview,
            })
        return summaries

    def _batch_sections(
        self, section_summaries: list[dict], batch_size: int = 5
    ) -> list[list[dict]]:
        """将章节按 batch_size 分组"""
        batches = []
        for i in range(0, len(section_summaries), batch_size):
            batches.append(section_summaries[i:i + batch_size])
        return batches

    def _generate_section_hierarchy_edges(
        self, structured_doc: StructuredDocument
    ) -> list[GraphEdge]:
        """从章节层级自动生成 PART_OF 边"""
        edges: list[GraphEdge] = []
        heading_stack: list[tuple[int, str]] = []  # [(level, node_id), ...]

        for i, section in enumerate(structured_doc.sections):
            node_id = f"section_{i}"
            level = section.level

            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()

            if heading_stack:
                parent_id = heading_stack[-1][1]
                edges.append(GraphEdge(
                    source_id=parent_id,
                    target_id=node_id,
                    relation=RelationType.PART_OF,
                    description=f"属于 {section.heading}",
                    source_sections=[section.heading],
                ))

            heading_stack.append((level, node_id))

        return edges

    def build(
        self,
        doc_id: str,
        structured_doc: StructuredDocument,
        document_title: str = "",
    ) -> KnowledgeGraph:
        """构建知识图谱的主入口"""
        # 1. 提取候选术语
        candidate_terms = self.extract_candidate_terms(structured_doc)
        term_names = [t[0] for t in candidate_terms[:100]]

        # 2. 生成章节摘要
        section_summaries = self._build_section_summaries(structured_doc)

        # 3. 一次性送 LLM 抽取（不再分批，避免多次串行调用）
        section_headings_str = ", ".join(
            s.heading for s in structured_doc.sections
        )
        all_sections_text = "\n\n".join(
            f"## {s['heading']}\n{s['preview']}" for s in section_summaries
        )

        try:
            result = self.llm.extract_concepts_and_relations(
                section_headings=section_headings_str,
                candidate_terms=", ".join(term_names),
                section_summaries=all_sections_text,
            )
            all_concepts = result.get("concepts", [])
            all_relations = result.get("relations", [])
        except Exception as e:
            import logging
            logging.error(f"LLM extraction failed: {e}")
            all_concepts = []
            all_relations = []

        # 4. 构建节点列表
        nodes: list[GraphNode] = []
        name_to_id: dict[str, str] = {}
        for i, c in enumerate(all_concepts):
            node_id = f"node_{i:04d}"
            name = c.get("name", "")
            node_type_str = c.get("type", "concept")
            try:
                node_type = NodeType(node_type_str)
            except ValueError:
                node_type = NodeType.CONCEPT

            try:
                imp = float(c.get("importance", 0.5))
            except (ValueError, TypeError):
                imp = 0.5

            nodes.append(GraphNode(
                id=node_id,
                name=name,
                type=node_type,
                importance=imp,
                summary=c.get("summary", ""),
                source_sections=[c.get("source_section", "")],
            ))
            name_to_id[name] = node_id

        # 5. 构建边列表
        edges: list[GraphEdge] = []
        for r in all_relations:
            src_id = name_to_id.get(r.get("source", ""))
            tgt_id = name_to_id.get(r.get("target", ""))
            if src_id and tgt_id:
                relation_str = r.get("relation", "related_to")
                try:
                    rel_type = RelationType(relation_str)
                except ValueError:
                    rel_type = RelationType.EXTENDS
                edges.append(GraphEdge(
                    source_id=src_id,
                    target_id=tgt_id,
                    relation=rel_type,
                    description=r.get("description", ""),
                ))

        nodes, edges = self._deduplicate(nodes, edges)

        # 7. 构建最终图谱
        kg = KnowledgeGraph(
            doc_id=doc_id,
            document_title=document_title,
            nodes=nodes,
            edges=edges,
            status=BuildingStatus.READY,
        )

        # 9. 持久化
        kg_store.save(kg)

        return kg

    def _deduplicate(
        self, nodes: list[GraphNode], edges: list[GraphEdge]
    ) -> tuple[list[GraphNode], list[GraphEdge]]:
        """去重合并同名节点和重复边，同时修复边的 ID 引用"""
        # 节点去重：同名节点合并
        seen: dict[str, GraphNode] = {}
        old_id_to_name: dict[str, str] = {}  # 记录旧 ID → 名称的映射
        for node in nodes:
            old_id_to_name[node.id] = node.name
            if node.name in seen:
                existing = seen[node.name]
                existing.source_sections = list(
                    set(existing.source_sections + node.source_sections)
                )
                existing.importance = max(existing.importance, node.importance)
            else:
                seen[node.name] = node

        deduped_nodes = list(seen.values())
        # 构建双重映射：旧 ID → 新 ID，名称 → 新 ID
        old_to_new: dict[str, str] = {}
        name_to_id: dict[str, str] = {}
        for i, n in enumerate(deduped_nodes):
            new_id = f"node_{i:04d}"
            name_to_id[n.name] = new_id
            # 所有旧 ID 指向同一名称的节点，都要映射到新 ID
        for old_id, name in old_id_to_name.items():
            if name in name_to_id:
                old_to_new[old_id] = name_to_id[name]
        # 更新节点 ID
        for node in deduped_nodes:
            node.id = name_to_id[node.name]

        # 边去重和 ID 修复
        edge_map: dict[tuple[str, str, str], GraphEdge] = {}
        for edge in edges:
            # 先通过旧 ID 映射修复，再尝试名称匹配
            new_src = old_to_new.get(edge.source_id, edge.source_id)
            new_tgt = old_to_new.get(edge.target_id, edge.target_id)
            key = (new_src, new_tgt, edge.relation.value)
            if key in edge_map:
                edge_map[key].source_sections = list(
                    set(edge_map[key].source_sections + edge.source_sections)
                )
            else:
                edge.source_id = new_src
                edge.target_id = new_tgt
                edge_map[key] = edge

        return deduped_nodes, list(edge_map.values())


# 模块级单例
kg_builder = KnowledgeGraphBuilder()
