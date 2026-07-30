"""知识图谱 JSON 持久化存储"""
import json
import os
from pathlib import Path
from models.knowledge_graph import KnowledgeGraph


class KGStore:
    """知识图谱文件存储，每个文档一个 JSON 文件"""

    def __init__(self, base_dir: str = "../kg_data"):
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)

    def _file_path(self, doc_id: str) -> Path:
        return self._base_dir / f"{doc_id}.json"

    def save(self, kg: KnowledgeGraph) -> None:
        """保存知识图谱到 JSON 文件"""
        path = self._file_path(kg.doc_id)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(kg.model_dump(mode="json"), f, ensure_ascii=False, indent=2)

    def load(self, doc_id: str) -> KnowledgeGraph | None:
        """加载知识图谱，不存在则返回 None"""
        path = self._file_path(doc_id)
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return KnowledgeGraph(**data)

    def exists(self, doc_id: str) -> bool:
        return self._file_path(doc_id).exists()

    def delete(self, doc_id: str) -> None:
        """删除知识图谱文件"""
        path = self._file_path(doc_id)
        if path.exists():
            os.remove(path)


# 模块级单例
kg_store = KGStore()
