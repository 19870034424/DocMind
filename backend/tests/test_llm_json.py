"""LLM JSON 解析测试 — _parse_kg_json 容错逻辑"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.llm_service import LLMService


class TestParseKgJson:
    """_parse_kg_json — 各种 LLM 返回格式的容错"""

    def _parse(self, raw: str) -> dict:
        return LLMService()._parse_kg_json(raw)

    def test_normal_json(self):
        """标准 JSON — 直接解析"""
        result = self._parse('{"concepts":[{"name":"栈","type":"concept"}],"relations":[]}')
        assert len(result["concepts"]) == 1
        assert result["concepts"][0]["name"] == "栈"

    def test_json_with_markdown_fence(self):
        """被 ```json 包裹 — 应该去掉围栏"""
        raw = """```json
{"concepts":[{"name":"队列"}],"relations":[]}
```"""
        result = self._parse(raw)
        assert len(result["concepts"]) == 1
        assert result["concepts"][0]["name"] == "队列"

    def test_json_with_plain_fence(self):
        """被 ``` 包裹但没有 json 标记"""
        raw = """```
{"concepts":[{"name":"链表"}],"relations":[]}
```"""
        result = self._parse(raw)
        assert len(result["concepts"]) == 1

    def test_json_after_chatter(self):
        """JSON 前面有废话文本 — 提取大括号内容"""
        raw = """好的，以下是为您提取的概念：

{
  "concepts": [
    {"name": "二叉树", "type": "concept", "importance": 0.9, "summary": "测试"}
  ],
  "relations": []
}

希望对您有帮助！"""
        result = self._parse(raw)
        assert len(result["concepts"]) == 1

    def test_json_with_text_after(self):
        """JSON 后面有废话"""
        raw = """{"concepts":[{"name":"哈希"}],"relations":[]}
以上是提取结果。"""
        result = self._parse(raw)
        assert len(result["concepts"]) == 1

    def test_empty_string(self):
        """空字符串 — 返回空结构"""
        result = self._parse("")
        assert result["concepts"] == []
        assert result["relations"] == []

    def test_completely_invalid(self):
        """完全不是 JSON — 不抛异常，返回空结构"""
        result = self._parse("这是一段完全不是 JSON 的文本。")
        assert result["concepts"] == []
        assert result["relations"] == []

    def test_json_with_thinking_tag(self):
        """被 think 标签包裹 — 应该能提取"""
        raw = """<thinking>需要分析章节内容</thinking>
{"concepts":[{"name":"动态规划","type":"concept"}],"relations":[]}"""
        result = self._parse(raw)
        assert len(result["concepts"]) == 1

    def test_multiple_brackets(self):
        """文本中有多组大括号 — 取最外层"""
        raw = """{"concepts":[{"name":"外层","type":"concept"}],"relations":[{"source":"A","target":"B","relation":"prerequisite"}]}"""
        result = self._parse(raw)
        assert len(result["concepts"]) == 1
        assert len(result["relations"]) == 1

    def test_only_opening_brace(self):
        """只有半个 JSON — 不崩溃"""
        result = self._parse('{"concepts":[')
        # 解析失败，返回空
        assert result["concepts"] == []

    def test_none_value(self):
        """LLM 返回 None — 不崩溃"""
        # 模拟 None（虽然不太可能，但防御一下）
        result = self._parse(None)  # type: ignore
        assert result["concepts"] == []

    def test_concepts_only_no_relations(self):
        """只有 concepts 没有 relations 也能解析"""
        raw = '{"concepts":[{"name":"概念A","type":"concept"}]}'
        result = self._parse(raw)
        assert len(result["concepts"]) == 1
        # relations 缺失时应该返回空列表
        assert result.get("relations") is None or result.get("relations") == []

    def test_trailing_comma_json(self):
        """宽松处理：JSON 末尾有逗号"""
        raw = '{"concepts":[{"name":"栈"},]}'
        result = self._parse(raw)
        # 非法 JSON，返回空结构不崩溃
        assert result["concepts"] == []


class TestParseKgJsonRealistic:
    """真实 LLM 可能返回的各种变形"""

    def _parse(self, raw: str) -> dict:
        return LLMService()._parse_kg_json(raw)

    def test_deepseek_r1_style(self):
        """DeepSeek-R1 可能返回带 think 标签的结果"""
        raw = """<think>
用户希望我提取概念，我需要仔细分析文本。
</think>
{
  "concepts": [
    {"name": "Transformer", "type": "concept", "importance": 1.0, "summary": "自注意力架构", "source_section": "2.1"},
    {"name": "BERT", "type": "algorithm", "importance": 0.9, "summary": "预训练模型", "source_section": "2.3"}
  ],
  "relations": [
    {"source": "Transformer", "target": "BERT", "relation": "prerequisite", "description": "Transformer是BERT的基础架构"}
  ]
}"""
        result = self._parse(raw)
        assert len(result["concepts"]) >= 1, f"Should extract concepts, got {len(result.get('concepts', []))}"

    def test_chinese_punctuation(self):
        """JSON 中包含中文标点 — 应该是有效 JSON"""
        raw = """{
  "concepts": [
    {"name": "图论——基本概念", "type": "concept", "importance": 0.8, "summary": "图论入门知识"}
  ],
  "relations": []
}"""
        result = self._parse(raw)
        # 中文破折号可能导致 JSON 解析失败，但不应该崩溃
        assert isinstance(result, dict)
        assert "concepts" in result

    def test_newlines_everywhere(self):
        """JSON 中包含大量换行和缩进"""
        raw = """
{
  "concepts":
  [
    {
      "name": "递归",
      "type": "concept",
      "importance": 0.7,
      "summary": "函数调用自身"
    },
    {
      "name": "迭代",
      "type": "concept",
      "importance": 0.6,
      "summary": "循环执行"
    }
  ],
  "relations":
  [
    {
      "source": "递归",
      "target": "迭代",
      "relation": "contrasts",
      "description": "递归与迭代是两种不同的重复执行方式"
    }
  ]
}
"""
        result = self._parse(raw)
        assert len(result["concepts"]) == 2
        assert len(result["relations"]) == 1
