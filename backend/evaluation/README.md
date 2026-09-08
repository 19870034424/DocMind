# 检索评估模块

量化评估 RAG 检索质量，支持标准 IR 指标（Recall@K, Precision@K, MRR, NDCG@K, MAP），对比纯向量检索与混合检索（BM25 + 向量 + RRF）的效果差异。

## 快速开始

```bash
cd backend

# 1. 生成评估数据集（上传文档后，LLM 辅助生成问题）
python -m evaluation.generator <doc_id> datasets/doc_<doc_id>.json 20 '文档标题'

# 2. 运行评估
python -m evaluation.runner <doc_id> datasets/doc_<doc_id>.json 5,10,20

# 3. 手动追加测试用例
python -c "
from evaluation import DatasetGenerator
gen = DatasetGenerator()
gen.add_manual_query('datasets/doc_xxx.json', '你的问题？', ['相关的chunk原文'], '备注')
"
```

## 指标说明

| 指标 | 含义 | 公式 | 典型目标值 |
|------|------|------|-----------|
| **Recall@K** | 前 K 个结果中找回的相关 chunk 占比 | `hits / total_relevant` | ≥ 0.80 |
| **Precision@K** | 前 K 个结果中相关 chunk 的占比 | `hits / K` | ≥ 0.30 |
| **MRR** | 第一个相关结果排名的倒数平均值 | `1 / rank_of_first_hit` | ≥ 0.70 |
| **NDCG@K** | 考虑排名位置的相关性得分 | `DCG / IDCG` | ≥ 0.75 |
| **MAP@K** | 平均精度均值 | `(1/R) Σ(P@i × rel_i)` | ≥ 0.60 |

所有指标使用二进制相关性（相关 = 1，不相关 = 0）。

### Recall@K 最重要

在 RAG 场景中，**Recall@K 是最核心的指标**——它直接衡量"用户问题的答案是否被检索到了"。如果答案根本不在检索结果里，LLM 再强也没用。

- Recall@5 = 0.9 意味着 10 个问题里 9 个的正确答案在 top-5 中
- Recall 不高的根源通常是 chunk 太小（信息被切散）或太大（信息被稀释）

### MRR — 衡量"第一个好结果多快出现"

MRR 高说明答案排得靠前。如果 MRR 低但 Recall@20 高，说明答案在结果里但位置靠后——考虑加精排（Cross-Encoder Reranker）。

## 理解评估结果

### 典型输出

```
=================================================================
  Evaluation: 智能客服系统技术方案 (deb7a78c141f)
  Queries: 32, K values: [5, 10, 20]
=================================================================
Metric                   Vector     Hybrid      Delta    Win
--------------------------------------------------------
recall@5                 0.9000     0.9000 +   0.0000   tie
recall@10                0.9000     0.9000 +   0.0000   tie
recall@20                0.9000     0.9000 +   0.0000   tie
mrr                      0.8833     0.8833 +   0.0000   tie
ndcg@5                   0.8877     0.8877 +   0.0000   tie
=================================================================
```

每列含义：
- **Metric**：指标名称
- **Vector**：纯向量检索平均值
- **Hybrid**：BM25 + 向量 + RRF 混合检索平均值
- **Delta**：Hybrid - Vector 的差值（+绿色表示 Hybrid 更好）
- **Win**：哪个更好（Hybrid / Vector / tie）

### 规模对结果的影响

| 文档规模 | 典型 chunk 数 | Vector vs Hybrid 差异 |
|---------|-------------|---------------------|
| 短文（< 20 页） | < 50 | 通常持平，两者都能准确检索 |
| 中篇（20-80 页） | 50-200 | 开始出现差异，关键词查询 Hybrid 占优 |
| 长篇（> 80 页） | 200+ | 差异显著，Hybrid 在 Recall@K 上通常高 3-8 个百分点 |

**注意**：小文档上两者持平是正常且合理的——向量和 BM25 在小搜索空间中都足够准确。

### 查询类型对差异的影响

| 查询类型 | 示例 | 谁更可能胜出 |
|---------|------|------------|
| 精确数字 | "QPS 目标是多少？" | BM25（Hybrid） |
| 专有名词 | "Milvus 几个节点？" | BM25（Hybrid） |
| 同义改写 | "怎么防止编造答案？" | 向量 |
| 概括性问题 | "系统架构是怎样的？" | 两者相当 |
| 跨段落 | "技术和成本相比如何？" | 向量 |

### 深入分析：per-query 结果

评估同时保存详细 JSON，可逐题分析：
```python
import json
with open('datasets/doc_xxx_results.json') as f:
    data = json.load(f)
# data['per_query']['vector'][i] vs data['per_query']['hybrid'][i]
# 对比每个 query 的各指标
```

## 数据集格式

```json
{
  "version": "1.0",
  "doc_id": "deb7a78c141f",
  "document_title": "智能客服系统技术方案",
  "num_chunks": 29,
  "queries": [
    {
      "id": "q1",
      "question": "系统的目标QPS是多少？",
      "relevant_chunk_texts": [
        "峰值 QPS 需达到 2000，平均响应延迟控制在 800ms 以内"
      ],
      "notes": "LLM-generated from chunk 4"
    }
  ]
}
```

- `question`：自然语言问题
- `relevant_chunk_texts`：正确答案所在的 chunk 原文（支持子串匹配）
- `notes`：标注问题来源或类别（便于分类分析）

## 创建高质量评估数据集

### 方法一：LLM 自动生成（推荐）

```bash
python -m evaluation.generator <doc_id> datasets/doc_<doc_id>.json 20 '文档标题'
```

对每个 chunk 调用 LLM 生成事实性问题，chunk 原文作为 ground truth。

**优点**：快速、覆盖面广、问题多样化  
**缺点**：LLM 生成的问题与原文太接近（检索容易），可能高估实际效果

### 方法二：手动追加高价值查询

```python
from evaluation import DatasetGenerator
gen = DatasetGenerator()
gen.add_manual_query(ds_path, question, [chunk_text], notes)
```

手动编写最能体现业务需求的问题，比如：
- **关键词精确查询**："QPS 2000"、"nlist=1024" → 测试 BM25
- **语义同义改写**："怎么防止胡编？" → 测试向量
- **跨段落综合**："技术选型和成本的关系？" → 测试多 chunk 检索

**建议**：LLM 生成 20 题打底 + 手动补充 5-10 题关键场景。

### 方法三：从对话日志中提取

最真实的方法——从实际用户的问答记录中提取问题和正确 chunk：
1. 记录用户问题和 LLM 最终引用的 chunk
2. 人工确认 chunk 确实回答了问题
3. 写入 eval 数据集

## 评估流程最佳实践

1. **基线建立**：首次评估记录所有指标作为 baseline
2. **变更验证**：改 chunk_size、换 embedding 模型、调 RRF k 值后重新评估，对比 baseline
3. **持续监控**：定期（如每周）用最新的评估集跑一次，防止退化
4. **分类分析**：按查询类型（关键词 vs 语义 vs 混合）分组统计，定位具体的薄弱环节

## 常见问题

**Q: Vector 和 Hybrid 总是持平，是不是模块有问题？**  
A: 小文档（< 50 chunks）上两者持平是正常的。差异在大规模文档（200+ chunks）上才能体现。可以尝试：增大测试文档、使用 K=1 或 K=2、构造精确关键词 vs 纯粹语义的对抗性查询。

**Q: 为什么不支持 graded relevance（0-3 分）？**  
A: 当前版本使用二进制相关性（相关/不相关），因为对于 chunk 检索，"这个 chunk 包含答案吗"是二值问题。Graded relevance 更适合段落/文档级检索评估，可以在后续版本中添加。

**Q: 如何评估 LLM 生成答案的质量？**  
A: 当前模块评估检索质量，非生成质量。评估生成质量需要 LLM-as-judge 或人工评估，这块在 Plan 中。检索质量是生成质量的前提——检索不好，答案一定不好。
