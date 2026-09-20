# DocMind RAG 回答模型 GRPO 训练方案(Search-R1 风格)

目标:把 DeepSeek-R1-Distill-Qwen-1.5B 用 GRPO 训成一个"教材领域、带引用、会拒答"的 RAG 回答模型,替换线上 DeepSeek API 调用。

## 输出格式约定(模型学会输出)

```
(自由推理, R1 蒸馏模型原生思维链)
<answer>根据资料回答的最终答案</answer>      ← 资料不足以回答时: <answer>无法从资料中回答</answer>
<cite>从所给资料中摘录的原句</cite>          ← 仅回答得了时需要
```

## 三项可验证奖励(rag_reward.py,纯字符串判定)

| 组件 | 判定方式 | 分值 |
|------|----------|------|
| 格式 | 同时含 `<answer>` 与 `<cite>`(该引用时) | +0.1 |
| 引用忠实度 | 引用句规范化后必须真的出现在所给资料里 | 0 ~ +0.4 |
| 答案正确性 | 资料足够时,gold key_phrases 在答案中的命中率 | 0 ~ +0.6 |
| 拒答正确 | 资料不足时正确拒答 +1.0;强行编造 -0.5;资料足够却拒答 -0.2 | ± |

## 目录内容

- `generate_training_data.py` — 复用 backend 的 chunker/llm_client 反推 QA,生成训练集(answerable 70% + unanswerable 30%)
- `rag_reward.py` — verl 自定义奖励函数,可单测: `python rag_reward.py --test`
- `train_rag_grpo_1.5b.sh` — 单卡 4090 的 verl GRPO 训练脚本(在 AutoDL 服务器跑)

## 步骤

### 1. 造数据(本地,需要 backend/.env 里的 DeepSeek key)

```bash
cd backend
../grpo_training/generate_training_data.py --num-answerable 2100 --num-unanswerable 900 \
    --output ../grpo_training/data/train.jsonl
```

数据源:优先读 `evaluation/datasets/chunks_*.json`;也可用 `--doc-id` 让它现解析。每条样本 = 1 次LLM调用(问题+参考答案+关键词一次生成)。

### 2. 服务器环境 + 模型(已规划,AutoDL)

- 模型: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` → `/root/autodl-tmp/r1-distill-qwen-1.5b`
- verl(vllm 预编译 wheel)/flash-attn 按 RM-R1 仓库 `README_zh.md` 安装

### 3. 训练(AutoDL, 1×4090)

```bash
cd /root/autodl-tmp/RM-R1
bash grpo_training/train_rag_grpo_1.5b.sh
```

要点:GRPO 组大小 `rollout.n=8`(同一题采样 8 个回答,组内算相对优势);KL 0.001 防跑飞;1.5B + bf16 + gradient checkpointing 在 24G 上稳妥。

### 4. 评测与上线

- 用 `backend/evaluation/runner.py` 的思路对比:DeepSeek API vs 训练模型(正确率/忠实度/拒答准确率)
- 合格后走 vLLM 起本地服务,`services/llm_client.py` 加一个 provider 即可切换

## 调参入口

- 奖励权重: `rag_reward.py` 顶部常量
- 拒答比例: 生成数据的 `--unanswerable-ratio`
- 难度: 生成数据的干扰 chunk 数量 `--num-context-chunks`(默认 4,越多越难)

---

# 附:裁判模型(RM-R1 式两两比较)

上面训的是**回答模型**;这部分沿用 RM-R1 论文思路,再训一个**裁判模型**,用于离线评测(替代/补充关键词指标)和 Best-of-N 选优。两个模型相互独立,可只训其一。

## 原理

裁判输入 = 同一问题 + 两条候选回答(A/B),输出推理链后以 `<answer>[[A]]</answer>` / `<answer>[[B]]</answer>` 给出判定。奖励极简可验证:只看输出末尾 80 字符,判对 +1、判错 −1(思维链里提前放标签不算数,防作弊)。

## 文件

- `generate_judge_data.py` — 生成两两比较数据。好答案 = gold answer + 忠实引用;坏答案由 LLM 按缺陷策略改写,覆盖 RAG 典型失败模式(引用不实/答一半/混入幻觉/过度拒答/资料不足时编造)。**含三类红队题型**: 常规抽取题(缺陷注入)、跨片段综合题(坏答案只综合一半)、术语混淆题(近义概念互换,引用真实但结论错)。沿用 RM-R1 防偏置:50% 交换 A/B 位置、好坏答案长度差 <20%、指令明确不受长度顺序影响
- `generate_pairs_from_sampling.py` — 红队候选采样配对:同一条查询用 4 种配置(温度 × 有无引用约束 × 上下文长度)采样候选回答,用 `rag_reward.py` 规则打分,组内最高/最低分自动配成一对——坏答案是模型的真实失败输出而非人造缺陷,无需成对标注。分差 <0.3 的对大概率丢弃(留 50% 作难样本)
- `judge_reward.py` — verl 自定义奖励(`compute_score`),可单测: `python judge_reward.py --test`
- `train_rag_judge_grpo_1.5b.sh` — 同一套 verl/1.5B/单卡 4090 配置的训练脚本
- `run_judge.py` — 推理入口:单次比较(`--question/--answer-a/--answer-b`)或批量评测(`--input pairs.jsonl`)

## 步骤

```bash
# 1. 造数据(本地,backend 目录下,需要 DeepSeek key;每条样本 2 次 LLM 调用)
python ../grpo_training/generate_judge_data.py --num-pairs 3000

# 2. 服务器训练(环境与 train_rag_grpo_1.5b.sh 相同)
bash grpo_training/train_rag_judge_grpo_1.5b.sh

# 3. merge checkpoint 后批量评测
python run_judge.py --model <merged_model路径> --input pairs.jsonl --output judged.jsonl
```

## 调参入口

- 缺陷策略配比: `generate_judge_data.py` 的 `FLAW_INSTRUCTIONS` / `make_pair` 里的 `rng.choices` 权重(flaw/cross_synth/term_swap = 5:2.5:2.5)
- 拷打(资料不足)比例: `--unanswerable-ratio`(默认 0.3)
- 判定严格度: 训练数据生成时的上下文片段数 `--num-context-chunks`(越多干扰越强,裁判越难)
- 采样配对的配置网格与最低分差: `generate_pairs_from_sampling.py` 的 `SAMPLING_CONFIGS` / `MIN_SCORE_GAP`

## 两份数据怎么选

| | `generate_judge_data.py` | `generate_pairs_from_sampling.py` |
|---|---|---|
| 坏答案来源 | LLM 按缺陷模板改写 | 多配置采样的真实失败输出 |
| LLM 调用/条 | 2 | 5 |
| 覆盖 | 题型全(含跨片段综合、术语混淆) | 失败分布更接近线上 |
| 建议 | 主体数据(约 3000 条) | 混入 1000 条左右增加真实性 |

两者输出格式相同,合并后一起训练即可。

