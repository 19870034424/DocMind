"""生成 DocMind RAG 回答模型的 GRPO 训练数据。

复用 backend 的 services.llm_client(需要 backend/.env 有 DeepSeek key)。
数据源: evaluation/datasets/chunks_*.json(评测时保存的分块)。

产出(默认写到 grpo_training/data/):
  train.jsonl / test.jsonl,每行:
  {
    "context_messages": [system, user],        # verl RubricRMDataset 直接可读
    "winner": "{...ground truth JSON...}"      # rag_reward.compute_score 的 ground_truth
  }

用法(在 backend 目录下):
  python ../grpo_training/generate_training_data.py --num-answerable 300 --num-unanswerable 100
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

MIN_CHUNK_LEN = 60  # 太短的 chunk 出不了像样的问题

SYSTEM_PROMPT = """你是一个严谨的文档问答助手。下面会给你编号的【资料】片段和一个【问题】。

规则:
1. 只依据资料回答,禁止使用资料之外的任何知识。
2. 先在内心逐步分析各片段是否包含答案,再输出结论。
3. 输出必须严格为以下格式之一。

若资料能回答:
<answer>简洁、完整的最终答案</answer>
<cite>资料中支持该答案的原句(逐字摘录,不要改写)</cite>

若资料不足以回答(缺失、无关或只有部分信息):
<answer>无法从资料中回答</answer>

注意:宁可直接引用原句也不要意译;拿不准就拒答,不要编造。"""

QUESTION_PROMPT = """基于以下文档片段，生成一个具体的事实性问题及其标准答案。

要求：
1. 问题应具体而非笼统，答案必须明确包含在片段中
2. 问题的措辞应与原文不完全相同（需要语义理解才能匹配到原文）
3. 标准答案用一两句话完整回答问题
4. key_phrases 为答案中 3~5 个必须出现的关键词/关键短语（用于自动判分，选最具区分度的实体、数字或术语）

只输出 JSON，不要任何其他文字：
{{"question": "...", "gold_answer": "...", "key_phrases": ["...", "..."]}}

文档片段：
{chunk_text}"""


def load_chunks() -> dict[str, list[dict]]:
    """按文档分组加载所有评测分块文件。"""
    datasets_dir = BACKEND_DIR / "evaluation" / "datasets"
    docs: dict[str, list[dict]] = {}
    for f in sorted(datasets_dir.glob("chunks_*.json")):
        try:
            chunks = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            print(f"跳过损坏文件: {f.name}")
            continue
        good = [c for c in chunks
                if isinstance(c.get("text"), str) and len(c["text"]) >= MIN_CHUNK_LEN]
        if good:
            docs[f.stem] = good
    if not docs:
        sys.exit("evaluation/datasets/ 下没有可用的 chunks_*.json。"
                 "请先在平台里对某文档跑一次评测生成,或手动放一份分块 JSON。")
    for name, cs in docs.items():
        print(f"  {name}: {len(cs)} chunks")
    return docs


def llm_json(prompt: str) -> dict | None:
    from services import llm_client

    try:
        resp = llm_client.chat_completion(
            [{"role": "user", "content": prompt}], max_tokens=500)
        text = llm_client._extract_text(resp) if hasattr(llm_client, "_extract_text") \
            else resp.choices[0].message.content
    except Exception as e:  # 网络/限流
        print(f"LLM 调用失败: {e}")
        return None
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    phrases = data.get("key_phrases") or []
    if not data.get("question") or not data.get("gold_answer") or not phrases:
        return None
    return data


def build_user_message(chunks: list[str], question: str) -> str:
    parts = ["【资料】"]
    for i, c in enumerate(chunks, 1):
        parts.append(f"[片段{i}]\n{c}")
    parts.append(f"\n【问题】{question}")
    return "\n\n".join(parts)


def make_answerable(docs: dict[str, list[dict]], rng: random.Random,
                    num_context: int) -> dict | None:
    names = list(docs)
    doc_key = rng.choice(names)
    doc = docs[doc_key]
    gold = rng.choice(doc)
    qa = llm_json(QUESTION_PROMPT.format(chunk_text=gold["text"]))
    if qa is None:
        return None
    other_names = [n for n in names if n != doc_key]
    rng.shuffle(other_names)
    distractors = []
    for n in other_names:
        if len(distractors) >= num_context - 1:
            break
        distractors.append(rng.choice(docs[n])["text"])
    same_doc = [c["text"] for c in doc if c is not gold]
    distractors += rng.sample(same_doc, min(num_context - len(distractors), len(same_doc)))
    context_chunks = distractors[:num_context - 1] + [gold["text"]]
    rng.shuffle(context_chunks)
    context = "\n\n".join(context_chunks)
    gt = {"answerable": True, "gold_answer": qa["gold_answer"],
          "key_phrases": qa["key_phrases"][:5], "context": context}
    return {"context_messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(context_chunks, qa["question"])},
    ], "winner": json.dumps(gt, ensure_ascii=False)}


def make_unanswerable(docs: dict[str, list[dict]], rng: random.Random,
                      num_context: int) -> dict | None:
    """问题来自文档 A,资料只给其他文档的片段 → 正确行为是拒答。"""
    names = list(docs)
    if len(names) < 2:
        return None
    q_doc_name, a_doc_name = rng.sample(names, 2)
    qa = llm_json(QUESTION_PROMPT.format(chunk_text=rng.choice(docs[q_doc_name])["text"]))
    if qa is None:
        return None
    pool = [c["text"] for c in docs[a_doc_name]]
    context_chunks = rng.sample(pool, min(num_context, len(pool)))
    context = "\n\n".join(context_chunks)
    gt = {"answerable": False, "gold_answer": "", "key_phrases": [], "context": context}
    return {"context_messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(context_chunks, qa["question"])},
    ], "winner": json.dumps(gt, ensure_ascii=False)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-answerable", type=int, default=2100)
    ap.add_argument("--num-unanswerable", type=int, default=900)
    ap.add_argument("--num-context-chunks", type=int, default=4)
    ap.add_argument("--test-size", type=int, default=200)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", default=str(Path(__file__).parent / "data"))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    docs = load_chunks()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lock = threading.Lock()
    dataset: list[dict] = []
    total = args.num_answerable + args.num_unanswerable
    print(f"\n开始生成 {total} 条样本(answerable {args.num_answerable} / "
          f"unanswerable {args.num_unanswerable}),{args.workers} 并发…\n")

    def task(kind: str) -> bool:
        sample = (make_answerable(docs, rng, args.num_context_chunks) if kind == "a"
                  else make_unanswerable(docs, rng, args.num_context_chunks))
        with lock:
            if sample:
                dataset.append(sample)
                return True
            return False

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(task, "a" if i < args.num_answerable else "u")
                   for i in range(total)]
        done = 0
        for f in as_completed(futures):
            done += 1
            if done % 50 == 0:
                print(f"  进度 {done}/{total}, 成功 {len(dataset)}")

    rng.shuffle(dataset)
    test = dataset[:args.test_size]
    train = dataset[args.test_size:]
    for name, data in (("test.jsonl", test), ("train.jsonl", train)):
        p = out_dir / name
        with open(p, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"✓ {p}: {len(data)} 条")
    n_ans = sum(1 for d in train
                if json.loads(d["winner"])["answerable"])
    print(f"训练集构成: answerable {n_ans} / unanswerable {len(train) - n_ans}")


if __name__ == "__main__":
    main()
