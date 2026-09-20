"""红队候选采样:同一查询用多种配置采样候选回答,用 rag_reward 规则打分,
组内最高/最低分自动配成裁判训练对——无需人工成对标注。

思路(对应 GRPO 的组内相对比较,只是这里用于产出裁判模型的监督对):
  1. 从 chunks_*.json 生成查询(answerable + unanswerable 拷打)
  2. 每条查询按配置网格采样 4 个候选(温度 × 有无引用约束 × 上下文长度)
  3. rag_reward.compute_score 给每个候选打分(格式/引用忠实度/关键词命中/拒答)
  4. best 与 worst 配成一对;分差太小(区分度不足)则丢弃

与 generate_judge_data.py 的区别:那边坏答案是 LLM 按缺陷模板"人造"的;
这边坏答案是不同配置下模型的真实失败输出,分布更接近线上。

每条样本的 LLM 调用 = 1(出题) + 4(采样) + 0(打分是纯规则),与那边相当。

用法(在 backend 目录下):
  python ../grpo_training/generate_pairs_from_sampling.py --num-pairs 1000
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import generate_judge_data as gjd          # noqa: E402  复用出题/上下文/格式
import generate_training_data as gtd       # noqa: E402  复用回答模型的 System Prompt
from rag_reward import compute_score       # noqa: E402  规则打分

BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

# (temperature, 强制引用, 上下文片段数)——刻意拉开候选质量差距
SAMPLING_CONFIGS = [
    (0.2, True, 4),    # 低温+引用约束+全上下文 → 预期最好
    (0.7, True, 4),    # 常规配置
    (0.9, True, 2),    # 高温+上下文缺失 → 容易漏要点
    (0.7, False, 4),   # 无引用约束 → 容易自由发挥/编造
]

MIN_SCORE_GAP = 0.3   # best/worst 分差低于此值丢弃(一对答案太接近,裁判学了没用)
MIN_GAP_FRACTION = 0.5  # 分差不足时,仍以该比例的概率保留(保留少量难样本)


def answer_with_config(context_chunks: list[str], question: str,
                       temperature: float, require_cite: bool) -> str | None:
    """用指定配置让 DeepSeek 生成一条候选回答。"""
    system = gtd.SYSTEM_PROMPT if require_cite else gtd.SYSTEM_PROMPT.replace(
        "<cite>资料中支持该答案的原句(逐字摘录,不要改写)</cite>\n", "")
    parts = ["【资料】"]
    for i, c in enumerate(context_chunks, 1):
        parts.append(f"[片段{i}]\n{c}")
    parts.append(f"\n【问题】{question}")
    prompt = "\n\n".join(parts)
    from services import llm_client
    try:
        resp = llm_client.chat_completion(
            [{"role": "system", "content": system},
             {"role": "user", "content": prompt + "\n\n请回答。"}],
            max_tokens=400, temperature=temperature, caller="judge_data_sampling")
        text = llm_client._extract_text(resp) if hasattr(llm_client, "_extract_text") \
            else resp.choices[0].message.content
        return (text or "").strip() or None
    except Exception as e:
        print(f"采样失败: {e}")
        return None


def make_sampled_pair(docs: dict[str, list[dict]], rng: random.Random,
                      num_context: int, unanswerable: bool) -> dict | None:
    names = list(docs)
    doc_key = rng.choice(names)
    gold_chunk = rng.choice(docs[doc_key])
    qa = gjd.llm_json(gjd.QUESTION_PROMPT.format(chunk_text=gold_chunk["text"]))
    if qa is None:
        return None

    if unanswerable:
        other = [n for n in names if n != doc_key]
        if not other:
            return None
        context_chunks, _ = gjd.pick_context(docs, None, num_context, rng)
        gt = {"answerable": False, "gold_answer": "", "key_phrases": [],
              "context": "\n\n".join(context_chunks)}
    else:
        context_chunks, _ = gjd.pick_context(docs, gold_chunk["text"], num_context, rng)
        gt = {"answerable": True, "gold_answer": qa["gold_answer"],
              "key_phrases": qa["key_phrases"][:5],
              "context": "\n\n".join(context_chunks)}
    gt_json = json.dumps(gt, ensure_ascii=False)

    # 多配置采样
    candidates: list[tuple[float, str]] = []
    for temperature, require_cite, n_ctx in SAMPLING_CONFIGS:
        if n_ctx < len(context_chunks):  # 缺上下文配置: 只给前 n_ctx 个片段(已 shuffle,含金片段可能被裁掉)
            ctx = context_chunks[:n_ctx]
        else:
            ctx = context_chunks
        answer = answer_with_config(ctx, qa["question"], temperature, require_cite)
        if answer:
            # 打分用的 context 必须与该候选实际看到的资料一致
            cand_gt = dict(gt, context="\n\n".join(ctx))
            score = compute_score("t", answer, json.dumps(cand_gt, ensure_ascii=False))
            candidates.append((score, answer))
    if len(candidates) < 3:
        return None

    candidates.sort(key=lambda x: x[0])
    worst_score, worst = candidates[0]
    best_score, best = candidates[-1]
    gap = best_score - worst_score
    if gap <= 0:
        return None  # 分数为 0: 两条回答等价,winner 标签只能是噪声
    if gap < MIN_SCORE_GAP and rng.random() >= MIN_GAP_FRACTION:
        return None
    if best_score <= 0:  # 全军覆没的组没有"好答案",配对会教坏裁判
        return None

    if rng.random() < 0.5:
        answer_a, answer_b, winner = best, worst, "model_a"
    else:
        answer_a, answer_b, winner = worst, best, "model_b"

    return {"context_messages": [
        {"role": "system", "content": gjd.JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": gjd.build_user_message(
            context_chunks, qa["question"], answer_a, answer_b)},
    ], "winner": winner, "flaw": "sampled", "unanswerable": unanswerable,
        "score_gap": round(gap, 3)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-pairs", type=int, default=1000)
    ap.add_argument("--unanswerable-ratio", type=float, default=0.3)
    ap.add_argument("--num-context-chunks", type=int, default=4)
    ap.add_argument("--test-size", type=int, default=100)
    ap.add_argument("--workers", type=int, default=4,
                    help="每条样本要串行发 5 次 LLM 请求,并发别开太大")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir",
                    default=str(Path(__file__).parent / "data_judge"))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    docs = gjd.load_chunks()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_unans = int(args.num_pairs * args.unanswerable_ratio)
    kinds = ["a"] * (args.num_pairs - n_unans) + ["u"] * n_unans
    rng.shuffle(kinds)

    lock = threading.Lock()
    dataset: list[dict] = []
    print(f"\n开始采样配对 {len(kinds)} 条(每条 {1 + len(SAMPLING_CONFIGS)} 次 LLM 调用,"
          f"{args.workers} 并发)…\n")

    def task(kind: str) -> bool:
        sample = make_sampled_pair(docs, rng, args.num_context_chunks, kind == "u")
        with lock:
            if sample:
                dataset.append(sample)
                return True
            return False

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(task, k) for k in kinds]
        done = 0
        failed = 0
        for f in as_completed(futures):
            done += 1
            try:
                f.result()  # 不调 result 的话线程异常会被静默吞掉
            except Exception as e:
                failed += 1
                print(f"  ✗ 样本生成异常: {type(e).__name__}: {e}")
            if done % 20 == 0:
                print(f"  进度 {done}/{len(kinds)}, 成功 {len(dataset)}")
    if failed:
        print(f"⚠ {failed} 条样本因异常被跳过")

    n_a = sum(1 for d in dataset if d["winner"] == "model_a")
    gaps = [d["score_gap"] for d in dataset]
    print(f"\n标签分布: model_a {n_a} / model_b {len(dataset) - n_a}")
    print(f"分差: min {min(gaps):.2f} / 均值 {sum(gaps) / len(gaps):.2f} / max {max(gaps):.2f}")
    print(f"构成: {dict(Counter('unans' if d['unanswerable'] else 'ans' for d in dataset))}")

    rng.shuffle(dataset)
    # test_size 上限为总量的 20%,小批量试跑时不至于把数据全划进测试集
    test_size = min(args.test_size, max(1, len(dataset) // 5))
    test = dataset[:test_size]
    train = dataset[test_size:]
    for name, data in (("test.jsonl", test), ("train.jsonl", train)):
        p = out_dir / f"{name.replace('.jsonl', '')}_sampled.jsonl"
        with open(p, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"✓ {p}: {len(data)} 条")
    print("\n注意: 输出带 _sampled 后缀,不会覆盖 generate_judge_data.py 的产物;"
          "两份数据可以合并训练。")


if __name__ == "__main__":
    main()
