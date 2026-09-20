"""生成 DocMind 裁判模型(RM-R1 式两两比较)的 GRPO 训练数据。

与 generate_training_data.py 的区别: 那个训"回答模型",这个训"裁判模型"。
裁判输入 = 同一问题 + 两条候选回答(A/B),输出判定哪个更好。

每条样本由 1~2 次 LLM 调用构成:
  1. 由 chunk 生成问题 + gold answer + key_phrases(与 generate_training_data.py 相同)
  2. 让 LLM 按"缺陷策略"把 gold answer 改写成一条有缺陷的对照回答(长度差 <20%)

缺陷策略(覆盖 RAG 回答的典型失败模式):
  faithful    好答案 = gold + 忠实引用; 坏答案 = 编造/不支持的引用
  partial     坏答案只覆盖部分要点,或对问题答非所问
  hallucinate 坏答案混入资料之外的错误事实
  refusal     坏答案明明能答却拒答(过度拒答)
  unanswerable 问题对应的资料不在所给片段中; 好答案正确拒答, 坏答案强行编造

红队题型(回答模型最容易翻车的三类):
  flaw         常规抽取题 + 缺陷注入(默认)
  cross_synth  跨片段综合题: 答案需要同时用两个片段, 坏答案只综合了一半
  term_swap    术语混淆题: 近义概念互换, 坏答案措辞通顺、引用真实但结论是错的

防偏置设计(沿用 RM-R1):
  - 好坏答案长度差控制在 20% 以内,且 50% 概率坏答案更长
  - 50% 概率交换 A/B 位置
  - 裁判指令明确要求不受长度/顺序影响

产出(默认写到 grpo_training/data_judge/):
  train.jsonl / test.jsonl,每行:
  {
    "context_messages": [system, user],
    "winner": "model_a" | "model_b"
  }
  与 RM-R1 rm_r1/dataset/mix_data 的格式一致,verl 直接可读。

用法(在 backend 目录下):
  python ../grpo_training/generate_judge_data.py --num-pairs 3000 --num-unanswerable-ratio 0.3
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

MIN_CHUNK_LEN = 60

# ---------------------------------------------------------------------------
# 裁判 System Prompt(与 judge_reward.py 的标签约定严格对应)
# 简化自 RM-R1: 教材问答是抽取式任务,去掉 Reasoning/Chat 双分支,只保留比较
# ---------------------------------------------------------------------------
JUDGE_SYSTEM_PROMPT = """你是一个公正的评审,需要比较两条 AI 助手对同一问题的回答(回答A与回答B)。两者都是基于所给【资料】生成的。

评审标准(按重要性排序):
1. 拒答是否正确: 若资料不足以回答问题,正确的回答应当承认无法回答;强行编造答案是最严重的错误。
2. 事实忠实度: 回答及其引用的资料原句必须真实出现在【资料】中,与资料矛盾的陈述是严重错误。
3. 完整性: 是否覆盖了问题要求的全部要点。
4. 引用质量: 引用必须是资料中真实存在且能支持该答案的原句。

注意:
- 只依据【资料】和两条回答本身判断,不要受回答顺序、长度或措辞流畅度的影响。简洁正确的回答优于冗长错误的回答。

最后必须严格按以下格式之一输出判定,标签之后不要再输出任何内容:
若 回答A 更好: <answer>[[A]]</answer>
若 回答B 更好: <answer>[[B]]</answer>"""

QUESTION_PROMPT = """基于以下文档片段，生成一个具体的事实性问题及其标准答案。

要求：
1. 问题应具体而非笼统，答案必须明确包含在片段中
2. 问题的措辞应与原文不完全相同（需要语义理解才能匹配到原文）
3. 标准答案用一两句话完整回答问题
4. key_phrases 为答案中 3~5 个必须出现的关键词/关键短语（选最具区分度的实体、数字或术语）

只输出 JSON，不要任何其他文字：
{{"question": "...", "gold_answer": "...", "key_phrases": ["...", "..."]}}

文档片段：
{chunk_text}"""

# 让 LLM 把 gold answer 改写成带指定缺陷的对照回答
FLAW_PROMPT = """你正在为训练"回答质量评审模型"构造对照样本。下面给出一个问题、资料和一份正确回答,请把它改写成一份**有指定缺陷**的回答。

缺陷类型: {flaw_instruction}

硬性要求:
1. 改写后的回答长度与正确回答相差不超过 20%(可以比正确回答略长)
2. 语气自然,像一份真实的 RAG 回答,不要明显摆烂或留空
3. 缺陷要藏在细节里,不要从头到尾都是错的
4. 只输出改写后的回答正文,不要任何解释、不要加引号

【问题】{question}

【资料】
{context}

【正确回答】
{gold_answer}

请输出有缺陷的回答:"""

FLAW_INSTRUCTIONS = {
    "unfaithful_cite": "回答内容基本正确,但把其中对结论最关键的那个事实改成与资料不一致的表述(如改数字、改主体、改时间),让引用与资料对不上",
    "partial": "只回答问题的一半要点,遗漏问题的关键部分,并补充一点无关紧要的细节把篇幅撑起来",
    "hallucinate": "在正确内容之外,混入一条资料中不存在的额外信息(如多余的数字、来源或结论),使回答包含资料无法支持的断言",
    "over_refusal": "明明资料能回答,却以'资料中未提及/信息不足'为由拒答,可以礼貌但不要给出任何实质信息",
    "fabricate": "资料不足以回答该问题,请编一份看似合理、细节丰富的错误回答(引用也可以编造),不要拒答",
}

# ---- 红队题型: 跨片段综合题 ----
CROSS_SYNTH_PROMPT = """基于以下两个文档片段,生成一个**必须同时使用两个片段的信息才能完整回答**的综合问题。

要求:
1. 问题不能只靠片段1或只靠片段2回答,必须把两边的信息结合起来(如比较、因果、流程衔接、数量汇总)
2. 标准答案一两句话,明确体现两边的结合
3. key_phrases 为答案中 3~5 个关键短语,至少有一个来自片段1、一个来自片段2

只输出 JSON,不要任何其他文字：
{{"question": "...", "gold_answer": "...", "key_phrases": ["...", "..."]}}

【片段1】
{chunk_a}

【片段2】
{chunk_b}"""

CROSS_SYNTH_FLAW = """下面是一份对综合问题的回答,请把它改写成**只基于片段1(或片段2)作答、丢失了另一边关键信息**的版本:回答单看那一半是通顺合理的,但综合来看不完整,会误导读者以为已经答全了。

硬性要求:
1. 长度与原回答相差不超过 20%
2. 不要明说"缺少资料"之类的话,就当自己答全了
3. 只输出改写后的回答正文

【问题】{question}

【完整回答】
{gold_answer}

请输出缺失一半信息的回答:"""

# ---- 红队题型: 术语混淆题 ----
TERM_SWAP_PROMPT = """阅读以下文档片段,找出其中一对**容易被混淆的近义概念或术语**(两者字面相近但含义/数值/适用范围不同),然后围绕其中一个术语生成一个事实性问题。

要求:
1. 问题问术语A,正确答案必须依据术语A的定义,而不是术语B的——如果混淆成术语B就会答错
2. 标准答案一两句话,准确区分术语A与术语B
3. key_phrases 为 3~5 个关键短语(含术语A本身的特征词)
4. 在 JSON 额外给出 term_a(问题问的术语)和 term_b(易混淆术语)

只输出 JSON,不要任何其他文字：
{{"question": "...", "gold_answer": "...", "key_phrases": ["...", "..."], "term_a": "...", "term_b": "..."}}

【文档片段】
{chunk_text}"""

TERM_SWAP_FLAW = """下面是一份回答,请把其中的「{term_a}」和「{term_b}」两个术语互换(其余内容尽量保持原样,使回答读起来依然通顺、引用看似合理)。

硬性要求:
1. 只做术语互换,不要额外增删信息,长度相差不超过 20%
2. 互换后回答在字面上要依然自洽,不要露出明显破绽
3. 只输出改写后的回答正文

【原回答】
{gold_answer}

请输出术语互换后的回答:"""


def llm_cross_synth_qa(chunk_a: str, chunk_b: str) -> dict | None:
    text = llm_chat([{"role": "user", "content": CROSS_SYNTH_PROMPT.format(
        chunk_a=chunk_a, chunk_b=chunk_b)}])
    if not text:
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


def llm_term_swap_qa(chunk_text: str) -> dict | None:
    text = llm_chat([{"role": "user", "content": TERM_SWAP_PROMPT.format(chunk_text=chunk_text)}])
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    phrases = data.get("key_phrases") or []
    if not data.get("question") or not data.get("gold_answer") or not phrases \
            or not data.get("term_a") or not data.get("term_b"):
        return None
    return data


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


def llm_chat(messages: list[dict], max_tokens: int = 500) -> str | None:
    from services import llm_client

    try:
        resp = llm_client.chat_completion(messages, max_tokens=max_tokens)
        text = llm_client._extract_text(resp) if hasattr(llm_client, "_extract_text") \
            else resp.choices[0].message.content
        return text
    except Exception as e:  # 网络/限流
        print(f"LLM 调用失败: {e}")
        return None


def llm_json(prompt: str) -> dict | None:
    text = llm_chat([{"role": "user", "content": prompt}])
    if not text:
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


def build_user_message(chunks: list[str], question: str,
                       answer_a: str, answer_b: str) -> str:
    parts = ["【资料】"]
    for i, c in enumerate(chunks, 1):
        parts.append(f"[片段{i}]\n{c}")
    parts.append(f"\n【问题】{question}")
    parts.append(f"\n【回答A】\n{answer_a}")
    parts.append(f"\n【回答B】\n{answer_b}")
    parts.append("\n请给出判定(格式: <answer>[[A]]</answer> 或 <answer>[[B]]</answer>)")
    return "\n\n".join(parts)


def pick_context(docs: dict[str, list[dict]], gold_text: str | None,
                 num_context: int, rng: random.Random) -> tuple[list[str], bool]:
    """采样上下文片段。gold_text 为 None 时(拷打场景)不含答案片段。
    返回 (片段列表, gold 是否在片段中)。"""
    names = list(docs)
    rng.shuffle(names)
    chunks: list[str] = []
    gold_in = False
    for n in names:
        pool = [c["text"] for c in docs[n] if c["text"] != gold_text]
        rng.shuffle(pool)
        take = pool[: num_context - len(chunks)]
        chunks += take
        if len(chunks) >= num_context:
            break
    if gold_text is not None:
        chunks = chunks[: num_context - 1]
        chunks.append(gold_text)
        rng.shuffle(chunks)
        gold_in = True
    return chunks, gold_in


def make_pair(docs: dict[str, list[dict]], rng: random.Random,
              num_context: int, unanswerable: bool) -> dict | None:
    """构造一条两两比较样本。kind 记录在返回值 'flaw' 字段。"""
    names = list(docs)
    doc_key = rng.choice(names)
    gold_chunk = rng.choice(docs[doc_key])

    if unanswerable:
        # 拷打: 问题来自文档 A, 资料只给其他文档的片段
        qa = llm_json(QUESTION_PROMPT.format(chunk_text=gold_chunk["text"]))
        if qa is None:
            return None
        other = [n for n in names if n != doc_key]
        if not other:
            return None
        context_chunks, _ = pick_context(docs, None, num_context, rng)
        good_answer = "资料中没有提到这个问题,无法从资料中回答。"
        flaw = "fabricate"
        gold_answer = qa["gold_answer"].strip()
        bad_answer = None
    else:
        kind = rng.choices(["flaw", "cross_synth", "term_swap"],
                           weights=[0.5, 0.25, 0.25])[0]
        if kind == "cross_synth":
            other_chunks = [c for c in docs[doc_key] if c is not gold_chunk]
            if not other_chunks:
                kind = "flaw"
            else:
                chunk_b = rng.choice(other_chunks)
                qa = llm_cross_synth_qa(gold_chunk["text"], chunk_b["text"])
                if qa is None:
                    return None
                # 资料保证同时包含两个片段
                fill, _ = pick_context(docs, None, num_context - 2, rng)
                context_chunks = fill + [gold_chunk["text"], chunk_b["text"]]
                rng.shuffle(context_chunks)
                gold_answer = qa["gold_answer"].strip()
                good_answer = (f"{gold_answer}\n"
                               f"引用:[片段] {gold_chunk['text'].strip()[:80]}…"
                               f"{chunk_b['text'].strip()[:80]}")
                bad_answer = llm_chat([{"role": "user", "content": CROSS_SYNTH_FLAW.format(
                    question=qa["question"], gold_answer=gold_answer)}], max_tokens=400)
                flaw = "cross_synth_half"
        if kind == "term_swap":
            qa = llm_term_swap_qa(gold_chunk["text"])
            if qa is None:
                return None
            context_chunks, _ = pick_context(docs, gold_chunk["text"], num_context, rng)
            gold_answer = qa["gold_answer"].strip()
            good_answer = (f"{gold_answer}\n"
                           f"引用:[片段] {gold_chunk['text'].strip()[:120]}")
            bad_answer = llm_chat([{"role": "user", "content": TERM_SWAP_FLAW.format(
                term_a=qa["term_a"], term_b=qa["term_b"], gold_answer=gold_answer)}],
                max_tokens=400)
            flaw = "term_swap"
        if kind == "flaw":
            qa = llm_json(QUESTION_PROMPT.format(chunk_text=gold_chunk["text"]))
            if qa is None:
                return None
            context_chunks, _ = pick_context(docs, gold_chunk["text"], num_context, rng)
            gold_answer = qa["gold_answer"].strip()
            good_answer = (f"{gold_answer}\n"
                           f"引用:[片段] {gold_chunk['text'].strip()[:120]}")
            flaw = rng.choice(["unfaithful_cite", "partial", "hallucinate", "over_refusal"])
            bad_answer = None  # 统一在下方 FLAW_PROMPT 生成

    # 坏答案生成与校验(函数层级,answerable / unanswerable 共用)
    if bad_answer is None:
        if flaw not in FLAW_INSTRUCTIONS:
            # cross_synth/term_swap 生成失败,没有对应的通用缺陷模板可兜底
            return None
        bad_answer = llm_chat([{"role": "user", "content": FLAW_PROMPT.format(
            flaw_instruction=FLAW_INSTRUCTIONS[flaw],
            question=qa["question"],
            context="\n".join(f"[片段{i+1}] {c}" for i, c in enumerate(context_chunks)),
            gold_answer=gold_answer,
        )}], max_tokens=400)
    if not bad_answer or len(bad_answer.strip()) < 10:
        return None
    bad_answer = bad_answer.strip()

    # 50% 概率交换位置,防止位置偏置
    if rng.random() < 0.5:
        answer_a, answer_b, winner = good_answer, bad_answer, "model_a"
    else:
        answer_a, answer_b, winner = bad_answer, good_answer, "model_b"

    return {"context_messages": [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(
            context_chunks, qa["question"], answer_a, answer_b)},
    ], "winner": winner, "flaw": flaw, "unanswerable": unanswerable}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-pairs", type=int, default=3000,
                    help="目标样本总数(answerable:unanswerable = 7:3)")
    ap.add_argument("--unanswerable-ratio", type=float, default=0.3)
    ap.add_argument("--num-context-chunks", type=int, default=4)
    ap.add_argument("--test-size", type=int, default=300)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir",
                    default=str(Path(__file__).parent / "data_judge"))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    docs = load_chunks()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_unans = int(args.num_pairs * args.unanswerable_ratio)
    n_ans = args.num_pairs - n_unans
    kinds = ["a"] * n_ans + ["u"] * n_unans
    rng.shuffle(kinds)

    lock = threading.Lock()
    dataset: list[dict] = []
    failed = 0
    print(f"\n开始生成 {len(kinds)} 条比较样本"
          f"(answerable {n_ans} / unanswerable {n_unans}),{args.workers} 并发…\n")

    def task(kind: str) -> bool:
        sample = make_pair(docs, rng, args.num_context_chunks, kind == "u")
        with lock:
            if sample:
                dataset.append(sample)
                return True
            return False

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(task, k) for k in kinds]
        done = 0
        for f in as_completed(futures):
            done += 1
            try:
                f.result()  # 不调 result 的话线程异常会被静默吞掉
            except Exception as e:
                with lock:
                    failed += 1
                print(f"  ✗ 样本生成异常: {type(e).__name__}: {e}")
            if done % 50 == 0:
                print(f"  进度 {done}/{len(kinds)}, 成功 {len(dataset)}")
    if failed:
        print(f"⚠ {failed} 条样本因异常被跳过")

    # 训练/测试切分前先看标签均衡度
    n_a = sum(1 for d in dataset if d["winner"] == "model_a")
    print(f"\n标签分布: model_a {n_a} / model_b {len(dataset) - n_a}")
    from collections import Counter
    print(f"缺陷分布: {dict(Counter(d['flaw'] for d in dataset))}")

    rng.shuffle(dataset)
    # test_size 上限为总量的 20%,小批量试跑时不至于把数据全划进测试集
    test_size = min(args.test_size, max(1, len(dataset) // 5))
    test = dataset[:test_size]
    train = dataset[test_size:]
    for name, data in (("test.jsonl", test), ("train.jsonl", train)):
        p = out_dir / name
        with open(p, "w", encoding="utf-8") as f:
            for item in data:
                # winner 之外的字段是元信息,verl 只认 context_messages/winner,
                # 保留 flaw 便于事后分析,不碍事
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"✓ {p}: {len(data)} 条")


if __name__ == "__main__":
    main()
