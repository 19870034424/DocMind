"""DocMind RAG 回答模型的可验证奖励函数(verl custom_reward_function 兼容)。

模型输出约定:
  <answer>最终答案</answer>      资料不足时: <answer>无法从资料中回答</answer>
  <cite>资料原句摘录</cite>       仅 answerable 样本要求

ground_truth 是 JSON 字符串(由 generate_training_data.py 写入 winner 字段):
  {"answerable": true, "gold_answer": "...", "key_phrases": ["..."], "context": "全部资料拼接"}

纯字符串判定,无模型调用。单测: python rag_reward.py --test
"""

from __future__ import annotations

import json
import re

# ---- 奖励权重(总范围约 [-0.5, 1.1]) ----
W_FORMAT = 0.1        # 格式正确
W_FAITHFUL = 0.4      # 引用忠实度(命中句占比)
W_CORRECT = 0.6       # 关键词命中率
R_REFUSE_OK = 1.0     # 该拒答时正确拒答
R_REFUSE_MISS = -0.2  # 有答案却拒答
R_HALLUCINATE = -0.5  # 无答案却强行回答

REFUSE_KEYWORDS = [
    "无法回答", "无法确定", "无法从资料", "资料中未", "资料中没有", "未提及",
    "没有提到", "未提供", "不足以", "信息不足", "无法找到",
]
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.S)
CITE_RE = re.compile(r"<cite>(.*?)</cite>", re.S)


def _norm(t: str) -> str:
    """与 backend/evaluation/metrics.py 同风格的规范化:去空白与标点,转小写。"""
    t = re.sub(r"\s+", "", t)
    t = re.sub(r"[。;!?,.;:!？、\"'“”‘’（）()《》<>【】\[\]{}—\-…·]", "", t)
    return t.lower()


def _split_sentences(t: str) -> list[str]:
    return [s for s in re.split(r"[。;;!！?\？\n]", t) if len(s.strip()) >= 6]


def parse_output(text: str) -> tuple[str | None, str | None]:
    """从模型输出中提取 answer 与 cite,均在最后一个思维链之后。"""
    a = ANSWER_RE.findall(text)
    c = CITE_RE.findall(text)
    answer = a[-1].strip() if a else None
    cite = c[-1].strip() if c else None
    return answer, cite


def is_refusal(answer: str | None) -> bool:
    if not answer:
        return True
    n = _norm(answer)
    return any(_norm(k) in n for k in REFUSE_KEYWORDS)


def citation_faithfulness(cite: str | None, context: str) -> float:
    """引用句规范化后逐句检查是否出现在资料中,返回命中占比。"""
    if not cite:
        return 0.0
    nc = _norm(context)
    if not nc:
        return 0.0
    if _norm(cite) in nc:
        return 1.0
    sents = _split_sentences(cite)
    if not sents:
        return 0.0
    hits = sum(1 for s in sents if _norm(s) in nc)
    return hits / len(sents)


def key_phrase_recall(answer: str, key_phrases: list[str]) -> float:
    if not key_phrases:
        return 0.0
    n = _norm(answer)
    hits = sum(1 for p in key_phrases if _norm(p) and _norm(p) in n)
    return hits / len(key_phrases)


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **_) -> float:
    """verl 自定义奖励入口(兼容位置签名与关键字调用)。"""
    try:
        gt = json.loads(ground_truth) if isinstance(ground_truth, str) else dict(ground_truth)
    except (json.JSONDecodeError, TypeError):
        return 0.0

    answerable = bool(gt.get("answerable", True))
    context = gt.get("context", "")
    answer, cite = parse_output(solution_str)

    if not answerable:
        # 应拒答:正确拒答给满分,编造答案重罚
        if is_refusal(answer):
            return R_REFUSE_OK + (W_FORMAT if cite is not None else 0.0)
        return R_HALLUCINATE

    # 应回答
    if is_refusal(answer):
        return R_REFUSE_MISS

    score = W_FORMAT
    if cite:
        score += W_FAITHFUL * citation_faithfulness(cite, context)
    score += W_CORRECT * key_phrase_recall(answer, gt.get("key_phrases", []))
    return score


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    assert args.test, "用法: python rag_reward.py --test"

    ctx = "DocMind采用FastAPI作为后端框架，前端使用React18与Vite构建。检索采用bge-m3嵌入模型，维度为1024。项目于2024年立项。"
    gt_ok = json.dumps({"answerable": True, "key_phrases": ["FastAPI", "bge-m3"],
                        "context": ctx}, ensure_ascii=False)
    gt_bad = json.dumps({"answerable": False, "context": ctx}, ensure_ascii=False)

    cases = [
        ("完美回答", gt_ok,
         "<think>分析资料...</think>\n<answer>后端用FastAPI，检索用bge-m3嵌入模型。</answer>\n<cite>DocMind采用FastAPI作为后端框架。检索采用bge-m3嵌入模型，维度为1024。</cite>",
         1.1, "格式+全忠实+全命中"),
        ("半对回答", gt_ok,
         "<answer>后端用FastAPI。</answer>\n<cite>采用FastAPI作为后端框架</cite>",
         0.1 + 0.4 + 0.3, "忠实但只命中一个关键词"),
        ("编造引用", gt_ok,
         "<answer>后端用Django，检索用bge-m3。</answer>\n<cite>系统采用Django框架开发</cite>",
         0.1 + 0.0 + 0.3, "引用不实+半命中"),
        ("有答案却拒答", gt_ok,
         "<answer>无法从资料中回答</answer>", R_REFUSE_MISS, "漏答惩罚"),
        ("正确拒答", gt_bad,
         "<answer>资料中未提及立项时间，无法回答。</answer>", R_REFUSE_OK, "该拒答时拒答"),
        ("无中生有", gt_bad,
         "<answer>项目于2023年立项。</answer>\n<cite>项目立项于2023年</cite>", R_HALLUCINATE, "幻觉重罚"),
    ]
    ok = True
    for name, gt, out, expect, note in cases:
        s = compute_score("t", out, gt)
        passed = abs(s - expect) < 1e-6
        ok &= passed
        print(f"{'PASS' if passed else 'FAIL'}  {name}: {s:.2f} (期望 {expect})  # {note}")
    print("\n全部通过" if ok else "\n存在失败用例")
