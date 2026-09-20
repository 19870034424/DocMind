"""DocMind 裁判模型(RM-R1 式)的可验证奖励函数(verl custom_reward_function 兼容)。

约定: 裁判在推理结束后,输出末尾必须严格是
  <answer>[[A]]</answer>  或  <answer>[[B]]</answer>
winner 即 ground_truth("model_a" / "model_b",由 generate_judge_data.py 写入)。

判定沿用 RM-R1 lm_as_judge.py 的防作弊方式: 只看输出的最后 80 个字符,
思维链里提前出现的标签不算数。判对 +1,判错 -1。

单测: python judge_reward.py --test
"""

from __future__ import annotations

LOOKBACK_CHARS = 80
CORRECT = 1.0
WRONG = -1.0


def judge_match(solution_str: str, winner: str) -> float:
    pred = solution_str[-LOOKBACK_CHARS:]
    if winner == "model_a":
        ok = "<answer>[[A]]</answer>" in pred and "<answer>[[B]]</answer>" not in pred
    elif winner == "model_b":
        ok = "<answer>[[B]]</answer>" in pred and "<answer>[[A]]</answer>" not in pred
    else:
        raise ValueError(f"非法 winner 标签: {winner!r}(检查数据集生成)")
    return CORRECT if ok else WRONG


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **_) -> float:
    """verl 自定义奖励入口(兼容位置签名与关键字调用)。"""
    return judge_match(solution_str, ground_truth)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    assert args.test, "用法: python judge_reward.py --test"

    cases = [
        # (模型输出, winner, 期望分, 说明)
        ("先分析资料……综合来看回答A更忠实。<answer>[[A]]</answer>", "model_a", 1.0, "判A且A赢"),
        ("先分析资料……综合来看回答A更忠实。<answer>[[A]]</answer>", "model_b", -1.0, "判A但B赢"),
        ("……引用是编造的,回答B更可靠。<answer>[[B]]</answer>", "model_b", 1.0, "判B且B赢"),
        ("<answer>[[A]]</answer>长篇分析……最后改判<answer>[[B]]</answer>", "model_a", -1.0,
         "思维链早期出现A标签不算,只看末尾"),
        ("两者都不好", "model_a", -1.0, "没输出标签"),
        ("判定:<answer>[[AB]]</answer>", "model_a", -1.0, "标签格式错"),
        ("分析……<answer>[[A]]</answer>和<answer>[[B]]</answer>都写了", "model_a", -1.0,
         "末尾两个标签都出现,视为不明确"),
    ]
    ok = True
    for out, winner, expect, note in cases:
        s = compute_score("t", out, winner)
        passed = abs(s - expect) < 1e-6
        ok &= passed
        print(f"{'PASS' if passed else 'FAIL'}  {note}: {s:+.1f} (期望 {expect:+.1f})")
    print("\n全部通过" if ok else "\n存在失败用例")
