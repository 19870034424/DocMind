"""加载训练好的裁判模型,对两条候选回答做两两评测。

用途:
  1. 离线评测: 对比 DocMind RAG 输出 vs 参考答案/DeepSeek API 输出
  2. Best-of-N: 对同一问题采样的多个回答两两比较选出最优

用法:
  # 单次比较(命令行直接给两段回答)
  python run_judge.py --model /path/to/merged_model \
      --question "DocMind 用什么嵌入模型?" --answer-a "bge-m3" --answer-b "text2vec"

  # 批量评测: 读 JSONL(每行 question/answer_a/answer_b,可选 context),输出判定
  python run_judge.py --model /path/to/merged_model --input pairs.jsonl --output judged.jsonl

在服务器上运行(模型需先 merge,见 demo/convert_fsdp_to_hf.py 的思路):
  python run_judge.py --model /root/autodl-tmp/docmind_judge/.../merged_model ...
"""

from __future__ import annotations

import argparse
import json
import re

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


def build_user_message(question: str, answer_a: str, answer_b: str,
                       context: str | None) -> str:
    parts = []
    if context:
        parts.append(f"【资料】\n{context}")
    parts.append(f"\n【问题】{question}")
    parts.append(f"\n【回答A】\n{answer_a}")
    parts.append(f"\n【回答B】\n{answer_b}")
    parts.append("\n请给出判定(格式: <answer>[[A]]</answer> 或 <answer>[[B]]</answer>)")
    return "\n".join(parts)


def parse_verdict(text: str) -> str:
    """从裁判输出末尾解析判定: 'A' / 'B' / 'invalid'。只看最后 80 字符,与训练奖励一致。"""
    pred = text[-80:]
    has_a = "<answer>[[A]]</answer>" in pred
    has_b = "<answer>[[B]]</answer>" in pred
    if has_a and not has_b:
        return "A"
    if has_b and not has_a:
        return "B"
    return "invalid"


def load_model(model_name: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def judge(model, tokenizer, question: str, answer_a: str, answer_b: str,
          context: str | None = None, max_new_tokens: int = 2048) -> dict:
    import torch

    conversation = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(question, answer_a, answer_b, context)},
    ]
    input_ids = tokenizer.apply_chat_template(
        conversation, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        generation = model.generate(
            input_ids=input_ids, max_new_tokens=max_new_tokens, do_sample=False)
    completion = tokenizer.decode(
        generation[0][len(input_ids[0]):], skip_special_tokens=True,
        clean_up_tokenization_spaces=True)
    return {"verdict": parse_verdict(completion), "reasoning": completion.strip()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="merge 后的裁判模型路径")
    ap.add_argument("--question")
    ap.add_argument("--answer-a")
    ap.add_argument("--answer-b")
    ap.add_argument("--context", help="可选: 资料原文,评测 RAG 输出时建议给")
    ap.add_argument("--input", help="批量模式: JSONL,每行 question/answer_a/answer_b[/context]")
    ap.add_argument("--output", default="judged.jsonl")
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    args = ap.parse_args()

    model, tokenizer = load_model(args.model)

    if args.input:
        rows = [json.loads(line) for line in open(args.input, encoding="utf-8") if line.strip()]
        with open(args.output, "w", encoding="utf-8") as f:
            n_valid = 0
            for i, row in enumerate(rows):
                result = judge(model, tokenizer, row["question"],
                               row["answer_a"], row["answer_b"],
                               row.get("context"), args.max_new_tokens)
                result.update({"index": i, "question": row["question"]})
                n_valid += result["verdict"] != "invalid"
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                print(f"[{i + 1}/{len(rows)}] {result['verdict']}: {row['question'][:40]}")
        print(f"\n✓ 完成,有效判定 {n_valid}/{len(rows)},结果写入 {args.output}")
    else:
        if not (args.question and args.answer_a and args.answer_b):
            ap.error("单次比较需要 --question/--answer-a/--answer-b,或用 --input 批量模式")
        result = judge(model, tokenizer, args.question, args.answer_a,
                       args.answer_b, args.context, args.max_new_tokens)
        print(f"判定: 回答{result['verdict']} 更好\n\n--- 裁判推理过程 ---\n{result['reasoning']}")


if __name__ == "__main__":
    main()
