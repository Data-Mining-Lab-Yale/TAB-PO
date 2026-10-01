#!/usr/bin/env python3
"""Reject context budgets that would truncate instructions or structured answers."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tabpo.data import read_arrow
from tabpo.prompting import completion_encoding, messages


def check(model, max_length, max_gen_toks=4096):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
    maxima = {}
    for split in ("train", "validation", "test"):
        prompts, fulls = [], []
        for i, row in enumerate(read_arrow(Path(__file__).resolve().parents[1] / f"data/scierc/{split}.arrow")):
            prefix = tokenizer.apply_chat_template(messages(row["query"]), tokenize=False, add_generation_prompt=True)
            prompt_length = len(tokenizer.encode(prefix, add_special_tokens=False))
            if prompt_length + max_gen_toks > max_length:
                raise ValueError(f"{split} row {i}: {prompt_length} prompt tokens + {max_gen_toks} generation reserve "
                                 f"exceed {max_length}. Increase --max-length to prevent lm-eval context truncation.")
            prompts.append(prompt_length)
            if split != "test":
                answer = json.dumps(json.loads(row["answer"]), ensure_ascii=False, separators=(",", ":"))
                _, enc, _ = completion_encoding(tokenizer, row["query"], answer, max_length)
                fulls.append(len(enc["input_ids"]))
        maxima[split] = {"max_prompt_tokens": max(prompts), "max_training_tokens": max(fulls) if fulls else None}
    return {"model": model, "max_length": max_length, "max_gen_toks": max_gen_toks, "splits": maxima}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--max-length", type=int, default=16384)
    p.add_argument("--max-gen-toks", type=int, default=4096)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    result = check(args.model, args.max_length, args.max_gen_toks)
    output = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    print(output)
