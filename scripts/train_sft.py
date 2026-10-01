#!/usr/bin/env python3
"""
Distributed completion-only SciERC SFT with PEFT (QLoRA/LoRA) on Arrow data:
  - input uses the model chat template and compact canonical JSON answer
  - loss only on answer tokens (prompt masked to -100)

Run:
  torchrun --nproc_per_node=1 scripts/train_sft.py ...
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Dict, Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tabpo.prompting import completion_encoding

# Avoid occasional tokenizer thread deadlocks on shared HPC
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from datasets import Dataset, load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    BitsAndBytesConfig,
    set_seed,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers.integrations import HfDeepSpeedConfig


# ---------------------------
# Tokenization / Label masking
# ---------------------------
def build_features(example: Dict[str, Any], tokenizer, max_length: int, system_prompt: str = "") -> Dict[str, Any]:
    answer = json.dumps(json.loads(example["answer"]), ensure_ascii=False, separators=(",", ":"))
    _, enc, labels = completion_encoding(tokenizer, example["query"], answer, max_length, system_prompt)
    return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": labels}

def load_dataset_any(path: str):
    p = Path(path)
    if p.is_dir():
        return load_from_disk(str(p))
    if p.is_file() and p.suffix == ".arrow":
        return Dataset.from_file(str(p))
    raise ValueError(f"Unsupported dataset path: {path}. Expected save_to_disk directory or .arrow file.")

def get_deepspeed_zero_stage(deepspeed_config_path: Optional[str]) -> int:
    """
    Returns DeepSpeed ZeRO stage from a JSON config.
    Returns 0 if no DeepSpeed config is provided.
    """
    if not deepspeed_config_path:
        return 0

    with open(deepspeed_config_path, "r") as f:
        ds_cfg = json.load(f)

    zero_cfg = ds_cfg.get("zero_optimization", {})
    if isinstance(zero_cfg, dict):
        return int(zero_cfg.get("stage", 0) or 0)

    return 0

class SimpleCollator(DataCollatorWithPadding):
    """Pads input_ids/attention_mask/labels to longest in batch. Pads labels with -100."""
    def __call__(self, features):
        batch = super().__call__([{k: v for k, v in f.items() if k != "labels"} for f in features])

        max_len = batch["input_ids"].shape[1]
        padded_labels = []
        for f in features:
            lab = f["labels"]
            pad_len = max_len - len(lab)
            padded_labels.append(lab + [-100] * pad_len)

        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./sft_peft_out")
    parser.add_argument("--max_length", type=int, default=2048)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--deepspeed_config", type=str, default=None)

    parser.add_argument("--use_qlora", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--valid_dataset_path", type=str, default=None)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )

    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--system_prompt", default="")
    args = parser.parse_args()
    set_seed(args.seed)
    if args.do_eval and not args.valid_dataset_path:
        parser.error("--do_eval requires a held-out --valid_dataset_path")

    # ---------------------------
    # DeepSpeed setup
    # ---------------------------
    ds_zero_stage = get_deepspeed_zero_stage(args.deepspeed_config)
    ds_is_zero3 = ds_zero_stage == 3

    # Important:
    # For DeepSpeed ZeRO-3, this object must be created BEFORE from_pretrained().
    # Keep this variable alive until after model loading.
    dschf = None
    if args.deepspeed_config and ds_is_zero3:
        if args.use_qlora:
            raise ValueError(
                "Do not combine --use_qlora with DeepSpeed ZeRO-3 in this SFT script. "
                "Use ZeRO-2 for QLoRA, or disable --use_qlora when using ZeRO-3."
            )
        dschf = HfDeepSpeedConfig(args.deepspeed_config)

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    if "meta-llama" in args.model_name.lower() and not hf_token:
        raise RuntimeError(
            "HF_TOKEN or HUGGING_FACE_HUB_TOKEN is required to access gated Llama models. "
            "Export HF_TOKEN before running this script."
        )

    target_modules = [
        x.strip()
        for x in args.lora_target_modules.split(",")
        if x.strip()
    ]

    # torchrun / DeepSpeed launcher sets these env vars
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if local_rank >= 0:
        torch.cuda.set_device(local_rank)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load dataset ----
    train_ds = load_dataset_any(args.dataset_path)
    eval_ds = load_dataset_any(args.valid_dataset_path) if args.valid_dataset_path else None
    if eval_ds is not None:
        if set(train_ds["query"]) & set(eval_ds["query"]):
            raise ValueError("SFT training and validation documents overlap")
    for path in (args.dataset_path, args.valid_dataset_path):
        if path and Path(path).stem == "test":
            raise ValueError("Held-out test data cannot be used in SFT")

    # ---- Tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=True,
        token=hf_token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ---- Model ----
    torch_dtype = torch.bfloat16 if args.bf16 else torch.float16

    quant_config: Optional[BitsAndBytesConfig] = None
    if args.use_qlora:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
        )

    # IMPORTANT:
    # - QLoRA + regular DDP / ZeRO-2: load each rank on its local GPU.
    # - DeepSpeed ZeRO-3 full/bf16 LoRA: do NOT use device_map.
    # - Never use device_map="auto" for distributed training.
    ddp_device_map = (
        {"": local_rank}
        if (args.use_qlora and local_rank >= 0 and not ds_is_zero3)
        else None
    )

    model_kwargs = dict(
        torch_dtype=torch_dtype,
        quantization_config=quant_config,
        token=hf_token,
    )

    if ddp_device_map is not None:
        model_kwargs["device_map"] = ddp_device_map

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        **model_kwargs,
    )

    # Stability for training
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    if args.use_qlora:
        model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    model = get_peft_model(model, lora_cfg)

    if rank == 0:
        model.print_trainable_parameters()

    # ---- Tokenize (simple + robust) ----
    def _map_fn(ex):
        return build_features(ex, tokenizer=tokenizer, max_length=args.max_length, system_prompt=args.system_prompt)

    train_tok = train_ds.map(_map_fn, remove_columns=train_ds.column_names, desc="Tokenizing train")
    if eval_ds is not None and args.do_eval:
        eval_tok = eval_ds.map(_map_fn, remove_columns=eval_ds.column_names, desc="Tokenizing eval")
        if len(eval_tok) == 0:
            eval_tok = None  # avoid DDP eval deadlocks
    else:
        eval_tok = None

    collator = SimpleCollator(tokenizer=tokenizer, padding=True)

    train_args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=args.logging_steps,
        logging_first_step=True,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,

        optim=("paged_adamw_8bit" if (args.use_qlora and not args.deepspeed_config) else "adamw_torch"),

        fp16=(not args.bf16),
        bf16=args.bf16,
        gradient_checkpointing=True,

        # DDP / DeepSpeed hygiene
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,

        # Safer defaults on HPC
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        dataloader_drop_last=False,

        # Optional evaluation
        eval_strategy=("steps" if (eval_tok is not None) else "no"),
        eval_steps=args.save_steps if (eval_tok is not None) else None,

        # True DeepSpeed hook
        deepspeed=args.deepspeed_config,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    # ---- Save adapter (rank0 only) ----
    adapter_dir = out_dir / "lora_adapter"
    trainer.save_model(str(adapter_dir))

    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(str(adapter_dir))
        print(f"\nSaved LoRA adapter to: {adapter_dir}", flush=True)


if __name__ == "__main__":
    main()
