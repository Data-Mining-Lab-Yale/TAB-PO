#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def max_lora_abs(peft_model, which="lora_B"):
    vals = []
    names = []

    for name, p in peft_model.named_parameters():
        if which in name:
            try:
                v = p.detach().float().abs().max().item()
                vals.append(v)
                names.append(name)
            except Exception:
                pass

    if not vals:
        return 0.0, []

    return max(vals), names[:20]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Base/SFT model name or path")
    ap.add_argument("--adapter", required=True, help="LoRA adapter directory")
    ap.add_argument("--out", required=True, help="Output dir for merged model")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    args = ap.parse_args()
    if Path(args.out).exists() and any(Path(args.out).iterdir()):
        ap.error("Output directory is nonempty; choose a new path to preserve existing models")

    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.dtype]

    print("=" * 80)
    print("MERGE SCIERC LORA ADAPTER")
    print("=" * 80)
    print(f"Base model/path: {args.base}")
    print(f"Adapter path:    {args.adapter}")
    print(f"Output path:     {args.out}")
    print(f"dtype:           {args.dtype}")
    print("=" * 80)

    if not os.path.isdir(args.adapter):
        raise FileNotFoundError(f"Adapter directory not found: {args.adapter}")

    adapter_config = os.path.join(args.adapter, "adapter_config.json")
    adapter_safetensors = os.path.join(args.adapter, "adapter_model.safetensors")
    adapter_bin = os.path.join(args.adapter, "adapter_model.bin")

    if not os.path.exists(adapter_config):
        raise FileNotFoundError(f"Missing adapter_config.json in: {args.adapter}")

    if not os.path.exists(adapter_safetensors) and not os.path.exists(adapter_bin):
        raise FileNotFoundError(
            f"No adapter_model.safetensors or adapter_model.bin found in: {args.adapter}"
        )

    tokenizer_source = args.adapter if (Path(args.adapter) / "tokenizer_config.json").exists() else args.base
    tok = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=dtype,
        device_map="auto",
    )

    print("\nLoading PEFT adapter...")
    peft_model = PeftModel.from_pretrained(base_model, args.adapter)

    lora_a_max, lora_a_names = max_lora_abs(peft_model, "lora_A")
    lora_b_max, lora_b_names = max_lora_abs(peft_model, "lora_B")

    print("\n" + "=" * 80)
    print("PEFT ADAPTER LOAD CHECK BEFORE MERGE")
    print("=" * 80)
    print(f"max_abs_lora_A: {lora_a_max:.8e}")
    print(f"max_abs_lora_B: {lora_b_max:.8e}")
    print("\nExample loaded lora_A parameter names:")
    for n in lora_a_names:
        print("  ", n)
    print("\nExample loaded lora_B parameter names:")
    for n in lora_b_names:
        print("  ", n)
    print("=" * 80)

    if lora_a_max == 0.0:
        raise RuntimeError(
            "Loaded adapter has all-zero LoRA A tensors. "
            "This means PEFT did not load the adapter correctly."
        )

    if lora_b_max == 0.0:
        raise RuntimeError(
            "Adapter merge aborted: loaded adapter has all-zero LoRA B tensors. "
            "Merging this adapter would produce a model identical or nearly "
            "identical to the base/SFT model. Fix the DPO adapter saving/loading "
            "path first."
        )

    print("\nMerging adapter into base model...")
    merged_model = peft_model.merge_and_unload()

    print(f"\nSaving merged model to: {args.out}")
    merged_model.save_pretrained(args.out, safe_serialization=True)
    tok.save_pretrained(args.out)

    print("\n" + "=" * 80)
    print(f"Merged model saved successfully to: {args.out}")
    print("=" * 80)


if __name__ == "__main__":
    main()
