#!/usr/bin/env python3
"""Inspect or run the validation-only SciERC pipeline, with resumable stages."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
STAGES = ("preflight", "sft", "merge-sft", "validation", "confusions", "preferences-train",
          "preferences-validation", "tabpo", "merge-tabpo", "test-sft", "test-tabpo")


def sample_path(directory: Path) -> str:
    paths = sorted(directory.rglob("*samples*.jsonl"))
    if len(paths) != 1:
        raise ValueError(f"Expected one validation sample log under {directory}, found {len(paths)}")
    return str(paths[0])


def command_plan(args):
    run = Path(args.run_dir)
    python = sys.executable
    train, valid = "data/scierc/train.arrow", "data/scierc/validation.arrow"
    sft, merged = run / "sft", run / "sft_merged"
    conf = run / "validation_confusions"
    pt, pv = run / "preferences_train", run / "preferences_validation"
    merged_tabpo = run / "tabpo_merged"
    launch = [python, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={args.num_gpus}"]
    def inference(model, tasks, output):
        return [python, "-m", "lm_eval", "--model", "vllm", "--model_args",
                f"pretrained={model},tensor_parallel_size={args.num_gpus},max_model_len={args.max_length},gpu_memory_utilization={args.gpu_memory_utilization},seed={args.seed}",
                "--tasks", tasks, "--include_path", "evaluation", "--batch_size", "1", "--num_fewshot", "0",
                "--apply_chat_template", "--log_samples", "--bootstrap_iters", "0",
                "--seed", f"{args.seed},{args.seed},{args.seed},{args.seed}", "--output_path", str(output)]
    def preferences(source, output, seed):
        return [python, "scripts/build_preference_dataset.py", "--input_arrow", source,
                "--output_dir", str(output), "--entity_confusion_file", str(conf / "entity_type_confusion.csv"),
                "--relation_confusion_file", str(conf / "relation_type_confusion.csv"),
                "--negatives_per_sample", str(args.negatives_per_sample), "--seed", str(seed)]
    plans = [
        [python, "scripts/check_token_lengths.py", "--model", args.model, "--max-length", str(args.max_length),
         "--output", str(run / "token_lengths.json")],
        launch + ["scripts/train_sft.py", "--dataset_path", train, "--valid_dataset_path", valid,
                  "--model_name", args.model, "--output_dir", str(sft), "--max_length", str(args.max_length),
                  "--batch_size", "1", "--grad_accum", "5", "--epochs", "3", "--lr", "2e-4",
                  "--lora_r", "16", "--lora_alpha", "32", "--lora_dropout", "0.05", "--do_eval", "--bf16", "--seed", str(args.seed)],
        [python, "scripts/merge_lora_adapter.py", "--base", args.model, "--adapter", str(sft / "lora_adapter"), "--out", str(merged), "--dtype", "bf16"],
        inference(merged, "scierc_validation", run / "validation_inference"),
        [python, "scripts/build_validation_confusions.py", "--samples_jsonl", "{validation_samples}", "--out_dir", str(conf),
         "--jaccard_threshold", "0.6"],
        preferences(train, pt, args.seed), preferences(valid, pv, args.seed + 1),
        launch + ["scripts/train_tabpo.py", "--model_name", str(merged), "--train_data_path", str(pt),
                  "--valid_data_path", str(pv), "--output_dir", str(run / "tabpo"), "--num_gpus", str(args.num_gpus),
                  "--max_length", str(args.max_length), "--max_prompt_length", str(args.max_length - 4096),
                  "--beta", "0.5", "--policy_prob_threshold", "0.87", "--sft_barrier_weight", str(args.barrier_weight),
                  "--num_train_epochs", "1", "--per_device_train_batch_size", "1", "--gradient_accumulation_steps", "1",
                  "--learning_rate", str(args.tabpo_lr), "--max_grad_norm", "1.0", "--bf16", "true", "--eval_strategy", "no", "--seed", str(args.seed)],
        [python, "scripts/merge_lora_adapter.py", "--base", str(merged), "--adapter", str(run / "tabpo"), "--out", str(merged_tabpo), "--dtype", "bf16"],
        inference(merged, "scierc_test", run / "sft_test_inference"),
        inference(merged_tabpo, "scierc_test", run / "tabpo_test_inference"),
    ]
    return list(zip(STAGES, plans))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct"))
    p.add_argument("--run-dir", default=os.environ.get("RUN_DIR", "outputs/qwen2_5_1_5b_seed42"))
    p.add_argument("--num-gpus", type=int, default=int(os.environ.get("NUM_GPUS", "1")))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-length", type=int, default=16384)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--tabpo-lr", type=float, default=2e-4)
    p.add_argument("--barrier-weight", type=float, default=0.5, help="Set 0 for plain DPO on the same preferences")
    p.add_argument("--negatives-per-sample", type=int, default=1)
    p.add_argument("--start-at", choices=STAGES, default=STAGES[0])
    p.add_argument("--stop-after", choices=STAGES, default=STAGES[-1])
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if args.num_gpus < 1 or args.max_length <= 4096 or args.tabpo_lr <= 0 or args.barrier_weight < 0:
        p.error("Require num-gpus >= 1, max-length > 4096, tabpo-lr > 0, barrier-weight >= 0")
    if args.negatives_per_sample < 1 or not 0 < args.gpu_memory_utilization < 1:
        p.error("Require negatives-per-sample >= 1 and gpu-memory-utilization in (0,1)")
    start, stop = STAGES.index(args.start_at), STAGES.index(args.stop_after)
    if start > stop:
        p.error("start-at must not follow stop-after")
    plans = command_plan(args)[start:stop + 1]
    for name, command in plans:
        print(f"[{name}] {shlex.join(command)}", flush=True)
    if args.dry_run:
        return
    os.chdir(ROOT)
    from verify_splits import audit
    data_audit = audit(ROOT / "data/scierc")
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() < args.num_gpus:
        raise RuntimeError("The full pipeline requires the requested number of visible CUDA GPUs")
    run = Path(args.run_dir)
    run.mkdir(parents=True, exist_ok=True)
    manifest_path = run / "run_manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))["configuration"]
        if start == 0:
            raise ValueError("Run already exists. Use --start-at to resume or choose a new --run-dir")
        changes = [key for key in ("model", "seed", "max_length", "tabpo_lr", "barrier_weight", "negatives_per_sample", "num_gpus", "gpu_memory_utilization") if old[key] != vars(args)[key]]
        if changes:
            raise ValueError(f"Resume configuration changed: {changes}; use a new run directory")
    packages = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "configuration": vars(args),
                "commands": plans, "packages": packages, "data_audit": data_audit,
                "preference_construction": "100% confusion-aware synthetic", "expert_disambiguation": False}
    (run / ("run_manifest.json" if not manifest_path.exists() else "resume_manifest.json")).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    with (run / "pip_freeze.txt").open("w", encoding="utf-8") as stream:
        subprocess.run([sys.executable, "-m", "pip", "freeze"], stdout=stream, check=True)
    for name, command in plans:
        if name == "confusions":
            command = [sample_path(run / "validation_inference") if x == "{validation_samples}" else x for x in command]
        print(f"Running {name}", flush=True)
        subprocess.run(command, check=True, cwd=ROOT)
        with (run / "completed_stages.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"stage": name, "command": command, "finished_utc": datetime.now(timezone.utc).isoformat()}) + "\n")


if __name__ == "__main__":
    main()
