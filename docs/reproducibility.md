# Running a SciERC experiment

## Environment

Use Linux, Python 3.11, CUDA-capable NVIDIA GPUs, and sufficient storage for base weights, two merged models, adapters, and reference caches. `requirements-cpu.txt` supports data, metrics, and preference construction. `requirements-training.txt` pins the direct training dependencies. `requirements-inference.txt` adds vLLM 0.11.2 and lm-evaluation-harness 0.4.8; `requirements.txt` includes both environments. We use the same harness version for task checks and inference.

Our dependencies pin direct package versions. The [validation record](validation.md) identifies tested environments; choose compatible drivers and run `pip check` after installation. vLLM 0.11.2's [CUDA requirements](https://github.com/vllm-project/vllm/blob/v0.11.2/requirements/cuda.txt) specify Torch 2.9.0, and its [common requirements](https://github.com/vllm-project/vllm/blob/v0.11.2/requirements/common.txt) accept Transformers ≥4.56 and <5. The included pins follow those constraints. Driver/CUDA compatibility remains host-specific. Do not install optional DeepSpeed/bitsandbytes unless selecting those code paths.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
python scripts/audit_data.py
python scripts/verify_release.py
python scripts/verify_tasks.py
python -m unittest discover -s tests -v
python scripts/run_pipeline.py --dry-run
```

For an offline CPU training check, install the training requirements with a compatible CPU Torch build and run `python scripts/smoke_test_training.py`. That check constructs a tiny random model and tokenizer locally; it does not download model weights. See our [validation record](validation.md).

## One complete run

```bash
python scripts/run_pipeline.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --run-dir outputs/qwen2_5_1_5b_seed42 \
  --num-gpus 1 --seed 42 --max-length 16384
```

The same options work through `bash scripts/run_pipeline.sh`. The wrapper also preserves `MODEL_NAME`, `RUN_DIR`, and `NUM_GPUS` environment defaults. Model authorization belongs in `HF_TOKEN`; gated model access must be arranged separately.

## Stages and outputs

| Stage | Input and role | Output under the run directory |
| --- | --- | --- |
| `preflight` | Audit canonical data; check selected tokenizer, prompt lengths, answer masks, context reserve. | `token_lengths.json` |
| `sft` | Train on 297 documents with completion-only loss and a separate validation dataset. | `sft/lora_adapter/` |
| `merge-sft` | Merge nonzero SFT LoRA weights into base. | `sft_merged/` |
| `validation` | Greedy SFT inference on exactly 53 validation documents. | `validation_inference/` with metrics and sample JSONL |
| `confusions` | Verify validation log membership/completeness/gold; extract confusion counts. | `validation_confusions/` CSVs and provenance summary |
| `preferences-train` | Construct pairs from training gold plus validation-derived confusion statistics. | `preferences_train/` and `build_summary.json` |
| `preferences-validation` | Construct pairs from separate validation gold. | `preferences_validation/` and summary |
| `tabpo` | Load merged SFT, cache frozen-reference scores, optimize preferences. | `tabpo/` adapter and training artifacts |
| `merge-tabpo` | Merge nonzero TAB-PO adapter into the SFT base. | `tabpo_merged/` |
| `test-sft` | SFT test inference using Appendix F metrics. | `sft_test_inference/` |
| `test-tabpo` | TAB-PO test inference using Appendix F metrics. | `tabpo_test_inference/` |

**Post-SFT confusion inference uses validation, not test.** The YAML field `test_split: validation` selects the Arrow validation partition through the harness evaluation API. Its file is `data/scierc/validation.arrow`. Test stages occur after both training stages. Bootstrap iterations are disabled because custom intermediate metric values are count tuples; the release does not report bootstrap intervals. YAML uses `max_gen_toks=4096`, supported by the harness adapter, rather than forwarding a duplicate `max_tokens` parameter to vLLM.

The orchestrator records commands, configuration, installed versions, split audit, `pip_freeze.txt`, and completed stages. It stops on the first failed subprocess. A validation directory must have exactly one sample log; archive stale logs before resuming inference rather than letting an arbitrary file be selected.

## Resume and ablate

```bash
python scripts/run_pipeline.py --run-dir outputs/qwen2_5_1_5b_seed42 --stop-after validation
python scripts/run_pipeline.py --run-dir outputs/qwen2_5_1_5b_seed42 --start-at confusions
python scripts/run_pipeline.py --run-dir outputs/qwen2_5_1_5b_dpo_seed42 --barrier-weight 0
```

Resume with the same model, seed, context, GPU count, sampling settings, and optimization parameters. Changing these requires a new run directory. Resume starts at the named stage; it does not guess that earlier stages succeeded. Existing generated preference and merged-model directories are intentionally not overwritten. `--dry-run` writes nothing and does not need model access or a GPU.

The DPO ablation changes only the barrier weight. Run independent optimization seeds in distinct directories and report individual scores, means, and uncertainty. Keep the data partition fixed when comparing optimization seeds.

## Pipeline defaults

| Setting | Pipeline default |
| --- | --- |
| SFT epochs / learning rate / gradient accumulation | 3 / 2e-4 / 5 |
| SFT LoRA rank / alpha / dropout | 16 / 32 / 0.05 |
| TAB-PO epochs / learning rate / gradient accumulation | 1 / 2e-4 / 1 |
| TAB-PO beta / tau / lambda | 0.5 / 0.87 / 0.5 |
| Gradient clipping | 1.0 |
| Precision / per-device batch size | bfloat16 / 1 |
| Maximum context / generation reserve | 16,384 / 4,096 tokens |
| Synthetic pairs requested per source document | 1 |
| Auxiliary token weighting / length normalization / class balancing | off |

Our pipeline uses a TAB-PO learning rate of 2e-4 and gradient clipping at 1.0. Inspect the dry-run for all parameters; direct script defaults may differ from orchestrator defaults. A context preflight fails instead of silently truncating the task.

## Automatic preference construction

We construct all preferences automatically from gold annotations and validation-derived confusion statistics: **100% confusion-aware synthetic pairs, with no expert disambiguation**. Training and validation preference pools use separate source splits. Build summaries record `synthetic_fraction: 1.0`, `expert_disambiguation: false`, family counts, and provenance. TAB-PO's default evaluation strategy is `no`; the separate preference validation pool is available when evaluation is enabled.

## Resource limits

GPU memory requirements depend on the model, context, precision, and host. Data-parallel Torch training duplicates model state unless a user explicitly configures sharding; `--num-gpus` also selects vLLM tensor parallelism for inference. The optional training DeepSpeed configuration is not a promise of 70B feasibility. Reference-cache preprocessing loads the full SFT base per worker before distributed optimization, so even a sharded training run can fail during caching. Choose a model and host that can hold that preprocessing workload, or implement and validate sharded cache construction before scaling.

Our [validation record](validation.md) separates completed CPU checks from GPU runtime verification.
