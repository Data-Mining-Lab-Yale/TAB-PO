# TAB-PO on SciERC

Our research implementation of **TAB-PO: Preference Optimization with a Token-Level Adaptive Barrier for Token-Critical Structured Generation**.

[Paper PDF](paper/tab-po.pdf) · [Reproduction guide](docs/reproducibility.md) · [Data card](data/README.md) · [Metrics and method](docs/methods.md) · [Validation record](docs/validation.md) · [Citation](CITATION.cff)

TAB-PO adds a confidence-gated preferred-token likelihood penalty to a reference-adjusted preference objective. This release runs the SciERC sequence: **SFT → validation inference → confusion-aware preferences → TAB-PO → held-out test inference**. Our frozen reference is the merged SFT model. Model outputs contain entities, typed relations, and coreference clusters as JSON.

We provide a document-level, iteratively stratified **297/53/100 train/validation/test split**, CPU data and metric checks, an offline training smoke test, and a resumable GPU pipeline. The paper directory contains the manuscript PDF and its description. Manuscript sources, original archives, and the raw notebook are excluded.

## Start on a CPU

Run commands from this repository's root. Use Python 3.11 for the documented training environment.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-cpu.txt
python scripts/audit_data.py
python scripts/verify_release.py
python -m unittest discover -s tests -v
python scripts/run_pipeline.py --dry-run
```

On Windows, activate with `.venv\Scripts\Activate.ps1`. CPU checks work on Windows; the vLLM training/inference pipeline targets Linux with CUDA. Torch-dependent objective tests skip when PyTorch is absent; see the [validation record](docs/validation.md) for the training checks.

## Run a GPU experiment

Use a Linux CUDA environment with a compatible NVIDIA driver and PyTorch installation. Run `python -m pip check` after installation; our [validation record](docs/validation.md) identifies tested environments.

```bash
python -m pip install -r requirements.txt
python scripts/run_pipeline.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --run-dir outputs/qwen2_5_1_5b_seed42 \
  --num-gpus 1 --seed 42
```

Set `HF_TOKEN` in your environment when a model requires authorization. No access token belongs in code or logs. Use `--dry-run` to inspect commands; `--stop-after validation` to stop after SFT validation inference; `--start-at confusions` to resume a completed SFT run. Existing nonempty preference and merged-model output directories are protected from overwrite. The [guide](docs/reproducibility.md) explains stage outputs, prerequisites, resource requirements and the DPO ablation.

## Confusion-aware preference data

**100% of preference pairs in this repository are constructed automatically using validation-derived confusion statistics; no expert disambiguation or adjudicated component is included.** Gold annotations supply chosen outputs. The generator builds schema-valid rejected outputs through entity/relation label confusions, missing records, extra relations, and relation relinking. Training and validation preferences use separate source documents. Build summaries record provenance, realized negative families, and the synthetic fraction.

## Evaluation

The `scierc_validation` and `scierc_test` tasks use the strict argument order defined in Appendix F for every relation type. Coreference mention matches require sentence-ID agreement. All F1 values aggregate TP/FP/FN across documents.

We use the paper's names: **Entity F1, Relation F1, Span F1, and Coreference F1**. Entity F1 measures document-level entity-type presence, Span F1 uses containment/Jaccard matching, and Coreference F1 evaluates mention pairs. See [metric definitions](docs/methods.md). Output keys are `entity_f1`, `relation_f1`, `span_f1`, and `coreference_f1`.

Existing `lm-eval --log_samples` predictions can be scored without model weights:

```bash
python scripts/evaluate_samples.py --samples path/to/samples.jsonl --split test
```

## Results reported in the manuscript

The table reports TAB-PO scores from our manuscript, averaged over three runs. Scores are percentages; evaluators emit fractions in [0, 1].

| Model | Entity F1 | Relation F1 | Span F1 | Coreference F1 |
| --- | ---: | ---: | ---: | ---: |
| Llama 3.1 8B | 91.50 | 56.63 | 86.31 | 46.24 |
| Llama 3.2 3B | 91.04 | 46.67 | 84.98 | 35.11 |
| Llama 3.3 70B | 92.21 | 58.68 | 88.11 | 60.80 |
| Qwen 2.5 1.5B | 90.85 | 38.31 | 83.07 | 24.76 |
| Qwen 2.5 7B | 91.50 | 48.88 | 83.38 | 41.19 |
| Qwen 2.5 14B | 91.64 | 55.14 | 85.20 | 52.95 |

## Repository map

| Path | Purpose |
| --- | --- |
| [paper/](paper/README.md) | Manuscript PDF. |
| [data/](data/README.md) | Arrow splits, hashes, label prevalence, source provenance. |
| [tabpo/objective.py](tabpo/objective.py) | Small, testable TAB-PO loss implementation. |
| [tabpo/prompting.py](tabpo/prompting.py) | Shared chat serialization and completion mask. |
| [scripts/](scripts/) | Data preparation, SFT, preference construction, TAB-PO, merging, evaluation, orchestration. |
| [evaluation/](evaluation/) | Independent validation/test tasks and scalar metric adapters. |
| [tests/](tests/) | Leakage, schema, objective, and metric regression checks. |
| [docs/](docs/) | Methods, reproduction steps, and validation record. |

## Scope, citation, and licensing

Our paper also studies PV-Miner; this repository covers SciERC. We provide the prepared SciERC records and split provenance. The original benchmark is [Luan et al., EMNLP 2018](https://aclanthology.org/D18-1360/). Cite it and TAB-PO using [CITATION.cff](CITATION.cff).

See [licensing status](LICENSING.md) for code and artifact terms.
