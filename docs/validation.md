# Validation record

We validate data integrity, preference construction, the TAB-PO objective, and Appendix F evaluation independently of model downloads. This record describes our local checks on October 1, 2026.

## Data and evaluation checks

| Check | Result |
| --- | --- |
| SciERC counts and schema | 297 training, 53 validation, 100 test documents; three string columns: `query`, `answer`, `context`. |
| Split membership | Seed-42 iterative stratification reproduces membership; no query overlap. |
| Metadata cleanup | All queries, contexts, answers, and row order match their pre-cleanup values. |
| Post-SFT inference | The pipeline selects `scierc_validation`, backed by `data/scierc/validation.arrow`. |
| Harness task loading and identity scoring | lm-eval 0.4.8 loads both tasks; gold-as-prediction scores are 1.0 for all four F1 metrics and JSON parse rate on all 53 validation and 100 test documents. |
| Relation argument order | Regression checks reject reversed arguments for every relation type in both scoring and confusion alignment. |

Preference checks cover distinct schema-valid negatives, validation-confusion provenance, separate source pools, and rejection of test input even when renamed. Construction summaries identify all pairs as synthetic and record that no expert disambiguation is used.

## Regression and training checks

The test suite covers split conservation, the three-column schema, provenance, strict relation order, sentence-aware coreference, micro aggregation, preference construction, prompt masks, and objective gradients. The CPU smoke test trains a tiny local model for two SFT and two TAB-PO steps and checks reference scores, finite losses, nonzero adapters, and save/reload behavior.

| Check | Result |
| --- | --- |
| Regression suite | **33 tests passed**, with no skips, including preference command-line integration. |
| CPU training smoke | **Passed**: two SFT steps, two TAB-PO steps, finite loss, consistent formatting, reference-cache verification, and adapter reload. |
| Release verification | **Passed**: local Markdown links, Python syntax, PDF-only paper directory, and credential-pattern scan. |
| Pipeline dry-run | **Passed**: validation-only post-SFT inference, separate preference pools, and strict test evaluation. |

## Run the checks

From the repository root:

```bash
python scripts/audit_data.py
python scripts/verify_release.py
python -m unittest discover -s tests -v
python scripts/run_pipeline.py --dry-run
```

With training dependencies and a CPU-compatible Torch build:

```bash
python scripts/smoke_test_training.py
```

With lm-evaluation-harness 0.4.8:

```bash
python scripts/verify_tasks.py
```

## Environment and coverage

Our local checks use Windows and Python 3.14.5. Training and inference installation instructions target Linux/Python 3.11. CPU checks and tiny-model training do not exercise CUDA, vLLM generation, distributed training, or large-model memory requirements. Direct dependency pins are specified in the requirements files; the orchestrator records installed versions and split hashes for each experiment.
