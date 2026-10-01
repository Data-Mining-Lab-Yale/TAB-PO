# Contributing

Describe the problem, the intended behavior, and any effect on the scientific protocol. Include the model/tokenizer versions, seed, split hashes, and a minimal failing example when relevant. Do not include access tokens or unpublished private data.

Run from the repository root:

```bash
python -m pip install -r requirements-cpu.txt
python scripts/audit_data.py
python scripts/verify_release.py
python -m unittest discover -s tests -v
```

For objective, tokenizer, adapter, or training-loop changes, install the training dependencies and also run `python scripts/smoke_test_training.py`. Tests must exercise a meaningful failure mode or scientific invariant, rather than mirroring the implementation.

Preserve held-out test documents and annotations. Document changes to split membership, matching rules, objective normalization, prompts, or sampling. Evaluation uses strict relation argument order from Appendix F. Include validation results when proposing a change. See [licensing status](LICENSING.md) for contribution and reuse terms.
