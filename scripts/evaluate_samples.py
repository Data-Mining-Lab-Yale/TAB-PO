#!/usr/bin/env python3
"""Score a complete held-out lm-eval sample log without loading model weights."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "evaluation")]
from tabpo.data import read_arrow, schema_errors, sha256
from verify_splits import audit
from build_validation_confusions import get_pred_response
import harness_metrics as metrics


def evaluate(path, split, data_dir):
    audit(data_dir)
    arrow = data_dir / f"{split}.arrow"
    gold = {row["query"]: row["answer"] for row in read_arrow(arrow)}
    seen, scores = set(), []
    schema_valid = 0
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        doc = record.get("doc", {})
        query = doc.get("query")
        if query not in gold or query in seen:
            raise ValueError(f"Line {number}: duplicated document or document outside {split}")
        if json.loads(doc.get("answer", "null")) != json.loads(gold[query]):
            raise ValueError(f"Line {number}: gold annotations differ from the audited split")
        prediction = get_pred_response(record)
        scores.append(metrics.counts(gold[query], prediction))
        try:
            schema_valid += not schema_errors(json.loads(metrics.u._strip_code_fences(prediction)))
        except (ValueError, TypeError):
            pass
        seen.add(query)
    if len(seen) != len(gold):
        raise ValueError(f"Need all {len(gold)} {split} predictions; got {len(seen)}")
    report = {"split": split, "documents": len(seen), "samples_sha256": sha256(path.read_bytes()),
              "data_sha256": sha256(arrow.read_bytes()), "schema_valid_rate": schema_valid / len(seen),
              "protocol": "paper_ordered", "metrics": {
                  key: metrics.micro_f1([doc[key] for doc in scores]) if key.endswith("f1")
                  else metrics.mean([doc[key] for doc in scores]) for key in scores[0]}}
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--samples", type=Path, required=True)
    p.add_argument("--split", choices=("validation", "test"), required=True)
    p.add_argument("--data-dir", type=Path, default=ROOT / "data/scierc")
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    report = json.dumps(evaluate(args.samples, args.split, args.data_dir), indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
