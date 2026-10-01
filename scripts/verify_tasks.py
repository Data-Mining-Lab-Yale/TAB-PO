#!/usr/bin/env python3
"""Load the real lm-eval YAML tasks and check gold-identity aggregation on CPU."""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "evaluation")]


def verify():
    from lm_eval.tasks import TaskManager, get_task_dict
    os.chdir(ROOT)
    names = ["scierc_validation", "scierc_test"]
    tasks = get_task_dict(names, task_manager=TaskManager(include_path=str(ROOT / "evaluation"), include_defaults=False))
    report = {}
    for name in names:
        task = tasks[name]
        # Both tasks use the harness evaluation split (test_split); the
        # validation task maps it to the separate validation Arrow file.
        documents = list(task.test_docs())
        expected_size = 100 if "test" in name else 53
        if len(documents) != expected_size:
            raise ValueError(f"{name}: expected {expected_size} documents, got {len(documents)}")
        values = [task.process_results(doc, [doc["answer"]]) for doc in documents]
        scores = {key: task.aggregation()[key]([row[key] for row in values]) for key in values[0]}
        if any(abs(value - 1.0) > 1e-8 for value in scores.values()):
            raise ValueError(f"{name}: gold identity failed: {scores}")
        report[name] = {"documents": len(documents), "gold_identity_metrics": scores}
    return report


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
