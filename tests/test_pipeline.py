import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "evaluation")]
from tabpo.data import read_arrow, schema_errors
from run_pipeline import command_plan
from evaluate_samples import evaluate
import harness_metrics


class PipelineTests(unittest.TestCase):
    def test_plan_keeps_test_in_final_stages(self):
        args = SimpleNamespace(run_dir="outputs/example", model="test/model", num_gpus=1, seed=42,
                               max_length=16384, gpu_memory_utilization=.85, negatives_per_sample=1,
                               tabpo_lr=2e-4, barrier_weight=.5)
        plan = dict(command_plan(args))
        self.assertIn("data/scierc/validation.arrow", plan["sft"])
        self.assertIn("scierc_validation", plan["validation"])
        self.assertIn("data/scierc/train.arrow", plan["preferences-train"])
        self.assertIn("data/scierc/validation.arrow", plan["preferences-validation"])
        for stage, command in plan.items():
            if not stage.startswith("test-"):
                self.assertFalse(any("scierc_test" in part or "test.arrow" in part for part in command))
        self.assertIn("scierc_test", plan["test-tabpo"])
        self.assertEqual(plan["test-tabpo"][plan["test-tabpo"].index("--tasks") + 1], "scierc_test")

    def test_malformed_field_types_do_not_crash_evaluator(self):
        pred = {"entities": [{"entity_id": [], "text": "word", "type": [], "sentence_id": 0}],
                "relations": [{"type": "Used-for", "arg1": [], "arg2": {}}], "coreference_clusters": []}
        values = harness_metrics.counts(json.dumps(pred), json.dumps(pred))
        self.assertEqual(values["relation_f1"], (0, 0, 0))

    def test_offline_evaluation_requires_complete_matching_split(self):
        rows = read_arrow(ROOT / "data/scierc/validation.arrow")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "samples.jsonl"
            path.write_text("".join(json.dumps({"doc": row, "filtered_resps": [row["answer"]]}) + "\n"
                                    for row in rows), encoding="utf-8")
            report = evaluate(path, "validation", ROOT / "data/scierc")
            self.assertEqual(report["documents"], 53)
            self.assertEqual(report["schema_valid_rate"], 1)
            self.assertEqual(report["protocol"], "paper_ordered")
            self.assertEqual(set(report["metrics"].values()), {1.0})
            with self.assertRaisesRegex(ValueError, "outside test"):
                evaluate(path, "test", ROOT / "data/scierc")

    @unittest.skipUnless(importlib.util.find_spec("datasets"), "preference integration requires datasets")
    def test_validation_to_preference_cli_and_test_rejection(self):
        from datasets import load_from_disk
        rows = read_arrow(ROOT / "data/scierc/validation.arrow")
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            path = tmp / "samples.jsonl"
            path.write_text("".join(json.dumps({"doc": row, "filtered_resps": [row["answer"]]}) + "\n"
                                    for row in rows), encoding="utf-8")
            conf = tmp / "confusions"
            subprocess.run([sys.executable, str(ROOT / "scripts/build_validation_confusions.py"),
                            "--samples_jsonl", str(path), "--out_dir", str(conf)], check=True, capture_output=True)
            command = [sys.executable, str(ROOT / "scripts/build_preference_dataset.py"),
                       "--entity_confusion_file", str(conf / "entity_type_confusion.csv"),
                       "--relation_confusion_file", str(conf / "relation_type_confusion.csv")]
            output = tmp / "preferences"
            subprocess.run(command + ["--input_arrow", str(ROOT / "data/scierc/train.arrow"),
                                       "--output_dir", str(output)], check=True, capture_output=True)
            pairs = list(load_from_disk(str(output)))
            summary = json.loads((output / "build_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["synthetic_fraction"], 1.0)
            self.assertFalse(summary["expert_disambiguation"])
            training_prompts = {r["query"] for r in read_arrow(ROOT / "data/scierc/train.arrow")}
            self.assertGreater(len(pairs), 0)
            for pair in pairs:
                self.assertEqual(json.loads(pair["metadata"])["source"], "synthetic")
                self.assertIn(pair["prompt"], training_prompts)
                self.assertFalse(schema_errors(json.loads(pair["rejected"])))
                self.assertNotEqual(pair["chosen"], pair["rejected"])
            renamed = tmp / "renamed.arrow"
            renamed.write_bytes((ROOT / "data/scierc/test.arrow").read_bytes())
            result = subprocess.run(command + ["--input_arrow", str(renamed), "--output_dir", str(tmp / "bad")],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("test data is forbidden", result.stderr)


if __name__ == "__main__":
    unittest.main()
