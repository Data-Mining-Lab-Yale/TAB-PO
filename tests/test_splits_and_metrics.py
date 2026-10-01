import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "evaluation")]
from tabpo.data import FIELDS, schema_errors, read_arrow, label_presence
from tabpo.splitting import iterative_split
from verify_splits import audit
from sample_provenance import verify_validation_samples
import harness_metrics as metrics
import scierc_utils as utils
import build_validation_confusions as confusion


def gold():
    return {"entities": [
        {"entity_id": "E1", "text": "GraphParser", "type": "Method", "sentence_id": 0},
        {"entity_id": "E2", "text": "parsing", "type": "Task", "sentence_id": 0},
        {"entity_id": "E3", "text": "GraphParser", "type": "Method", "sentence_id": 1}],
        "relations": [{"relation_id": "R1", "type": "Used-for", "arg1": "E1", "arg2": "E2", "sentence_id": 0}],
        "coreference_clusters": [{"cluster_id": "C1", "members": ["E1", "E3"]}]}


class SplitTests(unittest.TestCase):
    def test_published_splits_conserved_and_reproduced(self):
        report = audit(ROOT / "data/scierc")
        self.assertTrue(report["stratification_reproduced"])
        self.assertTrue(report["test_documents_unchanged"])
        self.assertEqual(report["query_overlap"], 0)

    def test_only_scierc_columns_are_published(self):
        self.assertEqual(FIELDS, ("query", "answer", "context"))
        for split in ("train", "validation", "test"):
            self.assertTrue(all(tuple(row) == FIELDS for row in read_arrow(ROOT / f"data/scierc/{split}.arrow")))

    def test_rare_labels_and_exact_capacities(self):
        labels = [{"common", "rare"} if i < 4 else {"common"} for i in range(20)]
        ti, vi = iterative_split(labels, 5, 42)
        self.assertEqual((len(ti), len(vi)), (15, 5))
        self.assertFalse(set(ti) & set(vi))
        self.assertEqual(sum("rare" in labels[i] for i in vi), 1)
        self.assertEqual((ti, vi), iterative_split(labels, 5, 42))

    def test_unlabeled_documents_are_retained(self):
        ti, vi = iterative_split([set()] * 10, 3)
        self.assertEqual((len(ti), len(vi)), (7, 3))
        self.assertEqual(sorted(ti + vi), list(range(10)))

    def test_test_sample_cannot_be_used_as_validation(self):
        row = read_arrow(ROOT / "data/scierc/test.arrow")[0]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "samples.jsonl"
            path.write_text(json.dumps({"doc": row, "filtered_resps": [row["answer"]]}) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "outside the validation"):
                verify_validation_samples(path, ROOT / "data/scierc")

    def test_incomplete_validation_log_fails(self):
        row = read_arrow(ROOT / "data/scierc/validation.arrow")[0]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "samples.jsonl"
            path.write_text(json.dumps({"doc": row, "filtered_resps": [row["answer"]]}) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Need all 53"):
                verify_validation_samples(path, ROOT / "data/scierc")


class MetricTests(unittest.TestCase):
    def setUp(self):
        self.answer = gold()

    def score(self, prediction):
        c = metrics.counts(json.dumps(self.answer), json.dumps(prediction))
        return {k: metrics.micro_f1([v]) for k, v in c.items() if k.endswith("f1")}

    def test_identity_scores_one(self):
        self.assertEqual(set(self.score(self.answer).values()), {1.0})

    def test_reversed_relation_is_rejected_for_every_type(self):
        pred = copy.deepcopy(self.answer)
        pred["relations"][0].update(arg1="E2", arg2="E1")
        for relation_type in utils.RELATION_SET:
            with self.subTest(relation_type=relation_type):
                self.answer["relations"][0]["type"] = relation_type
                pred["relations"][0]["type"] = relation_type
                self.assertEqual(self.score(pred)["relation_f1"], 0)
                aggregate = utils.evaluate_scierc_agg([[json.dumps(self.answer), json.dumps(pred)]])
                self.assertEqual(aggregate["relation"]["f1"], 0)

    def test_entity_presence_does_not_imply_correct_mentions(self):
        pred = copy.deepcopy(self.answer)
        for e in pred["entities"]:
            e["text"] = "unrelated evidence"
        self.assertEqual(self.score(pred)["entity_f1"], 1)
        self.assertEqual(self.score(pred)["span_f1"], 0)

    def test_duplicate_relations_count_as_false_positives(self):
        pred = copy.deepcopy(self.answer)
        pred["relations"].append(copy.deepcopy(pred["relations"][0]))
        self.assertAlmostEqual(self.score(pred)["relation_f1"], 2 / 3)

    def test_coreference_wrong_sentence_is_not_a_match(self):
        pred = copy.deepcopy(self.answer)
        pred["entities"][2]["sentence_id"] = 9
        self.assertEqual(self.score(pred)["coreference_f1"], 0)

    def test_invalid_json_parse_rate_is_zero(self):
        values = metrics.counts(json.dumps(self.answer), "not JSON")
        self.assertEqual(values["json_parse_rate"], 0)
        self.assertEqual(metrics.micro_f1([values["span_f1"]]), 0)

    def test_micro_f1_aggregates_counts_not_document_f1(self):
        self.assertAlmostEqual(metrics.micro_f1([(1, 0, 0), (0, 9, 9)]), 0.1)

    def test_adapter_matches_strict_aggregation(self):
        pred = copy.deepcopy(self.answer)
        pred["entities"][1]["type"] = "Method"
        pairs = [[json.dumps(self.answer), json.dumps(pred)]]
        old = utils.evaluate_scierc_agg(pairs)
        new = self.score(pred)
        for key, old_key in (("entity_f1", "entity"), ("relation_f1", "relation"),
                             ("span_f1", "span"), ("coreference_f1", "coreference")):
            self.assertAlmostEqual(new[key], old[old_key]["f1"], places=4)

    def test_confusion_requires_both_endpoints_to_match(self):
        gt = {"arg1_text": "network", "arg2_text": "image classification"}
        pred = {"arg1_text": "network", "arg2_text": "image unrelated"}
        self.assertEqual(confusion.relation_endpoint_score(gt, pred, jaccard_threshold=0.6), 0)

    def test_confusion_alignment_requires_order_for_every_type(self):
        for relation_type in utils.RELATION_SET:
            gt = {"type": relation_type, "arg1_text": "network", "arg2_text": "classification"}
            pred = {"type": relation_type, "arg1_text": "classification", "arg2_text": "network"}
            self.assertEqual(confusion.relation_match_score(gt, pred), (0, "none"))

    def test_coreference_requires_sentence_ids(self):
        pred = copy.deepcopy(self.answer)
        del pred["entities"][2]["sentence_id"]
        self.assertEqual(self.score(pred)["coreference_f1"], 0)


class SchemaTests(unittest.TestCase):
    def test_valid_graph(self):
        self.assertEqual(schema_errors(gold()), [])

    def test_dangling_endpoint_rejected(self):
        obj = gold()
        obj["relations"][0]["arg1"] = "missing"
        self.assertTrue(schema_errors(obj))

    def test_malformed_records_do_not_crash(self):
        obj = gold()
        obj["entities"].append(None)
        self.assertTrue(schema_errors(obj))

    def test_singleton_coreference_rejected(self):
        obj = gold()
        obj["coreference_clusters"][0]["members"] = ["E1"]
        self.assertTrue(schema_errors(obj))


if __name__ == "__main__":
    unittest.main()
