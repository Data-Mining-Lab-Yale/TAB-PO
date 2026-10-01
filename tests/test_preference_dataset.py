import importlib.util
from pathlib import Path
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_preference_dataset.py"
spec = importlib.util.spec_from_file_location("preferences", SCRIPT)
preferences = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preferences)


class PreferenceDatasetTests(unittest.TestCase):
    def setUp(self):
        self.gold = {
            "entities": [
                {"entity_id": "E1", "text": "method", "type": "Method", "sentence_id": 0},
                {"entity_id": "E2", "text": "task", "type": "Task", "sentence_id": 0},
                {"entity_id": "E3", "text": "score", "type": "Metric", "sentence_id": 0},
            ],
            "relations": [
                {"relation_id": "R1", "type": "Used-for", "arg1": "E1", "arg2": "E2", "sentence_id": 0},
            ],
            "coreference_clusters": [],
        }

    def test_generated_pairs_are_distinct_and_schema_valid(self):
        row = {"query": "Extract entities and relations", "answer": preferences.serialize(self.gold)}
        entity_confusions = {"Method": {"Task": 3}}
        relation_confusions = {"Used-for": {"Evaluate-for": 2}}
        pairs = preferences.synthetic_pairs([row], entity_confusions, relation_confusions, 4, 42)
        self.assertGreaterEqual(len(pairs), 2)
        self.assertEqual(len({p["rejected"] for p in pairs}), len(pairs))
        for pair in pairs:
            self.assertNotEqual(pair["chosen"], pair["rejected"])
            self.assertTrue(preferences.valid(__import__("json").loads(pair["rejected"])))

    def test_invalid_cross_sentence_relation_is_rejected(self):
        self.gold["entities"][1]["sentence_id"] = 1
        self.assertFalse(preferences.valid(self.gold))


if __name__ == "__main__":
    unittest.main()
