#!/usr/bin/env python3
"""Build 100% confusion-aware synthetic SciERC preferences without expert adjudication."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import csv
import json
from pathlib import Path
import random
import uuid
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tabpo.data import schema_errors, sha256, query_hash

ENTITY_TYPES = ("Task", "Method", "Metric", "Material", "OtherScientificTerm", "Generic")
RELATION_TYPES = ("Used-for", "Evaluate-for", "Feature-of", "Part-of", "Hyponym-of", "Compare", "Conjunction")


def read_confusions(path: Path, gold_key: str, pred_key: str) -> dict[str, Counter]:
    counts: dict[str, Counter] = defaultdict(Counter)
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            gold, pred = row[gold_key], row[pred_key]
            count = int(row["Count"])
            if count < 0:
                raise ValueError(f"Negative confusion count in {path.name}")
            if gold != pred:
                counts[gold][pred] += count
    return counts


def serialize(obj: dict) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def valid(obj: dict) -> bool:
    return not schema_errors(obj)


def weighted_choice(rng: random.Random, options: list, weights: list[int | float]):
    return rng.choices(options, weights=weights, k=1)[0]


def label_flip(obj: dict, rng: random.Random, entity_confusions: dict, relation_confusions: dict, epsilon: float = 1.0):
    candidates = []
    for kind, field, table in (("entity", "entities", entity_confusions), ("relation", "relations", relation_confusions)):
        allowed = ENTITY_TYPES if kind == "entity" else RELATION_TYPES
        for index, record in enumerate(obj[field]):
            alternatives = {k: table.get(record.get("type"), {}).get(k, 0) + epsilon
                            for k in allowed if k != record.get("type")}
            if alternatives:
                candidates.append((kind, field, index, alternatives))
    if not candidates:
        return None
    weights = [sum((entity_confusions if c[0] == "entity" else relation_confusions)
                   .get(obj[c[1]][c[2]]["type"], {}).values()) + epsilon for c in candidates]
    kind, field, index, alternatives = weighted_choice(rng, candidates, weights)
    negative = copy.deepcopy(obj)
    negative[field][index]["type"] = weighted_choice(rng, list(alternatives), list(alternatives.values()))
    return negative, f"{kind}_label_confusion"


def missing_record(obj: dict, rng: random.Random, entity_confusions=None, relation_confusions=None):
    candidates = [("relations", i) for i in range(len(obj["relations"]))]
    candidates += [("entities", i) for i in range(len(obj["entities"]))]
    if not candidates:
        return None
    weights = [sum(((entity_confusions or {}) if field == "entities" else (relation_confusions or {}))
                   .get(obj[field][index]["type"], {}).values()) + 1 for field, index in candidates]
    field, index = weighted_choice(rng, candidates, weights)
    negative = copy.deepcopy(obj)
    removed = negative[field].pop(index)
    if field == "entities":
        entity_id = removed["entity_id"]
        negative["relations"] = [r for r in negative["relations"] if entity_id not in (r["arg1"], r["arg2"])]
        for cluster in negative["coreference_clusters"]:
            cluster["members"] = [m for m in cluster["members"] if m != entity_id]
        negative["coreference_clusters"] = [c for c in negative["coreference_clusters"] if len(c["members"]) >= 2]
    return negative, f"missing_{field[:-1]}"


def extra_relation(obj: dict, rng: random.Random, relation_confusions: dict):
    existing = {frozenset((r["arg1"], r["arg2"])) for r in obj["relations"]}
    candidates = [(a, b) for a in obj["entities"] for b in obj["entities"]
                  if a["entity_id"] != b["entity_id"] and a.get("sentence_id") == b.get("sentence_id")
                  and frozenset((a["entity_id"], b["entity_id"])) not in existing]
    if not candidates:
        return None
    a, b = rng.choice(candidates)
    negative = copy.deepcopy(obj)
    used_ids = {r.get("relation_id") for r in negative["relations"]}
    number = 1
    while f"R{number}" in used_ids:
        number += 1
    observed = Counter(r.get("type") for r in obj["relations"] if r.get("type") in RELATION_TYPES)
    observed.update({k: sum(v.values()) for k, v in relation_confusions.items() if k in RELATION_TYPES})
    relation_type = weighted_choice(rng, list(RELATION_TYPES), [observed.get(t, 0) + 1 for t in RELATION_TYPES])
    negative["relations"].append({"relation_id": f"R{number}", "type": relation_type,
                                  "arg1": a["entity_id"], "arg2": b["entity_id"],
                                  "sentence_id": a.get("sentence_id")})
    return negative, "extra_relation"


def relink(obj: dict, rng: random.Random, relation_confusions=None):
    entity_by_id = {e["entity_id"]: e for e in obj["entities"]}
    candidates = []
    for index, rel in enumerate(obj["relations"]):
        sid = entity_by_id[rel["arg1"]].get("sentence_id")
        for entity in obj["entities"]:
            candidate_id = entity["entity_id"]
            already_gold = any(r["type"] == rel["type"] and r["arg1"] == rel["arg1"] and r["arg2"] == candidate_id
                               for r in obj["relations"])
            if entity.get("sentence_id") == sid and candidate_id not in (rel["arg1"], rel["arg2"]) and not already_gold:
                candidates.append((index, candidate_id))
    if not candidates:
        return None
    index, candidate_id = weighted_choice(rng, candidates,
        [sum((relation_confusions or {}).get(obj["relations"][i]["type"], {}).values()) + 1 for i, _ in candidates])
    negative = copy.deepcopy(obj)
    negative["relations"][index]["arg2"] = candidate_id
    return negative, "relation_relink"


def synthetic_pairs(rows: list[dict], entity_confusions: dict, relation_confusions: dict,
                    negatives_per_sample: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    output = []
    for row_index, row in enumerate(rows):
        chosen = json.loads(row["answer"])
        if not valid(chosen):
            raise ValueError(f"Invalid gold schema in source row {row_index}")
        seen = set()
        builders = [lambda: label_flip(chosen, rng, entity_confusions, relation_confusions),
                    lambda: missing_record(chosen, rng, entity_confusions, relation_confusions),
                    lambda: extra_relation(chosen, rng, relation_confusions),
                    lambda: relink(chosen, rng, relation_confusions)]
        for _ in range(negatives_per_sample):
            rng.shuffle(builders)
            result = None
            for _attempt in range(32):
                candidate = rng.choice(builders)()
                if candidate is not None and valid(candidate[0]) and serialize(candidate[0]) not in seen and candidate[0] != chosen:
                    result = candidate
                    break
            if result is None:
                break
            rejected, family = result
            seen.add(serialize(rejected))
            output.append({"prompt": row["query"], "chosen": serialize(chosen),
                           "rejected": serialize(rejected),
                           "metadata": json.dumps({"source_row": row_index, "negative_family": family,
                                                   "source_query_sha256": query_hash(row),
                                                   "source": "synthetic"})})
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_arrow", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--entity_confusion_file", type=Path, required=True)
    parser.add_argument("--relation_confusion_file", type=Path, required=True)
    parser.add_argument("--negatives_per_sample", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", type=Path, default=Path(__file__).resolve().parents[1] / "data/scierc")
    args = parser.parse_args()
    if args.negatives_per_sample < 1:
        parser.error("negatives_per_sample must be >= 1")
    if args.input_arrow.stem == "test":
        parser.error("Test data must not be used to build preference pairs")
    from datasets import Dataset
    import pyarrow as pa
    from verify_splits import audit
    audit(args.data_dir)
    manifest = json.loads((args.data_dir / "split_manifest.json").read_text(encoding="utf-8"))
    source_hash = sha256(args.input_arrow.read_bytes())
    source_split = next((s for s in ("train", "validation") if source_hash == manifest["splits"][s]["sha256"]), None)
    if source_split is None:
        parser.error("input_arrow must match an audited training or validation split; test data is forbidden")
    provenance_path = args.entity_confusion_file.parent / "confusion_summary.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("validation_arrow_sha256") != manifest["splits"]["validation"]["sha256"]:
        parser.error("Confusions must have verified provenance from this validation split")
    for path in (args.entity_confusion_file, args.relation_confusion_file):
        if provenance.get("csv_sha256", {}).get(path.name) != sha256(path.read_bytes()):
            parser.error(f"Confusion CSV checksum does not match provenance: {path.name}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("output_dir is nonempty; choose a new directory")
    source = Dataset.from_file(str(args.input_arrow))
    entity_confusions = read_confusions(args.entity_confusion_file, "Ground_Truth_EntityType", "Predicted_EntityType")
    relation_confusions = read_confusions(args.relation_confusion_file, "Ground_Truth_RelationType", "Predicted_RelationType")
    synthetic = synthetic_pairs(list(source), entity_confusions, relation_confusions,
                                args.negatives_per_sample, args.seed)
    pairs = synthetic
    random.Random(args.seed).shuffle(pairs)
    if not pairs:
        raise ValueError("No preference pairs could be generated")
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    # Dataset.from_list fingerprints an in-memory Arrow table through dill.
    # Writing a temporary Arrow stream also works with Python/Arrow versions
    # where that fingerprint serialization is unsupported.
    arrow_path = args.output_dir.parent / f".preferences-{uuid.uuid4().hex}.arrow"
    table = pa.Table.from_pylist(pairs)
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    arrow_path.write_bytes(sink.getvalue().to_pybytes())
    try:
        Dataset.from_file(str(arrow_path)).save_to_disk(str(args.output_dir))
    finally:
        arrow_path.unlink(missing_ok=True)
    summary = {"input_rows": len(source), "synthetic_pairs": len(synthetic),
               "total_pairs": len(pairs), "seed": args.seed,
               "construction": "confusion-aware synthetic", "synthetic_fraction": 1.0,
               "expert_disambiguation": False,
               "source_split": source_split, "source_arrow_sha256": source_hash,
               "validation_arrow_sha256": provenance["validation_arrow_sha256"],
               "confusion_samples_sha256": provenance["samples_sha256"],
               "negative_families": dict(Counter(json.loads(p["metadata"])["negative_family"] for p in synthetic)),
               "unpaired_source_rows": len(source) - len({json.loads(p["metadata"])["source_row"] for p in synthetic})}
    (args.output_dir / "build_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
