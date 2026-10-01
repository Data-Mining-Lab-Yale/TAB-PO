"""Verify split checksums, schema, conservation, and iterative stratification."""
from __future__ import annotations
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tabpo.data import FIELDS, label_presence, query_hash, read_arrow, row_hash, schema_errors, sha256
from tabpo.splitting import iterative_split


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit(data_dir: Path) -> dict:
    manifest = json.loads((data_dir / "split_manifest.json").read_text(encoding="utf-8"))
    require(manifest.get("schema_version") == 2 and manifest.get("columns") == list(FIELDS),
            "Expected the three-column SciERC schema")
    identities, all_rows, results = {}, {}, {}
    for split, expected in manifest["splits"].items():
        path = data_dir / f"{split}.arrow"
        require(sha256(path.read_bytes()) == expected["sha256"], f"{split}: checksum mismatch")
        rows = read_arrow(path)
        require(len(rows) == expected["rows"], f"{split}: incorrect row count")
        for i, row in enumerate(rows):
            require(tuple(row) == FIELDS, f"{split}:{i}: unexpected columns")
            require(all(isinstance(row[k], str) for k in FIELDS), f"{split}:{i}: non-string column")
            require(not schema_errors(json.loads(row["answer"])), f"{split}:{i}: invalid gold schema")
        hashes = [query_hash(row) for row in rows]
        require(hashes == expected["query_hashes"], f"{split}: membership mismatch")
        require(len(set(hashes)) == len(hashes), f"{split}: duplicate document")
        identities[split], all_rows[split] = set(hashes), rows
        results[split] = {"rows": len(rows), "valid_json_answers": len(rows), "sha256": expected["sha256"]}
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        require(not identities[a] & identities[b], f"{a}/{b}: document overlap")
    size = manifest["source_training_rows"]
    pool = [None] * size
    seen_indices = set()
    for split in ("train", "validation"):
        indices = manifest["split_indices"][split]
        require(len(indices) == len(all_rows[split]), f"{split}: index count mismatch")
        for i, row in zip(indices, all_rows[split]):
            require(type(i) is int and 0 <= i < size and i not in seen_indices, "Invalid or repeated source index")
            pool[i] = row
            seen_indices.add(i)
    require(all(row is not None for row in pool), "Training pool is not conserved")
    require([row_hash(row) for row in pool] == manifest["source_row_hashes"], "Source annotations were altered")
    ti, vi = iterative_split([label_presence(row) for row in pool], len(all_rows["validation"]), manifest["seed"])
    require(ti == manifest["split_indices"]["train"] and vi == manifest["split_indices"]["validation"],
            "Iterative stratification does not reproduce saved membership")
    for label, per_split in manifest["label_statistics"].items():
        for split, expected in per_split.items():
            actual = sum(label in label_presence(row) for row in all_rows[split])
            require(actual == expected["documents"], f"{split}: stale prevalence for {label}")
            require(abs(actual / len(all_rows[split]) - expected["prevalence"]) < 1e-12, "Stale prevalence")
    require(len(all_rows["test"]) == manifest["source_test_rows"], "Held-out test row count changed")
    require([row_hash(row) for row in all_rows["test"]] == manifest["source_test_row_hashes"],
            "Held-out test documents or annotations were altered")
    results.update(query_overlap=0, training_pool_conserved=True, stratification_reproduced=True,
                   gold_schema_valid=True, test_documents_unchanged=True, columns=list(FIELDS))
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data/scierc")
    args = parser.parse_args()
    print(json.dumps(audit(args.data_dir), indent=2))
