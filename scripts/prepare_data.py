#!/usr/bin/env python3
"""Prepare an auditable document-level SciERC train/validation/test partition."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pyarrow as pa
from tabpo.data import FIELDS, label_presence, query_hash, row_hash, schema_errors, sha256, write_arrow
from tabpo.splitting import iterative_split


def archive_rows(path: Path) -> tuple[list[dict], bytes, str]:
    with zipfile.ZipFile(path) as archive:
        matches = [n for n in archive.namelist() if n.strip("./") == "data-00000-of-00001.arrow"]
        if len(matches) != 1:
            raise ValueError(f"{path.name}: expected one top-level source Arrow; found {matches}")
        raw = archive.read(matches[0])
        return pa.ipc.open_stream(pa.BufferReader(raw)).read_all().to_pylist(), raw, matches[0]


def prepare(train_archive: Path, test_archive: Path, output_dir: Path, seed: int = 42,
            validation_size: int = 53) -> dict:
    pool, train_raw, train_member = archive_rows(train_archive)
    test, test_raw, test_member = archive_rows(test_archive)
    identities = {}
    for name, rows in (("train_pool", pool), ("test", test)):
        identities[name] = {query_hash(row) for row in rows}
        if len(identities[name]) != len(rows):
            raise ValueError(f"Duplicate documents in {name}")
        for i, row in enumerate(rows):
            if any(not isinstance(row.get(f), str) for f in FIELDS):
                raise ValueError(f"{name} row {i}: all source columns must be strings")
            errors = schema_errors(json.loads(row["answer"]))
            if errors:
                raise ValueError(f"{name} row {i}: {errors}")
    if identities["train_pool"] & identities["test"]:
        raise ValueError("Training pool overlaps held-out test documents")
    labels = [label_presence(row) for row in pool]
    ti, vi = iterative_split(labels, validation_size, seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = {"train": [pool[i] for i in ti], "validation": [pool[i] for i in vi], "test": test}
    for name in splits:
        write_arrow(output_dir / f"{name}.arrow", splits[name])
    all_labels = sorted(set().union(*labels))
    stats = {}
    for label in all_labels:
        stats[label] = {}
        for name, rows in splits.items():
            presence = sum(label in label_presence(row) for row in rows)
            stats[label][name] = {"documents": presence, "prevalence": presence / len(rows)}
    manifest = {
        "schema_version": 2, "columns": list(FIELDS),
        "method": "first-order iterative multilabel stratification with exact capacities",
        "implementation": "tabpo.splitting.iterative_split", "seed": seed,
        "stratification_unit": "complete prepared document (query)",
        "stratification_labels": all_labels, "label_statistics": stats,
        "source_training_rows": len(pool), "source_test_rows": len(test),
        "source_training_arrow_sha256": sha256(train_raw), "source_test_arrow_sha256": sha256(test_raw),
        "source_archives": {"train": train_archive.name, "test": test_archive.name},
        "source_members": {"train": train_member, "test": test_member},
        "split_indices": {"train": ti, "validation": vi},
        "source_row_hashes": [row_hash(row) for row in pool],
        "source_test_row_hashes": [row_hash(row) for row in test],
        "splits": {name: {"rows": len(rows), "sha256": sha256((output_dir / f"{name}.arrow").read_bytes()),
                           "query_hashes": [query_hash(row) for row in rows]} for name, rows in splits.items()},
        "verification_passed": True,
        "note": "Document-level stratified split; only query, answer, and context are retained. Test documents and annotations are unchanged.",
    }
    (output_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-archive", type=Path, required=True)
    p.add_argument("--test-archive", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("data/scierc"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--validation-size", type=int, default=53)
    args = p.parse_args()
    if args.output_dir.exists() and any(args.output_dir.glob("*.arrow")):
        p.error("Output already contains Arrow files. Use a new directory to preserve existing splits.")
    manifest = prepare(args.train_archive, args.test_archive, args.output_dir, args.seed, args.validation_size)
    print(json.dumps({"method": manifest["method"], "seed": args.seed,
                      "rows": {s: d["rows"] for s, d in manifest["splits"].items()},
                      "verification_passed": True}, indent=2))


if __name__ == "__main__":
    main()
