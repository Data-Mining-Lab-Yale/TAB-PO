"""Check complete lm-eval logs against the audited validation documents."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tabpo.data import query_hash, read_arrow, sha256
from verify_splits import audit


def verify_validation_samples(path, data_dir):
    audit(data_dir)
    expected = {row["query"]: row["answer"] for row in read_arrow(data_dir / "validation.arrow")}
    seen = set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        doc = record.get("doc", {})
        query = doc.get("query")
        if query not in expected or query in seen:
            raise ValueError(f"Sample line {number} is duplicated or is outside the validation split")
        if json.loads(doc.get("answer", "null")) != json.loads(expected[query]):
            raise ValueError(f"Sample line {number}: gold answer differs from validation annotations")
        if not record.get("filtered_resps") and not record.get("resps"):
            raise ValueError(f"Sample line {number}: no model response field")
        seen.add(query)
    if len(seen) != len(expected):
        raise ValueError(f"Need all {len(expected)} validation predictions; got {len(seen)}")
    return {"validation_arrow_sha256": sha256((data_dir / "validation.arrow").read_bytes()),
            "samples_sha256": sha256(path.read_bytes()), "validated_documents": len(seen),
            "source_split": "validation"}
