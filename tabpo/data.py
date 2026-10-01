"""Arrow I/O and stable row identities, independent of Hugging Face Datasets."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pyarrow as pa

FIELDS = ("query", "answer", "context")
ENTITY_TYPES = ("Task", "Method", "Metric", "Material", "OtherScientificTerm", "Generic")
RELATION_TYPES = ("Used-for", "Evaluate-for", "Feature-of", "Part-of", "Hyponym-of", "Compare", "Conjunction")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def query_hash(row: dict) -> str:
    return sha256(row["query"].encode("utf-8"))


def row_hash(row: dict) -> str:
    return sha256(json.dumps({k: row[k] for k in FIELDS}, sort_keys=True,
                             ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def read_arrow(path: str | Path) -> list[dict]:
    with pa.memory_map(str(path), "r") as source:
        return pa.ipc.open_stream(source).read_all().to_pylist()


def write_arrow(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    schema = pa.schema([(field, pa.string()) for field in FIELDS])
    with pa.OSFile(str(path), "wb") as sink:
        with pa.ipc.new_stream(sink, schema) as writer:
            writer.write_table(pa.Table.from_pylist(rows, schema=schema))


def schema_errors(obj: object) -> list[str]:
    """Validate the prepared SciERC annotation contract."""
    if not isinstance(obj, dict):
        return ["answer must be an object"]
    if any(not isinstance(obj.get(k), list) for k in ("entities", "relations", "coreference_clusters")):
        return ["entities, relations, coreference_clusters must be arrays"]
    errors = []
    entities = {}
    for e in obj["entities"]:
        if not isinstance(e, dict):
            errors.append("entity must be an object")
            continue
        eid = e.get("entity_id")
        if not isinstance(eid, str) or not eid or eid in entities:
            errors.append("entity_id must be a unique nonempty string")
        else:
            entities[eid] = e
        if e.get("type") not in ENTITY_TYPES or not isinstance(e.get("text"), str) or not e.get("text", "").strip():
            errors.append("entity requires valid type and nonempty text")
        if type(e.get("sentence_id")) is not int or e["sentence_id"] < 0:
            errors.append("entity sentence_id must be a nonnegative integer")
    relation_ids = set()
    for r in obj["relations"]:
        if not isinstance(r, dict):
            errors.append("relation must be an object")
            continue
        rid = r.get("relation_id")
        if not isinstance(rid, str) or not rid or rid in relation_ids:
            errors.append("relation_id must be a unique nonempty string")
        if isinstance(rid, str):
            relation_ids.add(rid)
        if r.get("type") not in RELATION_TYPES:
            errors.append("unknown relation type")
        a, b = r.get("arg1"), r.get("arg2")
        if not isinstance(a, str) or not isinstance(b, str) or a not in entities or b not in entities or a == b:
            errors.append("relation endpoints must name distinct existing entities")
            continue
        sid = entities[a].get("sentence_id")
        if type(r.get("sentence_id")) is not int or sid != entities[b].get("sentence_id") or r.get("sentence_id") != sid:
            errors.append("relation and endpoints must have the same sentence_id")
    cluster_ids = set()
    clustered = set()
    for c in obj["coreference_clusters"]:
        if not isinstance(c, dict):
            errors.append("coreference cluster must be an object")
            continue
        cid = c.get("cluster_id")
        if not isinstance(cid, str) or not cid or cid in cluster_ids:
            errors.append("cluster_id must be a unique nonempty string")
        if isinstance(cid, str):
            cluster_ids.add(cid)
        members = c.get("members")
        if not isinstance(members, list) or len(members) < 2 or any(not isinstance(m, str) or m not in entities for m in members):
            errors.append("coreference requires at least two existing entity IDs")
            continue
        if len(set(members)) != len(members) or clustered.intersection(members):
            errors.append("coreference members must be unique and belong to one cluster")
        clustered.update(members)
    return errors


def label_presence(row: dict) -> set[str]:
    obj = json.loads(row["answer"])
    labels = {"entity:" + e["type"] for e in obj["entities"]}
    labels.update("relation:" + r["type"] for r in obj["relations"])
    if any(len(c["members"]) >= 2 for c in obj["coreference_clusters"]):
        labels.add("has_coreference")
    return labels
