#!/usr/bin/env python3
"""
Build SCIERC entity-type and relation-type confusion CSVs from vLLM/lm_eval
jsonl inference samples.

SCIERC expected answer/prediction format:
{
  "entities": [
    {
      "entity_id": "E1",
      "text": "...",
      "type": "Task|Method|Metric|Material|OtherScientificTerm|Generic",
      "sentence_id": 0
    }
  ],
  "relations": [
    {
      "relation_id": "R1",
      "type": "Used-for|Evaluate-for|Feature-of|Part-of|Hyponym-of|Compare|Conjunction",
      "arg1": "E1",
      "arg2": "E2",
      "sentence_id": 0
    }
  ],
  "coreference_clusters": [
    {
      "cluster_id": "C1",
      "members": ["E1", "E2"]
    }
  ]
}

Outputs:
  1) entity_type_confusion.csv
       Ground_Truth_EntityType,Predicted_EntityType,Count

  2) relation_type_confusion.csv
       Ground_Truth_RelationType,Predicted_RelationType,Count

  3) Dense versions for inspection:
       entity_type_confusion_dense.csv
       relation_type_confusion_dense.csv

  4) matched_mismatch_examples.json

  5) confusion_summary.json

Important design choices:
1) Entity confusions are counted only for matched GT/pred entity spans.
   Pure missed entities and pure spurious entities are not folded into the
   confusion matrix because the DPO-data builder should handle missing/extra
   entities separately.

2) Relation confusions are counted only for matched GT/pred relation endpoints.
   We resolve relation arg IDs to entity texts and match relations by endpoint
   text similarity, not raw entity IDs, because generated entity IDs may differ.

3) Relation endpoint order is strict for every relation type, as in Appendix F.

4) Raw label strings are preserved exactly as they appear in samples, except that
   unknown/missing labels are ignored unless --include_unknown_labels is enabled.

Example:
    python scripts/build_validation_confusions.py \
        --samples_jsonl /path/to/samples.jsonl \
        --out_dir /path/to/confusions \
        --jaccard_threshold 0.6
"""

import argparse
import ast
import csv
import html
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ============================================================================
# SCIERC label sets
# ============================================================================

ENTITY_TYPE_SET = {
    "Task",
    "Method",
    "Metric",
    "Material",
    "OtherScientificTerm",
    "Generic",
}

RELATION_TYPE_SET = {
    "Used-for",
    "Evaluate-for",
    "Feature-of",
    "Part-of",
    "Hyponym-of",
    "Compare",
    "Conjunction",
}

TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


# ============================================================================
# General text helpers
# ============================================================================

def normalize_space(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def norm_text(text: Any) -> str:
    return normalize_space(text).lower()


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(norm_text(text))


def token_jaccard(a: str, b: str) -> float:
    sa = set(tokenize(a))
    sb = set(tokenize(b))
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def containment_or_jaccard(a: str, b: str) -> float:
    """
    Slightly more forgiving span similarity:
      - exact normalized match => 1.0
      - full containment => 1.0
      - otherwise token Jaccard
    """
    a_norm = norm_text(a)
    b_norm = norm_text(b)

    if not a_norm and not b_norm:
        return 1.0
    if not a_norm or not b_norm:
        return 0.0
    if a_norm == b_norm:
        return 1.0
    if a_norm in b_norm or b_norm in a_norm:
        return 1.0

    return token_jaccard(a_norm, b_norm)


# ============================================================================
# Robust JSON parsing
# ============================================================================

def strip_code_fences(text: str) -> str:
    text = str(text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def extract_first_json_object(text: str) -> Optional[str]:
    """
    Extract first balanced {...} JSON-like object from a string.
    Handles braces inside quoted strings.
    """
    text = strip_code_fences(text)
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(text)):
        ch = text[i]

        if escape:
            escape = False
            continue

        if ch == "\\":
            escape = True
            continue

        if ch == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]

    return None


def normalize_scierc_obj(obj: Any) -> Dict[str, List[Dict[str, Any]]]:
    """
    Normalize parsed object into the SCIERC shape.
    """
    if not isinstance(obj, dict):
        return {
            "entities": [],
            "relations": [],
            "coreference_clusters": [],
        }

    entities = obj.get("entities", [])
    relations = obj.get("relations", [])
    coref = obj.get("coreference_clusters", [])

    if not isinstance(entities, list):
        entities = []
    if not isinstance(relations, list):
        relations = []
    if not isinstance(coref, list):
        coref = []

    return {
        "entities": entities,
        "relations": relations,
        "coreference_clusters": coref,
    }


def safe_json_loads(text: Any) -> Dict[str, List[Dict[str, Any]]]:
    """
    Robust JSON parser for model outputs.

    Tries:
      1) already-dict input
      2) direct json.loads
      3) first balanced JSON object
      4) html-unescape + normalized smart quotes + json.loads
      5) ast.literal_eval fallback
      6) regex salvage for entities/relations/coref
    """
    if isinstance(text, dict):
        return normalize_scierc_obj(text)

    if text is None:
        return normalize_scierc_obj({})

    raw = strip_code_fences(str(text))

    # 1) Direct JSON
    try:
        return normalize_scierc_obj(json.loads(raw))
    except Exception:
        pass

    # 2) First balanced JSON object
    candidate = extract_first_json_object(raw)
    if candidate is not None:
        try:
            return normalize_scierc_obj(json.loads(candidate))
        except Exception:
            pass

    # 3) HTML unescape + smart quote normalization
    raw2 = html.unescape(raw)
    raw2 = (
        raw2.replace("“", '"')
            .replace("”", '"')
            .replace("’", "'")
            .replace("‘", "'")
    )

    candidate2 = extract_first_json_object(raw2)
    if candidate2 is not None:
        try:
            return normalize_scierc_obj(json.loads(candidate2))
        except Exception:
            pass

        # 4) Python literal fallback
        try:
            return normalize_scierc_obj(ast.literal_eval(candidate2))
        except Exception:
            pass

    # 5) Regex salvage
    salvaged = {
        "entities": [],
        "relations": [],
        "coreference_clusters": [],
    }

    entity_pattern = re.compile(
        r'\{\s*"entity_id"\s*:\s*"([^"]+)"\s*,\s*'
        r'"text"\s*:\s*"([^"]*)"\s*,\s*'
        r'"type"\s*:\s*"([^"]+)"\s*,\s*'
        r'"sentence_id"\s*:\s*(-?\d+)\s*\}'
    )

    relation_pattern = re.compile(
        r'\{\s*"relation_id"\s*:\s*"([^"]+)"\s*,\s*'
        r'"type"\s*:\s*"([^"]+)"\s*,\s*'
        r'"arg1"\s*:\s*"([^"]+)"\s*,\s*'
        r'"arg2"\s*:\s*"([^"]+)"\s*,\s*'
        r'"sentence_id"\s*:\s*(-?\d+)\s*\}'
    )

    cluster_pattern = re.compile(
        r'\{\s*"cluster_id"\s*:\s*"([^"]+)"\s*,\s*'
        r'"members"\s*:\s*\[([^\]]*)\]\s*\}'
    )

    for m in entity_pattern.finditer(raw2):
        salvaged["entities"].append(
            {
                "entity_id": m.group(1),
                "text": m.group(2),
                "type": m.group(3),
                "sentence_id": int(m.group(4)),
            }
        )

    for m in relation_pattern.finditer(raw2):
        salvaged["relations"].append(
            {
                "relation_id": m.group(1),
                "type": m.group(2),
                "arg1": m.group(3),
                "arg2": m.group(4),
                "sentence_id": int(m.group(5)),
            }
        )

    for m in cluster_pattern.finditer(raw2):
        raw_members = [x.strip() for x in m.group(2).split(",") if x.strip()]
        members = [x.strip().strip('"').strip("'") for x in raw_members]
        salvaged["coreference_clusters"].append(
            {
                "cluster_id": m.group(1),
                "members": members,
            }
        )

    return normalize_scierc_obj(salvaged)


# ============================================================================
# lm_eval/vLLM sample helpers
# ============================================================================

def get_pred_response(record: Dict[str, Any]) -> str:
    """
    Prefer filtered_resps[0]; otherwise fall back to resps[0][0].
    This matches typical lm_eval --log_samples output.
    """
    filtered = record.get("filtered_resps")
    if filtered and len(filtered) > 0:
        return filtered[0]

    resps = record.get("resps")
    if resps and len(resps) > 0:
        first = resps[0]
        if isinstance(first, list) and len(first) > 0:
            return first[0]
        if isinstance(first, str):
            return first

    raise ValueError("No prediction response found in filtered_resps or resps")


def get_gold_answer_text(record: Dict[str, Any]) -> str:
    """
    Common locations for gold answer in lm_eval samples:
      record["doc"]["answer"]
      record["doc"]["target"]
      record["target"]
    """
    doc = record.get("doc", {})
    if isinstance(doc, dict):
        if "answer" in doc:
            return doc["answer"]
        if "target" in doc:
            return doc["target"]

    if "target" in record:
        return record["target"]

    raise ValueError("No gold answer found in record['doc']['answer'], record['doc']['target'], or record['target']")


def get_scierc_obj_from_answer(answer_text: Any) -> Dict[str, List[Dict[str, Any]]]:
    return safe_json_loads(answer_text)


# ============================================================================
# SCIERC object cleaning
# ============================================================================

def safe_int(x: Any) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None


def clean_entities(
    obj: Dict[str, List[Dict[str, Any]]],
    include_unknown_labels: bool = False,
) -> List[Dict[str, Any]]:
    cleaned = []

    for ent in obj.get("entities", []):
        if not isinstance(ent, dict):
            continue

        ent_id = str(ent.get("entity_id", "")).strip()
        text = normalize_space(ent.get("text", ""))
        ent_type = normalize_space(ent.get("type", ""))
        sent_id = safe_int(ent.get("sentence_id"))

        if not text:
            continue

        if not include_unknown_labels and ent_type not in ENTITY_TYPE_SET:
            continue

        cleaned.append(
            {
                "entity_id": ent_id,
                "text": text,
                "type": ent_type,
                "sentence_id": sent_id,
            }
        )

    return cleaned


def clean_relations(
    obj: Dict[str, List[Dict[str, Any]]],
    include_unknown_labels: bool = False,
) -> List[Dict[str, Any]]:
    cleaned = []

    for rel in obj.get("relations", []):
        if not isinstance(rel, dict):
            continue

        rel_id = str(rel.get("relation_id", "")).strip()
        rel_type = normalize_space(rel.get("type", ""))
        arg1 = str(rel.get("arg1", "")).strip()
        arg2 = str(rel.get("arg2", "")).strip()
        sent_id = safe_int(rel.get("sentence_id"))

        if not arg1 or not arg2:
            continue

        if not include_unknown_labels and rel_type not in RELATION_TYPE_SET:
            continue

        cleaned.append(
            {
                "relation_id": rel_id,
                "type": rel_type,
                "arg1": arg1,
                "arg2": arg2,
                "sentence_id": sent_id,
            }
        )

    return cleaned


def build_entity_id_map(entities: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {
        ent["entity_id"]: ent
        for ent in entities
        if ent.get("entity_id")
    }


# ============================================================================
# Entity matching
# ============================================================================

def pair_entities(
    gt_entities: List[Dict[str, Any]],
    pred_entities: List[Dict[str, Any]],
    jaccard_threshold: float = 0.6,
) -> Tuple[List[Tuple[int, int, float, str]], List[int], List[int]]:
    """
    One-to-one entity span alignment.

    Phase 1: exact normalized text match first.
    Phase 2: fuzzy token-Jaccard / containment matching on remaining entities.

    Returns:
        pairs: list of (gt_idx, pred_idx, score, match_type)
        unmatched_gt: remaining GT indices
        unmatched_pred: remaining predicted indices
    """
    used_gt = set()
    used_pred = set()
    pairs: List[Tuple[int, int, float, str]] = []

    # Exact normalized span matches first
    gt_by_text: Dict[str, List[int]] = {}
    for i, ent in enumerate(gt_entities):
        key = norm_text(ent["text"])
        gt_by_text.setdefault(key, []).append(i)

    for j, ent in enumerate(pred_entities):
        key = norm_text(ent["text"])
        for i in gt_by_text.get(key, []):
            if i not in used_gt and j not in used_pred:
                pairs.append((i, j, 1.0, "exact"))
                used_gt.add(i)
                used_pred.add(j)
                break

    # Fuzzy matches on remaining entities
    candidates = []
    for i, g in enumerate(gt_entities):
        if i in used_gt:
            continue

        for j, p in enumerate(pred_entities):
            if j in used_pred:
                continue

            score = containment_or_jaccard(g["text"], p["text"])
            if score >= jaccard_threshold:
                candidates.append(
                    (
                        score,
                        norm_text(g["text"]) == norm_text(p["text"]),
                        g["type"] == p["type"],
                        g.get("sentence_id") == p.get("sentence_id"),
                        i,
                        j,
                    )
                )

    candidates.sort(reverse=True)

    for score, exact_flag, same_type, same_sentence, i, j in candidates:
        if i in used_gt or j in used_pred:
            continue

        pairs.append((i, j, float(score), "fuzzy"))
        used_gt.add(i)
        used_pred.add(j)

    unmatched_gt = [i for i in range(len(gt_entities)) if i not in used_gt]
    unmatched_pred = [j for j in range(len(pred_entities)) if j not in used_pred]

    return pairs, unmatched_gt, unmatched_pred


# ============================================================================
# Relation matching
# ============================================================================

def relation_to_text_item(
    rel: Dict[str, Any],
    entity_map: Dict[str, Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    Convert a relation with arg IDs into a relation with endpoint texts.
    Returns None if endpoints cannot be resolved.
    """
    arg1_id = rel.get("arg1")
    arg2_id = rel.get("arg2")

    if arg1_id not in entity_map or arg2_id not in entity_map:
        return None

    ent1 = entity_map[arg1_id]
    ent2 = entity_map[arg2_id]

    arg1_text = normalize_space(ent1.get("text", ""))
    arg2_text = normalize_space(ent2.get("text", ""))

    if not arg1_text or not arg2_text:
        return None

    return {
        "type": rel.get("type", ""),
        "arg1_text": arg1_text,
        "arg2_text": arg2_text,
        "arg1_sentence_id": ent1.get("sentence_id"),
        "arg2_sentence_id": ent2.get("sentence_id"),
        "relation_sentence_id": rel.get("sentence_id"),
        "relation_id": rel.get("relation_id", ""),
        "arg1": arg1_id,
        "arg2": arg2_id,
    }


def relation_endpoint_score(
    gt_rel: Dict[str, Any],
    pred_rel: Dict[str, Any],
    jaccard_threshold: float = 0.6,
) -> float:
    """
    Score relation endpoint match by endpoint text similarity.
    """
    s1 = containment_or_jaccard(gt_rel["arg1_text"], pred_rel["arg1_text"])
    s2 = containment_or_jaccard(gt_rel["arg2_text"], pred_rel["arg2_text"])

    return (s1 + s2) / 2.0 if min(s1, s2) >= jaccard_threshold else 0.0


def relation_sentence_compatible(
    gt_rel: Dict[str, Any],
    pred_rel: Dict[str, Any],
    use_relation_sentence_id: bool = False,
) -> bool:
    """
    Optional sentence-id matching.
    Disabled by default because generated outputs may not reliably preserve sentence IDs.
    """
    if not use_relation_sentence_id:
        return True

    gt_sent = gt_rel.get("relation_sentence_id")
    pred_sent = pred_rel.get("relation_sentence_id")

    if gt_sent is None or pred_sent is None:
        return True

    return gt_sent == pred_sent


def relation_match_score(
    gt_rel: Dict[str, Any],
    pred_rel: Dict[str, Any],
    jaccard_threshold: float = 0.6,
    use_relation_sentence_id: bool = False,
) -> Tuple[float, str]:
    """
    Match relation endpoints regardless of relation type.
    This is used to decide whether two relations refer to the same endpoint pair.
    Then the type mismatch becomes the relation-type confusion.

    Returns:
        (score, orientation)
        orientation is "direct" or "none"
    """
    if not relation_sentence_compatible(gt_rel, pred_rel, use_relation_sentence_id):
        return 0.0, "none"

    direct = relation_endpoint_score(gt_rel, pred_rel, jaccard_threshold=jaccard_threshold)
    if direct >= jaccard_threshold:
        return direct, "direct"

    return 0.0, "none"


def pair_relations(
    gt_rel_items: List[Dict[str, Any]],
    pred_rel_items: List[Dict[str, Any]],
    jaccard_threshold: float = 0.6,
    use_relation_sentence_id: bool = False,
) -> Tuple[List[Tuple[int, int, float, str]], List[int], List[int]]:
    """
    One-to-one relation alignment by resolved endpoint texts.

    This matches relation endpoint pairs first; then type differences are counted
    as relation-type confusions.
    """
    used_gt = set()
    used_pred = set()
    pairs: List[Tuple[int, int, float, str]] = []

    candidates = []

    for i, g in enumerate(gt_rel_items):
        for j, p in enumerate(pred_rel_items):
            score, orientation = relation_match_score(
                g,
                p,
                jaccard_threshold=jaccard_threshold,
                use_relation_sentence_id=use_relation_sentence_id,
            )

            if score >= jaccard_threshold:
                candidates.append(
                    (
                        score,
                        g["type"] == p["type"],
                        orientation == "direct",
                        i,
                        j,
                        orientation,
                    )
                )

    candidates.sort(reverse=True)

    for score, same_type, direct_flag, i, j, orientation in candidates:
        if i in used_gt or j in used_pred:
            continue

        pairs.append((i, j, float(score), orientation))
        used_gt.add(i)
        used_pred.add(j)

    unmatched_gt = [i for i in range(len(gt_rel_items)) if i not in used_gt]
    unmatched_pred = [j for j in range(len(pred_rel_items)) if j not in used_pred]

    return pairs, unmatched_gt, unmatched_pred


# ============================================================================
# CSV writers
# ============================================================================

def write_long_csv(
    path: Path,
    rows: List[Tuple[str, str, int]],
    header_a: str,
    header_b: str,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([header_a, header_b, "Count"])
        for a, b, count in rows:
            writer.writerow([a, b, count])


def write_dense_csv(
    path: Path,
    counts: Counter,
    row_header: str,
    col_header: str,
) -> None:
    row_labels = sorted({k[0] for k in counts})
    col_labels = sorted({k[1] for k in counts})

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([row_header] + col_labels)
        for r in row_labels:
            writer.writerow([r] + [counts.get((r, c), 0) for c in col_labels])


# ============================================================================
# Main confusion builder
# ============================================================================

def build_confusions(
    samples_jsonl: Path,
    out_dir: Path,
    jaccard_threshold: float = 0.6,
    include_correct_pairs: bool = False,
    include_unknown_labels: bool = False,
    use_relation_sentence_id: bool = False,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    entity_type_confusions = Counter()
    relation_type_confusions = Counter()

    stats = Counter()
    example_rows = []

    with samples_jsonl.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            stats["records_total"] += 1

            try:
                record = json.loads(line)

                gt_obj = get_scierc_obj_from_answer(get_gold_answer_text(record))
                pred_obj = get_scierc_obj_from_answer(get_pred_response(record))

                gt_entities = clean_entities(
                    gt_obj,
                    include_unknown_labels=include_unknown_labels,
                )
                pred_entities = clean_entities(
                    pred_obj,
                    include_unknown_labels=include_unknown_labels,
                )

                gt_relations = clean_relations(
                    gt_obj,
                    include_unknown_labels=include_unknown_labels,
                )
                pred_relations = clean_relations(
                    pred_obj,
                    include_unknown_labels=include_unknown_labels,
                )

            except Exception as e:
                stats["records_parse_failed"] += 1
                if len(example_rows) < 100:
                    example_rows.append(
                        {
                            "line_num": line_num,
                            "type": "parse_error",
                            "detail": str(e),
                        }
                    )
                continue

            stats["gt_entities_total"] += len(gt_entities)
            stats["pred_entities_total"] += len(pred_entities)
            stats["gt_relations_total"] += len(gt_relations)
            stats["pred_relations_total"] += len(pred_relations)

            # ------------------------------------------------------------
            # Entity-type confusions
            # ------------------------------------------------------------
            ent_pairs, unmatched_gt_ent, unmatched_pred_ent = pair_entities(
                gt_entities,
                pred_entities,
                jaccard_threshold=jaccard_threshold,
            )

            stats["matched_entity_pairs_total"] += len(ent_pairs)
            stats["unmatched_gt_entities_total"] += len(unmatched_gt_ent)
            stats["unmatched_pred_entities_total"] += len(unmatched_pred_ent)

            for gt_idx, pred_idx, score, match_type in ent_pairs:
                g = gt_entities[gt_idx]
                p = pred_entities[pred_idx]

                gt_type = g["type"]
                pred_type = p["type"]

                if include_correct_pairs or gt_type != pred_type:
                    entity_type_confusions[(gt_type, pred_type)] += 1

                if len(example_rows) < 100 and gt_type != pred_type:
                    example_rows.append(
                        {
                            "line_num": line_num,
                            "kind": "entity_type_confusion",
                            "match_type": match_type,
                            "score": round(score, 4),
                            "gt_entity_type": gt_type,
                            "pred_entity_type": pred_type,
                            "gt_text": g["text"],
                            "pred_text": p["text"],
                            "gt_sentence_id": g.get("sentence_id"),
                            "pred_sentence_id": p.get("sentence_id"),
                        }
                    )

            # ------------------------------------------------------------
            # Relation-type confusions
            # ------------------------------------------------------------
            gt_entity_map = build_entity_id_map(gt_entities)
            pred_entity_map = build_entity_id_map(pred_entities)

            gt_rel_items = []
            for rel in gt_relations:
                item = relation_to_text_item(rel, gt_entity_map)
                if item is not None:
                    gt_rel_items.append(item)
                else:
                    stats["gt_relations_unresolved_args"] += 1

            pred_rel_items = []
            for rel in pred_relations:
                item = relation_to_text_item(rel, pred_entity_map)
                if item is not None:
                    pred_rel_items.append(item)
                else:
                    stats["pred_relations_unresolved_args"] += 1

            rel_pairs, unmatched_gt_rel, unmatched_pred_rel = pair_relations(
                gt_rel_items,
                pred_rel_items,
                jaccard_threshold=jaccard_threshold,
                use_relation_sentence_id=use_relation_sentence_id,
            )

            stats["matched_relation_pairs_total"] += len(rel_pairs)
            stats["unmatched_gt_relations_total"] += len(unmatched_gt_rel)
            stats["unmatched_pred_relations_total"] += len(unmatched_pred_rel)

            for gt_idx, pred_idx, score, orientation in rel_pairs:
                g = gt_rel_items[gt_idx]
                p = pred_rel_items[pred_idx]

                gt_type = g["type"]
                pred_type = p["type"]

                if include_correct_pairs or gt_type != pred_type:
                    relation_type_confusions[(gt_type, pred_type)] += 1

                if len(example_rows) < 100 and gt_type != pred_type:
                    example_rows.append(
                        {
                            "line_num": line_num,
                            "kind": "relation_type_confusion",
                            "orientation": orientation,
                            "score": round(score, 4),
                            "gt_relation_type": gt_type,
                            "pred_relation_type": pred_type,
                            "gt_arg1_text": g["arg1_text"],
                            "gt_arg2_text": g["arg2_text"],
                            "pred_arg1_text": p["arg1_text"],
                            "pred_arg2_text": p["arg2_text"],
                            "gt_relation_sentence_id": g.get("relation_sentence_id"),
                            "pred_relation_sentence_id": p.get("relation_sentence_id"),
                        }
                    )

    # Sort rows by descending count, then lexicographically
    entity_rows = sorted(
        [
            (gt, pred, count)
            for (gt, pred), count in entity_type_confusions.items()
        ],
        key=lambda x: (-x[2], x[0], x[1]),
    )

    relation_rows = sorted(
        [
            (gt, pred, count)
            for (gt, pred), count in relation_type_confusions.items()
        ],
        key=lambda x: (-x[2], x[0], x[1]),
    )

    # Long CSVs consumed by SCIERC DPO data prep
    write_long_csv(
        out_dir / "entity_type_confusion.csv",
        entity_rows,
        "Ground_Truth_EntityType",
        "Predicted_EntityType",
    )

    write_long_csv(
        out_dir / "relation_type_confusion.csv",
        relation_rows,
        "Ground_Truth_RelationType",
        "Predicted_RelationType",
    )

    # Compatibility aliases if your next-stage script expects old names.
    # These are intentionally entity/relation CSVs with old generic headers.
    write_long_csv(
        out_dir / "code_confusion.csv",
        entity_rows,
        "Ground_Truth_Code",
        "Predicted_Code",
    )

    write_long_csv(
        out_dir / "subcode_confusion.csv",
        relation_rows,
        "Ground_Truth_SubCode",
        "Predicted_SubCode",
    )

    # Dense matrices for inspection
    write_dense_csv(
        out_dir / "entity_type_confusion_dense.csv",
        entity_type_confusions,
        "Ground_Truth_EntityType",
        "Predicted_EntityType",
    )

    write_dense_csv(
        out_dir / "relation_type_confusion_dense.csv",
        relation_type_confusions,
        "Ground_Truth_RelationType",
        "Predicted_RelationType",
    )

    # Save sample mismatch examples
    with (out_dir / "matched_mismatch_examples.json").open("w", encoding="utf-8") as f:
        json.dump(example_rows, f, indent=2, ensure_ascii=False)

    summary = {
        **stats,
        "jaccard_threshold": jaccard_threshold,
        "include_correct_pairs": include_correct_pairs,
        "include_unknown_labels": include_unknown_labels,
        "relation_argument_order": "strict",
        "use_relation_sentence_id": use_relation_sentence_id,
        "unique_entity_type_pairs": len(entity_rows),
        "unique_relation_type_pairs": len(relation_rows),
        "top_10_entity_type_confusions": entity_rows[:10],
        "top_10_relation_type_confusions": relation_rows[:10],
        "output_files": {
            "entity_type_confusion": str(out_dir / "entity_type_confusion.csv"),
            "relation_type_confusion": str(out_dir / "relation_type_confusion.csv"),
            "entity_type_confusion_dense": str(out_dir / "entity_type_confusion_dense.csv"),
            "relation_type_confusion_dense": str(out_dir / "relation_type_confusion_dense.csv"),
            "compat_code_confusion": str(out_dir / "code_confusion.csv"),
            "compat_subcode_confusion": str(out_dir / "subcode_confusion.csv"),
            "examples": str(out_dir / "matched_mismatch_examples.json"),
            "summary": str(out_dir / "confusion_summary.json"),
        },
    }

    with (out_dir / "confusion_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--samples_jsonl",
        type=Path,
        required=True,
        help="Path to lm_eval/vLLM samples jsonl file.",
    )

    parser.add_argument(
        "--out_dir",
        type=Path,
        required=True,
        help="Directory where confusion CSVs and summary files will be written.",
    )

    parser.add_argument(
        "--jaccard_threshold",
        type=float,
        default=0.6,
        help="Threshold for entity text and relation endpoint matching.",
    )

    parser.add_argument(
        "--include_correct_pairs",
        action="store_true",
        help="Also include diagonal/correct pairs in the output CSVs.",
    )

    parser.add_argument(
        "--include_unknown_labels",
        action="store_true",
        help="Keep labels outside the known SCIERC entity/relation type sets.",
    )

    parser.add_argument(
        "--use_relation_sentence_id",
        action="store_true",
        help="Require relation sentence_id compatibility when matching relations.",
    )

    args = parser.parse_args()

    from sample_provenance import verify_validation_samples
    from tabpo.data import sha256
    data_dir = Path(__file__).resolve().parents[1] / "data/scierc"
    provenance = verify_validation_samples(args.samples_jsonl, data_dir)
    if not 0 < args.jaccard_threshold <= 1:
        parser.error("jaccard_threshold must be in (0,1]")

    summary = build_confusions(
        samples_jsonl=args.samples_jsonl,
        out_dir=args.out_dir,
        jaccard_threshold=args.jaccard_threshold,
        include_correct_pairs=args.include_correct_pairs,
        include_unknown_labels=args.include_unknown_labels,
        use_relation_sentence_id=args.use_relation_sentence_id,
    )

    summary.update(provenance)
    summary["csv_sha256"] = {name: sha256((args.out_dir / name).read_bytes())
                             for name in ("entity_type_confusion.csv", "relation_type_confusion.csv")}
    (args.out_dir / "confusion_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
