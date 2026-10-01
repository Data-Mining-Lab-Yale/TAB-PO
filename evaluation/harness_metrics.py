"""Scalar micro-F1 adapters for the SciERC metrics defined in Appendix F."""
from __future__ import annotations
import json
import scierc_utils as u


def counts(gold, prediction):
    g, p = u.safe_json_loads(gold), u.safe_json_loads(prediction)
    gt, pt = set(u._entity_type_list(g)), set(u._entity_type_list(p))
    result = {"entity_f1": (len(gt & pt), len(pt - gt), len(gt - pt))}
    gs, ps = u._span_list(g), u._span_list(p)
    used = set()
    tp = 0
    for pred in ps:
        for j, true in enumerate(gs):
            if j not in used and u._span_match(true, pred):
                used.add(j)
                tp += 1
                break
    result["span_f1"] = (tp, len(ps) - tp, len(gs) - tp)
    result["relation_f1"] = u._greedy_match_by_score(u._relation_items(g), u._relation_items(p),
                                                   u._relation_match_score)
    result["coreference_f1"] = u._greedy_match_by_score(u._coref_pair_items(g), u._coref_pair_items(p), u._coref_pair_match_score)
    try:
        obj = json.loads(u._strip_code_fences(prediction))
        valid = isinstance(obj, dict) and all(isinstance(obj.get(k), list) for k in ("entities", "relations", "coreference_clusters"))
    except (TypeError, ValueError):
        valid = False
    result["json_parse_rate"] = float(valid)
    return result


def process_results(doc, results):
    return counts(doc["answer"], results[0])


def micro_f1(items):
    tp, fp, fn = (sum(x[i] for x in items) for i in (0, 1, 2))
    return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0


def mean(items):
    return sum(items) / len(items) if items else 0.0
