import re
import json
import ast
import html
from itertools import combinations

# SciERC metrics follow Appendix F. Relation arguments are ordered;
# coreference mention identity includes the sentence ID.


ENTITY_TYPE_SET = [
    "Task",
    "Method",
    "Metric",
    "Material",
    "OtherScientificTerm",
    "Generic",
]

RELATION_SET = [
    "Used-for",
    "Evaluate-for",
    "Feature-of",
    "Part-of",
    "Hyponym-of",
    "Compare",
    "Conjunction",
]

JACCARD_THRESHOLD = 0.6


def _normalize_whitespace(text):
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def _norm_text(text):
    return _normalize_whitespace(text).lower()


def _strip_code_fences(text):
    text = str(text).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _extract_first_json_object(text):
    text = _strip_code_fences(text)
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


def _normalize_scierc_obj(obj):
    if not isinstance(obj, dict):
        return {"entities": [], "relations": [], "coreference_clusters": []}

    entities = obj.get("entities", [])
    relations = obj.get("relations", [])
    coref = obj.get("coreference_clusters", [])

    entities = entities if isinstance(entities, list) else []
    relations = relations if isinstance(relations, list) else []
    coref = coref if isinstance(coref, list) else []

    return {
        "entities": [e for e in entities if isinstance(e, dict)],
        "relations": [r for r in relations if isinstance(r, dict)],
        "coreference_clusters": coref,
    }


def safe_json_loads(input_str):
    if isinstance(input_str, dict):
        return _normalize_scierc_obj(input_str)

    if input_str is None:
        return {"entities": [], "relations": [], "coreference_clusters": []}

    text = _strip_code_fences(str(input_str))

    # 1) direct JSON
    try:
        return _normalize_scierc_obj(json.loads(text))
    except Exception:
        pass

    # 2) first balanced JSON object inside the text
    candidate = _extract_first_json_object(text)
    if candidate is not None:
        try:
            return _normalize_scierc_obj(json.loads(candidate))
        except Exception:
            pass

    # 3) HTML-unescape + normalized quotes + JSON
    text2 = html.unescape(text)
    text2 = text2.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    candidate2 = _extract_first_json_object(text2)
    if candidate2 is not None:
        try:
            return _normalize_scierc_obj(json.loads(candidate2))
        except Exception:
            pass

    # 4) Python literal fallback
    if candidate2 is not None:
        try:
            return _normalize_scierc_obj(ast.literal_eval(candidate2))
        except Exception:
            pass

    # 5) Salvage entity/relation/coreference lists if the full object is malformed
    salvaged = {"entities": [], "relations": [], "coreference_clusters": []}

    entity_pattern = re.compile(
        r'\{\s*"entity_id"\s*:\s*"([^"]+)"\s*,\s*"text"\s*:\s*"([^"]*)"\s*,\s*"type"\s*:\s*"([^"]+)"\s*,\s*"sentence_id"\s*:\s*(-?\d+)\s*\}'
    )
    relation_pattern = re.compile(
        r'\{\s*"relation_id"\s*:\s*"([^"]+)"\s*,\s*"type"\s*:\s*"([^"]+)"\s*,\s*"arg1"\s*:\s*"([^"]+)"\s*,\s*"arg2"\s*:\s*"([^"]+)"\s*,\s*"sentence_id"\s*:\s*(-?\d+)\s*\}'
    )
    cluster_pattern = re.compile(
        r'\{\s*"cluster_id"\s*:\s*"([^"]+)"\s*,\s*"members"\s*:\s*\[([^\]]*)\]\s*\}'
    )

    for m in entity_pattern.finditer(text2):
        salvaged["entities"].append(
            {
                "entity_id": m.group(1),
                "text": m.group(2),
                "type": m.group(3),
                "sentence_id": int(m.group(4)),
            }
        )

    for m in relation_pattern.finditer(text2):
        salvaged["relations"].append(
            {
                "relation_id": m.group(1),
                "type": m.group(2),
                "arg1": m.group(3),
                "arg2": m.group(4),
                "sentence_id": int(m.group(5)),
            }
        )

    for m in cluster_pattern.finditer(text2):
        raw_members = [x.strip() for x in m.group(2).split(",") if x.strip()]
        members = [x.strip().strip('"').strip("'") for x in raw_members]
        salvaged["coreference_clusters"].append(
            {
                "cluster_id": m.group(1),
                "members": members,
            }
        )

    return _normalize_scierc_obj(salvaged)


def evaluate_scierc(items):
    return items


def _entity_type_list(obj):
    return [
        ent.get("type")
        for ent in obj.get("entities", [])
        if isinstance(ent.get("type"), str) and ent.get("type") in ENTITY_TYPE_SET
    ]


def _span_list(obj):
    return [
        _normalize_whitespace(ent.get("text", ""))
        for ent in obj.get("entities", [])
        if _normalize_whitespace(ent.get("text", ""))
    ]


def _entity_map(obj):
    entity_map = {}
    for ent in obj.get("entities", []):
        if not isinstance(ent, dict):
            continue

        ent_id = ent.get("entity_id")
        if isinstance(ent_id, (str, int)):
            entity_map[ent_id] = ent
            entity_map[str(ent_id)] = ent

    return entity_map


def _safe_int(x):
    try:
        return int(x)
    except Exception:
        return None


def _span_similarity(phrase1, phrase2):
    phrase1 = _normalize_whitespace(phrase1)
    phrase2 = _normalize_whitespace(phrase2)
    if not phrase1 or not phrase2:
        return 0.0
    if is_full_containment_match(phrase1, phrase2) or is_full_containment_match(phrase2, phrase1):
        return 1.0
    return calculate_jaccard_for_tokens(phrase1, phrase2)


def _span_match(phrase1, phrase2, jaccard_threshold=JACCARD_THRESHOLD):
    sim = _span_similarity(phrase1, phrase2)
    return sim >= jaccard_threshold


def _relation_items(obj):
    entity_map = _entity_map(obj)
    items = []

    for rel in obj.get("relations", []):
        rel_type = rel.get("type")
        arg1 = rel.get("arg1")
        arg2 = rel.get("arg2")

        if not isinstance(rel_type, str) or rel_type not in RELATION_SET:
            continue
        if not isinstance(arg1, (str, int)) or not isinstance(arg2, (str, int)):
            continue
        if arg1 not in entity_map or arg2 not in entity_map:
            continue

        ent1 = entity_map[arg1]
        ent2 = entity_map[arg2]
        arg1_text = _normalize_whitespace(ent1.get("text", ""))
        arg2_text = _normalize_whitespace(ent2.get("text", ""))
        if not arg1_text or not arg2_text:
            continue

        items.append(
            {
                "type": rel_type,
                "arg1_text": arg1_text,
                "arg2_text": arg2_text,
                "arg1_sentence_id": _safe_int(ent1.get("sentence_id")),
                "arg2_sentence_id": _safe_int(ent2.get("sentence_id")),
                "relation_sentence_id": _safe_int(rel.get("sentence_id")),
            }
        )

    return items


def _ordered_relation_similarity(t_arg1, t_arg2, p_arg1, p_arg2):
    sim1 = _span_similarity(t_arg1, p_arg1)
    sim2 = _span_similarity(t_arg2, p_arg2)
    if sim1 >= JACCARD_THRESHOLD and sim2 >= JACCARD_THRESHOLD:
        return (sim1 + sim2) / 2.0
    return 0.0


def _relation_match_score(true_rel, pred_rel):
    if true_rel["type"] != pred_rel["type"]:
        return 0.0
    return _ordered_relation_similarity(
        true_rel["arg1_text"], true_rel["arg2_text"],
        pred_rel["arg1_text"], pred_rel["arg2_text"]
    )

def _greedy_match_by_score(true_items, pred_items, match_score_fn):
    candidate_pairs = []
    for t_idx, t_item in enumerate(true_items):
        for p_idx, p_item in enumerate(pred_items):
            score = match_score_fn(t_item, p_item)
            if score > 0:
                candidate_pairs.append((score, t_idx, p_idx))

    candidate_pairs.sort(reverse=True, key=lambda x: x[0])

    matched_true = set()
    matched_pred = set()
    true_positive = 0

    for score, t_idx, p_idx in candidate_pairs:
        if t_idx in matched_true or p_idx in matched_pred:
            continue
        matched_true.add(t_idx)
        matched_pred.add(p_idx)
        true_positive += 1

    false_positive = len(pred_items) - len(matched_pred)
    false_negative = len(true_items) - len(matched_true)

    return true_positive, false_positive, false_negative

def _entity_to_mention(ent):
    if not isinstance(ent, dict):
        return None

    text = _normalize_whitespace(ent.get("text", ""))
    if not text:
        return None

    return {
        "text": text,
        "sentence_id": _safe_int(ent.get("sentence_id")),
    }


def _lookup_entity(entity_map, member_id):
    if member_id is None:
        return None

    try:
        ent = entity_map.get(member_id)
    except TypeError:
        ent = None

    if ent is not None:
        return ent

    return entity_map.get(str(member_id))


def _member_to_mentions(member, entity_map):
    # Case 1: member is a dict, e.g. {"entity_id": "E1"} or {"text": "..."}
    if isinstance(member, dict):
        for key in ("entity_id", "id", "member_id", "mention_id"):
            if key in member:
                ent = _lookup_entity(entity_map, member.get(key))
                mention = _entity_to_mention(ent)
                if mention is not None:
                    return [mention]

        # Direct mention object fallback.
        text = _normalize_whitespace(
            member.get("text", "")
            or member.get("mention", "")
            or member.get("span", "")
        )
        if text:
            return [
                {
                    "text": text,
                    "sentence_id": _safe_int(member.get("sentence_id")),
                }
            ]

        return []

    # Case 2: member is a nested list, e.g. [{"entity_id": "E1"}, {"entity_id": "E2"}]
    if isinstance(member, (list, tuple)):
        mentions = []
        for sub_member in member:
            mentions.extend(_member_to_mentions(sub_member, entity_map))
        return mentions

    # Case 3: member is normally a string entity id, e.g. "E1"
    ent = _lookup_entity(entity_map, member)
    mention = _entity_to_mention(ent)
    if mention is not None:
        return [mention]

    # Case 4: model may output raw mention text instead of entity ids.
    # Avoid treating obvious entity-id-like strings such as E1/T12 as text.
    text = _normalize_whitespace(member)
    if text and not re.fullmatch(r"[A-Za-z]*\d+", text):
        return [{"text": text, "sentence_id": None}]

    return []


def _cluster_to_raw_members(cluster):
    # Expected format:
    # {"cluster_id": "C1", "members": ["E1", "E2"]}
    if isinstance(cluster, dict):
        raw_members = cluster.get("members", [])
    # Common malformed-but-valid model output:
    # ["E1", "E2"]
    elif isinstance(cluster, (list, tuple)):
        raw_members = cluster
    else:
        raw_members = []

    if raw_members is None:
        return []

    if not isinstance(raw_members, list):
        raw_members = [raw_members]

    return raw_members

def _coref_pair_items(obj):
    entity_map = _entity_map(obj)
    pair_items = []

    coref_clusters = obj.get("coreference_clusters", [])
    if not isinstance(coref_clusters, list):
        return pair_items

    for cluster in coref_clusters:
        members = []

        for raw_member in _cluster_to_raw_members(cluster):
            members.extend(_member_to_mentions(raw_member, entity_map))

        # Keep repeated surface forms as separate mentions; do not collapse with set().
        for m1, m2 in combinations(range(len(members)), 2):
            pair_items.append(
                {
                    "m1": members[m1],
                    "m2": members[m2],
                }
            )

    return pair_items


def _mention_match_score(true_mention, pred_mention):
    text_score = _span_similarity(true_mention["text"], pred_mention["text"])
    if text_score < JACCARD_THRESHOLD:
        return 0.0

    true_sid = true_mention.get("sentence_id")
    pred_sid = pred_mention.get("sentence_id")
    if true_sid is None or pred_sid is None or true_sid != pred_sid:
        return 0.0

    return text_score


def _coref_pair_match_score(true_pair, pred_pair):
    direct1 = _mention_match_score(true_pair["m1"], pred_pair["m1"])
    direct2 = _mention_match_score(true_pair["m2"], pred_pair["m2"])
    direct = (direct1 + direct2) / 2.0 if direct1 > 0 and direct2 > 0 else 0.0

    reverse1 = _mention_match_score(true_pair["m1"], pred_pair["m2"])
    reverse2 = _mention_match_score(true_pair["m2"], pred_pair["m1"])
    reverse = (reverse1 + reverse2) / 2.0 if reverse1 > 0 and reverse2 > 0 else 0.0

    return max(direct, reverse)


def evaluate_scierc_agg(items):
    true_answer = [item[0] for item in items]
    pred_answer = [item[1] for item in items]

    true_entity_types = []
    pred_entity_types = []

    true_spans = []
    pred_spans = []

    true_relation_items = []
    pred_relation_items = []

    true_coref_items = []
    pred_coref_items = []

    for t_answer, p_answer in zip(true_answer, pred_answer):
        t_obj = safe_json_loads(t_answer)
        p_obj = safe_json_loads(p_answer)

        true_entity_types.append(_entity_type_list(t_obj))
        pred_entity_types.append(_entity_type_list(p_obj))

        true_spans.append(_span_list(t_obj))
        pred_spans.append(_span_list(p_obj))

        true_relation_items.append(_relation_items(t_obj))
        pred_relation_items.append(_relation_items(p_obj))

        true_coref_items.append(_coref_pair_items(t_obj))
        pred_coref_items.append(_coref_pair_items(p_obj))

    precision_entity, recall_entity, f1_entity = calculate_entity_type(true_entity_types, pred_entity_types)
    precision_relation, recall_relation, f1_relation = calculate_relation_prf(true_relation_items, pred_relation_items)
    precision_span, recall_span, f1_span = relaxed_match_evaluation_with_full_containment(
        true_spans, pred_spans, jaccard_threshold=JACCARD_THRESHOLD
    )
    precision_coref, recall_coref, f1_coref = calculate_coref_prf(true_coref_items, pred_coref_items)

    return {
        "entity": {"P": round(precision_entity, 4), "R": round(recall_entity, 4), "f1": round(f1_entity, 4)},
        "relation": {"P": round(precision_relation, 4), "R": round(recall_relation, 4), "f1": round(f1_relation, 4)},
        "span": {"P": round(precision_span, 4), "R": round(recall_span, 4), "f1": round(f1_span, 4)},
        "coreference": {"P": round(precision_coref, 4), "R": round(recall_coref, 4), "f1": round(f1_coref, 4)},
    }


def calculate_entity_type(true_entity_types, pred_entity_types):
    return calculate_set_prf([set(x) & set(ENTITY_TYPE_SET) for x in true_entity_types],
                             [set(x) & set(ENTITY_TYPE_SET) for x in pred_entity_types])


def calculate_relation_prf(true_relation_items, pred_relation_items):
    true_positive = 0
    false_positive = 0
    false_negative = 0

    for true_items, pred_items in zip(true_relation_items, pred_relation_items):
        tp, fp, fn = _greedy_match_by_score(true_items, pred_items, _relation_match_score)
        true_positive += tp
        false_positive += fp
        false_negative += fn

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1


def calculate_coref_prf(true_coref_items, pred_coref_items):
    true_positive = 0
    false_positive = 0
    false_negative = 0

    for true_items, pred_items in zip(true_coref_items, pred_coref_items):
        tp, fp, fn = _greedy_match_by_score(true_items, pred_items, _coref_pair_match_score)
        true_positive += tp
        false_positive += fp
        false_negative += fn

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1


def calculate_set_prf(true_sets, pred_sets):
    true_positive = 0
    false_positive = 0
    false_negative = 0

    for true_set, pred_set in zip(true_sets, pred_sets):
        true_positive += len(true_set & pred_set)
        false_positive += len(pred_set - true_set)
        false_negative += len(true_set - pred_set)

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1


def calculate_jaccard_for_tokens(phrase1, phrase2):
    set1 = set(phrase1.lower().split())
    set2 = set(phrase2.lower().split())

    intersection = set1.intersection(set2)
    union = set1.union(set2)

    if len(union) == 0:
        return 0
    return len(intersection) / len(union)


def is_full_containment_match(phrase1, phrase2):
    set1 = set(phrase1.lower().split())
    set2 = set(phrase2.lower().split())
    return set1.issubset(set2)


def relaxed_match_evaluation_with_full_containment(true_entities_list, pred_entities_list, jaccard_threshold=0.6):
    true_positive = 0
    false_positive = 0
    false_negative = 0

    for true_entities, pred_entities in zip(true_entities_list, pred_entities_list):
        matched_true = [False] * len(true_entities)
        matched_pred = [False] * len(pred_entities)

        for i, pred_entity in enumerate(pred_entities):
            for j, true_entity in enumerate(true_entities):
                if not matched_true[j] and not matched_pred[i]:
                    if is_full_containment_match(true_entity, pred_entity) or is_full_containment_match(pred_entity, true_entity):
                        true_positive += 1
                        matched_true[j] = True
                        matched_pred[i] = True
                    elif calculate_jaccard_for_tokens(pred_entity, true_entity) >= jaccard_threshold:
                        true_positive += 1
                        matched_true[j] = True
                        matched_pred[i] = True

        false_positive += matched_pred.count(False)
        false_negative += matched_true.count(False)

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1
