#!/usr/bin/env python3
"""
Token-weighted TAB-PO / DPO training for SCIERC.

Expected DPO dataset columns:
  - prompt
  - chosen
  - rejected
  - metadata optional

Expected chosen/rejected JSON format:
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

This is revised from the EPPC/PV trainer that weighted:
  Code / Sub-code / Span

For SCIERC, token weighting uses:
  entity type       -> entity["type"]
  relation type     -> relation["type"]
  entity text/span  -> entity["text"]
  relation args     -> relation["arg1"], relation["arg2"] IDs, optional

Core DPO objective:
  L_DPO = -log sigmoid(beta * [(log pi(y+|x)-log pi(y-|x))
                              -(log ref(y+|x)-log ref(y-|x))])

Adaptive barrier:
  Adds SFT-style NLL on chosen tokens whose current policy token probability
  is below policy_prob_threshold.

Important:
  - Prompt tokens are masked.
  - Only answer tokens contribute to chosen/rejected log-probabilities.
  - Per-token logp uses fused cross-entropy, avoiding full log_softmax materialization.
"""

from __future__ import annotations

import argparse
import ast
import html
import json
import os
import re
import math
import contextlib
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tabpo.prompting import completion_encoding, messages
from tabpo.objective import tabpo_loss

import torch
import torch.nn.functional as F
from datasets import load_from_disk
from peft import LoraConfig, TaskType, get_peft_model

try:
    from peft import get_peft_model_state_dict
except ImportError:
    from peft.utils import get_peft_model_state_dict
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed


# ============================================================================
# Config
# ============================================================================

@dataclass
class Config:
    model_name: str
    train_data_path: str
    valid_data_path: str
    output_dir: str

    max_length: int = 8192
    max_prompt_length: int = 7600

    # DPO parameters
    beta: float = 0.5
    policy_prob_threshold: float = 0.87
    sft_barrier_weight: float = 0.5

    # Memory/speed optimizations
    cache_reference_logps: bool = True
    ref_logp_cache_batch_size: int = 1
    concat_policy_forward: bool = True

    # SCIERC token weighting
    weight_entity_type_value: float = 1.0
    weight_relation_type_value: float = 1.0
    weight_entity_text_value: float = 1.0
    weight_relation_arg_value: float = 1.0
    weight_diff_tokens: float = 1.0

    # Weight normalization / length normalization
    normalize_by_weight_mass: bool = True
    enable_length_norm: bool = False
    length_norm_by: str = "tokens"  # "tokens" | "weight_mass"

    # Class balancing / example reweighting for SCIERC
    enable_class_balance: bool = False
    class_balance_strategy: str = "effective_num"  # "effective_num" | "inv_freq" | "inv_sqrt"
    class_balance_beta: float = 0.99
    class_balance_alpha: float = 1.0
    class_balance_combine: str = "max"  # "max" | "geom" | "mean"
    class_balance_max_weight: float = 3.0
    class_balance_use_entity_type: bool = True
    class_balance_use_relation_type: bool = True

    # Backward-compatible aliases; not used directly except CLI mapping.
    weight_code_value: float = 1.0
    weight_subcode_value: float = 1.0
    weight_span_value: float = 1.0

    # Training
    num_train_epochs: float = 3
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 2
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    logging_steps: int = 10
    save_steps: int = 50
    eval_steps: Optional[int] = None
    eval_strategy: str = "no"

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )

    # Runtime
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True
    save_total_limit: Optional[int] = None
    deepspeed_config: Optional[str] = None
    seed: int = 42


# ============================================================================
# Model path detection
# ============================================================================

def detect_model_source(model_identifier: str) -> Tuple[str, str]:
    path = Path(model_identifier)
    if path.exists() and path.is_dir():
        return "local", str(path.resolve())
    return "huggingface", model_identifier


# ============================================================================
# Robust SCIERC JSON parsing for token-value extraction
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


def _strip_code_fences(text: str) -> str:
    text = str(text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _extract_first_json_object(text: str) -> Optional[str]:
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


def _normalize_scierc_obj(obj: Any) -> Dict[str, List[Dict[str, Any]]]:
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


def safe_scierc_json_loads(text: Any) -> Dict[str, List[Dict[str, Any]]]:
    if isinstance(text, dict):
        return _normalize_scierc_obj(text)

    raw = _strip_code_fences(str(text or ""))

    try:
        return _normalize_scierc_obj(json.loads(raw))
    except Exception:
        pass

    candidate = _extract_first_json_object(raw)
    if candidate is not None:
        try:
            return _normalize_scierc_obj(json.loads(candidate))
        except Exception:
            pass

    raw2 = html.unescape(raw)
    raw2 = (
        raw2.replace("“", '"')
            .replace("”", '"')
            .replace("’", "'")
            .replace("‘", "'")
    )

    candidate2 = _extract_first_json_object(raw2)
    if candidate2 is not None:
        try:
            return _normalize_scierc_obj(json.loads(candidate2))
        except Exception:
            pass

        try:
            return _normalize_scierc_obj(ast.literal_eval(candidate2))
        except Exception:
            pass

    return {
        "entities": [],
        "relations": [],
        "coreference_clusters": [],
    }


# Fallback regexes if JSON parsing fails or partial values are still useful.
_ENTITY_TYPE_FIELD_RE = re.compile(
    r'\{\s*"entity_id"\s*:\s*"[^"]*"\s*,\s*"text"\s*:\s*"[^"]*"\s*,\s*"type"\s*:\s*"([^"]+)"',
    flags=re.DOTALL,
)
_ENTITY_TEXT_RE = re.compile(r'"text"\s*:\s*"([^"]+)"')
_RELATION_TYPE_FIELD_RE = re.compile(
    r'\{\s*"relation_id"\s*:\s*"[^"]*"\s*,\s*"type"\s*:\s*"([^"]+)"',
    flags=re.DOTALL,
)
_ARG_RE = re.compile(r'"arg[12]"\s*:\s*"([^"]+)"')
_MEMBERS_RE = re.compile(r'"members"\s*:\s*\[([^\]]*)\]')


def _uniq(xs: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in xs:
        x = str(x)
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def extract_values_from_jsonish(text: str) -> Dict[str, List[str]]:
    """
    Extract SCIERC values for token weighting.

    Returns:
      {
        "entity_type": [...],
        "relation_type": [...],
        "entity_text": [...],
        "relation_arg": [...],
        "coref_member": [...]
      }
    """
    obj = safe_scierc_json_loads(text)

    entity_types: List[str] = []
    relation_types: List[str] = []
    entity_texts: List[str] = []
    relation_args: List[str] = []
    coref_members: List[str] = []

    for ent in obj.get("entities", []):
        if not isinstance(ent, dict):
            continue

        ent_type = str(ent.get("type", "")).strip()
        ent_text = str(ent.get("text", "")).strip()
        ent_id = str(ent.get("entity_id", "")).strip()

        if ent_type:
            entity_types.append(ent_type)
        if ent_text:
            entity_texts.append(ent_text)
        if ent_id:
            relation_args.append(ent_id)

    for rel in obj.get("relations", []):
        if not isinstance(rel, dict):
            continue

        rel_type = str(rel.get("type", "")).strip()
        arg1 = str(rel.get("arg1", "")).strip()
        arg2 = str(rel.get("arg2", "")).strip()

        if rel_type:
            relation_types.append(rel_type)
        if arg1:
            relation_args.append(arg1)
        if arg2:
            relation_args.append(arg2)

    for clus in obj.get("coreference_clusters", []):
        if not isinstance(clus, dict):
            continue
        members = clus.get("members", [])
        if isinstance(members, list):
            for m in members:
                m = str(m).strip()
                if m:
                    coref_members.append(m)

    # Regex fallback can recover values even if JSON parser returned empty lists.
    if not entity_types:
        entity_types.extend(_ENTITY_TYPE_FIELD_RE.findall(text))
    if not entity_texts:
        entity_texts.extend(_ENTITY_TEXT_RE.findall(text))
    if not relation_types:
        relation_types.extend(_RELATION_TYPE_FIELD_RE.findall(text))
    if not relation_args:
        relation_args.extend(_ARG_RE.findall(text))

    if not coref_members:
        for m in _MEMBERS_RE.findall(text):
            parts = [x.strip().strip('"').strip("'") for x in m.split(",")]
            coref_members.extend([x for x in parts if x])

    return {
        "entity_type": _uniq(entity_types),
        "relation_type": _uniq(relation_types),
        "entity_text": _uniq(entity_texts),
        "relation_arg": _uniq(relation_args),
        "coref_member": _uniq(coref_members),
    }

# =============================================================================
# CLASS IMBALANCE: EXAMPLE REWEIGHTING FOR SCIERC DPO
# =============================================================================

def compute_class_weights(
    counts: Dict[str, int],
    strategy: str,
    beta: float = 0.999,
    alpha: float = 1.0,
) -> Dict[str, float]:
    if not counts:
        return {}

    total = float(sum(counts.values()))
    out: Dict[str, float] = {}

    for k, n in counts.items():
        n = max(int(n), 1)

        if strategy == "effective_num":
            w = (1.0 - beta) / (1.0 - (beta ** n))
        elif strategy == "inv_freq":
            w = (total / n) ** alpha
        elif strategy == "inv_sqrt":
            w = (total / n) ** (0.5 * alpha)
        else:
            raise ValueError(f"Unknown class_balance_strategy: {strategy}")

        out[k] = float(w)

    # Normalize mean class weight to 1.0
    mean_w = sum(out.values()) / max(len(out), 1)
    if mean_w > 0:
        out = {k: v / mean_w for k, v in out.items()}

    return out


def add_example_weights_dpo(train_ds, cfg: Config):
    """
    Adds column: example_weight.

    PV Miner balances Code/Sub-code.
    SCIERC adaptation:
      Code     -> entity_type
      Sub-code -> relation_type

    Uses only the CHOSEN response, matching the PV DPO implementation.
    """
    if not cfg.enable_class_balance:
        return train_ds.map(
            lambda ex: {"example_weight": 1.0},
            desc="Setting example_weight=1.0",
        )

    entity_ctr = Counter()
    relation_ctr = Counter()

    # Count SCIERC classes from chosen response only
    for ex in train_ds:
        vals = extract_values_from_jsonish(ex.get("chosen", ""))

        if cfg.class_balance_use_entity_type:
            for v in vals.get("entity_type", []):
                entity_ctr[v] += 1

        if cfg.class_balance_use_relation_type:
            for v in vals.get("relation_type", []):
                relation_ctr[v] += 1

    entity_w = (
        compute_class_weights(
            dict(entity_ctr),
            cfg.class_balance_strategy,
            cfg.class_balance_beta,
            cfg.class_balance_alpha,
        )
        if cfg.class_balance_use_entity_type
        else {}
    )

    relation_w = (
        compute_class_weights(
            dict(relation_ctr),
            cfg.class_balance_strategy,
            cfg.class_balance_beta,
            cfg.class_balance_alpha,
        )
        if cfg.class_balance_use_relation_type
        else {}
    )

    def _combine(a: float, b: float) -> float:
        if cfg.class_balance_combine == "max":
            return max(a, b)
        if cfg.class_balance_combine == "geom":
            return math.sqrt(max(a, 1e-8) * max(b, 1e-8))
        if cfg.class_balance_combine == "mean":
            return 0.5 * (a + b)
        raise ValueError(f"Unknown class_balance_combine: {cfg.class_balance_combine}")

    def _one(ex):
        vals = extract_values_from_jsonish(ex.get("chosen", ""))

        ew_entity = 1.0
        ew_relation = 1.0

        if cfg.class_balance_use_entity_type:
            entity_types = vals.get("entity_type", [])
            if entity_types:
                ew_entity = max(entity_w.get(v, 1.0) for v in entity_types)

        if cfg.class_balance_use_relation_type:
            relation_types = vals.get("relation_type", [])
            if relation_types:
                ew_relation = max(relation_w.get(v, 1.0) for v in relation_types)

        ew = float(min(_combine(ew_entity, ew_relation), cfg.class_balance_max_weight))
        return {"example_weight": ew}

    train_ds = train_ds.map(_one, desc="Computing SCIERC example weights")

    # Normalize mean example weight to 1.0
    ws = [float(x) for x in train_ds["example_weight"]]
    mean_w = sum(ws) / max(len(ws), 1)

    if mean_w > 0:
        def _norm(ex):
            ew = float(ex["example_weight"]) / float(mean_w)
            ew = float(min(ew, cfg.class_balance_max_weight))
            return {"example_weight": ew}

        train_ds = train_ds.map(_norm, desc="Normalizing example weights mean=1.0")

    ws = [float(x) for x in train_ds["example_weight"]]

    print("\n" + "=" * 80)
    print("SCIERC CLASS BALANCE SUMMARY DPO")
    print("=" * 80)
    print(
        f"Strategy: {cfg.class_balance_strategy} | "
        f"beta={cfg.class_balance_beta} | "
        f"alpha={cfg.class_balance_alpha} | "
        f"combine={cfg.class_balance_combine} | "
        f"max_w={cfg.class_balance_max_weight}"
    )
    print(f"Unique entity_type classes: {len(entity_w)}")
    print(f"Unique relation_type classes: {len(relation_w)}")
    print(f"Example weight: min={min(ws):.3f} mean={sum(ws)/len(ws):.3f} max={max(ws):.3f}")
    print("=" * 80 + "\n")

    return train_ds

def _find_subsequence(
    haystack: List[int],
    needle: List[int],
    start: int,
    end: int,
) -> Optional[int]:
    if not needle or end - start < len(needle):
        return None

    for i in range(start, end - len(needle) + 1):
        if haystack[i:i + len(needle)] == needle:
            return i

    return None


# ============================================================================
# Data collator
# ============================================================================

class DPODataCollatorTokenWeighted:
    """
    Builds two sequences per example:
      chosen_text   = chat(system, user, assistant=chosen)
      rejected_text = chat(system, user, assistant=rejected)

    Masks all tokens before the answer substring.
    Applies SCIERC token-value weights inside the chosen/rejected answer JSON.
    """

    def __init__(
        self,
        tokenizer,
        cfg: Config,
        system_prompt: str,
    ):
        self.tok = tokenizer
        self.cfg = cfg
        self.system_prompt = system_prompt

        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

        self.pad_id = self.tok.pad_token_id
        self._logged_responses = False
        self._logged_mask_fallback = False
        self._logged_value_extraction = False

        if not getattr(self.tok, "is_fast", False):
            print(
                "⚠️  WARNING: tokenizer is not fast; offset_mapping unavailable. "
                "Using token-subsequence fallback for masking/value weighting."
            )

    def _truncate_prompt(self, prompt: str) -> str:
        ids = self.tok.encode(prompt, add_special_tokens=False)

        if len(ids) <= self.cfg.max_prompt_length:
            return prompt

        raise ValueError(f"Prompt has {len(ids)} tokens, exceeding max_prompt_length={self.cfg.max_prompt_length}. "
                         "Increase the limit to preserve the complete extraction instruction.")

    def _build_full_text(self, prompt: str, answer: str) -> str:
        """
        Use the model chat template with explicit assistant content.
        """
        return self.tok.apply_chat_template(
            messages(prompt, answer, self.system_prompt),
            tokenize=False,
            add_generation_prompt=False,
        )

    def _make_labels_from_answer_offsets(
        self,
        full_text: str,
        input_ids: List[int],
        offsets: List[Tuple[int, int]],
        answer: str,
    ) -> List[int]:
        labels = list(input_ids)
        idx = full_text.rfind(answer)

        if idx == -1:
            return [-100] * len(labels)

        resp_start_char = idx

        for i, (s, e) in enumerate(offsets):
            if s < resp_start_char:
                labels[i] = -100

        return labels

    def _make_labels_from_answer_token_fallback(
        self,
        input_ids: List[int],
        answer: str,
    ) -> List[int]:
        labels = list(input_ids)

        answer_ids = self.tok.encode(answer, add_special_tokens=False)
        if not answer_ids:
            return [-100] * len(labels)

        pos = _find_subsequence(input_ids, answer_ids, 0, len(input_ids))
        if pos is None:
            return [-100] * len(labels)

        for i in range(pos):
            labels[i] = -100

        if not self._logged_mask_fallback:
            print(
                "⚠️  Using token-subsequence fallback masking. "
                "A fast tokenizer is recommended for more reliable masking."
            )
            self._logged_mask_fallback = True

        return labels

    def _init_weights(self, labels: List[int]) -> List[float]:
        return [0.0 if lab == -100 else 1.0 for lab in labels]

    def _scierc_targets(self, assistant_content: str) -> List[Tuple[str, float, str]]:
        """
        Build value targets as:
          (value_string, weight, category)
        """
        vals = extract_values_from_jsonish(assistant_content)

        if not self._logged_value_extraction:
            print("\n" + "=" * 80)
            print("SCIERC TOKEN-WEIGHT VALUE EXTRACTION EXAMPLE")
            print("=" * 80)
            print("entity_type:", vals["entity_type"][:10])
            print("relation_type:", vals["relation_type"][:10])
            print("entity_text:", vals["entity_text"][:10])
            print("relation_arg:", vals["relation_arg"][:10])
            print("=" * 80 + "\n")
            self._logged_value_extraction = True

        targets: List[Tuple[str, float, str]] = []

        for v in vals["entity_type"]:
            targets.append((v, self.cfg.weight_entity_type_value, "entity_type"))

        for v in vals["relation_type"]:
            targets.append((v, self.cfg.weight_relation_type_value, "relation_type"))

        for v in vals["entity_text"]:
            targets.append((v, self.cfg.weight_entity_text_value, "entity_text"))

        for v in vals["relation_arg"]:
            targets.append((v, self.cfg.weight_relation_arg_value, "relation_arg"))

        # Coref members are entity IDs; use relation_arg weight as the safest default.
        for v in vals["coref_member"]:
            targets.append((v, self.cfg.weight_relation_arg_value, "coref_member"))

        return targets

    def _apply_value_weights_with_offsets(
        self,
        full_text: str,
        offsets: List[Tuple[int, int]],
        labels: List[int],
        weights: List[float],
        assistant_content: str,
    ) -> None:
        base = full_text.rfind(assistant_content)
        if base == -1:
            return

        targets: List[Tuple[int, int, float, str]] = []

        for value, weight, category in self._scierc_targets(assistant_content):
            if not value:
                continue

            start_local = assistant_content.find(value)
            while start_local != -1:
                s = base + start_local
                e = s + len(value)
                targets.append((s, e, weight, category))
                start_local = assistant_content.find(value, start_local + 1)

        if not targets:
            return

        for i, (a, b) in enumerate(offsets):
            if labels[i] == -100:
                continue

            # Special tokens often have (0, 0)
            if a == b == 0:
                continue

            for s, e, weight, category in targets:
                if a < e and b > s:
                    weights[i] = max(weights[i], weight)

    def _apply_value_weights_fallback_token_search(
        self,
        input_ids: List[int],
        labels: List[int],
        weights: List[float],
        assistant_content: str,
    ) -> None:
        resp_start = next((i for i, lab in enumerate(labels) if lab != -100), len(labels))
        resp_end = len(labels)

        for value, weight, category in self._scierc_targets(assistant_content):
            if not value:
                continue

            value_ids = self.tok.encode(value, add_special_tokens=False)
            pos = _find_subsequence(input_ids, value_ids, resp_start, resp_end)

            while pos is not None:
                for j in range(pos, min(pos + len(value_ids), len(weights))):
                    if labels[j] != -100:
                        weights[j] = max(weights[j], weight)

                next_start = pos + max(1, len(value_ids))
                pos = _find_subsequence(input_ids, value_ids, next_start, resp_end)

    def _apply_diff_weight(
        self,
        chosen_ids: List[int],
        chosen_labels: List[int],
        chosen_w: List[float],
        rejected_ids: List[int],
        rejected_labels: List[int],
        rejected_w: List[float],
        w_diff: float,
    ) -> None:
        c0 = next((i for i, lab in enumerate(chosen_labels) if lab != -100), None)
        r0 = next((i for i, lab in enumerate(rejected_labels) if lab != -100), None)

        if c0 is None or r0 is None:
            return

        c_len = sum(1 for i in range(c0, len(chosen_labels)) if chosen_labels[i] != -100)
        r_len = sum(1 for i in range(r0, len(rejected_labels)) if rejected_labels[i] != -100)
        L = min(c_len, r_len)

        for k in range(L):
            ci = c0 + k
            ri = r0 + k

            if chosen_labels[ci] == -100 or rejected_labels[ri] == -100:
                continue

            if chosen_ids[ci] != rejected_ids[ri]:
                chosen_w[ci] *= w_diff
                rejected_w[ri] *= w_diff

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        has_ref_logps = all(
            ("ref_chosen_logp" in ex and "ref_rejected_logp" in ex)
            for ex in features
        )

        batch = {
            "chosen_input_ids": [],
            "chosen_attention_mask": [],
            "chosen_labels": [],
            "chosen_weights": [],
            "rejected_input_ids": [],
            "rejected_attention_mask": [],
            "rejected_labels": [],
            "rejected_weights": [],
            "example_weight": [],
        }

        if has_ref_logps:
            batch["ref_chosen_logp"] = []
            batch["ref_rejected_logp"] = []

        for idx, ex in enumerate(features):
            prompt = self._truncate_prompt(ex["prompt"])
            chosen_resp = ex["chosen"]
            rejected_resp = ex["rejected"]

            batch["example_weight"].append(float(ex.get("example_weight", 1.0)))
            if has_ref_logps:
                batch["ref_chosen_logp"].append(float(ex["ref_chosen_logp"]))
                batch["ref_rejected_logp"].append(float(ex["ref_rejected_logp"]))

            if not self._logged_responses:
                print(f"\n{'=' * 80}")
                print(f"EXAMPLE #{idx + 1} - SCIERC CHOSEN vs REJECTED")
                print(f"{'=' * 80}")
                print(f"\n✅ CHOSEN RESPONSE first 600 chars:\n{chosen_resp[:600]}")
                print(f"\n❌ REJECTED RESPONSE first 600 chars:\n{rejected_resp[:600]}")
                print(f"{'=' * 80}\n")
                self._logged_responses = True

            chosen_text, chosen_enc, c_labels = completion_encoding(
                self.tok, prompt, chosen_resp, self.cfg.max_length, self.system_prompt)
            rejected_text, rejected_enc, r_labels = completion_encoding(
                self.tok, prompt, rejected_resp, self.cfg.max_length, self.system_prompt)

            c_ids = chosen_enc["input_ids"]
            r_ids = rejected_enc["input_ids"]
            c_am = chosen_enc["attention_mask"]
            r_am = rejected_enc["attention_mask"]

            c_offsets = chosen_enc.get("offset_mapping", None)
            r_offsets = rejected_enc.get("offset_mapping", None)

            c_w = self._init_weights(c_labels)
            r_w = self._init_weights(r_labels)

            if c_offsets is not None:
                self._apply_value_weights_with_offsets(
                    chosen_text,
                    c_offsets,
                    c_labels,
                    c_w,
                    chosen_resp,
                )
            else:
                self._apply_value_weights_fallback_token_search(
                    c_ids,
                    c_labels,
                    c_w,
                    chosen_resp,
                )

            if r_offsets is not None:
                self._apply_value_weights_with_offsets(
                    rejected_text,
                    r_offsets,
                    r_labels,
                    r_w,
                    rejected_resp,
                )
            else:
                self._apply_value_weights_fallback_token_search(
                    r_ids,
                    r_labels,
                    r_w,
                    rejected_resp,
                )

            if self.cfg.weight_diff_tokens and self.cfg.weight_diff_tokens > 1.0:
                self._apply_diff_weight(
                    c_ids,
                    c_labels,
                    c_w,
                    r_ids,
                    r_labels,
                    r_w,
                    self.cfg.weight_diff_tokens,
                )

            batch["chosen_input_ids"].append(c_ids)
            batch["chosen_attention_mask"].append(c_am)
            batch["chosen_labels"].append(c_labels)
            batch["chosen_weights"].append(c_w)

            batch["rejected_input_ids"].append(r_ids)
            batch["rejected_attention_mask"].append(r_am)
            batch["rejected_labels"].append(r_labels)
            batch["rejected_weights"].append(r_w)

        max_len = max(
            max(len(x) for x in batch["chosen_input_ids"]),
            max(len(x) for x in batch["rejected_input_ids"]),
        )

        def pad_int(seqs, pad_val):
            return [s + [pad_val] * (max_len - len(s)) for s in seqs]

        def pad_float(seqs, pad_val):
            return [s + [pad_val] * (max_len - len(s)) for s in seqs]

        batch["chosen_input_ids"] = pad_int(batch["chosen_input_ids"], self.pad_id)
        batch["rejected_input_ids"] = pad_int(batch["rejected_input_ids"], self.pad_id)
        batch["chosen_attention_mask"] = pad_int(batch["chosen_attention_mask"], 0)
        batch["rejected_attention_mask"] = pad_int(batch["rejected_attention_mask"], 0)
        batch["chosen_labels"] = pad_int(batch["chosen_labels"], -100)
        batch["rejected_labels"] = pad_int(batch["rejected_labels"], -100)
        batch["chosen_weights"] = pad_float(batch["chosen_weights"], 0.0)
        batch["rejected_weights"] = pad_float(batch["rejected_weights"], 0.0)

        out: Dict[str, torch.Tensor] = {}

        for k in batch:
            if ("weights" in k) or (k == "example_weight") or k.startswith("ref_"):
                out[k] = torch.tensor(batch[k], dtype=torch.float32)
            else:
                out[k] = torch.tensor(batch[k], dtype=torch.long)

        return out

def _supervised_logits_window(labels: torch.Tensor) -> Tuple[int, int]:
    """
    Return:
      start: original logits start position.
             For causal LM, logits at position i-1 predict label at position i.
      keep:  number of trailing logits to request through num_logits_to_keep.

    labels shape: [B, T]
    """
    valid = labels.ne(-100)
    counts = valid.sum(dim=1)

    if torch.any(counts == 0):
        bad_rows = torch.nonzero(counts == 0, as_tuple=False).flatten().detach().cpu().tolist()
        raise ValueError(
            f"Found batch rows with zero supervised tokens: rows={bad_rows}. "
            "This usually means the assistant answer was truncated away. "
            "Use tok.truncation_side='left' and reduce max_prompt_length."
        )

    first_idx = valid.float().argmax(dim=1)
    start = max(int(first_idx.min().item()) - 1, 0)
    keep = labels.size(1) - start
    keep = max(2, keep)

    return start, keep


def _get_shifted_logits_labels_weights(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Return only the answer-supervised shifted logits/labels/weights.

    This keeps the full prompt in input_ids as context, but avoids computing
    or at least avoids passing full prompt logits into cross_entropy.
    """
    start, keep = _supervised_logits_window(labels)

    try:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            num_logits_to_keep=keep,
        )

        logits = outputs.logits

        if logits.size(1) == keep:
            # logits correspond to original positions [start : T]
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, start + 1:].contiguous()
            shift_weights = weights[:, start + 1:].contiguous()

        elif logits.size(1) == labels.size(1):
            # Model ignored num_logits_to_keep; crop manually.
            shift_logits = logits[:, start:-1, :].contiguous()
            shift_labels = labels[:, start + 1:].contiguous()
            shift_weights = weights[:, start + 1:].contiguous()

        else:
            raise RuntimeError(
                f"Unexpected logits length {logits.size(1)}. "
                f"Expected keep={keep} or full sequence length={labels.size(1)}."
            )

    except TypeError:
        # Fallback for model classes that do not accept num_logits_to_keep.
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )

        logits = outputs.logits
        shift_logits = logits[:, start:-1, :].contiguous()
        shift_labels = labels[:, start + 1:].contiguous()
        shift_weights = weights[:, start + 1:].contiguous()

    if shift_logits.shape[:2] != shift_labels.shape:
        raise RuntimeError(
            f"Shift shape mismatch: "
            f"logits={tuple(shift_logits.shape)}, "
            f"labels={tuple(shift_labels.shape)}, "
            f"weights={tuple(shift_weights.shape)}"
        )

    vocab_size = shift_logits.size(-1)
    bad = (shift_labels != -100) & ((shift_labels < 0) | (shift_labels >= vocab_size))

    if torch.any(bad):
        bad_vals = shift_labels[bad][:20].detach().cpu().tolist()
        raise ValueError(
            f"Labels outside vocabulary range. "
            f"vocab_size={vocab_size}, bad label examples={bad_vals}"
        )

    return shift_logits, shift_labels, shift_weights

# ============================================================================
# Logp helper: memory-safe CE instead of full log_softmax
# ============================================================================

def weighted_sequence_logp(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    cfg: Config,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      seq_logp:
        - weighted SUM logp if cfg.enable_length_norm=False
        - weighted AVERAGE logp if cfg.enable_length_norm=True

      denom:
        - token count or weight mass, depending on cfg.length_norm_by
    """
    shift_logits, shift_labels, shift_weights = _get_shifted_logits_labels_weights(
        model,
        input_ids,
        attention_mask,
        labels,
        weights,
    )

    per_token_nll = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        reduction="none",
        ignore_index=-100,
    ).view(shift_labels.size())

    per_token_logps = -per_token_nll
    mask = (shift_labels != -100).float()
    tok_count = mask.sum(dim=1).clamp(min=1.0)

    w = shift_weights * mask

    # Same mechanism as PV code:
    # optionally normalize token weights so valid-token average weight is about 1.0.
    if cfg.normalize_by_weight_mass:
        w_sum_raw = w.sum(dim=1).clamp(min=1e-6)
        w = w * (tok_count / w_sum_raw).unsqueeze(1)

    sum_logp = (per_token_logps * w).sum(dim=1)

    if cfg.enable_length_norm:
        if cfg.length_norm_by == "weight_mass":
            denom = w.sum(dim=1).clamp(min=1e-6)
        elif cfg.length_norm_by == "tokens":
            denom = tok_count
        else:
            raise ValueError(f"Unknown length_norm_by: {cfg.length_norm_by}")

        seq_logp = sum_logp / denom
        return seq_logp, denom

    denom = torch.ones_like(tok_count)
    return sum_logp, denom


def unwrap_for_peft(model: torch.nn.Module) -> torch.nn.Module:
    """
    Works for plain PEFT model, DDP-wrapped model, and DeepSpeedEngine-wrapped model.
    We need the underlying PEFT model so disable_adapter() actually works.
    """
    m = model
    for _ in range(8):
        if hasattr(m, "disable_adapter") or hasattr(m, "disable_adapter_layers"):
            return m
        if hasattr(m, "module"):
            m = m.module
            continue
        break
    return m


class RefAdapterOff:
    """
    Use the same PEFT model with adapter disabled as reference policy.
    Works under DDP and DeepSpeed wrappers.
    """

    def __init__(self, model: torch.nn.Module):
        self.model = unwrap_for_peft(model)
        self._has_ctx = hasattr(self.model, "disable_adapter")
        self._ctx = None

    def __enter__(self):
        if self._has_ctx:
            self._ctx = self.model.disable_adapter()
            return self._ctx.__enter__()

        if hasattr(self.model, "disable_adapter_layers"):
            self.model.disable_adapter_layers()

        return None

    def __exit__(self, exc_type, exc, tb):
        if self._has_ctx:
            return self._ctx.__exit__(exc_type, exc, tb)

        if hasattr(self.model, "enable_adapter_layers"):
            self.model.enable_adapter_layers()

        return False

def add_reference_logps_to_dataset(
    ds,
    model: torch.nn.Module,
    collator: DPODataCollatorTokenWeighted,
    cfg: Config,
    split_name: str,
    batch_size: int = 1,
):
    """
    Precompute fixed reference log-probs once.

    Reference policy = same SFT-merged base model with LoRA adapter disabled.
    This preserves the DPO objective exactly as long as the same tokenizer,
    masking, token weights, normalize_by_weight_mass, and length_norm settings
    are used during caching and training.
    """
    if "ref_chosen_logp" in ds.column_names or "ref_rejected_logp" in ds.column_names:
        raise ValueError("Input contains unverified reference scores. Remove reference columns and "
                         "recompute them with this SFT model, tokenizer, mask, and weighting configuration.")

    batch_size = max(1, int(batch_size))

    # Ensure model is on GPU for caching.
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cpu" and torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
        if local_rank >= 0:
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cuda")
        model.to(device)

    print("\n" + "=" * 80)
    print(f"CACHING REFERENCE LOG-PROBS: {split_name}")
    print("=" * 80)
    print(f"Samples: {len(ds)}")
    print(f"Cache batch size: {batch_size}")
    print(f"Device: {device}")
    print("=" * 80)

    was_training = model.training
    model.eval()

    ref_chosen_all: List[float] = []
    ref_rejected_all: List[float] = []

    with torch.no_grad():
        ctx = RefAdapterOff(model) if (
            hasattr(unwrap_for_peft(model), "disable_adapter")
            or hasattr(unwrap_for_peft(model), "disable_adapter_layers")
        ) else None

        if ctx is None:
            context = contextlib.nullcontext()
        else:
            context = ctx

        with context:
            for start in range(0, len(ds), batch_size):
                end = min(start + batch_size, len(ds))
                features = [ds[i] for i in range(start, end)]
                batch = collator(features)

                c_ids = batch["chosen_input_ids"].to(device)
                c_am = batch["chosen_attention_mask"].to(device)
                c_lab = batch["chosen_labels"].to(device)
                c_w = batch["chosen_weights"].to(device)

                r_ids = batch["rejected_input_ids"].to(device)
                r_am = batch["rejected_attention_mask"].to(device)
                r_lab = batch["rejected_labels"].to(device)
                r_w = batch["rejected_weights"].to(device)

                ref_c, _ = weighted_sequence_logp(
                    model,
                    c_ids,
                    c_am,
                    c_lab,
                    c_w,
                    cfg=cfg,
                )
                ref_r, _ = weighted_sequence_logp(
                    model,
                    r_ids,
                    r_am,
                    r_lab,
                    r_w,
                    cfg=cfg,
                )

                ref_chosen_all.extend(ref_c.detach().float().cpu().tolist())
                ref_rejected_all.extend(ref_r.detach().float().cpu().tolist())

                if start == 0 or ((start // batch_size) % 100 == 0):
                    print(
                        f"{split_name}: cached {end}/{len(ds)} examples",
                        flush=True,
                    )

    if was_training:
        model.train()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if len(ref_chosen_all) != len(ds) or len(ref_rejected_all) != len(ds):
        raise RuntimeError(
            f"Reference cache length mismatch for {split_name}: "
            f"chosen={len(ref_chosen_all)}, rejected={len(ref_rejected_all)}, dataset={len(ds)}"
        )

    ds = ds.add_column("ref_chosen_logp", ref_chosen_all)
    ds = ds.add_column("ref_rejected_logp", ref_rejected_all)

    print(f"✓ Finished caching reference log-probs for {split_name}")
    return ds

# ============================================================================
# Trainer
# ============================================================================

class TWCDPOTrainer(Trainer):
    def __init__(self, *args, cfg: Config, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = cfg

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        c_ids = inputs["chosen_input_ids"]
        c_am = inputs["chosen_attention_mask"]
        c_lab = inputs["chosen_labels"]
        c_w = inputs["chosen_weights"]

        r_ids = inputs["rejected_input_ids"]
        r_am = inputs["rejected_attention_mask"]
        r_lab = inputs["rejected_labels"]
        r_w = inputs["rejected_weights"]

        exw = inputs.get("example_weight", None)
        if (not self.cfg.enable_class_balance) or (exw is None):
            exw = torch.ones(c_ids.size(0), device=c_ids.device, dtype=torch.float32)
        else:
            exw = exw.to(c_ids.device).float().view(-1)

        device = c_ids.device

        with torch.no_grad():
            c_valid_tokens = (c_lab[:, 1:] != -100).sum(dim=1)
            r_valid_tokens = (r_lab[:, 1:] != -100).sum(dim=1)

            if self.state.global_step < 5 or self.state.global_step % 100 == 0:
                print(
                    f"SCIERC DPO token check step={self.state.global_step}: "
                    f"chosen_valid={c_valid_tokens.tolist()} "
                    f"rejected_valid={r_valid_tokens.tolist()} "
                    f"seq_len={c_ids.shape[1]}",
                    flush=True,
                )

            if torch.any(c_valid_tokens == 0) or torch.any(r_valid_tokens == 0):
                raise ValueError(
                    f"Zero supervised tokens detected at step={self.state.global_step}. "
                    f"chosen_valid={c_valid_tokens.tolist()}, "
                    f"rejected_valid={r_valid_tokens.tolist()}. "
                    "The answer was likely truncated away."
                )

        def _seq_logp_and_token_logps(_ids, _am, _lab, _wts):
            shift_logits, shift_labels, shift_weights = _get_shifted_logits_labels_weights(
                model,
                _ids,
                _am,
                _lab,
                _wts,
            )

            per_token_nll = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
                reduction="none",
                ignore_index=-100,
            ).view(shift_labels.size())

            token_logps = -per_token_nll
            token_mask = shift_labels != -100
            mask_f = token_mask.float()

            w = shift_weights * mask_f

            tok_count = mask_f.sum(dim=1).clamp(min=1.0)

            if self.cfg.normalize_by_weight_mass:
                w_sum_raw = w.sum(dim=1).clamp(min=1e-6)
                w = w * (tok_count / w_sum_raw).unsqueeze(1)

            sum_logp = (token_logps * w).sum(dim=1)

            if self.cfg.enable_length_norm:
                if self.cfg.length_norm_by == "weight_mass":
                    denom = w.sum(dim=1).clamp(min=1e-6)
                elif self.cfg.length_norm_by == "tokens":
                    denom = tok_count
                else:
                    raise ValueError(f"Unknown length_norm_by: {self.cfg.length_norm_by}")

                seq_logp = sum_logp / denom
            else:
                seq_logp = sum_logp

            return seq_logp, token_logps, token_mask, w

        # Policy logps: adapter ON
        # If enabled, concatenate chosen/rejected along batch dimension.
        # This is safe here because the collator already pads chosen and rejected
        # to the same max_len within the batch.
        if self.cfg.concat_policy_forward:
            all_ids = torch.cat([c_ids, r_ids], dim=0)
            all_am = torch.cat([c_am, r_am], dim=0)
            all_lab = torch.cat([c_lab, r_lab], dim=0)
            all_w = torch.cat([c_w, r_w], dim=0)

            (
                pi_all,
                token_logps_all,
                token_mask_all,
                token_w_all,
            ) = _seq_logp_and_token_logps(
                all_ids,
                all_am,
                all_lab,
                all_w,
            )

            B = c_ids.size(0)

            pi_c = pi_all[:B]
            pi_r = pi_all[B:]

            c_token_logps = token_logps_all[:B]
            c_token_mask = token_mask_all[:B]
            c_token_w = token_w_all[:B]

        else:
            pi_c, c_token_logps, c_token_mask, c_token_w = _seq_logp_and_token_logps(
                c_ids,
                c_am,
                c_lab,
                c_w,
            )
            pi_r, _, _, _ = _seq_logp_and_token_logps(
                r_ids,
                r_am,
                r_lab,
                r_w,
            )

        # Reference logps.
        # Preferred path: use cached fixed-reference values.
        # Fallback path: recompute adapter-off reference logps if cache columns
        # are missing, preserving backward compatibility.
        if "ref_chosen_logp" in inputs and "ref_rejected_logp" in inputs:
            ref_c = inputs["ref_chosen_logp"].to(device=device).float().view(-1)
            ref_r = inputs["ref_rejected_logp"].to(device=device).float().view(-1)
        else:
            was_training = model.training
            model.eval()

            with torch.no_grad():
                with RefAdapterOff(model):
                    ref_c, _ = weighted_sequence_logp(
                        model,
                        c_ids,
                        c_am,
                        c_lab,
                        c_w,
                        cfg=self.cfg,
                    )
                    ref_r, _ = weighted_sequence_logp(
                        model,
                        r_ids,
                        r_am,
                        r_lab,
                        r_w,
                        cfg=self.cfg,
                    )

            if was_training:
                model.train()
        total, loss_terms = tabpo_loss(
            pi_c, pi_r, ref_c, ref_r, c_token_logps, c_token_mask, c_token_w,
            beta=self.cfg.beta, tau=self.cfg.policy_prob_threshold,
            barrier_weight=self.cfg.sft_barrier_weight, example_weights=exw,
            gated_weight_normalization=self.cfg.enable_length_norm and self.cfg.length_norm_by == "weight_mass")
        # Diagnostics use the exact terms from the shared, independently tested objective.
        pi_logratio = (pi_c - pi_r).float()
        ref_logratio = (ref_c - ref_r).float()
        dpo_logits = pi_logratio - ref_logratio

        p_pref = torch.sigmoid(self.cfg.beta * dpo_logits)
        dpo_loss = loss_terms["preference"]
        dpo_loss_mean = (dpo_loss * exw).sum() / exw.sum().clamp(min=1e-8)

        # Adaptive SFT barrier over under-confident chosen tokens
        gate = loss_terms["gate"]
        barrier_per_ex = loss_terms["barrier"]

        barrier_loss = (barrier_per_ex * exw).sum() / exw.sum().clamp(min=1e-8)


        with torch.no_grad():
            chosen_tok_count = c_token_mask.float().sum(dim=1).clamp(min=1.0)
            if self.cfg.enable_length_norm:
                policy_chosen_prob_geom = torch.exp(pi_c)
            else:
                policy_chosen_prob_geom = torch.exp(pi_c / chosen_tok_count)
            active_tokens = c_token_mask.float().sum().clamp(min=1.0)
            barrier_active_pct = gate.sum() / active_tokens

            supervised_tokens = c_token_mask.float().sum(dim=1).mean()

            metrics = {
                "loss": total.detach().float(),
                "loss/dpo": dpo_loss_mean.detach().float(),
                "loss/barrier": barrier_loss.detach().float(),
                "policy/prob_chosen_geom": policy_chosen_prob_geom.mean().detach().float(),
                "barrier/active_pct": barrier_active_pct.detach().float(),
                "pref/p_pref": p_pref.mean().detach().float(),
                "logits/mean": dpo_logits.mean().detach().float(),
                "logits/max": dpo_logits.max().detach().float(),
                "logits/min": dpo_logits.min().detach().float(),
                "tokens/chosen_supervised_mean": supervised_tokens.detach().float(),
            }

        self.log({k: v.item() for k, v in metrics.items()})

        return (total, metrics) if return_outputs else total

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            loss = self.compute_loss(model, inputs)
        return loss.detach(), None, None


def _is_rank0() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def _dist_barrier():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def unwrap_peft_model(model: torch.nn.Module) -> torch.nn.Module:
    """
    Unwrap DeepSpeedEngine/DDP to recover the underlying PEFT model.
    """
    m = model
    for _ in range(20):
        if hasattr(m, "peft_config") and hasattr(m, "save_pretrained"):
            return m
        if hasattr(m, "module"):
            m = m.module
            continue
        break
    return m


def _get_active_adapter_name(peft_model: torch.nn.Module) -> str:
    adapter_name = getattr(peft_model, "active_adapter", "default")

    if callable(adapter_name):
        adapter_name = adapter_name()

    if isinstance(adapter_name, (list, tuple)):
        adapter_name = adapter_name[0]

    if adapter_name is None:
        adapter_name = "default"

    return str(adapter_name)


def _normalize_lora_key_for_peft_save(key: str, adapter_name: str) -> str:
    """
    Convert runtime PEFT parameter keys to adapter checkpoint keys.

    Runtime PEFT parameter:
      base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight

    Correct saved adapter key:
      base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight

    If the saved checkpoint keeps '.default.' inside lora_A/lora_B keys,
    PEFT may not map the tensor back correctly on PeftModel.from_pretrained.
    """
    while key.startswith("module."):
        key = key[len("module.") :]

    replacements = [
        (f".lora_A.{adapter_name}.weight", ".lora_A.weight"),
        (f".lora_B.{adapter_name}.weight", ".lora_B.weight"),
        (f".lora_embedding_A.{adapter_name}", ".lora_embedding_A"),
        (f".lora_embedding_B.{adapter_name}", ".lora_embedding_B"),
        (f".lora_magnitude_vector.{adapter_name}", ".lora_magnitude_vector"),
    ]

    for old, new in replacements:
        key = key.replace(old, new)

    return key


def gather_lora_state_dict_zero3_strict(peft_model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """
    Gather LoRA parameters under DeepSpeed ZeRO-3.

    Important:
      This returns raw runtime PEFT keys first.
      These keys must be converted to PEFT adapter checkpoint keys before saving.
    """
    raw_state_dict: Dict[str, torch.Tensor] = {}

    try:
        import deepspeed
    except Exception:
        deepspeed = None

    for name, param in peft_model.named_parameters():
        if "lora_" not in name:
            continue

        if deepspeed is not None and hasattr(param, "ds_id"):
            with deepspeed.zero.GatheredParameters([param], modifier_rank=0):
                if _is_rank0():
                    raw_state_dict[name] = param.detach().float().cpu().clone()
        else:
            if _is_rank0():
                raw_state_dict[name] = param.detach().float().cpu().clone()

    if _is_rank0():
        if not raw_state_dict:
            raise RuntimeError(
                "DPO adapter save failed: gathered LoRA state_dict is EMPTY."
            )

        lora_a = {k: v for k, v in raw_state_dict.items() if "lora_A" in k}
        lora_b = {k: v for k, v in raw_state_dict.items() if "lora_B" in k}

        if not lora_a or not lora_b:
            raise RuntimeError(
                f"DPO adapter save failed: missing LoRA A/B tensors. "
                f"num_lora_A={len(lora_a)}, num_lora_B={len(lora_b)}"
            )

        max_abs = max(v.abs().max().item() for v in raw_state_dict.values())
        b_max_abs = max(v.abs().max().item() for v in lora_b.values())
        b_l2 = sum((v ** 2).sum().item() for v in lora_b.values()) ** 0.5

        print("=" * 80)
        print("DPO LoRA ZERO-3 RAW GATHER VALIDATION")
        print("=" * 80)
        print(f"Raw LoRA tensors gathered: {len(raw_state_dict)}")
        print(f"Raw LoRA A tensors:        {len(lora_a)}")
        print(f"Raw LoRA B tensors:        {len(lora_b)}")
        print(f"Raw max abs all LoRA:      {max_abs:.8e}")
        print(f"Raw max abs LoRA B:        {b_max_abs:.8e}")
        print(f"Raw L2 norm LoRA B:        {b_l2:.8e}")
        print("Example raw keys:")
        for k in list(raw_state_dict.keys())[:8]:
            print("  ", k)
        print("=" * 80)

        if max_abs == 0.0:
            raise RuntimeError("DPO adapter save failed: all LoRA tensors are exactly zero.")

        if b_max_abs == 0.0:
            raise RuntimeError(
                "DPO adapter save failed: all LoRA B tensors are zero. "
                "This means the trained adapter update was not captured."
            )

    return raw_state_dict


def build_peft_loadable_adapter_state_dict(
    peft_model: torch.nn.Module,
    raw_lora_state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Convert raw runtime PEFT/ZeRO-3 LoRA keys into adapter-save keys.

    Important:
    We do this manually and deterministically from the gathered raw LoRA
    parameters. Do not depend on get_peft_model_state_dict(...) here, because
    with ZeRO-3/wrapped models it can return a non-empty state dict whose keys
    still do not reload correctly.
    """
    adapter_name = _get_active_adapter_name(peft_model)

    if not _is_rank0():
        return {}

    adapter_state_dict: Dict[str, torch.Tensor] = {}

    for k, v in raw_lora_state_dict.items():
        if "lora_" not in k:
            continue

        nk = _normalize_lora_key_for_peft_save(k, adapter_name)

        # Defensive cleanup for common wrappers.
        while nk.startswith("module."):
            nk = nk[len("module."):]
        while nk.startswith("_fsdp_wrapped_module."):
            nk = nk[len("_fsdp_wrapped_module."):]

        adapter_state_dict[nk] = v.detach().cpu().contiguous()

    lora_keys = [k for k in adapter_state_dict if "lora_" in k]
    b_keys = [k for k in adapter_state_dict if "lora_B" in k]

    if not lora_keys:
        raise RuntimeError("PEFT-loadable adapter state_dict has no LoRA keys.")
    if not b_keys:
        raise RuntimeError("PEFT-loadable adapter state_dict has no LoRA B keys.")

    b_max_abs = max(adapter_state_dict[k].float().abs().max().item() for k in b_keys)

    print("=" * 80)
    print("PEFT-LOADABLE ADAPTER KEY VALIDATION")
    print("=" * 80)
    print(f"Adapter name:              {adapter_name}")
    print(f"Saved-format tensor keys:  {len(adapter_state_dict)}")
    print(f"Saved-format LoRA keys:    {len(lora_keys)}")
    print(f"Saved-format LoRA B keys:  {len(b_keys)}")
    print(f"Saved-format max LoRA B:   {b_max_abs:.8e}")
    print("Example saved-format keys:")
    for k in list(adapter_state_dict.keys())[:12]:
        print("  ", k)
    print("=" * 80)

    if b_max_abs == 0.0:
        raise RuntimeError("PEFT-loadable adapter LoRA B tensors are all zero.")

    return adapter_state_dict


def save_lora_adapter_only_zero3(trainer, tokenizer, output_dir: str):
    """
    Save a PEFT-loadable LoRA adapter under DeepSpeed ZeRO-3.

    Critical behavior:
      1. Unwrap DeepSpeed/DDP to the underlying PEFT model.
      2. Gather partitioned LoRA tensors from ZeRO-3.
      3. Convert runtime keys:
            module.base_model...lora_B.default.weight
         into PEFT adapter checkpoint keys:
            base_model...lora_B.weight
      4. Save adapter_config.json and adapter_model.safetensors directly.
      5. Validate that disk weights are nonzero and key format is reloadable.
    """
    import os
    import torch
    from safetensors.torch import save_file, load_file

    os.makedirs(output_dir, exist_ok=True)

    _dist_barrier()

    wrapped_model = getattr(trainer, "model_wrapped", None)
    if wrapped_model is None:
        wrapped_model = trainer.model

    peft_model = unwrap_peft_model(wrapped_model)

    if not (hasattr(peft_model, "peft_config") and hasattr(peft_model, "save_pretrained")):
        raise RuntimeError(
            f"Could not unwrap to a PEFT model. Got type={type(peft_model)}. "
            "Saving would likely produce a no-op adapter."
        )

    adapter_name = _get_active_adapter_name(peft_model)

    raw_lora_state_dict = gather_lora_state_dict_zero3_strict(peft_model)
    adapter_state_dict = build_peft_loadable_adapter_state_dict(
        peft_model,
        raw_lora_state_dict,
    )

    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(output_dir)

        # Save adapter config.
        if adapter_name not in peft_model.peft_config:
            raise RuntimeError(
                f"Active adapter '{adapter_name}' not found in peft_model.peft_config. "
                f"Available adapters: {list(peft_model.peft_config.keys())}"
            )

        peft_config = peft_model.peft_config[adapter_name]

        old_inference_mode = getattr(peft_config, "inference_mode", None)
        try:
            peft_config.inference_mode = True
        except Exception:
            pass

        peft_config.save_pretrained(output_dir)

        if old_inference_mode is not None:
            try:
                peft_config.inference_mode = old_inference_mode
            except Exception:
                pass

        # Remove stale/partial files before writing.
        adapter_safetensors = os.path.join(output_dir, "adapter_model.safetensors")
        adapter_bin = os.path.join(output_dir, "adapter_model.bin")

        for stale_path in (adapter_safetensors, adapter_bin):
            if os.path.exists(stale_path):
                os.remove(stale_path)

        # Direct deterministic save.
        save_file(
            adapter_state_dict,
            adapter_safetensors,
            metadata={"format": "pt"},
        )

        # Validate actual disk file.
        if not os.path.exists(adapter_safetensors):
            raise RuntimeError(
                f"Direct adapter save failed: {adapter_safetensors} was not created."
            )

        saved = load_file(adapter_safetensors)

        lora_keys = [k for k in saved.keys() if "lora_" in k]
        b_keys = [k for k in saved.keys() if "lora_B" in k]

        max_abs_saved = (
            max(saved[k].float().abs().max().item() for k in lora_keys)
            if lora_keys else 0.0
        )
        b_max_abs_saved = (
            max(saved[k].float().abs().max().item() for k in b_keys)
            if b_keys else 0.0
        )

        bad_keys = [
            k for k in saved.keys()
            if f".lora_A.{adapter_name}." in k
            or f".lora_B.{adapter_name}." in k
            or k.startswith("module.")
        ]

        print("=" * 80)
        print("DPO LoRA SAVE VALIDATION AFTER DIRECT DISK WRITE")
        print("=" * 80)
        print(f"Saved path:             {adapter_safetensors}")
        print(f"Saved tensor keys:      {len(saved)}")
        print(f"Saved LoRA keys:        {len(lora_keys)}")
        print(f"Saved LoRA B keys:      {len(b_keys)}")
        print(f"Saved max abs LoRA:     {max_abs_saved:.8e}")
        print(f"Saved max abs LoRA B:   {b_max_abs_saved:.8e}")
        print("Example disk keys:")
        for k in list(saved.keys())[:12]:
            print("  ", k)
        print("=" * 80)

        if not os.path.exists(os.path.join(output_dir, "adapter_config.json")):
            raise RuntimeError("adapter_config.json was not saved.")

        if bad_keys:
            raise RuntimeError(
                "Saved adapter still has non-loadable key names. Examples:\n"
                + "\n".join(bad_keys[:10])
            )

        if not lora_keys:
            raise RuntimeError("Saved adapter contains no LoRA tensors.")

        if not b_keys:
            raise RuntimeError("Saved adapter contains no LoRA B tensors.")

        if max_abs_saved == 0.0:
            raise RuntimeError("Saved adapter LoRA tensors are all zero.")

        if b_max_abs_saved == 0.0:
            raise RuntimeError("Saved adapter LoRA B tensors are all zero.")

    _dist_barrier()

def validate_saved_adapter_reload_effect(base_model_path: str, adapter_dir: str):
    """
    Validate that the saved adapter is not only nonzero on disk,
    but also loads into PEFT active LoRA modules and changes logits.
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    print("=" * 80)
    print("DPO ADAPTER RELOAD FUNCTIONAL VALIDATION")
    print("=" * 80)
    print("base_model_path:", base_model_path)
    print("adapter_dir:", adapter_dir)

    tok = AutoTokenizer.from_pretrained(base_model_path, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    prompt = (
        "Extract SCIERC entities, relations, and coreference clusters as valid JSON.\n"
        "Text: Neural networks are used for relation extraction.\n"
        "JSON:"
    )

    inputs = tok(prompt, return_tensors="pt").to("cuda")

    base = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        use_cache=False,
    ).eval()

    peft_model = PeftModel.from_pretrained(base, adapter_dir).eval()

    loaded_lora_keys = []
    loaded_lora_b_keys = []
    loaded_lora_b_max = 0.0

    for name, param in peft_model.named_parameters():
        if "lora_" in name:
            loaded_lora_keys.append(name)
        if "lora_B" in name:
            loaded_lora_b_keys.append(name)
            loaded_lora_b_max = max(
                loaded_lora_b_max,
                param.detach().float().abs().max().item(),
            )

    print("Loaded PEFT LoRA keys:", len(loaded_lora_keys))
    print("Loaded PEFT LoRA B keys:", len(loaded_lora_b_keys))
    print("Loaded PEFT max_abs_lora_B:", loaded_lora_b_max)

    if not loaded_lora_keys:
        raise RuntimeError("Reloaded PEFT model has no LoRA keys.")
    if loaded_lora_b_max == 0.0:
        raise RuntimeError(
            "Reloaded PEFT adapter has all-zero LoRA B weights. "
            "The saved adapter file is nonzero, but PEFT is not loading it correctly. "
            "This is likely a saved-key naming/mapping issue."
        )

    with torch.no_grad():
        with peft_model.disable_adapter():
            base_logits = peft_model(**inputs).logits[:, -1, :].float().cpu()

        adapter_logits = peft_model(**inputs).logits[:, -1, :].float().cpu()

    diff = (adapter_logits - base_logits).abs().max().item()
    print("MAX_LOGIT_DIFF: disable_adapter vs adapter_enabled =", diff)

    if diff == 0.0:
        raise RuntimeError(
            "Reloaded adapter has nonzero LoRA tensors but exactly zero functional logit effect. "
            "Either the adapter is not active, the key mapping is wrong, or the update is below bf16 resolution."
        )

    print("✓ Reloaded DPO adapter is functionally active.")
    print("=" * 80)

def str2bool(v):
    if isinstance(v, bool):
        return v

    s = str(v).strip().lower()

    if s in ("1", "true", "t", "yes", "y"):
        return True
    if s in ("0", "false", "f", "no", "n"):
        return False

    raise argparse.ArgumentTypeError(f"Boolean value expected, got: {v}")

# ============================================================================
# CLI
# ============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train SCIERC token-weighted TAB-PO / DPO model."
    )

    # Required
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--train_data_path", type=str, required=True)
    parser.add_argument("--valid_data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    # Runtime
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--system_prompt",
        type=str,
        default="",
    )

    parser.add_argument("--bf16", type=lambda x: str(x).lower() == "true", default=True)
    parser.add_argument("--fp16", type=lambda x: str(x).lower() == "true", default=False)
    parser.add_argument("--gradient_checkpointing", type=lambda x: str(x).lower() == "true", default=True)
    parser.add_argument("--deepspeed_config", type=str, default=None)

    parser.add_argument("--cache_reference_logps", type=str2bool, default=True)
    parser.add_argument("--ref_logp_cache_batch_size", type=int, default=1)
    parser.add_argument("--concat_policy_forward", type=str2bool, default=True)

    # Length
    parser.add_argument("--max_length", type=int, default=8192)
    parser.add_argument("--max_prompt_length", type=int, default=8192)

    # DPO parameters
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--policy_prob_threshold", type=float, default=0.87)
    parser.add_argument("--sft_barrier_weight", type=float, default=0.5)

    # SCIERC token weighting
    parser.add_argument("--weight_entity_type_value", type=float, default=1.0)
    parser.add_argument("--weight_relation_type_value", type=float, default=1.0)
    parser.add_argument("--weight_entity_text_value", type=float, default=1.0)
    parser.add_argument("--weight_relation_arg_value", type=float, default=1.0)
    parser.add_argument("--weight_diff_tokens", type=float, default=1.0)
    # Weight normalization / length normalization
    parser.add_argument("--normalize_by_weight_mass", type=str2bool, default=True)
    parser.add_argument("--enable_length_norm", type=str2bool, default=False)
    parser.add_argument(
        "--length_norm_by",
        type=str,
        default="tokens",
        choices=["tokens", "weight_mass"],
    )

    # Class balancing / example weighting
    parser.add_argument("--enable_class_balance", type=str2bool, default=False)
    parser.add_argument(
        "--class_balance_strategy",
        type=str,
        default="effective_num",
        choices=["effective_num", "inv_freq", "inv_sqrt"],
    )
    parser.add_argument("--class_balance_beta", type=float, default=0.99)
    parser.add_argument("--class_balance_alpha", type=float, default=1.0)
    parser.add_argument(
        "--class_balance_combine",
        type=str,
        default="max",
        choices=["max", "geom", "mean"],
    )
    parser.add_argument("--class_balance_max_weight", type=float, default=3.0)
    parser.add_argument("--class_balance_use_entity_type", type=str2bool, default=True)
    parser.add_argument("--class_balance_use_relation_type", type=str2bool, default=True)
    # Backward-compatible aliases from old EPPC/PV code.
    # If user passes old names, map them to the closest SCIERC meaning:
    #   code    -> entity type
    #   subcode -> relation type
    #   span    -> entity text
    parser.add_argument("--weight_code_value", type=float, default=None)
    parser.add_argument("--weight_subcode_value", type=float, default=None)
    parser.add_argument("--weight_span_value", type=float, default=None)

    # Training
    parser.add_argument("--num_train_epochs", type=float, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--eval_steps", type=int, default=None)
    parser.add_argument(
        "--eval_strategy",
        type=str,
        default="no",
        choices=["no", "steps", "epoch"],
    )
    parser.add_argument("--save_total_limit", type=int, default=None)

    # LoRA
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )

    return parser


def make_config(args: argparse.Namespace) -> Config:
    # Backward-compatible aliases
    weight_entity_type_value = args.weight_entity_type_value
    weight_relation_type_value = args.weight_relation_type_value
    weight_entity_text_value = args.weight_entity_text_value

    if args.weight_code_value is not None:
        weight_entity_type_value = args.weight_code_value

    if args.weight_subcode_value is not None:
        weight_relation_type_value = args.weight_subcode_value

    if args.weight_span_value is not None:
        weight_entity_text_value = args.weight_span_value

    lora_target_modules = tuple(
        x.strip()
        for x in args.lora_target_modules.split(",")
        if x.strip()
    )

    return Config(
        model_name=args.model_name,
        train_data_path=args.train_data_path,
        valid_data_path=args.valid_data_path,
        output_dir=args.output_dir,

        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,

        beta=args.beta,
        policy_prob_threshold=args.policy_prob_threshold,
        sft_barrier_weight=args.sft_barrier_weight,

        cache_reference_logps=args.cache_reference_logps,
        ref_logp_cache_batch_size=args.ref_logp_cache_batch_size,
        concat_policy_forward=args.concat_policy_forward,

        weight_entity_type_value=weight_entity_type_value,
        weight_relation_type_value=weight_relation_type_value,
        weight_entity_text_value=weight_entity_text_value,
        weight_relation_arg_value=args.weight_relation_arg_value,
        weight_diff_tokens=args.weight_diff_tokens,

        normalize_by_weight_mass=args.normalize_by_weight_mass,
        enable_length_norm=args.enable_length_norm,
        length_norm_by=args.length_norm_by,

        enable_class_balance=args.enable_class_balance,
        class_balance_strategy=args.class_balance_strategy,
        class_balance_beta=args.class_balance_beta,
        class_balance_alpha=args.class_balance_alpha,
        class_balance_combine=args.class_balance_combine,
        class_balance_max_weight=args.class_balance_max_weight,
        class_balance_use_entity_type=args.class_balance_use_entity_type,
        class_balance_use_relation_type=args.class_balance_use_relation_type,

        weight_code_value=weight_entity_type_value,
        weight_subcode_value=weight_relation_type_value,
        weight_span_value=weight_entity_text_value,

        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy=args.eval_strategy,
        save_total_limit=args.save_total_limit,

        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=lora_target_modules,

        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        deepspeed_config=args.deepspeed_config,
        seed=args.seed,
    )


# ============================================================================
# Main
# ============================================================================

def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    cfg = make_config(args)
    set_seed(cfg.seed)
    if cfg.beta <= 0 or not 0 < cfg.policy_prob_threshold < 1 or cfg.sft_barrier_weight < 0:
        parser.error("Require beta > 0, 0 < policy_prob_threshold < 1, sft_barrier_weight >= 0")
    if cfg.bf16 and cfg.fp16:
        parser.error("Choose either bf16 or fp16")

    os.makedirs(cfg.output_dir, exist_ok=True)

    print("=" * 80)
    print("SCIERC Token-Weighted TAB-PO / DPO with Adaptive Barrier")
    print("=" * 80)

    source_type, model_path = detect_model_source(cfg.model_name)
    print(f"Model source: {source_type}")
    print(f"Model path/ID: {model_path}")

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    print("\nLoading tokenizer...")
    tok = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        token=hf_token,
    )

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    tok.truncation_side = "left"
    tok.padding_side = "right"
    print("✓ Tokenizer loaded")

    print("\nLoading DPO datasets...")
    train_ds = load_from_disk(cfg.train_data_path)
    valid_ds = load_from_disk(cfg.valid_data_path)
    if set(train_ds["prompt"]) & set(valid_ds["prompt"]):
        raise ValueError("Preference training and validation prompts overlap")

    print(f"Train: {len(train_ds)} samples")
    print(f"Valid: {len(valid_ds)} samples")
    print(f"Train columns: {train_ds.column_names}")
    print(f"Valid columns: {valid_ds.column_names}")

    needed = {"prompt", "chosen", "rejected"}
    for name, ds in [("train", train_ds), ("valid", valid_ds)]:
        cols = set(ds.column_names)
        missing = needed - cols
        if missing:
            raise ValueError(f"{name} dataset missing columns: {missing}. Found columns: {ds.column_names}")

    # Add example_weight column for class balancing.
    # Train uses SCIERC class-balanced weights if enabled.
    # Validation stays neutral.
    train_ds = add_example_weights_dpo(train_ds, cfg)
    valid_ds = valid_ds.map(
        lambda ex: {"example_weight": 1.0},
        desc="Setting valid example_weight=1.0",
    )

    eval_steps = cfg.eval_steps if cfg.eval_steps is not None else cfg.save_steps

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if local_rank >= 0:
        torch.cuda.set_device(local_rank)

    print("\nLoading model.")

    # IMPORTANT:
    # Do NOT create TrainingArguments with deepspeed before this point.
    # If TrainingArguments(deepspeed=...) is created before model loading,
    # Transformers/DeepSpeed ZeRO-3 can partition parameters during load.
    # Then the reference-cache raw forward can fail with:
    # RuntimeError: 'weight' must be 2-D
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float16 if cfg.fp16 else torch.float32,
        device_map=None,
        use_cache=False,
        token=hf_token,
    )

    print("✓ Base model loaded")

    print("\nTraining configuration:")
    print(json.dumps({
        "max_length": cfg.max_length,
        "max_prompt_length": cfg.max_prompt_length,
        "beta": cfg.beta,
        "policy_prob_threshold": cfg.policy_prob_threshold,
        "sft_barrier_weight": cfg.sft_barrier_weight,
        "cache_reference_logps": cfg.cache_reference_logps,
        "ref_logp_cache_batch_size": cfg.ref_logp_cache_batch_size,
        "concat_policy_forward": cfg.concat_policy_forward,
        "weight_entity_type_value": cfg.weight_entity_type_value,
        "weight_relation_type_value": cfg.weight_relation_type_value,
        "weight_entity_text_value": cfg.weight_entity_text_value,
        "weight_relation_arg_value": cfg.weight_relation_arg_value,
        "weight_diff_tokens": cfg.weight_diff_tokens,
        "num_train_epochs": cfg.num_train_epochs,
        "per_device_train_batch_size": cfg.per_device_train_batch_size,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "learning_rate": cfg.learning_rate,
        "warmup_ratio": cfg.warmup_ratio,
        "max_grad_norm": cfg.max_grad_norm,
        "logging_steps": cfg.logging_steps,
        "save_steps": cfg.save_steps,
        "eval_strategy": cfg.eval_strategy,
        "eval_steps": cfg.eval_steps,
        "lora_r": cfg.lora_r,
        "lora_alpha": cfg.lora_alpha,
        "lora_dropout": cfg.lora_dropout,
        "lora_target_modules": cfg.lora_target_modules,
        "normalize_by_weight_mass": cfg.normalize_by_weight_mass,
        "enable_length_norm": cfg.enable_length_norm,
        "length_norm_by": cfg.length_norm_by,
        "enable_class_balance": cfg.enable_class_balance,
        "class_balance_strategy": cfg.class_balance_strategy,
        "class_balance_beta": cfg.class_balance_beta,
        "class_balance_alpha": cfg.class_balance_alpha,
        "class_balance_combine": cfg.class_balance_combine,
        "class_balance_max_weight": cfg.class_balance_max_weight,
        "class_balance_use_entity_type": cfg.class_balance_use_entity_type,
        "class_balance_use_relation_type": cfg.class_balance_use_relation_type,
        "deepspeed_config": cfg.deepspeed_config,
    }, indent=2))

    print("\nCreating SCIERC token-weighted data collator...")
    collator = DPODataCollatorTokenWeighted(
        tok,
        cfg=cfg,
        system_prompt=args.system_prompt,
    )

    # Cache reference log-probs using the plain SFT-merged base model,
    # BEFORE attaching the trainable DPO LoRA adapter and BEFORE creating
    # TrainingArguments(deepspeed=...).
    if cfg.cache_reference_logps:
        train_ds = add_reference_logps_to_dataset(
            train_ds,
            model=model,
            collator=collator,
            cfg=cfg,
            split_name="train",
            batch_size=cfg.ref_logp_cache_batch_size,
        )

        if cfg.eval_strategy != "no":
            valid_ds = add_reference_logps_to_dataset(
                valid_ds,
                model=model,
                collator=collator,
                cfg=cfg,
                split_name="valid",
                batch_size=cfg.ref_logp_cache_batch_size,
            )

        # If DeepSpeed will initialize training, free the full reference-cache
        # model from GPU first. Trainer/DeepSpeed will place/partition it again.
        if cfg.deepspeed_config is not None and torch.cuda.is_available():
            model.to("cpu")
            torch.cuda.empty_cache()

    print("\nApplying LoRA...")
    lora = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.lora_target_modules),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    if torch.cuda.is_available() and cfg.deepspeed_config is None:
        if local_rank >= 0:
            model.to(torch.device("cuda", local_rank))
        else:
            model.to("cuda")

    if cfg.gradient_checkpointing:
        # Use reentrant checkpointing with DeepSpeed/ZeRO.
        # Non-reentrant checkpointing can fail with metadata mismatch during
        # recomputation when ZeRO partitions/gathers tensors.
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": True}
            )
        except TypeError:
            model.gradient_checkpointing_enable()

        model.enable_input_require_grads()

        if hasattr(model, "config"):
            model.config.use_cache = False

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        max_grad_norm=cfg.max_grad_norm,
        bf16=cfg.bf16,
        fp16=cfg.fp16,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        save_strategy="no",
        report_to="none",
        seed=cfg.seed,
        data_seed=cfg.seed,
        remove_unused_columns=False,
        gradient_checkpointing=cfg.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": True} if cfg.gradient_checkpointing else None,
        eval_strategy=cfg.eval_strategy,
        eval_steps=eval_steps if cfg.eval_strategy == "steps" else None,
        ddp_find_unused_parameters=False if world_size > 1 else None,
        deepspeed=cfg.deepspeed_config,
    )

    trainer = TWCDPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds if cfg.eval_strategy != "no" else None,
        data_collator=collator,
        tokenizer=tok,
        cfg=cfg,
    )

    print("\nStarting SCIERC DPO training...")
    trainer.train()

    print(f"\nSaving LoRA adapter only to {cfg.output_dir}...")
    save_lora_adapter_only_zero3(trainer, tok, cfg.output_dir)

    run_reload_validation = os.environ.get(
        "RUN_DPO_RELOAD_VALIDATION_INSIDE_TRAINER",
        "0",
    ).strip().lower() in ("1", "true", "yes", "y")

    if trainer.is_world_process_zero() and run_reload_validation:
        validate_saved_adapter_reload_effect(
            base_model_path=cfg.model_name,
            adapter_dir=cfg.output_dir,
        )

    model_info = {
        "task": "SCIERC",
        "method": "Token-weighted TAB-PO / DPO with adaptive barrier",
        "base_model": cfg.model_name,
        "base_model_source": source_type,
        "base_model_resolved_path": model_path,
        "training_completed": True,
        "dpo_dataset_columns": ["prompt", "chosen", "rejected"],
        "scierc_weight_mapping": {
            "entity_type": cfg.weight_entity_type_value,
            "relation_type": cfg.weight_relation_type_value,
            "entity_text": cfg.weight_entity_text_value,
            "relation_arg": cfg.weight_relation_arg_value,
            "diff_tokens": cfg.weight_diff_tokens,
        },
        "config": {
            "max_length": cfg.max_length,
            "max_prompt_length": cfg.max_prompt_length,
            "beta": cfg.beta,
            "policy_prob_threshold": cfg.policy_prob_threshold,
            "sft_barrier_weight": cfg.sft_barrier_weight,
            "num_train_epochs": cfg.num_train_epochs,
            "per_device_train_batch_size": cfg.per_device_train_batch_size,
            "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
            "learning_rate": cfg.learning_rate,
            "warmup_ratio": cfg.warmup_ratio,
            "max_grad_norm": cfg.max_grad_norm,
            "lora_r": cfg.lora_r,
            "lora_alpha": cfg.lora_alpha,
            "lora_dropout": cfg.lora_dropout,
            "lora_target_modules": list(cfg.lora_target_modules),
            "normalize_by_weight_mass": cfg.normalize_by_weight_mass,
            "enable_length_norm": cfg.enable_length_norm,
            "length_norm_by": cfg.length_norm_by,
            "enable_class_balance": cfg.enable_class_balance,
            "class_balance_strategy": cfg.class_balance_strategy,
            "class_balance_beta": cfg.class_balance_beta,
            "class_balance_alpha": cfg.class_balance_alpha,
            "class_balance_combine": cfg.class_balance_combine,
            "class_balance_max_weight": cfg.class_balance_max_weight,
            "class_balance_use_entity_type": cfg.class_balance_use_entity_type,
            "class_balance_use_relation_type": cfg.class_balance_use_relation_type,
        },
    }

    if trainer.is_world_process_zero():
        with open(os.path.join(cfg.output_dir, "model_info.json"), "w", encoding="utf-8") as f:
            json.dump(model_info, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("SCIERC DPO TRAINING COMPLETE")
    print("=" * 80)
    print(f"Output saved to: {cfg.output_dir}")


if __name__ == "__main__":
    main()
