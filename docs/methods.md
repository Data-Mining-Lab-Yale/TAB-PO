# Methods and metrics

## TAB-PO objective

For a prompt and chosen/rejected completion pair, let `s+` and `s-` be sums of completion-token log probabilities, and let `s_ref+`, `s_ref-` use our frozen merged SFT model:

```text
L_pref = -log sigmoid(beta * [(s+ - s-) - (s_ref+ - s_ref-)])
gate_t = stop_gradient(1[p_policy(chosen_t | prefix) < tau])
L_barrier = sum_t gate_t * [-log p_policy(chosen_t | prefix)] / max(1, sum_t gate_t)
L_TABPO = L_pref + lambda * L_barrier
```

The denominator counts selected, nonmasked chosen tokens. An empty gate gives zero barrier. Our default pipeline uses `beta=0.5`, `tau=0.87`, and `lambda=0.5`; `lambda=0` gives DPO on the same preferences. [objective.py](../tabpo/objective.py) implements the loss used by the training loop.

The confidence gate selects under-confident tokens within the chosen completion mask, including ontology, serialization, and end-of-turn tokens. Ontology tokens receive the barrier whenever their policy probability falls below `tau`. Gate decisions and reference scores are detached from gradients. Padding and prompt tokens contribute to neither term. The barrier is a soft likelihood penalty.

The training script also exposes token weighting, length normalization, and class balancing. These extensions are off by default. Reference log probabilities are cached before optimization using the same completion mask and frozen SFT weights.

## Prompt formatting and context

SFT, preferences, and inference use the selected model's chat template with the prepared query as one user message. Chosen/rejected JSON is compactly serialized. Fast-tokenizer character offsets mask the prompt and retain completion tokens without inserting a second BOS token. SFT and TAB-PO use the same [formatting helper](../tabpo/prompting.py).

Context overflow raises an error. Preflight reserves 4,096 generation tokens after each input prompt and checks complete training/validation gold chats. Generated preferences are checked again during collation. Increase the context limit if necessary, within the model's supported context and available memory. Test inputs are inspected only for context budgeting.

## Confusion-aware preference construction

We construct **100% of preference pairs automatically**, with no expert disambiguation or adjudicated component in this repository. Post-SFT inference runs on the 53 held-out validation documents. The confusion builder verifies the complete validation log against canonical queries and gold annotations before extracting entity- and relation-type confusion counts.

Gold chosen outputs come from the selected training or validation source split. Our generator uses four negative families: label confusions, missing records, extra relations, and relation relinking. Vulnerable records and replacement labels are sampled using validation confusion counts with additive smoothing of 1.0. Relation alignment requires both ordered endpoint matches to pass the relaxed threshold independently.

Removing an entity also removes incident relations and cleans coreference clusters. Extra relations use existing same-sentence entities and fresh relation IDs. Relinking chooses a different valid same-sentence endpoint and excludes existing gold typed edges. Rejected outputs are schema-valid, distinct from chosen outputs, and unique within a document. Build summaries report realized family counts and any unpaired documents.

Every command-line preference build verifies source-split and confusion provenance. Training and validation preferences use separate document pools. CSV hashes, prediction-log hashes, and the synthetic fraction are recorded. Test data is excluded from preference construction.

## Evaluation metrics

| Paper metric | Output key | Definition |
| --- | --- | --- |
| Entity F1 | `entity_f1` | Micro F1 over each document's set of recognized entity types. |
| Relation F1 | `relation_f1` | Relation type and two relaxed-matching argument texts, with strict argument order for every relation type. |
| Span F1 | `span_f1` | One-to-one mention-text matching by normalized token-set containment in either direction or Jaccard ≥ 0.6. |
| Coreference F1 | `coreference_f1` | Unordered mention pairs induced by clusters, matched one-to-one using relaxed endpoint text and exact sentence IDs. |

Both validation and test tasks use this single evaluation protocol from Appendix F. F1 aggregates TP/FP/FN across documents. Record IDs link records within each output; predicted IDs need not equal gold IDs. Span matching uses deterministic prediction-first greedy assignment; relation/coreference matching uses deterministic descending-score greedy assignment. A zero total denominator returns zero.

The parser supports code fences, balanced-object extraction, Python-literal fallback, and partial-record recovery. Semantic metrics use recognized ontology labels and resolved references. `json_parse_rate` reports responses that parse as objects containing the three array fields; offline evaluation additionally reports `schema_valid_rate`. Entity F1's unit is document-level type presence, Span F1 uses relaxed text matching, and Coreference F1 is pair-based.
