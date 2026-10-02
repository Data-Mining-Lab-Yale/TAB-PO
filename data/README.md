# Prepared SciERC data

We partition our **350 prepared training documents** into **297 training and 53 validation documents**, with seed 42. The separate **100-document test partition** retains its original queries, contexts, annotations, and document order. All three Arrow files contain only the SciERC fields `query`, `answer`, and `context`. Validation is a subset of the prepared training pool.

## Partition protocol

[prepare_data.py](../scripts/prepare_data.py) selects the single top-level `data-00000-of-00001.arrow` member of each archive. It ignores nested historical split files and filesystem artifacts. It checks gold schema validity, duplicate queries, and training/test separation before partitioning.

[iterative_split](../tabpo/splitting.py) implements first-order iterative multilabel stratification following the rare-label allocation principle of Sechidis, Tsoumakas, and Vlahavas (2011), with exact capacities. Each document has presence indicators for six entity types, seven relation types, and whether it has any coreference link. The algorithm repeatedly allocates documents with the rarest remaining label to the split with greatest remaining demand for that label, breaks ties by remaining capacity and a seeded RNG, and updates all labels on that document. Unlabelled documents fill remaining capacities. Sorted traversal makes seeded ties repeatable. This is an explicitly implemented capacity-constrained variant, not a claim of identical membership to another library's implementation.

Complete prepared documents stay together; annotations are never split. First-order stratification approximates label marginals and does not promise exact label counts or preserve every higher-order label combination. The table below makes residual imbalance inspectable.

The manifest records original Arrow hashes, source row hashes, split indices, each split's query hashes, label prevalence, and file hashes. [verify_splits.py](../scripts/verify_splits.py) reconstructs the original pool, repeats the split algorithm, verifies the exact membership, and checks all saved statistics. Regeneration validates the prepared annotations and reproduces document membership.

## Files and checksums

| Split | Documents | SHA-256 |
| --- | ---: | --- |
| train | 297 | `e712e1642f3f286427788cebbd9a099fae769a51b8e039226c0ddbd7eaa30b4a` |
| validation | 53 | `e6ed22095af1a5b006e5a7290d681cd1afe6ab70096fa65a03e58b0d2c4987a2` |
| test | 100 | `e155b58498c1deb3c0fac0428d43d0aab68d3410e5fda90850401c994aa4debd` |

See [split_manifest.json](scierc/split_manifest.json) for source checksums, document identities, split indices, schema, and label statistics.

## Document-level stratification labels

Counts denote documents containing the label, not annotation instances.

| Indicator | Train (297) | Validation (53) | Test (100) |
| --- | ---: | ---: | ---: |
| entity:Generic | 256 | 48 | 82 |
| entity:Material | 184 | 29 | 67 |
| entity:Method | 271 | 52 | 96 |
| entity:Metric | 93 | 17 | 38 |
| entity:OtherScientificTerm | 264 | 43 | 93 |
| entity:Task | 254 | 45 | 88 |
| has_coreference | 277 | 51 | 91 |
| relation:Compare | 106 | 19 | 29 |
| relation:Conjunction | 171 | 30 | 55 |
| relation:Evaluate-for | 125 | 22 | 48 |
| relation:Feature-of | 94 | 17 | 38 |
| relation:Hyponym-of | 120 | 22 | 28 |
| relation:Part-of | 89 | 16 | 35 |
| relation:Used-for | 286 | 51 | 96 |

Test labels are shown for transparency and were not used in split allocation or preference construction.

## Schema and inspection

Every Arrow row has three string columns: `query`, `answer`, and `context`. The query contains the complete prepared instruction and document; the answer is a JSON object with `entities`, `relations`, and `coreference_clusters`.

Entities use unique string IDs, nonempty text, an ontology type, and a nonnegative integer sentence ID. Relations point to distinct existing entities in the same sentence and use unique relation IDs. Coreference clusters contain at least two existing entity IDs; members occur in at most one cluster. [tabpo/data.py](../tabpo/data.py) specifies the contract. All 450 gold answers pass these structural checks. These checks cannot certify annotation correctness or that every evidence string occurs in the document.

```python
from tabpo.data import read_arrow
rows = read_arrow("data/scierc/validation.arrow")
print(len(rows))  # 53
print(rows[0].keys())
```

## Regenerate into a separate directory

The original archives are deliberately excluded from the public release. If you already possess them:

```bash
python scripts/prepare_data.py \
  --train-archive /path/to/scierc_train_arrow.zip \
  --test-archive /path/to/scierc_test_arrow.zip \
  --output-dir /tmp/scierc-regenerated --seed 42 --validation-size 53
python scripts/verify_splits.py --data-dir /tmp/scierc-regenerated
```

Arrow serialization can vary between library versions. We preserve and verify the held-out test's queries, contexts, annotations, and row order through source-row hashes; dropping metadata changes file bytes. Audit reconstructed identities and memberships as well as checksums. The main pipeline deliberately uses the checked-in canonical split; changing it requires updating its task YAML paths and provenance together.

## Data boundaries

SFT uses training data with a separate validation dataset. Post-SFT inference for confusion extraction uses exactly the 53 validation documents. The confusion builder rejects duplicate, incomplete, or foreign/test prediction logs. Preference train and validation sets come from distinct document pools, and provenance checks reject test Arrow even if renamed. Test generation runs after both model training stages.

The original publication is [Luan et al. (2018)](https://aclanthology.org/D18-1360/).
