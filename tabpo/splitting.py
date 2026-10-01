"""First-order iterative multilabel stratification with exact split capacities.

Rare labels are assigned first. For each document, select the split with the
largest remaining demand for that label, breaking ties by remaining capacity
and then a seeded RNG. Never split annotations within a document. This follows
the iterative allocation of Sechidis et al. (2011), with a hard capacity guard.
"""
from __future__ import annotations

from collections import Counter
import random


def iterative_split(labels: list[set[str]], validation_size: int, seed: int = 42) -> tuple[list[int], list[int]]:
    n = len(labels)
    if not 0 < validation_size < n:
        raise ValueError("validation_size must leave at least one document in each split")
    rng = random.Random(seed)
    capacity = [n - validation_size, validation_size]
    counts = Counter(label for row in labels for label in row)
    demand = [{label: count * size / n for label, count in counts.items()} for size in capacity]
    remaining = set(range(n))
    assigned = [[], []]
    while remaining:
        active = Counter(label for i in sorted(remaining) for label in sorted(labels[i]))
        if active:
            rare_count = min(active.values())
            rare = rng.choice(sorted(label for label, count in active.items() if count == rare_count))
            candidates = [i for i in sorted(remaining) if rare in labels[i]]
        else:
            rare = None
            candidates = sorted(remaining)
        rng.shuffle(candidates)
        for i in candidates:
            available = [s for s in (0, 1) if capacity[s] > 0]
            if rare is not None:
                best = max(demand[s][rare] for s in available)
                available = [s for s in available if demand[s][rare] == best]
            best_capacity = max(capacity[s] for s in available)
            available = [s for s in available if capacity[s] == best_capacity]
            split = rng.choice(available)
            assigned[split].append(i)
            capacity[split] -= 1
            for label in sorted(labels[i]):
                demand[split][label] -= 1
            remaining.remove(i)
    return sorted(assigned[0]), sorted(assigned[1])
