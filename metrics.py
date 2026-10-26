"""Metrics. Every architecture gets the unified Recall@10 plus its paper's native metrics.

A ranking is a list of doc_ids (best first). qrels map qid -> {gold_id: grade}.
For gold_level == "paper", retrieved page ids are mapped to their paper before matching.
"""
from __future__ import annotations

import math
import random
from typing import Dict, List, Optional

import numpy as np


def _hits(ranking: List[str], gold: Dict[str, float], doc2paper: Optional[Dict[str, str]]) -> List[str]:
    """Gold items covered by the ranking, in rank order, each counted once."""
    seen, out = set(), []
    for d in ranking:
        key = doc2paper.get(d, d) if doc2paper else d
        if key in gold and key not in seen:
            seen.add(key)
            out.append(key)
    return out


def recall_at(ranking, gold, k, doc2paper=None) -> float:
    """Unified Recall@k = |Gold ∩ Top-k| / |Gold| (paper-level dedup when doc2paper is given)."""
    if not gold:
        return float("nan")
    return len(_hits(ranking[:k], gold, doc2paper)) / len(gold)


def mrr_at(ranking, gold, k) -> float:
    for i, d in enumerate(ranking[:k]):
        if d in gold:
            return 1.0 / (i + 1)
    return 0.0


def r_precision(ranking, gold) -> float:
    r = len(gold)
    return sum(1 for d in ranking[:r] if d in gold) / r if r else float("nan")


def average_precision(ranking, gold) -> float:
    if not gold:
        return float("nan")
    hits, s = 0, 0.0
    for i, d in enumerate(ranking):
        if d in gold:
            hits += 1
            s += hits / (i + 1)
    return s / len(gold)


def ndcg_exp_pct(ranking, grades: Dict[str, float], pool_size: int, p: float = 0.1) -> float:
    """DORIS-MAE NDCG_exp@p%: gain 2^rel / log2(i+2), cutoff = ceil(p * pool size)."""
    k = max(1, math.ceil(p * pool_size))
    dcg = sum((2 ** grades.get(d, 0.0)) / math.log2(i + 2) for i, d in enumerate(ranking[:k]))
    ideal = sorted(grades.values(), reverse=True)[:k]
    ideal += [0.0] * (k - len(ideal))
    idcg = sum((2 ** g) / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else float("nan")


def mean(xs) -> float:
    xs = [x for x in xs if not (isinstance(x, float) and math.isnan(x))]
    return float(np.mean(xs)) if xs else float("nan")


def bootstrap_se(values: List[float], n: int = 1000, seed: int = 0) -> float:
    vals = [v for v in values if not math.isnan(v)]
    if len(vals) < 2:
        return float("nan")
    rng = random.Random(seed)
    means = [np.mean([vals[rng.randrange(len(vals))] for _ in vals]) for _ in range(n)]
    return float(np.std(means))


def pct(x: float) -> Optional[float]:
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(100 * x, 2)
