"""Architecture-specific evaluation. Each returns {"recall@10": ..., "native": {...}, "breakdown": {...}}
with all numbers in percent, plus per-query Recall@10."""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List

from .metrics import (average_precision, bootstrap_se, mean, mrr_at, ndcg_exp_pct, pct, r_precision,
                      recall_at)
from .schema import Task


def _ids(r):
    return [d for d, _ in r]


def _recalls(task: Task, rankings, ks=(1, 5, 10, 20, 50, 100)):
    d2p = task.doc2paper if task.gold_level == "paper" else None
    per = {}
    for q in task.queries:
        rk = _ids(rankings.get(q.qid, []))
        per[q.qid] = {k: recall_at(rk, task.qrels[q.qid], k, d2p) for k in ks}
    return per


def eval_A1(task: Task, rankings) -> dict:
    per = _recalls(task, rankings)
    broad = [q for q in task.queries if q.meta.get("specificity") == 0]
    spec = [q for q in task.queries if q.meta.get("specificity") == 1]
    native = {"broad_R@20": pct(mean(per[q.qid][20] for q in broad)),
              "specific_R@5": pct(mean(per[q.qid][5] for q in spec)),
              "specific_R@20": pct(mean(per[q.qid][20] for q in spec)),
              "n_broad": len(broad), "n_specific": len(spec)}
    breakdown = {}
    for sub in sorted({q.meta.get("subset", "all") for q in task.queries}):
        qs = [q for q in task.queries if q.meta.get("subset", "all") == sub]
        breakdown[sub] = {
            "broad_R@20": pct(mean(per[q.qid][20] for q in qs if q.meta.get("specificity") == 0)),
            "specific_R@5": pct(mean(per[q.qid][5] for q in qs if q.meta.get("specificity") == 1)),
            "specific_R@20": pct(mean(per[q.qid][20] for q in qs if q.meta.get("specificity") == 1)),
            "R@10": pct(mean(per[q.qid][10] for q in qs)), "n": len(qs)}
    return _pack(per, native, breakdown)


def eval_A2(task: Task, rankings) -> dict:
    rows = {}
    skipped = 0
    for q in task.queries:
        grades = task.qrels[q.qid]
        rel = {d: g for d, g in grades.items() if g >= task.relevant_threshold}
        if not rel:
            skipped += 1
            continue
        rk = _ids(rankings[q.qid])
        # DORIS-MAE conventions (calibrated against the paper's random baseline: NDCGexp 7.31, MRR@10 3.59):
        #  - NDCG gain uses the summed aspect score S(p|Q) = mean grade x number of aspect units;
        #  - MRR@10 is the reciprocal rank of the top-scored abstract(s) in the pool.
        units = q.meta.get("n_aspect_units") or 1
        summed = {d: g * units for d, g in grades.items()}
        top = max(grades.values())
        best = {d: 1.0 for d, g in grades.items() if g == top and g > 0}
        rows[q.qid] = {"R@5": recall_at(rk, rel, 5), "R@10": recall_at(rk, rel, 10), "R@20": recall_at(rk, rel, 20),
                       "RP": r_precision(rk, rel), "NDCGexp@10%": ndcg_exp_pct(rk, summed, len(task.pools[q.qid])),
                       "MRR@10": mrr_at(rk, best, 10), "MAP": average_precision(rk, rel)}
    native = {}
    for m in ["R@5", "R@20", "RP", "NDCGexp@10%", "MRR@10", "MAP"]:
        vals = [r[m] for r in rows.values()]
        native[m] = pct(mean(vals))
        native[m + "_se"] = pct(bootstrap_se(vals))
    native["n_queries_with_relevant"] = len(rows)
    native["n_queries_without_relevant"] = skipped
    per = {qid: {10: r["R@10"], 5: r["R@5"], 20: r["R@20"]} for qid, r in rows.items()}
    return _pack(per, native, {})


def eval_A3(task: Task, rankings) -> dict:
    per = _recalls(task, rankings)
    native = {f"R@{k}": pct(mean(p[k] for p in per.values())) for k in (1, 5, 20)}
    return _pack(per, native, {})


def eval_A4(task: Task, rankings) -> dict:
    per = _recalls(task, rankings)
    by_paper: Dict[str, List[str]] = defaultdict(list)
    for q in task.queries:
        by_paper[task.groups[q.qid]["target"]].append(q.qid)
    native = {}
    for k in (5, 10, 20):
        hit = {qid: per[qid][k] >= 1.0 for qid in per}
        native[f"AllAspect@{k}"] = pct(mean(float(all(hit[x] for x in qs)) for qs in by_paper.values()))
        native[f"AnyAspect@{k}"] = pct(mean(float(any(hit[x] for x in qs)) for qs in by_paper.values()))
        native[f"AspectCoverage@{k}"] = pct(mean(mean(float(hit[x]) for x in qs) for qs in by_paper.values()))
    native["n_target_papers"] = len(by_paper)
    native["avg_queries_per_paper"] = round(len(task.queries) / max(1, len(by_paper)), 2)
    breakdown = {}
    for g in sorted({task.groups[q.qid]["aspect_group"] for q in task.queries}):
        qs = [q.qid for q in task.queries if task.groups[q.qid]["aspect_group"] == g]
        breakdown[g] = {f"R@{k}": pct(mean(per[x][k] for x in qs)) for k in (5, 10, 20)}
        breakdown[g]["n"] = len(qs)
    mm = [q.qid for q in task.queries if q.meta.get("is_multimodal")]
    if mm:
        txt = [q.qid for q in task.queries if not q.meta.get("is_multimodal")]
        breakdown["text-referenced"] = {"R@10": pct(mean(per[x][10] for x in txt)), "n": len(txt)}
        breakdown["multimodal-referenced"] = {"R@10": pct(mean(per[x][10] for x in mm)), "n": len(mm)}
    return _pack(per, native, breakdown)


def _pack(per, native, breakdown) -> dict:
    r10 = [p[10] for p in per.values()]
    return {"recall@10": pct(mean(r10)), "recall@10_se": pct(bootstrap_se(r10)), "native": native,
            "breakdown": breakdown,
            "per_query_recall@10": {q: (None if math.isnan(p[10]) else round(p[10], 4)) for q, p in per.items()}}


EVALUATORS = {"A1": eval_A1, "A2": eval_A2, "A3": eval_A3, "A4": eval_A4}


def evaluate(task: Task, rankings) -> dict:
    return EVALUATORS[task.arch](task, rankings)
