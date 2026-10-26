"""Collect results/*.json into one payload for the dashboard. Only real result files are shown;
cells/models without a result are marked pending / api / n/a / external."""
from __future__ import annotations

import glob
import json
import os
import time

from .refine import ARCH_NAMES, NATIVE
from .zoo import M, REPORTED, ZOO, applicable

DATASET_NAMES = {"litsearch": "LitSearch", "doris_mae": "DORIS-MAE", "irpapers": "IRPAPERS", "maple": "MAPLE"}

ARCH_INFO = {
    "A1": {"paper": "LitSearch (EMNLP 2024)",
           "query": "Natural-language literature-search question, tagged broad / specific",
           "document": "Paper: title + abstract", "answer": "Set of gold papers (binary)",
           "search": "Full corpus", "native_metrics": "R@20 (broad), R@5 / R@20 (specific)"},
    "A2": {"paper": "DORIS-MAE (NeurIPS 2023 D&B)",
           "query": "Complex multi-aspect paragraph (whole, or sentence-by-sentence)",
           "document": "Abstract", "answer": "Graded relevance (0–2 per aspect); relevant iff mean ≥ 1",
           "search": "Re-rank ~100-paper candidate pool", "native_metrics": "R@5, R@20, RP, NDCGexp@10%, MRR@10, MAP"},
    "A3": {"paper": "IRPAPERS (2026)",
           "query": "Page-grounded question", "document": "Page: OCR text or page image",
           "answer": "Exactly one gold page", "search": "Full page collection", "native_metrics": "Recall@1, @5, @20"},
    "A4": {"paper": "MAPLE (2026)",
           "query": "Aspect queries (motivation / method / experiment-result), many per paper",
           "document": "Whole paper: holistic full text (truncated to model max input)",
           "answer": "One target paper per query, grouped by paper", "search": "Full corpus",
           "native_metrics": "AllAspect / AnyAspect / AspectCoverage @k; per-aspect R@k"},
}

DEFAULT_CELL = {  # FRAMEWORK.md §5, shown before a cell has been prepared
    ("A1", "doris_mae"): ("faithful", ["Graded relevance binarised (S/|Q| ≥ 1)", "Searches all 363K abstracts"]),
    ("A1", "irpapers"): ("faithful", ["Pages lifted to papers (title + page 1)", "166-paper corpus"]),
    ("A1", "maple"): ("faithful", ["Title + abstract representation"]),
    ("A2", "litsearch"): ("faithful", ["DORIS-style pools built", "Binary gold → grade 2"]),
    ("A2", "irpapers"): ("faithful", ["Pages lifted to papers", "Pools built"]),
    ("A2", "maple"): ("faithful", ["DORIS-style pools built", "Binary gold → grade 2"]),
    ("A3", "litsearch"): ("approximate", ["Full text → pseudo-pages", "Paper-level gold"]),
    ("A3", "doris_mae"): ("approximate", ["Abstract = one page", "Paper-level gold"]),
    ("A3", "maple"): ("approximate", ["Full text → pseudo-pages", "Paper-level gold"]),
    ("A4", "litsearch"): ("faithful", ["Full text holistic", "Multi-gold → query/target pairs"]),
    ("A4", "doris_mae"): ("approximate", ["Abstract stands in for full text", "Aspects → aspect queries"]),
    ("A4", "irpapers"): ("approximate", ["Pages → whole paper", "Aspect group heuristic from page position"]),
}


def build(results_dir: str) -> dict:
    tasks = {}
    for f in glob.glob(os.path.join(results_dir, "tasks", "*.json")):
        t = json.load(open(f))
        tasks[(t["arch"], t["dataset"])] = t
    res = {}
    for f in glob.glob(os.path.join(results_dir, "*__*__*.json")):
        r = json.load(open(f))
        r.pop("per_query_recall@10", None)
        r.pop("traceback", None)
        res[(r["arch"], r["dataset"], r["model_key"])] = r

    cells = {}
    for arch in ZOO:
        for ds in DATASET_NAMES:
            native = NATIVE[arch] == ds
            t = tasks.get((arch, ds))
            has_img = ds == "irpapers" and arch in ("A3", "A4")
            if t:
                fid, devs, card, stats = t["fidelity"], t["deviations"], t["format_card"], t["stats"]
            else:
                fid, devs = ("native", []) if native else DEFAULT_CELL[(arch, ds)]
                card, stats = None, None
            models = []
            for key in ZOO[arch]:
                r = res.get((arch, ds, key))
                if r is None:
                    st, reason = applicable(arch, key, has_img, ds)
                    r = {"status": "pending" if st == "runnable" else st, "reason": reason}
                r = {k: v for k, v in r.items() if k != "task"}
                r.update(key=key, name=M[key]["name"], tier=M[key]["tier"], ckpt=M[key].get("ckpt"),
                         assumed=bool(M[key].get("assumed")))
                if native and key in REPORTED.get(arch, {}) and "reported" not in r:
                    r["reported"] = dict(zip(REPORTED[arch]["_metrics"], REPORTED[arch][key]))
                models.append(r)
            done = [m for m in models if m["status"] == "done" and m.get("recall@10") is not None]
            best = max(done, key=lambda m: m["recall@10"]) if done else None
            cells[f"{arch}__{ds}"] = {
                "arch": arch, "dataset": ds, "native": native, "fidelity": fid, "deviations": devs,
                "format_card": card, "stats": stats, "prepared": bool(t), "models": models,
                "n_done": len(done), "n_runnable": sum(m["status"] in ("done", "pending", "failed") for m in models),
                "best": {"key": best["key"], "name": best["name"], "recall@10": best["recall@10"]} if best else None}
    return {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "archs": [{"id": a, "name": ARCH_NAMES[a], **ARCH_INFO[a], "native_dataset": NATIVE[a],
                   "zoo": [{"key": k, "name": M[k]["name"], "tier": M[k]["tier"], "ckpt": M[k].get("ckpt"),
                            "assumed": bool(M[k].get("assumed")), "note": M[k].get("note")} for k in ZOO[a]]}
                  for a in ZOO],
        "datasets": [{"id": d, "name": n} for d, n in DATASET_NAMES.items()],
        "cells": cells,
        "n_results": sum(c["n_done"] for c in cells.values()),
    }


if __name__ == "__main__":
    import sys
    d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "results")
    print(json.dumps(build(d), indent=1)[:3000])
