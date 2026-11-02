"""Cross-audit analysis over results/*.json  ->  site/analysis.json (read by the dashboard).

    python -m xaudit.analysis            (or: python xaudit/analysis.py)

Pure numpy; no model code is imported, so it runs anywhere. Sections:
  validation  diagonal cells vs the papers' reported numbers (gate = within +-3 points)
  matrix      Recall@10 per model per cell, BM25 baseline, best model
  ranking     Kendall tau between an architecture's native ranking and its rankings on the other datasets
  format      the same model + dataset scored under each architecture's format (BM25 is in all 16 cells)
  significance  paired sign-flip permutation tests on per-query Recall@10 (adjacent ranks, best vs BM25),
                Holm-corrected within each cell; 'top tier' = models not significantly below the best
"""
from __future__ import annotations

import glob
import importlib.util
import json
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("_zoo", os.path.join(HERE, "zoo.py"))
zoo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(zoo)

ARCHS = ["A1", "A2", "A3", "A4"]
ARCH_NAMES = {"A1": "LitSearch", "A2": "DORIS-MAE", "A3": "IRPAPERS", "A4": "MAPLE"}
DATASETS = ["litsearch", "doris_mae", "irpapers", "maple"]
DS_NAMES = {"litsearch": "LitSearch", "doris_mae": "DORIS-MAE", "irpapers": "IRPAPERS", "maple": "MAPLE"}
NATIVE = {"A1": "litsearch", "A2": "doris_mae", "A3": "irpapers", "A4": "maple"}
GATE = 3.0
N_PERM = 20000
ALPHA = 0.05


def load(results_dir):
    res = {}
    for f in glob.glob(os.path.join(results_dir, "*__*__*.json")):
        r = json.load(open(f))
        res[(r["arch"], r["dataset"], r["model_key"])] = r
    return res


def name(key):
    return zoo.M[key]["name"]


# ---------------------------------------------------------------- stats helpers
def kendall_tau_b(x, y):
    n = len(x)
    conc = disc = tx = ty = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx, dy = np.sign(x[i] - x[j]), np.sign(y[i] - y[j])
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                tx += 1
            elif dy == 0:
                ty += 1
            elif dx == dy:
                conc += 1
            else:
                disc += 1
    den = np.sqrt((conc + disc + tx) * (conc + disc + ty))
    return float((conc - disc) / den) if den else None


def perm_test(a, b, rng):
    """Two-sided paired sign-flip permutation test on per-query differences."""
    d = np.asarray(a, float) - np.asarray(b, float)
    obs = abs(d.mean())
    if not np.any(d):
        return 1.0
    signs = rng.choice([-1.0, 1.0], size=(N_PERM, len(d)))
    null = np.abs((signs * d).mean(axis=1))
    return float((np.sum(null >= obs - 1e-12) + 1) / (N_PERM + 1))


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj = [0.0] * len(ps)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


def paired(r1, r2):
    p1, p2 = r1.get("per_query_recall@10") or {}, r2.get("per_query_recall@10") or {}
    common = [k for k in p1 if k in p2]
    if len(common) < 0.95 * max(len(p1), len(p2), 1) or not common:
        return None
    return [p1[k] for k in common], [p2[k] for k in common]


# ---------------------------------------------------------------- sections
def validation(res):
    out = []
    for a in ARCHS:
        ds = NATIVE[a]
        rep = zoo.REPORTED.get(a, {})
        mets = rep.get("_metrics", [])
        for key in zoo.ZOO[a]:
            if key not in rep:
                continue
            r = res.get((a, ds, key))
            row = {"arch": a, "dataset": ds, "key": key, "name": name(key), "tier": zoo.M[key]["tier"]}
            if not r or r.get("status") != "done":
                row["status"] = "not_run" if zoo.M[key]["tier"] in ("api", "external") else "pending"
                out.append(row)
                continue
            nat = r.get("native", {})
            cmp_ = []
            for m, v in zip(mets, rep[key]):
                got = nat.get(m)
                if got is None:
                    continue
                cmp_.append({"metric": m, "reported": v, "measured": round(got, 2), "diff": round(got - v, 2)})
            worst = max((abs(c["diff"]) for c in cmp_), default=None)
            signed = max(cmp_, key=lambda c: abs(c["diff"]))["diff"] if cmp_ else None
            row.update(status="done", metrics=cmp_, max_abs_diff=worst, worst_signed=signed,
                       direction=(None if signed is None or abs(signed) <= GATE else ("higher" if signed > 0 else "lower")),
                       verdict=("pass" if worst is not None and worst <= GATE else "outside"),
                       note=r.get("note", ""))
            out.append(row)
    return out


def matrix(res):
    cells = {}
    for a in ARCHS:
        for ds in DATASETS:
            ms = []
            for key in zoo.ZOO[a]:
                r = res.get((a, ds, key))
                if r and r.get("status") == "done" and r.get("recall@10") is not None:
                    ms.append({"key": key, "name": name(key), "r10": r["recall@10"],
                               "se": r.get("recall@10_se"), "has_pq": bool(r.get("per_query_recall@10"))})
            ms.sort(key=lambda m: -m["r10"])
            bm = next((m["r10"] for m in ms if m["key"] == "bm25"), None)
            pending = []
            for key in zoo.ZOO[a]:
                r = res.get((a, ds, key))
                st, _ = zoo.applicable(a, key, ds == "irpapers" and a in ("A3", "A4"), ds)
                if st == "runnable" and (r is None or r.get("status") not in ("done", "too_large")):
                    pending.append(name(key))
            cells[f"{a}__{ds}"] = {"arch": a, "dataset": ds, "native": NATIVE[a] == ds, "models": ms,
                                   "bm25": bm, "best": ms[0] if ms else None,
                                   "best_gain_over_bm25": round(ms[0]["r10"] - bm, 2) if ms and bm is not None else None,
                                   "pending": pending}
    return cells


def ranking(cells):
    out = []
    for a in ARCHS:
        nat = {m["key"]: m["r10"] for m in cells[f"{a}__{NATIVE[a]}"]["models"]}
        nat_best = max(nat, key=nat.get) if nat else None
        for ds in DATASETS:
            if ds == NATIVE[a]:
                continue
            oth = {m["key"]: m["r10"] for m in cells[f"{a}__{ds}"]["models"]}
            common = [k for k in nat if k in oth]
            row = {"arch": a, "dataset": ds, "n_models": len(common), "few_models": len(common) < 4}
            if len(common) >= 3:
                row["tau"] = round(kendall_tau_b([nat[k] for k in common], [oth[k] for k in common]), 3)
                ob = max(common, key=lambda k: oth[k])
                nb = max(common, key=lambda k: nat[k])
                row.update(native_best=name(nb), foreign_best=name(ob), winner_kept=(ob == nb))
                # largest rank moves
                rn = {k: i for i, k in enumerate(sorted(common, key=lambda k: -nat[k]))}
                ro = {k: i for i, k in enumerate(sorted(common, key=lambda k: -oth[k]))}
                moves = sorted(common, key=lambda k: -abs(rn[k] - ro[k]))[:3]
                row["biggest_moves"] = [{"name": name(k), "native_rank": rn[k] + 1, "rank": ro[k] + 1}
                                        for k in moves if rn[k] != ro[k]]
            out.append(row)
    return out


def format_effect(res):
    """Same checkpoint on the same dataset under different architecture formats."""
    groups = {"BM25": {a: "bm25" for a in ARCHS},
              "E5-large-v2": {"A1": "e5-large-v2", "A2": "e5-large-v2-doris"},
              "Instructor-XL": {"A1": "instructor-xl-litsearch", "A4": "instructor-xl-maple"},
              "GritLM-7B": {"A1": "gritlm-7b-litsearch", "A4": "gritlm-7b-maple"}}
    out = []
    for g, km in groups.items():
        for ds in DATASETS:
            row = {"model": g, "dataset": ds, "by_arch": {}}
            for a, key in km.items():
                r = res.get((a, ds, key))
                if r and r.get("status") == "done":
                    row["by_arch"][a] = r["recall@10"]
            if len(row["by_arch"]) >= 2:
                v = list(row["by_arch"].values())
                row["spread"] = round(max(v) - min(v), 2)
                out.append(row)
    return out


def significance(res, cells, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for cid, c in cells.items():
        ms = c["models"]
        if len(ms) < 2:
            continue
        tests = []
        for i in range(len(ms) - 1):
            tests.append(("adjacent", ms[i], ms[i + 1]))
        bm = next((m for m in ms if m["key"] == "bm25"), None)
        if bm and ms[0]["key"] != "bm25" and (len(ms) < 2 or ms[1]["key"] != "bm25"):
            tests.append(("best_vs_bm25", ms[0], bm))
        best = ms[0]
        for m in ms[1:]:
            if not any(t[1] is best and t[2] is m for t in tests):
                tests.append(("best_vs", best, m))
        rows, ps = [], []
        for kind, m1, m2 in tests:
            pr = paired(res[(c["arch"], c["dataset"], m1["key"])], res[(c["arch"], c["dataset"], m2["key"])])
            row = {"kind": kind, "a": m1["name"], "b": m2["name"], "diff": round(m1["r10"] - m2["r10"], 2)}
            if pr is None:
                row["p"] = None
                row["reason"] = "per-query scores unavailable"
            else:
                row["n"] = len(pr[0])
                row["p"] = perm_test(pr[0], pr[1], rng)
                ps.append(row["p"])
            rows.append(row)
        adj = holm(ps) if ps else []
        it = iter(adj)
        for row in rows:
            if row["p"] is not None:
                row["p_holm"] = round(next(it), 4)
                row["p"] = round(row["p"], 4)
                row["significant"] = row["p_holm"] < ALPHA
        tier = [best["name"]] if best["has_pq"] else None
        for row in (rows if tier else []):
            if row["kind"] in ("best_vs", "adjacent", "best_vs_bm25") and row["a"] == best["name"] \
                    and row.get("p_holm") is not None and not row["significant"] and row["b"] not in tier:
                tier.append(row["b"])
        untested = [r["b"] for r in rows if r["a"] == best["name"] and r["p"] is None] if tier else []
        out[cid] = {"tests": rows, "top_tier": tier, "top_tier_untested": untested,
                    "untestable": [m["name"] for m in ms if not m["has_pq"]]}
    return out


def headline(val, cells, rank, fmt):
    done = [v for v in val if v.get("status") == "done"]
    passed = [v for v in done if v["verdict"] == "pass"]
    taus = [r["tau"] for r in rank if r.get("tau") is not None and not r["few_models"]]
    kept = [r for r in rank if r.get("winner_kept") is not None]
    bm = [f for f in fmt if f["model"] == "BM25"]
    return {
        "n_results": sum(len(c["models"]) for c in cells.values()),
        "n_pending": sum(len(c["pending"]) for c in cells.values()),
        "validation": {"checked": len(done), "passed": len(passed),
                       "outside_higher": sum(v.get("direction") == "higher" for v in done),
                       "outside_lower": sum(v.get("direction") == "lower" for v in done),
                       "outside": [f'{v["name"]} ({ARCH_NAMES[v["arch"]]})' for v in done if v["verdict"] != "pass"]},
        "ranking": {"mean_tau": round(float(np.mean(taus)), 3) if taus else None,
                    "min_tau": min(taus) if taus else None, "max_tau": max(taus) if taus else None,
                    "winner_kept": f"{sum(r['winner_kept'] for r in kept)}/{len(kept)}"},
        "bm25_format_spread": {DS_NAMES[f["dataset"]]: f["spread"] for f in bm},
    }


def build(results_dir):
    res = load(results_dir)
    val = validation(res)
    cells = matrix(res)
    rank = ranking(cells)
    fmt = format_effect(res)
    sig = significance(res, cells)
    return {"generated": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "gate": GATE, "alpha": ALPHA, "n_perm": N_PERM,
            "arch_names": ARCH_NAMES, "dataset_names": DS_NAMES, "native": NATIVE,
            "headline": headline(val, cells, rank, fmt),
            "validation": val, "cells": cells, "ranking": rank, "format": fmt, "significance": sig}


if __name__ == "__main__":
    root = os.path.dirname(HERE)
    out = build(os.path.join(root, "results"))
    p = os.path.join(root, "site", "analysis.json")
    json.dump(out, open(p, "w"), indent=1)
    print("wrote", p)
    print(json.dumps(out["headline"], indent=1))
