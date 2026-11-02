"""Run the 16-cell cross-audit.

Examples
  python -m xaudit.run --prepare                                  # refine all 16 cells, write task cards
  python -m xaudit.run --arch A1 --dataset litsearch --model bm25 # one model in one cell
  python -m xaudit.run --arch all --dataset all --tier local       # everything that fits a 6 GB GPU
  python -m xaudit.run --arch all --dataset all --tier kaggle      # the 7–8B models (run on Kaggle)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import time
import traceback

from .encoders import make_runner
from .evaluate import evaluate
from .refine import ARCH_NAMES, refine
from .sources import CACHE, DATASETS, load_source
from .zoo import M, REPORTED, ZOO, applicable

ROOT = os.path.join(os.path.dirname(__file__), "..")
RESULTS = os.environ.get("XAUDIT_RESULTS", os.path.join(ROOT, "results"))


def _dense_pool_fn():
    """Dense part of DORIS-style pooling: bge-small-en-v1.5 (not in any architecture's zoo)."""
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except Exception:
        return None

    def build(abstracts):
        m = SentenceTransformer("BAAI/bge-small-en-v1.5")
        ids = list(abstracts)
        D = m.encode([abstracts[i] for i in ids], batch_size=128, normalize_embeddings=True, show_progress_bar=True)

        def search(query, k):
            q = m.encode(["Represent this sentence for searching relevant passages: " + query],
                         normalize_embeddings=True)[0]
            s = D @ q
            return [ids[j] for j in np.argsort(-s)[:k]]
        return search
    return build


_MEMO = {}


def get_task(arch: str, dataset: str, max_pages_per_paper: int = 0, rebuild: bool = False):
    # The page cap only matters for A3; and if an uncapped (more faithful) A3 cell already exists, reuse it
    # instead of rebuilding (rebuilding the big page cells takes ~30 min and several GB).
    if arch != "A3":
        max_pages_per_paper = 0
    elif os.path.exists(os.path.join(CACHE, f"cap_{arch}_{dataset}.txt")):
        # memory fallback for the very large page cells (cap recorded as a deviation by refine_A3)
        max_pages_per_paper = int(open(os.path.join(CACHE, f"cap_{arch}_{dataset}.txt")).read().strip() or 6)
    elif max_pages_per_paper and os.path.exists(os.path.join(CACHE, f"task_{arch}_{dataset}_mp0.pkl")):
        max_pages_per_paper = 0
    mk = (arch, dataset, max_pages_per_paper)
    if mk in _MEMO and not rebuild:
        return _MEMO[mk]
    _MEMO.clear()   # keep only one (possibly large) task in memory
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"task_{arch}_{dataset}_mp{max_pages_per_paper}.pkl")
    src = None
    if os.path.exists(path) and not rebuild:
        with open(path, "rb") as f:
            task = pickle.load(f)
    else:
        src = load_source(dataset, need_fulltext=arch in ("A3", "A4"))
        task = refine(arch, src, dense_pool_fn=_dense_pool_fn() if arch == "A2" else None,
                      max_pages_per_paper=max_pages_per_paper)
        loader, task.page_image_loader = task.page_image_loader, None
        with open(path, "wb") as f:
            pickle.dump(task, f, protocol=pickle.HIGHEST_PROTOCOL)
        task.page_image_loader = loader
    if task.has_page_images and task.page_image_loader is None:
        from .sources import _irpapers_image_loader
        task.page_image_loader = _irpapers_image_loader()
    _MEMO[mk] = task
    return task


def task_card(task) -> dict:
    return {"arch": task.arch, "arch_name": ARCH_NAMES[task.arch], "dataset": task.dataset,
            "fidelity": task.fidelity, "deviations": task.deviations, "format_card": task.format_card,
            "stats": task.stats(), "unit": task.unit, "gold_level": task.gold_level,
            "has_page_images": task.has_page_images}


def _best_modes(evals: dict) -> dict:
    """DORIS-MAE reports, per metric, the better of whole-query and sentence-by-sentence processing."""
    if len(evals) == 1:
        return next(iter(evals.values()))
    out = json.loads(json.dumps(next(iter(evals.values()))))
    for key in out["native"]:
        vals = [e["native"].get(key) for e in evals.values()]
        vals = [v for v in vals if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v))]
        if vals and not key.endswith("_se") and not key.startswith("n_"):
            out["native"][key] = max(vals)
    best = max(evals, key=lambda m: evals[m]["recall@10"] or -1)
    out["recall@10"] = evals[best]["recall@10"]
    out["per_query_recall@10"] = evals[best]["per_query_recall@10"]
    out["modes"] = {m: {"recall@10": e["recall@10"], "native": e["native"]} for m, e in evals.items()}
    return out


def _free(runner):
    """Release the model and GPU memory before the next model (models otherwise pile up in VRAM and spill
    into slow shared memory on a 6 GB card)."""
    import gc
    for attr in ("_m", "_bert", "_lin", "_tok", "_p", "parts", "text", "img"):
        if hasattr(runner, attr):
            try:
                delattr(runner, attr)
            except Exception:
                pass
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def run_one(arch, dataset, key, args) -> dict:
    task = get_task(arch, dataset, args.max_pages_per_paper)
    status, reason = applicable(arch, key, task.has_page_images, dataset)
    rec = {"arch": arch, "arch_name": ARCH_NAMES[arch], "dataset": dataset, "model_key": key,
           "model_name": M[key]["name"], "tier": M[key]["tier"], "status": status, "reason": reason,
           "task": task_card(task)}
    if status != "runnable":
        return rec
    t0 = time.time()
    runner = make_runner(key, arch, batch_size=args.batch_size, max_len=args.max_len, k=args.k)
    evals = {}
    for mode in runner.modes():
        rankings = runner.rank(task, mode)
        if M[key]["kind"] == "mm_hybrid":
            sweep = runner.extra["alpha_sweep_rankings"]
            scored = {name: evaluate(task, r) for name, r in sweep.items()}
            best = max((n for n in scored if n.startswith("rsf")), key=lambda n: scored[n]["native"]["R@1"] or -1)
            rec["alpha_sweep"] = {n: {"recall@10": s["recall@10"], **s["native"]} for n, s in scored.items()}
            rec["headline_config"] = best
            evals[mode] = scored[best]
        elif M[key]["kind"] == "muvera":
            sweep = runner.extra["ef_sweep_rankings"]
            rec["ef_sweep"] = {ef: {"recall@10": evaluate(task, r)["recall@10"], **evaluate(task, r)["native"]}
                               for ef, r in sweep.items()}
            evals[mode] = evaluate(task, rankings)
        else:
            evals[mode] = evaluate(task, rankings)
    rec.update(_best_modes(evals))
    _free(runner)
    rec.update(status="done", config=runner.config, runtime_s=round(time.time() - t0, 1),
               timestamp=time.strftime("%Y-%m-%d %H:%M:%S"))
    if dataset == {"A1": "litsearch", "A2": "doris_mae", "A3": "irpapers", "A4": "maple"}[arch]:
        ref = REPORTED.get(arch, {})
        if key in ref:
            rec["reported"] = dict(zip(ref["_metrics"], ref[key]))
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", default="all", help="A1|A2|A3|A4|all")
    ap.add_argument("--dataset", default="all", help="litsearch|doris_mae|irpapers|maple|all")
    ap.add_argument("--model", default="all", help="model key from xaudit/zoo.py, or all")
    ap.add_argument("--tier", default="local", help="local|kaggle|all (used when --model all)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=None, help="override a model's max input tokens")
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--max-pages-per-paper", type=int, default=0, help="A3 pseudo-page cap (0 = all)")
    ap.add_argument("--prepare", action="store_true", help="only refine cells and write task cards")
    ap.add_argument("--force", action="store_true", help="re-run even if a result file exists")
    args = ap.parse_args()

    archs = ["A1", "A2", "A3", "A4"] if args.arch == "all" else [args.arch]
    datasets = DATASETS if args.dataset == "all" else [args.dataset]
    os.makedirs(os.path.join(RESULTS, "tasks"), exist_ok=True)
    # cheapest cells first, so the most results land soonest; the huge page/abstract cells run last
    COST = {("A3", "maple"): 100, ("A3", "litsearch"): 90, ("A3", "doris_mae"): 80, ("A1", "doris_mae"): 70,
            ("A4", "doris_mae"): 70, ("A4", "maple"): 60, ("A4", "litsearch"): 50, ("A1", "maple"): 40,
            ("A1", "litsearch"): 30}
    pairs = sorted(((a, d) for a in archs for d in datasets), key=lambda p: COST.get(p, 0))
    for arch, ds in pairs:
        if True:
            try:
                task = get_task(arch, ds, args.max_pages_per_paper, rebuild=args.prepare and args.force)
            except FileNotFoundError as e:
                print(f"[skip] {arch} x {ds}: {e}")
                continue
            with open(os.path.join(RESULTS, "tasks", f"{arch}__{ds}.json"), "w") as f:
                json.dump(task_card(task), f, indent=1)
            print(f"[cell] {arch} ({ARCH_NAMES[arch]}) x {ds}: {task.fidelity}, {task.stats()}")
            if args.prepare:
                continue
            keys = ZOO[arch] if args.model == "all" else [args.model]
            for key in keys:
                if key not in ZOO[arch]:
                    print(f"  [skip] {key} is not in {arch}'s zoo"); continue
                if args.model == "all" and args.tier != "all" and M[key]["tier"] != args.tier:
                    continue
                out = os.path.join(RESULTS, f"{arch}__{ds}__{key}.json")
                if os.path.exists(out) and not args.force:
                    try:
                        prev = json.load(open(out)).get("status")
                    except Exception:
                        prev = None
                    if prev == "done":
                        print(f"  [done] {key}"); continue
                    print(f"  [retry] {key} (previous attempt: {prev})")
                print(f"  [run ] {key} ...", flush=True)
                try:
                    rec = run_one(arch, ds, key, args)
                except Exception as e:  # keep going; record the failure honestly
                    import gc; gc.collect()
                    rec = {"arch": arch, "dataset": ds, "model_key": key, "model_name": M[key]["name"],
                           "tier": M[key]["tier"], "status": "failed", "reason": f"{type(e).__name__}: {e}",
                           "traceback": traceback.format_exc()[-2000:]}
                if rec["status"] in ("done", "failed"):
                    with open(out, "w") as f:
                        json.dump(rec, f, indent=1)
                print(f"         -> {rec['status']}"
                      + (f"  Recall@10 = {rec.get('recall@10')}" if rec["status"] == "done" else f"  ({rec.get('reason')})"))


if __name__ == "__main__":
    main()
