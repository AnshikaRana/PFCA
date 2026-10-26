"""Load the four datasets from data/raw/ into their richest native form (schema.Source).

Expected layout (created by download_data.sh):
  data/raw/litsearch/query.jsonl, corpus_clean.jsonl
  data/raw/doris_mae/DORIS-MAE_dataset_v1.json
  data/raw/irpapers/queries.jsonl, docs_text.jsonl
  data/raw/maple/queries/*.jsonl, corpus/*.jsonl, representations/{abstract,fulltext}/*.jsonl
"""
from __future__ import annotations

import glob
import json
import os
import pickle
import re
from typing import Dict, Iterator, List

from .schema import Aspect, Page, Paper, Source, SourceQuery

DATA_ROOT = os.environ.get("XAUDIT_DATA", os.path.join(os.path.dirname(__file__), "..", "data"))
RAW = lambda *p: os.path.join(DATA_ROOT, "raw", *p)  # noqa: E731
CACHE = os.path.join(DATA_ROOT, "cache")

DATASETS = ["litsearch", "doris_mae", "irpapers", "maple"]


def _jsonl(path: str) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _need(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}. Run download_data.sh first (see README).")


# --------------------------------------------------------------------------- LitSearch
def load_litsearch(need_fulltext: bool = True) -> Source:
    qpath, cpath = RAW("litsearch", "query.jsonl"), RAW("litsearch", "corpus_clean.jsonl")
    _need(qpath); _need(cpath)
    papers: Dict[str, Paper] = {}
    for r in _jsonl(cpath):
        pid = str(r["corpusid"])
        papers[pid] = Paper(pid=pid, title=r.get("title") or "", abstract=r.get("abstract") or "",
                            fulltext=(r.get("full_paper") or "") if need_fulltext else "")
    queries = []
    for i, r in enumerate(_jsonl(qpath)):
        gold = {str(c): 1.0 for c in r["corpusids"] if str(c) in papers}
        queries.append(SourceQuery(
            qid=f"ls{i}", text=r["query"], gold=gold,
            meta={"query_set": r.get("query_set"), "specificity": int(r.get("specificity", 1)),
                  "quality": r.get("quality"),
                  "subset": "inline" if str(r.get("query_set", "")).startswith("inline") else "author"}))
    return Source("litsearch", "paper", papers, queries, has_fulltext=need_fulltext)


# --------------------------------------------------------------------------- DORIS-MAE
def _as_list(x):
    if isinstance(x, dict):
        return [x[k] for k in sorted(x, key=lambda k: int(k) if str(k).isdigit() else str(k))]
    return list(x)


def load_doris_mae(**_) -> Source:
    path = RAW("doris_mae", "DORIS-MAE_dataset_v1.json")
    _need(path)
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    keys = {k.lower().replace(" ", "_"): k for k in d}
    corpus = _as_list(d[keys["corpus"]])
    queries_raw = _as_list(d[keys["query"]])
    ann = _as_list(d[keys["annotation"]])
    id2aspect = {str(k): v for k, v in d[keys["aspect_id2aspect"]].items()}

    papers: Dict[str, Paper] = {}
    for idx, a in enumerate(corpus):
        pid = str(a.get("abstract_id", idx))
        papers[pid] = Paper(pid=pid, title=a.get("title") or "", abstract=a.get("original_abstract") or "")

    scores: Dict[tuple, float] = {}
    for a in ann:
        scores[(str(a["aspect_id"]), str(a["abstract_id"]))] = float(a["score"])

    queries = []
    for qi, q in enumerate(queries_raw):
        asp_tree = q.get("aspects", {})
        aspects, all_ids = [], []
        for aid, subs in asp_tree.items():
            subs = [str(s) for s in (subs or [])]
            aspects.append(Aspect(str(aid), id2aspect.get(str(aid), ""), subs))
            all_ids += [str(aid)] + subs
        all_ids = list(dict.fromkeys(all_ids))
        pool = [str(p) for p in q.get("candidate_pool", [])]
        gold, aspect_grades = {}, {}
        for pid in pool:
            s = sum(scores.get((aid, pid), 0.0) for aid in all_ids)
            gold[pid] = s / max(1, len(all_ids))          # S(p|Q)/|Q| in [0,2]
        for asp in aspects:
            aspect_grades[asp.aspect_id] = {pid: scores.get((asp.aspect_id, pid), 0.0) for pid in pool}
        queries.append(SourceQuery(
            qid=f"dm{qi}", text=q["query_text"], gold=gold, relevant_threshold=1.0, pool=pool,
            aspects=aspects, aspect_grades=aspect_grades,
            meta={"query_type": q.get("query_type"), "n_aspects": len(aspects), "n_aspect_units": len(all_ids)}))
    return Source("doris_mae", "paper", papers, queries, has_fulltext=False,
                  notes=["DORIS-MAE provides abstracts only (no full text, no page images)."])


# --------------------------------------------------------------------------- IRPAPERS
def _irpapers_image_loader():
    """Returns iter_images(page_ids) -> yields (page_id, PIL.Image) one at a time from the page image
    files written by download_data.py (data/raw/irpapers/pages/<page_id>.img)."""
    def iter_images(page_ids: List[str]):
        from PIL import Image
        for pid in page_ids:
            path = RAW("irpapers", "pages", f"{pid}.img")
            if os.path.exists(path):
                with Image.open(path) as im:
                    yield pid, im.convert("RGB")
    return iter_images


def load_irpapers(**_) -> Source:
    qpath, dpath = RAW("irpapers", "queries.jsonl"), RAW("irpapers", "docs_text.jsonl")
    _need(qpath); _need(dpath)
    pages: Dict[str, Page] = {}
    papers: Dict[str, Paper] = {}
    for r in _jsonl(dpath):
        pdf = str(int(r["pdf_id_x"]))
        pg = int(r["page_number"])
        page_id = r.get("dataset_id") or f"{pdf}_{pg}"
        pages[page_id] = Page(page_id, pdf, pg, r.get("transcription") or "")
        papers.setdefault(pdf, Paper(pid=pdf, title=r.get("pdf_title") or ""))
    for p in papers.values():
        p.page_ids = sorted([pg.page_id for pg in pages.values() if pg.pid == p.pid],
                            key=lambda x: pages[x].number)
        first = pages[p.page_ids[0]].text if p.page_ids else ""
        p.abstract = first                       # page 1 (title + abstract area) = paper "abstract"
        p.fulltext = "\n\n".join(pages[x].text for x in p.page_ids)
    queries = []
    for r in _jsonl(qpath):
        gold_page = f"{int(r['pdf_id'])}_{int(r['page_number'])}"
        if gold_page not in pages:
            continue
        n_pages = len(papers[str(int(r["pdf_id"]))].page_ids)
        queries.append(SourceQuery(
            qid=f"ir{r['dataset_id']}", text=r["question"], gold={gold_page: 1.0},
            meta={"answer": r.get("answer"), "page_number": int(r["page_number"]), "n_pages": n_pages,
                  "pdf_id": str(int(r["pdf_id"]))}))
    return Source("irpapers", "page", papers, queries, pages=pages, has_fulltext=True,
                  has_page_images=True, page_image_loader=_irpapers_image_loader())


# --------------------------------------------------------------------------- MAPLE
_SLUG_HASH = re.compile(r"-[0-9a-f]{8,12}$")


def _deslug(s: str) -> str:
    if " " in s or "-" not in s:
        return s
    return _SLUG_HASH.sub("", s).replace("-", " ").strip()


def load_maple(need_fulltext: bool = True) -> Source:
    base = RAW("maple")
    _need(os.path.join(base, "queries", "text_referenced.jsonl"))
    papers: Dict[str, Paper] = {}
    for f in sorted(glob.glob(os.path.join(base, "corpus", "*.jsonl"))):
        for r in _jsonl(f):
            pid = r["paper_id"]
            papers[pid] = Paper(pid=pid, title=_deslug(r.get("title") or pid))
    for f in sorted(glob.glob(os.path.join(base, "representations", "abstract", "*.jsonl"))):
        for r in _jsonl(f):
            if r["paper_id"] in papers:
                papers[r["paper_id"]].abstract = r.get("text") or ""
    has_ft = False
    if need_fulltext:
        ft_files = sorted(glob.glob(os.path.join(base, "representations", "fulltext", "*.jsonl")))
        for f in ft_files:
            for r in _jsonl(f):
                if r["paper_id"] in papers:
                    papers[r["paper_id"]].fulltext = r.get("text") or ""
        has_ft = bool(ft_files)
    queries = []
    for fname, mm in [("text_referenced.jsonl", False), ("multimodal_referenced.jsonl", True)]:
        p = os.path.join(base, "queries", fname)
        if not os.path.exists(p):
            continue
        for r in _jsonl(p):
            if r["target_paper_id"] not in papers:
                continue
            queries.append(SourceQuery(
                qid=f"mp{r['query_id']}", text=r["query_text"], gold={r["target_paper_id"]: 1.0},
                meta={"aspect_group": r.get("aspect_group"), "is_multimodal": bool(r.get("is_multimodal", mm))}))
    notes = [] if has_ft else ["MAPLE full text not downloaded: holistic full-text falls back to abstract."]
    return Source("maple", "paper", papers, queries, has_fulltext=has_ft, notes=notes)


LOADERS = {"litsearch": load_litsearch, "doris_mae": load_doris_mae, "irpapers": load_irpapers, "maple": load_maple}


def load_source(name: str, need_fulltext: bool = True, use_cache: bool = True) -> Source:
    os.makedirs(CACHE, exist_ok=True)
    cp = os.path.join(CACHE, f"source_{name}_{'ft' if need_fulltext else 'noft'}.pkl")
    if use_cache and os.path.exists(cp):
        with open(cp, "rb") as f:
            src = pickle.load(f)
        if name == "irpapers":
            src.page_image_loader = _irpapers_image_loader()
        return src
    src = LOADERS[name](need_fulltext=need_fulltext)
    if use_cache:
        loader, src.page_image_loader = src.page_image_loader, None
        with open(cp, "wb") as f:
            pickle.dump(src, f, protocol=pickle.HIGHEST_PROTOCOL)
        src.page_image_loader = loader
    return src
