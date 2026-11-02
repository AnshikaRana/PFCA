"""Refinement operators R_A: rebuild any Source in the exact Query/Document/Answer format of
architecture A (see FRAMEWORK.md §3, rules R1–R5)."""
from __future__ import annotations

import random
import re
from typing import Callable, Dict, List, Optional

from .bm25 import SparseIndex
from .schema import Source, Task, TaskQuery

ARCH_NAMES = {"A1": "LitSearch", "A2": "DORIS-MAE", "A3": "IRPAPERS", "A4": "MAPLE"}
NATIVE = {"A1": "litsearch", "A2": "doris_mae", "A3": "irpapers", "A4": "maple"}
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"(])")


def split_sentences(text: str) -> List[str]:
    s = [x.strip() for x in _SENT.split(text or "") if x.strip()]
    return s or [text]


def title_abstract(p) -> str:
    ab = p.abstract or " ".join((p.fulltext or "").split()[:300])
    return f"{p.title}. {ab}".strip(". ").strip() if p.title else ab


def pseudo_pages(text: str, words: int = 800) -> List[str]:
    toks = (text or "").split()
    return [" ".join(toks[i:i + words]) for i in range(0, len(toks), words)] or [""]


def _relevant_papers(src: Source, q) -> Dict[str, float]:
    """Gold lifted to paper ids, relevant items only, binary."""
    rel = q.relevant()
    if src.native_unit == "page":
        return {src.pages[g].pid: 1.0 for g in rel}
    return {g: 1.0 for g in rel}


def _fid(src: Source, arch: str, devs: List[str]):
    if src.name == NATIVE[arch]:
        return "native", []
    return ("approximate" if any(d.startswith("[approx]") for d in devs) else "faithful"), devs


# ----------------------------------------------------------------------------- A1 LitSearch
def refine_A1(src: Source, **_) -> Task:
    docs = {pid: title_abstract(p) for pid, p in src.papers.items()}
    queries, qrels, devs, skipped = [], {}, [], 0
    for q in src.queries:
        gold = _relevant_papers(src, q)
        if not gold:
            skipped += 1
            continue
        meta = dict(q.meta)
        if src.name != "litsearch":
            meta["specificity"] = 1 if len(gold) <= 5 else 0
            meta["specificity_source"] = "derived(|gold|<=5 -> specific)"
            meta["subset"] = "foreign"
        queries.append(TaskQuery(q.qid, q.text, meta))
        qrels[q.qid] = gold
    if src.name == "doris_mae":
        devs += ["Graded relevance binarised with DORIS's own rule S(p|Q)/|Q| >= 1.",
                 "Search space = all 363,133 abstracts (LitSearch searches the full corpus), not the 100-paper pools.",
                 "Specificity derived from gold-set size (<=5 specific, else broad)."]
    if src.name == "irpapers":
        devs += ["Pages lifted to papers: document = title + page-1 transcription (abstract area).",
                 "Gold page mapped to its paper; corpus is only 166 papers.",
                 "All questions are single-gold, so all are 'specific'."]
    if src.name == "maple":
        devs += ["Title + abstract representation (MAPLE 'abstract'); titles rebuilt from paper slugs where needed.",
                 "One target paper per query, so all queries are 'specific'."]
    if skipped:
        devs.append(f"{skipped} queries dropped: no relevant document.")
    fid, devs = _fid(src, "A1", devs)
    return Task("A1", src.name, "paper", docs, {k: p.title for k, p in src.papers.items()},
                {k: p.abstract for k, p in src.papers.items()}, queries, qrels, fidelity=fid, deviations=devs,
                format_card={"query": "Natural-language literature-search question (+ broad/specific tag)",
                             "document": "Paper = title + abstract",
                             "answer": "Set of gold papers, binary relevance",
                             "search": f"Full corpus ({len(docs):,} papers)"})


# ----------------------------------------------------------------------------- A2 DORIS-MAE
def build_pool(qid: str, query: str, gold: List[str], bm25: SparseIndex,
               dense_fn: Optional[Callable[[str, int], List[str]]], size: int = 100) -> List[str]:
    """DORIS-MAE pooling: ~80 lexical (paragraph + sentence level) + ~20 dense, gold included."""
    size = min(size, len(bm25.doc_ids))
    lists = [[d for d, _ in bm25.search(query, k=size)]]
    lists += [[d for d, _ in bm25.search(s, k=size)] for s in split_sentences(query)[:12]]
    pool, seen = [], set()
    i = 0
    while len(pool) < min(80, size) and any(i < len(l) for l in lists):
        for l in lists:
            if i < len(l) and l[i] not in seen and len(pool) < min(80, size):
                seen.add(l[i]); pool.append(l[i])
        i += 1
    if dense_fn is not None:
        for d in dense_fn(query, 60):
            if len(pool) >= size:
                break
            if d not in seen:
                seen.add(d); pool.append(d)
    for d in lists[0]:
        if len(pool) >= size:
            break
        if d not in seen:
            seen.add(d); pool.append(d)
    missing = [g for g in gold if g not in seen]
    if missing:
        non_gold = [d for d in pool if d not in set(gold)]
        keep = max(0, size - len(gold))
        pool = [d for d in pool if d in set(gold)] + non_gold[:keep] + missing
    pool = list(dict.fromkeys(pool))
    random.Random(qid).shuffle(pool)          # remove any positional signal
    return pool


def refine_A2(src: Source, dense_pool_fn: Optional[Callable] = None, **_) -> Task:
    abstracts = {pid: (p.abstract or p.title) for pid, p in src.papers.items()}
    queries, qrels, pools, devs = [], {}, {}, []
    native = src.name == "doris_mae"
    if native:
        for q in src.queries:
            pool = list(q.pool)
            random.Random(q.qid).shuffle(pool)
            pools[q.qid] = pool
            qrels[q.qid] = {d: q.gold.get(d, 0.0) for d in pool}
            queries.append(TaskQuery(q.qid, q.text, q.meta, split_sentences(q.text)))
    else:
        bm25 = SparseIndex(list(abstracts), list(abstracts.values()))
        dense = dense_pool_fn(abstracts) if dense_pool_fn else None
        for q in src.queries:
            gold = _relevant_papers(src, q)
            if not gold:
                continue
            pools[q.qid] = build_pool(q.qid, q.text, list(gold), bm25, dense)
            qrels[q.qid] = {d: (2.0 if d in gold else 0.0) for d in pools[q.qid]}
            queries.append(TaskQuery(q.qid, q.text, q.meta, split_sentences(q.text)))
        devs += ["Candidate pools built with the DORIS recipe: BM25 paragraph+sentence (~80) + 20 dense "
                 "(bge-small-en-v1.5, outside every zoo) + gold, shuffled."
                 + ("" if dense else " [dense part skipped: sentence-transformers unavailable]"),
                 "Binary gold mapped to graded scale: gold = 2, other pooled = 0 (relevant iff >= 1).",
                 "Query has no aspect decomposition; sentence-by-sentence mode uses sentence splitting."]
        if src.name == "irpapers":
            devs.append("Pages lifted to papers: abstract = page-1 transcription; gold page -> its paper.")
    union = sorted({d for p in pools.values() for d in p})
    docs = {d: abstracts[d] for d in union}
    fid, devs = _fid(src, "A2", devs)
    return Task("A2", src.name, "abstract", docs, {d: src.papers[d].title for d in union},
                {d: abstracts[d] for d in union}, queries, qrels, pools=pools, relevant_threshold=1.0,
                fidelity=fid, deviations=devs,
                format_card={"query": "Complex multi-aspect paragraph (whole string or sentence-by-sentence)",
                             "document": "Abstract",
                             "answer": "Graded relevance over the candidate pool; relevant iff score >= 1",
                             "search": f"Re-rank ~{round(sum(map(len, pools.values())) / max(1, len(pools)))}-paper pool per query"})


# ----------------------------------------------------------------------------- A3 IRPAPERS
def refine_A3(src: Source, page_words: int = 800, max_pages_per_paper: int = 0, **_) -> Task:
    devs = []
    if src.native_unit == "page":
        docs = {pid: pg.text for pid, pg in src.pages.items()}
        doc2paper = {pid: pg.pid for pid, pg in src.pages.items()}
        qrels = {q.qid: dict(q.relevant()) for q in src.queries}
        queries = [TaskQuery(q.qid, q.text, q.meta) for q in src.queries]
        gold_level = "doc"
    else:
        docs, doc2paper = {}, {}
        for pid, p in src.papers.items():
            body = p.fulltext if p.fulltext else title_abstract(p)
            pages = pseudo_pages(body, page_words)
            if max_pages_per_paper:
                pages = pages[:max_pages_per_paper]
            for i, t in enumerate(pages):
                did = f"{pid}::p{i + 1}"
                docs[did] = (f"{p.title}\n{t}" if i == 0 and p.title and p.fulltext else t)
                doc2paper[did] = pid
        queries, qrels = [], {}
        for q in src.queries:
            gold = _relevant_papers(src, q)
            if gold:
                queries.append(TaskQuery(q.qid, q.text, q.meta))
                qrels[q.qid] = gold
        gold_level = "paper"
        devs.append("[approx] No real pages: full text split into ~%d-word pseudo-pages." % page_words
                    if src.has_fulltext else "[approx] No full text or pages: each abstract is a single 'page'.")
        devs.append("[approx] Gold is paper-level: a query succeeds if any page of a gold paper is retrieved "
                    "(IRPAPERS gold is a single page).")
        devs.append("Image-track models are N/A: this dataset has no page images.")
        if max_pages_per_paper:
            devs.append(f"Pages capped at {max_pages_per_paper} per paper.")
    fid, devs = _fid(src, "A3", devs)
    return Task("A3", src.name, "page", docs, {}, {}, queries, qrels, gold_level=gold_level,
                doc2paper=doc2paper, fidelity=fid, deviations=devs, has_page_images=src.has_page_images,
                page_image_loader=src.page_image_loader,
                format_card={"query": "Page-grounded question",
                             "document": "Page (OCR text; page image for image models)",
                             "answer": "One gold page (Recall@K = Success@K)" if gold_level == "doc"
                             else "Gold paper(s); success if any of its pages is retrieved",
                             "search": f"Full page collection ({len(docs):,} pages)"})


# ----------------------------------------------------------------------------- A4 MAPLE
def _page_position_group(page: int, n: int) -> str:
    r = page / max(1, n)
    return "motivation" if r <= 0.25 else ("method" if r <= 0.6 else "experiment/result")


def refine_A4(src: Source, **_) -> Task:
    devs = []
    docs = {}
    for pid, p in src.papers.items():
        if src.name == "maple":
            docs[pid] = p.fulltext or title_abstract(p)
        elif p.fulltext:
            docs[pid] = f"{p.title}\n\n{p.fulltext}" if p.title else p.fulltext
        else:
            docs[pid] = title_abstract(p)
    queries, qrels, groups = [], {}, {}

    def add(qid, text, target, group, meta):
        queries.append(TaskQuery(qid, text, dict(meta, aspect_group=group)))
        qrels[qid] = {target: 1.0}
        groups[qid] = {"target": target, "aspect_group": group}

    for q in src.queries:
        if src.name == "maple":
            add(q.qid, q.text, next(iter(q.gold)), q.meta.get("aspect_group") or "unlabeled", q.meta)
        elif src.name == "irpapers":
            gold = _relevant_papers(src, q)
            for t in gold:
                add(q.qid, q.text, t, _page_position_group(q.meta["page_number"], q.meta["n_pages"]), q.meta)
        elif src.name == "doris_mae":
            for t in q.relevant():
                for asp in q.aspects:
                    if asp.text and q.aspect_grades.get(asp.aspect_id, {}).get(t, 0) >= 1:
                        add(f"{q.qid}:{asp.aspect_id}#{t}", asp.text, t, "unlabeled", q.meta)
        else:  # litsearch
            gold = _relevant_papers(src, q)
            for t in gold:
                add(f"{q.qid}#{t}" if len(gold) > 1 else q.qid, q.text, t, "unlabeled", q.meta)
    if src.name == "litsearch":
        devs += ["Holistic document = title + full paper text (truncated to each model's max input).",
                 "Multi-gold questions expanded into one (query, target) pair per gold paper.",
                 "No aspect labels: aspect-group breakdown is 'unlabeled'; most papers have one query, "
                 "so AllAspect ~= AnyAspect."]
    if src.name == "irpapers":
        devs += ["Pages lifted to whole papers: full text = all page transcriptions concatenated "
                 "(multimodal models can use the page images).",
                 "[approx] Aspect group is a heuristic from page position (<=25% motivation, <=60% method, "
                 "else experiment/result)."]
    if src.name == "doris_mae":
        devs += ["[approx] No full text: the abstract stands in for the holistic paper.",
                 "Aspect queries = DORIS aspects with score >= 1 for each relevant paper (the paper is their target).",
                 "Search space = all 363,133 abstracts."]
    if src.name == "maple" and not src.has_fulltext:
        devs.append("[approx] Full text not downloaded: abstract used instead.")
    fid, devs = _fid(src, "A4", devs)
    return Task("A4", src.name, "paper", docs, {k: p.title for k, p in src.papers.items()},
                {k: p.abstract for k, p in src.papers.items()}, queries, qrels, groups=groups,
                fidelity=fid, deviations=devs, has_page_images=src.has_page_images,
                page_image_loader=src.page_image_loader,
                format_card={"query": "Aspect query (motivation / method / experiment-result), many per paper",
                             "document": "Whole paper, holistic full text (truncated to model max input)",
                             "answer": "One target paper per query; queries grouped by paper",
                             "search": f"Full corpus ({len(docs):,} papers)"})


REFINERS = {"A1": refine_A1, "A2": refine_A2, "A3": refine_A3, "A4": refine_A4}


def refine(arch: str, src: Source, **kw) -> Task:
    return REFINERS[arch](src, **kw)
