"""Core data structures.

Source  = a dataset loaded in its *richest native form* (papers, pages, graded labels, pools, aspects).
Task    = a Source refined into one architecture's exact Query / Document / Answer format.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional


@dataclass
class Paper:
    pid: str
    title: str = ""
    abstract: str = ""
    fulltext: str = ""                      # "" if the dataset has no full text
    page_ids: List[str] = field(default_factory=list)  # only for page-native datasets (IRPAPERS)


@dataclass
class Page:
    page_id: str
    pid: str
    number: int
    text: str


@dataclass
class Aspect:
    aspect_id: str
    text: str
    sub_aspect_ids: List[str] = field(default_factory=list)


@dataclass
class SourceQuery:
    qid: str
    text: str
    meta: Dict = field(default_factory=dict)
    # Gold labels in the dataset's native unit ("paper" or "page").
    # grade = native graded score (DORIS: S(p|Q)/|Q| in [0,2]; binary datasets: 1.0)
    gold: Dict[str, float] = field(default_factory=dict)
    relevant_threshold: float = 1.0
    pool: Optional[List[str]] = None        # DORIS-MAE native candidate pool (paper ids)
    aspects: List[Aspect] = field(default_factory=list)          # DORIS-MAE only
    aspect_grades: Dict[str, Dict[str, float]] = field(default_factory=dict)  # aspect_id -> {pid: score 0-2}

    def relevant(self) -> Dict[str, float]:
        return {d: g for d, g in self.gold.items() if g >= self.relevant_threshold}


@dataclass
class Source:
    name: str                       # litsearch | doris_mae | irpapers | maple
    native_unit: str                # "paper" | "page"
    papers: Dict[str, Paper]
    queries: List[SourceQuery]
    pages: Dict[str, Page] = field(default_factory=dict)
    has_fulltext: bool = False
    has_page_images: bool = False
    page_image_loader: Optional[Callable[[List[str]], list]] = None   # page_ids -> PIL images
    notes: List[str] = field(default_factory=list)


@dataclass
class TaskQuery:
    qid: str
    text: str
    meta: Dict = field(default_factory=dict)
    sentences: List[str] = field(default_factory=list)


@dataclass
class Task:
    """One cell of the 4x4 matrix: architecture `arch` applied to dataset `dataset`."""
    arch: str
    dataset: str
    unit: str                                  # document unit the models see: paper | page | abstract
    docs: Dict[str, str]                       # doc_id -> text exactly as the architecture formats it
    doc_titles: Dict[str, str]
    doc_abstracts: Dict[str, str]
    queries: List[TaskQuery]
    qrels: Dict[str, Dict[str, float]]         # qid -> {gold_id: grade}
    gold_level: str = "doc"                    # "doc" or "paper" (A3 on paper-gold datasets)
    doc2paper: Dict[str, str] = field(default_factory=dict)
    pools: Optional[Dict[str, List[str]]] = None   # A2 only
    relevant_threshold: float = 1.0
    groups: Dict[str, Dict] = field(default_factory=dict)  # A4: qid -> {"target": pid, "aspect_group": str}
    fidelity: str = "native"                   # native | faithful | approximate
    deviations: List[str] = field(default_factory=list)
    format_card: Dict[str, str] = field(default_factory=dict)
    has_page_images: bool = False
    page_image_loader: Optional[Callable] = None

    def stats(self) -> Dict:
        return {
            "n_docs": len(self.docs),
            "n_queries": len(self.queries),
            "avg_gold_per_query": round(sum(sum(1 for g in v.values() if g >= self.relevant_threshold)
                                            for v in self.qrels.values()) / max(1, len(self.qrels)), 2),
            "avg_pool_size": (round(sum(len(p) for p in self.pools.values()) / max(1, len(self.pools)), 1)
                              if self.pools else None),
        }
