"""Self-contained Lucene-style BM25 and TF-IDF (no Java needed).

BM25 follows Lucene's formula (as used by Pyserini / Weaviate):
  idf = ln(1 + (N - df + 0.5) / (df + 0.5))
  w   = idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))
Tokenisation: lowercase alphanumeric tokens, Lucene English stop-words, Porter stemming when
PyStemmer (or NLTK) is installed. The stemmer actually used is recorded in each result file.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy import sparse

LUCENE_STOP = set("a an and are as at be but by for if in into is it no not of on or such that the their "
                  "then there these they this to was will with".split())
_TOK = re.compile(r"[a-z0-9]+")


def _stemmer():
    try:
        import Stemmer  # PyStemmer
        s = Stemmer.Stemmer("porter")
        return s.stemWord, "porter(PyStemmer)"
    except Exception:
        pass
    try:
        from nltk.stem import PorterStemmer
        ps = PorterStemmer()
        return ps.stem, "porter(nltk)"
    except Exception:
        return (lambda w: w), "none"


_STEM, STEMMER_NAME = _stemmer()
_cache: Dict[str, str] = {}


def tokenize(text: str) -> List[str]:
    out = []
    for t in _TOK.findall((text or "").lower()):
        if t in LUCENE_STOP:
            continue
        s = _cache.get(t)
        if s is None:
            s = _STEM(t)
            if len(_cache) < 2_000_000:
                _cache[t] = s
        out.append(s)
    return out


class SparseIndex:
    """BM25 (default) or TF-IDF index over a list of documents."""

    def __init__(self, doc_ids: Sequence[str], texts: Sequence[str], method: str = "bm25",
                 k1: float = 0.9, b: float = 0.4):
        self.doc_ids = list(doc_ids)
        self.pos = {d: i for i, d in enumerate(self.doc_ids)}
        self.method, self.k1, self.b = method, k1, b
        vocab: Dict[str, int] = {}
        rows, cols, vals, dls = [], [], [], []
        for i, t in enumerate(texts):
            c = Counter(tokenize(t))
            dls.append(sum(c.values()))
            for term, tf in c.items():
                j = vocab.setdefault(term, len(vocab))
                rows.append(i); cols.append(j); vals.append(tf)
        self.vocab = vocab
        n, v = len(self.doc_ids), max(1, len(vocab))
        tf = sparse.csr_matrix((np.array(vals, dtype=np.float32), (rows, cols)), shape=(n, v))
        df = np.bincount(np.array(cols, dtype=np.int64), minlength=v).astype(np.float32)
        dl = np.array(dls, dtype=np.float32)
        if method == "bm25":
            idf = np.log1p((n - df + 0.5) / (df + 0.5))
            avgdl = max(1e-9, dl.mean()) if n else 1.0
            tf = tf.tocoo()
            denom = tf.data + k1 * (1 - b + b * dl[tf.row] / avgdl)
            w = idf[tf.col] * tf.data * (k1 + 1) / denom
            self.W = sparse.csc_matrix((w.astype(np.float32), (tf.row, tf.col)), shape=(n, v))
        else:  # tf-idf (sklearn-style smooth idf, l2-normalised docs, cosine)
            idf = np.log((1 + n) / (1 + df)) + 1
            W = tf.multiply(idf[None, :]).tocsr()
            norms = np.sqrt(np.asarray(W.multiply(W).sum(axis=1))).ravel() + 1e-12
            self.W = sparse.diags(1 / norms) @ W
            self.W = self.W.tocsc()
            self.idf = idf

    def _qvec(self, query: str) -> Optional[sparse.csr_matrix]:
        c = Counter(t for t in tokenize(query) if t in self.vocab)
        if not c:
            return None
        cols = [self.vocab[t] for t in c]
        vals = np.array(list(c.values()), dtype=np.float32)
        if self.method != "bm25":
            vals = vals * self.idf[cols]
            vals = vals / (np.linalg.norm(vals) + 1e-12)
        return sparse.csr_matrix((vals, ([0] * len(cols), cols)), shape=(1, len(self.vocab)))

    def score_all(self, query: str) -> np.ndarray:
        q = self._qvec(query)
        if q is None:
            return np.zeros(len(self.doc_ids), dtype=np.float32)
        return np.asarray((self.W @ q.T).todense()).ravel()

    def search(self, query: str, k: int = 100, restrict: Optional[Sequence[str]] = None):
        s = self.score_all(query)
        if restrict is not None:
            idx = np.array([self.pos[d] for d in restrict if d in self.pos], dtype=np.int64)
            sub = s[idx]
            order = np.argsort(-sub, kind="stable")[:k]
            return [(self.doc_ids[idx[i]], float(sub[i])) for i in order]
        k = min(k, len(s))
        top = np.argpartition(-s, k - 1)[:k] if k < len(s) else np.arange(len(s))
        top = top[np.argsort(-s[top], kind="stable")]
        return [(self.doc_ids[i], float(s[i])) for i in top]
