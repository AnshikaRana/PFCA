"""Hand-checked unit tests for metrics, fusion, MUVERA FDE and refinement rules.
Run:  python tests/test_core.py   (or pytest)"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from xaudit.metrics import average_precision, mrr_at, ndcg_exp_pct, r_precision, recall_at  # noqa: E402
from xaudit.encoders import MuveraRunner, maxsim, rrf, rsf, topk_dense  # noqa: E402
from xaudit.bm25 import SparseIndex  # noqa: E402
from xaudit.refine import build_pool, pseudo_pages, split_sentences  # noqa: E402


def close(a, b, tol=1e-6):
    assert abs(a - b) < tol, (a, b)


def test_recall():
    rk = ["a", "b", "c", "d"]
    close(recall_at(rk, {"c": 1, "z": 1}, 2), 0.0)
    close(recall_at(rk, {"c": 1, "z": 1}, 3), 0.5)
    # paper-level gold: two pages of the same gold paper count once
    d2p = {"p1": "A", "p2": "A", "p3": "B"}
    close(recall_at(["p1", "p2", "p3"], {"A": 1, "B": 1}, 2, d2p), 0.5)
    close(recall_at(["p1", "p2", "p3"], {"A": 1, "B": 1}, 3, d2p), 1.0)


def test_rank_metrics():
    rk = ["x", "a", "y", "b"]
    gold = {"a": 1, "b": 1}
    close(mrr_at(rk, gold, 10), 0.5)
    close(r_precision(rk, gold), 0.5)
    close(average_precision(rk, gold), (1 / 2 + 2 / 4) / 2)


def test_ndcg_exp():
    grades = {"a": 2.0, "b": 1.0, "c": 0.0, "d": 0.0, "e": 0.0, "f": 0.0, "g": 0.0, "h": 0.0, "i": 0.0, "j": 0.0,
              "k": 0.0, "l": 0.0, "m": 0.0, "n": 0.0, "o": 0.0, "p": 0.0, "q": 0.0, "r": 0.0, "s": 0.0, "t": 0.0}
    # pool 20 -> cutoff ceil(2) = 2
    close(ndcg_exp_pct(["a", "b"] + list("cdefghijklmnopqrst"), grades, 20), 1.0)
    v = ndcg_exp_pct(["b", "a"] + list("cdefghijklmnopqrst"), grades, 20)
    exp = (2 ** 1 / math.log2(2) + 2 ** 2 / math.log2(3)) / (2 ** 2 / math.log2(2) + 2 ** 1 / math.log2(3))
    close(v, exp)


def test_fusion():
    a = [("d1", 10.0), ("d2", 5.0), ("d3", 0.0)]
    b = [("d3", 1.0), ("d2", 0.5)]
    out = dict(rsf([a, b], [0.5, 0.5], 10))
    close(out["d1"], 0.5); close(out["d2"], 0.5 * 0.5 + 0.5 * 0.0); close(out["d3"], 0.5)
    r = dict(rrf([a, b], [1, 1], 10))
    close(r["d3"], 1 / 63 + 1 / 61)


def test_dense_topk_and_l2():
    rng = np.random.default_rng(0)
    D = rng.standard_normal((1000, 8)).astype(np.float32)
    Q = D[[3, 500]] + 1e-3
    res = topk_dense(Q, D, [str(i) for i in range(1000)], 5, sim="l2", chunk=128)
    assert res[0][0][0] == "3" and res[1][0][0] == "500"


def test_muvera_fde_approximates_maxsim():
    rng = np.random.default_rng(1)
    docs = [rng.standard_normal((30, 16)) for _ in range(40)]
    docs = [d / np.linalg.norm(d, axis=1, keepdims=True) for d in docs]
    q = docs[7][:5] + 0.05 * rng.standard_normal((5, 16))
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    F = np.stack([MuveraRunner.fde(d, False) for d in docs])
    qf = MuveraRunner.fde(q, True)
    top = np.argsort(-(F @ qf))[:10]
    assert 7 in top, top
    assert max(range(40), key=lambda i: maxsim(q, docs[i])) == 7


def test_bm25_and_pool():
    ids = [f"d{i}" for i in range(50)]
    texts = [f"topic{i} common words here" for i in range(50)]
    idx = SparseIndex(ids, texts)
    assert idx.search("topic7", k=1)[0][0] == "d7"
    pool = build_pool("q1", "topic7 words. topic9 here.", ["d42"], idx, None, size=20)
    assert len(pool) == 20 and "d42" in pool and "d7" in pool and len(set(pool)) == 20


def test_text_utils():
    assert split_sentences("A first one. Then a second! Third?") == ["A first one.", "Then a second!", "Third?"]
    assert len(pseudo_pages(" ".join(["w"] * 1700), 800)) == 3


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
