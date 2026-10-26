"""Exercise every runner code path (dense, pool re-rank, sentence mode, hybrid, multimodal hybrid,
MUVERA, mode selection) with a fake bag-of-words encoder, so no model download is needed.
Usage: python tests/test_runners.py <synthetic data dir> <results dir>"""
import os
import sys
import zlib

import numpy as np

DATA, RES = sys.argv[1], sys.argv[2]
os.environ["XAUDIT_DATA"], os.environ["XAUDIT_RESULTS"] = DATA, RES
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from xaudit import encoders, run  # noqa: E402
from xaudit.bm25 import tokenize  # noqa: E402


def bow(texts, dim=256):
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        for w in tokenize(t):
            out[i, zlib.crc32(w.encode()) % dim] += 1
    out /= np.linalg.norm(out, axis=1, keepdims=True) + 1e-9
    return out.astype(np.float16)


class FakeDense(encoders.Runner):
    def encode(self, texts, is_query, doc_ids=None, task=None):
        return bow(texts)


class FakeImage(encoders.ColPaliRunner):
    def doc_vectors(self, task):
        return {d: bow(task.docs[d].split(". ")[:6]).astype(np.float32) for d in task.docs}

    def query_vectors(self, task):
        return [bow([q.text]).astype(np.float32) for q in task.queries]


class FakeMuvera(encoders.MuveraRunner, FakeImage):
    pass


for k in ["st", "instructor", "hf", "specter2", "gritlm", "splade", "colbert"]:
    encoders.KINDS[k] = FakeDense
encoders.KINDS["colpali"] = FakeImage
encoders.KINDS["muvera"] = FakeMuvera
encoders.CACHE_DIR = os.path.join(DATA, "cache", "emb")


class Args:
    batch_size, max_len, k, max_pages_per_paper = 8, None, 100, 0


import json  # noqa: E402

checks = [("A2", "doris_mae", "colbert-v2"), ("A2", "doris_mae", "sentbert"), ("A2", "maple", "e5-large-v2-doris"),
          ("A1", "litsearch", "e5-large-v2"), ("A3", "irpapers", "hybrid-text"), ("A3", "irpapers", "mm-hybrid"),
          ("A3", "irpapers", "muvera"), ("A3", "litsearch", "colqwen2"), ("A4", "irpapers", "scincl"),
          ("A4", "doris_mae", "gritlm-7b-maple"), ("A1", "maple", "gpt4o-rerank-bm25"),
          ("A4", "maple", "qwen3-vl-embed-8b")]
for arch, ds, key in checks:
    rec = run.run_one(arch, ds, key, Args)
    summary = {k: rec.get(k) for k in ("status", "recall@10", "headline_config")}
    if "modes" in rec:
        summary["modes"] = {m: v["recall@10"] for m, v in rec["modes"].items()}
    if "ef_sweep" in rec:
        summary["ef_sweep_R@10"] = {e: v["recall@10"] for e, v in rec["ef_sweep"].items()}
    print(arch, ds, key, json.dumps(summary))
    if rec["status"] == "done":
        assert rec["recall@10"] is not None and 0 <= rec["recall@10"] <= 100
print("all runner paths OK")
