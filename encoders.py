"""Model runners. Every runner returns rankings {qid: [(doc_id, score), ...]} for a refined Task.

Heavy libraries (torch, sentence-transformers, transformers, adapters, gritlm, pylate, colpali-engine)
are imported lazily, so BM25/TF-IDF runs need only numpy + scipy.
"""
from __future__ import annotations

import hashlib
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .bm25 import STEMMER_NAME, SparseIndex
from .refine import split_sentences
from .schema import Task
from .zoo import BM25_PARAMS, M

Rankings = Dict[str, List[Tuple[str, float]]]
CACHE_DIR = os.path.join(os.environ.get("XAUDIT_DATA", os.path.join(os.path.dirname(__file__), "..", "data")),
                         "cache", "emb")


# ============================================================================ helpers
def _device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _hash(*parts) -> str:
    h = hashlib.sha1()
    for p in parts:
        if isinstance(p, (list, tuple)):
            for x in p:
                h.update(str(x).encode("utf-8", "ignore")); h.update(b"\x1f")
        else:
            h.update(str(p).encode("utf-8", "ignore"))
        h.update(b"\x1e")
    return h.hexdigest()[:20]


def _cached(key: str, fn):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, key + ".npy")
    if os.path.exists(path):
        return np.load(path)
    arr = fn()
    np.save(path, arr)
    return arr


def topk_dense(Q: np.ndarray, D: np.ndarray, doc_ids: Sequence[str], k: int, sim: str = "cos",
               chunk: int = 50_000) -> List[List[Tuple[str, float]]]:
    """Exact search (no ANN). sim: cos/dot -> inner product; l2 -> negative squared distance."""
    Q = Q.astype(np.float32)
    k = min(k, len(doc_ids))
    best_s = np.full((len(Q), 0), -np.inf, dtype=np.float32)
    best_i = np.zeros((len(Q), 0), dtype=np.int64)
    for start in range(0, len(D), chunk):
        Dc = D[start:start + chunk].astype(np.float32)
        S = Q @ Dc.T
        if sim == "l2":
            S = 2 * S - (Dc * Dc).sum(1)[None, :]
        kk = min(k, S.shape[1])
        idx = np.argpartition(-S, kk - 1, axis=1)[:, :kk]
        s = np.take_along_axis(S, idx, 1)
        best_s = np.concatenate([best_s, s], 1)
        best_i = np.concatenate([best_i, idx + start], 1)
        if best_s.shape[1] > k:
            keep = np.argpartition(-best_s, k - 1, axis=1)[:, :k]
            best_s = np.take_along_axis(best_s, keep, 1)
            best_i = np.take_along_axis(best_i, keep, 1)
    out = []
    for s, i in zip(best_s, best_i):
        o = np.argsort(-s, kind="stable")
        out.append([(doc_ids[i[j]], float(s[j])) for j in o])
    return out


def _clip(texts, max_tokens: int):
    """Cut each text to ~2 words per allowed token before tokenizing. The model truncates at max_tokens anyway;
    this only avoids tokenizing whole 6,000-word papers (identical model input, much faster)."""
    limit = max(64, 2 * int(max_tokens))
    out = []
    for t in texts:
        w = t.split()
        out.append(t if len(w) <= limit else " ".join(w[:limit]))
    return out


def _split_title(task: Task, did: str, text: str) -> Tuple[str, str]:
    t = task.doc_titles.get(did, "") if task.doc_titles else ""
    if t and text.startswith(t):
        return t, text[len(t):].lstrip(". \n")
    return "", text


# ============================================================================ base class
class Runner:
    def __init__(self, spec: dict, arch: str, batch_size: int = 32, max_len: Optional[int] = None,
                 k: int = 100):
        self.spec, self.arch, self.bs, self.k = spec, arch, batch_size, k
        self.max_len = max_len or spec.get("max_len", 512)
        self.config = {"checkpoint": spec.get("ckpt"), "max_len": self.max_len,
                       "assumed_checkpoint": bool(spec.get("assumed"))}
        self.extra: Dict = {}

    def modes(self) -> List[str]:
        return self.spec.get("modes", ["paragraph"]) if self.arch == "A2" else ["paragraph"]

    # subclasses implement encode(texts, is_query) -> np.ndarray
    def encode(self, texts: List[str], is_query: bool, doc_ids: Optional[List[str]] = None,
               task: Optional[Task] = None) -> np.ndarray:
        raise NotImplementedError

    def _doc_texts(self, task: Task, ids: List[str]) -> List[str]:
        return [task.docs[d] for d in ids]

    def rank(self, task: Task, mode: str = "paragraph") -> Rankings:
        sim = self.spec.get("sim", "cos")
        ids = list(task.docs)
        if mode == "sentence":
            return self._rank_sentence(task, ids)
        key = _hash(self.spec["key"], self.max_len, task.arch, task.dataset, "doc", ids[:50], len(ids),
                    sum(len(task.docs[d]) for d in ids))
        D = _cached(key, lambda: self.encode(self._doc_texts(task, ids), False, ids, task))
        Q = self.encode([q.text for q in task.queries], True)
        if task.pools:   # A2: re-rank each pool
            pos = {d: i for i, d in enumerate(ids)}
            out = {}
            for qi, q in enumerate(task.queries):
                pool = task.pools[q.qid]
                idx = np.array([pos[d] for d in pool])
                Dp = D[idx].astype(np.float32)
                s = Dp @ Q[qi].astype(np.float32)
                if sim == "l2":
                    s = 2 * s - (Dp * Dp).sum(1)
                o = np.argsort(-s, kind="stable")
                out[q.qid] = [(pool[j], float(s[j])) for j in o]
            return out
        res = topk_dense(Q, D, ids, self.k, sim)
        return {q.qid: r for q, r in zip(task.queries, res)}

    def _rank_sentence(self, task: Task, ids: List[str]) -> Rankings:
        """DORIS-MAE sentence-by-sentence mode: Rel(p|Q) = mean_i max_j cos(q_i, p_j)."""
        sents, owner = [], []
        for d in ids:
            for s in split_sentences(task.docs[d]):
                sents.append(s); owner.append(d)
        key = _hash(self.spec["key"], self.max_len, task.arch, task.dataset, "sent", len(sents), sents[:50])
        S = _cached(key, lambda: self.encode(sents, False))
        S = S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-9)
        by_doc: Dict[str, List[int]] = {}
        for i, d in enumerate(owner):
            by_doc.setdefault(d, []).append(i)
        out = {}
        for q in task.queries:
            qs = q.sentences or split_sentences(q.text)
            Qv = self.encode(qs, True)
            Qv = Qv / (np.linalg.norm(Qv, axis=1, keepdims=True) + 1e-9)
            pool = task.pools[q.qid] if task.pools else ids
            scores = []
            for d in pool:
                sim = Qv @ S[by_doc[d]].T
                scores.append(float(sim.max(axis=1).mean()))
            o = np.argsort(-np.array(scores), kind="stable")[: (len(pool) if task.pools else self.k)]
            out[q.qid] = [(pool[j], scores[j]) for j in o]
        return out


# ============================================================================ sparse
class SparseRunner(Runner):
    def rank(self, task: Task, mode: str = "paragraph") -> Rankings:
        k1, b = BM25_PARAMS[task.arch]
        method = "bm25" if self.spec["kind"] == "bm25" else "tfidf"
        ids = list(task.docs)
        index = SparseIndex(ids, [task.docs[d] for d in ids], method=method, k1=k1, b=b)
        self.config = {"method": method, "k1": k1 if method == "bm25" else None, "b": b if method == "bm25" else None,
                       "stemmer": STEMMER_NAME, "tokenizer": "lowercase alnum + Lucene stopwords"}
        out = {}
        for q in task.queries:
            if task.pools:
                out[q.qid] = index.search(q.text, k=len(task.pools[q.qid]), restrict=task.pools[q.qid])
            else:
                out[q.qid] = index.search(q.text, k=self.k)
        return out


# ============================================================================ dense text models
class STRunner(Runner):
    """sentence-transformers models (GTR, E5, ANCE, SentBERT, Arctic, BGE-M3, Qwen3-Embedding)."""

    def _model(self):
        if not hasattr(self, "_m"):
            from sentence_transformers import SentenceTransformer
            dev = _device()
            kw = {"model_kwargs": {"torch_dtype": "float16"}} if dev == "cuda" else {}
            self._m = SentenceTransformer(self.spec["ckpt"], device=dev, trust_remote_code=True, **kw)
            self._m.max_seq_length = self.max_len
            self.config.update(device=dev, q_prefix=self.spec.get("q_prefix") or self.spec.get("q_prompt_name"),
                               d_prefix=self.spec.get("d_prefix"))
        return self._m

    def encode(self, texts, is_query, doc_ids=None, task=None):
        m = self._model()
        kw = {}
        if is_query and self.spec.get("q_prompt_name"):
            kw["prompt_name"] = self.spec["q_prompt_name"]
        elif self.spec.get("q_prefix" if is_query else "d_prefix"):
            kw["prompt"] = self.spec["q_prefix" if is_query else "d_prefix"]
        return m.encode(_clip(texts, self.max_len), batch_size=self.bs, convert_to_numpy=True,
                        show_progress_bar=not is_query,
                        normalize_embeddings=self.spec.get("sim") == "cos", **kw).astype(np.float16)


class InstructorRunner(STRunner):
    """INSTRUCTOR via sentence-transformers: instruction is prepended but excluded from pooling."""

    def _model(self):
        m = super()._model()
        if hasattr(m, "set_pooling_include_prompt"):
            m.set_pooling_include_prompt(False)
        self.config.update(q_instr=self.spec.get("q_instr"), d_instr=self.spec.get("d_instr"))
        return m

    def encode(self, texts, is_query, doc_ids=None, task=None):
        m = self._model()
        instr = self.spec.get("q_instr") if is_query else self.spec.get("d_instr")
        kw = {"prompt": instr + " "} if instr else {}
        return m.encode(_clip(texts, self.max_len), batch_size=self.bs, convert_to_numpy=True,
                        show_progress_bar=not is_query, normalize_embeddings=True, **kw).astype(np.float16)


class HFRunner(Runner):
    """Plain transformers encoders with CLS or mean pooling (SimCSE, SPECTER, SimLM, ERNIE, SciBERT,
    SciNCL, LLAMA)."""

    def _load(self):
        if not hasattr(self, "_m"):
            import torch
            from transformers import AutoModel, AutoTokenizer
            dev = _device()
            self._tok = AutoTokenizer.from_pretrained(self.spec["ckpt"])
            if self._tok.pad_token is None:
                self._tok.pad_token = self._tok.eos_token
            dtype = torch.float16 if dev == "cuda" else torch.float32
            self._m = AutoModel.from_pretrained(self.spec["ckpt"], torch_dtype=dtype).to(dev).eval()
            self._dev = dev
            self.config.update(device=dev, pooling=self.spec.get("pool", "cls"))

    def _format(self, texts, is_query, doc_ids, task):
        if is_query or task is None or self.spec["key"] != "scincl":
            return texts
        sep = self._tok.sep_token or " "
        out = []
        for d, t in zip(doc_ids, texts):
            title, rest = _split_title(task, d, t)
            out.append(f"{title}{sep}{rest}" if title else t)
        return out

    def encode(self, texts, is_query, doc_ids=None, task=None):
        import torch
        self._load()
        texts = _clip(self._format(texts, is_query, doc_ids, task), self.max_len)
        outs = []
        rng = range(0, len(texts), self.bs)
        if not is_query and len(texts) > 5 * self.bs:
            try:
                from tqdm import tqdm
                rng = tqdm(rng, desc="Batches", mininterval=10)
            except Exception:
                pass
        for i in rng:
            batch = self._tok(texts[i:i + self.bs], padding=True, truncation=True, max_length=self.max_len,
                              return_tensors="pt").to(self._dev)
            with torch.no_grad():
                h = self._m(**batch).last_hidden_state
            if self.spec.get("pool") == "mean":
                mask = batch["attention_mask"].unsqueeze(-1).to(h.dtype)
                v = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
            else:
                v = h[:, 0]
            v = v.float()
            if self.spec.get("sim") == "cos":
                v = torch.nn.functional.normalize(v, dim=-1)
            outs.append(v.cpu().numpy().astype(np.float16))
        return np.concatenate(outs) if outs else np.zeros((0, 1), np.float16)


class Specter2Runner(HFRunner):
    def _load(self):
        if not hasattr(self, "_m"):
            import torch
            from adapters import AutoAdapterModel
            from transformers import AutoTokenizer
            dev = _device()
            self._tok = AutoTokenizer.from_pretrained(self.spec["ckpt"])
            m = AutoAdapterModel.from_pretrained(self.spec["ckpt"])
            if self.spec.get("adapter"):
                loaded, errs = None, []
                for kw in ({"source": "hf", "load_as": "specter2"}, {"load_as": "specter2"}, {}):
                    try:
                        loaded = m.load_adapter(self.spec["adapter"], set_active=True, **kw)
                        break
                    except Exception as e:  # adapters versions differ in how they resolve Hub adapters
                        errs.append(f"{type(e).__name__}: {e}")
                if loaded is None:  # last resort: download the adapter folder ourselves and load it from disk
                    try:
                        from huggingface_hub import snapshot_download
                        local = snapshot_download(self.spec["adapter"])
                        loaded = m.load_adapter(local, load_as="specter2", set_active=True)
                    except Exception as e:
                        errs.append(f"local: {type(e).__name__}: {e}")
                if loaded is None:
                    raise RuntimeError("could not load SPECTER2 adapter: " + " | ".join(errs))
                self.config["adapter_loaded_as"] = loaded
            self._m = m.to(dev).eval()
            if dev == "cuda":
                self._m = self._m.half()
            self._dev = dev
            self.config.update(device=dev, adapter=self.spec.get("adapter") or "none (specter2_base)")

    def _format(self, texts, is_query, doc_ids, task):
        if is_query or task is None:
            return texts
        out = []
        for d, t in zip(doc_ids, texts):
            title, rest = _split_title(task, d, t)
            out.append(f"{title}{self._tok.sep_token}{rest}" if title else t)
        return out

    def encode(self, texts, is_query, doc_ids=None, task=None):
        self.spec = dict(self.spec, pool="cls")
        return super().encode(texts, is_query, doc_ids, task)


class GritLMRunner(Runner):
    def _model(self):
        if not hasattr(self, "_m"):
            from gritlm import GritLM
            self._m = GritLM(self.spec["ckpt"], torch_dtype="auto", mode="embedding")
            self.config.update(q_instr=self.spec.get("q_instr"))
        return self._m

    def encode(self, texts, is_query, doc_ids=None, task=None):
        instr = self.spec.get("q_instr") if is_query else None
        fmt = f"<|user|>\n{instr}\n<|embed|>\n" if instr else "<|embed|>\n"
        v = self._model().encode(_clip(texts, self.max_len), instruction=fmt, batch_size=self.bs,
                                 max_length=self.max_len)
        v = np.asarray(v, dtype=np.float32)
        return (v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)).astype(np.float16)


class SpladeRunner(Runner):
    """SPLADE-v2: vocab-sized sparse vectors, max(log(1+relu(logits))) pooling, dot product."""

    def _load(self):
        if not hasattr(self, "_m"):
            import torch
            from transformers import AutoModelForMaskedLM, AutoTokenizer
            self._dev = _device()
            self._tok = AutoTokenizer.from_pretrained(self.spec["ckpt"])
            self._m = AutoModelForMaskedLM.from_pretrained(self.spec["ckpt"]).to(self._dev).eval()
            if self._dev == "cuda":
                self._m = self._m.half()

    def encode(self, texts, is_query, doc_ids=None, task=None):
        import torch
        from scipy import sparse
        self._load()
        rows = []
        for i in range(0, len(texts), self.bs):
            b = self._tok(texts[i:i + self.bs], padding=True, truncation=True, max_length=self.max_len,
                          return_tensors="pt").to(self._dev)
            with torch.no_grad():
                logits = self._m(**b).logits.float()
            v = torch.max(torch.log1p(torch.relu(logits)) * b["attention_mask"].unsqueeze(-1), dim=1).values
            rows.append(sparse.csr_matrix(v.cpu().numpy()))
        return sparse.vstack(rows).tocsr()

    def rank(self, task: Task, mode: str = "paragraph") -> Rankings:
        ids = list(task.docs)
        D = self.encode(self._doc_texts(task, ids), False)
        Q = self.encode([q.text for q in task.queries], True)
        pos = {d: i for i, d in enumerate(ids)}
        out = {}
        for qi, q in enumerate(task.queries):
            pool = task.pools[q.qid] if task.pools else ids
            s = np.asarray((D[[pos[d] for d in pool]] @ Q[qi].T).todense()).ravel()
            o = np.argsort(-s, kind="stable")[: (len(pool) if task.pools else self.k)]
            out[q.qid] = [(pool[j], float(s[j])) for j in o]
        return out


def maxsim(q: np.ndarray, d: np.ndarray) -> float:
    return float((q @ d.T).max(axis=1).sum())


class ColBERTRunner(Runner):
    """ColBERT-v2 implemented directly with transformers (no extra package): BERT + 128-d linear head,
    [unused0]/[unused1] query/document markers, queries padded with [MASK] to 32 tokens, docs <= 180 tokens,
    L2-normalised token vectors, MaxSim scoring."""

    def _model(self):
        if not hasattr(self, "_m"):
            import torch
            from huggingface_hub import hf_hub_download
            from transformers import AutoModel, AutoTokenizer
            self._dev = _device()
            self._tok = AutoTokenizer.from_pretrained(self.spec["ckpt"])
            self._bert = AutoModel.from_pretrained(self.spec["ckpt"]).to(self._dev).eval()
            sd = None
            for fname in ("model.safetensors", "pytorch_model.bin"):
                try:
                    path = hf_hub_download(self.spec["ckpt"], fname)
                    if fname.endswith(".safetensors"):
                        from safetensors.torch import load_file
                        sd = load_file(path)
                    else:
                        sd = torch.load(path, map_location="cpu")
                    break
                except Exception:
                    continue
            w = next(v for k, v in sd.items() if k.endswith("linear.weight"))
            self._lin = torch.nn.Linear(w.shape[1], w.shape[0], bias=False)
            self._lin.weight.data = w.float()
            self._lin = self._lin.to(self._dev).eval()
            self._m = True
            self.config.update(device=self._dev, implementation="native transformers ColBERT-v2")
        return self._m

    def _encode_raw(self, texts, is_query):
        import torch
        self._model()
        out = []
        for i in range(0, len(texts), self.bs):
            batch = [("[unused0] " if is_query else "[unused1] ") + t for t in texts[i:i + self.bs]]
            if is_query:
                enc = self._tok(batch, padding="max_length", truncation=True, max_length=32, return_tensors="pt")
                ids = enc["input_ids"]
                ids[ids == self._tok.pad_token_id] = self._tok.mask_token_id
                mask = torch.ones_like(ids)
            else:
                enc = self._tok(batch, padding=True, truncation=True, max_length=180, return_tensors="pt")
                ids, mask = enc["input_ids"], enc["attention_mask"]
            with torch.no_grad():
                h = self._bert(input_ids=ids.to(self._dev), attention_mask=mask.to(self._dev)).last_hidden_state
                v = torch.nn.functional.normalize(self._lin(h.float()), dim=-1).cpu().numpy()
            m = mask.numpy().astype(bool)
            for j in range(len(batch)):
                out.append(v[j][m[j]] if not is_query else v[j])
        return out

    def _enc(self, texts, is_query):
        return [np.asarray(x, dtype=np.float32) for x in self._encode_raw(texts, is_query)]

    def rank(self, task: Task, mode: str = "paragraph") -> Rankings:
        ids = list(task.docs)
        D = dict(zip(ids, self._enc(self._doc_texts(task, ids), False)))
        out = {}
        for q in task.queries:
            pool = task.pools[q.qid] if task.pools else ids
            if mode == "sentence":
                qv = self._enc(q.sentences or split_sentences(q.text), True)
                s = [float(np.mean([maxsim(x, D[d]) for x in qv])) for d in pool]
            else:
                qv = self._enc([q.text], True)[0]
                s = [maxsim(qv, D[d]) for d in pool]
            o = np.argsort(-np.array(s), kind="stable")[: (len(pool) if task.pools else self.k)]
            out[q.qid] = [(pool[j], s[j]) for j in o]
        return out


# ============================================================================ image late-interaction (IRPAPERS)
class ColPaliRunner(Runner):
    def _load(self):
        if hasattr(self, "_m"):
            return
        import torch
        import colpali_engine.models as cm
        fam = self.spec["family"]
        names = {"colpali": ("ColPali", "ColPaliProcessor"), "colqwen2": ("ColQwen2", "ColQwen2Processor"),
                 "colmodernvbert": ("ColModernVBert", "ColModernVBertProcessor")}[fam]
        Model, Proc = getattr(cm, names[0]), getattr(cm, names[1])
        dev = _device()
        self._m = Model.from_pretrained(self.spec["ckpt"], torch_dtype=torch.bfloat16 if dev == "cuda" else
                                        torch.float32, device_map=dev).eval()
        self._p = Proc.from_pretrained(self.spec["ckpt"])
        self._dev = dev
        self.config.update(device=dev, family=fam)

    def doc_vectors(self, task: Task) -> Dict[str, np.ndarray]:
        import torch
        if hasattr(self, "_docv"):
            return self._docv
        key = _hash(self.spec["key"], "pages", len(task.docs))
        path = os.path.join(CACHE_DIR, key + ".npz")
        if os.path.exists(path):
            z = np.load(path)
            self._docv = {k: z[k] for k in z.files}
            return self._docv
        self._load()
        ids = list(task.docs)
        vecs, batch_ids, batch_imgs = {}, [], []

        def flush():
            if not batch_imgs:
                return
            b = self._p.process_images(batch_imgs).to(self._dev)
            with torch.no_grad():
                e = self._m(**b)
            for pid, v, m in zip(batch_ids, e, b.get("attention_mask", [None] * len(batch_ids))):
                v = v.float().cpu().numpy()
                if m is not None:
                    v = v[m.bool().cpu().numpy()]
                vecs[pid] = v.astype(np.float16)
            batch_ids.clear(); batch_imgs.clear()

        for pid, img in task.page_image_loader(ids):
            batch_ids.append(pid); batch_imgs.append(img)
            if len(batch_imgs) >= self.bs:
                flush()
        flush()
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.savez(path, **vecs)
        self._docv = vecs
        return vecs

    def query_vectors(self, task: Task) -> List[np.ndarray]:
        import torch
        self._load()
        out = []
        for i in range(0, len(task.queries), self.bs):
            b = self._p.process_queries([q.text for q in task.queries[i:i + self.bs]]).to(self._dev)
            with torch.no_grad():
                e = self._m(**b)
            for v, m in zip(e, b["attention_mask"]):
                out.append(v.float().cpu().numpy()[m.bool().cpu().numpy()])
        return out

    def rank(self, task: Task, mode: str = "paragraph") -> Rankings:
        D = self.doc_vectors(task)
        ids = list(D)
        out = {}
        for q, qv in zip(task.queries, self.query_vectors(task)):
            s = np.array([maxsim(qv, D[d].astype(np.float32)) for d in ids])
            o = np.argsort(-s, kind="stable")[: self.k]
            out[q.qid] = [(ids[j], float(s[j])) for j in o]
        return out


class MuveraRunner(ColPaliRunner):
    """MUVERA fixed-dimensional encodings (k_sim=4, d_proj=16, reps=10) + exact MaxSim rescoring of the
    top-ef candidates. Candidate generation uses exact FDE inner products (no HNSW approximation)."""

    def __init__(self, spec, arch, **kw):
        super().__init__(dict(M[spec["base"]], key=spec["key"] + "-base"), arch, **kw)
        self.spec_m = spec

    @staticmethod
    def fde(vectors: np.ndarray, is_query: bool, k_sim=4, d_proj=16, reps=10, seed=0) -> np.ndarray:
        dim = vectors.shape[1]
        parts = []
        for r in range(reps):
            rng = np.random.default_rng(seed + r)
            H = rng.standard_normal((dim, k_sim))
            P = rng.choice([-1.0, 1.0], size=(dim, d_proj)) / np.sqrt(d_proj)
            bits = (vectors @ H > 0).astype(np.int64)
            codes = bits @ (1 << np.arange(k_sim))
            nb = 1 << k_sim
            out = np.zeros((nb, dim), dtype=np.float32)
            for c in range(nb):
                sel = vectors[codes == c]
                if len(sel):
                    out[c] = sel.sum(0) if is_query else sel.mean(0)
                elif not is_query:
                    cb = (c >> np.arange(k_sim)) & 1
                    ham = (bits != cb).sum(1)
                    out[c] = vectors[int(np.argmin(ham))]
            parts.append((out @ P).ravel())
        return np.concatenate(parts)

    def rank(self, task: Task, mode: str = "paragraph") -> Rankings:
        D = self.doc_vectors(task)
        ids = list(D)
        F = np.stack([self.fde(D[d].astype(np.float32), False) for d in ids])
        sweep = {}
        qvs = self.query_vectors(task)
        result = None
        for ef in (256, 512, 1024):
            out = {}
            for q, qv in zip(task.queries, qvs):
                qf = self.fde(qv, True)
                cand = np.argsort(-(F @ qf))[: min(ef, len(ids))]
                s = np.array([maxsim(qv, D[ids[j]].astype(np.float32)) for j in cand])
                o = np.argsort(-s, kind="stable")[: self.k]
                out[q.qid] = [(ids[cand[j]], float(s[j])) for j in o]
            sweep[ef] = out
        self.extra["ef_sweep_rankings"] = {str(ef): r for ef, r in sweep.items()}
        self.config.update(k_sim=4, d_proj=16, reps=10, ef_headline=1024, candidate_search="exact FDE (no HNSW)")
        return sweep[1024]


# ============================================================================ fusion (IRPAPERS)
def _minmax(r: List[Tuple[str, float]]) -> Dict[str, float]:
    if not r:
        return {}
    s = np.array([x for _, x in r])
    lo, hi = s.min(), s.max()
    return {d: (float((v - lo) / (hi - lo)) if hi > lo else 1.0) for d, v in r}


def rsf(lists: List[List[Tuple[str, float]]], weights: List[float], k: int) -> List[Tuple[str, float]]:
    tot: Dict[str, float] = {}
    for r, w in zip(lists, weights):
        for d, v in _minmax(r).items():
            tot[d] = tot.get(d, 0.0) + w * v
    return sorted(tot.items(), key=lambda x: -x[1])[:k]


def rrf(lists, weights, k, c=60):
    tot: Dict[str, float] = {}
    for r, w in zip(lists, weights):
        for rank, (d, _) in enumerate(r):
            tot[d] = tot.get(d, 0.0) + w / (c + rank + 1)
    return sorted(tot.items(), key=lambda x: -x[1])[:k]


class HybridRunner(Runner):
    def __init__(self, spec, arch, **kw):
        super().__init__(spec, arch, **kw)
        self.parts = [make_runner(p, arch, **kw) for p in spec["parts"]]

    def rank(self, task: Task, mode: str = "paragraph") -> Rankings:
        part_r = [p.rank(task) for p in self.parts]
        self.config = {"fusion": "relative score fusion (min-max, equal weights)",
                       "parts": {p.spec["key"]: p.config for p in self.parts}}
        return {q.qid: rsf([r[q.qid] for r in part_r], [0.5, 0.5], self.k) for q in task.queries}


class MMHybridRunner(Runner):
    """Multimodal hybrid: (1-α)·hybrid-text + α·image, RSF and RRF, α ∈ {0, 0.1, ..., 1}."""

    def __init__(self, spec, arch, **kw):
        super().__init__(spec, arch, **kw)
        self.text = make_runner(spec["parts"][0], arch, **kw)
        self.img = make_runner(spec["parts"][1], arch, **kw)

    def rank(self, task: Task, mode: str = "paragraph") -> Rankings:
        t, i = self.text.rank(task), self.img.rank(task)
        sweep = {}
        for fusion, fn in (("rsf", rsf), ("rrf", rrf)):
            for a in [round(x * 0.1, 1) for x in range(11)]:
                sweep[f"{fusion}@{a}"] = {q.qid: fn([t[q.qid], i[q.qid]], [1 - a, a], self.k) for q in task.queries}
        self.extra["alpha_sweep_rankings"] = sweep
        self.config = {"fusion": "RSF and RRF(k=60) over alpha in [0,1]; headline = best RSF alpha by Recall@1 "
                                 "(as the paper reports its best configuration)",
                       "text": self.text.config, "image": self.img.config}
        return sweep["rsf@0.5"]  # replaced by best-alpha selection in evaluate()


# ============================================================================ factory
KINDS = {"bm25": SparseRunner, "tfidf": SparseRunner, "st": STRunner, "instructor": InstructorRunner,
         "hf": HFRunner, "specter2": Specter2Runner, "gritlm": GritLMRunner, "splade": SpladeRunner,
         "colbert": ColBERTRunner, "colpali": ColPaliRunner, "muvera": MuveraRunner, "hybrid": HybridRunner,
         "mm_hybrid": MMHybridRunner}


def make_runner(key: str, arch: str, **kw) -> Runner:
    spec = M[key]
    if spec["kind"] not in KINDS:
        raise ValueError(f"{key} ({spec['kind']}) is not runnable inside the kit: {spec.get('note', '')}")
    return KINDS[spec["kind"]](spec, arch, **kw)
