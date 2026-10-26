# PFCA — Protocol-Faithful Cross-Audit

A framework for evaluating **four retrieval architectures** (the evaluation pipelines of
LitSearch, DORIS-MAE, IRPAPERS and MAPLE) on **all four datasets**: 4 × 4 = **16 evaluations**.

The previous plan put every dataset into one shared format. PFCA works the other way round:
**each dataset is refined into the exact format that each paper's architecture expects**. The
architecture stays as the paper defines it, and the data is adapted to it.

---

## 1. Core idea

An **architecture card** describes each paper:

```
A = ⟨ Query format, Document format, Answer format, Search space, Model zoo, Native metrics ⟩
```

A **refinement operator** `R_A` turns any dataset `D` into an instance of that card:

```
cell(A, D) = Evaluate_A( Models_A , R_A(D) )
```

- **Diagonal cells** (A's own dataset): `R_A` is the identity. They must reproduce the paper's
  reported numbers, and this is how the setup is validated.
- **Off-diagonal cells**: `R_A` uses the fixed refinement rules in §3. The rules are the same for
  every dataset, so all off-diagonal cells are built the same way.

---

## 2. The four architecture cards (taken from the papers)

### A1 — LitSearch (Ajith et al., EMNLP 2024)

| Factor | Specification |
|---|---|
| **Query format** | One natural-language literature-search question. Tags: `query_set` (inline-citation / author-written) and `specificity` (0 = broad, at most 20 papers fit; 1 = specific, at most 5 papers fit). |
| **Document format** | One **paper**, represented as **title + abstract** (the default setting). |
| **Answer format** | A **set of gold paper IDs** (usually 1), with **binary** relevance. |
| **Search space** | The **full corpus** (64,183 papers). |
| **Model zoo** | BM25 · GTR-T5-large · Instructor-XL · E5-large-v2 · GritLM-7B · *(API: GPT-4o reranking, GPT-4o one-hop reranking)* |
| **Encoding** | Instructor query: "Represent the research question for retrieving relevant research paper abstracts:"; doc: "Represent the title and abstract of the research paper for retrieval:". GritLM query instruction: "Given a research query, retrieve the title and abstract of the relevant research paper". Max length 512 tokens (GritLM 2048). |
| **Native metrics** | **R@20 on broad** questions; **R@5 and R@20 on specific** questions; reported per subset (inline / author). |

### A2 — DORIS-MAE (Wang et al., NeurIPS 2023 D&B)

| Factor | Specification |
|---|---|
| **Query format** | One **complex, multi-aspect paragraph** (95–226 words), decomposed into aspects and sub-aspects. Models encode it either as **one string** or **sentence by sentence** (mean–max aggregation), and the better of the two is reported. |
| **Document format** | One **abstract** (one vector, or one vector per sentence). |
| **Answer format** | **Graded relevance**: `S(p|Q) = Σ_{a∈Q} S(a,p)` with `S(a,p) ∈ {0,1,2}`. A paper is **relevant iff `S(p|Q)/|Q| ≥ 1`**. Graded scores feed nDCG. |
| **Search space** | **Re-ranking a per-query candidate pool** of about 100 abstracts (built with TF-IDF/BM25 at sentence and paragraph level for about 80, a dense model for about 20, plus citation neighbours). |
| **Model zoo** | E5-L-v2 · RocketQA-v2 · SimCSE · SPLADE-v2 · SPECTER-v2 · TSAspire · SentBERT · OTAspire · ANCE-FirstP · LLAMA · SimLM · BM25 · ColBERT-v2 · TF-IDF · ERNIE · SciBERT · *(API: ada-002)* |
| **Native metrics** | **R@5, R@20, R-Precision, NDCG<sub>exp</sub>@10% (gain = 2^rel / log2(i+2), cutoff = 10% of pool), MRR@10, MAP**, with bootstrap standard errors. |

### A3 — IRPAPERS (Weaviate, 2026)

| Factor | Specification |
|---|---|
| **Query format** | One **page-grounded question** (one question per non-reference page, "needle in the haystack"). |
| **Document format** | One **page**, as an **OCR text transcription** (text track) or a **page image** (image track). |
| **Answer format** | **Exactly one gold page**, so Recall@K = Success@K. |
| **Search space** | The **full page collection** (3,230 pages from 166 papers). |
| **Model zoo** | BM25 · Arctic-embed 2.0 · **Hybrid text** (BM25 + Arctic, relative score fusion, equal weights) · ColModernVBERT · ColPali · ColQwen2 · **Multimodal hybrid** (hybrid text + ColModernVBERT, RSF/RRF, α sweep) · MUVERA (ef sweep) · *(API: Voyage 3 Large, Cohere Embed v4)* |
| **Native metrics** | **Recall@1, Recall@5, Recall@20**. |

### A4 — MAPLE (2026)

| Factor | Specification |
|---|---|
| **Query format** | **Aspect queries**: several queries per target paper, each about one aspect, labelled `motivation`, `method` or `experiment/result`. Text-referenced and multimodal-referenced queries are kept separate. |
| **Document format** | One **whole paper** as a **single holistic representation**: extracted **full text truncated to each model's maximum input** (text models) or page screenshots (multimodal models). Variants: abstract, and 512-token chunks with max or top-3-average pooling. |
| **Answer format** | **One target paper per query**, with queries **grouped by paper**. |
| **Search space** | The **full corpus** (73,973 papers, including about 24K hard negatives). |
| **Model zoo** | BM25 · SPECTER2 · SciNCL · BGE-M3 · Instructor-XL · Qwen3-Embed-8B · GritLM-7B · Qwen3-VL-Embed-8B · Ops-MM-Embed-7B |
| **Encoding** | Query instruction for Qwen3, Instructor, GritLM, Qwen3-VL and Ops-MM: "Represent this paper search query for retrieval of relevant scientific papers". |
| **Native metrics** | **Paper level**: AllAspect@k, AnyAspect@k, AspectCoverage@k. **Query level**: Recall@k per aspect group (k = 5, 20). |

---

## 3. Refinement rules (how each dataset is rebuilt for each architecture)

The same rules apply in every cell.

**R1 — Query-format rule.** The query text is passed through unchanged. If the target format
needs structure the dataset lacks, the structure is **derived by a fixed rule** and marked
`derived`. It is never generated by an LLM.
- LitSearch specificity: specific if |gold| ≤ 5, otherwise broad (the paper's own definition).
- DORIS sentence-by-sentence mode: sentence split.
- MAPLE aspect group: DORIS aspects become aspect queries; IRPAPERS gets a page-position heuristic (marked `heuristic`); otherwise `unlabeled`.

**R2 — Document-unit rule.** The dataset is mapped to the architecture's unit.
- **Lift** to a coarser unit when needed: page → paper (IRPAPERS under A1, A2, A4). A paper is
  built from its pages, using the title + page 1 as the "abstract", or all pages as the
  "full text".
- **Split** to a finer unit when needed: paper → pseudo-pages of about 800 words (LitSearch and
  MAPLE full text under A3).
- If the dataset has no data for the requested unit, the **best available text** is used and the
  cell is marked: DORIS has abstracts only, so its "full text" (A4) and "page" (A3) are the
  abstract itself.
- Image-only models run only where page images exist (IRPAPERS). Elsewhere they are marked
  **N/A (no page images)**, not run on a substitute.

**R3 — Answer-format rule.** Gold labels are translated, never invented.
- Binary → graded (A2): gold = 2, non-gold = 0, so the "relevant iff ≥ 1" rule still holds.
- Graded → binary (A1, A3, A4): DORIS papers with `S(p|Q)/|Q| ≥ 1` count as gold.
- Many-gold → one-target (A4): each (query, gold paper) pair becomes one aspect query for that paper.
- Page gold under a paper unit: the gold paper is the paper that contains the gold page.
- Paper gold under a page unit (A3 on LitSearch, DORIS-MAE, MAPLE): a query **succeeds if any
  page of a gold paper** is in the top K. This is marked `paper-level gold`.

**R4 — Search-space rule.**
- A1, A3, A4 search the **whole** refined corpus.
- A2 needs **candidate pools**. Datasets without pools get a DORIS-style pool: gold papers plus
  BM25 at paragraph and sentence level up to 80, plus 20 from a dense model that is **not in any
  zoo** (`BAAI/bge-small-en-v1.5`), filled to 100 with further BM25 results. The pool size is
  capped at the corpus size.

**R5 — Fidelity flags.** Each cell records one of:
- `native`: diagonal, identical to the paper.
- `faithful`: only R1–R4 transformations, and no information lost.
- `approximate`: something was substituted (an abstract standing in for full text, heuristic
  labels, or paper-level gold). The exact deviations are listed on the dashboard.

---

## 4. The unified headline metric: Recall@10

Recall@10 is computed for **every model in every cell**, using that architecture's answer format:

```
Recall@10 = mean over evaluation queries q of  |Gold_q ∩ Top10_q| / |Gold_q|
```

| Architecture | Evaluation query | Gold_q | Top-10 drawn from |
|---|---|---|---|
| A1 LitSearch | question | gold papers | full corpus |
| A2 DORIS-MAE | complex query | papers with `S/|Q| ≥ 1` | candidate pool |
| A3 IRPAPERS | question | the gold page (or gold paper's pages, counted once) | page collection |
| A4 MAPLE | (aspect query, target) pair | target paper | full corpus |

With a single gold item, Recall@10 equals Success@10, which matches IRPAPERS' and MAPLE's own
definitions. Each cell also reports the **paper's native metrics** (§2), so a diagonal cell can be
checked against the published table.

---

## 5. The 16-cell matrix

| Architecture ↓ / Dataset → | LitSearch | DORIS-MAE | IRPAPERS | MAPLE |
|---|---|---|---|---|
| **A1 LitSearch** | **native** | faithful (363K abstracts; graded → binary) | faithful (pages → papers, 166 papers) | faithful (title + abstract) |
| **A2 DORIS-MAE** | faithful (built pools, binary → graded) | **native** | faithful (pages → papers, pool capped at 166) | faithful (built pools) |
| **A3 IRPAPERS** | approximate (full text → pseudo-pages, paper-level gold) | approximate (abstract = page) | **native** | approximate (full text → pseudo-pages, paper-level gold) |
| **A4 MAPLE** | faithful (full text, pair expansion) | approximate (abstract as full text; aspects → aspect queries) | approximate (pages → whole paper; aspect groups from page position, a heuristic) | **native** |

---

## 6. What counts as a valid result

1. **Validation gate**: a diagonal cell must be within ±3 points of the paper's reported number
   for that model. If it isn't, the architecture's off-diagonal cells are flagged until it is fixed.
2. Models only ever see the query and document text in the architecture's format. Gold labels are
   used only for scoring, and for pool membership in A2 (as in DORIS-MAE, where only pooled papers
   are labelled).
3. Every result file records: model checkpoint, instructions, maximum length, pool or corpus size,
   fidelity flag, deviations, and per-query scores.
4. API models are **listed but not run** (zero-cost configuration).
5. **No synthetic numbers**: the dashboard only displays result files produced by real runs.
