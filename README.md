# PFCA: Protocol-Faithful Cross-Audit of scientific literature retrieval

Code, results and dashboard for the paper *A Protocol-Faithful Cross-Audit of Scientific Literature Search* (Anshika Rana and Rishabh Kaushal, Indira Gandhi Delhi Technical University for Women).

**Live results dashboard: https://anshikarana.github.io/PFCA/** (every cell, every model, the validation checks and the analysis; the same page as the local `serve.py` website).

![PFCA architecture](docs/pfca_architecture.png)

The audit takes the evaluation protocols (architectures) of **LitSearch, DORIS-MAE, IRPAPERS and MAPLE** and runs each one on **all four datasets**, giving **16 evaluation cells**. In each cell the dataset is rebuilt in the exact query, document and answer format that the paper's architecture expects, every open model listed in that paper is run, and **Recall@10** is reported for all of them together with the paper's own native metrics.

## Results (114 open-model runs, all complete)

Best Recall@10 (%) per cell, with the best model in brackets. Bold cells are each paper's own dataset (native).

| Architecture ↓ / Dataset → | LitSearch | DORIS-MAE | IRPAPERS | MAPLE |
|---|---|---|---|---|
| A1 LitSearch | **75.2 (GritLM-7B)** | 14.3 (Instructor-XL) | 97.2 (GritLM-7B) | 61.0 (GritLM-7B) |
| A2 DORIS-MAE | 61.9 (E5-L-v2) | **26.9 (E5-L-v2)** | 97.2 (TF-IDF) | 39.4 (SPECTER-v2) |
| A3 IRPAPERS | 68.0 (Hybrid text) | 19.5 (Hybrid text) | **91.6 (MM hybrid)** | 40.5 (Hybrid text) |
| A4 MAPLE | 73.1 (Qwen3-8B) | 5.1 (BM25) | 98.3 (BM25) | **56.5 (GritLM-7B)** |

Main findings:

- **Native advantage holds only for LitSearch and DORIS-MAE.** Their best Recall@10 falls by 2.1–13.3 and 7.5–21.8 points under the other protocols. It holds only partly for MAPLE (its own protocol, 56.5, trails LitSearch's title-and-abstract protocol, 61.0) and not for IRPAPERS (91.6 natively, 97.2–98.3 elsewhere).
- **The protocol, not the query, drives recall.** Recall falls as more documents are ranked per query (Spearman ρ = −0.57 over the 16 cells): DORIS-MAE drops when its 100-abstract pools become a 363K-abstract corpus, and IRPAPERS rises when its 3,230 pages collapse into 166 papers. MAPLE papers are found more easily from title and abstract than from truncated full text.
- **Query length explains nothing.** DORIS-MAE's queries are about seven times longer than LitSearch's but score lowest under every protocol, and the same BM25 scorer moves by 10–19 points on identical queries when only the protocol changes.
- **Rankings transfer poorly.** Mean Kendall τ = 0.41 across datasets; the native winner keeps first place in only 3 of 9 comparisons, and only 5 of 16 cells have a statistically clear winner.
- **Reproduction:** 16 of 31 diagonal model checks fall within ±3 points of the published numbers; most deviations lie above the papers' values.

Three runs (7–8B models over the 363K DORIS-MAE abstracts) were not run because they exceed one free 12-hour GPU session. Paid API models are listed but not run.

## Repository layout

| Stage | Files |
|---|---|
| Downloading | `download_data.sh`, `download_data.py` |
| Refinement (R_A) | `xaudit/sources.py`, `xaudit/refine.py`, `xaudit/schema.py` |
| Evaluation | `xaudit/zoo.py`, `xaudit/encoders.py`, `xaudit/bm25.py`, `xaudit/evaluate.py`, `xaudit/metrics.py`, `xaudit/run.py` |
| Analysis | `xaudit/aggregate.py`, `xaudit/analysis.py` |
| Dashboard | `site/` + `serve.py` (local, live) and `docs/` + `build_pages.py` (GitHub Pages, static) |
| Large models on Kaggle | `kaggle/` |
| Results of the 114 runs | `results/*.json` (per-query scores included where available) |

- `FRAMEWORK.md`: the auditing framework (architecture cards, refinement rules, the 16-cell matrix).
- `results/` contains the result files reported in the paper. Delete or move them to start from scratch; the dashboard only shows result files that exist.

## Dashboard

- **Online:** https://anshikarana.github.io/PFCA/ is a static copy of the dashboard built from `results/`.
- **Locally (live):** `python serve.py`, then open http://localhost:8000. It re-reads `results/` on every load.
- **Updating the online copy** after new results:
  ```bash
  python xaudit/analysis.py     # recompute validation, rankings, format effect, significance
  python build_pages.py         # rebuild docs/index.html, docs/data.json, docs/analysis.json
  git add docs results site && git commit -m "Update results" && git push
  ```
- **Turning GitHub Pages on (once):** repository *Settings → Pages → Build and deployment → Deploy from a branch*, branch `main`, folder `/docs`.

## 1. Setup (WSL2 Ubuntu, once)

```bash
cd PFCA                                 # this repository (inside the Linux filesystem)
conda create -n xaudit python=3.11 -y && conda activate xaudit
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
bash download_data.sh                   # ~12 GB; resumes if interrupted
python tests/test_core.py               # quick self-check (no downloads)
```

## 2. Run

```bash
python -m xaudit.run --prepare                                # refine all 16 cells (no GPU needed)
python serve.py                                               # open http://localhost:8000 in a 2nd terminal
python -m xaudit.run --arch all --dataset all --model bm25    # fastest first results, all 16 cells
python -m xaudit.run --arch all --dataset all --tier local    # every model that fits a 6 GB GPU
```

Then run the 7–8B models on Kaggle (`kaggle/kaggle_run.md`) and copy their JSON files into `results/`.

Useful flags: `--arch A1|A2|A3|A4`, `--dataset litsearch|doris_mae|irpapers|maple`,
`--model <key>` (keys are in `xaudit/zoo.py`), `--batch-size 8` (if you run out of GPU memory),
`--max-len 2048` (e.g. BGE-M3 on 6 GB), `--max-pages-per-paper 30` (to shrink the A3 pseudo-page
cells), `--force` (re-run a model).

### Suggested order (most results per hour)
1. BM25 everywhere: minutes per cell, and it gives a first full 4×4 heatmap.
2. Diagonal cells with local models (`--dataset` set to each architecture's own dataset). These
   are the **validation gate**: they should land within ±3 points of the paper's numbers, which
   the website shows in grey next to yours.
3. The off-diagonal cells, smallest first: IRPAPERS, then LitSearch and MAPLE, then the 363K-abstract
   DORIS-MAE corpus under A1 and A4.

### Rough local cost on an RTX 3050 6 GB (estimates; your timings will vary)
| Cell type | Encoded text | Base-size model | Large model |
|---|---|---|---|
| A2 (pools) | ~10K abstracts | minutes | ~10–20 min |
| A1/A4 on LitSearch or MAPLE | 64–74K papers | ~20–40 min | 1–3 h |
| A1/A4 on DORIS-MAE | 363K abstracts | ~1–2 h | 4–8 h (overnight) |
| A3 on LitSearch/MAPLE pseudo-pages | ~0.5–1M pages | hours | use `--max-pages-per-paper` or Kaggle |

## 3. What runs where

| Status on the website | Meaning |
|---|---|
| **Done** | Real run, numbers from a local GPU or Kaggle. |
| **Pending** | Runnable; not run yet. |
| **Kaggle GPU** | 7–8B model: run on Kaggle (`kaggle/kaggle_run.md`). |
| **API · not run** | Paid API (GPT-4o rerankers, ada-002, Voyage, Cohere). Listed but not run (zero-cost configuration). |
| **External code** | Needs the authors' own code: RocketQA-v2 (PaddlePaddle), TSAspire / OTAspire (allenai/aspire), Qwen3-VL-Embed-8B / Ops-MM-Embed-7B (MAPLE screenshots). Run them with the paper's repo and add a result file (below). |
| **N/A** | Image model on a dataset with no page images. |

### Adding a result from external code
Save `results/<ARCH>__<dataset>__<model_key>.json`:
```json
{"arch": "A2", "dataset": "doris_mae", "model_key": "rocketqa-v2", "model_name": "RocketQA-v2",
 "status": "done", "recall@10": 31.2, "native": {"R@5": 15.6, "R@20": 45.4},
 "config": {"source": "DORIS-MAE repo, evaluation.py -o query"}}
```

## 4. Fidelity notes
- Diagonal cells use each paper's own data, format and metric definitions.
- Off-diagonal cells are labelled **faithful** or **approximate**. Every transformation is listed
  in the cell's detail view and in `results/tasks/*.json`.
- Checkpoints marked `*` on the website were not named in the paper, so the canonical public
  checkpoint was used: SimCSE, SPLADE-v2, SentBERT, LLAMA, ColPali, ColQwen2.
- Excluded: DORIS-MAE's in-domain fine-tuned variants (SPECTER-ID, SciBERT-ID). They need
  training, not evaluation.
- BM25 is a built-in Lucene-formula implementation (no Java). The tokenizer and stemmer used are
  recorded in each result.
- IRPAPERS used Weaviate (HNSW, ef=160). Here, search is exact, which can only match or beat HNSW.
- Seven Kaggle runs have incomplete per-query tables (duplicate query ids in the Kaggle output); their
  aggregate scores are valid, but they are excluded from significance tests (shown as "–" on the dashboard).

## 5. Files
```
FRAMEWORK.md          auditing framework
xaudit/sources.py     load the 4 datasets in their native richness
xaudit/refine.py      R_A: rebuild a dataset in each architecture's query/document/answer format
xaudit/zoo.py         model zoos per paper, checkpoints, paper-reported numbers
xaudit/encoders.py    all model runners (BM25/TF-IDF, dense, SPLADE, ColBERT, ColPali family, MUVERA, fusion)
xaudit/evaluate.py    native metrics per architecture + unified Recall@10
xaudit/run.py         CLI
xaudit/aggregate.py   builds the dashboard data from results/
xaudit/analysis.py    validation gate, Kendall tau, format effect, significance -> site/analysis.json
serve.py, site/       local website (live)
build_pages.py, docs/ GitHub Pages copy of the website (static)
tests/                metric/fusion unit tests + synthetic end-to-end test (never shown as results)
kaggle/               how to run the 7-8B models
```

## Citation

```bibtex
@misc{rana2026pfca,
  title  = {A Protocol-Faithful Cross-Audit of Scientific Literature Search},
  author = {Rana, Anshika and Kaushal, Rishabh},
  year   = {2026},
  note   = {Indira Gandhi Delhi Technical University for Women. Code: https://github.com/AnshikaRana/PFCA}
}
```
