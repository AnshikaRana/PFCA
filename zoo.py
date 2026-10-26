"""Model zoos of the four architectures, exactly as listed in each paper, plus the paper-reported
numbers for the diagonal (validation) cells.

tier:  local    -> fits a 6 GB RTX 3050 (fp16)
       kaggle   -> needs a 16 GB GPU (Kaggle T4 / Colab)
       api      -> paid API, listed but NOT run (zero-cost configuration)
       external -> needs the authors' own code; drop its result JSON into results/ (see README)
assumed: True when the paper did not name the exact checkpoint and we picked the canonical one.
"""

LITSEARCH_Q = "Represent the research question for retrieving relevant research paper abstracts:"
LITSEARCH_D = "Represent the title and abstract of the research paper for retrieval:"
LITSEARCH_GRIT = "Given a research query, retrieve the title and abstract of the relevant research paper"
MAPLE_Q = "Represent this paper search query for retrieval of relevant scientific papers"

M = {
    # ---------------- shared / A1 LitSearch
    "bm25": dict(name="BM25", kind="bm25", tier="local"),
    "tfidf": dict(name="TF-IDF", kind="tfidf", tier="local"),
    "gtr-t5-large": dict(name="GTR-T5-large", kind="st", ckpt="sentence-transformers/gtr-t5-large",
                         max_len=512, sim="cos", tier="local"),
    "instructor-xl-litsearch": dict(name="Instructor-XL", kind="instructor", ckpt="hkunlp/instructor-xl",
                                    q_instr=LITSEARCH_Q, d_instr=LITSEARCH_D, max_len=512, sim="cos", tier="kaggle",
                                    note="1.2B model: ~20 s per batch on a 6 GB GPU, so it runs on Kaggle."),
    "e5-large-v2": dict(name="E5-large-v2", kind="st", ckpt="intfloat/e5-large-v2", q_prefix="query: ",
                        d_prefix="passage: ", max_len=512, sim="cos", tier="local"),
    "gritlm-7b-litsearch": dict(name="GritLM-7B", kind="gritlm", ckpt="GritLM/GritLM-7B", q_instr=LITSEARCH_GRIT,
                                max_len=2048, sim="cos", tier="kaggle"),
    "gpt4o-rerank-bm25": dict(name="GPT-4o reranking (w/ BM25)", kind="api", tier="api"),
    "gpt4o-onehop-bm25": dict(name="GPT-4o one-hop (w/ BM25)", kind="api", tier="api"),
    "gpt4o-rerank-gritlm": dict(name="GPT-4o reranking (w/ GritLM)", kind="api", tier="api"),
    "gpt4o-onehop-gritlm": dict(name="GPT-4o one-hop (w/ GritLM)", kind="api", tier="api"),
    # ---------------- A2 DORIS-MAE
    "e5-large-v2-doris": dict(name="E5-L-v2", kind="st", ckpt="intfloat/e5-large-v2", q_prefix="query: ",
                              d_prefix="passage: ", max_len=512, sim="cos", tier="local"),
    "rocketqa-v2": dict(name="RocketQA-v2", kind="external", tier="external",
                        note="PaddlePaddle model; run with the DORIS-MAE repo, drop result JSON in results/."),
    "simcse": dict(name="SimCSE", kind="hf", ckpt="princeton-nlp/sup-simcse-bert-base-uncased", pool="cls",
                   max_len=512, sim="cos", tier="local", assumed=True),
    "splade-v2": dict(name="SPLADE-v2", kind="splade", ckpt="naver/splade_v2_max", max_len=256, tier="local",
                      assumed=True),
    "specter-v2": dict(name="SPECTER-v2", kind="specter2", ckpt="allenai/specter2_base", adapter="allenai/specter2",
                       max_len=512, sim="l2", tier="local"),
    "sentbert": dict(name="SentBERT", kind="st", ckpt="sentence-transformers/all-mpnet-base-v2", max_len=384,
                     sim="cos", tier="local", modes=["sentence"], assumed=True),
    "ts-aspire": dict(name="TSAspire", kind="external", tier="external",
                      note="Needs allenai/aspire code (multi-vector, L2); drop result JSON in results/."),
    "ot-aspire": dict(name="OTAspire", kind="external", tier="external",
                      note="Needs allenai/aspire code (optimal transport); drop result JSON in results/."),
    "ance-firstp": dict(name="ANCE-FirstP", kind="st", ckpt="sentence-transformers/msmarco-roberta-base-ance-firstp",
                        max_len=512, sim="dot", tier="local", modes=["sentence"]),
    "llama-7b": dict(name="LLAMA", kind="hf", ckpt="huggyllama/llama-7b", pool="mean", max_len=512, sim="cos",
                     tier="kaggle", assumed=True),
    "simlm": dict(name="SimLM", kind="hf", ckpt="intfloat/simlm-base-msmarco-finetuned", pool="cls", max_len=512,
                  sim="dot", tier="local"),
    "colbert-v2": dict(name="ColBERT-v2", kind="colbert", ckpt="colbert-ir/colbertv2.0", tier="local",
                       modes=["paragraph", "sentence"]),
    "ernie": dict(name="ERNIE", kind="hf", ckpt="nghuyong/ernie-2.0-base-en", pool="mean", max_len=512, sim="cos",
                  tier="local"),
    "scibert": dict(name="SciBERT", kind="hf", ckpt="allenai/scibert_scivocab_uncased", pool="mean", max_len=512,
                    sim="cos", tier="local"),
    "ada-002": dict(name="ada-002", kind="api", tier="api"),
    # ---------------- A3 IRPAPERS
    "arctic-embed-2": dict(name="Arctic-embed 2.0", kind="st", ckpt="Snowflake/snowflake-arctic-embed-l-v2.0",
                           q_prompt_name="query", max_len=1024, sim="cos", tier="local"),
    "hybrid-text": dict(name="Hybrid text (BM25 + Arctic, RSF)", kind="hybrid", parts=["bm25", "arctic-embed-2"],
                        tier="local"),
    "colmodernvbert": dict(name="ColModernVBERT", kind="colpali", ckpt="ModernVBERT/colmodernvbert",
                           family="colmodernvbert", modality="image", tier="local"),
    "colpali": dict(name="ColPali", kind="colpali", ckpt="vidore/colpali-v1.3", family="colpali", modality="image",
                    tier="kaggle", assumed=True),
    "colqwen2": dict(name="ColQwen2", kind="colpali", ckpt="vidore/colqwen2-v1.0", family="colqwen2",
                     modality="image", tier="kaggle", assumed=True),
    "mm-hybrid": dict(name="Multimodal hybrid (text hybrid + ColModernVBERT)", kind="mm_hybrid",
                      parts=["hybrid-text", "colmodernvbert"], modality="image", tier="local"),
    "muvera": dict(name="MUVERA (ColModernVBERT FDE, ef=1024)", kind="muvera", base="colmodernvbert",
                   modality="image", tier="local"),
    "voyage-3-large": dict(name="Voyage 3 Large", kind="api", tier="api"),
    "cohere-embed-v4": dict(name="Cohere Embed v4", kind="api", tier="api"),
    # ---------------- A4 MAPLE
    "specter2-maple": dict(name="SPECTER2", kind="specter2", ckpt="allenai/specter2_base", adapter=None,
                           max_len=512, sim="cos", tier="local"),
    "scincl": dict(name="SciNCL", kind="hf", ckpt="malteos/scincl", pool="cls", max_len=512, sim="cos", tier="local"),
    "bge-m3": dict(name="BGE-M3", kind="st", ckpt="BAAI/bge-m3", max_len=2048, sim="cos", tier="local",
                   note="Run at 2,048 tokens on a 6 GB GPU (MAPLE used its 8,192 max); recorded in each result."),
    "instructor-xl-maple": dict(name="Instructor-XL", kind="instructor", ckpt="hkunlp/instructor-xl",
                                q_instr=MAPLE_Q, d_instr=None, max_len=512, sim="cos", tier="kaggle",
                                note="1.2B model: ~20 s per batch on a 6 GB GPU, so it runs on Kaggle."),
    "qwen3-embed-8b": dict(name="Qwen3-Embed-8B", kind="st", ckpt="Qwen/Qwen3-Embedding-8B",
                           q_prefix=f"Instruct: {MAPLE_Q}\nQuery:", max_len=8192, sim="cos", tier="kaggle"),
    "gritlm-7b-maple": dict(name="GritLM-7B", kind="gritlm", ckpt="GritLM/GritLM-7B", q_instr=MAPLE_Q,
                            max_len=4096, sim="cos", tier="kaggle"),
    "qwen3-vl-embed-8b": dict(name="Qwen3-VL-Embed-8B", kind="external", modality="image", tier="external",
                              note="Screenshot-based; run with the MAPLE repo. Only IRPAPERS has page images."),
    "ops-mm-embed-7b": dict(name="Ops-MM-Embed-7B", kind="external", modality="image", tier="external",
                            note="Screenshot-based; run with the MAPLE repo. Only IRPAPERS has page images."),
}
for k, v in M.items():
    v["key"] = k

ZOO = {
    "A1": ["bm25", "gtr-t5-large", "instructor-xl-litsearch", "e5-large-v2", "gritlm-7b-litsearch",
           "gpt4o-rerank-bm25", "gpt4o-onehop-bm25", "gpt4o-rerank-gritlm", "gpt4o-onehop-gritlm"],
    "A2": ["e5-large-v2-doris", "rocketqa-v2", "ada-002", "simcse", "splade-v2", "specter-v2", "ts-aspire",
           "sentbert", "ot-aspire", "ance-firstp", "llama-7b", "simlm", "bm25", "colbert-v2", "tfidf", "ernie",
           "scibert"],
    "A3": ["bm25", "arctic-embed-2", "hybrid-text", "colmodernvbert", "colpali", "colqwen2", "mm-hybrid",
           "muvera", "voyage-3-large", "cohere-embed-v4"],
    "A4": ["bm25", "specter2-maple", "scincl", "bge-m3", "instructor-xl-maple", "qwen3-embed-8b",
           "gritlm-7b-maple", "qwen3-vl-embed-8b", "ops-mm-embed-7b"],
}

# BM25 parameters per architecture (Pyserini defaults for A1/A2/A4; Weaviate defaults for A3)
BM25_PARAMS = {"A1": (0.9, 0.4), "A2": (0.9, 0.4), "A3": (1.2, 0.75), "A4": (0.9, 0.4)}

# Paper-reported numbers for the diagonal cells (validation gate).
REPORTED = {
    "A1": {  # LitSearch Table 3, "Avg." columns
        "_metrics": ["broad_R@20", "specific_R@5"],
        "bm25": [39.9, 50.0], "gtr-t5-large": [43.8, 39.6], "instructor-xl-litsearch": [56.5, 52.3],
        "e5-large-v2": [55.4, 56.2], "gritlm-7b-litsearch": [70.8, 74.8],
        "gpt4o-rerank-bm25": [59.9, 68.0], "gpt4o-onehop-bm25": [64.8, 68.6],
        "gpt4o-rerank-gritlm": [75.3, 79.2], "gpt4o-onehop-gritlm": [73.2, 77.0],
    },
    "A2": {  # DORIS-MAE Table 2
        "_metrics": ["R@5", "R@20", "RP", "NDCGexp@10%", "MRR@10", "MAP"],
        "e5-large-v2-doris": [16.51, 43.77, 37.46, 25.90, 14.85, 40.49],
        "rocketqa-v2": [15.63, 45.41, 34.36, 30.30, 20.87, 40.18],
        "ada-002": [15.38, 42.84, 35.81, 27.46, 19.88, 40.37],
        "simcse": [14.90, 42.62, 35.27, 26.88, 21.19, 39.02],
        "splade-v2": [14.78, 40.14, 31.65, 26.08, 17.82, 37.23],
        "specter-v2": [14.50, 43.36, 33.41, 25.65, 17.19, 37.12],
        "ts-aspire": [14.26, 41.25, 33.81, 26.63, 15.59, 37.00],
        "sentbert": [14.09, 44.69, 33.79, 21.88, 13.23, 37.75],
        "ot-aspire": [13.34, 42.25, 33.63, 25.52, 14.18, 36.70],
        "ance-firstp": [13.21, 34.54, 30.51, 20.30, 13.87, 34.53],
        "llama-7b": [12.74, 34.51, 28.33, 16.65, 11.78, 31.29],
        "simlm": [12.68, 35.67, 28.90, 18.91, 11.29, 33.06],
        "bm25": [8.47, 30.50, 21.94, 13.23, 9.19, 25.99],
        "colbert-v2": [8.45, 27.86, 22.33, 12.57, 6.69, 25.80],
        "tfidf": [10.71, 29.22, 24.79, 18.25, 12.41, 28.77],
        "ernie": [6.49, 22.58, 20.18, 9.66, 3.77, 22.71],
        "scibert": [5.13, 17.99, 17.13, 7.50, 3.41, 20.34],
    },
    "A3": {  # IRPAPERS Table 2 / Section 4.1 / Table 4
        "_metrics": ["R@1", "R@5", "R@20"],
        "hybrid-text": [46, 78, 91], "colmodernvbert": [43, 78, 93], "colpali": [45, 79, 93],
        "colqwen2": [49, 81, 94], "mm-hybrid": [49, 81, 95], "muvera": [41, 75, 88],
        "cohere-embed-v4": [58, 87, 97], "voyage-3-large": [52, 86, 95],
    },
    "A4": {  # MAPLE Table 5
        "_metrics": ["AllAspect@5", "AllAspect@20", "AnyAspect@5", "AnyAspect@20", "AspectCoverage@5",
                     "AspectCoverage@20"],
        "bm25": [0.0, 0.0, 68.6, 89.0, 15.4, 28.3], "specter2-maple": [0.0, 1.4, 38.1, 55.2, 12.3, 22.5],
        "scincl": [0.5, 1.4, 63.3, 75.7, 17.8, 30.7], "bge-m3": [1.0, 2.9, 64.3, 80.0, 21.9, 33.3],
        "instructor-xl-maple": [0.0, 2.9, 68.1, 80.5, 21.0, 34.1],
        "ops-mm-embed-7b": [2.4, 5.2, 59.5, 82.4, 21.3, 40.3],
        "qwen3-embed-8b": [3.3, 11.0, 75.7, 91.0, 29.9, 53.3],
        "qwen3-vl-embed-8b": [5.2, 11.4, 81.0, 91.4, 37.2, 57.3],
        "gritlm-7b-maple": [3.3, 15.7, 89.5, 98.1, 43.0, 61.8],
    },
}


def model(key: str) -> dict:
    return M[key]


def applicable(arch: str, key: str, task_has_images: bool, dataset: str = "") -> tuple:
    """(status, reason) for a model in a given cell before running."""
    m = M[key]
    if m["tier"] == "api":
        return "api", "Paid API model: listed, not run (open-models-only configuration)."
    if m.get("modality") == "image" and dataset == "maple":
        return "external", ("MAPLE page screenshots live in the 390 GB pdf_shards; run this model with the "
                            "MAPLE repo and drop its result JSON into results/.")
    if m.get("modality") == "image" and not task_has_images:
        return "na", "Needs page images; this dataset has none (text-only)."
    if m["tier"] == "external":
        return "external", m.get("note", "Needs the authors' code.")
    return "runnable", ""
