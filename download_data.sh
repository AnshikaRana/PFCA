#!/usr/bin/env bash
# Download the four datasets into data/raw/ (about 12 GB in total).
# Safe to re-run: downloads resume.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data/raw/{litsearch,doris_mae,irpapers,maple}

echo "== LitSearch (HF princeton-nlp/LitSearch)"
python - <<'EOF'
from datasets import load_dataset
for cfg in ["query", "corpus_clean"]:
    ds = load_dataset("princeton-nlp/LitSearch", cfg, split="full")
    ds.to_json(f"data/raw/litsearch/{cfg}.jsonl")
    print(cfg, len(ds))
EOF

echo "== DORIS-MAE (Zenodo 8299749, dataset JSON only, 1.1 GB)"
wget -c -O data/raw/doris_mae/DORIS-MAE_dataset_v1.json \
  "https://zenodo.org/records/8299749/files/DORIS-MAE_dataset_v1.json?download=1"

echo "== IRPAPERS (HF weaviate/IRPAPERS; page images stay in the HF cache for the image models)"
python - <<'EOF'
from datasets import load_dataset
q = load_dataset("weaviate/IRPAPERS", "queries", split="train"); q.to_json("data/raw/irpapers/queries.jsonl")
d = load_dataset("weaviate/IRPAPERS", "docs", split="train")
d.remove_columns(["base64_str"]).to_json("data/raw/irpapers/docs_text.jsonl")
print("queries", len(q), "pages", len(d))
EOF

echo "== MAPLE (HF kai-02/MAPLE) -- ONLY queries, corpus, abstract + fulltext (~5.1 GB). The repo is 401 GB: never download it all."
if command -v hf >/dev/null 2>&1; then HF=hf; else HF=huggingface-cli; fi
$HF download kai-02/MAPLE --repo-type dataset \
  --include "queries/*" "corpus/*" "representations/abstract/*" "representations/fulltext/*" "README.md" \
  --local-dir data/raw/maple

echo "Done. Next: python -m xaudit.run --prepare"
