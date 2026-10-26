"""Download the four datasets into data/raw/ (about 12 GB). Works on Windows, macOS and Linux.
Run:  python download_data.py      (safe to re-run; finished parts are skipped)"""
import os
import urllib.request

from datasets import load_dataset
from huggingface_hub import snapshot_download

os.chdir(os.path.dirname(os.path.abspath(__file__)))
for d in ["litsearch", "doris_mae", "irpapers", "maple"]:
    os.makedirs(os.path.join("data", "raw", d), exist_ok=True)

print("== LitSearch")
for cfg in ["query", "corpus_clean"]:
    out = f"data/raw/litsearch/{cfg}.jsonl"
    if not os.path.exists(out):
        load_dataset("princeton-nlp/LitSearch", cfg, split="full").to_json(out)

print("== DORIS-MAE (1.1 GB, may take a while)")
out = "data/raw/doris_mae/DORIS-MAE_dataset_v1.json"
if not os.path.exists(out):
    urllib.request.urlretrieve(
        "https://zenodo.org/records/8299749/files/DORIS-MAE_dataset_v1.json?download=1", out + ".part")
    os.replace(out + ".part", out)

print("== IRPAPERS")
if not os.path.exists("data/raw/irpapers/queries.jsonl"):
    load_dataset("weaviate/IRPAPERS", "queries", split="train").to_json("data/raw/irpapers/queries.jsonl")
if not os.path.exists("data/raw/irpapers/docs_text.jsonl"):
    # The docs table is one 3.8 GB CSV whose base64 page images are too large for `datasets`/Arrow,
    # so stream it row by row: text goes to docs_text.jsonl, each page image to its own file.
    import base64, csv, json, sys
    from huggingface_hub import hf_hub_download
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    path = hf_hub_download("weaviate/IRPAPERS", "irpapers-docs.csv", repo_type="dataset")
    os.makedirs("data/raw/irpapers/pages", exist_ok=True)
    n = 0
    with open(path, newline="", encoding="utf-8") as f, \
         open("data/raw/irpapers/docs_text.jsonl.part", "w", encoding="utf-8") as out:
        for r in csv.DictReader(f):
            b64 = r.pop("base64_str", "") or ""
            if b64:
                with open(f"data/raw/irpapers/pages/{r['dataset_id']}.img", "wb") as im:
                    im.write(base64.b64decode(b64))
            r["pdf_id_x"] = int(float(r["pdf_id_x"]))
            r["page_number"] = int(float(r["page_number"]))
            out.write(json.dumps(r) + "\n")
            n += 1
    os.replace("data/raw/irpapers/docs_text.jsonl.part", "data/raw/irpapers/docs_text.jsonl")
    print("IRPAPERS pages:", n)

print("== MAPLE (only the ~5 GB we need; the full repo is 401 GB)")
import time
for attempt in range(1, 21):   # network drops are common on long downloads: retry, resuming each time
    try:
        snapshot_download("kai-02/MAPLE", repo_type="dataset", local_dir="data/raw/maple", max_workers=2,
                          allow_patterns=["queries/*", "corpus/*", "representations/abstract/*",
                                          "representations/fulltext/*", "README.md"])
        break
    except Exception as e:
        print(f"MAPLE download attempt {attempt} failed ({type(e).__name__}); retrying in 30 s...", flush=True)
        time.sleep(30)
else:
    raise SystemExit("MAPLE download kept failing - check the internet connection and re-run.")

print("Done. Next: python -m xaudit.run --prepare")
