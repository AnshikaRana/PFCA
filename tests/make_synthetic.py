"""Write a TINY synthetic copy of all four raw dataset formats, for testing the pipeline only.
These numbers are meaningless and must never be shown as results (tests write to a temp folder)."""
import json
import os
import random
import sys

WORDS = ("retrieval dense sparse transformer contrastive citation graph aspect query abstract benchmark "
         "pruning distillation tokenizer multimodal image page layout ocr fusion ranking reranker embedding "
         "scientific literature dataset annotation hierarchy pooling chunk holistic encoder decoder").split()


def text(rng, topic, n):
    return " ".join(rng.choice(topic * 3 + WORDS) for _ in range(n)).capitalize() + "."


def main(root):
    rng = random.Random(0)
    topics = [rng.sample(WORDS, 4) for _ in range(40)]
    w = lambda p, rows: (os.makedirs(os.path.dirname(p), exist_ok=True),  # noqa: E731
                         open(p, "w").write("\n".join(json.dumps(r) for r in rows) + "\n"))
    raw = os.path.join(root, "raw")
    # LitSearch
    corpus = [{"corpusid": 1000 + i, "title": " ".join(topics[i]), "abstract": text(rng, topics[i], 40),
               "citations": [], "full_paper": " ".join(text(rng, topics[i], 60) for _ in range(30))}
              for i in range(40)]
    qs = [{"query_set": rng.choice(["inline_acl", "manual_iclr"]), "query": "papers on " + text(rng, topics[i], 8),
           "specificity": i % 2, "quality": 2, "corpusids": [1000 + i] + ([1000 + i + 1] if i % 5 == 0 else [])}
          for i in range(12)]
    w(os.path.join(raw, "litsearch", "corpus_clean.jsonl"), corpus)
    w(os.path.join(raw, "litsearch", "query.jsonl"), qs)
    # DORIS-MAE
    abstracts = [{"original_abstract": text(rng, topics[i % 40], 50), "title": " ".join(topics[i % 40]),
                  "abstract_id": i} for i in range(120)]
    queries, ann, id2asp, aid = [], [], {}, 0
    for qi in range(6):
        aspects = {}
        sents = []
        for a in range(3):
            id2asp[str(aid)] = text(rng, topics[qi * 3 + a], 10)
            sents.append(id2asp[str(aid)])
            aspects[str(aid)] = [str(aid + 1)]
            id2asp[str(aid + 1)] = text(rng, topics[qi * 3 + a], 5)
            aid += 2
        pool = rng.sample(range(120), 30)
        for p in pool:
            for k, subs in aspects.items():
                for x in [k] + subs:
                    match = abstracts[p]["title"].split()[0] in id2asp[x].lower()
                    ann.append({"aspect_id": int(x), "abstract_id": p, "score": 2 if match else rng.choice([0, 0, 1])})
        queries.append({"query_text": " ".join(sents), "query_type": "ML", "idea_from": 0, "candidate_pool": pool,
                        "aspects": aspects})
    w_json = {"aspect2aspect_id": {v: k for k, v in id2asp.items()}, "aspect_id2aspect": id2asp, "Query": queries,
              "Corpus": abstracts, "Annotation": ann}
    os.makedirs(os.path.join(raw, "doris_mae"), exist_ok=True)
    json.dump(w_json, open(os.path.join(raw, "doris_mae", "DORIS-MAE_dataset_v1.json"), "w"))
    # IRPAPERS
    docs, irq = [], []
    for pdf in range(8):
        for pg in range(1, 7):
            docs.append({"dataset_id": f"{pdf}_{pg}", "pdf_id_x": pdf, "pdf_title": " ".join(topics[pdf]),
                         "page_number": pg, "transcription": text(rng, topics[(pdf * 6 + pg) % 40], 120)})
            if pg <= 3:
                irq.append({"dataset_id": f"{pdf}_{pg}", "pdf_id": pdf, "pdf_title": " ".join(topics[pdf]),
                            "page_number": pg, "question": "what " + text(rng, topics[(pdf * 6 + pg) % 40], 8),
                            "answer": "x"})
    w(os.path.join(raw, "irpapers", "docs_text.jsonl"), docs)
    w(os.path.join(raw, "irpapers", "queries.jsonl"), irq)
    # MAPLE
    base = os.path.join(raw, "maple")
    pids = [f"paper-{i}-abcdef1234" for i in range(50)]
    w(os.path.join(base, "corpus", "corpus-00000-of-00001.jsonl"), [{"paper_id": p, "title": p} for p in pids])
    w(os.path.join(base, "representations", "abstract", "abstract-00000-of-00001.jsonl"),
      [{"paper_id": p, "representation_type": "abstract", "text": text(rng, topics[i % 40], 40)}
       for i, p in enumerate(pids)])
    w(os.path.join(base, "representations", "fulltext", "fulltext-00000-of-00001.jsonl"),
      [{"paper_id": p, "representation_type": "fulltext", "text": text(rng, topics[i % 40], 900)}
       for i, p in enumerate(pids)])
    mq = [{"query_id": f"q{i}", "query_text": "find papers " + text(rng, topics[i % 10], 8),
           "aspect_group": ["motivation", "method", "experiment/result"][i % 3], "target_paper_id": pids[i % 10],
           "is_multimodal": False} for i in range(30)]
    w(os.path.join(base, "queries", "text_referenced.jsonl"), mq)
    print("synthetic data written to", root)


if __name__ == "__main__":
    main(sys.argv[1])
