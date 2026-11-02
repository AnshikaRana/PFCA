"""Build the static GitHub Pages copy of the results dashboard into docs/.

    python build_pages.py

It writes docs/index.html (the same page as site/index.html), docs/data.json (all result files,
the same payload serve.py builds live) and docs/analysis.json. Re-run it after adding results or
running `python xaudit/analysis.py`, then commit docs/. GitHub Pages serves the folder at
https://anshikarana.github.io/PFCA/ (Settings -> Pages -> Deploy from branch: main, folder /docs).
"""
import json
import os
import shutil

from xaudit.aggregate import build

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(HERE, "docs")


def main():
    os.makedirs(DOCS, exist_ok=True)
    data = build(os.path.join(HERE, "results"))
    with open(os.path.join(DOCS, "data.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    shutil.copyfile(os.path.join(HERE, "site", "analysis.json"), os.path.join(DOCS, "analysis.json"))
    with open(os.path.join(HERE, "site", "index.html"), encoding="utf-8") as f:
        html = f.read()
    # Static hosting: no live server, so no auto-refresh and a different hint if data.json is missing.
    html = html.replace("load(); setInterval(load, 30000);", "load();")
    html = html.replace("last refreshed (auto every 30 s)", "results snapshot (static copy)")
    html = html.replace("Start the site with <code>python serve.py</code>.",
                        "Run <code>python build_pages.py</code> and commit <code>docs/</code>.")
    with open(os.path.join(DOCS, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)
    open(os.path.join(DOCS, ".nojekyll"), "w").close()
    print("docs/ updated:", data.get("n_results"), "results")


if __name__ == "__main__":
    main()
