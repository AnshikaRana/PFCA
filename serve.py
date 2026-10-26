"""Local results website.

    python serve.py            -> http://localhost:8000
    python serve.py --port 8080 --results path/to/results

The page re-reads results/ every time it loads (and auto-refreshes every 30 s), so results appear
as soon as each run finishes. Nothing is sent anywhere: it only listens on 127.0.0.1.
"""
import argparse
import json
import os
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

from xaudit.aggregate import build

HERE = os.path.dirname(os.path.abspath(__file__))


class Handler(SimpleHTTPRequestHandler):
    results_dir = os.path.join(HERE, "results")

    def do_GET(self):
        if self.path.split("?")[0] == "/data.json":
            body = json.dumps(build(self.results_dir)).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        return super().do_GET()

    def log_message(self, *a):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--results", default=os.environ.get("XAUDIT_RESULTS", os.path.join(HERE, "results")))
    a = ap.parse_args()
    Handler.results_dir = a.results
    os.makedirs(os.path.join(a.results, "tasks"), exist_ok=True)
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), partial(Handler, directory=os.path.join(HERE, "site")))
    print(f"PFCA dashboard: http://localhost:{a.port}   (results from {a.results})  Ctrl+C to stop")
    srv.serve_forever()


if __name__ == "__main__":
    main()
