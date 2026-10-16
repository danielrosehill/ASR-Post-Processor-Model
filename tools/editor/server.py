# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.8",
#   "transformers>=5.20,<6",
# ]
# ///
"""Local triples editor: raw transcript -> stock S1-mini output -> your edit.

  uv run tools/editor/server.py            # then open http://127.0.0.1:8765

Reads data/triples/queue.jsonl (built by scripts/build_queue.py), writes your edits to
data/triples/triples.jsonl and caches S1-mini outputs in data/triples/s1-cache.jsonl.
S1-mini runs on the CPU; outputs for the whole queue are generated in a background
thread at startup, and the "Run S1-mini" button regenerates one item on demand.
Binds to localhost only. See docs/decisions/0002-triples-training-data.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training"))
import s1mini_lora as fmt  # noqa: E402  single source of the prompt format

DATA = ROOT / "data" / "triples"
QUEUE = DATA / "queue.jsonl"
TRIPLES = DATA / "triples.jsonl"
CACHE = DATA / "s1-cache.jsonl"
HTML = Path(__file__).with_name("index.html")
TAXONOMY = json.loads((ROOT / "taxonomy" / "transformations.json").read_text())

lock = threading.Lock()  # guards files and the in-memory dicts
model_lock = threading.Lock()  # one generation at a time
_model = {}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


def write_jsonl(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    tmp.replace(path)


def key(raw: str) -> str:
    return hashlib.sha256(f"{fmt.BASE_REVISION}\n{fmt.control_line(None)}\n{raw}".encode()).hexdigest()[:16]


queue = read_jsonl(QUEUE)
triples = {t["id"]: t for t in read_jsonl(TRIPLES)}
cache = {c["key"]: c for c in read_jsonl(CACHE)}
status = {"model": "not loaded", "precompute": "idle"}


def s1(raw: str, force: bool = False) -> dict:
    k = key(raw)
    if not force and k in cache:
        return cache[k]
    with model_lock:
        if "m" not in _model:
            status["model"] = "loading"
            from transformers import AutoModelForCausalLM, AutoTokenizer

            _model["tok"] = AutoTokenizer.from_pretrained(fmt.BASE_MODEL, revision=fmt.BASE_REVISION)
            _model["m"] = AutoModelForCausalLM.from_pretrained(fmt.BASE_MODEL, revision=fmt.BASE_REVISION, dtype="auto").eval()
            status["model"] = "ready"
        import torch

        tok, m = _model["tok"], _model["m"]
        enc = tok(fmt.render_prompt(raw, None), return_tensors="pt", add_special_tokens=False)
        n = enc.input_ids.shape[1]
        t0 = time.time()
        with torch.no_grad():
            out = m.generate(**enc, max_new_tokens=int(1.3 * n) + 32, do_sample=False)
        text = tok.decode(out[0][n:], skip_special_tokens=True).strip()
    rec = {"key": k, "output": text, "seconds": round(time.time() - t0, 1), "control": fmt.control_line(None),
           "model": fmt.BASE_MODEL, "revision": fmt.BASE_REVISION}
    with lock:
        cache[k] = rec
        write_jsonl(CACHE, list(cache.values()))
    return rec


def precompute() -> None:
    todo = [q for q in queue if key(q["raw"]) not in cache]
    for i, q in enumerate(todo, 1):
        status["precompute"] = f"{i}/{len(todo)}"
        try:
            s1(q["raw"])
        except Exception as e:  # keep going; the item can be retried from the GUI
            print(f"S1-mini failed on {q['id']}: {e}", file=sys.stderr)
    status["precompute"] = "done"


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code: int, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def body(self) -> dict:
        return json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            return self.send(200, HTML.read_bytes(), "text/html; charset=utf-8")
        if self.path == "/api/state":
            with lock:
                items = [q | {"s1": cache.get(key(q["raw"]))} for q in queue]
                return self.send(200, {
                    "items": items,
                    "triples": triples,
                    "status": status,
                    "taxonomy": [{"id": t["id"], "layer": t["layer"], "description": t["description"]}
                                 for t in TAXONOMY["transformations"] if t["owner"] == "postprocessor"],
                })
        if self.path == "/api/status":
            return self.send(200, status | {"cached": len(cache)})
        self.send(404, {"error": "not found"})

    def do_POST(self):
        b = self.body()
        if self.path == "/api/s1":
            item = next((q for q in queue if q["id"] == b["id"]), None)
            if not item:
                return self.send(404, {"error": "unknown id"})
            return self.send(200, s1(item["raw"], force=bool(b.get("force"))))
        if self.path == "/api/save":
            rec = b["triple"]
            if rec.get("control") not in fmt.LAYER_CONTROLS.values():
                return self.send(400, {"error": "bad control"})
            rec["updated"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            with lock:
                triples[rec["id"]] = rec
                order = {q["id"]: i for i, q in enumerate(queue)}
                write_jsonl(TRIPLES, sorted(triples.values(), key=lambda t: order.get(t["id"], 1e9)))
            return self.send(200, {"ok": True, "saved": len(triples)})
        if self.path == "/api/add":
            raw = " ".join(b["raw"].split())
            if not raw:
                return self.send(400, {"error": "empty"})
            item = {"id": f"new-{key(raw)[:8]}", "source": "new", "source_id": None, "chunk": "1/1",
                    "asr": b.get("asr") or "nvidia/parakeet-tdt-0.6b-v2", "raw": raw, "words": len(raw.split())}
            with lock:
                if not any(q["id"] == item["id"] for q in queue):
                    queue.append(item)
                    write_jsonl(QUEUE, queue)
            threading.Thread(target=s1, args=(raw,), daemon=True).start()
            return self.send(200, item)
        self.send(404, {"error": "not found"})


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-precompute", action="store_true")
    a = p.parse_args()
    if not queue:
        sys.exit(f"{QUEUE} is empty; run python scripts/build_queue.py first")
    if not a.no_precompute:
        threading.Thread(target=precompute, daemon=True).start()
    print(f"{len(queue)} items, {len(triples)} saved, {len(cache)} S1-mini outputs cached")
    print(f"open http://127.0.0.1:{a.port}")
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
