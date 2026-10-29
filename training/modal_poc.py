"""Proof-of-concept LoRA of S1-mini by Superwhisper on Daniel's before/after edits, on Modal (Daniel, 2026-10-29:
"train up a first experimental version so I can see if this is going to work"). Not the stage-3 run.

  with-secrets uv run --with anthropic --with huggingface_hub python training/modal_poc.py data   # -> data/private/poc/
  MODAL_PROFILE=danielrosehill uv run --with modal modal run training/modal_poc.py              # train + held-out infer

data   Pairs from the saved annotations (scripts/before_after.py `annotations`): items with a spoken edit command
       (COMMAND) and the old hist-* items (replaced by their rewrites, `rewrite`) are left out; the 21 hist2-* items
       that carry Daniel's edit transposed (`carried_output`) go in as training pairs. A fixed 9 of the real edits,
       one or two per use case, are held out as test. -> data/private/poc/{train,test}.jsonl, control [Layers: 1,2].
run    training/s1mini_lora.py `train` on an L4/A10G (rank 16, 4 epochs, batch 4, lr 2e-4), then `infer` on the
       test set twice: stock S1-mini (its own control line) and base + adapter. -> data/private/poc/{base,lora}.jsonl
       and the adapter in data/private/poc/adapter/ (not pushed: the data includes obfuscated history).
"""

from __future__ import annotations

import io
import json
import random
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
POC = ROOT / "data" / "private" / "poc"
CONTROL = "[Layers: 1,2]"


def data() -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    import before_after as b
    recs = [r for r in b.annotations() if not b.COMMAND.search(r["input"]) and not r["id"].startswith("hist-")]
    rng, test = random.Random(0), []
    for uc in sorted({r["use_case"] for r in recs}):
        k = [r for r in recs if r["use_case"] == uc]
        test += rng.sample(k, min(len(k) // 4 or 1, 2))
    test = test[:9]
    tid = {r["id"] for r in test}
    pair = lambda i, inp, out, src: {"id": i, "control": CONTROL, "input": inp, "output": out, "source": src}
    train = [pair(r["id"], r["input"], r["output"], "annotation") for r in recs if r["id"] not in tid]
    train += [pair(h["id"], h["input"], h["carried_output"], "carried") for h in b.read("history2.jsonl") if "carried_output" in h]
    POC.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("test", [pair(r["id"], r["input"], r["output"], "annotation") for r in test])):
        (POC / f"{name}.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")
    print(f"train {len(train)} ({sum(x['source'] == 'carried' for x in train)} carried), test {len(test)} -> {POC}")


if __name__ == "__main__" and sys.argv[1:] == ["data"]:
    data()
    sys.exit()

import modal  # noqa: E402

app = modal.App("s1-mini-poc-lora")
image = (modal.Image.debian_slim(python_version="3.12")
         .uv_pip_install("torch>=2.8", "transformers>=5.20,<6", "peft>=0.21,<1", "trl>=1.16,<2", "datasets>=3.0", "accelerate>=1.0")
         .add_local_file(ROOT / "training" / "s1mini_lora.py", "/root/s1mini_lora.py"))


@app.function(image=image, gpu=["L4", "A10G"], timeout=3600)
def run(train: str, test: str) -> dict:
    import subprocess
    Path("/tmp/train.jsonl").write_text(train)
    Path("/tmp/test.jsonl").write_text(test)
    py = ["python", "/root/s1mini_lora.py"]
    subprocess.run(py + ["train", "--data", "/tmp/train.jsonl", "--output-dir", "/tmp/adapter", "--epochs", "4",
                         "--batch-size", "4", "--grad-accum", "1", "--lr", "2e-4"], check=True)
    subprocess.run(py + ["infer", "--data", "/tmp/test.jsonl", "--layers", "none", "--out", "/tmp/base.jsonl"], check=True)
    subprocess.run(py + ["infer", "--data", "/tmp/test.jsonl", "--adapter", "/tmp/adapter", "--out", "/tmp/lora.jsonl"], check=True)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        t.add("/tmp/adapter", arcname="adapter")
    return {"base": Path("/tmp/base.jsonl").read_text(), "lora": Path("/tmp/lora.jsonl").read_text(), "adapter": buf.getvalue()}


@app.local_entrypoint()
def main():
    res = run.remote((POC / "train.jsonl").read_text(), (POC / "test.jsonl").read_text())
    (POC / "base.jsonl").write_text(res["base"])
    (POC / "lora.jsonl").write_text(res["lora"])
    tarfile.open(fileobj=io.BytesIO(res["adapter"]), mode="r:gz").extractall(POC)
    print(f"wrote {POC}/base.jsonl, lora.jsonl, adapter/")
