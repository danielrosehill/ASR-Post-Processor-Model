"""Stock S1-mini by Superwhisper on every before/after pool item, on Modal (GPU), for the annotator's middle column.

  MODAL_PROFILE=danielrosehill modal run scripts/s1_modal.py   # reads data/private/{history2 (else history),vocab,fill}.jsonl and every synth round (data/synth/<tag>.json), writes data/private/s1.jsonl

Prompt: training/s1mini_lora.py `render_prompt(text, None)`, i.e. stock S1-mini's pinned control line
(`[Styling: semi-formal] [Structure: prose] [Context: general]`, no Layers axis), greedy decoding,
max_new_tokens = 1.3 x input tokens + 32, as the card prescribes. Model pinned to BASE_REVISION.
Items already in s1.jsonl are skipped.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
PRIVATE = ROOT / "data" / "private"
sys.path.insert(0, str(ROOT / "training"))

app = modal.App("s1-mini-before-after")
image = (modal.Image.debian_slim(python_version="3.12")
         .uv_pip_install("torch==2.8.0", "transformers>=4.51,<5", "accelerate")
         .add_local_file(ROOT / "training" / "s1mini_lora.py", "/root/s1mini_lora.py"))


@app.function(image=image, gpu=["L4", "A10G", "T4"], timeout=1800)  # first free GPU type
def generate(texts: list[str]) -> list[str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import s1mini_lora as fmt

    tok = AutoTokenizer.from_pretrained(fmt.BASE_MODEL, revision=fmt.BASE_REVISION, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(fmt.BASE_MODEL, revision=fmt.BASE_REVISION,
                                                 torch_dtype=torch.bfloat16).to("cuda").eval()
    prompts = [fmt.render_prompt(t, None) for t in texts]
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))  # similar lengths per batch
    out: dict[int, str] = {}
    for k in range(0, len(order), 16):
        idx = order[k:k + 16]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        n = max(len(tok(texts[i], add_special_tokens=False).input_ids) for i in idx)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=int(1.3 * n) + 32, do_sample=False,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        for j, i in enumerate(idx):
            out[i] = tok.decode(gen[j, enc.input_ids.shape[1]:], skip_special_tokens=True).strip()
        print(f"{len(out)}/{len(texts)} done", flush=True)
    return [out[i] for i in range(len(texts))]


@app.local_entrypoint()
def main():
    hist = "history2.jsonl" if (PRIVATE / "history2.jsonl").exists() else "history.jsonl"  # before_after.py rewrite
    items = [json.loads(l) for f in [hist, "vocab.jsonl", "fill.jsonl"] + [f"{p.stem}.jsonl" for p in sorted((ROOT / "data" / "synth").glob("*.json"))] if (PRIVATE / f).exists()
             for l in (PRIVATE / f).read_text(encoding="utf-8").splitlines()]
    path = PRIVATE / "s1.jsonl"
    have = {json.loads(l)["id"] for l in path.read_text().splitlines()} if path.exists() else set()
    todo = [it for it in items if it["id"] not in have]
    print(f"{len(todo)} items to run")
    if not todo:
        return
    outs = generate.remote([it["input"] for it in todo])
    with path.open("a", encoding="utf-8") as f:
        for it, o in zip(todo, outs):
            f.write(json.dumps({"id": it["id"], "s1": o, "s1_model": "superwhisper/s1-mini@88f6b158"}, ensure_ascii=False) + "\n")
    print(f"wrote {len(todo)} -> data/private/s1.jsonl")
