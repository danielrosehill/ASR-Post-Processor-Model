"""Merge a LoRA from training/modal_poc.py into S1-mini by Superwhisper and publish one standalone model (Daniel,
2026-10-29: no separate adapter to load). On Modal (CPU): merge, save safetensors, convert to GGUF Q8_0 with
llama.cpp's convert_hf_to_gguf.py, push both to one public HF repo, danielrosehill/S1-ASR-Cleanup, with the S1-mini licence, NOTICE and model card.

  with-secrets bash -c 'MODAL_PROFILE=danielrosehill uv run --with modal modal run training/modal_release.py'

Reads the adapter from data/private/poc/adapter/. The HF token goes to Modal as an ephemeral secret for this run.
Licence (superwhisper/s1-mini LICENSE): Apache 2.0 plus one term, the model keeps its name "S1-mini" by
"Superwhisper", exact capitalisation, in any derivative; LICENSE and NOTICE are kept.
"""

from __future__ import annotations

import io
import os
import tarfile
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
ADAPTER = ROOT / "data" / "private" / "poc" / "adapter"
REPO = "danielrosehill/S1-ASR-Cleanup"  # Daniel, 2026-10-29; still identified as S1-mini by Superwhisper (licence)
VERSION = "v1.0"  # small project tier (release-versioning habit)
BASE, BASE_REV = "superwhisper/s1-mini", "88f6b15896c73bbb13a3b596e0afe8ea0d5150b4"
LLAMA_CPP = "b6900"  # llama.cpp release tag whose convert_hf_to_gguf.py is used

CARD = """---
license: other
license_name: s1-mini-license
license_link: LICENSE
base_model: superwhisper/s1-mini
base_model_relation: finetune
library_name: transformers
pipeline_tag: text-generation
language:
  - en
tags:
  - dictation
  - transcript-cleanup
  - gguf
---

# S1-ASR-Cleanup ({version}): a fine-tune of S1-mini by Superwhisper

S1-ASR-Cleanup is a fine-tune of **S1-mini by Superwhisper** ([`superwhisper/s1-mini`](https://huggingface.co/superwhisper/s1-mini),
Qwen3-0.6B) that cleans raw speech-to-text dictation the way one user edits his own. One standalone model: the LoRA
is merged in, nothing else to load.

## What it does differently from stock S1-mini

- **Paragraphs.** Long dictation is split into short paragraphs, one thought each; greetings and sign-offs on their
  own lines. Stock S1-mini returns one block.
- **Dates, deterministic.** "the twelfth of March", "March twelfth" -> `March 12`; "Thursday the twelfth of March" ->
  `Thursday, March 12`; a bare "by the fifteenth" -> `by the 15th`.
- **Money.** "two thousand two hundred shekels" -> `2,200 NIS`; "forty bucks" -> `$40`; `€35`, `$2 million`.
- **Informal contractions expanded.** gonna, wanna, gotta, hafta, lemme, dunno, 'cause, outta, tryna, c'mon, 'em,
  shoulda and the like become `going to`, `want to`, `have to`, ...; ordinary contractions (I'm, don't) stay.
- **Spelled-out clarifications dropped.** "Ronen, that's R O N E N" -> `Ronen`.
- **Mid-sentence capitals.** The capital (and stray full stop) a recogniser leaves after a pause is lowercased:
  "we could Use those" -> `we could use those`; real capitals (names, I, WhatsApp) stay.
- Also: disfluencies and false starts, self-corrections resolved to the final intent, sentences broken at a pause
  rejoined, restated points and asides to people in the room dropped. Romanised Hebrew words and tech names kept.

## Examples

Outputs of this model and of stock S1-mini (same prompt, greedy) on held-out pairs, not cherry-picked beyond one per
behaviour:

{examples}

**Status: {version}, proof of concept.** Trained on {pairs} before/after pairs: the user's own edits from
[`danielrosehill/ASR-Cleanup-Before-After`](https://huggingface.co/datasets/danielrosehill/ASR-Cleanup-Before-After)
plus synthetic pairs targeting each behaviour above (LoRA rank 16, 4 epochs). Small data; expect rough edges.

## Files

- `model.safetensors` and config: for transformers.
- `s1-asr-cleanup-{version}-Q8_0.gguf`: for llama.cpp and anything built on it.

## Prompt

The same format as S1-mini, with the system prompt verbatim from its card, `enable_thinking=False` (the empty think
block below) and greedy decoding. The control line ends with `[Layers: 1,2]`:

```
<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{control}
<transcript><|im_end|>
<|im_start|>assistant
<think>

</think>

```

Keep inputs under about 1,000 tokens and set `max_new_tokens` to about 1.3 x input tokens + 32.

## Licence

Apache 2.0 with the S1-mini naming term (see `LICENSE` and `NOTICE`): any use, distribution or integration must
identify the model as "S1-mini" by "Superwhisper". Training code:
[`danielrosehill/ASR-Post-Processor-Model`](https://github.com/danielrosehill/ASR-Post-Processor-Model).
"""

app = modal.App("s1-asr-cleanup-release")
image = (modal.Image.debian_slim(python_version="3.12").apt_install("git")
         .uv_pip_install("torch>=2.8", "transformers>=5.20,<6", "peft>=0.21,<1", "accelerate>=1.0", "huggingface_hub",
                         "sentencepiece", "protobuf", "numpy")
         .run_commands(f"git clone --depth 1 --branch {LLAMA_CPP} https://github.com/ggml-org/llama.cpp /llama.cpp",
                       "pip install /llama.cpp/gguf-py")
         .add_local_file(ROOT / "training" / "s1mini_lora.py", "/root/s1mini_lora.py"))


@app.function(image=image, cpu=4, memory=16384, timeout=3600)
def release(adapter_tgz: bytes, pairs: int, examples: str) -> str:
    import json
    import subprocess

    import torch
    from huggingface_hub import HfApi, hf_hub_download
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import s1mini_lora as fmt
    tarfile.open(fileobj=io.BytesIO(adapter_tgz), mode="r:gz").extractall("/tmp", filter="data")
    out = Path("/tmp/merged")
    model = AutoModelForCausalLM.from_pretrained(BASE, revision=BASE_REV, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, "/tmp/adapter").merge_and_unload()
    model.save_pretrained(out)
    AutoTokenizer.from_pretrained(BASE, revision=BASE_REV).save_pretrained(out)
    for f in ("LICENSE", "NOTICE"):
        (out / f).write_bytes(Path(hf_hub_download(BASE, f, revision=BASE_REV)).read_bytes())
    gguf = f"s1-asr-cleanup-{VERSION}-Q8_0.gguf"
    subprocess.run(["python", "/llama.cpp/convert_hf_to_gguf.py", str(out), "--outtype", "q8_0",
                    "--outfile", str(out / gguf)], check=True)
    (out / "README.md").write_text(CARD.format(version=VERSION, pairs=pairs, examples=examples, system=fmt.SYSTEM_PROMPT,
                                               control=fmt.control_line("1,2")))
    (out / "release.json").write_text(json.dumps({"version": VERSION, "base_model": BASE, "base_revision": BASE_REV,
                                                  "pairs": pairs, "lora": json.loads(Path("/tmp/adapter/run.json").read_text())["args"]}, indent=2))
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(REPO, exist_ok=True)
    api.update_repo_settings(REPO, private=False)
    c = api.upload_folder(repo_id=REPO, folder_path=str(out), commit_message=f"{VERSION}: S1-mini by Superwhisper + LoRA on {pairs} pairs")
    api.create_tag(REPO, tag=VERSION, revision=c.oid)
    return f"https://huggingface.co/{REPO} ({VERSION}, {c.oid[:8]})"


@app.local_entrypoint()
def main():
    import json
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        t.add(ADAPTER, arcname="adapter")
    pairs = json.loads((ADAPTER / "run.json").read_text())["pairs"]
    fn = release.with_options(secrets=[modal.Secret.from_dict({"HF_TOKEN": os.environ["HF_TOKEN"]})])
    print(fn.remote(buf.getvalue(), pairs, examples()))


def examples() -> str:
    """One held-out behaviour probe per target (data/private/poc/test.jsonl, source target-* / capitals): input, stock
    S1-mini (base.jsonl), this model (lora.jsonl). Probes are synthetic, so nothing private goes on the card."""
    import json
    rd = lambda f: {r["id"]: r for r in map(json.loads, (ADAPTER.parent / f).read_text().splitlines())}
    test, base, lora = rd("test.jsonl"), rd("base.jsonl"), rd("lora.jsonl")
    seen, parts = set(), []
    for i, t in test.items():
        if t["source"] == "annotation" or t["input"] == t["output"] or t["source"] in seen or i not in lora:
            continue
        seen.add(t["source"])
        q = lambda s: "\n".join("> " + l if l else ">" for l in s.strip().splitlines())
        parts.append(f"**{t['source'].removeprefix('target-')}**\n\nRaw transcript:\n\n{q(t['input'])}\n\n"
                     f"Stock S1-mini:\n\n{q(base[i]['prediction'])}\n\nS1-ASR-Cleanup:\n\n{q(lora[i]['prediction'])}")
    return "\n\n".join(parts)
