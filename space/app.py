"""Before/after annotator for the S1-mini clean-up LoRA (decision 0003).

Three columns (Daniel, 2026-10-29): the original raw input; stock S1-mini's output with its changes
against the original marked (struck = removed, green = added); and the edit box, which starts as
S1-mini's output, so the saved edit records where Daniel diverges from S1-mini. Haiku's draft
(under his foundation prompt) can be loaded into the edit box with a button.
"No change needed" saves the raw input unchanged (a counterexample). Each save is committed to the
dataset as annotations/<id>.jsonl with the signed-in user's OAuth token. The next item comes from the
use case with the fewest annotations, so the mix stays balanced.
"""

from __future__ import annotations

import difflib
import html
import json
import os
import random
from datetime import datetime, timezone

import gradio as gr
from huggingface_hub import HfApi, hf_hub_download

DATASET = os.environ.get("DATASET_REPO", "danielrosehill/ASR-Cleanup-Before-After")
api = HfApi()


def load_pool() -> list[dict]:
    path = hf_hub_download(DATASET, "pool/pool.jsonl", repo_type="dataset", force_download=True)
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


POOL = load_pool()
BY_ID = {r["id"]: r for r in POOL}


def done_ids() -> set[str]:
    files = api.list_repo_files(DATASET, repo_type="dataset")
    return {f[len("annotations/"):-len(".jsonl")] for f in files if f.startswith("annotations/") and f.endswith(".jsonl")}


def diff_html(a: str, b: str) -> str:
    """Word diff of raw a against text b: removed words struck in red, added words in green."""
    aw, bw = a.split(), b.split()
    out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=aw, b=bw, autojunk=False).get_opcodes():
        if op in ("delete", "replace"):
            out.append(f'<del style="color:#b42318">{html.escape(" ".join(aw[i1:i2]))}</del>')
        if op in ("insert", "replace"):
            out.append(f'<ins style="color:#067647;text-decoration:none;background:#dcfae6">{html.escape(" ".join(bw[j1:j2]))}</ins>')
        if op == "equal":
            out.append(html.escape(" ".join(aw[i1:i2])))
    return f'<div style="line-height:1.6;font-size:15px">{" ".join(out)}</div>'


def pick_next(skipped: list[str]) -> tuple[dict | None, str]:
    done = done_ids()
    todo = [r for r in POOL if r["id"] not in done and r["id"] not in skipped]
    counts: dict[str, int] = {}
    for i in done:
        if i in BY_ID:
            counts[BY_ID[i]["use_case"]] = counts.get(BY_ID[i]["use_case"], 0) + 1
    progress = (f"**{len(done)}** annotated of {len(POOL)} · "
                + " · ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    if not todo:
        return None, progress
    least = min(counts.get(r["use_case"], 0) for r in todo)
    return random.choice([r for r in todo if counts.get(r["use_case"], 0) == least]), progress


def start(item: dict) -> str:
    return item.get("s1") or item["draft"]


def show(item: dict | None, progress: str):
    if item is None:
        return None, "", "", "", "Nothing left to annotate.", progress
    terms = f" · keep: {', '.join(item['protect_terms'])}" if item.get("protect_terms") else ""
    meta = f"**{item['id']}** · {item['use_case']} · {item['words_in']} words{terms}"
    original = f'<div style="line-height:1.6;font-size:15px;white-space:pre-wrap">{html.escape(item["input"])}</div>'
    s1 = diff_html(item["input"], item["s1"]) if item.get("s1") else "<em>No S1-mini output for this item.</em>"
    return item["id"], original, s1, start(item), meta, progress


def on_load(skipped):
    return show(*pick_next(skipped or []))


def on_skip(item_id, skipped):
    skipped = (skipped or []) + ([item_id] if item_id else [])
    return (*show(*pick_next(skipped)), skipped)


def save(item_id, output, skipped, force, profile, oauth_token, no_change=False):
    if profile is None or oauth_token is None:
        raise gr.Error("Sign in with Hugging Face first.")
    if not item_id:
        raise gr.Error("No item loaded.")
    item = BY_ID[item_id]
    output = item["input"] if no_change else output.strip()
    missing = [t for t in item.get("protect_terms") or [] if t not in output]
    if missing and not force:
        raise gr.Error(f"Protected term missing from the output: {', '.join(missing)}. Tick 'Save anyway' if that's intended.")
    record = {
        "id": item_id, "source": item["source"], "use_case": item["use_case"], "control": item["control"],
        "input": item["input"], "s1": item.get("s1"), "s1_model": item.get("s1_model"),
        "draft": item["draft"], "draft_model": item["draft_model"], "output": output,
        "edited_vs_s1": output != item.get("s1"), "edited_vs_draft": output != item["draft"],
        "no_change": output == item["input"],
        "protect_terms": item.get("protect_terms") or [], "words_in": item["words_in"],
        "annotator": profile.username, "annotated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    HfApi(token=oauth_token.token).upload_file(
        path_or_fileobj=(json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8"),
        path_in_repo=f"annotations/{item_id}.jsonl", repo_id=DATASET, repo_type="dataset",
        commit_message=f"Annotate {item_id}" + (" (no change)" if record["no_change"] else ""),
    )
    gr.Info(f"Saved {item_id}.")
    return (*show(*pick_next(skipped or [])), skipped, False)


def on_save(item_id, output, skipped, force, profile: gr.OAuthProfile | None, oauth_token: gr.OAuthToken | None):
    return save(item_id, output, skipped, force, profile, oauth_token)


def on_no_change(item_id, output, skipped, force, profile: gr.OAuthProfile | None, oauth_token: gr.OAuthToken | None):
    return save(item_id, output, skipped, force, profile, oauth_token, no_change=True)


with gr.Blocks(title="ASR Clean-up Annotator") as demo:
    gr.Markdown(
        "# ASR clean-up: before → after\n"
        "**Original** is the raw transcript; **S1-mini** is stock S1-mini's clean-up with its changes marked; "
        "**Your edit** starts as S1-mini's text: fix it into the output you want, then **Save & next**. "
        "**No change needed** saves the original as the answer; **Load Haiku draft** swaps in Haiku's version. "
        f"Saves go to [`{DATASET}`](https://huggingface.co/datasets/{DATASET})."
    )
    gr.LoginButton()
    item_id, skipped = gr.State(None), gr.State([])
    meta, progress = gr.Markdown(), gr.Markdown()
    with gr.Row(equal_height=True):
        with gr.Column():
            gr.Markdown("**Original**")
            original = gr.HTML()
        with gr.Column():
            gr.Markdown("**S1-mini** (changes vs original)")
            s1 = gr.HTML()
        with gr.Column():
            after = gr.Textbox(label="Your edit", lines=18, max_lines=60, autoscroll=False)
    with gr.Row():
        save_btn = gr.Button("Save & next", variant="primary")
        same_btn = gr.Button("No change needed")
        reset_btn = gr.Button("Reset to S1-mini")
        haiku_btn = gr.Button("Load Haiku draft")
        skip_btn = gr.Button("Skip")
        force = gr.Checkbox(label="Save anyway (protected term removed on purpose)", value=False)

    outs = [item_id, original, s1, after, meta, progress]
    demo.load(on_load, [skipped], outs)
    save_btn.click(on_save, [item_id, after, skipped, force], outs + [skipped, force])
    same_btn.click(on_no_change, [item_id, after, skipped, force], outs + [skipped, force])
    skip_btn.click(on_skip, [item_id, skipped], outs + [skipped])
    reset_btn.click(lambda i: start(BY_ID[i]) if i else "", [item_id], [after])
    haiku_btn.click(lambda i: BY_ID[i]["draft"] if i else "", [item_id], [after])

if __name__ == "__main__":
    demo.launch()
