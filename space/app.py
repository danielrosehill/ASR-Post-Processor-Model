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
import queue
import random
import re
import threading
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


def repo_ids(files: list[str], folder: str, ext: str) -> set[str]:
    return {f[len(folder) + 1:-len(ext)] for f in files if f.startswith(folder + "/") and f.endswith(ext)}


# Advancing used to wait for a Hub commit and then a full file listing (several seconds per item). Now
# the done/removed sets are read once at startup and kept in memory, and commits go through one
# background worker in order; a failed commit is retried and reported on the next action.
_files = api.list_repo_files(DATASET, repo_type="dataset")
DONE, REMOVED = repo_ids(_files, "annotations", ".jsonl"), repo_ids(_files, "removed", ".json")
UPLOADS: queue.Queue = queue.Queue()
FAILED: list[str] = []


def _uploader():
    while True:
        token, path, data, message = UPLOADS.get()
        for attempt in range(3):
            try:
                HfApi(token=token).upload_file(path_or_fileobj=data, path_in_repo=path, repo_id=DATASET,
                                               repo_type="dataset", commit_message=message)
                break
            except Exception as e:  # noqa: BLE001
                if attempt == 2:
                    FAILED.append(f"{path}: {e}")


threading.Thread(target=_uploader, daemon=True).start()


def commit(token: str, path: str, data: bytes, message: str) -> None:
    UPLOADS.put((token, path, data, message))
    if FAILED:
        gr.Warning("Some saves failed and were not written: " + "; ".join(FAILED[-3:]))


def done_ids() -> tuple[set[str], set[str]]:
    return DONE, REMOVED


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
    return f'<div>{" ".join(out)}</div>'


def pick_next(skipped: list[str]) -> tuple[dict | None, str]:
    done, removed = done_ids()
    todo = [r for r in POOL if r["id"] not in done and r["id"] not in removed and r["id"] not in skipped]
    counts: dict[str, int] = {}
    for i in done:
        if i in BY_ID:
            counts[BY_ID[i]["use_case"]] = counts.get(BY_ID[i]["use_case"], 0) + 1
    progress = (f"**{len(done)}** annotated of {len(POOL) - len(removed & set(BY_ID))} · {len(removed)} removed · "
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
    original = card("Original", f'<div style="white-space:pre-wrap">{html.escape(item["input"])}</div>')
    s1 = card("S1-mini (changes vs original)", diff_html(item["input"], item["s1"]) if item.get("s1") else "<em>No S1-mini output.</em>")
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
    # Case- and punctuation-insensitive, and a warning only: the first version blocked saves on false
    # alarms (Daniel, 2026-10-29), e.g. a term that changed case or picked up a hyphen.
    norm = lambda x: re.sub(r"[^0-9a-z\u0590-\u05ff]+", " ", x.lower()).strip()
    missing = [t for t in item.get("protect_terms") or [] if f" {norm(t)} " not in f" {norm(output)} "]
    if missing:
        gr.Warning(f"Saved, but check: protected term not found in your edit: {', '.join(missing)}")
    record = {
        "id": item_id, "source": item["source"], "use_case": item["use_case"], "control": item["control"],
        "input": item["input"], "s1": item.get("s1"), "s1_model": item.get("s1_model"),
        "draft": item["draft"], "draft_model": item["draft_model"], "output": output,
        "edited_vs_s1": output != item.get("s1"), "edited_vs_draft": output != item["draft"],
        "no_change": output == item["input"],
        "protect_terms": item.get("protect_terms") or [], "words_in": item["words_in"],
        "annotator": profile.username, "annotated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    DONE.add(item_id)
    commit(oauth_token.token, f"annotations/{item_id}.jsonl", (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8"),
           f"Annotate {item_id}" + (" (no change)" if record["no_change"] else ""))
    RECENT.insert(0, record)
    del RECENT[8:]
    return (*show(*pick_next(skipped or [])), skipped, False, recent_html())


RECENT: list[dict] = []  # this session's saves, newest first


def recent_html() -> str:
    if not RECENT:
        return "<em>Your saves this session appear here.</em>"
    rows = "".join(
        f'<div class="card small"><div class="label">{html.escape(r["id"])} · {html.escape(r["use_case"])}'
        f'{" · no change" if r["no_change"] else ""}</div>{diff_html(r["input"], r["output"])}</div>' for r in RECENT)
    return rows


def on_remove(item_id, skipped, profile: gr.OAuthProfile | None, oauth_token: gr.OAuthToken | None):
    if profile is None or oauth_token is None:
        raise gr.Error("Sign in with Hugging Face first.")
    if not item_id:
        raise gr.Error("No item loaded.")
    record = {"id": item_id, "removed_by": profile.username,
              "removed_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    REMOVED.add(item_id)
    commit(oauth_token.token, f"removed/{item_id}.json", json.dumps(record).encode("utf-8"), f"Remove {item_id} (invalid)")
    gr.Info(f"Removed {item_id}.")
    return (*show(*pick_next(skipped or [])), skipped)


def on_save(item_id, output, skipped, force, profile: gr.OAuthProfile | None, oauth_token: gr.OAuthToken | None):
    return save(item_id, output, skipped, force, profile, oauth_token)


def on_no_change(item_id, output, skipped, force, profile: gr.OAuthProfile | None, oauth_token: gr.OAuthToken | None):
    return save(item_id, output, skipped, force, profile, oauth_token, no_change=True)


CSS = """
.card { border: 1px solid #d0d5dd; border-radius: 10px; padding: 14px 16px; margin-bottom: 12px; background: #fcfcfd; }
.card .label { font-size: 13px; font-weight: 600; color: #475467; text-transform: uppercase; letter-spacing: .04em; margin-bottom: 6px; }
.card div, .card { font-size: 18px; line-height: 1.6; }
.card.small, .card.small div { font-size: 15px; }
#edit textarea { font-size: 18px !important; line-height: 1.6 !important; }
"""


def card(label: str, body: str) -> str:
    return f'<div class="card"><div class="label">{label}</div>{body}</div>'


with gr.Blocks(title="ASR Clean-up Annotator", css=CSS) as demo:
    gr.Markdown(
        "# ASR clean-up: before → after\n"
        "Left: the **original** and **S1-mini**'s clean-up (red struck = removed, green = added). Right: **your edit**, "
        "starting from S1-mini's text. Fix it, then **Save & next**. **No change needed** saves the original as the answer; "
        "**Remove (invalid)** drops the item. "
        f"Saves go to [`{DATASET}`](https://huggingface.co/datasets/{DATASET})."
    )
    gr.LoginButton()
    item_id, skipped = gr.State(None), gr.State([])
    with gr.Row():
        meta, progress = gr.Markdown(), gr.Markdown()
    with gr.Row():
        with gr.Column(scale=1):
            original = gr.HTML()
            s1 = gr.HTML()
        with gr.Column(scale=1):
            after = gr.Textbox(label="Your edit", lines=14, max_lines=60, autoscroll=False, elem_id="edit")
            with gr.Row():
                save_btn = gr.Button("Save & next", variant="primary")
                same_btn = gr.Button("No change needed")
            with gr.Row():
                reset_btn = gr.Button("Reset to S1-mini", size="sm")
                haiku_btn = gr.Button("Load Haiku draft", size="sm")
                skip_btn = gr.Button("Skip", size="sm")
                remove_btn = gr.Button("Remove (invalid)", variant="stop", size="sm")
            force = gr.State(False)
    gr.Markdown("### Recent saves")
    recent = gr.HTML(recent_html())

    outs = [item_id, original, s1, after, meta, progress]
    demo.load(on_load, [skipped], outs)
    save_btn.click(on_save, [item_id, after, skipped, force], outs + [skipped, force, recent])
    same_btn.click(on_no_change, [item_id, after, skipped, force], outs + [skipped, force, recent])
    remove_btn.click(on_remove, [item_id, skipped], outs + [skipped])
    skip_btn.click(on_skip, [item_id, skipped], outs + [skipped])
    reset_btn.click(lambda i: start(BY_ID[i]) if i else "", [item_id], [after])
    haiku_btn.click(lambda i: BY_ID[i]["draft"] if i else "", [item_id], [after])

if __name__ == "__main__":
    demo.launch()
