"""Before/after annotator for the S1-mini clean-up LoRA (decision 0003).

Three columns (Daniel, 2026-10-29): the original raw input; stock S1-mini's output with its changes
against the original marked (struck = removed, green = added); and the edit box, which starts as
S1-mini's output, so the saved edit records where Daniel diverges from S1-mini. Haiku's draft
(under his foundation prompt) can be loaded into the edit box with a button.
"No change needed" saves the raw input unchanged (a counterexample). Each save is committed to the
dataset as annotations/<id>.jsonl with the signed-in user's OAuth token. The next item comes from the
use case with the fewest annotations, so the mix stays balanced.

Deploy (restarts the Space, which re-reads pool/pool.jsonl):
  hf upload danielrosehill/ASR-Cleanup-Annotator space/ . --repo-type space --exclude "__pycache__/*"
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
            out.append(f'<del style="color:#9a3412;text-decoration-color:#ea580c">{html.escape(" ".join(aw[i1:i2]))}</del>')
        if op in ("insert", "replace"):
            out.append(f'<ins style="color:#067647;text-decoration:none;background:#dcfae6">{html.escape(" ".join(bw[j1:j2]))}</ins>')
        if op == "equal":
            out.append(html.escape(" ".join(aw[i1:i2])))
    return f'<div>{" ".join(out)}</div>'


def copyable(text: str) -> str:
    """The original as clickable sentences inside paragraphs, for building the edit by copy-paste (Daniel,
    2026-10-29). Click copies a sentence; shift-click copies from the last clicked sentence to this one (a
    paragraph-to-be); clicking a paragraph's ¶ copies the whole paragraph. The handlers are COPY_JS."""
    paras = [p for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
    out = []
    for p in paras:
        sents = [x for x in re.split(r"(?<=[.!?])\s+(?=\S)", p.strip()) if x]
        spans = " ".join(f'<span class="sent">{html.escape(x)}</span>' for x in sents)
        out.append(f'<p class="para"><span class="pmark" title="Copy paragraph">¶</span> {spans}</p>')
    return f'<div class="copyable">{"".join(out)}</div>'


COPY_JS = """
<script>
(() => {
  let last = null;
  const copy = async (text, els) => {
    try { await navigator.clipboard.writeText(text); }
    catch (e) { const t = document.createElement('textarea'); t.value = text; document.body.appendChild(t); t.select(); document.execCommand('copy'); t.remove(); }
    els.forEach(el => { el.classList.add('copied'); setTimeout(() => el.classList.remove('copied'), 700); });
  };
  document.addEventListener('click', e => {
    const all_ = e.target.closest('.card .copyall');
    if (all_) { copy(all_.parentElement.querySelector('.fulltext').textContent, [all_]); return; }
    const mark = e.target.closest('.copyable .pmark');
    if (mark) { const s = [...mark.parentElement.querySelectorAll('.sent')]; copy(s.map(x => x.textContent).join(' '), s); return; }
    const el = e.target.closest('.copyable .sent');
    if (!el || window.getSelection().toString()) return;  // a drag-select still copies normally
    const all = [...el.closest('.copyable').querySelectorAll('.sent')];
    let pick = [el];
    if (e.shiftKey && last && all.includes(last)) {
      const [a, b] = [all.indexOf(last), all.indexOf(el)].sort((x, y) => x - y);
      pick = all.slice(a, b + 1);
      window.getSelection().removeAllRanges();
    }
    last = el;
    copy(pick.map(x => x.textContent).join(' '), pick);
  });
})();
</script>
"""


def pick_next(skipped: list[str]) -> tuple[dict | None, str]:
    done, removed = done_ids()
    # hist-* are being replaced by their rewrites, hist2-* (before_after.py rewrite, 2026-10-29); never serve them
    todo = [r for r in POOL if r["id"] not in done and r["id"] not in removed and r["id"] not in skipped
            and not r["id"].startswith("hist-")]
    counts: dict[str, int] = {}
    for i in done:
        if i in BY_ID:
            counts[BY_ID[i]["use_case"]] = counts.get(BY_ID[i]["use_case"], 0) + 1
    progress = (f"**{len(done)}** annotated of {len(POOL) - len(removed & set(BY_ID))} · {len(removed)} removed · "
                + " · ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    if not todo:
        return None, progress
    # Review queue first (2026-10-29): synthetic pairs (`pairs`, `target`) and carried edits to check before training
    todo = [r for r in todo if r.get("review")] or todo
    least = min(counts.get(r["use_case"], 0) for r in todo)
    return random.choice([r for r in todo if counts.get(r["use_case"], 0) == least]), progress


def start(item: dict) -> str:
    """The learned draft (`before_after.py adapt`, few-shot on Daniel's saved edits) when the pool has one, else
    stock S1-mini (Daniel, 2026-10-29: start closer to his edits so there is less to change)."""
    return item.get("adapted") or item.get("s1") or item["draft"]


def start_kind(item: dict) -> str:
    return "adapted" if item.get("adapted") else "s1" if item.get("s1") else "draft"


def show(item: dict | None, progress: str):
    if item is None:
        return None, "", "", "", "Nothing left to annotate.", progress
    terms = f" · keep: {', '.join(item['protect_terms'])}" if item.get("protect_terms") else ""
    began = {"adapted": f"learned draft ({item.get('adapt_round')})", "s1": "S1-mini", "draft": "Haiku draft"}[start_kind(item)]
    meta = f"**{item['id']}** · {item['use_case']} · {item['words_in']} words{terms} · edit starts from {began}"
    original = card("Original · click a sentence to copy it, shift-click another to copy the run between", copyable(item["input"]), "original", item["input"])
    s1 = card("S1-mini (changes vs original)", diff_html(item["input"], item["s1"]) if item.get("s1") else "<em>No S1-mini output.</em>", "s1", item.get("s1"))
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
        "adapted": item.get("adapted"), "adapt_round": item.get("adapt_round"), "start": start_kind(item),
        "edited_vs_s1": output != item.get("s1"), "edited_vs_draft": output != item["draft"],
        "edited_vs_start": output != start(item),
        "no_change": output == item["input"],
        "protect_terms": item.get("protect_terms") or [], "words_in": item["words_in"],
        "annotator": profile.username, "annotated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    DONE.add(item_id)
    commit(oauth_token.token, f"annotations/{item_id}.jsonl", (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8"),
           f"Annotate {item_id}" + (" (no change)" if record["no_change"] else ""))
    return (*show(*pick_next(skipped or [])), skipped, False)


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
#edit textarea { font-size: 18px !important; line-height: 1.6 !important; background: #f3fbf6 !important; border-color: #b7e4c7 !important; }
.card.original { background: #fdf3f2; border-color: #f4c7c3; }
.card.s1 { background: #fff7ed; border-color: #fbd5a8; }
.copyable .para { margin: 0 0 10px; }
.copyable .sent { cursor: copy; border-radius: 4px; padding: 1px 2px; transition: background .2s; }
.copyable .sent:hover { background: #fde2df; }
.copyable .sent.copied { background: #b7e4c7; }
.copyable .pmark { cursor: copy; color: #b42318; opacity: .45; font-weight: 600; margin-right: 2px; }
.copyable .pmark:hover { opacity: 1; }
.card .copyall { float: right; cursor: copy; text-transform: none; letter-spacing: 0; font-weight: 500; padding: 1px 8px; border: 1px solid #d0d5dd; border-radius: 6px; background: #fff; }
.card .copyall:hover { background: #f2f4f7; }
.card .copyall.copied { background: #b7e4c7; }
"""


def card(label: str, body: str, kind: str = "", full: str | None = None) -> str:
    """`full`: the card's whole text, behind a copy icon (Daniel, 2026-10-29: paste S1-mini or the original over
    the learned draft and start from there). Kept in a hidden <pre> so paragraph breaks survive the copy."""
    btn = (f'<span class="copyall" title="Copy all">⧉ copy</span><pre class="fulltext" hidden>{html.escape(full)}</pre>'
           if full else "")
    return f'<div class="card {kind}"><div class="label">{label}{btn}</div>{body}</div>'


with gr.Blocks(title="ASR Clean-up Annotator", css=CSS, head=COPY_JS) as demo:
    gr.Markdown(
        "# ASR clean-up: before → after\n"
        "Left: the **original** and **S1-mini**'s clean-up (red struck = removed, green = added). Right: **your edit**, "
        "starting from the learned draft (re-drafted from your earlier edits) or, without one, S1-mini's text. Fix it, then **Save & next**. **No change needed** saves the original as the answer; "
        "**Remove (invalid)** drops the item. Click a sentence of the original to copy it (shift-click a second one for the run between). "
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
            after = gr.Textbox(label="Your edit", lines=10, max_lines=60, autoscroll=False, elem_id="edit")
            with gr.Row():
                save_btn = gr.Button("Save & next", variant="primary")
                same_btn = gr.Button("No change needed")
            with gr.Row():
                reset_btn = gr.Button("Reset to S1-mini", size="sm")
                learned_btn = gr.Button("Learned draft", size="sm")
                haiku_btn = gr.Button("Load Haiku draft", size="sm")
                skip_btn = gr.Button("Skip", size="sm")
                remove_btn = gr.Button("Remove (invalid)", variant="stop", size="sm")
            force = gr.State(False)

    outs = [item_id, original, s1, after, meta, progress]
    demo.load(on_load, [skipped], outs)
    save_btn.click(on_save, [item_id, after, skipped, force], outs + [skipped, force])
    same_btn.click(on_no_change, [item_id, after, skipped, force], outs + [skipped, force])
    remove_btn.click(on_remove, [item_id, skipped], outs + [skipped])
    skip_btn.click(on_skip, [item_id, skipped], outs + [skipped])
    reset_btn.click(lambda i: (BY_ID[i].get("s1") or BY_ID[i]["draft"]) if i else "", [item_id], [after])
    haiku_btn.click(lambda i: BY_ID[i]["draft"] if i else "", [item_id], [after])
    learned_btn.click(lambda i: start(BY_ID[i]) if i else "", [item_id], [after])

if __name__ == "__main__":
    demo.launch()
