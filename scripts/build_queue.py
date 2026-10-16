"""Build the editing queue for the triples set: real raw dictation, chunked to editable size.

  python scripts/build_queue.py            # writes data/triples/queue.jsonl

Standard library only. Sources (all public, read 2026-10-16):

  danielrosehill/Voice-Notes-Screened         data/notes.jsonl, 65 real voice notes (Voicenotes ASR)
  danielrosehill/Noisy-Voice-Notes            transcripts/*.md, 7 real noisy voice notes (ElevenLabs Scribe)
  danielrosehill/Transcription-Cleanup-Trainer data/whisper-transcripts/1.txt, one 6-minute answer (whisper-1)
  context/transcript-raw.txt                  this project's own voice note (AssemblyAI Universal)

None of these is Parakeet output; see docs/decisions/0002-triples-training-data.md for why they are used anyway.
Ids are stable across rebuilds (source id + chunk index), so saved edits stay attached to their item.
Items added in the editor (source "new") are carried over on rebuild.
"""

from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "triples" / "queue.jsonl"
HF = "https://huggingface.co/datasets/danielrosehill"

MAX_WORDS = 300  # S1-mini's card recommends under ~1,000 tokens; 300 words also keeps one item editable
TARGET_WORDS = 220


def fetch(url: str) -> str:
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read().decode("utf-8")


def flatten(text: str) -> str:
    """Parakeet emits one block with no line breaks; match that."""
    text = re.sub(r"<br\s*/?>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def chunk(text: str) -> list[str]:
    words = text.split()
    if len(words) <= MAX_WORDS:
        return [text]
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, cur = [], []
    for s in sentences:
        cur.append(s)
        if sum(len(x.split()) for x in cur) >= TARGET_WORDS:
            chunks.append(" ".join(cur))
            cur = []
    if cur:
        if chunks and sum(len(x.split()) for x in cur) < 60:
            chunks[-1] += " " + " ".join(cur)  # fold a short tail into the previous chunk
        else:
            chunks.append(" ".join(cur))
    return chunks


def items(source: str, source_id: str, asr: str, text: str):
    parts = chunk(flatten(text))
    for i, part in enumerate(parts):
        suffix = f"-c{i + 1}" if len(parts) > 1 else ""
        yield {
            "id": f"{source}-{source_id}{suffix}",
            "source": source,
            "source_id": source_id,
            "chunk": f"{i + 1}/{len(parts)}",
            "asr": asr,
            "raw": part,
            "words": len(part.split()),
        }


def main() -> None:
    rows = []

    for line in fetch(f"{HF}/Voice-Notes-Screened/resolve/main/data/notes.jsonl").splitlines():
        if line.strip():
            n = json.loads(line)
            rows += items("vns", n["id"], "voicenotes.com pipeline (engine not recorded)", n["text"])

    meta = fetch(f"{HF}/Noisy-Voice-Notes/resolve/main/metadata.csv").splitlines()
    for line in meta[1:]:
        nid = line.split(",", 1)[0]
        md = fetch(f"{HF}/Noisy-Voice-Notes/resolve/main/transcripts/{nid}.md")
        body = md.split("\n---\n", 1)[-1]  # drop YAML front matter
        body = re.sub(r"^#.*$", "", body, flags=re.M)  # drop the title heading
        rows += items("nvn", nid, "ElevenLabs Scribe (via voicenotes.com)", body)

    rows += items(
        "tct", "1", "openai/whisper-1",
        fetch(f"{HF}/Transcription-Cleanup-Trainer/resolve/main/data/whisper-transcripts/1.txt"),
    )
    rows += items("voicenote", "2026-10-16", "AssemblyAI Universal", (ROOT / "context" / "transcript-raw.txt").read_text())

    # Keep items pasted into the editor with "+ New"; they exist nowhere else.
    if OUT.exists():
        rows += [r for r in map(json.loads, OUT.read_text().splitlines()) if r.get("source") == "new"]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    by = {}
    for r in rows:
        by[r["source"]] = by.get(r["source"], 0) + 1
    print(f"{len(rows)} items -> {OUT.relative_to(ROOT)}  {by}")


if __name__ == "__main__":
    main()
