# 0002: POC training data is about 100 hand-edited triples (raw → S1-mini → speaker)

- **Date:** 2026-10-16
- **Status:** accepted
- **Supersedes:** the 400-pair synthetic set as the primary POC training data (Phase 1 step 3 of
  [`../development-plan.md`](../development-plan.md)), which becomes optional augmentation
- **Depends on:** [0001](0001-fine-tune-s1-mini.md)

## Context

The target, in the speaker's words from [`../../context/transcript-raw.txt`](../../context/transcript-raw.txt):
a light-touch, general-purpose rewrite, good enough for everything from a WhatsApp message to a report, but "not so
specific that it's already too shaped for general text". In a follow-up request (2026-10-16) the speaker put it as
"not too much, not too little".

Since the base is S1-mini (0001), the cheapest way to define that level is to show the speaker what S1-mini already
does and record only where they disagree.

S1-mini's training data cannot be recovered. Checked 2026-10-16: the `superwhisper` account on HF has two repos,
`s1-mini` and `s1-mini-GGUF`, and no datasets. The model card describes only the 7,519-case held-out set, not the
training set. Stock S1-mini's outputs on our own raw text are the next best view of its behaviour.

## Decision

1. **Triples, about 100.** Each item is a raw transcript, stock S1-mini's output on it (its pinned control line, no
   `Layers` axis), and the speaker's final version.
2. **The speaker edits S1-mini's output by default, not the raw text.** The editor pre-fills the "Yours" box with
   S1-mini's output. "Start from raw" is one click away for items where S1-mini went wrong, and the choice is recorded
   per item as `started_from`.
   - Less typing: most of layer 1 is already done.
   - The difference between S1-mini's output and the speaker's version is exactly what the adapter has to learn, and
     the editor shows it live.
   - "S1-mini is right" is a one-click verdict. The accept rate measures how close stock S1-mini already is to the
     target, which feeds the Phase 2 B2 exit test directly.
   - Risk: anchoring, meaning the speaker accepts S1-mini choices they would not have made themselves. Mitigations:
     the diff can be switched to compare against raw, and the guide text in the editor states the target.
3. **Training pairs are raw → final**, not S1-mini output → final. At inference the adapter sees raw text. The
   S1-mini output is kept in each triple for analysis and for B2 scoring.
4. **One general-purpose level, labelled `[Layers: 1,2]`.** The speaker's "not too much, not too little" level
   includes repetition and non-directed-speech removal, which are layer 2. A per-item "Layer 1 only" toggle exists
   but is optional. Separate layer-1 targets are deferred to the Phase 4 A/B test. If one level turns out to be all
   that is wanted, the `Layers` axis can be dropped.
5. **Raw sources: real dictation that is already public, chunked to at most 300 words** (98 items, built by
   [`../../scripts/build_queue.py`](../../scripts/build_queue.py)):

   | Source | Items | ASR |
   |---|---|---|
   | [`Voice-Notes-Screened`](https://huggingface.co/datasets/danielrosehill/Voice-Notes-Screened) (65 notes) | 71 | voicenotes.com pipeline |
   | [`Noisy-Voice-Notes`](https://huggingface.co/datasets/danielrosehill/Noisy-Voice-Notes) (7 notes) | 12 | ElevenLabs Scribe |
   | [`Transcription-Cleanup-Trainer`](https://huggingface.co/datasets/danielrosehill/Transcription-Cleanup-Trainer) sample 1 | 4 | whisper-1 |
   | [`context/transcript-raw.txt`](../../context/transcript-raw.txt) | 11 | AssemblyAI Universal |

   None of it is Parakeet output, so its punctuation habits differ from the deployment input. That is accepted for
   the POC: real spontaneous speech carries the disfluencies that matter, which synthetic text imitates badly.
   Fresh Parakeet dictations can be pasted into the editor ("+ New", recorded as `source: "new"`) to top the set up
   and shift the mix towards Parakeet.
6. **Evaluation is unchanged:** the 40 recorded scenario clips stay evaluation-only and go only to
   `danielrosehill/Parakeet-Dictation-Cleanup-Eval`. Two inputs move from "supplementary evaluation" to training:
   `Voice-Notes-Screened` and `context/transcript-raw.txt`. `Noisy-Voice-Notes` describes itself as an evaluation set;
   it is used for training here because nothing else evaluates against it in this project.

This repeats the layout of `Transcription-Cleanup-Trainer` (raw / automatic cleanup / manual cleanup), with S1-mini
replacing Gemini as the automatic step and an editor that makes each item quick to do.

## Workflow

```bash
python scripts/build_queue.py        # data/triples/queue.jsonl (already built; rerun only to refresh)
uv run tools/editor/server.py        # http://127.0.0.1:8765; S1-mini outputs fill in the background
python scripts/pairs.py from-triples data/triples/triples.jsonl > data/triples/pairs.jsonl
python scripts/pairs.py validate data/triples/pairs.jsonl
```

Files in `data/triples/`, all committed:

| File | Written by | Content |
|---|---|---|
| `queue.jsonl` | `build_queue.py`, editor "+ New" | `id`, `source`, `source_id`, `chunk`, `asr`, `raw`, `words` |
| `s1-cache.jsonl` | editor | Stock S1-mini output per raw text, keyed by a hash of revision, control line and raw text |
| `triples.jsonl` | editor | `raw`, `s1_output`, `final`, `control`, `started_from`, `verdict` (`edited` / `s1-accepted` / `raw-kept` / `skipped`), `applied`, `notes` |

## Consequences

- Phase 0's style guide (five to ten hand-cleaned examples) is now the first ten items done in the editor.
- The Phase 1 exit test becomes: at least 90 non-skipped triples, with the S1-mini accept rate recorded. Taxonomy
  coverage is checked from the optional `applied` labels where present. The synthetic set is generated only if the
  first adapter underfits a category the triples rarely contain, such as `L2.non-directed-speech`.
- If the accept rate is high (most items need no edit), that is the Phase 2 signal that stock S1-mini is close
  enough and the fine-tune should narrow to the categories the speaker did edit.
