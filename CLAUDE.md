# CLAUDE.md

Planning and (later) working repo for a small text-in/text-out model that cleans raw Parakeet v2
dictation. Public repo: never add private material (see "Do not" below).

## Read first, in this order

1. `project.json`: current stage and status.
2. `docs/development-plan.md`: phases with exit tests. Find the first phase whose exit test has not passed; that is the work.
3. `taxonomy/transformations.json`: the scope. Every transformation has an `owner`; only `owner: "postprocessor"` items are this model's job.
4. `docs/spec.md`: input/output contract and evaluation metrics.
5. `context/transcript-raw.txt`: the speaker's own words. When a doc and the transcript disagree about intent, the transcript wins; fix the doc.

## Ground rules

- **POC scale first.** Smallest data and compute that answers the question. Do not scale data before Phase 4 says the approach works.
- **Light touch.** The model preserves wording and meaning. Any transformation that changes structure, order, tone or format belongs to the `llm-pass` owner, not here.
- **Homophones and vocabulary are ASR problems.** Do not add them to post-processor training data; they belong in the ASR fine-tune repos.
- **The 40-clip real evaluation set is never trained on.**
- **Taxonomy ids are labels.** Training pairs record `applied: [ids]` from `taxonomy/transformations.json`. Adding, renaming or re-owning a transformation means editing that file and noting why in `docs/development-plan.md`.
- The laptop has no ML accelerator: training runs on a hosted GPU.
- Python environments use `uv`.

## Related repos and datasets

- `~/repos/github/audio-dictation/ASR-Regex-Post-Processing` (public): the deterministic pre-pass. If a transformation is a safe regex, it goes there.
- `~/repos/github/audio-dictation/English-Hebrew-Codeswitching-Parakeet-Finetune` (public): ASR-side fine-tune.
- HF datasets (all public): `danielrosehill/Transcription-Cleanup-Trainer`, `danielrosehill/Read-Aloud-Voice-Sentences`, `danielrosehill/Voice-Notes-Screened`, `danielrosehill/Text-Transformation-Prompts-300525`. What each is good for: `docs/research.md`.

## Do not

- Link or name private repositories or private datasets in this repo. Check visibility (`gh repo view <repo> --json visibility`, or the HF API `private` field) before adding any link.
- Commit raw voice-note archives or unscreened transcripts.
