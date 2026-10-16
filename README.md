# ASR Post-Processor Model

Planning repo for a small, light-touch, text-in/text-out model that turns raw speech-to-text output
into clean, formatted text that can be sent without editing.

The input is what [Parakeet v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) produces in daily
dictation (through the [Whisperian](https://whisperian.app) Android app): already punctuated and cased,
but with fillers, false starts, *gonna*, no paragraph breaks, the odd repeated point, and sometimes
a sentence said to someone at the door. The output is the same text with those fixed and nothing else
changed. Restyling, restructuring and reformatting are a separate, later step.

**Status (2026-10-16): planning.** No data, no model yet. Start with the
[development plan](docs/development-plan.md).

## Why a model and not a prompt or a regex

Dictation gets to 90% usable quickly; the last 10% is what makes text sendable. Part of that gap is
the ASR's job (vocabulary, homophones, Hebrew words in English speech), and part is reshaping the text.
Regex handles the deterministic part. A system-prompted LLM can do the rest but tends to over-edit and
is hard to keep at exactly the same level of touch across a WhatsApp message and a long report. A small
fine-tuned model can be trained on one person's exact preferences. Whether it actually beats a
prompted LLM is tested in Phase 2 of the plan, not assumed.

## The layered taxonomy

| Layer | What | Examples | Owner |
|---|---|---|---|
| 1 | Foundational | fillers, stutters, false starts, *gonna* -> *going to*, pause-induced sentence breaks, paragraph breaks | this model |
| 2 | Light editorial | repeated points, spoken self-corrections, speech aimed at someone else ("Come in. Thanks, Jim.") | this model, as a separate variant |
| 3 | Editing | headings, reordering, formality, format | system-prompted LLM pass |
| - | Not text problems | homophones, vocabulary, code-switched words | ASR fine-tune |

Full list with examples: [`taxonomy/transformations.json`](taxonomy/transformations.json).

## Contents

| Path | What |
|---|---|
| [`docs/development-plan.md`](docs/development-plan.md) | Phased plan, POC first; recommended training-data approach and why |
| [`docs/spec.md`](docs/spec.md) | Input/output contract, model candidates, evaluation method, out of scope |
| [`docs/research.md`](docs/research.md) | Prior art: punctuation, disfluency, ASR error correction, paragraph segmentation, small base models |
| [`taxonomy/transformations.json`](taxonomy/transformations.json) | Machine-readable layered taxonomy |
| [`context/voice-note.mp3`](context/voice-note.mp3) | The original 17-minute voice note setting out the idea (2026-10-16) |
| [`context/transcript-raw.txt`](context/transcript-raw.txt) | Its raw transcript (AssemblyAI Universal), unedited. Also a good long-form test input. |
| [`project.json`](project.json) | Machine-readable project state |

## Related work

- [ASR-Regex-Post-Processing](https://github.com/danielrosehill/ASR-Regex-Post-Processing): Android-safe
  (ICU) regex snippets for ASR cleanup. The deterministic pass that runs before this model.
- [English-Hebrew-Codeswitching-Parakeet-Finetune](https://github.com/danielrosehill/English-Hebrew-Codeswitching-Parakeet-Finetune):
  the ASR-side fine-tune. Vocabulary and code-switching errors belong there, not here.
- [Transcription-Cleanup-Trainer](https://huggingface.co/datasets/danielrosehill/Transcription-Cleanup-Trainer)
  (HF dataset, 2025-12-18): an earlier start on the same problem; one completed raw/cleaned sample.
- [Read-Aloud-Voice-Sentences](https://huggingface.co/datasets/danielrosehill/Read-Aloud-Voice-Sentences)
  (HF dataset): phone recordings with Parakeet-style references; source of no-op pairs.
- [Voice-Notes-Screened](https://huggingface.co/datasets/danielrosehill/Voice-Notes-Screened)
  (HF dataset): 65 privacy-screened real voice-note transcripts, raw; extra evaluation input.
- [Text-Transformation-Prompts-300525](https://huggingface.co/datasets/danielrosehill/Text-Transformation-Prompts-300525)
  (HF dataset): the system-prompt library for the layer-3 reformatting pass.

## Licence

To be decided at first release.
