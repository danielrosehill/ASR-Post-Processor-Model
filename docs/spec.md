# Spec

Version 0.1, 2026-10-16. Draft for Phase 0 of [`development-plan.md`](development-plan.md).

## Input/output contract

**Input**

- One transcript from one ASR session, UTF-8 English text.
- Expected source: `nvidia/parakeet-tdt-0.6b-v2` (stock or a personal fine-tune), optionally after the
  regex pre-pass from [ASR-Regex-Post-Processing](https://github.com/danielrosehill/ASR-Regex-Post-Processing).
  So: punctuated, cased, digits mostly already normalised, a single block with no line breaks.
- Length: 1 word to roughly 2,000 words (a 15-minute note). Inputs over the model's comfortable window
  (target: 1,000 tokens, the same limit S1-mini documents) are chunked on sentence boundaries with one
  sentence of overlap by the caller; the model itself sees one chunk.
- A control line selecting the layers to apply, adopting the pattern S1-mini uses
  (see [`research.md`](research.md#s1-mini-is-superwhisper-s1-mini)):
  `[Layers: 1]` or `[Layers: 1,2]`. On the wire it is appended to S1-mini's own pinned control line
  (`[Styling: semi-formal] [Structure: prose] [Context: general] [Layers: 1]`); training pairs store only the
  `[Layers: …]` part. Full prompt: [`fine-tuning.md`](fine-tuning.md#prompt-format). One model with a control line, not two models, is the default (fewest
  moving parts). Two separately trained variants are the fallback if the control line does not separate cleanly.

**Output**

- The cleaned text only. No preamble, no quotes, no explanation.
- Paragraphs separated by exactly one blank line (`\n\n`). No other Markdown: no headings, bullets or emphasis.
- Wording is preserved. The model may only:
  - delete tokens (fillers, repetitions, reparanda, non-directed speech),
  - expand the informal contractions listed under `L1.informal-contractions`,
  - fix punctuation and casing at a join it made (`L1.pause-induced-sentence-breaks`) or after a deletion,
  - insert paragraph breaks.
- Never inserts new content words, never changes numbers, names or technical terms, never changes
  spelling of a word it keeps (that is the ASR fine-tune's job).
- If the whole input is filler or non-directed speech, return an empty string.
- If nothing needs fixing, return the input unchanged. This is expected to be the most common case for short messages.

## Transformation taxonomy

Machine-readable source of truth: [`../taxonomy/transformations.json`](../taxonomy/transformations.json).
Ids are the labels attached to every training pair.

| Id | Layer | Owner | Status |
|---|---|---|---|
| `L1.filler-removal` | 1 | postprocessor | in scope |
| `L1.stutter-repetition` | 1 | postprocessor | in scope |
| `L1.false-start-repair` | 1 | postprocessor | in scope |
| `L1.informal-contractions` | 1 | postprocessor | in scope |
| `L1.pause-induced-sentence-breaks` | 1 | postprocessor | in scope |
| `L1.paragraph-boundaries` | 1 | postprocessor | in scope; fallback is a separate segmenter |
| `L2.content-repetition` | 2 | postprocessor | in scope |
| `L2.spoken-self-correction` | 2 | postprocessor | in scope |
| `L2.non-directed-speech` | 2 | postprocessor | experimental |
| `L2.train-of-thought-recovery` | 2 | postprocessor | experimental |
| `L2.verbal-hedges` | 2 | postprocessor | experimental |
| `L3.headings-and-structure` | 3 | llm-pass | excluded |
| `L3.reordering` | 3 | llm-pass | excluded |
| `L3.style-and-formality` | 3 | llm-pass | excluded |
| `X.homophones` | - | asr-finetune | excluded |
| `X.vocabulary-and-code-switching` | - | asr-finetune | excluded |
| `X.cross-utterance-restart` | - | out-of-scope | excluded |
| `X.punctuation-casing-from-scratch` | - | out-of-scope | excluded |
| `X.spoken-formatting-commands` | - | out-of-scope (regex) | excluded |
| `X.number-and-entity-formatting` | - | out-of-scope (ASR + regex) | excluded |

Layer 1 is "every output gets this". Layer 2 is light editorial judgement, gated by the control line so it can
be A/B tested for going too far. Layer 3 is editing proper and belongs to a system-prompted LLM pass
(the prompts in [`Text-Transformation-Prompts-300525`](https://huggingface.co/datasets/danielrosehill/Text-Transformation-Prompts-300525)).

### The boundary question

The voice note's central question is which layer of the stack owns what. The working rule:

| Question | If yes |
|---|---|
| Does deciding need the audio (acoustics, the speaker's accent)? | ASR fine-tune |
| Is it a fixed pattern with near-zero false positives? | Regex pre-pass |
| Does it need context, but keeps wording and meaning? | This model |
| Does it change structure, order, tone, or depend on where the text is going? | LLM pass |

The rule is a hypothesis. Phase 2 baselines test it: any category the frontier-LLM baseline (B4) cannot do
reliably from text alone moves out of this model.

## Model candidates

Parameter counts are the HF `safetensors.total` field, read 2026-10-16. Tied-embedding models are counted
twice by that field; the unique count is given where the card states it.

| Model | Params | Licence | Role |
|---|---|---|---|
| [`google/gemma-3-270m-it`](https://huggingface.co/google/gemma-3-270m-it) | 268M | Gemma terms | Smallest probe: does a model this size learn layer 1 at all? |
| [`HuggingFaceTB/SmolLM2-360M-Instruct`](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) | 362M | Apache-2.0 | Alternative tiny probe with a permissive licence |
| [`superwhisper/s1-mini`](https://huggingface.co/superwhisper/s1-mini) | 752M stored / 596M unique | Apache-2.0 + naming clause | **Chosen POC base (2026-10-16).** See [`fine-tuning.md`](fine-tuning.md) |
| [`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B) | 752M stored / 596M unique | Apache-2.0 | Control run: S1-mini's base, so the comparison isolates S1-mini's cleanup training |
| [`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B) | 873M | Apache-2.0 | Fallback. Newest small Qwen (2026-02-28) |
| [`Qwen/Qwen3.5-2B`](https://huggingface.co/Qwen/Qwen3.5-2B) | 2.27B | Apache-2.0 | Step-up if 0.8B fails layer 2 |
| [`google/flan-t5-base`](https://huggingface.co/google/flan-t5-base) | 248M | Apache-2.0 | Seq2seq control: encoder-decoder suits copy-heavy rewriting |

Decision (2026-10-16): LoRA on S1-mini, with the same LoRA on Qwen3-0.6B as the control. It starts already
knowing fillers and self-corrections, is a proven size for this task, and is under the 1B budget in `project.json`.
Gemma 3 270M, Qwen3.5-0.8B and the others are fallbacks if both runs fail the Phase 3 exit.

## Training data format

JSONL, one pair per line:

```json
{
  "id": "syn-000123",
  "source": "synthetic | real-parakeet | read-aloud-noop | taxonomy-example",
  "use_case": "message | email | note | long",
  "control": "[Layers: 1,2]",
  "input": "raw transcript text",
  "output": "cleaned text",
  "applied": ["L1.filler-removal", "L1.paragraph-boundaries", "L2.non-directed-speech"],
  "asr_model": "nvidia/parakeet-tdt-0.6b-v2 | null for synthetic",
  "words_in": 182
}
```

`taxonomy-example` marks the smoke-test pairs only. Validate a file with `python scripts/pairs.py validate <file>`.

Each synthetic raw text yields two pairs, one per control line, with layer-2 transformations applied only in the `[Layers: 1,2]` target.

## Evaluation

Evaluation set: the 40 real spontaneous Parakeet v2 clips from Phase 1, hand-cleaned at both layer settings,
never trained on. Supplementary: [`Voice-Notes-Screened`](https://huggingface.co/datasets/danielrosehill/Voice-Notes-Screened) (real raw dictation, not Parakeet) for robustness, and
[`context/transcript-raw.txt`](../context/transcript-raw.txt) as a single long-form stress test.

Metrics, all computed per taxonomy id where the pair labels allow:

| Metric | Definition | Target (POC) |
|---|---|---|
| Edit precision / recall / F1 | Align raw to reference and raw to output at word level; compare the sets of deleted or changed raw tokens. Standard disfluency-detection scoring, extended to substitutions. | F1 above S1-mini (B2) and the zero-shot base (B3) |
| No-op change rate | On inputs whose reference equals the input, share of words changed | under 1% |
| Insertion rate | Content words in the output that are in neither the input nor the contraction-expansion list | 0 in 40 clips |
| WER vs reference | Word error rate of output against hand-cleaned reference | report only; secondary |
| Paragraph boundary F1 | Boundaries match if within one sentence of a reference boundary; also Pk | report; compared with SaT |
| Pairwise preference | Speaker blind-judges output pairs (model vs baselines, L1 vs L1+2) on 20 clips | model preferred over or tied with B4 in at least half |
| Latency | Wall-clock per 200-word input on the target device | decide after Whisperian constraints are known |

Baselines (see [`development-plan.md`](development-plan.md#phase-2-baselines-no-training)): identity, regex
only, S1-mini, base model zero-shot, frontier LLM with system prompt, and SaT for paragraphs only.

The insertion rate and no-op change rate are the light-touch guardrails; a model that scores well on edit F1
but fails either is rejected.

## Out of scope

- Homophones, misrecognised vocabulary, code-switched Hebrew words: the ASR fine-tunes.
- Capitalisation after a stop-and-restart across separate recordings: the model never sees the previous recording.
- Punctuation and casing restoration from scratch: Parakeet already does it.
- Spoken formatting commands and number formatting: regex and ASR.
- Headings, reordering, summarising, formality, format conversion: the LLM pass.
- Languages other than English; Hebrew-script output.
