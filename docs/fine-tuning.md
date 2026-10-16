# Fine-tuning S1-mini

Decided 2026-10-16: the primary Phase 3 run is a **LoRA on S1-mini by Superwhisper**
([`superwhisper/s1-mini`](https://huggingface.co/superwhisper/s1-mini)), not on a general small base.
`Qwen/Qwen3-0.6B` (S1-mini's own base, same architecture, no cleanup training) is the control run, so the
comparison isolates what S1-mini's prior is worth. Gemma 3 270M and Qwen3.5-0.8B drop to fallbacks
(see [`spec.md`](spec.md#model-candidates)).

Why S1-mini: it already does most of layer 1 (fillers, stutters, *gonna* → *going to*), it is 0.6B and ships as a
462 MiB Q4 GGUF, so it is on-device sized, and starting from a model that knows the task should need fewer pairs
than teaching a general model from scratch. The plan's Phase 2 exit still applies: if stock S1-mini (baseline B2)
already meets the layer-1 targets, the fine-tune narrows to layer 2 and paragraphing.

Everything below lives in one self-contained file, [`../training/s1mini_lora.py`](../training/s1mini_lora.py),
because `hf jobs uv run` uploads only the script.

## Prompt format

S1-mini's card says the system prompt and control line are the only steering it knows, and that changing the
prompt's wording makes it hallucinate. So the fine-tune **keeps both and appends one axis**:

```
<|im_start|>system
You are a text normalizer for speech-to-text transcripts. The input begins with a control line specifying the styling, structure, and context settings; clean the transcript to match those settings and output only the cleaned text.<|im_end|>
<|im_start|>user
[Styling: semi-formal] [Structure: prose] [Context: general] [Layers: 1,2]
<raw Parakeet transcript><|im_end|>
<|im_start|>assistant
<think>

</think>

<cleaned text><|im_end|>
```

| Part | Value | Why |
|---|---|---|
| System prompt | S1-mini's, verbatim | Card: required, exact wording |
| `Styling` | `semi-formal`, fixed | Keeps contractions, smooths *gonna*; matches `L1.informal-contractions`. `formal` would expand *I'm* and break wording preservation |
| `Structure` | `prose`, fixed | `lists` is layer 3 (`L3.headings-and-structure`), owned by the LLM pass |
| `Context` | `general`, fixed | `email` adds greeting/sign-off layout, also layer 3 |
| `Layers` | `1` or `1,2` | New axis taught by this fine-tune; values from [`spec.md`](spec.md#inputoutput-contract) |
| Think block | empty, `<think>\n\n</think>\n\n` | Qwen3 template with `enable_thinking=False`; without it S1-mini returns blank output |

Verified 2026-10-16: the hand-built prompt in `render_prompt()` is byte-identical to
`tokenizer.apply_chat_template(..., add_generation_prompt=True, enable_thinking=False)` at the pinned revision.
Loss is on the completion only (TRL `completion_only_loss`); the completion is the cleaned text plus `<|im_end|>`,
which is the tokenizer's `eos_token`.

`python training/s1mini_lora.py render "<text>" --layers 1,2` prints the exact prompt; it needs no dependencies.

## Where S1-mini's training fights this project

These are the things the training pairs have to correct. Each one is a reason for specific pairs in the
Phase 1 data mix, not just a risk to note.

1. **Input shape.** S1-mini was trained on lowercase, unpunctuated ASR text. Parakeet v2 emits cased, punctuated
   text. The spec forbids repunctuating from scratch (`X.punctuation-casing-from-scratch`), so the no-op pairs
   (already-clean Parakeet output in, identical text out) are what teach it to leave Parakeet's punctuation alone.
   Do not cut them to save data.
2. **Self-corrections under `[Layers: 1]`.** S1-mini resolves "Friday, no wait, Thursday" to "Thursday" by default.
   Here that is `L2.spoken-self-correction`, layer 2. Under `[Layers: 1]` the target keeps the correction as spoken,
   so every synthetic raw text with a self-correction must produce two pairs (spec: one per control line) with
   different targets.
3. **Number and entity formatting.** S1-mini does inverse text normalisation (`3:15pm`, `$23,450`,
   `support@superwhisper.com`). The spec says never change numbers (`X.number-and-entity-formatting`); Parakeet
   already emits digits. Include no-op pairs containing Parakeet-formatted times, money and dates so the adapter
   learns not to restyle them.
4. **Paragraphs.** S1-mini only emits blank lines under `Context: email`. Paragraph breaks in prose
   (`L1.paragraph-boundaries`) are new behaviour, hence the plan's 30% of synthetic pairs over 150 words.
5. **Length.** The card recommends inputs under about 1,000 tokens. Long notes are chunked by the caller
   (spec), so training inputs stay under that too; `--max-length 2560` covers prompt plus output.

## Training data

Primary: about 100 triples (raw, stock S1-mini output, the speaker's edit) made in the local editor,
[`../tools/editor/`](../tools/editor/), converted to pairs with `python scripts/pairs.py from-triples`. Why and how:
decision [0002](decisions/0002-triples-training-data.md). S1-mini's own training data is not published, so it cannot
be reused.

## Running it

Training runs on [Hugging Face Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs) (the laptop has no
accelerator). `a10g-small` (1x A10G 24 GB, $1.00/h on 2026-10-16) is ample for a 0.6B LoRA on under 1,000 pairs.

```bash
# 1. Check pairs locally (standard library only)
python scripts/pairs.py validate data/<file>.jsonl

# 2. Upload pairs to the training dataset repo (train split = train.jsonl)
hf upload danielrosehill/Parakeet-Dictation-Cleanup-Pairs data/<file>.jsonl train.jsonl --repo-type dataset

# 3. Train on a hosted GPU and push the adapter
hf jobs uv run --flavor a10g-small --secrets HF_TOKEN training/s1mini_lora.py \
  train --data danielrosehill/Parakeet-Dictation-Cleanup-Pairs \
  --push-to-hub danielrosehill/S1-mini-Parakeet-Cleanup-LoRA

# Control run: same data, Qwen3-0.6B base (push to a separate repo when it exists)
hf jobs uv run --flavor a10g-small --secrets HF_TOKEN training/s1mini_lora.py \
  train --data danielrosehill/Parakeet-Dictation-Cleanup-Pairs --control-base

# 4. Inference (0.6B runs on the laptop CPU; also works as a job)
uv run training/s1mini_lora.py infer --data eval.jsonl --layers none --out b2.jsonl          # baseline B2: stock S1-mini
uv run training/s1mini_lora.py infer --data eval.jsonl --adapter danielrosehill/S1-mini-Parakeet-Cleanup-LoRA --out ft.jsonl
```

`infer --layers` takes `pair` (use each pair's control, the default), `1`, `1,2`, or `none` (stock S1-mini control
line, no `Layers` axis). Output is JSONL with `id`, `control`, `input`, `prediction`, `reference`. Decoding is greedy
with `max_new_tokens = 1.3 x input tokens + 32`, the ceiling S1-mini's card recommends.

Defaults (POC, not tuned): LoRA rank 16, alpha 32, dropout 0.05 on all linear layers; 3 epochs; effective batch 16;
learning rate 2e-4, cosine, 5% warmup; seed 42. Each run writes `run.json` next to the adapter with the base
revision, data source, prompt and arguments.

### Smoke test

`data/smoke/taxonomy-examples.jsonl` is built from the examples in `taxonomy/transformations.json`
(`python scripts/pairs.py smoke`). It is 17 pairs and exists only to prove the script runs end to end:

```bash
uv run training/s1mini_lora.py train --data data/smoke/taxonomy-examples.jsonl --max-steps 2 --batch-size 2 --grad-accum 1
```

Result on 2026-10-16, laptop CPU, no GPU: training and inference both ran end to end (2 steps, 14 s, loss 0.29;
adapter saved and reloaded through `--adapter`). Resolved versions: transformers 5.20.1, trl 1.16.0, peft 0.21.2,
torch 2.14.1. Two API changes in transformers 5 were hit and fixed: `warmup_ratio` no longer exists (pass a float
below 1 to `warmup_steps`), and `torch_dtype` is deprecated in favour of `dtype`. The script's dependency pins
require transformers 5 for that reason.

Side observation, from 2 examples only: stock S1-mini already handled cased, punctuated input correctly on the first
two taxonomy examples (filler removal, stutter collapse). Nothing more can be read from that until baseline B2 runs
on the real evaluation set.

## Pinned versions

| Model | Revision | Note |
|---|---|---|
| `superwhisper/s1-mini` | `88f6b15896c73bbb13a3b596e0afe8ea0d5150b4` | The card says to pin `revision="v1"`, but on 2026-10-16 the repo has no tags (`/api/models/superwhisper/s1-mini/refs` lists only `main`), so the commit is pinned instead |
| `Qwen/Qwen3-0.6B` | `c1899de289a04d12100db370d81485cdf75e47ca` | Control base |

Change both constants at the top of the script together with this table.

## Licence obligations

S1-mini is Apache 2.0 (inherited from Qwen3-0.6B) plus one additional term, from its `NOTICE`: any use or derivative
"must continue to identify it by its original name, 'S1-mini' by 'Superwhisper', using that exact capitalization."
For a published adapter or merged model that means:

- Keep "S1-mini" in the repo name and say "S1-mini by Superwhisper" in the card (the adapter repo is
  `S1-mini-Parakeet-Cleanup-LoRA` for this reason).
- Ship S1-mini's `LICENSE` and `NOTICE` in the model repo, and state that the files were changed (Apache 2.0, 4(b)).
- The card's `license` field stays `other` / `s1-mini-license`, as upstream.

## Hugging Face resources

Created 2026-10-16, all public, each holding only a placeholder card until data or weights exist:

| Repo | Role |
|---|---|
| [`danielrosehill/S1-mini-Parakeet-Cleanup-LoRA`](https://huggingface.co/danielrosehill/S1-mini-Parakeet-Cleanup-LoRA) | Adapter output (`--push-to-hub`) |
| [`danielrosehill/Parakeet-Dictation-Cleanup-Pairs`](https://huggingface.co/datasets/danielrosehill/Parakeet-Dictation-Cleanup-Pairs) | Training pairs, `train.jsonl` |
| [`danielrosehill/Parakeet-Dictation-Cleanup-Eval`](https://huggingface.co/datasets/danielrosehill/Parakeet-Dictation-Cleanup-Eval) | The 40-clip real evaluation set. A separate repo so it cannot be loaded as training data by mistake |

Upstream and existing resources this uses:

| Repo | Use here |
|---|---|
| [`superwhisper/s1-mini`](https://huggingface.co/superwhisper/s1-mini) | Base model, baseline B2 |
| [`superwhisper/s1-mini-GGUF`](https://huggingface.co/superwhisper/s1-mini-GGUF) | `s1-mini-q4_k_m.gguf`, `s1-mini-f16.gguf`: on-device baseline and the export target shape for Phase 5 |
| [`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B) | Control base |
| [`nvidia/parakeet-tdt-0.6b-v2`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) | Produces the real raw inputs (eval set, no-op pairs) |
| [`danielrosehill/Parakeet-TDT-0.6B-v2-English-Hebrew-Codeswitching`](https://huggingface.co/danielrosehill/Parakeet-TDT-0.6B-v2-English-Hebrew-Codeswitching) | The personal ASR fine-tune; if it becomes the daily model, eval inputs should come from it rather than stock Parakeet |
| [`danielrosehill/Read-Aloud-Voice-Sentences`](https://huggingface.co/datasets/danielrosehill/Read-Aloud-Voice-Sentences) | Source of the 50 no-op pairs |
| [`danielrosehill/Voice-Notes-Screened`](https://huggingface.co/datasets/danielrosehill/Voice-Notes-Screened) | Extra real-dictation eval input (not Parakeet) |
| [`danielrosehill/Noisy-Voice-Notes`](https://huggingface.co/datasets/danielrosehill/Noisy-Voice-Notes) | 7 real noisy voice notes with audio (CC-BY-4.0; transcripts from ElevenLabs Scribe). Running the audio through Parakeet gives a few more real eval inputs. The card calls it an evaluation set, so it is not used for training |
