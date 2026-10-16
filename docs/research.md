# Prior art

Verified 2026-10-16. **Confirmed** means the page, HF API record or GitHub API record was opened on that
date. **Inferred** means it comes from a search result, a backbone size or a name, and was not read directly.
Parameter counts are the HF `safetensors.total` field unless noted.

## Summary

- The "S1 mini" from the voice note is **Superwhisper S1-mini**, a Qwen3-0.6B fine-tune released 2026-08-12 for
  exactly this job. It covers fillers, self-corrections, register (*gonna*), number formatting, lists and email
  layout. It does not claim general paragraph breaking, content-repetition removal beyond stutters, or removal of
  speech aimed at other people. Those three are where this project can add something.
- Several other dictation-cleanup fine-tunes of 0.08B to 2B exist on HF, mostly for Mac apps, mostly without evaluation.
- Paragraph segmentation has a ready-made, MIT-licensed tool (SaT / wtpsplit) and a 2025 paper defining it as a task for speech.
- Removing non-directed speech from inside a dictation, from text alone, has no prior art found. The nearest
  literature (device-directed speech detection, addressee detection) classifies whole utterances, mostly with audio.

## S1-mini is Superwhisper S1-mini

| | |
|---|---|
| Model | [`superwhisper/s1-mini`](https://huggingface.co/superwhisper/s1-mini) (confirmed, card read); GGUF: [`superwhisper/s1-mini-GGUF`](https://huggingface.co/superwhisper/s1-mini-GGUF) |
| Blog | <https://superwhisper.com/blog/s1> (confirmed) |
| Created | 2026-08-12 (HF API) |
| Base | `Qwen/Qwen3-0.6B` (HF API `base_model`) |
| Size | 751.6M stored; the card says 596M unique (tied `lm_head` stored twice). Q4 GGUF about 462 MiB |
| Licence | Apache-2.0 plus a clause requiring the name "S1-mini by Superwhisper" (HF shows `other`) |
| Input limit | About 1,000 tokens per the card; chunk longer transcripts |

How it works: a fixed system prompt plus a control line,
`[Styling: casual|semi-casual|semi-formal|formal] [Structure: prose|lists] [Context: general|email]`. Run with
thinking disabled (`enable_thinking=False`) or it returns blank output. It removes fillers, resolves
self-corrections to the final value, punctuates and cases, and converts numbers, dates, currency and email
addresses. *gonna* becomes *going to* at `semi-formal`. Filler-only input returns an empty string.

Vendor-reported, not independently checked: 94.8% token accuracy on 7,519 held-out English cases; the blog adds
98.6% correct withholding on non-transcribable input and 97.6% correct list-vs-prose choice.

What this means here: S1-mini is baseline B2 in the plan, its control-line design is adopted in the spec
(`[Layers: 1]` / `[Layers: 1,2]`), and it is itself a candidate starting point for a LoRA.

Not it (confirmed by name): `simplescaling/s1` (test-time-scaling reasoning on Qwen2.5-32B), `fishaudio/s1-mini`
(text-to-speech), `internlm/Intern-S1-mini` (science multimodal).

### Other dictation-cleanup fine-tunes on HF

All confirmed via the HF API; none has a published evaluation the agent could find.

| Model | Base | Params | Licence | Note |
|---|---|---|---|---|
| [`meetboswell/boswell-dictation-cleanup`](https://huggingface.co/meetboswell/boswell-dictation-cleanup) | Qwen3-1.7B, 4-bit MLX | 1.72B | Apache-2.0 | For the Boswell Mac app; card says no evaluation |
| [`ReFyneLabs/simplewords-dictation-cleanup-v2`](https://huggingface.co/ReFyneLabs/simplewords-dictation-cleanup-v2) | Qwen3.5-2B, QLoRA, MLX | 1.88B | Apache-2.0 | |
| [`getonit/transcription-cleanup-llama3.2-1b`](https://huggingface.co/getonit/transcription-cleanup-llama3.2-1b) | Llama-3.2-1B, MLX | 1.24B | none stated | About 5k downloads, empty card |
| [`tsilahi/looma-dictation-slm`](https://huggingface.co/tsilahi/looma-dictation-slm) | flan-t5-small | 77M | Apache-2.0 | The seq2seq route in practice |

### Dictation apps with a cleanup pass

- **Wispr Flow**: a fine-tuned Llama cleanup pass served on Baseten, P99 under 700 ms. Source <https://baseten.co/customers/wispr-flow> (inferred: search summary, page not opened).
- **Handy** <https://github.com/cjpais/Handy> (MIT): optional post-processing toggle. Confirmed.
- **VoiceInk** <https://github.com/Beingpax/VoiceInk>: context-aware "AI enhancement" that reads screen content. Confirmed.
- **Whisperian** <https://whisperian.app>: the speaker's daily driver; supports small-LLM and regex post-processors (per [ASR-Regex-Post-Processing](https://github.com/danielrosehill/ASR-Regex-Post-Processing)). Which model formats it accepts is an open question in the plan.

## Punctuation and truecasing

Parakeet v2 already punctuates and cases, so these are baselines or label sources, not a stage in the pipeline.

| Model | What | Size | Licence | Status |
|---|---|---|---|---|
| [`oliverguhr/fullstop-punctuation-multilang-large`](https://huggingface.co/oliverguhr/fullstop-punctuation-multilang-large) | Punctuation tagging, XLM-R large | 559M | MIT | confirmed |
| [`1-800-BAD-CODE/punctuation_fullstop_truecase_english`](https://huggingface.co/1-800-BAD-CODE/punctuation_fullstop_truecase_english) | Punctuation, truecasing, sentence boundaries in one ONNX; package <https://github.com/1-800-BAD-CODE/punctuators> | about 50M (inferred from 209.5 MB ONNX) | Apache-2.0 | confirmed |
| [`1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase`](https://huggingface.co/1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase) | Multilingual version | about 280M (inferred) | Apache-2.0 | confirmed |
| [`felflare/bert-restore-punctuation`](https://huggingface.co/felflare/bert-restore-punctuation) | Punctuation and casing, bert-base-uncased | about 110M | MIT | confirmed |
| [`kredor/punctuate-all`](https://huggingface.co/kredor/punctuate-all) | Multilingual punctuation, xlm-roberta-base | about 278M (inferred) | MIT | confirmed |
| NeMo `punctuation_en_bert` / `punctuation_en_distilbert` ([NGC](https://api.ngc.nvidia.com/v2/models/nvidia/nemo/punctuation_en_bert)) | Comma, period, question mark and capitalisation per word | about 110M / 66M (inferred) | NGC terms | confirmed |

## Disfluency detection and removal

Datasets and generation:

- **Switchboard disfluency annotations** (Treebank-3, [LDC99T42](https://catalog.ldc.upenn.edu/LDC99T42)): the standard corpus; paid LDC licence, so not usable here. Confirmed.
- **Disfl-QA** ([arXiv 2106.04016](https://arxiv.org/abs/2106.04016), [GitHub](https://github.com/google-research-datasets/Disfl-QA)): human-written disfluent rewrites of SQuAD questions. Confirmed; licence not confirmed.
- **LARD: Large-scale Artificial Disfluency Generation** ([arXiv 2201.05041](https://arxiv.org/abs/2201.05041)): rules for synthesising repetitions, replacements and restarts. Confirmed. **Directly reusable** for the corruption step of synthetic pair generation (taxonomy `L1.stutter-repetition`, `L1.false-start-repair`, `L2.spoken-self-correction`).

Papers:

- Jamshid Lou and Johnson, "Improving Disfluency Detection by Self-Training a Self-Attentive Model" ([arXiv 2004.05323](https://arxiv.org/abs/2004.05323)). Confirmed. Source of the token-level edit F1 used in the spec.
- **DRES: Benchmarking LLMs for Disfluency Removal** ([arXiv 2509.20321v1](https://arxiv.org/abs/2509.20321v1)). Confirmed. Text-only benchmark on Switchboard. Findings that matter here: segmenting the input helps; reasoning models over-delete fluent words; fine-tuning reaches near state of the art but hurts generalisation. Later versions were retitled. The over-deletion finding is why the spec has a no-op change rate and insertion rate as hard guardrails.
- VocalBench-DF ([arXiv 2510.15406](https://arxiv.org/abs/2510.15406v1)): speech-LLM robustness to disfluent audio. Inferred (not opened); loosely relevant.

Small taggers on HF (confirmed via API): `stillerman/fdt-disfluency-*` (4M to 66M; the distilbert-66m-v5 is 65.2M, Apache-2.0, with ONNX and transformers.js builds), `arielcerdap/modernbert-base-multiclass-disfluency-v2` (150M, multiclass tags), `hafidev/bert-base-uncased-*-disfluency-detection` (separate taggers for filled pauses, restarts, discourse markers). Deletion-only and cheap; candidates for a "faithful deletion" baseline or pre-pass.

## ASR error correction with LLMs

These fix recognition errors (usually from N-best lists). That is closer to the ASR fine-tune's job than to this
model's, but they define the evaluation vocabulary for the field.

- **HyPoradise** (NeurIPS 2023, [arXiv 2309.15701](https://arxiv.org/abs/2309.15701)): N-best to transcript generative error correction benchmark. Confirmed.
- **Whispering LLaMA** ([arXiv 2310.06434](https://arxiv.org/abs/2310.06434)): fuses audio features into error correction. Confirmed.
- **GenSEC challenge** ([arXiv 2409.09785](https://arxiv.org/abs/2409.09785)): post-ASR correction, speaker tagging and emotion. Confirmed.
- HF models tagged "asr-correction" are almost all non-English T5/mT5 (for example `Den4ikAI/rut5_base_asr_error_correction`). Nothing for English dictation beyond the cleanup fine-tunes above.

## Paragraph and text segmentation

- **SaT / wtpsplit** <https://github.com/segment-any-text/wtpsplit> (MIT). Confirmed: **does paragraph segmentation**. The README says SaT models "are trained to predict newline probability, they can segment text into paragraphs in addition to sentences"; call `sat.split(text, do_paragraph_segmentation=True)` and tune `paragraph_threshold`. Paper [arXiv 2406.16678](https://arxiv.org/abs/2406.16678). Models [`segment-any-text/sat-3l-sm`](https://huggingface.co/segment-any-text/sat-3l-sm) (214M) and [`segment-any-text/sat-12l-sm`](https://huggingface.co/segment-any-text/sat-12l-sm) (277M, re-checked via API), MIT. Baseline B5; also a possible labeller for paragraph boundaries in synthetic data.
- **Retkowski and Waibel, "Paragraph Segmentation Revisited: Towards a Standard Task for Structuring Speech"** ([arXiv 2512.24517](https://arxiv.org/abs/2512.24517)). Confirmed. TEDPara and YTSegPara benchmarks, constrained LLM decoding that inserts breaks without changing the text, and a compact model, MiniSeg. The closest match to `L1.paragraph-boundaries`; its benchmarks are an external check on the paragraph metric.
- WtP, predecessor of SaT ([arXiv 2305.18893](https://arxiv.org/abs/2305.18893)). Confirmed.
- Zhang et al., self-adaptive sliding window for spoken document segmentation, ASRU 2021 ([arXiv 2107.09278](https://arxiv.org/abs/2107.09278)). Confirmed.
- Koshorek et al. 2018, "Text Segmentation as a Supervised Learning Task", Wiki-727k ([arXiv 1803.09337](https://arxiv.org/abs/1803.09337)). Confirmed.
- Lukasik et al. 2020, "Text Segmentation by Cross Segment Attention" ([arXiv 2004.14535](https://arxiv.org/abs/2004.14535)). Confirmed.
- Lai, Farrús and Moore 2020, lexical and prosodic features for paragraph segmentation ([UPF repository](https://repositori.upf.edu/items/785aa291-7f6c-4929-ac57-26e6f50abad3)). Inferred (search result).

The voice note calls paragraphs "the blind spot of post-processing". The literature agrees it is under-served, but
SaT means a separate paragraph model already exists off the shelf. The plan's Phase 4 decides between that and
learning paragraphs inside the cleanup model, with fewest moving parts as the tie-break.

## Non-directed speech

The literature terms are **device-directed speech detection** and **addressee detection**. No dictation-specific term was found.

- Mallidi et al., "Device-directed Utterance Detection" ([arXiv 1808.02504](https://arxiv.org/abs/1808.02504)). Confirmed (Amazon affiliation inferred).
- Wagner et al., "Multimodal Data and Resource Efficient Device-Directed Speech Detection with Large Foundation Models" ([arXiv 2312.03632](https://arxiv.org/abs/2312.03632)). Confirmed. Uses ASR 1-best text plus decoder signals with an LLM, so text carries part of the signal.
- "Device-Directed Speech Detection for Follow-up Conversations Using Large Language Models" ([arXiv 2411.00023](https://arxiv.org/abs/2411.00023)). Confirmed.
- Review of speech-based addressee detection, Frontiers in Computer Science 2022 ([PDF](https://www.frontiersin.org/articles/10.3389/fcomp.2022.831784/pdf)). Inferred (search result).

Gap: all of this classifies whole utterances, mostly with audio. Deleting "Come in. Thanks, Jim." from the middle of
an email dictation, from text alone, appears unstudied. That is why `L2.non-directed-speech` is marked experimental.

## Inverse text normalisation

- NeMo ITN (WFST grammars, [arXiv 2104.05055](https://arxiv.org/abs/2104.05055), code <https://github.com/NVIDIA/NeMo-text-processing>, Apache-2.0). Confirmed.
- Thutmose Tagger, single-pass neural ITN ([arXiv 2208.00064](https://arxiv.org/abs/2208.00064)). Confirmed.

S1-mini folds ITN into the model. Here it stays out (`X.number-and-entity-formatting`): Parakeet v2 already emits digits and the model is trained not to touch numbers.

## Small base models

See the table in [`spec.md`](spec.md#model-candidates). The wider list checked on 2026-10-16 via the HF API:

| Id | Params | Licence |
|---|---|---|
| `google/gemma-3-270m-it` | 268M | Gemma terms |
| `LiquidAI/LFM2-350M` | 354M | LFM Open |
| `HuggingFaceTB/SmolLM2-360M-Instruct` | 362M | Apache-2.0 |
| `Qwen/Qwen2.5-0.5B-Instruct` | 494M | Apache-2.0 |
| `Qwen/Qwen3-0.6B` | 752M stored / 596M unique | Apache-2.0 |
| `Qwen/Qwen3.5-0.8B` | 873M | Apache-2.0 |
| `google/gemma-3-1b-it` | 1.00B | Gemma terms |
| `LiquidAI/LFM2.5-1.2B-Instruct` | 1.17B | other |
| `meta-llama/Llama-3.2-1B-Instruct` | 1.24B | Llama 3.2 Community |
| `Qwen/Qwen2.5-1.5B-Instruct` | 1.54B | Apache-2.0 |
| `HuggingFaceTB/SmolLM2-1.7B-Instruct` | 1.71B | Apache-2.0 |
| `Qwen/Qwen3-1.7B` | 2.03B stored | Apache-2.0 |
| `Qwen/Qwen3.5-2B` | 2.27B | Apache-2.0 |
| `google/flan-t5-base` / `flan-t5-large` | 248M / 783M | Apache-2.0 |
| `google-t5/t5-small` | 61M | Apache-2.0 |

No Qwen small text model newer than Qwen3.5 (2026-02-28) appeared in a newest-first listing of the Qwen org. The smallest Gemma 4 is `google/gemma-4-E2B-it` (multimodal, 5.12B stored), too large for this budget.

## The speaker's own datasets

Checked 2026-10-16 via the HF API and dataset cards. All are public.

### danielrosehill/Transcription-Cleanup-Trainer

<https://huggingface.co/datasets/danielrosehill/Transcription-Cleanup-Trainer>, created 2025-12-18. Confirmed (files downloaded and read).

- **What it is:** the scaffold of a paired dataset for exactly this problem, aiming at a "Goldilocks" cleanup level. Ten open-ended spoken-answer prompts (`questions.json`, `questions/question-01..10.md`, topics: software workflow, work-life balance, explaining LLMs, ...), with a four-way layout per sample: `data/audio/`, `data/whisper-transcripts/` (OpenAI `whisper-1`), `data/auto-cleanup/` (Gemini 2.5 Flash via OpenRouter) and `data/manual-cleanups/` (hand-edited).
- **What it holds:** **one completed sample out of ten** (`dataset.json` says 10% complete): a 367-second answer to question 1, 911 words raw, 891 after Gemini, 859 after hand cleanup. The other nine are prompts only. The card still says "License: Private" in its body although the repo is public with no licence set.
- **Reusable as training data?** Not as pairs. One sample is not a dataset, the raw side is Whisper rather than Parakeet, and the hand cleanup goes beyond layer 1 (it rewrites the opening sentence, adds clarifying parentheticals such as "(to the AI tool)", and leaves typos like "woud" and "lof"). What is reusable:
  - the **hand cleanup as a style reference**, above all for paragraphing: it breaks into very short paragraphs, often one or two sentences, which confirms the "shorter paragraphs" preference in the voice note;
  - the **Gemini auto-cleanup vs hand cleanup diff** as a concrete picture of where a prompted LLM under-edits (it kept "Basically," and some "kind of", and used fewer, longer paragraphs) — a ready example for baseline B4;
  - the **ten question prompts** for the spontaneous Phase 1 recordings, which need prompted rather than read speech.

### Other datasets checked

| Dataset | Used here? | Why |
|---|---|---|
| [`Read-Aloud-Voice-Sentences`](https://huggingface.co/datasets/danielrosehill/Read-Aloud-Voice-Sentences) | Yes: no-op pairs | 1,834 rows of phone audio with references "punctuated and cased like Parakeet v2"; `email`, `message`, `long` categories. Read speech, so no disfluencies. |
| [`Voice-Notes-Screened`](https://huggingface.co/datasets/danielrosehill/Voice-Notes-Screened) | Yes: extra eval input | 65 privacy-screened real voice-note transcripts, raw ASR with fillers kept on purpose. Not Parakeet output. |
| [`Text-Transformation-Prompts-300525`](https://huggingface.co/datasets/danielrosehill/Text-Transformation-Prompts-300525) | Reference only | System prompts for the layer-3 LLM pass. Its `basic_text_fixes` prompt is a starting point for the B3/B4 baseline prompt. |
| [`Tech-Sentences`](https://huggingface.co/datasets/danielrosehill/Tech-Sentences), [`Tech-Sentences-For-ASR-Training`](https://huggingface.co/datasets/danielrosehill/Tech-Sentences-For-ASR-Training) | No | ASR vocabulary data; owned by the ASR fine-tune. |
| `Noisy-Voice-Notes` | No | 7 noisy clips with ElevenLabs Scribe transcripts; a noise/ASR probe, not a cleanup set. |
| `Podcast-ASR-Evaluation` | No | One podcast episode with ground truth and several ASR runs, for WER benchmarking (inferred from the file listing; card not read). |
