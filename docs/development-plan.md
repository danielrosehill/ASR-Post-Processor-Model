# Development plan

Written 2026-10-16 from the voice note in [`../context/transcript-raw.txt`](../context/transcript-raw.txt).
Status: **planning**. Nothing has been trained and no data has been generated yet.

## The goal in one paragraph

Build a small, light-touch, text-in/text-out model that turns raw Parakeet v2 dictation into
text that can be sent as-is. It does foundational cleanup (disfluencies, informal contractions,
paragraph breaks) and, in a second variant, light editorial cleanup (repetition, speech aimed at
someone else in the room). It does not restyle or restructure. The use cases run from a
one-line WhatsApp message to a long report, so the model has to leave text alone more often than
it changes it.

## Where this sits in the stack

```
audio
  -> Parakeet v2 (stock, or the personal / code-switching fine-tunes)  # words: vocab, homophones, Hebrew
  -> regex pre-pass (ASR-Regex-Post-Processing)                       # deterministic: fillers, spoken commands
  -> THIS MODEL                                                       # layer 1 (+ layer 2 variant)
  -> optional system-prompted LLM pass                                # layer 3: format, style, structure
```

The rule that decides ownership, applied in [`../taxonomy/transformations.json`](../taxonomy/transformations.json):

1. **Needs the audio to decide?** ASR fine-tune (homophones, vocabulary, code-switched words).
2. **Expressible as a pattern with near-zero false positives?** Regex pre-pass.
3. **Needs context but preserves meaning and wording?** This model.
4. **Changes structure, order, tone or format, or depends on the destination?** LLM pass.

The project tests that boundary empirically rather than assuming it: the baselines in Phase 2 include a
system-prompted LLM doing the same job, so the comparison says whether a fine-tune is worth it at all.

## Recommended training-data approach

The voice note sets out three options and leaves the choice to Claude. Recommendation:
**Approach A (fully synthetic pairs), calibrated against a small Approach C set, with the C
recordings spoken freely instead of read aloud.** Approach B is dropped.

Why:

- **Reading aloud does not produce the targets.** Every layer-1 and layer-2 transformation is a
  feature of spontaneous speech: fillers, false starts, restarts, repetition, a door knock. Read
  sentences (B, or C as first described) come out fluent, so Parakeet transcribes them nearly
  clean and the pairs teach almost nothing except "leave it alone". That is still useful, but it
  needs only a few dozen examples, not a recording session.
- **Synthetic pairs give exact labels and exact control.** Generate the *clean* text first, then
  corrupt it with tagged transformations from the taxonomy. The target is known by construction,
  each pair carries the list of taxonomy ids applied, and the mix can be balanced by layer, length
  and use case. That makes the layer-1 vs layer-1+2 variants and the A/B tests cheap: the same raw
  text gets two targets.
- **But synthetic raw text drifts from what Parakeet really emits.** LLM-written "raw ASR" tends to
  be too messy in the wrong ways and too clean in the right ones. A small real Parakeet v2 set fixes
  that: it is the style reference for the corruption prompt and the evaluation set. This is the
  part of approach C worth keeping, and it matches the preference for one ASR tool across devices.
- **Why not B.** Spreading recordings across Whisper, Mistral and others trains for error profiles
  the speaker never sees, and each model formats differently (Whisper hallucinations, punctuation
  conventions), which adds noise to a 100-example set. It would make sense for a public,
  model-agnostic release; that is a later decision, not the POC's.
- **Why not the voice-note archive.** As the voice note says: PII redaction, and transcripts from
  older, worse models. The screened public subset ([`Voice-Notes-Screened`](https://huggingface.co/datasets/danielrosehill/Voice-Notes-Screened), 65 rows,
  raw ASR with fillers kept on purpose) is still useful as extra *evaluation input*, because it is real
  spontaneous dictation, but it is not Parakeet output.

Free material that already exists:

- [`danielrosehill/Read-Aloud-Voice-Sentences`](https://huggingface.co/datasets/danielrosehill/Read-Aloud-Voice-Sentences) (1,834 rows on
  2026-10-16, phone audio, `text` already "punctuated and cased like Parakeet v2", categories include
  `email`, `message`, `long`). Running its audio through stock Parakeet v2 gives real Parakeet output with
  a clean reference. Use it for **no-op pairs** (input already clean, output identical) and to measure
  Parakeet's punctuation habits. It cannot teach disfluency removal because the speech is read.
- [`danielrosehill/Transcription-Cleanup-Trainer`](https://huggingface.co/datasets/danielrosehill/Transcription-Cleanup-Trainer):
  one completed sample (a 6-minute spontaneous answer, Whisper raw / Gemini 2.5 Flash cleanup / hand cleanup).
  See [`research.md`](research.md#danielrosehilltranscription-cleanup-trainer). Its hand cleanup is the
  best existing statement of the target style, especially paragraphing, but it goes past layer 1 (it
  rewrites openings and adds parenthetical asides), so it is a style reference, not a training pair.

## Phases

Each phase has an exit test. Stop or re-plan at any exit that fails.

### Phase 0: lock the spec (no compute)

- Review [`spec.md`](spec.md) and the taxonomy. Decide the three experimental layer-2 items
  (`L2.non-directed-speech`, `L2.train-of-thought-recovery`, `L2.verbal-hedges`): in the POC or not.
- Write the target-style guide: five to ten hand-cleaned real examples, one per use case (WhatsApp
  message, email, note to self, long report section). Paragraph length preference is set here.
- Exit: the speaker agrees each example is "what I'd send".

### Phase 1: data (POC scale)

1. **Real set, 40 clips.** Record 40 spontaneous dictations into Parakeet v2 (prompted, like the
   Transcription-Cleanup-Trainer questions, not read). Mix: 15 short messages (under 30 s), 15 emails
   or notes (30 to 90 s), 10 long (2 min+, for paragraphs). Stage at least 5 with a deliberate interruption
   (someone at the door) for `L2.non-directed-speech`. Hand-clean each to the style guide, at layer 1 and
   at layer 1+2. These 40 are **evaluation only**, never trained on.
   The scenario cards are in [`../recording/scenarios.json`](../recording/scenarios.json) (written 2026-10-16):
   fictional cast and details so no real PII enters the set, a delivery cue per card that stages the
   targeted disfluency, and the taxonomy ids each card is meant to exercise. Record them on the phone with
   the [recording app](../app/README.md) (https://danielrosehill.github.io/ASR-Post-Processor-Model/app/),
   which keeps takes on the device and exports a zip with a `manifest.json`.
2. **No-op set, 50 pairs.** From Read-Aloud-Voice-Sentences, `email`/`message`/`long` rows transcribed
   by stock Parakeet v2 where the output already matches the reference.
3. **Synthetic set, 400 pairs.** Clean-first generation: an LLM writes clean target text across the same
   use-case mix; a second step injects transformations from the taxonomy, conditioned on 10 real raw
   Parakeet transcripts from step 1 as style examples. Each pair records `applied: [taxonomy ids]`.
   Length mix weighted towards long texts (at least 30% over 150 words) so paragraph boundaries appear.
   Second family LLM reviews a 10% sample for realism.
- Exit: on a blind sample of 20 synthetic raw texts mixed with 20 real ones, the speaker cannot reliably
  tell them apart, and every taxonomy id in scope appears at least 20 times.

### Phase 2: baselines (no training)

Run all of these on the 40-clip real eval set, scored with the metrics in [`spec.md`](spec.md#evaluation):

| # | System | Why |
|---|---|---|
| B0 | Identity (raw = output) | Floor; shows how much there is to fix |
| B1 | Regex pre-pass only | What the deterministic layer already achieves |
| B2 | [`superwhisper/s1-mini`](https://huggingface.co/superwhisper/s1-mini) at `semi-formal` / `prose` | The existing tool for exactly this job (the "S1 mini" in the voice note) |
| B3 | Base small model, zero-shot, system prompt | What the fine-tune adds over prompting the same weights |
| B4 | Frontier LLM with a careful system prompt | Ceiling; the "system-prompted LLM pass" alternative |
| B5 | SaT (`segment-any-text/sat-12l-sm`) paragraphs only | Whether paragraphing needs the generative model at all |

- Exit: if S1-mini (B2), optionally followed by SaT (B5), already meets the targets on layer 1, do not train a
  layer-1 model; the project narrows to layer 2 (repetition, non-directed speech), which S1-mini does not claim
  to cover. If B3 matches B4, ship a prompt instead of a fine-tune. If B4 itself fails a category (likely
  `L2.non-directed-speech`), that category is too hard from text alone and moves to `llm-pass` or out of scope.

### Phase 3: POC fine-tune

- Base model: Gemma 3 270M as a cheap probe, then Qwen3.5-0.8B (see [`spec.md`](spec.md#model-candidates)),
  LoRA, 400 synthetic + 50 no-op pairs. One model, two control lines (`[Layers: 1]`, `[Layers: 1,2]`), the
  pattern S1-mini uses. Separately trained L1 and L1+L2 variants are the fallback.
- One extra run: LoRA on top of S1-mini itself, to see whether starting from a cleanup model needs fewer pairs.
- Train on a rented or hosted GPU; the laptop has no ML accelerator.
- Exit: the `[Layers: 1]` output beats B2 and B3 on edit F1 and does not exceed 1% word changes on no-op inputs. If the
  smallest base fails, step up one size once before re-planning.

### Phase 4: A/B and boundary decisions

- Blind pairwise comparison on the real set: `[Layers: 1]` vs `[Layers: 1,2]` vs S1-mini vs B4. For each layer-2 item, does adding it
  improve preference or go too far?
- Decide: does paragraphing work inside the model, or does it need its own segmenter (see
  [`research.md`](research.md#paragraph-and-text-segmentation))? Fewest moving parts wins a tie.
- Record the outcome per taxonomy id: keep, move owner, or drop. Update `taxonomy/transformations.json`.

### Phase 5: deploy (only after Phase 4 passes)

- Export (GGUF or ONNX) and test as a post-processor in daily dictation. Confirm what Whisperian's
  post-processing step accepts before choosing the format (open question below).
- Scale data only if daily use shows a gap the POC set did not cover.

## Open questions

- What model formats and sizes does Whisperian's LLM post-processing step accept, and does it pass any
  preceding text as context? (Decides deployment, and whether `X.cross-utterance-restart` is ever solvable.)
- Does one model handle both a 10-word message and a 1,500-word note, or does long text need chunking with
  overlap? The model's context window and latency on the phone decide this.
- Public release shape: personal model only, or a model-agnostic variant (which would bring approach B back).
