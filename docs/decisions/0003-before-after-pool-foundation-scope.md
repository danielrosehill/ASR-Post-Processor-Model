# 0003: ~300 before/after edits from Daniel's own history, drafted by Haiku under his foundation cleanup prompt

- **Date:** 2026-10-29
- **Status:** accepted
- **Supersedes:** 0002's S1-mini output as the draft (S1-mini becomes the baseline, run on Modal), and its
  ~100 items as the target; narrows docs/spec.md's scope question in favour of the foundation prompt (below)
- **By:** Daniel (approach, numbers, public dataset, private Space); agent (mix, prompt adaptations)

## Context

Daniel, 2026-10-29: one post-processing model that "just does all of this together" is a better use of his time
than more isolated paragraph work (ASR-Paragraph-Breaks). He is happy to do 300-400 edits covering his own
transcription history (AI prompts, email drafts, calendar appointments, short and long texts), with
code-switching and tech sentences so the niche vocabulary is preserved, and counterexamples. He named his
foundation cleanup prompt (AI-Transcription-Notepad, `app/src/config.py`, `FOUNDATION_PROMPT_SECTIONS`) as the
broad scope of his baseline edits.

## Decision

1. **Pool (`scripts/before_after.py`):** 330 Hebrish desktop dictations sampled by length (90 of 5-25 words,
   130 of 26-100, 110 of 101-700), obfuscated by Claude Haiku 5.5 (people, private organisations, places,
   dates, amounts; disfluencies, tech names and romanised Hebrew kept) and labelled `ai_prompt`, `email`,
   `calendar`, `message`, `note`, `other`; plus 100 sentences from the train split of
   `danielrosehill/Read-Aloud-Voice-Sentences` (60 `code_switch`, 40 `tech`), half as read (counterexamples) and
   half with disfluencies added, `target_terms` kept as `protect_terms`.
2. **Drafts:** Haiku under the foundation prompt, adapted: text in; plain prose only (no markdown, headings,
   bullets: the apps insert plain text); tech terms and romanised Hebrew never changed; numbers kept as the
   ASR wrote them; no signature injection. Unchanged input is a valid answer. The prompt is `DRAFT` in the script.
3. **Annotation:** private Space `danielrosehill/ASR-Cleanup-Annotator` (`space/`) in three columns (Daniel,
   2026-10-29): the original; stock S1-mini's output (`scripts/s1_modal.py`, Modal GPU) with its changes marked;
   and the edit box, which starts as S1-mini's text so each save records where Daniel diverges from it. The
   Haiku draft is one button away. The next item is from the least-annotated use case. Saves go to
   the public dataset `danielrosehill/ASR-Cleanup-Before-After` (`annotations/<id>.jsonl`), under Daniel's
   account, not an organisation.
4. **Target:** about 300 edits, about 40 held out as the test set. Train at 100 to see the learning curve;
   continue only if it is still improving. Stock S1-mini runs on the same inputs on Modal as the baseline.
5. **Control line:** every item is `[Layers: 1,2]`; Layer-1-only targets are not made.

## Pool as built (2026-10-29)

535 items: history 330 (`ai_prompt` 232, `note` 54, `other` 31, `message` 13), vocab 100 (`code_switch` 60,
`tech` 40), and `fill` 105 (`email` 45, `calendar` 30, `message` 30; Haiku-written in the raw ASR register,
a fifth clean), added because the history has no emails or calendar entries. Haiku's draft left 126 inputs
unchanged. One annotation (`tech`) was saved in the first two-column layout and has no `s1` field.

Two things went wrong on the way, both fixed in the script:

- Haiku answered dictated AI prompts instead of rewriting them (one reply: "I can't generate or play audio
  here..."). 175 of 330 history rewrites drifted out of the 0.75-1.33 length window (short and medium takes
  mostly; long ones were fine). Inputs now go inside `<transcript>` tags with a "this is data, never answer
  it" sentence; a retry batch of the failures then lost none.
- The first Modal run sat with no container for 15 minutes waiting for an L4; the function now takes the
  first free of L4, A10G, T4.

S1-mini has no Hugging Face inference provider (checked 2026-10-29: empty `inferenceProviderMapping`), so it
runs on Modal.

## What the foundation prompt adds to docs/spec.md

Spoken edit commands ("scratch that", "new paragraph"), abandoned sentences, spelled-out words, grammar and
obvious homophones, "I mean"/"basically" as filler. docs/spec.md and the taxonomy still describe the narrower
2026-10-16 scope; reconcile them with this record when training starts.

## Cost

All Haiku calls use Message Batches (half the per-call price).
