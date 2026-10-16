# 0001: Fine-tune S1-mini rather than a general small base

- **Date:** 2026-10-16
- **Status:** accepted
- **Decided by:** the speaker, after the prior-art review in [`../research.md`](../research.md#s1-mini-is-superwhisper-s1-mini)

## Decision

The Phase 3 fine-tune is a LoRA on [`superwhisper/s1-mini`](https://huggingface.co/superwhisper/s1-mini), pinned
at commit `88f6b15896c73bbb13a3b596e0afe8ea0d5150b4`. The same LoRA on `Qwen/Qwen3-0.6B` (S1-mini's own base) is the
control run. Gemma 3 270M and Qwen3.5-0.8B, the earlier recommendation in [`../spec.md`](../spec.md#model-candidates),
become fallbacks.

S1-mini's system prompt and control line are kept verbatim, pinned to
`[Styling: semi-formal] [Structure: prose] [Context: general]`, and the fine-tune appends `[Layers: 1]` or
`[Layers: 1,2]`.

## Why

- S1-mini already does most of layer 1 (fillers, stutters, *gonna* → *going to*), so it should need fewer pairs.
- At 0.6B with a 462 MiB Q4 GGUF, it is on-device sized.
- Its card warns that rewording the system prompt degrades output, so the format extends it rather than replacing it.
- The Qwen3-0.6B control isolates what S1-mini's cleanup training is worth.

## Consequences

- S1-mini's own habits conflict with the spec in four places (input shape, self-corrections under `[Layers: 1]`,
  number formatting, prose paragraphs). The training data has to correct each one; see
  [`../fine-tuning.md`](../fine-tuning.md#where-s1-minis-training-fights-this-project).
- A published adapter inherits S1-mini's naming clause: it must keep "S1-mini" by "Superwhisper" in its name and
  card, and ship upstream `LICENSE` and `NOTICE`.
- Implementation: [`../../training/s1mini_lora.py`](../../training/s1mini_lora.py), smoke-tested on CPU 2026-10-16.
