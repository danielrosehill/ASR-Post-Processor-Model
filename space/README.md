---
title: ASR Clean-up Annotator
emoji: ✍️
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
license: mit
hf_oauth: true
hf_oauth_scopes:
  - write-repos
short_description: Before/after edits of raw ASR text for a clean-up model
---

# ASR Clean-up Annotator

Private Space. Shows one raw transcript at a time from
[`danielrosehill/ASR-Cleanup-Before-After`](https://huggingface.co/datasets/danielrosehill/ASR-Cleanup-Before-After)
with Claude Haiku's draft clean-up; the annotator edits the draft to the wanted output. Source:
`space/` in the ASR-Post-Processor-Model repo. Saves use the signed-in user's OAuth token.
