# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.8",
#   "transformers>=5.20,<6",
#   "peft>=0.21,<1",
#   "trl>=1.16,<2",
#   "datasets>=3.0",
#   "accelerate>=1.0",
# ]
# ///
"""LoRA fine-tune of S1-mini by Superwhisper for light-touch Parakeet v2 cleanup.

Self-contained on purpose: `hf jobs uv run` uploads only this file, so the prompt
format lives here and nowhere else. Subcommands:

  render  print the exact prompt string for one input (check the format by eye)
  train   LoRA on JSONL pairs in the docs/spec.md format; optional push to the Hub
  infer   greedy decoding over JSONL inputs, base or base+adapter (also runs baseline B2)

See docs/fine-tuning.md for why the format is what it is and how to run each step.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# --- Base model -------------------------------------------------------------------
# superwhisper/s1-mini has no git tags on 2026-10-16 (its card says revision="v1",
# but /api/models/superwhisper/s1-mini/refs lists only main), so pin the commit.
BASE_MODEL = "superwhisper/s1-mini"
BASE_REVISION = "88f6b15896c73bbb13a3b596e0afe8ea0d5150b4"
# Control run: same architecture, no cleanup training (docs/development-plan.md, Phase 3).
CONTROL_MODEL = "Qwen/Qwen3-0.6B"
CONTROL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"

# --- Prompt format ----------------------------------------------------------------
# S1-mini's system prompt, verbatim from its model card. The card warns that changing
# the wording degrades output, so it is kept and the new axis is appended instead.
SYSTEM_PROMPT = (
    "You are a text normalizer for speech-to-text transcripts. The input begins "
    "with a control line specifying the styling, structure, and context settings; "
    "clean the transcript to match those settings and output only the cleaned text."
)
# S1-mini's own axes, pinned to the register this project targets: semi-formal keeps
# contractions and turns gonna into going to (= L1.informal-contractions), prose and
# general stop it from making lists or email layout (both L3, owned by the llm-pass).
S1_CONTROL = "[Styling: semi-formal] [Structure: prose] [Context: general]"
# The new axis this fine-tune teaches. Values match docs/spec.md.
LAYER_CONTROLS = {"1": "[Layers: 1]", "1,2": "[Layers: 1,2]"}
# Qwen3 template with enable_thinking=False ends the prompt with this empty think
# block; S1-mini was trained on it and returns blank output without it.
ASSISTANT_PREFIX = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
END_OF_TURN = "<|im_end|>"


def control_line(layers: str | None) -> str:
    """layers=None gives stock S1-mini's control line (baseline B2)."""
    if layers is None:
        return S1_CONTROL
    return f"{S1_CONTROL} {LAYER_CONTROLS[layers]}"


def layers_from_control(control: str) -> str:
    for key, value in LAYER_CONTROLS.items():
        if control == value:
            return key
    raise ValueError(f"unknown control {control!r}; expected one of {list(LAYER_CONTROLS.values())}")


def render_prompt(transcript: str, layers: str | None) -> str:
    """The full prompt up to and including the assistant prefix, built by hand so it
    matches the literal string on the S1-mini card without loading a tokenizer."""
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}{END_OF_TURN}\n"
        f"<|im_start|>user\n{control_line(layers)}\n{transcript}{END_OF_TURN}\n"
        f"{ASSISTANT_PREFIX}"
    )


# --- Data -------------------------------------------------------------------------
REQUIRED_FIELDS = ("id", "control", "input", "output")


def load_pairs(spec: str, split: str = "train") -> list[dict]:
    """spec is a local .jsonl path or an HF dataset repo id (optionally repo:split)."""
    path = Path(spec)
    if path.suffix == ".jsonl":
        if not path.exists():
            raise FileNotFoundError(path)
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    else:
        from datasets import load_dataset

        repo, _, repo_split = spec.partition(":")
        rows = list(load_dataset(repo, split=repo_split or split))
    for row in rows:
        missing = [f for f in REQUIRED_FIELDS if f not in row]
        if missing:
            raise ValueError(f"pair {row.get('id')!r} missing {missing}")
        layers_from_control(row["control"])
    return rows


# --- Subcommands ------------------------------------------------------------------
def cmd_render(args: argparse.Namespace) -> None:
    sys.stdout.write(render_prompt(args.text, None if args.layers == "none" else args.layers))
    sys.stdout.write("\n")


def cmd_train(args: argparse.Namespace) -> None:
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    base, revision = (CONTROL_MODEL, CONTROL_REVISION) if args.control_base else (BASE_MODEL, BASE_REVISION)
    rows = load_pairs(args.data)
    # Prompt/completion strings: TRL masks the prompt, so loss is on the cleaned text only.
    ds = Dataset.from_list(
        [
            {
                "prompt": render_prompt(r["input"], layers_from_control(r["control"])),
                "completion": r["output"] + END_OF_TURN,
            }
            for r in rows
        ]
    )
    print(f"{len(ds)} pairs from {args.data}; base {base}@{revision[:8]}", flush=True)

    tok = AutoTokenizer.from_pretrained(base, revision=revision)
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    model = AutoModelForCausalLM.from_pretrained(
        base, revision=revision, dtype=torch.bfloat16 if use_bf16 else torch.float32
    )

    peft_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank * 2,
        lora_dropout=0.05,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    cfg = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=0.05,  # transformers 5: a float < 1 is a ratio (warmup_ratio was removed)
        max_length=args.max_length,
        completion_only_loss=True,
        bf16=use_bf16,
        logging_steps=5,
        save_strategy="no",
        report_to="none",
        seed=args.seed,
        push_to_hub=bool(args.push_to_hub),
        hub_model_id=args.push_to_hub,
        hub_private_repo=False,
    )
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok, peft_config=peft_config)
    trainer.train()
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)
    (Path(args.output_dir) / "run.json").write_text(
        json.dumps(
            {
                "base_model": base,
                "base_revision": revision,
                "data": args.data,
                "pairs": len(ds),
                "system_prompt": SYSTEM_PROMPT,
                "control_lines": {k: control_line(k) for k in LAYER_CONTROLS},
                "args": vars(args) | {"func": None},
            },
            indent=2,
        )
    )
    if args.push_to_hub:
        trainer.push_to_hub()
        print(f"pushed adapter to https://huggingface.co/{args.push_to_hub}", flush=True)


def cmd_infer(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base, revision = (CONTROL_MODEL, CONTROL_REVISION) if args.control_base else (BASE_MODEL, BASE_REVISION)
    tok = AutoTokenizer.from_pretrained(base, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(base, revision=revision, dtype="auto")
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    rows = load_pairs(args.data)
    out = open(args.out, "w") if args.out else sys.stdout
    for r in rows:
        # --layers overrides the pair's control; "none" = stock S1-mini control line (B2).
        layers = layers_from_control(r["control"]) if args.layers == "pair" else None if args.layers == "none" else args.layers
        enc = tok(render_prompt(r["input"], layers), return_tensors="pt", add_special_tokens=False).to(device)
        n_in = enc.input_ids.shape[1]
        # Card guidance: output tracks input length; 1.3 x input + 32 is a safe ceiling.
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=int(1.3 * n_in) + 32, do_sample=False)
        pred = tok.decode(gen[0][n_in:], skip_special_tokens=True).strip()
        out.write(json.dumps({"id": r["id"], "control": control_line(layers), "input": r["input"], "prediction": pred, "reference": r["output"]}) + "\n")
        out.flush()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(required=True)

    r = sub.add_parser("render", help="print the exact prompt for one transcript")
    r.add_argument("text")
    r.add_argument("--layers", choices=["1", "1,2", "none"], default="1")
    r.set_defaults(func=cmd_render)

    t = sub.add_parser("train", help="LoRA fine-tune")
    t.add_argument("--data", required=True, help="local .jsonl or HF dataset repo id[:split]")
    t.add_argument("--output-dir", default="outputs/s1mini-lora")
    t.add_argument("--control-base", action="store_true", help=f"train on {CONTROL_MODEL} instead (control run)")
    t.add_argument("--rank", type=int, default=16)
    t.add_argument("--epochs", type=float, default=3)
    t.add_argument("--max-steps", type=int, default=-1, help="override epochs; 2 is a smoke test")
    t.add_argument("--batch-size", type=int, default=8)
    t.add_argument("--grad-accum", type=int, default=2)
    t.add_argument("--lr", type=float, default=2e-4)
    t.add_argument("--max-length", type=int, default=2560, help="prompt + completion tokens; spec chunks input at ~1,000")
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--push-to-hub", metavar="REPO_ID", help="e.g. danielrosehill/S1-mini-Parakeet-Cleanup-LoRA")
    t.set_defaults(func=cmd_train)

    i = sub.add_parser("infer", help="greedy decoding over JSONL inputs")
    i.add_argument("--data", required=True)
    i.add_argument("--adapter", help="local dir or HF repo id of a LoRA adapter; omit for the base model")
    i.add_argument("--control-base", action="store_true")
    i.add_argument("--layers", choices=["pair", "1", "1,2", "none"], default="pair")
    i.add_argument("--out", help="output .jsonl (default stdout)")
    i.set_defaults(func=cmd_infer)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
