"""Training-pair utilities. Standard library only.

  python scripts/pairs.py validate data/smoke/taxonomy-examples.jsonl
  python scripts/pairs.py smoke > data/smoke/taxonomy-examples.jsonl

validate checks a JSONL file against the pair format in docs/spec.md and the ids in
taxonomy/transformations.json. smoke rebuilds the smoke-test set from the taxonomy's
own examples: enough to prove the training script runs end to end, nothing more.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TAXONOMY = json.loads((ROOT / "taxonomy" / "transformations.json").read_text())
POSTPROCESSOR_IDS = {t["id"]: t for t in TAXONOMY["transformations"] if t["owner"] == "postprocessor"}

CONTROLS = {"[Layers: 1]", "[Layers: 1,2]"}
SOURCES = {"synthetic", "real-parakeet", "read-aloud-noop", "taxonomy-example"}
USE_CASES = {"message", "email", "note", "long"}
REQUIRED = ("id", "source", "control", "input", "output", "applied")


def validate(path: str) -> int:
    errors, seen = [], set()
    for n, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        where = f"{path}:{n} ({row.get('id')})"
        for f in REQUIRED:
            if f not in row:
                errors.append(f"{where}: missing {f}")
        if row.get("id") in seen:
            errors.append(f"{where}: duplicate id")
        seen.add(row.get("id"))
        if row.get("control") not in CONTROLS:
            errors.append(f"{where}: control {row.get('control')!r} not in {sorted(CONTROLS)}")
        if row.get("source") not in SOURCES:
            errors.append(f"{where}: source {row.get('source')!r} not in {sorted(SOURCES)}")
        if "use_case" in row and row["use_case"] not in USE_CASES:
            errors.append(f"{where}: use_case {row['use_case']!r} not in {sorted(USE_CASES)}")
        for tid in row.get("applied", []):
            if tid not in POSTPROCESSOR_IDS:
                errors.append(f"{where}: applied id {tid!r} is not a postprocessor-owned taxonomy id")
            elif row.get("control") == "[Layers: 1]" and POSTPROCESSOR_IDS[tid]["layer"] == 2:
                errors.append(f"{where}: layer-2 id {tid} applied under [Layers: 1]")
    for e in errors:
        print(e, file=sys.stderr)
    print(f"{len(seen)} pairs, {len(errors)} errors", file=sys.stderr)
    return 1 if errors else 0


def smoke() -> int:
    for tid, t in POSTPROCESSOR_IDS.items():
        ex = t["example"]
        control = "[Layers: 1]" if t["layer"] == 1 else "[Layers: 1,2]"
        row = {
            "id": f"tax-{tid}",
            "source": "taxonomy-example",
            "use_case": "note",
            "control": control,
            "input": ex["raw"],
            "output": ex["clean"],
            "applied": [tid],
            "asr_model": None,
            "words_in": len(ex["raw"].split()),
        }
        print(json.dumps(row, ensure_ascii=False))
        if t["layer"] == 1:  # layer 1 applies under both control lines
            print(json.dumps(row | {"id": f"tax-{tid}-l12", "control": "[Layers: 1,2]"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "smoke":
        sys.exit(smoke())
    if len(sys.argv) == 3 and sys.argv[1] == "validate":
        sys.exit(validate(sys.argv[2]))
    print(__doc__, file=sys.stderr)
    sys.exit(2)
