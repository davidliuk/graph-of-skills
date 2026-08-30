from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path


LABEL_FIELDS = (
    "valid_relation",
    "type_correct",
    "direction_correct",
    "corrected_type",
    "notes",
)


def apply_decisions(
    annotations: Path,
    decisions_path: Path,
    output: Path,
) -> None:
    decisions = json.loads(decisions_path.read_text(encoding="utf-8"))
    with annotations.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    sample_ids = {str(row.get("sample_id", "")) for row in rows}
    if sample_ids != set(decisions):
        raise ValueError(
            f"Decision/sample mismatch: missing={sorted(sample_ids - set(decisions))}, "
            f"extra={sorted(set(decisions) - sample_ids)}"
        )
    for row in rows:
        decision = decisions[row["sample_id"]]
        for field in LABEL_FIELDS:
            row[field] = str(decision.get(field, ""))

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.{os.getpid()}.tmp"
    try:
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply frozen blind edge decisions.")
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    apply_decisions(args.annotations, args.decisions, args.output)


if __name__ == "__main__":
    main()
