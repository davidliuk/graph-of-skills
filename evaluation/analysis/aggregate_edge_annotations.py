from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .manifest import atomic_write_json


LABEL_FIELDS = ("valid_relation", "type_correct", "direction_correct")
VALID_LABELS = {"yes", "no", "uncertain", "n/a"}


def _normalize_label(value: Any) -> str:
    label = str(value or "").strip().lower()
    aliases = {"na": "n/a", "not applicable": "n/a"}
    return aliases.get(label, label)


def _metric(rows: list[dict[str, str]], field: str) -> dict[str, Any]:
    counts = Counter(_normalize_label(row.get(field, "")) for row in rows)
    yes = counts["yes"]
    no = counts["no"]
    decided = yes + no
    return {
        "yes": yes,
        "no": no,
        "uncertain": counts["uncertain"],
        "n/a": counts["n/a"],
        "missing": counts[""],
        "precision_excluding_uncertain": yes / decided if decided else None,
        "decided_count": decided,
    }


def _summarize(rows: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "count": len(rows),
        **{field: _metric(rows, field) for field in LABEL_FIELDS},
    }


def aggregate_annotations(
    csv_path: Path,
    key_path: Path,
    *,
    require_complete: bool = True,
) -> dict[str, Any]:
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8", newline="")))
    key_rows = json.loads(key_path.read_text(encoding="utf-8"))
    key_by_id = {str(row["sample_id"]): row for row in key_rows}

    seen: set[str] = set()
    joined: list[dict[str, str]] = []
    for row in rows:
        sample_id = str(row.get("sample_id", "")).strip()
        if not sample_id or sample_id in seen:
            raise ValueError(f"Missing or duplicate sample_id: {sample_id!r}")
        if sample_id not in key_by_id:
            raise ValueError(f"Annotation sample absent from key: {sample_id}")
        seen.add(sample_id)
        normalized = {
            field: _normalize_label(row.get(field, "")) for field in LABEL_FIELDS
        }
        for field, label in normalized.items():
            if label and label not in VALID_LABELS:
                raise ValueError(f"Invalid {field} label for {sample_id}: {label!r}")
        if require_complete and not normalized["valid_relation"]:
            raise ValueError(f"Missing valid_relation label for {sample_id}")
        joined.append(
            {
                "sample_id": sample_id,
                "stratum": str(key_by_id[sample_id].get("stratum", "")),
                **normalized,
            }
        )

    missing_ids = sorted(set(key_by_id) - seen)
    if require_complete and missing_ids:
        raise ValueError(f"Annotation CSV is missing {len(missing_ids)} keyed samples")

    strata = sorted({row["stratum"] for row in joined})
    return {
        "schema_version": 1,
        "annotation_csv": str(csv_path.resolve()),
        "annotation_key": str(key_path.resolve()),
        "missing_keyed_sample_ids": missing_ids,
        "overall": _summarize(joined),
        "by_stratum": {
            stratum: _summarize([row for row in joined if row["stratum"] == stratum])
            for stratum in strata
        },
    }


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Blind Edge Annotation Summary",
        "",
        "Uncertain and n/a labels are preserved and excluded from precision denominators.",
        "",
        "| Stratum | N | Valid relation | Type correct | Direction correct |",
        "|---|---:|---:|---:|---:|",
    ]
    rows = {"overall": result["overall"], **result["by_stratum"]}
    for name, row in rows.items():
        values = []
        for field in LABEL_FIELDS:
            metric = row[field]
            precision = metric["precision_excluding_uncertain"]
            values.append(
                "n/a"
                if precision is None
                else f"{precision:.3f} ({metric['yes']}/{metric['decided_count']})"
            )
        lines.append(
            f"| {name} | {row['count']} | {values[0]} | {values[1]} | {values[2]} |"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate blind edge annotations.")
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = aggregate_annotations(
        args.annotations,
        args.key,
        require_complete=not args.allow_incomplete,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output_dir / "summary.json", result)
    markdown = render_markdown(result)
    (args.output_dir / "results.md").write_text(markdown, encoding="utf-8")
    print(markdown, end="")


if __name__ == "__main__":
    main()
