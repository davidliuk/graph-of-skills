from __future__ import annotations

import csv
import json

from evaluation.analysis.aggregate_edge_annotations import aggregate_annotations


def test_aggregate_annotations_excludes_uncertain_from_precision(tmp_path):
    csv_path = tmp_path / "labels.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "sample_id",
                "valid_relation",
                "type_correct",
                "direction_correct",
            ],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "sample_id": "s1",
                    "valid_relation": "yes",
                    "type_correct": "yes",
                    "direction_correct": "yes",
                },
                {
                    "sample_id": "s2",
                    "valid_relation": "no",
                    "type_correct": "no",
                    "direction_correct": "no",
                },
                {
                    "sample_id": "s3",
                    "valid_relation": "uncertain",
                    "type_correct": "uncertain",
                    "direction_correct": "n/a",
                },
            ]
        )
    key_path = tmp_path / "key.json"
    key_path.write_text(
        json.dumps(
            [
                {"sample_id": "s1", "stratum": "deterministic_io/dependency"},
                {"sample_id": "s2", "stratum": "llm_validated/workflow"},
                {"sample_id": "s3", "stratum": "llm_validated/semantic"},
            ]
        ),
        encoding="utf-8",
    )

    result = aggregate_annotations(csv_path, key_path, require_complete=True)

    assert result["overall"]["valid_relation"]["yes"] == 1
    assert result["overall"]["valid_relation"]["no"] == 1
    assert result["overall"]["valid_relation"]["uncertain"] == 1
    assert result["overall"]["valid_relation"]["precision_excluding_uncertain"] == 0.5
    assert result["by_stratum"]["llm_validated/workflow"]["count"] == 1
