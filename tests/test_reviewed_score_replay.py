from __future__ import annotations

import csv
import json
from pathlib import Path

from image_clustering.evaluation.reviewed_groups import prepare_dataset
from image_clustering.evaluation.reviewed_predictions import (
    load_prediction_source,
    normalize_prediction,
)
from image_clustering.evaluation.reviewed_score_replay import (
    load_reviewed_candidate_csv,
    tune_candidate_replay,
)
from reviewed_fixture import build_package


CANDIDATE_FIELDS = [
    "index",
    "first_image_id",
    "second_image_id",
    "sequence_id",
    "index_gap",
    "review_tier",
    "priority_reason",
    "same_occluded_probability",
    "same_document_probability",
    "occluded_given_same_probability",
    "deterministic_same_document",
    "automatic_link_eligible",
    "hard_contradiction",
    "raw_hard_contradiction",
    "acceptance_conflict",
    "same_component",
    "common_accepted_neighbors",
    "registration_fallback_used",
    "registration_alignment_score",
    "decision_reason",
]


def _candidate_id(image_id: str) -> str:
    parts = image_id.split("/")
    stem = Path(parts[-1]).stem
    return f"{parts[-2]}/{stem}.j2k"


def _write_candidates(prepared: Path, path: Path) -> None:
    rows = []
    index = 0
    for filename in ("positive_pairs.jsonl", "negative_pairs.jsonl"):
        for line in (prepared / filename).read_text().splitlines():
            truth = json.loads(line)
            accepted = truth["truth"] == "accepted"
            p_same = 0.98 if accepted else 0.25
            q = 0.90 if accepted else 0.20
            rows.append(
                {
                    "index": index,
                    "first_image_id": _candidate_id(truth["image_a"]),
                    "second_image_id": _candidate_id(truth["image_b"]),
                    "sequence_id": truth["sequence_id"],
                    "index_gap": abs(
                        truth["sequence_index_a"] - truth["sequence_index_b"]
                    ),
                    "review_tier": "raw_candidate",
                    "priority_reason": "fixture",
                    "same_occluded_probability": p_same * q,
                    "same_document_probability": p_same,
                    "occluded_given_same_probability": q,
                    "deterministic_same_document": "True",
                    "automatic_link_eligible": "True",
                    "hard_contradiction": "False",
                    "raw_hard_contradiction": "False",
                    "acceptance_conflict": "False",
                    "same_component": "False",
                    "common_accepted_neighbors": 0,
                    "registration_fallback_used": "False",
                    "registration_alignment_score": 0.9 if accepted else 0.4,
                    "decision_reason": "fixture",
                }
            )
            index += 1
    rows.append(
        {
            **rows[0],
            "index": index,
            "first_image_id": "unknown/unknown-1.j2k",
            "second_image_id": "unknown/unknown-2.j2k",
        }
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANDIDATE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def test_prediction_csv_parses_boolean_strings(tmp_path: Path) -> None:
    path = tmp_path / "predictions.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "first_image_id",
                "second_image_id",
                "deterministic_same_document",
                "automatic_link_eligible",
                "hard_contradiction",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "first_image_id": "a.jpg",
                "second_image_id": "b.jpg",
                "deterministic_same_document": "False",
                "automatic_link_eligible": "False",
                "hard_contradiction": "False",
            }
        )
    raw, components = load_prediction_source(path)
    normalized = normalize_prediction(raw[0])
    assert components is None
    assert normalized["deterministic_same_document"] is False
    assert normalized["automatic_edge"] is False


def test_score_replay_maps_ids_and_finds_perfect_safe_fit(tmp_path: Path) -> None:
    csv_path, jsonl_path = build_package(tmp_path)
    prepared = tmp_path / "prepared"
    prepare_dataset(
        csv_path,
        prepared,
        jsonl_path=jsonl_path,
        package_root=tmp_path,
    )
    candidates = tmp_path / "occlusion_candidates.csv"
    _write_candidates(prepared, candidates)
    _, matched, ingestion = load_reviewed_candidate_csv(csv_path, candidates)
    assert len(matched) == 266
    assert ingestion["unmatched_candidate_rows"] == 1

    output = tmp_path / "replay"
    summary = tune_candidate_replay(
        csv_path,
        candidates,
        prepared,
        output,
    )
    selection = summary["selection_tuned"]
    overfit = summary["full_dataset_overfit_diagnostic"]
    assert selection["fit_metrics"]["zero_false_links"]
    assert selection["locked_audit_metrics"]["perfect_group_fit"]
    assert overfit["fit_metrics"]["perfect_group_fit"]
    assert overfit["promotion_eligible"] is False
    assert (output / "selection_tuned_predictions.jsonl").is_file()
    assert (output / "full_dataset_overfit_predictions.jsonl").is_file()
