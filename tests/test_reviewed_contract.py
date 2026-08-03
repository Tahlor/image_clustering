from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from image_clustering.evaluation.reviewed_groups import (
    CONTRACT,
    evaluate_predictions,
    expand_pairs,
    load_csv_manifest,
    make_grouped_splits,
    prepare_dataset,
    validate_manifest,
)
from reviewed_fixture import build_package


def test_integrity_contract_and_pair_expansion(tmp_path: Path) -> None:
    csv_path, jsonl_path = build_package(tmp_path)
    report = validate_manifest(
        csv_path, jsonl_path=jsonl_path, package_root=tmp_path
    )
    assert report["status"] == "pass"
    assert report["counts"] == {
        "rows": 422,
        "original_clusters": 200,
        "accepted_clusters": 134,
        "rejected_clusters": 66,
        "accepted_images": 282,
        "rejected_images": 140,
        "positive_pairs": 183,
        "negative_pairs": 83,
    }
    positives, negatives = expand_pairs(load_csv_manifest(csv_path))
    assert len(positives) == CONTRACT.positive_pairs
    assert len(negatives) == CONTRACT.negative_pairs


def test_integrity_fails_instead_of_repairing_labels(tmp_path: Path) -> None:
    csv_path, jsonl_path = build_package(tmp_path)
    lines = csv_path.read_text().splitlines()
    csv_path.write_text("\n".join(lines[:-1]) + "\n")
    report = validate_manifest(csv_path, jsonl_path=jsonl_path, package_root=tmp_path)
    assert report["status"] == "fail"
    assert any("expected 422 rows" in error for error in report["errors"])
    with pytest.raises(ValueError, match="integrity"):
        prepare_dataset(
            csv_path,
            tmp_path / "prepared",
            jsonl_path=jsonl_path,
            package_root=tmp_path,
        )


def test_grouped_split_keeps_clusters_and_sequences_together(tmp_path: Path) -> None:
    csv_path, _ = build_package(tmp_path)
    rows = load_csv_manifest(csv_path)
    assignments = make_grouped_splits(rows)
    by_cluster = {row.original_cluster_id: row.split for row in assignments}
    sequence_splits = {}
    for row in rows:
        sequence_splits.setdefault(row.sequence_id, set()).add(
            by_cluster[row.original_cluster_id]
        )
    assert all(len(splits) == 1 for splits in sequence_splits.values())
    nine = [row for row in assignments if row.cluster_size == 9]
    assert len(nine) == 1


def test_staging_restores_source_sequence_grouping(tmp_path: Path) -> None:
    csv_path, jsonl_path = build_package(tmp_path)
    prepared = tmp_path / "prepared"
    stage = tmp_path / "stage"
    summary = prepare_dataset(
        csv_path,
        prepared,
        jsonl_path=jsonl_path,
        package_root=tmp_path,
        stage_root=stage,
    )
    assert summary["staged_image_count"] == 422
    rows = list(csv.DictReader((prepared / "evaluation_input_manifest.csv").open()))
    assert len(rows) == 422
    by_proposal = {}
    for row in rows:
        by_proposal.setdefault(row["original_cluster_id"], set()).add(
            str(Path(row["staged_path"]).parent)
        )
        staged_path = Path(row["staged_path"])
        assert staged_path.is_file()
        expected = f"{row['original_cluster_id']}/{Path(row['image_id']).name}"
        assert row["prediction_image_id"] == expected
        assert staged_path.relative_to(stage).as_posix() == expected
    assert all(len(paths) == 1 for paths in by_proposal.values())


def test_native_clustering_components_are_authoritative(tmp_path: Path) -> None:
    csv_path, jsonl_path = build_package(tmp_path)
    prepared = tmp_path / "prepared"
    prepare_dataset(
        csv_path,
        prepared,
        jsonl_path=jsonl_path,
        package_root=tmp_path,
    )
    positive = json.loads(
        (prepared / "positive_pairs.jsonl").read_text().splitlines()[0]
    )
    clustering = {
        "schema_version": 1,
        "comparisons": [
            {
                "first_image_id": positive["image_a"],
                "second_image_id": positive["image_b"],
                "same_document": False,
                "automatic_link_eligible": False,
                "occlusion_candidate_flag": False,
            }
        ],
        "clusters": [
            {
                "cluster_id": "native_component",
                "image_ids": [positive["image_a"], positive["image_b"]],
            }
        ],
    }
    path = tmp_path / "clustering.json"
    path.write_text(json.dumps(clustering))
    metrics = evaluate_predictions(prepared, path, tmp_path / "result")
    assert metrics["pair_metrics"]["same_document_recall_connected"] > 0
    first = json.loads(
        (tmp_path / "result/pair_results.jsonl").read_text().splitlines()[0]
    )
    assert first["pair_status"] == "transitively_connected"


def test_staged_prediction_ids_are_canonicalized(tmp_path: Path) -> None:
    csv_path, jsonl_path = build_package(tmp_path)
    prepared = tmp_path / "prepared"
    stage = tmp_path / "stage"
    prepare_dataset(
        csv_path,
        prepared,
        jsonl_path=jsonl_path,
        package_root=tmp_path,
        stage_root=stage,
    )
    positive = json.loads(
        (prepared / "positive_pairs.jsonl").read_text().splitlines()[0]
    )
    input_rows = list(
        csv.DictReader((prepared / "evaluation_input_manifest.csv").open())
    )
    by_truth = {row["image_id"]: row["prediction_image_id"] for row in input_rows}
    predictions = tmp_path / "staged_predictions.jsonl"
    predictions.write_text(
        json.dumps(
            {
                "image_a": by_truth[positive["image_a"]],
                "image_b": by_truth[positive["image_b"]],
                "same_document": True,
                "automatic_link_eligible": True,
                "same_document_probability": 0.99,
                "occluded_given_same_probability": 0.5,
                "same_clean_probability": 0.495,
                "same_occluded_probability": 0.495,
                "different_document_probability": 0.01,
            }
        )
        + "\n"
    )
    evaluate_predictions(prepared, predictions, tmp_path / "result")
    first = json.loads(
        (tmp_path / "result/pair_results.jsonl").read_text().splitlines()[0]
    )
    assert first["pair_status"] == "deterministically_linked"
