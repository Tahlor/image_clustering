from __future__ import annotations

import json
from pathlib import Path

import pytest

from image_clustering.evaluation.reviewed_groups import (
    apply_isotonic,
    compare_runs,
    evaluate_predictions,
    fit_real_calibration,
    prepare_dataset,
)
from reviewed_fixture import build_package, write_predictions


def test_probability_alone_never_creates_graph_edge(tmp_path: Path) -> None:
    csv_path, jsonl_path = build_package(tmp_path)
    prepared = tmp_path / "prepared"
    prepare_dataset(
        csv_path,
        prepared,
        jsonl_path=jsonl_path,
        package_root=tmp_path,
    )
    negative = json.loads(
        (prepared / "negative_pairs.jsonl").read_text().splitlines()[0]
    )
    prediction = tmp_path / "predictions.jsonl"
    prediction.write_text(
        json.dumps(
            {
                "image_a": negative["image_a"],
                "image_b": negative["image_b"],
                "same_document": False,
                "automatic_link_eligible": False,
                "hard_contradiction": True,
                "occlusion_candidate_flag": True,
                "same_document_probability": 0.999,
                "occluded_given_same_probability": 0.999,
                "same_clean_probability": 0.000999,
                "same_occluded_probability": 0.998001,
                "different_document_probability": 0.001,
            }
        )
        + "\n"
    )
    metrics = evaluate_predictions(prepared, prediction, tmp_path / "result")
    assert metrics["pair_metrics"]["confusion_matrix_automatic_edge"]["fp"] == 0
    assert metrics["promotion_gates"]["zero_reviewed_negative_automatic_edges"]


def test_false_negative_edge_fails_both_promotion_gates(tmp_path: Path) -> None:
    csv_path, jsonl_path = build_package(tmp_path)
    prepared = tmp_path / "prepared"
    prepare_dataset(
        csv_path,
        prepared,
        jsonl_path=jsonl_path,
        package_root=tmp_path,
    )
    predictions = tmp_path / "predictions.jsonl"
    write_predictions(prepared, predictions, false_edge=True)
    metrics = evaluate_predictions(prepared, predictions, tmp_path / "result")
    assert metrics["pair_metrics"]["confusion_matrix_automatic_edge"]["fp"] == 1
    assert not metrics["promotion_gates"]["zero_reviewed_negative_automatic_edges"]
    assert not metrics["promotion_gates"]["zero_contaminated_reviewed_components"]
    assert not metrics["promotion_gates"]["passed"]


def test_full_fixture_passes_and_writes_required_outputs(tmp_path: Path) -> None:
    csv_path, jsonl_path = build_package(tmp_path)
    prepared = tmp_path / "prepared"
    prepare_dataset(
        csv_path,
        prepared,
        jsonl_path=jsonl_path,
        package_root=tmp_path,
    )
    predictions = tmp_path / "predictions.jsonl"
    write_predictions(prepared, predictions)
    output = tmp_path / "result"
    metrics = evaluate_predictions(prepared, predictions, output)
    assert metrics["promotion_gates"]["passed"]
    assert metrics["pair_metrics"]["automatic_link_precision"] == 1.0
    assert metrics["group_metrics"]["accepted_group_recovery"] == 1.0
    for filename in (
        "pair_results.jsonl",
        "group_results.jsonl",
        "failure_analysis.csv",
        "reliability_table.csv",
        "risk_coverage_curve.csv",
        "metrics.json",
    ):
        assert (output / filename).exists()


def test_calibration_uses_selection_only_and_is_review_only(tmp_path: Path) -> None:
    csv_path, jsonl_path = build_package(tmp_path)
    prepared = tmp_path / "prepared"
    prepare_dataset(
        csv_path,
        prepared,
        jsonl_path=jsonl_path,
        package_root=tmp_path,
    )
    predictions = tmp_path / "predictions.jsonl"
    write_predictions(prepared, predictions)
    result = tmp_path / "result"
    evaluate_predictions(prepared, predictions, result)
    calibrator = fit_real_calibration(
        prepared,
        result / "pair_results.jsonl",
        tmp_path / "calibration.json",
    )
    assert calibrator["calibration_source"] == "reviewed-real-selection-only"
    assert calibrator["locked_audit_used_for_fit"] is False
    assert calibrator["graph_edge_policy"].startswith("unchanged")
    knots = calibrator["identity_isotonic_knots"]
    assert 0 <= apply_isotonic(0.5, knots) <= 1


def test_compare_rejects_population_change(tmp_path: Path) -> None:
    before = tmp_path / "before.json"
    after = tmp_path / "after.json"
    base = {
        "pair_count": 266,
        "prepared_dir": "same",
        "group_metrics": {
            "accepted_group_recovery": 0.5,
            "rejected_group_separation": 1.0,
            "contaminated_component_count": 0,
        },
        "pair_metrics": {
            "same_document_recall_connected": 0.5,
            "candidate_recall": 0.8,
            "candidate_precision": 0.7,
            "automatic_link_precision": 1.0,
            "automatic_link_recall": 0.4,
            "negative_false_link_rate": 0.0,
        },
        "probability_metrics": {
            "log_loss": 0.5,
            "brier_score": 0.2,
            "expected_calibration_error": 0.1,
        },
        "operational_metrics": {"review_fraction": 0.3},
        "promotion_gates": {"passed": True},
    }
    before.write_text(json.dumps(base))
    changed = dict(base, pair_count=265)
    after.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="populations differ"):
        compare_runs(before, after, tmp_path / "comparison.json")


def test_real_calibration_changes_probabilities_not_edges(tmp_path: Path) -> None:
    csv_path, jsonl_path = build_package(tmp_path)
    prepared = tmp_path / "prepared"
    prepare_dataset(
        csv_path,
        prepared,
        jsonl_path=jsonl_path,
        package_root=tmp_path,
    )
    predictions = tmp_path / "predictions.jsonl"
    write_predictions(prepared, predictions)
    baseline_dir = tmp_path / "baseline"
    evaluate_predictions(prepared, predictions, baseline_dir)
    calibrator_path = tmp_path / "calibrator.json"
    fit_real_calibration(
        prepared,
        baseline_dir / "pair_results.jsonl",
        calibrator_path,
    )
    calibrated_dir = tmp_path / "calibrated"
    evaluate_predictions(
        prepared,
        predictions,
        calibrated_dir,
        calibrator_path=calibrator_path,
    )
    baseline = json.loads((baseline_dir / "metrics.json").read_text())
    calibrated = json.loads((calibrated_dir / "metrics.json").read_text())
    assert calibrated["probability_calibration_source"] == (
        "reviewed-real-selection-only"
    )
    assert calibrated["pair_metrics"] == baseline["pair_metrics"]
    assert calibrated["group_metrics"] == baseline["group_metrics"]
    assert calibrated["by_split"]["locked_audit"]["promotion_gates"]["passed"]
