"""Compute and write grouped reviewed-real evaluation reports."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from image_clustering.evaluation.reviewed_metrics_helpers import (
    binary_probability_metrics,
    calibration_table,
    ece,
    fraction,
    mean,
    review_budget_curve,
    safe_log_loss,
    split_metrics,
)
from image_clustering.evaluation.reviewed_models import SCHEMA_VERSION, sha256
from image_clustering.evaluation.reviewed_validate import write_csv, write_jsonl


def _state_calibration(pair_rows: list[dict[str, Any]]) -> dict[str, Any]:
    state_rows = [
        row
        for row in pair_rows
        if row["derived_occlusion_subtype"]
        in {"same_clean", "same_occluded", "mixed_or_multi_state", "different_document"}
    ]
    for row in state_rows:
        row["truth_same_clean"] = int(
            row["derived_occlusion_subtype"] == "same_clean"
        )
        row["truth_same_occluded"] = int(
            row["derived_occlusion_subtype"]
            in {"same_occluded", "mixed_or_multi_state"}
        )
        row["truth_different_document"] = int(
            not row["truth_same_document"]
        )
    by_size = {}
    for category in sorted(
        {row["derived_occlusion_size_category"] for row in state_rows}
    ):
        rows = [
            row
            for row in state_rows
            if row["derived_occlusion_size_category"] == category
        ]
        by_size[category] = binary_probability_metrics(
            rows,
            probability_key="same_occluded_probability",
            truth_key="truth_same_occluded",
        )
    return {
        "same_clean_calibration": binary_probability_metrics(
            state_rows,
            probability_key="same_clean_probability",
            truth_key="truth_same_clean",
        ),
        "same_occluded_calibration": binary_probability_metrics(
            state_rows,
            probability_key="same_occluded_probability",
            truth_key="truth_same_occluded",
        ),
        "negative_calibration": binary_probability_metrics(
            state_rows,
            probability_key="different_document_probability",
            truth_key="truth_different_document",
        ),
        "calibration_by_occlusion_size": by_size,
    }


def calculate_metrics(
    pair_rows: list[dict[str, Any]],
    group_rows: list[dict[str, Any]],
    *,
    prepared_dir: Path,
    predictions_path: Path,
    calibrator_path: Path | None,
    calibration_source: str,
    run_label: str,
) -> dict[str, Any]:
    positive = [row for row in pair_rows if row["truth_same_document"]]
    negative = [row for row in pair_rows if not row["truth_same_document"]]
    accepted = [row for row in group_rows if row["review_decision"] == "accepted"]
    rejected = [row for row in group_rows if row["review_decision"] == "rejected"]
    contaminated = [row for row in rejected if row["group_status"] == "contaminated"]
    same_clean = [
        row for row in positive if row["derived_occlusion_subtype"] == "same_clean"
    ]
    same_occluded = [
        row
        for row in positive
        if row["derived_occlusion_subtype"] in {"same_occluded", "mixed_or_multi_state"}
    ]
    candidate_tp = sum(bool(row["candidate_flag"]) for row in positive)
    candidate_fp = sum(bool(row["candidate_flag"]) for row in negative)
    auto_tp = sum(bool(row["automatic_edge"]) for row in positive)
    auto_fp = sum(bool(row["automatic_edge"]) for row in negative)
    reliability = calibration_table(
        pair_rows,
        probability_key="same_document_probability",
        truth_key="truth_same_document",
    )
    registration = [
        row
        for row in pair_rows
        if row["registration_model"] is not None
        or row["registration_alignment_score"] is not None
        or row["feature_overlap"] is not None
    ]
    runtimes = [
        float(row["runtime_seconds"])
        for row in pair_rows
        if row["runtime_seconds"] is not None
    ]
    by_split = {
        split: split_metrics(
            [row for row in pair_rows if row["split"] == split],
            [row for row in group_rows if row["split"] == split],
        )
        for split in sorted({row["split"] for row in pair_rows})
    }
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "run_label": run_label,
        "prepared_dir": str(prepared_dir.resolve()),
        "predictions_path": str(predictions_path.resolve()),
        "predictions_sha256": sha256(predictions_path),
        "calibrator": str(calibrator_path.resolve()) if calibrator_path else None,
        "probability_calibration_source": calibration_source,
        "pair_count": len(pair_rows),
        "by_split": by_split,
        "group_metrics": {
            "accepted_group_recovery": fraction(
                sum(row["group_status"] == "recovered" for row in accepted),
                len(accepted),
            ),
            "complete_component_recovery_count": sum(
                row["group_status"] == "recovered" for row in accepted
            ),
            "rejected_group_separation": fraction(
                sum(row["group_status"] == "separated" for row in rejected),
                len(rejected),
            ),
            "contaminated_component_count": len(contaminated),
            "image_weighted_contamination": fraction(
                sum(row["cluster_size"] for row in contaminated),
                sum(row["cluster_size"] for row in rejected),
            ),
            "multi_image_accepted_group_recall": fraction(
                sum(
                    row["group_status"] == "recovered"
                    for row in accepted
                    if row["cluster_size"] > 2
                ),
                sum(row["cluster_size"] > 2 for row in accepted),
            ),
        },
        "pair_metrics": {
            "same_document_recall_connected": fraction(
                sum(bool(row["same_component"]) for row in positive), len(positive)
            ),
            "same_occluded_candidate_recall": fraction(
                sum(bool(row["candidate_flag"]) for row in same_occluded),
                len(same_occluded),
            ),
            "candidate_precision": fraction(
                candidate_tp, candidate_tp + candidate_fp
            ),
            "candidate_recall": fraction(candidate_tp, len(positive)),
            "automatic_link_precision": fraction(auto_tp, auto_tp + auto_fp),
            "automatic_link_recall": fraction(auto_tp, len(positive)),
            "negative_false_link_rate": fraction(auto_fp, len(negative)),
            "contaminated_negative_pair_rate": fraction(
                sum(bool(row["same_component"]) for row in negative), len(negative)
            ),
            "confusion_matrix_automatic_edge": {
                "tp": auto_tp,
                "fn": len(positive) - auto_tp,
                "fp": auto_fp,
                "tn": len(negative) - auto_fp,
            },
        },
        "probability_metrics": {
            "log_loss": mean(
                safe_log_loss(
                    int(row["truth_same_document"]),
                    float(row["same_document_probability"]),
                )
                for row in pair_rows
            ),
            "brier_score": mean(
                (float(row["same_document_probability"]) - row["truth_same_document"])
                ** 2
                for row in pair_rows
            ),
            "expected_calibration_error": ece(reliability),
            "reliability_table": reliability,
            "same_clean_mean_p_same_clean": mean(
                float(row["same_clean_probability"]) for row in same_clean
            ),
            "same_occluded_mean_p_same_occluded": mean(
                float(row["same_occluded_probability"]) for row in same_occluded
            ),
            "negative_mean_p_different": mean(
                float(row["different_document_probability"]) for row in negative
            ),
            **_state_calibration(pair_rows),
        },
        "operational_metrics": {
            "review_volume_pairs": sum(
                bool(row["candidate_flag"]) for row in pair_rows
            ),
            "review_fraction": fraction(
                sum(bool(row["candidate_flag"]) for row in pair_rows), len(pair_rows)
            ),
            "review_budget_curve": review_budget_curve(pair_rows),
            "registration_success": fraction(
                sum(
                    bool(row["registration_model"])
                    or bool(row["feature_overlap"])
                    or bool(row["registration_alignment_score"])
                    for row in registration
                ),
                len(registration),
            ),
            "registration_rows": len(registration),
            "ecc_frequency": fraction(
                sum(bool(row["registration_fallback_used"]) for row in registration),
                len(registration),
            ),
            "runtime_seconds_sum": sum(runtimes) if runtimes else None,
            "runtime_seconds_mean": mean(runtimes),
        },
        "promotion_gates": {
            "zero_reviewed_negative_automatic_edges": auto_fp == 0,
            "zero_contaminated_reviewed_components": not contaminated,
            "passed": auto_fp == 0 and not contaminated,
        },
        "limitations": (
            [
                "same-clean and same-occluded calibration require completed "
                "accepted-group subtype annotations"
            ]
            if not same_clean or not same_occluded
            else []
        ),
    }
    return metrics


def write_reports(
    output_dir: Path,
    pair_rows: list[dict[str, Any]],
    group_rows: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    metrics: Mapping[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "pair_results.jsonl", pair_rows)
    write_csv(output_dir / "pair_results.csv", pair_rows)
    write_jsonl(output_dir / "group_results.jsonl", group_rows)
    write_csv(output_dir / "group_results.csv", group_rows)
    write_jsonl(output_dir / "failure_analysis.jsonl", failures)
    write_csv(output_dir / "failure_analysis.csv", failures)
    write_csv(
        output_dir / "reliability_table.csv",
        metrics["probability_metrics"]["reliability_table"],
    )
    write_csv(
        output_dir / "risk_coverage_curve.csv",
        metrics["operational_metrics"]["review_budget_curve"],
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
