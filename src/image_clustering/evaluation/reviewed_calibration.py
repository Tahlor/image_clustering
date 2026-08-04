"""Selection-only real calibration and exact before/after comparison."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from image_clustering.evaluation.reviewed_models import (
    SCHEMA_VERSION,
    load_jsonl,
    parse_bool,
)
from image_clustering.evaluation.reviewed_predictions import apply_isotonic
from image_clustering.evaluation.reviewed_prepare import load_subtypes
from image_clustering.evaluation.reviewed_subtypes import (
    validate_completed_subtypes,
)


def _pav_fit(points: Sequence[tuple[float, int]]) -> list[tuple[float, float]]:
    if not points:
        raise ValueError("cannot fit a calibrator without observations")
    ordered = sorted(points)
    blocks: list[list[float]] = []
    for score, truth in ordered:
        blocks.append([score, score, float(truth), 1.0])
        while len(blocks) >= 2:
            left, right = blocks[-2], blocks[-1]
            if left[2] / left[3] <= right[2] / right[3]:
                break
            blocks[-2:] = [
                [left[0], right[1], left[2] + right[2], left[3] + right[3]]
            ]
    knots = []
    for low, high, total, count in blocks:
        value = total / count
        knots.extend([(low, value), (high, value)])
    compact = []
    for point in knots:
        if compact and point == compact[-1]:
            continue
        compact.append(point)
    return compact


def fit_real_calibration(
    prepared_dir: Path,
    pair_results_path: Path,
    output_path: Path,
    *,
    subtype_path: Path | None = None,
) -> dict[str, Any]:
    splits = {
        row["original_cluster_id"]: row["split"]
        for row in load_jsonl(prepared_dir / "split_manifest.jsonl")
    }
    rows = load_jsonl(pair_results_path)
    selection = [
        row for row in rows if splits[row["original_cluster_id"]] == "selection"
    ]
    identity_knots = _pav_fit(
        [
            (
                float(row["same_document_probability"]),
                int(row["truth_same_document"]),
            )
            for row in selection
        ]
    )
    resolved_subtype_path = subtype_path or (
        prepared_dir / "accepted_group_occlusion_subtypes.csv"
    )
    subtype_validation = (
        validate_completed_subtypes(prepared_dir, resolved_subtype_path)
        if subtype_path is not None
        else None
    )
    subtypes = load_subtypes(resolved_subtype_path)
    conditional = [
        row
        for row in selection
        if row["truth_same_document"]
        and row["original_cluster_id"] in subtypes
        and subtypes[row["original_cluster_id"]].get(
            "material_occlusion_metric_included"
        )
    ]
    conditional_knots = (
        _pav_fit(
            [
                (
                    float(row["occluded_given_same_probability"]),
                    int(
                        parse_bool(
                            subtypes[row["original_cluster_id"]][
                                "material_occlusion_metric_included"
                            ]
                        )
                    ),
                )
                for row in conditional
            ]
        )
        if conditional
        else None
    )
    summary = json.loads(
        (prepared_dir / "dataset_preparation_summary.json").read_text(
            encoding="utf-8"
        )
    )
    calibrator = {
        "schema_version": SCHEMA_VERSION,
        "calibration_source": "reviewed-real-selection-only",
        "prepared_manifest_sha256": summary["manifest_sha256"],
        "selection_pair_count": len(selection),
        "identity_isotonic_knots": identity_knots,
        "conditional_occlusion_isotonic_knots": conditional_knots,
        "conditional_occlusion_truth": "material_occlusion_metric_included",
        "subtype_validation": subtype_validation,
        "graph_edge_policy": (
            "unchanged; calibrated probabilities are review-only"
        ),
        "locked_audit_used_for_fit": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(calibrator, indent=2, sort_keys=True), encoding="utf-8"
    )
    return calibrator


def compare_runs(
    before_path: Path,
    after_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    before = json.loads(before_path.read_text(encoding="utf-8"))
    after = json.loads(after_path.read_text(encoding="utf-8"))
    if before["pair_count"] != after["pair_count"]:
        raise ValueError("before/after pair populations differ")
    if before["prepared_dir"] != after["prepared_dir"]:
        raise ValueError("before/after prepared datasets differ")
    paths = (
        ("group_metrics", "accepted_group_recovery"),
        ("group_metrics", "rejected_group_separation"),
        ("group_metrics", "contaminated_component_count"),
        ("pair_metrics", "same_document_recall_connected"),
        ("pair_metrics", "candidate_recall"),
        ("pair_metrics", "candidate_precision"),
        ("pair_metrics", "automatic_link_precision"),
        ("pair_metrics", "automatic_link_recall"),
        ("pair_metrics", "negative_false_link_rate"),
        ("probability_metrics", "log_loss"),
        ("probability_metrics", "brier_score"),
        ("probability_metrics", "expected_calibration_error"),
        ("operational_metrics", "review_fraction"),
    )
    comparisons = []
    for section, metric in paths:
        left, right = before[section][metric], after[section][metric]
        delta = (
            right - left
            if isinstance(left, (int, float))
            and isinstance(right, (int, float))
            and left is not None
            and right is not None
            else None
        )
        comparisons.append(
            {
                "section": section,
                "metric": metric,
                "before": left,
                "after": right,
                "delta": delta,
            }
        )
    report = {
        "schema_version": SCHEMA_VERSION,
        "before": str(before_path),
        "after": str(after_path),
        "same_evaluation_population": True,
        "comparisons": comparisons,
        "before_promotion_gates": before["promotion_gates"],
        "after_promotion_gates": after["promotion_gates"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


__all__ = ["apply_isotonic", "compare_runs", "fit_real_calibration"]
