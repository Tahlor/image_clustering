"""Orchestrate reviewed-real pair, component, and report evaluation."""

from __future__ import annotations

from collections.abc import Collection
from pathlib import Path
from typing import Any

from image_clustering.evaluation.reviewed_models import load_jsonl
from image_clustering.evaluation.reviewed_pair_eval import (
    build_components,
    build_group_rows,
    build_pair_rows,
    normalized_predictions,
)
from image_clustering.evaluation.reviewed_predictions import (
    load_calibrator,
    load_prediction_source,
    prediction_id_map,
)
from image_clustering.evaluation.reviewed_prepare import load_subtypes
from image_clustering.evaluation.reviewed_reporting import (
    calculate_metrics,
    write_reports,
)

KNOWN_SPLITS = frozenset({"development", "selection", "locked_audit"})


def _normalize_include_splits(
    include_splits: Collection[str] | None,
) -> frozenset[str] | None:
    if include_splits is None:
        return None
    normalized = frozenset(str(split) for split in include_splits)
    if not normalized:
        raise ValueError("include_splits must not be empty")
    unknown = normalized - KNOWN_SPLITS
    if unknown:
        raise ValueError(f"unknown reviewed split(s): {sorted(unknown)}")
    return normalized


def evaluate_predictions(
    prepared_dir: Path,
    predictions_path: Path,
    output_dir: Path,
    *,
    subtype_path: Path | None = None,
    calibrator_path: Path | None = None,
    run_label: str = "evaluation",
    include_splits: Collection[str] | None = None,
) -> dict[str, Any]:
    selected_splits = _normalize_include_splits(include_splits)
    positive = load_jsonl(prepared_dir / "positive_pairs.jsonl")
    negative = load_jsonl(prepared_dir / "negative_pairs.jsonl")
    group_truth = load_jsonl(prepared_dir / "canonical_reviewed_groups.jsonl")
    splits = {
        row["original_cluster_id"]: row
        for row in load_jsonl(prepared_dir / "split_manifest.jsonl")
    }
    subtypes = load_subtypes(
        subtype_path
        or prepared_dir / "accepted_group_occlusion_subtypes.csv"
    )
    calibrator = load_calibrator(calibrator_path, prepared_dir=prepared_dir)
    raw, native_components = load_prediction_source(
        predictions_path,
        id_map=prediction_id_map(prepared_dir),
    )
    predictions = normalized_predictions(raw, calibrator)
    image_ids = sorted(
        {image_id for group in group_truth for image_id in group["image_ids"]}
    )
    union = build_components(image_ids, predictions, native_components)
    pair_rows, failures = build_pair_rows(
        positive + negative,
        predictions,
        union,
        splits,
        subtypes,
        calibrator,
    )
    group_rows = build_group_rows(
        group_truth,
        pair_rows,
        union,
        splits,
        subtypes,
    )
    if selected_splits is not None:
        selected_group_ids = {
            original_cluster_id
            for original_cluster_id, assignment in splits.items()
            if assignment["split"] in selected_splits
        }
        pair_rows = [
            row
            for row in pair_rows
            if row["original_cluster_id"] in selected_group_ids
        ]
        group_rows = [
            row
            for row in group_rows
            if row["original_cluster_id"] in selected_group_ids
        ]
        failures = [
            row
            for row in failures
            if row["original_cluster_id"] in selected_group_ids
        ]
        if not pair_rows or not group_rows:
            raise ValueError(
                "requested reviewed splits contain no evaluable pairs or groups"
            )
    calibration_source = (
        calibrator["calibration_source"] if calibrator else "uncalibrated"
    )
    metrics = calculate_metrics(
        pair_rows,
        group_rows,
        prepared_dir=prepared_dir,
        predictions_path=predictions_path,
        calibrator_path=calibrator_path,
        calibration_source=calibration_source,
        run_label=run_label,
    )
    evaluated_splits = sorted({row["split"] for row in pair_rows})
    metrics["evaluated_splits"] = evaluated_splits
    metrics["locked_audit_included"] = "locked_audit" in evaluated_splits
    write_reports(output_dir, pair_rows, group_rows, failures, metrics)
    return metrics
