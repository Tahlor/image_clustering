"""Orchestrate reviewed-real pair, component, and report evaluation."""

from __future__ import annotations

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


def evaluate_predictions(
    prepared_dir: Path,
    predictions_path: Path,
    output_dir: Path,
    *,
    subtype_path: Path | None = None,
    calibrator_path: Path | None = None,
    run_label: str = "evaluation",
) -> dict[str, Any]:
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
    write_reports(output_dir, pair_rows, group_rows, failures, metrics)
    return metrics
