"""Build reviewed pair and group outcomes from production predictions."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from image_clustering.evaluation.reviewed_models import parse_bool
from image_clustering.evaluation.reviewed_predictions import (
    UnionFind,
    calibrate_prediction,
    normalize_prediction,
)


def _failure_category(row: Mapping[str, Any]) -> str:
    reason = str(row.get("failure_or_demotion_reason", "")).lower()
    if not row.get("prediction_present"):
        return "candidate_generation"
    if "registration" in reason or "overlap" in reason or "match" in reason:
        return "registration"
    if "transform" in reason or "reprojection" in reason:
        return "transform_quality"
    if "visibility" in reason or "mask" in reason:
        return "visibility_or_occlusion_localization"
    if "outside" in reason or "exterior" in reason:
        return "exterior_agreement"
    if row.get("hard_contradiction") and row.get("truth_same_document"):
        return "incorrect_contradiction"
    if row.get("pair_status") == "contaminated_component":
        return "graph_bridge_behavior"
    if "filename" in reason:
        return "filename_safety_demotion"
    if row.get("truth_same_document"):
        return "same_document_probability_or_deterministic_decision"
    return "suspicious_negative_probability"


def _optional_annotation_bool(
    annotation: Mapping[str, str],
    field: str,
) -> bool | None:
    value = annotation.get(field)
    if value is None or not str(value).strip():
        return None
    return parse_bool(value)


def normalized_predictions(
    raw_predictions: Sequence[Mapping[str, Any]],
    calibrator: Mapping[str, Any] | None,
) -> dict[tuple[str, str], dict[str, Any]]:
    output = {}
    for raw in raw_predictions:
        row = calibrate_prediction(normalize_prediction(raw), calibrator)
        key = (row["image_a"], row["image_b"])
        if key in output:
            raise ValueError(f"duplicate pair prediction: {key}")
        output[key] = row
    return output


def build_components(
    image_ids: Sequence[str],
    predictions: Mapping[tuple[str, str], Mapping[str, Any]],
    native_components: Mapping[str, str] | None,
) -> UnionFind:
    union = UnionFind(image_ids)
    if native_components is None:
        for row in predictions.values():
            if row["automatic_edge"]:
                union.union(row["image_a"], row["image_b"])
        return union
    grouped: dict[str, list[str]] = defaultdict(list)
    for image_id in image_ids:
        grouped[native_components.get(image_id, f"singleton:{image_id}")].append(
            image_id
        )
    for component in grouped.values():
        for image_id in component[1:]:
            union.union(component[0], image_id)
    return union


def _missing_prediction(
    image_a: str,
    image_b: str,
    calibrator: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return calibrate_prediction(
        normalize_prediction(
            {
                "image_a": image_a,
                "image_b": image_b,
                "reason": "pair_not_compared",
            }
        ),
        calibrator,
    )


def build_pair_rows(
    truth_pairs: Sequence[Mapping[str, Any]],
    predictions: Mapping[tuple[str, str], Mapping[str, Any]],
    union: UnionFind,
    splits: Mapping[str, Mapping[str, Any]],
    subtypes: Mapping[str, Mapping[str, str]],
    calibrator: Mapping[str, Any] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pair_rows = []
    failures = []
    for truth in truth_pairs:
        key = tuple(sorted((truth["image_a"], truth["image_b"])))
        prediction = predictions.get(key) or _missing_prediction(
            key[0], key[1], calibrator
        )
        accepted = truth["truth"] == "accepted"
        connected = union.find(key[0]) == union.find(key[1])
        if accepted:
            status = (
                "deterministically_linked"
                if prediction["automatic_edge"]
                else "transitively_connected"
                if connected
                else "review_only"
                if prediction["candidate_flag"]
                else "missed_entirely"
            )
        else:
            status = (
                "automatic_false_link"
                if prediction["automatic_edge"]
                else "contaminated_component"
                if connected
                else "candidate_only"
                if prediction["candidate_flag"]
                else "correctly_separated"
            )
        annotation = subtypes.get(truth["original_cluster_id"], {})
        subtype = annotation.get(
            "occlusion_subtype",
            "uncertain_occlusion_subtype" if accepted else "different_document",
        )
        material_metric = _optional_annotation_bool(
            annotation,
            "material_occlusion_metric_included",
        )
        row = {
            **truth,
            "split": splits[truth["original_cluster_id"]]["split"],
            "derived_occlusion_subtype": subtype,
            "derived_visual_relationship_category": annotation.get(
                "visual_relationship_category", "unannotated"
            ),
            "derived_visual_overlay_category": annotation.get(
                "visual_overlay_category", "unannotated"
            ),
            "derived_material_occlusion_metric_included": material_metric,
            "derived_occlusion_size_category": annotation.get(
                "occlusion_size_category", "unannotated"
            ),
            "truth_same_document": int(accepted),
            "pair_status": status,
            "deterministic_decision": prediction["deterministic_same_document"],
            "candidate_flag": prediction["candidate_flag"],
            "automatic_link_eligible": prediction["automatic_link_eligible"],
            "automatic_edge": prediction["automatic_edge"],
            "same_component": connected,
            "hard_contradiction": prediction["hard_contradiction"],
            "registration_model": prediction.get("registration_model"),
            "registration_fallback_used": prediction.get(
                "registration_fallback_used", False
            ),
            "registration_alignment_score": prediction.get(
                "registration_alignment_score"
            ),
            "feature_overlap": prediction.get("feature_overlap"),
            "same_document_probability": prediction["same_document_probability"],
            "occluded_given_same_probability": prediction[
                "occluded_given_same_probability"
            ],
            "same_clean_probability": prediction["same_clean_probability"],
            "same_occluded_probability": prediction["same_occluded_probability"],
            "different_document_probability": prediction[
                "different_document_probability"
            ],
            "probability_calibration_source": prediction[
                "probability_calibration_source"
            ],
            "branch": prediction.get("branch"),
            "review_tier": prediction.get("review_tier"),
            "reason": prediction.get("reason", ""),
            "failure_or_demotion_reason": prediction.get(
                "failure_or_demotion_reason", prediction.get("reason", "")
            ),
            "runtime_seconds": prediction.get("runtime_seconds"),
            "prediction_present": key in predictions,
        }
        pair_rows.append(row)
        if status in {
            "review_only",
            "missed_entirely",
            "automatic_false_link",
            "contaminated_component",
        }:
            failures.append({**row, "failure_category": _failure_category(row)})
    return pair_rows, failures


def build_group_rows(
    group_truth: Sequence[Mapping[str, Any]],
    pair_rows: Sequence[Mapping[str, Any]],
    union: UnionFind,
    splits: Mapping[str, Mapping[str, Any]],
    subtypes: Mapping[str, Mapping[str, str]],
) -> list[dict[str, Any]]:
    output = []
    for group in group_truth:
        roots = {union.find(image_id) for image_id in group["image_ids"]}
        accepted = group["review_decision"] == "accepted"
        complete = len(roots) == (1 if accepted else len(group["image_ids"]))
        connected_pairs = sum(
            row["same_component"]
            for row in pair_rows
            if row["original_cluster_id"] == group["original_cluster_id"]
            and row["truth_same_document"]
        )
        annotation = subtypes.get(group["original_cluster_id"], {})
        output.append(
            {
                "original_cluster_id": group["original_cluster_id"],
                "review_decision": group["review_decision"],
                "split": splits[group["original_cluster_id"]]["split"],
                "cluster_size": group["cluster_size"],
                "derived_occlusion_subtype": annotation.get(
                    "occlusion_subtype", "not_applicable"
                ),
                "derived_visual_relationship_category": annotation.get(
                    "visual_relationship_category", "unannotated"
                ),
                "derived_visual_overlay_category": annotation.get(
                    "visual_overlay_category", "unannotated"
                ),
                "derived_material_occlusion_metric_included": (
                    _optional_annotation_bool(
                        annotation,
                        "material_occlusion_metric_included",
                    )
                ),
                "predicted_component_count": len(roots),
                "complete_component_recovery": complete if accepted else None,
                "complete_rejected_separation": complete if not accepted else None,
                "connected_positive_pairs": connected_pairs,
                "group_status": (
                    "recovered"
                    if accepted and complete
                    else "split"
                    if accepted
                    else "separated"
                    if complete
                    else "contaminated"
                ),
                "image_ids": group["image_ids"],
            }
        )
    return output
