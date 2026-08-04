"""Validate completed visual evidence for accepted reviewed groups."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from image_clustering.evaluation.reviewed_models import (
    CONTRACT,
    load_jsonl,
    parse_bool,
)
from image_clustering.evaluation.reviewed_prepare import load_subtypes

ALLOWED_VISUAL_RELATIONSHIPS = {
    "identical_or_near_identical",
    "material_physical_occlusion",
    "same_document_state_or_content_change",
    "visual_only_overlay",
    "mixed_or_multi_state",
    "uncertain_or_other",
}
ALLOWED_OVERLAY_CATEGORIES = {
    "none",
    "large_number",
    "stamp_or_seal",
    "card_or_label",
    "other_visual_overlay",
    "uncertain",
}
ALLOWED_HIDDEN_CONTENT_RISKS = {
    "none",
    "low",
    "medium",
    "high",
    "uncertain",
}
ALLOWED_OCCLUSION_SIZES = {
    "none",
    "small",
    "medium",
    "large",
    "uncertain",
}
ALLOWED_REGISTRATION_DIFFICULTIES = {
    "easy",
    "moderate",
    "hard",
    "uncertain",
}
REQUIRED_COMPLETED_FIELDS = {
    "member_image_ids_json",
    "visual_relationship_category",
    "visual_overlay_category",
    "material_occlusion_metric_included",
    "affected_image_id",
    "occluded_image_id",
    "better_view_image_id",
    "meaningful_hidden_content_risk",
    "occlusion_size_category",
    "registration_difficulty",
    "evidence",
    "uncertainty_notes",
    "annotator_method",
}


def _required_text(
    row: dict[str, str],
    field: str,
    cluster_id: str,
) -> str:
    if field not in row:
        raise ValueError(f"Subtype row for {cluster_id} is missing {field}")
    return str(row[field] or "").strip()


def _member_ids(row: dict[str, str], cluster_id: str) -> list[str]:
    raw = _required_text(row, "member_image_ids_json", cluster_id)
    try:
        values = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid member_image_ids_json for {cluster_id}"
        ) from error
    if not isinstance(values, list) or not values:
        raise ValueError(
            "member_image_ids_json must be a non-empty list: "
            f"{cluster_id}"
        )
    members = [str(value).strip() for value in values]
    if any(not value for value in members) or len(set(members)) != len(members):
        raise ValueError(
            f"Invalid or duplicate member image IDs for {cluster_id}"
        )
    return members


def _optional_member(
    row: dict[str, str],
    field: str,
    cluster_id: str,
    expected_members: set[str],
) -> str:
    value = _required_text(row, field, cluster_id)
    if value and value not in expected_members:
        raise ValueError(f"{field} is not a member of {cluster_id}: {value}")
    return value


def validate_completed_subtypes(
    prepared_dir: Path,
    subtype_path: Path,
) -> dict[str, Any]:
    """Require one complete, internally consistent row per accepted group."""
    groups = [
        row
        for row in load_jsonl(prepared_dir / "canonical_reviewed_groups.jsonl")
        if row["review_decision"] == "accepted"
    ]
    expected = {
        row["original_cluster_id"]: list(row["image_ids"])
        for row in groups
    }
    if len(expected) != CONTRACT.accepted_clusters:
        raise ValueError(
            "Prepared accepted-group count differs from the reviewed contract: "
            f"{len(expected)}"
        )

    annotations = load_subtypes(subtype_path)
    missing = sorted(set(expected) - set(annotations))
    extra = sorted(set(annotations) - set(expected))
    if missing or extra:
        raise ValueError(
            "Completed subtype sidecar does not match accepted groups; "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )

    relationship_counts: Counter[str] = Counter()
    overlay_counts: Counter[str] = Counter()
    material_count = 0

    for cluster_id, expected_member_list in sorted(expected.items()):
        row = annotations[cluster_id]
        missing_fields = sorted(REQUIRED_COMPLETED_FIELDS - set(row))
        if missing_fields:
            raise ValueError(
                f"Subtype row for {cluster_id} is missing fields: "
                f"{missing_fields}"
            )
        members = _member_ids(row, cluster_id)
        if members != expected_member_list:
            raise ValueError(
                f"member_image_ids_json differs from authority for {cluster_id}"
            )
        expected_members = set(expected_member_list)

        subtype = _required_text(row, "occlusion_subtype", cluster_id)
        if subtype == "uncertain_occlusion_subtype":
            raise ValueError(f"Unresolved occlusion subtype for {cluster_id}")
        relationship = _required_text(
            row,
            "visual_relationship_category",
            cluster_id,
        )
        if relationship not in ALLOWED_VISUAL_RELATIONSHIPS:
            raise ValueError(
                f"Unsupported visual relationship for {cluster_id}: "
                f"{relationship}"
            )
        if relationship == "uncertain_or_other":
            raise ValueError(f"Unresolved visual relationship for {cluster_id}")

        overlay = _required_text(row, "visual_overlay_category", cluster_id)
        if overlay not in ALLOWED_OVERLAY_CATEGORIES:
            raise ValueError(
                f"Unsupported visual overlay for {cluster_id}: {overlay}"
            )
        if overlay == "uncertain":
            raise ValueError(f"Unresolved visual overlay for {cluster_id}")

        material = parse_bool(
            _required_text(
                row,
                "material_occlusion_metric_included",
                cluster_id,
            )
        )
        risk = _required_text(
            row,
            "meaningful_hidden_content_risk",
            cluster_id,
        )
        if risk not in ALLOWED_HIDDEN_CONTENT_RISKS or risk == "uncertain":
            raise ValueError(
                f"Invalid hidden-content risk for {cluster_id}: {risk}"
            )
        size = _required_text(row, "occlusion_size_category", cluster_id)
        if size not in ALLOWED_OCCLUSION_SIZES or size == "uncertain":
            raise ValueError(f"Invalid occlusion size for {cluster_id}: {size}")
        difficulty = _required_text(
            row,
            "registration_difficulty",
            cluster_id,
        )
        if (
            difficulty not in ALLOWED_REGISTRATION_DIFFICULTIES
            or difficulty == "uncertain"
        ):
            raise ValueError(
                f"Invalid registration difficulty for {cluster_id}: "
                f"{difficulty}"
            )

        affected = _optional_member(
            row,
            "affected_image_id",
            cluster_id,
            expected_members,
        )
        occluded = _optional_member(
            row,
            "occluded_image_id",
            cluster_id,
            expected_members,
        )
        better = _optional_member(
            row,
            "better_view_image_id",
            cluster_id,
            expected_members,
        )
        evidence = _required_text(row, "evidence", cluster_id)
        annotator = _required_text(row, "annotator_method", cluster_id)
        if not evidence or not annotator:
            raise ValueError(
                f"Evidence and annotator method are required: {cluster_id}"
            )

        if material:
            material_count += 1
            if subtype not in {"same_occluded", "mixed_or_multi_state"}:
                raise ValueError(
                    f"Material metric row has incompatible subtype: {cluster_id}"
                )
            if relationship not in {
                "material_physical_occlusion",
                "mixed_or_multi_state",
            }:
                raise ValueError(
                    "Material metric row has incompatible relationship: "
                    f"{cluster_id}"
                )
            if not affected or not occluded or not better:
                raise ValueError(
                    "Material occlusion requires affected, occluded, and "
                    f"better IDs: {cluster_id}"
                )
            if occluded == better:
                raise ValueError(
                    f"Occluded and better image IDs must differ: {cluster_id}"
                )
            if risk == "none" or size == "none":
                raise ValueError(
                    "Material occlusion requires non-none risk and size: "
                    f"{cluster_id}"
                )
        else:
            if subtype == "same_occluded":
                raise ValueError(
                    "same_occluded subtype requires inclusion in the material "
                    f"occlusion metric: {cluster_id}"
                )
            if relationship == "material_physical_occlusion":
                raise ValueError(
                    "Physical occlusion was excluded from its required metric: "
                    f"{cluster_id}"
                )
            if subtype == "same_clean" and (risk != "none" or size != "none"):
                raise ValueError(
                    f"same_clean row must use none risk and size: {cluster_id}"
                )
            if occluded:
                raise ValueError(
                    f"Non-material row must not set occluded_image_id: {cluster_id}"
                )

        if relationship == "identical_or_near_identical":
            if subtype != "same_clean" or material or overlay != "none":
                raise ValueError(
                    f"Identical relationship has contradictory fields: {cluster_id}"
                )
        if relationship == "visual_only_overlay":
            if material or overlay == "none" or not affected:
                raise ValueError(
                    f"Visual-only overlay row is incomplete: {cluster_id}"
                )

        relationship_counts[relationship] += 1
        overlay_counts[overlay] += 1

    return {
        "accepted_group_count": len(annotations),
        "material_occlusion_metric_group_count": material_count,
        "visual_relationship_counts": dict(sorted(relationship_counts.items())),
        "visual_overlay_counts": dict(sorted(overlay_counts.items())),
        "complete": True,
    }


__all__ = [
    "ALLOWED_HIDDEN_CONTENT_RISKS",
    "ALLOWED_OCCLUSION_SIZES",
    "ALLOWED_OVERLAY_CATEGORIES",
    "ALLOWED_REGISTRATION_DIFFICULTIES",
    "ALLOWED_VISUAL_RELATIONSHIPS",
    "REQUIRED_COMPLETED_FIELDS",
    "validate_completed_subtypes",
]
