"""Acceptance decisions for content-aware pair scoring."""

from __future__ import annotations

import numpy as np

from image_clustering.clustering.config import ClusterConfig
from image_clustering.clustering.content import ContentMetrics
from image_clustering.clustering.models import Registration


def _near_duplicate(
    registration: Registration,
    change: dict[str, float],
    content: ContentMetrics,
    config: ClusterConfig,
) -> bool:
    return (
        registration.feature_overlap >= config.duplicate_min_feature_overlap
        and change["changed_fraction"] <= config.duplicate_max_changed_fraction
        and content.unmatched_ink_fraction
        <= config.duplicate_max_unmatched_ink_fraction
        and content.unmatched_ink_union_fraction
        <= config.duplicate_max_unmatched_ink_union_fraction
        and content.ink_mismatch_tiles_fraction
        <= config.duplicate_max_ink_mismatch_tiles_fraction
    )


def _physical_occlusion(
    registration: Registration,
    content: ContentMetrics,
    config: ClusterConfig,
) -> bool:
    if registration.feature_overlap < config.occlusion_min_feature_overlap:
        return False
    if not 1 <= content.occlusion_candidate_count <= 2:
        return False
    maximum_area = (
        min(config.occlusion_max_area_fraction, 0.70)
        if content.page_count == 1
        else config.occlusion_max_area_fraction
    )
    large_clean_geometry = (
        content.page_count == 2
        and content.occlusion_area_fraction
        <= config.occlusion_large_clean_max_area_fraction
        and content.unmatched_ink_union_fraction
        <= config.occlusion_large_clean_max_unmatched_ink_union_fraction
        and content.outside_unmatched_ink_fraction <= 0.005
        and content.outside_unmatched_ink_union_fraction <= 0.02
        and content.outside_ink_mismatch_tiles_fraction <= 0.05
    )
    strong_material_boundary = (
        content.occlusion_residual_capture
        >= config.occlusion_strong_boundary_min_capture
        and content.occlusion_material_fraction >= 0.50
        and content.occlusion_boundary_score >= 1.50
    )
    if not (
        config.occlusion_min_area_fraction <= content.occlusion_area_fraction
        and (content.occlusion_area_fraction <= maximum_area or large_clean_geometry)
    ):
        return False
    if (
        content.outside_unmatched_ink_fraction
        > config.occlusion_max_outside_unmatched_ink_fraction
        or content.outside_unmatched_ink_union_fraction
        > config.occlusion_max_outside_unmatched_ink_union_fraction
        or content.outside_ink_mismatch_tiles_fraction
        > (
            0.45
            if strong_material_boundary
            else (
                min(config.occlusion_max_outside_ink_mismatch_tiles_fraction, 0.20)
                if content.page_count == 1
                else config.occlusion_max_outside_ink_mismatch_tiles_fraction
            )
        )
    ):
        return False

    clean_outside_explanation = (
        content.occlusion_residual_capture >= config.occlusion_min_residual_capture
        and content.outside_unmatched_ink_union_fraction <= 0.04
        and content.outside_unmatched_ink_fraction <= 0.01
        and content.outside_ink_mismatch_tiles_fraction <= 0.15
        and (
            content.ink_mismatch_tiles_fraction <= 0.20
            or content.unmatched_ink_union_fraction <= 0.06
        )
    )
    material_explanation = (
        content.occlusion_residual_capture >= config.occlusion_min_residual_capture
        and content.occlusion_material_fraction
        >= config.occlusion_min_material_fraction
    )
    standard_explanation = (
        content.occlusion_residual_capture >= config.occlusion_min_residual_capture
        and content.occlusion_boundary_score >= config.occlusion_min_boundary_score
    )
    strong_boundary_explanation = (
        content.occlusion_residual_capture
        >= config.occlusion_strong_boundary_min_capture
        and content.occlusion_boundary_score >= config.occlusion_strong_boundary_score
    )
    full_page_explanation = (
        content.full_page_occlusion_count >= 1
        and content.occlusion_residual_capture
        >= config.occlusion_strong_boundary_min_capture
    )
    return (
        strong_material_boundary
        or large_clean_geometry
        or clean_outside_explanation
        or material_explanation
        or standard_explanation
        or strong_boundary_explanation
        or full_page_explanation
    )


def _hard_contradiction(
    accepted: bool,
    content: ContentMetrics,
    config: ClusterConfig,
) -> bool:
    """Return whether distributed content mismatch should block graph bridging."""
    plausible_multi_occlusion = (
        content.occlusion_candidate_count >= 1
        and content.occlusion_material_fraction >= 0.50
    )
    distributed_ink_and_residual = (
        content.ink_mismatch_tiles_fraction
        >= config.contradiction_min_ink_mismatch_tiles_fraction
        and content.residual_tiles_changed_fraction
        >= config.contradiction_min_residual_tiles_changed_fraction
        and content.unmatched_ink_union_fraction
        >= config.contradiction_min_unmatched_ink_union_fraction
    )
    overwhelming_ink_disagreement = (
        content.ink_mismatch_tiles_fraction
        >= config.contradiction_overwhelming_ink_tiles_fraction
        and content.unmatched_ink_union_fraction
        >= config.contradiction_overwhelming_unmatched_ink_union_fraction
        and content.outside_unmatched_ink_union_fraction
        >= config.contradiction_overwhelming_outside_ink_union_fraction
    )
    return (
        not accepted
        and not plausible_multi_occlusion
        and (distributed_ink_and_residual or overwhelming_ink_disagreement)
    )


def _decision(
    registration: Registration,
    change: dict[str, float],
    content: ContentMetrics,
    config: ClusterConfig,
) -> tuple[bool, str | None, str]:
    if change["valid_fraction"] < config.min_valid_fraction:
        return False, None, "insufficient valid overlap after registration"
    if _near_duplicate(registration, change, content, config):
        return True, "near_duplicate", "near-exact document-specific ink agreement"
    if _physical_occlusion(registration, content, config):
        return (
            True,
            "physical_occlusion",
            "coherent physical occlusion with near-exact outside ink agreement",
        )
    if registration.feature_overlap < config.occlusion_min_feature_overlap:
        return False, None, "too little document-specific exact feature support"
    if content.ink_mismatch_tiles_fraction >= 0.20:
        return False, None, "distributed coherent ink disagreement"
    if content.occlusion_candidate_count == 0:
        return False, None, "no physical occlusion explains the disagreement"
    if (
        content.outside_unmatched_ink_union_fraction
        > config.occlusion_max_outside_unmatched_ink_union_fraction
    ):
        return (
            False,
            None,
            "document-specific ink disagrees outside candidate occlusion",
        )
    return False, None, "occlusion geometry or outside agreement was insufficient"


def _confidence(
    accepted: bool,
    branch: str | None,
    registration: Registration,
    content: ContentMetrics,
) -> float:
    support = float(np.clip((registration.feature_overlap - 0.06) / 0.24, 0, 1))
    if branch == "near_duplicate":
        ink = float(np.clip(1.0 - content.unmatched_ink_union_fraction / 0.02, 0, 1))
        score = 0.45 * support + 0.55 * ink
    elif branch == "physical_occlusion":
        outside = float(
            np.clip(1.0 - content.outside_unmatched_ink_union_fraction / 0.115, 0, 1)
        )
        capture = float(np.clip(content.occlusion_residual_capture / 0.70, 0, 1))
        score = 0.30 * support + 0.40 * outside + 0.30 * capture
    else:
        disagreement = float(np.clip(content.ink_mismatch_tiles_fraction / 0.40, 0, 1))
        score = 0.35 * support + 0.15 * (1.0 - disagreement)
    return float(max(score, 0.50) if accepted else min(score, 0.49))
