"""Acceptance decisions for content-aware pair scoring."""

from __future__ import annotations

import numpy as np

from image_clustering.clustering.config import ClusterConfig
from image_clustering.clustering.content import ContentMetrics
from image_clustering.clustering.models import Registration


def _registration_support(
    registration: Registration,
    feature_overlap_threshold: float,
    config: ClusterConfig,
) -> bool:
    if registration.fallback_used:
        return registration.alignment_score >= config.ecc_min_correlation
    return registration.feature_overlap >= feature_overlap_threshold


def _near_duplicate(
    registration: Registration,
    change: dict[str, float],
    content: ContentMetrics,
    config: ClusterConfig,
) -> bool:
    return (
        _registration_support(
            registration,
            config.duplicate_min_feature_overlap,
            config,
        )
        and change["changed_fraction"] <= config.duplicate_max_changed_fraction
        and content.unmatched_ink_fraction
        <= config.duplicate_max_unmatched_ink_fraction
        and content.unmatched_ink_union_fraction
        <= config.duplicate_max_unmatched_ink_union_fraction
        and content.ink_mismatch_tiles_fraction
        <= config.duplicate_max_ink_mismatch_tiles_fraction
    )


def _looks_like_full_page_text_replacement(
    content: ContentMetrics,
    config: ClusterConfig,
) -> bool:
    """Detect broad filled-text replacement misread as a whole-page sheet.

    At lower working resolutions, handwriting and printed values can blur into broad
    smooth residuals. The full-page shortcut must not treat that as a physical sheet
    unless the block has stronger material contrast or denser inside replacement.
    """
    page_support = (
        content.occlusion_area_fraction
        * max(content.page_count, 1)
        / max(content.occlusion_candidate_count, 1)
    )
    return (
        content.full_page_occlusion_count > 0
        and page_support >= 0.75
        and content.ink_mismatch_tiles_fraction
        >= config.occlusion_full_page_text_min_ink_mismatch_tiles_fraction
        and content.occlusion_material_median
        < config.occlusion_full_page_text_max_material_median
        and content.inside_unmatched_ink_union_fraction
        < config.occlusion_full_page_text_max_inside_unmatched_ink_union_fraction
    )


def _localized_text_erasure(
    content: ContentMetrics,
    config: ClusterConfig,
) -> bool:
    """Detect a near-background sheet from localized text loss.

    Some paper overlays are nearly the same brightness as the underlying page. They
    erase a coherent block of print and handwriting but contribute little smooth
    grayscale material contrast. This path requires that one block capture nearly
    all mismatch and that the registered exterior remain clean. Page-wide different
    records remain excluded by the full-page text-replacement veto.
    """
    page_support = (
        content.occlusion_area_fraction
        * max(content.page_count, 1)
        / max(content.occlusion_candidate_count, 1)
    )
    return (
        not _looks_like_full_page_text_replacement(content, config)
        and content.occlusion_ink_mismatch_capture
        >= max(config.occlusion_evidence_full_ink_mismatch_capture, 0.85)
        and content.occlusion_localization_contrast
        >= config.occlusion_evidence_full_localization_contrast
        and content.outside_unmatched_ink_union_fraction
        <= config.occlusion_clean_max_outside_unmatched_ink_union_fraction
        and content.outside_ink_mismatch_tiles_fraction
        <= config.occlusion_geometric_max_outside_ink_mismatch_tiles_fraction
        and page_support >= config.occlusion_geometric_min_page_support_fraction
        and content.occlusion_residual_capture
        >= config.occlusion_geometric_min_residual_capture
        and content.occlusion_rectangularity
        >= config.occlusion_min_support_fill_fraction
    )


def _looks_like_different_filled_record(
    content: ContentMetrics,
    config: ClusterConfig,
) -> bool:
    """Detect a preserved form populated with different record-specific text."""
    overwhelming_replacement = (
        content.unmatched_ink_union_fraction
        >= config.occlusion_different_record_min_unmatched_ink_union_fraction
        and content.ink_mismatch_tiles_fraction
        >= config.occlusion_different_record_min_ink_mismatch_tiles_fraction
        and content.occlusion_material_median
        < config.occlusion_different_record_max_material_median
    )
    thin_text_replacement = (
        content.unmatched_ink_union_fraction
        >= config.contradiction_text_min_unmatched_ink_union_fraction
        and content.ink_mismatch_tiles_fraction
        >= config.contradiction_text_min_ink_mismatch_tiles_fraction
        and content.inside_unmatched_ink_union_fraction
        < config.occlusion_evidence_min_inside_unmatched_ink_union_fraction
        and content.occlusion_material_median < 0.006
        and not _localized_text_erasure(content, config)
    )
    distributed_outside_replacement = (
        content.outside_unmatched_ink_union_fraction
        >= config.occlusion_evidence_distributed_outside_union_fraction
        and content.outside_ink_mismatch_tiles_fraction
        >= config.occlusion_evidence_distributed_outside_tiles_fraction
        and content.occlusion_ink_mismatch_capture < 0.60
    )
    return (
        overwhelming_replacement
        or thin_text_replacement
        or distributed_outside_replacement
        or _looks_like_full_page_text_replacement(content, config)
    )


def _physical_occlusion(
    registration: Registration,
    change: dict[str, float],
    content: ContentMetrics,
    config: ClusterConfig,
) -> bool:
    """Return whether a large contiguous physical overlay explains the pair."""
    if not _registration_support(
        registration,
        config.occlusion_min_feature_overlap,
        config,
    ):
        return False
    if not 1 <= content.occlusion_candidate_count <= 2:
        return False
    if _looks_like_different_filled_record(content, config):
        return False
    if (
        content.occlusion_ink_mismatch_capture
        < config.occlusion_evidence_min_ink_mismatch_capture
    ):
        return False
    if (
        content.occlusion_localization_contrast
        < config.occlusion_evidence_min_localization_contrast
    ):
        return False

    localized_text_erasure = _localized_text_erasure(content, config)
    block_replacement = (
        content.inside_unmatched_ink_union_fraction
        >= config.occlusion_evidence_min_inside_unmatched_ink_union_fraction
        or content.occlusion_material_median >= 0.006
        or localized_text_erasure
    )
    if not block_replacement:
        return False

    changed_fraction = change["changed_fraction"]
    strong_change = changed_fraction >= config.occlusion_strong_changed_fraction
    extreme_change = (
        changed_fraction >= config.occlusion_extreme_changed_fraction
        and content.occlusion_residual_capture
        >= config.occlusion_extreme_min_residual_capture
        and content.occlusion_material_median
        >= config.occlusion_extreme_min_material_median
    )

    page_support_fraction = (
        content.occlusion_area_fraction
        * content.page_count
        / max(content.occlusion_candidate_count, 1)
    )
    minimum_support = (
        config.occlusion_strong_min_page_support_fraction
        if strong_change
        else config.occlusion_min_page_support_fraction
    )
    if (
        page_support_fraction < minimum_support
        or content.occlusion_rectangularity
        < config.occlusion_min_support_fill_fraction
        or content.occlusion_residual_capture < config.occlusion_min_residual_capture
    ):
        return False

    if extreme_change:
        maximum_outside_union = (
            config.occlusion_extreme_max_outside_unmatched_ink_union_fraction
        )
        maximum_outside_tiles = (
            config.occlusion_extreme_max_outside_ink_mismatch_tiles_fraction
        )
    elif strong_change:
        maximum_outside_union = (
            config.occlusion_strong_max_outside_unmatched_ink_union_fraction
        )
        maximum_outside_tiles = (
            config.occlusion_strong_max_outside_ink_mismatch_tiles_fraction
        )
    else:
        maximum_outside_union = (
            config.occlusion_max_outside_unmatched_ink_union_fraction
        )
        maximum_outside_tiles = (
            config.occlusion_max_outside_ink_mismatch_tiles_fraction
        )
    if (
        content.outside_unmatched_ink_union_fraction > maximum_outside_union
        or content.outside_ink_mismatch_tiles_fraction > maximum_outside_tiles
    ):
        return False

    material_change = (
        changed_fraction >= config.occlusion_material_changed_fraction
        and content.occlusion_material_median
        >= config.occlusion_min_material_median
    )
    clean_large_change = (
        changed_fraction >= config.occlusion_clean_changed_fraction
        and content.outside_unmatched_ink_union_fraction
        <= config.occlusion_clean_max_outside_unmatched_ink_union_fraction
        and page_support_fraction
        >= config.occlusion_clean_min_page_support_fraction
    )
    geometric_material_change = (
        page_support_fraction
        >= config.occlusion_geometric_min_page_support_fraction
        and content.occlusion_residual_capture
        >= config.occlusion_geometric_min_residual_capture
        and content.occlusion_material_median
        >= config.occlusion_geometric_min_material_median
        and content.outside_unmatched_ink_union_fraction
        <= config.occlusion_geometric_max_outside_unmatched_ink_union_fraction
        and content.outside_ink_mismatch_tiles_fraction
        <= config.occlusion_geometric_max_outside_ink_mismatch_tiles_fraction
    )
    return (
        strong_change
        or material_change
        or clean_large_change
        or geometric_material_change
        or extreme_change
        or localized_text_erasure
    )


def _hard_contradiction(
    accepted: bool,
    content: ContentMetrics,
    config: ClusterConfig,
) -> bool:
    """Return whether distributed content mismatch should block graph bridging."""
    if accepted:
        return False

    different_filled_record = _looks_like_different_filled_record(content, config)
    distributed_text_replacement = (
        content.unmatched_ink_union_fraction
        >= config.contradiction_text_min_unmatched_ink_union_fraction
        and content.ink_mismatch_tiles_fraction
        >= config.contradiction_text_min_ink_mismatch_tiles_fraction
        and content.occlusion_material_median
        <= config.contradiction_text_max_material_median
    )
    plausible_multi_occlusion = (
        content.occlusion_candidate_count == 2
        and content.occlusion_material_fraction >= 0.50
        and content.outside_unmatched_ink_union_fraction <= 0.04
        and content.outside_ink_mismatch_tiles_fraction <= 0.15
    )
    distributed_ink = (
        content.ink_mismatch_tiles_fraction
        >= config.contradiction_min_ink_mismatch_tiles_fraction
        and content.unmatched_ink_union_fraction
        >= config.contradiction_min_unmatched_ink_union_fraction
    )
    distributed_exterior_residual = (
        content.residual_tiles_changed_fraction
        >= config.contradiction_min_residual_tiles_changed_fraction
        and content.outside_unmatched_ink_union_fraction
        >= config.contradiction_min_outside_unmatched_ink_union_fraction
        and content.outside_ink_mismatch_tiles_fraction
        >= config.contradiction_min_outside_ink_mismatch_tiles_fraction
    )
    overwhelming_ink_disagreement = (
        content.ink_mismatch_tiles_fraction
        >= config.contradiction_overwhelming_ink_tiles_fraction
        and content.unmatched_ink_union_fraction
        >= config.contradiction_overwhelming_unmatched_ink_union_fraction
        and content.outside_unmatched_ink_union_fraction
        >= config.contradiction_overwhelming_outside_ink_union_fraction
    )
    return different_filled_record or distributed_text_replacement or (
        not plausible_multi_occlusion
        and (
            distributed_ink
            or distributed_exterior_residual
            or overwhelming_ink_disagreement
        )
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
    if _looks_like_different_filled_record(content, config):
        return (
            False,
            None,
            "same form structure but page-wide document-specific ink replacement",
        )
    if _physical_occlusion(registration, change, content, config):
        return (
            True,
            "physical_occlusion",
            "large contiguous physical occlusion with near-exact outside agreement",
        )
    if not _registration_support(
        registration,
        config.occlusion_min_feature_overlap,
        config,
    ):
        return False, None, "too little document-specific registration support"
    if content.ink_mismatch_tiles_fraction >= 0.20:
        return False, None, "distributed coherent ink disagreement"
    if content.occlusion_candidate_count == 0:
        return (
            False,
            None,
            "no large contiguous physical occlusion explains the disagreement",
        )
    if (
        content.outside_unmatched_ink_union_fraction
        > config.occlusion_max_outside_unmatched_ink_union_fraction
    ):
        return (
            False,
            None,
            "document-specific ink disagrees outside candidate occlusion",
        )
    if _hard_contradiction(False, content, config):
        return (
            False,
            None,
            "distributed document-specific ink or exterior disagreement",
        )
    return (
        False,
        None,
        "occlusion support, density, or outside agreement was insufficient",
    )


def _confidence(
    accepted: bool,
    branch: str | None,
    registration: Registration,
    content: ContentMetrics,
) -> float:
    if registration.fallback_used:
        support = float(
            np.clip((registration.alignment_score - 0.30) / 0.50, 0, 1)
        )
    else:
        support = float(np.clip((registration.feature_overlap - 0.06) / 0.24, 0, 1))
    if branch == "near_duplicate":
        ink = float(np.clip(1.0 - content.unmatched_ink_union_fraction / 0.02, 0, 1))
        score = 0.45 * support + 0.55 * ink
    elif branch == "physical_occlusion":
        outside = float(
            np.clip(1.0 - content.outside_unmatched_ink_union_fraction / 0.08, 0, 1)
        )
        capture = float(np.clip(content.occlusion_residual_capture / 0.70, 0, 1))
        score = 0.30 * support + 0.40 * outside + 0.30 * capture
    else:
        disagreement = float(np.clip(content.ink_mismatch_tiles_fraction / 0.40, 0, 1))
        score = 0.35 * support + 0.15 * (1.0 - disagreement)
    return float(max(score, 0.50) if accepted else min(score, 0.49))
