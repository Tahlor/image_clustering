"""Pairwise same-physical-document scoring."""

from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np

from image_clustering.clustering.candidate_scoring import pair_probabilities
from image_clustering.clustering.config import ClusterConfig
from image_clustering.clustering.content import (
    ContentMetrics,
    analyze_content,
    local_dissimilarity,
)
from image_clustering.clustering.content_geometry import _tile_bounds
from image_clustering.clustering.models import (
    ImageFeatures,
    PairComparison,
    Registration,
)
from image_clustering.clustering.registration import (
    register_pair,
    source_pixel_transform,
    warp_current,
)
from image_clustering.clustering.scoring_decision import (
    _confidence,
    _decision,
    _hard_contradiction,
)

_FILENAME_SUFFIX = re.compile(r"^(?P<prefix>.*?)(?P<number>\d+)$")


def _filename_sequence_position(path: Path) -> tuple[str, int] | None:
    """Return the stable nonnumeric prefix and terminal numeric capture index."""
    match = _FILENAME_SUFFIX.match(path.stem)
    if match is None or not match.group("prefix"):
        return None
    return match.group("prefix"), int(match.group("number"))


def _automatic_link_safety_reason(
    previous: ImageFeatures,
    current: ImageFeatures,
    registration: Registration,
    content: ContentMetrics,
    branch: str | None,
    config: ClusterConfig,
) -> str | None:
    """Return why a deterministic match must remain review-only.

    These checks do not alter the continuous candidate score. They only prevent weak
    or operationally implausible matches from becoming graph edges.
    """
    previous_position = _filename_sequence_position(previous.image.path)
    current_position = _filename_sequence_position(current.image.path)
    if previous_position is not None and current_position is not None:
        previous_prefix, previous_number = previous_position
        current_prefix, current_number = current_position
        if (
            config.automatic_link_require_same_filename_prefix
            and previous_prefix != current_prefix
        ):
            return "filename sequence prefixes differ"
        if (
            previous_prefix == current_prefix
            and abs(previous_number - current_number)
            > config.automatic_link_max_numeric_filename_gap
        ):
            return "filename capture positions are too far apart for an automatic edge"

    if (
        not config.automatic_link_allow_full_page_ecc
        and registration.fallback_used
        and content.full_page_occlusion_count > 0
    ):
        return "full-page ECC match requires review"

    if (
        branch == "physical_occlusion"
        and (
            registration.fallback_used
            or registration.feature_overlap
            < config.automatic_link_min_physical_occlusion_feature_overlap
        )
    ):
        return (
            "physical-occlusion match lacks strong document-specific feature "
            "overlap"
        )

    dirty_exterior = (
        branch == "physical_occlusion"
        and content.outside_unmatched_ink_union_fraction
        >= config.occlusion_dirty_exterior_min_unmatched_ink_union_fraction
        and content.outside_ink_mismatch_tiles_fraction
        >= config.occlusion_dirty_exterior_min_ink_mismatch_tiles_fraction
    )
    strong_identity_support = (
        registration.feature_overlap
        >= config.occlusion_dirty_exterior_min_feature_overlap
        or registration.alignment_score
        >= config.occlusion_dirty_exterior_min_alignment_score
    )
    if dirty_exterior and not strong_identity_support:
        return "dirty occlusion exterior lacks strong registration support"
    return None


def _fit_exposure(values: np.ndarray, targets: np.ndarray) -> tuple[float, float]:
    """Return a bounded robust linear map from aligned to reference intensity."""
    variance = float(np.var(values))
    covariance = float(np.cov(values, targets, bias=True)[0, 1])
    scale = float(np.clip(covariance / max(variance, 1e-6), 0.75, 1.35))
    offset = float(np.median(targets - scale * values))
    return scale, offset


def _normalize_brightness(
    reference: np.ndarray,
    aligned: np.ndarray,
    valid_mask: np.ndarray,
    config: ClusterConfig | None = None,
) -> np.ndarray:
    """Normalize exposure from stable textured regions, not the possible occluder.

    A large dark or light sheet can dominate whole-image median/MAD statistics and
    manufacture disagreement in the truly unchanged exterior. We first locate the
    lowest-residual tiles after an additive centering pass, then fit a bounded robust
    linear exposure map using only textured, non-saturated pixels in those tiles.
    """
    config = config or ClusterConfig()
    core = cv2.erode(valid_mask, np.ones((9, 9), np.uint8)) > 0
    if int(core.sum()) < 1000:
        return aligned

    reference_float = reference.astype(np.float32)
    aligned_float = aligned.astype(np.float32)
    additive_offset = float(np.median((reference_float - aligned_float)[core]))
    centered = np.clip(aligned_float + additive_offset, 0, 255)
    residual = np.abs(reference_float - centered)

    rows = max(4, config.tile_rows * 2)
    columns = max(4, config.tile_columns * 2)
    tile_scores: list[tuple[float, int, int]] = []
    for row in range(rows):
        for column in range(columns):
            x0, y0, x1, y1 = _tile_bounds(
                row,
                column,
                reference.shape,
                rows,
                columns,
            )
            tile_core = core[y0:y1, x0:x1]
            if tile_core.mean() < 0.70:
                continue
            score = float(np.median(residual[y0:y1, x0:x1][tile_core]))
            tile_scores.append((score, row, column))
    if not tile_scores:
        return centered.astype(np.uint8)

    tile_scores.sort()
    stable_count = max(
        8,
        round(config.residual_stable_fraction * len(tile_scores)),
    )
    stable = np.zeros_like(core)
    for _, row, column in tile_scores[:stable_count]:
        x0, y0, x1, y1 = _tile_bounds(
            row,
            column,
            reference.shape,
            rows,
            columns,
        )
        stable[y0:y1, x0:x1] = True
    stable &= core

    gradient_x = cv2.Scharr(reference, cv2.CV_32F, 1, 0)
    gradient_y = cv2.Scharr(reference, cv2.CV_32F, 0, 1)
    texture = cv2.magnitude(gradient_x, gradient_y)
    if int(stable.sum()) < 1000:
        return centered.astype(np.uint8)
    texture_floor = float(np.percentile(texture[stable], 55))
    textured = (
        stable
        & (texture >= texture_floor)
        & (reference > 15)
        & (reference < 248)
        & (aligned > 15)
        & (aligned < 248)
    )
    if int(textured.sum()) < 1000:
        return centered.astype(np.uint8)

    values = aligned[textured].astype(np.float64)
    targets = reference[textured].astype(np.float64)
    if values.size > 100_000:
        stride = max(1, values.size // 100_000)
        values = values[::stride]
        targets = targets[::stride]

    scale, offset = _fit_exposure(values, targets)
    fit_residual = np.abs(targets - (scale * values + offset))
    keep = fit_residual <= np.quantile(fit_residual, 0.80)
    if int(keep.sum()) >= 500:
        scale, offset = _fit_exposure(values[keep], targets[keep])

    normalized = aligned_float * scale + offset
    return np.clip(normalized, 0, 255).astype(np.uint8)


def _change_metrics(
    reference: np.ndarray,
    aligned: np.ndarray,
    valid_mask: np.ndarray,
    config: ClusterConfig,
) -> dict[str, float]:
    """Retain coarse legacy metrics for diagnostics and backward compatibility."""
    core = cv2.erode(valid_mask, np.ones((9, 9), np.uint8)) > 0
    valid_fraction = float(core.mean())
    if int(core.sum()) < 1000:
        return {
            "valid_fraction": valid_fraction,
            "changed_fraction": 1.0,
            "stable_fraction": 0.0,
            "tiles_changed_fraction": 1.0,
            "largest_change_share": 0.0,
        }
    dissimilarity = local_dissimilarity(reference=reference, aligned=aligned)
    raw_changed = (dissimilarity > config.change_threshold) & core
    changed_fraction = float(raw_changed.sum() / max(core.sum(), 1))
    stable_fraction = 1.0 - changed_fraction

    tile_changed = []
    height, width = reference.shape
    for row in range(config.tile_rows):
        for column in range(config.tile_columns):
            y0 = round(row * height / config.tile_rows)
            y1 = round((row + 1) * height / config.tile_rows)
            x0 = round(column * width / config.tile_columns)
            x1 = round((column + 1) * width / config.tile_columns)
            tile_core = core[y0:y1, x0:x1]
            if tile_core.mean() < 0.5:
                continue
            tile_fraction = float(raw_changed[y0:y1, x0:x1][tile_core].mean())
            tile_changed.append(tile_fraction > config.tile_changed_threshold)
    tiles_changed_fraction = float(np.mean(tile_changed)) if tile_changed else 1.0

    mask = raw_changed.astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    close_size = max(7, (round(min(reference.shape) * 0.018) // 2) * 2 + 1)
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        np.ones((close_size, close_size), np.uint8),
    )
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    areas = [int(stats[index, cv2.CC_STAT_AREA]) for index in range(1, count)]
    coherent_area = sum(areas)
    largest_change_share = max(areas, default=0) / max(coherent_area, 1)
    return {
        "valid_fraction": valid_fraction,
        "changed_fraction": changed_fraction,
        "stable_fraction": stable_fraction,
        "tiles_changed_fraction": tiles_changed_fraction,
        "largest_change_share": float(largest_change_share),
    }


def score_pair(
    previous: ImageFeatures,
    current: ImageFeatures,
    index_gap: int,
    config: ClusterConfig,
) -> PairComparison:
    """Score whether two captures show the same physical document scene."""
    registration = register_pair(previous=previous, current=current, config=config)
    if not registration.accepted:
        return PairComparison(
            first_image_id=previous.image.image_id,
            second_image_id=current.image.image_id,
            sequence_id=previous.image.sequence_id,
            index_gap=index_gap,
            same_document=False,
            confidence=0.0,
            reason=registration.reason or "registration rejected",
            good_match_count=registration.good_match_count,
            registration_fallback_used=registration.fallback_used,
            registration_alignment_score=registration.alignment_score,
        )
    aligned, valid_mask = warp_current(
        current_gray=current.gray,
        registration=registration,
        previous_shape=previous.gray.shape,
    )
    aligned = _normalize_brightness(
        reference=previous.gray,
        aligned=aligned,
        valid_mask=valid_mask,
        config=config,
    )
    change = _change_metrics(
        reference=previous.gray,
        aligned=aligned,
        valid_mask=valid_mask,
        config=config,
    )
    content = analyze_content(
        reference=previous.gray,
        aligned=aligned,
        valid_mask=valid_mask,
        config=config,
    )
    deterministic_accepted, branch, deterministic_reason = _decision(
        registration=registration,
        change=change,
        content=content,
        config=config,
    )
    safety_reason = (
        _automatic_link_safety_reason(
            previous=previous,
            current=current,
            registration=registration,
            content=content,
            branch=branch,
            config=config,
        )
        if deterministic_accepted
        else None
    )
    accepted = deterministic_accepted and safety_reason is None
    reason = (
        deterministic_reason
        if safety_reason is None
        else f"review-only deterministic match: {safety_reason}"
    )
    contradiction = _hard_contradiction(
        accepted=deterministic_accepted,
        content=content,
        config=config,
    )
    confidence = _confidence(
        accepted=accepted,
        branch=branch,
        registration=registration,
        content=content,
    )
    probabilities = pair_probabilities(
        registration=registration,
        change=change,
        content=content,
        accepted=accepted,
        hard_contradiction=contradiction,
        candidate_threshold=config.occlusion_candidate_probability_threshold,
        config=config,
    )
    return PairComparison(
        first_image_id=previous.image.image_id,
        second_image_id=current.image.image_id,
        sequence_id=previous.image.sequence_id,
        index_gap=index_gap,
        same_document=accepted,
        confidence=confidence,
        reason=reason,
        registration_model=registration.model,
        transform=source_pixel_transform(
            registration=registration,
            previous_scale=previous.scale,
            current_scale=current.scale,
        ),
        good_match_count=registration.good_match_count,
        inlier_count=registration.inlier_count,
        inlier_ratio=registration.inlier_ratio,
        feature_overlap=registration.feature_overlap,
        median_reprojection_error=registration.median_reprojection_error,
        registration_fallback_used=registration.fallback_used,
        registration_alignment_score=registration.alignment_score,
        valid_fraction=change["valid_fraction"],
        changed_fraction=change["changed_fraction"],
        stable_fraction=change["stable_fraction"],
        tiles_changed_fraction=change["tiles_changed_fraction"],
        largest_change_share=change["largest_change_share"],
        unmatched_ink_fraction=content.unmatched_ink_fraction,
        unmatched_ink_union_fraction=content.unmatched_ink_union_fraction,
        ink_mismatch_tiles_fraction=content.ink_mismatch_tiles_fraction,
        coherent_ink_component_count=content.coherent_ink_component_count,
        largest_ink_component_fraction=content.largest_ink_component_fraction,
        residual_tiles_changed_fraction=content.residual_tiles_changed_fraction,
        occlusion_candidate_count=content.occlusion_candidate_count,
        occlusion_area_fraction=content.occlusion_area_fraction,
        occlusion_residual_capture=content.occlusion_residual_capture,
        occlusion_rectangularity=content.occlusion_rectangularity,
        occlusion_boundary_score=content.occlusion_boundary_score,
        occlusion_material_fraction=content.occlusion_material_fraction,
        occlusion_material_median=content.occlusion_material_median,
        outside_unmatched_ink_fraction=content.outside_unmatched_ink_fraction,
        outside_unmatched_ink_union_fraction=(
            content.outside_unmatched_ink_union_fraction
        ),
        outside_ink_mismatch_tiles_fraction=(
            content.outside_ink_mismatch_tiles_fraction
        ),
        inside_unmatched_ink_union_fraction=(
            content.inside_unmatched_ink_union_fraction
        ),
        occlusion_ink_mismatch_capture=content.occlusion_ink_mismatch_capture,
        occlusion_localization_contrast=content.occlusion_localization_contrast,
        full_page_occlusion_count=content.full_page_occlusion_count,
        shallow_occlusion_count=content.shallow_occlusion_count,
        page_count=content.page_count,
        hard_contradiction=contradiction,
        branch=branch,
        probability_model_version=probabilities.model_version,
        same_document_probability=probabilities.same_document,
        occluded_given_same_probability=probabilities.occluded_given_same,
        raw_occluded_given_same_probability=(
            probabilities.raw_occluded_given_same
        ),
        occlusion_evidence=probabilities.occlusion_evidence,
        same_clean_probability=probabilities.same_clean,
        same_occluded_probability=probabilities.same_occluded,
        different_document_probability=probabilities.different_document,
        occlusion_candidate_flag=probabilities.candidate_flag,
        automatic_link_eligible=probabilities.automatic_link_eligible,
    )
