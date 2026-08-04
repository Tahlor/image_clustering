"""Continuous pair probabilities used for recall-first occlusion review.

The compact model in this module is intentionally not an automatic-link model. It was
fit on the versioned Vermont synthetic benchmark and is used to rank pairs for review.
The conservative deterministic decision and hard-contradiction rules remain the sole
authority for graph edges.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from image_clustering.clustering.config import ClusterConfig
from image_clustering.clustering.content import ContentMetrics
from image_clustering.clustering.models import Registration

_MODEL_VERSION = "vermont-synthetic-logit-v3-full-page-text-gate"

_FEATURE_NAMES = (
    "registration_fallback",
    "registration_inlier_ratio",
    "registration_feature_overlap",
    "registration_x_span",
    "registration_y_span",
    "valid_fraction",
    "changed_fraction",
    "unmatched_ink_fraction",
    "unmatched_ink_union_fraction",
    "largest_ink_component_fraction",
    "residual_tiles_changed_fraction",
    "ink_mismatch_tiles_fraction",
    "occlusion_candidate_count",
    "occlusion_area_fraction",
    "occlusion_residual_capture",
    "occlusion_rectangularity",
    "occlusion_material_fraction",
    "occlusion_material_median",
    "outside_unmatched_ink_fraction",
    "outside_unmatched_ink_union_fraction",
    "outside_ink_mismatch_tiles_fraction",
    "visible_exterior_fraction",
)

# Development fit plus Platt calibration on the separate selection split. Runtime
# inference is a pair of tiny linear models and does not require scikit-learn.
_IDENTITY_INTERCEPT = -25.032403217297773
_IDENTITY_COEFFICIENTS = (
    -3.30523264275,
    1.54165319374,
    10.8673154655,
    3.02391782739,
    2.95748489785,
    14.318220195,
    1.3601496971,
    12.911490102,
    9.98239985807,
    -25.104974442,
    2.94313315568,
    -7.41409816239,
    -0.113294723829,
    -2.76520017095,
    0.717476905671,
    1.26170244453,
    -0.450872517669,
    0.468714776354,
    6.20489537405,
    -1.86654378948,
    -0.382983552484,
    2.76520017095,
)
_OCCLUSION_INTERCEPT = -40.3619545391732
_OCCLUSION_COEFFICIENTS = (
    -0.0862595982891,
    4.63165727698,
    -39.7763158242,
    5.40568144254,
    -3.80712561427,
    29.3484218594,
    12.8020717636,
    112.588110255,
    30.6888764169,
    -367.37276427,
    -4.59443003663,
    41.4537202459,
    2.20572211637,
    1.35998083064,
    2.2674292895,
    5.08228494261,
    7.48353954155,
    3.74987617885,
    68.8749943246,
    -10.1167134239,
    0.103324514267,
    -1.35998083064,
)


@dataclass(frozen=True)
class PairProbabilities:
    """Coherent three-state probabilities plus action-oriented diagnostics."""

    same_document: float
    occluded_given_same: float
    same_clean: float
    same_occluded: float
    different_document: float
    candidate_flag: bool
    automatic_link_eligible: bool
    raw_occluded_given_same: float
    occlusion_evidence: float
    model_version: str = _MODEL_VERSION


def _sigmoid(value: float) -> float:
    value = max(-40.0, min(40.0, value))
    return 1.0 / (1.0 + math.exp(-value))


def _linear_probability(
    intercept: float,
    coefficients: tuple[float, ...],
    values: tuple[float, ...],
) -> float:
    if len(coefficients) != len(values):
        raise ValueError("Probability feature vector has the wrong length")
    terms = zip(coefficients, values, strict=True)
    linear_value = sum(
        coefficient * value for coefficient, value in terms
    )
    return _sigmoid(intercept + linear_value)


def _ramp(value: float, floor: float, full: float) -> float:
    if full <= floor:
        return float(value >= full)
    return float(max(0.0, min(1.0, (value - floor) / (full - floor))))


def _full_page_text_replacement(
    content: ContentMetrics,
    config: ClusterConfig,
) -> bool:
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


def _occlusion_evidence(
    content: ContentMetrics,
    config: ClusterConfig,
) -> float:
    """Measure whether one physical block explains the text-channel mismatch.

    The synthetic classifier learned that broad disagreement often accompanies an
    occlusion. Real same-template/different-record pages violate that shortcut: the
    form registers well while handwriting changes throughout the page. This gate
    requires a contiguous candidate to capture the mismatch and requires the text
    channel outside the candidate to remain substantially stable.
    """
    if not 1 <= content.occlusion_candidate_count <= 2:
        return 0.0

    page_support = (
        content.occlusion_area_fraction
        * max(content.page_count, 1)
        / max(content.occlusion_candidate_count, 1)
    )
    support = _ramp(
        page_support,
        config.occlusion_min_page_support_fraction * 0.45,
        config.occlusion_min_page_support_fraction,
    )
    residual_capture = _ramp(
        content.occlusion_residual_capture,
        config.occlusion_min_residual_capture * 0.45,
        min(1.0, config.occlusion_min_residual_capture * 1.75),
    )
    ink_capture = _ramp(
        content.occlusion_ink_mismatch_capture,
        config.occlusion_evidence_min_ink_mismatch_capture,
        config.occlusion_evidence_full_ink_mismatch_capture,
    )
    localization_contrast = _ramp(
        content.occlusion_localization_contrast,
        config.occlusion_evidence_min_localization_contrast,
        config.occlusion_evidence_full_localization_contrast,
    )
    inside_disagreement = _ramp(
        content.inside_unmatched_ink_union_fraction,
        config.occlusion_evidence_min_inside_unmatched_ink_union_fraction,
        config.occlusion_evidence_full_inside_unmatched_ink_union_fraction,
    )
    shape = _ramp(
        content.occlusion_rectangularity,
        config.occlusion_min_support_fill_fraction * 0.50,
        max(0.60, config.occlusion_min_support_fill_fraction * 2.5),
    )
    material = _ramp(
        content.occlusion_material_median,
        0.006,
        max(0.04, config.occlusion_min_material_median * 2.0),
    )

    outside_union = 1.0 - _ramp(
        content.outside_unmatched_ink_union_fraction,
        config.occlusion_clean_max_outside_unmatched_ink_union_fraction,
        config.occlusion_extreme_max_outside_unmatched_ink_union_fraction,
    )
    outside_tiles = 1.0 - _ramp(
        content.outside_ink_mismatch_tiles_fraction,
        config.occlusion_geometric_max_outside_ink_mismatch_tiles_fraction,
        config.occlusion_extreme_max_outside_ink_mismatch_tiles_fraction,
    )
    exterior_agreement = min(outside_union, outside_tiles)

    # A full-page insert has little or no exterior to score. Material change is
    # then the only acceptable substitute; this does not rescue ordinary filled
    # forms because their broad difference is mostly thin text, not a sheet.
    if content.full_page_occlusion_count and page_support >= 0.70:
        exterior_agreement = max(exterior_agreement, 0.50 * material)

    localization = (
        0.55 * ink_capture
        + 0.25 * localization_contrast
        + 0.20 * inside_disagreement
    )
    block_replacement = max(inside_disagreement, material)
    evidence = (
        0.12 * support
        + 0.22 * residual_capture
        + 0.22 * localization
        + 0.18 * exterior_agreement
        + 0.08 * shape
        + 0.18 * block_replacement
    )

    distributed_replacement = (
        content.outside_unmatched_ink_union_fraction
        >= config.occlusion_evidence_distributed_outside_union_fraction
        and content.outside_ink_mismatch_tiles_fraction
        >= config.occlusion_evidence_distributed_outside_tiles_fraction
    )
    weak_localization = (
        content.occlusion_ink_mismatch_capture
        < config.occlusion_evidence_min_ink_mismatch_capture
        and content.occlusion_residual_capture
        < config.occlusion_min_residual_capture
    )
    thin_text_only = (
        content.inside_unmatched_ink_union_fraction
        < config.occlusion_evidence_min_inside_unmatched_ink_union_fraction
        and content.occlusion_material_median < 0.006
    )
    if (
        distributed_replacement
        or weak_localization
        or thin_text_only
        or _full_page_text_replacement(content, config)
    ):
        evidence *= config.occlusion_evidence_distributed_penalty
    return float(max(0.0, min(1.0, evidence)))


def _feature_values(
    registration: Registration,
    change: dict[str, float],
    content: ContentMetrics,
) -> tuple[float, ...]:
    visible_exterior = max(0.0, 1.0 - content.occlusion_area_fraction)
    values = (
        float(registration.fallback_used),
        float(registration.inlier_ratio),
        float(registration.feature_overlap),
        float(registration.x_span),
        float(registration.y_span),
        float(change["valid_fraction"]),
        float(change["changed_fraction"]),
        float(content.unmatched_ink_fraction),
        float(content.unmatched_ink_union_fraction),
        float(content.largest_ink_component_fraction),
        float(content.residual_tiles_changed_fraction),
        float(content.ink_mismatch_tiles_fraction),
        float(content.occlusion_candidate_count),
        float(content.occlusion_area_fraction),
        float(content.occlusion_residual_capture),
        float(content.occlusion_rectangularity),
        float(content.occlusion_material_fraction),
        float(content.occlusion_material_median),
        float(content.outside_unmatched_ink_fraction),
        float(content.outside_unmatched_ink_union_fraction),
        float(content.outside_ink_mismatch_tiles_fraction),
        visible_exterior,
    )
    if not all(math.isfinite(value) for value in values):
        return tuple(0.0 if not math.isfinite(value) else value for value in values)
    return values


def pair_probabilities(
    registration: Registration,
    change: dict[str, float],
    content: ContentMetrics,
    *,
    accepted: bool,
    hard_contradiction: bool,
    candidate_threshold: float,
    config: ClusterConfig | None = None,
) -> PairProbabilities:
    """Return synthetic-calibrated probabilities and safe action flags.

    ``candidate_flag`` is deliberately recall-oriented. ``automatic_link_eligible``
    never follows the probability directly; it requires the existing conservative
    deterministic acceptance and the absence of a hard contradiction.
    """
    values = _feature_values(
        registration=registration,
        change=change,
        content=content,
    )
    p_same = _linear_probability(
        _IDENTITY_INTERCEPT,
        _IDENTITY_COEFFICIENTS,
        values,
    )
    raw_occluded_given_same = _linear_probability(
        _OCCLUSION_INTERCEPT,
        _OCCLUSION_COEFFICIENTS,
        values,
    )
    evidence_config = config or ClusterConfig()
    occlusion_evidence = _occlusion_evidence(content, evidence_config)
    p_occluded_given_same = raw_occluded_given_same * occlusion_evidence
    p_same_occluded = p_same * p_occluded_given_same
    p_same_clean = p_same * (1.0 - p_occluded_given_same)
    return PairProbabilities(
        same_document=p_same,
        occluded_given_same=p_occluded_given_same,
        same_clean=p_same_clean,
        same_occluded=p_same_occluded,
        different_document=1.0 - p_same,
        candidate_flag=p_same_occluded >= candidate_threshold,
        automatic_link_eligible=accepted and not hard_contradiction,
        raw_occluded_given_same=raw_occluded_given_same,
        occlusion_evidence=occlusion_evidence,
    )


__all__ = ["PairProbabilities", "pair_probabilities"]
