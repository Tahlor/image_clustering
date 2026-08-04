"""Sequence-aware ranking for recall-first occlusion review candidates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from image_clustering.clustering.config import ClusterConfig
from image_clustering.clustering.content import ContentMetrics
from image_clustering.clustering.models import ClusteringResult, PairComparison
from image_clustering.clustering.scoring_decision import _hard_contradiction


@dataclass(frozen=True)
class OcclusionReviewCandidate:
    """One pair ranked for human or stronger-model review, never auto-linked."""

    first_image_id: str
    second_image_id: str
    sequence_id: str
    index_gap: int
    review_tier: int
    priority_reason: str
    same_occluded_probability: float
    same_document_probability: float
    occluded_given_same_probability: float
    deterministic_same_document: bool
    automatic_link_eligible: bool
    hard_contradiction: bool
    raw_hard_contradiction: bool
    acceptance_conflict: bool
    same_component: bool
    common_accepted_neighbors: tuple[str, ...]
    registration_fallback_used: bool
    registration_alignment_score: float
    decision_reason: str
    raw_occluded_given_same_probability: float = 0.0
    occlusion_evidence: float = 0.0
    inside_unmatched_ink_union_fraction: float = 0.0
    occlusion_ink_mismatch_capture: float = 0.0
    occlusion_localization_contrast: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert the candidate to a JSON-serializable row."""
        value = asdict(self)
        value["common_accepted_neighbors"] = list(self.common_accepted_neighbors)
        return value


def _cluster_lookup(result: ClusteringResult) -> dict[str, str]:
    return {
        image_id: cluster.cluster_id
        for cluster in result.clusters
        for image_id in cluster.image_ids
    }


def _accepted_neighbors(
    result: ClusteringResult,
) -> dict[str, set[str]]:
    neighbors = {image.image_id: set() for image in result.images}
    for comparison in result.comparisons:
        if not comparison.same_document:
            continue
        neighbors.setdefault(comparison.first_image_id, set()).add(
            comparison.second_image_id
        )
        neighbors.setdefault(comparison.second_image_id, set()).add(
            comparison.first_image_id
        )
    return neighbors


def _content_metrics(comparison: PairComparison) -> ContentMetrics:
    return ContentMetrics(
        unmatched_ink_fraction=comparison.unmatched_ink_fraction,
        unmatched_ink_union_fraction=comparison.unmatched_ink_union_fraction,
        ink_mismatch_tiles_fraction=comparison.ink_mismatch_tiles_fraction,
        coherent_ink_component_count=comparison.coherent_ink_component_count,
        largest_ink_component_fraction=comparison.largest_ink_component_fraction,
        residual_tiles_changed_fraction=comparison.residual_tiles_changed_fraction,
        occlusion_candidate_count=comparison.occlusion_candidate_count,
        occlusion_area_fraction=comparison.occlusion_area_fraction,
        occlusion_residual_capture=comparison.occlusion_residual_capture,
        occlusion_rectangularity=comparison.occlusion_rectangularity,
        occlusion_boundary_score=comparison.occlusion_boundary_score,
        occlusion_material_fraction=comparison.occlusion_material_fraction,
        occlusion_material_median=comparison.occlusion_material_median,
        outside_unmatched_ink_fraction=(
            comparison.outside_unmatched_ink_fraction
        ),
        outside_unmatched_ink_union_fraction=(
            comparison.outside_unmatched_ink_union_fraction
        ),
        outside_ink_mismatch_tiles_fraction=(
            comparison.outside_ink_mismatch_tiles_fraction
        ),
        full_page_occlusion_count=comparison.full_page_occlusion_count,
        shallow_occlusion_count=comparison.shallow_occlusion_count,
        page_count=comparison.page_count,
        inside_unmatched_ink_union_fraction=(
            comparison.inside_unmatched_ink_union_fraction
        ),
        occlusion_ink_mismatch_capture=(
            comparison.occlusion_ink_mismatch_capture
        ),
        occlusion_localization_contrast=(
            comparison.occlusion_localization_contrast
        ),
    )


def _raw_hard_contradiction(
    comparison: PairComparison,
    config: ClusterConfig,
) -> bool:
    """Evaluate contradiction evidence without suppressing accepted pairs."""
    content_metrics_present = (
        comparison.unmatched_ink_union_fraction < 1.0
        or comparison.ink_mismatch_tiles_fraction < 1.0
        or comparison.residual_tiles_changed_fraction < 1.0
        or comparison.occlusion_candidate_count > 0
    )
    if not content_metrics_present:
        return False
    return _hard_contradiction(
        accepted=False,
        content=_content_metrics(comparison),
        config=config,
    )


def _tier(
    comparison: PairComparison,
    *,
    acceptance_conflict: bool,
    same_component: bool,
    common_neighbors: tuple[str, ...],
) -> tuple[int, str]:
    if acceptance_conflict:
        return 0, "accepted edge conflicts with raw contradiction evidence"
    if comparison.same_document:
        return 5, "deterministically accepted without a contradiction conflict"
    if comparison.hard_contradiction:
        return 4, "hard contradiction; expert review only"
    if common_neighbors:
        return 1, "common accepted neighbor supports a sequence bridge"
    if same_component:
        return 2, "already connected transitively by conservative edges"
    return 3, "high-scoring rejected pair without graph support"


def rank_occlusion_candidates(
    result: ClusteringResult,
    *,
    config: ClusterConfig | None = None,
    include_accepted: bool = False,
    include_unflagged: bool = False,
) -> tuple[OcclusionReviewCandidate, ...]:
    """Rank candidate pairs using safe sequence context without changing clusters.

    The function never creates an edge. It merely places candidate-scored pairs into
    interpretable tiers. Common-neighbor and same-component evidence can prioritize a
    review. Accepted edges whose raw ink evidence is contradictory are always surfaced
    as audit conflicts, but their operational decision is not changed.
    """
    config = config or ClusterConfig()
    cluster_by_image = _cluster_lookup(result)
    accepted_neighbors = _accepted_neighbors(result)
    candidates: list[OcclusionReviewCandidate] = []
    for comparison in result.comparisons:
        if (
            not comparison.same_document
            and not include_unflagged
            and not comparison.occlusion_candidate_flag
        ):
            continue
        raw_contradiction = _raw_hard_contradiction(comparison, config)
        acceptance_conflict = comparison.same_document and raw_contradiction
        if (
            not include_accepted
            and comparison.same_document
            and not acceptance_conflict
        ):
            continue
        first_cluster = cluster_by_image.get(comparison.first_image_id)
        second_cluster = cluster_by_image.get(comparison.second_image_id)
        same_component = (
            first_cluster is not None
            and second_cluster is not None
            and first_cluster == second_cluster
        )
        common_neighbors = tuple(
            sorted(
                accepted_neighbors.get(comparison.first_image_id, set())
                & accepted_neighbors.get(comparison.second_image_id, set())
            )
        )
        tier, priority_reason = _tier(
            comparison,
            acceptance_conflict=acceptance_conflict,
            same_component=same_component,
            common_neighbors=common_neighbors,
        )
        candidates.append(
            OcclusionReviewCandidate(
                first_image_id=comparison.first_image_id,
                second_image_id=comparison.second_image_id,
                sequence_id=comparison.sequence_id,
                index_gap=comparison.index_gap,
                review_tier=tier,
                priority_reason=priority_reason,
                same_occluded_probability=comparison.same_occluded_probability,
                same_document_probability=comparison.same_document_probability,
                occluded_given_same_probability=(
                    comparison.occluded_given_same_probability
                ),
                deterministic_same_document=comparison.same_document,
                automatic_link_eligible=comparison.automatic_link_eligible,
                hard_contradiction=comparison.hard_contradiction,
                raw_hard_contradiction=raw_contradiction,
                acceptance_conflict=acceptance_conflict,
                same_component=same_component,
                common_accepted_neighbors=common_neighbors,
                registration_fallback_used=(
                    comparison.registration_fallback_used
                ),
                registration_alignment_score=(
                    comparison.registration_alignment_score
                ),
                decision_reason=comparison.reason,
                raw_occluded_given_same_probability=(
                    comparison.raw_occluded_given_same_probability
                ),
                occlusion_evidence=comparison.occlusion_evidence,
                inside_unmatched_ink_union_fraction=(
                    comparison.inside_unmatched_ink_union_fraction
                ),
                occlusion_ink_mismatch_capture=(
                    comparison.occlusion_ink_mismatch_capture
                ),
                occlusion_localization_contrast=(
                    comparison.occlusion_localization_contrast
                ),
            )
        )
    candidates.sort(
        key=lambda candidate: (
            candidate.review_tier,
            -candidate.same_occluded_probability,
            candidate.index_gap,
            candidate.first_image_id,
            candidate.second_image_id,
        )
    )
    return tuple(candidates)


__all__ = ["OcclusionReviewCandidate", "rank_occlusion_candidates"]
