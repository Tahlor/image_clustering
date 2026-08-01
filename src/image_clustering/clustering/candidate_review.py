"""Sequence-aware ranking for recall-first occlusion review candidates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from image_clustering.clustering.models import ClusteringResult, PairComparison


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
    same_component: bool
    common_accepted_neighbors: tuple[str, ...]
    registration_fallback_used: bool
    registration_alignment_score: float
    decision_reason: str

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
        neighbors[comparison.first_image_id].add(comparison.second_image_id)
        neighbors[comparison.second_image_id].add(comparison.first_image_id)
    return neighbors


def _tier(
    comparison: PairComparison,
    *,
    same_component: bool,
    common_neighbors: tuple[str, ...],
) -> tuple[int, str]:
    if comparison.same_document:
        return 0, "deterministically accepted; include only for audit"
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
    include_accepted: bool = False,
    include_unflagged: bool = False,
) -> tuple[OcclusionReviewCandidate, ...]:
    """Rank candidate pairs using safe sequence context without changing clusters.

    The function never creates an edge. It merely places candidate-scored pairs into
    interpretable tiers. Common-neighbor and same-component evidence can prioritize a
    review, while a hard contradiction always moves the pair to the highest-risk tier.
    """
    cluster_by_image = _cluster_lookup(result)
    accepted_neighbors = _accepted_neighbors(result)
    candidates: list[OcclusionReviewCandidate] = []
    for comparison in result.comparisons:
        if not include_unflagged and not comparison.occlusion_candidate_flag:
            continue
        if not include_accepted and comparison.same_document:
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
                same_component=same_component,
                common_accepted_neighbors=common_neighbors,
                registration_fallback_used=(
                    comparison.registration_fallback_used
                ),
                registration_alignment_score=(
                    comparison.registration_alignment_score
                ),
                decision_reason=comparison.reason,
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
