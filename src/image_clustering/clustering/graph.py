"""Conservative graph construction for complete physical-document clusters."""

from __future__ import annotations

from image_clustering.clustering.models import ImageCluster, PairComparison


def _pair_key(first: str, second: str) -> tuple[str, str]:
    return tuple(sorted((first, second)))


def _is_hard_contradiction(comparison: PairComparison) -> bool:
    """Return whether a rejected pair blocks a transitive component merge.

    Registration failure is not a contradiction because two heavily occluded
    views may share no direct visible region. A well-registered pair with
    distributed document-specific disagreement is a hard negative.
    """
    if comparison.hard_contradiction:
        return True
    content_metrics_present = (
        comparison.unmatched_ink_union_fraction < 1.0
        or comparison.ink_mismatch_tiles_fraction < 1.0
        or comparison.residual_tiles_changed_fraction < 1.0
        or comparison.occlusion_candidate_count > 0
    )
    if content_metrics_present:
        return False
    return (
        not comparison.same_document
        and comparison.registration_model is not None
        and comparison.valid_fraction >= 0.65
        and comparison.feature_overlap >= 0.06
        and comparison.changed_fraction >= 0.65
    )


def _components(
    image_ids: list[str],
    comparisons: list[PairComparison],
    max_cluster_size: int | None = None,
) -> list[list[str]]:
    parent = {image_id: image_id for image_id in image_ids}
    members = {image_id: {image_id} for image_id in image_ids}
    lookup = {
        _pair_key(comparison.first_image_id, comparison.second_image_id): comparison
        for comparison in comparisons
    }

    def find(image_id: str) -> str:
        while parent[image_id] != image_id:
            parent[image_id] = parent[parent[image_id]]
            image_id = parent[image_id]
        return image_id

    def union(first: str, second: str) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root == second_root:
            return
        if (
            max_cluster_size is not None
            and len(members[first_root]) + len(members[second_root])
            > max_cluster_size
        ):
            return
        if len(members[first_root]) < len(members[second_root]):
            first_root, second_root = second_root, first_root
        parent[second_root] = first_root
        members[first_root].update(members.pop(second_root))

    accepted = sorted(
        (comparison for comparison in comparisons if comparison.same_document),
        key=lambda comparison: comparison.confidence,
        reverse=True,
    )
    for comparison in accepted:
        first_root = find(comparison.first_image_id)
        second_root = find(comparison.second_image_id)
        if first_root == second_root:
            continue
        contradiction = any(
            _is_hard_contradiction(lookup[key])
            for first_member in members[first_root]
            for second_member in members[second_root]
            if (key := _pair_key(first_member, second_member)) in lookup
        )
        if not contradiction:
            union(comparison.first_image_id, comparison.second_image_id)

    grouped: dict[str, list[str]] = {}
    for image_id in image_ids:
        grouped.setdefault(find(image_id), []).append(image_id)
    return list(grouped.values())


def _representative(
    component: list[str],
    comparisons: list[PairComparison],
) -> str:
    weighted_degree = {image_id: 0.0 for image_id in component}
    component_set = set(component)
    for comparison in comparisons:
        if (
            comparison.same_document
            and comparison.first_image_id in component_set
            and comparison.second_image_id in component_set
        ):
            weighted_degree[comparison.first_image_id] += comparison.confidence
            weighted_degree[comparison.second_image_id] += comparison.confidence
    return max(
        component,
        key=lambda image_id: (
            weighted_degree[image_id],
            -component.index(image_id),
        ),
    )


def build_clusters(
    sequence_id: str,
    image_ids: list[str],
    comparisons: list[PairComparison],
    cluster_id_start: int = 1,
    max_cluster_size: int | None = None,
) -> list[ImageCluster]:
    """Convert accepted pair edges into complete conservative components."""
    order = {image_id: index for index, image_id in enumerate(image_ids)}
    clusters = []
    for offset, component in enumerate(
        _components(
            image_ids=image_ids,
            comparisons=comparisons,
            max_cluster_size=max_cluster_size,
        )
    ):
        component.sort(key=order.__getitem__)
        clusters.append(
            ImageCluster(
                cluster_id=f"cluster_{cluster_id_start + offset:05d}",
                sequence_id=sequence_id,
                image_ids=tuple(component),
                representative_image_id=_representative(
                    component=component,
                    comparisons=comparisons,
                ),
            )
        )
    return clusters
