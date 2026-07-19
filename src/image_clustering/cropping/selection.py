from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import Config
from .documents import (
    absolute_detail_score,
    boundary_score,
    content_score,
    page_frontness_score,
)
from .image_ops import (
    bbox_area,
    bbox_intersection,
    correlation,
    fingerprint,
    invert_transform,
    transform_bbox,
)
from .models import BBox, Submission

State = tuple[float, int, BBox, np.ndarray, float, float]


def _dedupe_state_crops(states: list[State], threshold: float) -> list[State]:
    kept: list[State] = []
    for state in sorted(states, key=lambda item: (item[5], item[4]), reverse=True):
        if any(correlation(state[3], prior[3]) >= threshold for prior in kept):
            continue
        kept.append(state)
    return kept


def _candidate_geometry(candidate: BBox, page: BBox) -> tuple[float, float, float]:
    page_width = max(page[2] - page[0], 1)
    page_height = max(page[3] - page[1], 1)
    width_ratio = (candidate[2] - candidate[0]) / page_width
    height_ratio = (candidate[3] - candidate[1]) / page_height
    coverage = bbox_area(bbox_intersection(candidate, page)) / max(
        bbox_area(page),
        1,
    )
    return width_ratio, height_ratio, coverage


def choose_canonical_pages(
    group_id: str,
    page_boxes_anchor: list[tuple[str, BBox]],
    candidate_boxes_by_side: dict[str, list[BBox]],
    transforms_to_anchor: dict[int, np.ndarray],
    image_shapes: dict[int, tuple[int, int]],
    images_gray: dict[int, np.ndarray],
    config: Config,
) -> tuple[list[Submission], dict[str, int]]:
    """Choose the best front-facing observation of each persistent page."""
    submissions: list[Submission] = []
    canonical_image_by_side: dict[str, int] = {}
    complete_threshold = float(
        config.get("selection.complete_max_occlusion_fraction", 0.025)
    )
    partial_threshold = float(
        config.get("selection.partial_max_occlusion_fraction", 0.55)
    )
    min_frontness = float(config.get("selection.min_page_frontness", 0.58))
    frontness_weight = float(
        config.get("selection.canonical_frontness_weight", 0.20)
    )
    detail_weight = float(config.get("selection.canonical_detail_weight", 0.80))
    visibility_ratio = float(
        config.get("selection.canonical_visibility_ratio", 0.90)
    )
    shallow_height = float(
        config.get("selection.shallow_band_max_height_fraction", 0.27)
    )
    shallow_width = float(
        config.get("selection.shallow_band_min_width_fraction", 0.52)
    )
    shallow_visibility_weight = float(
        config.get("selection.shallow_visibility_weight", 0.35)
    )
    min_region_coverage = float(
        config.get("selection.min_unique_region_page_coverage", 0.25)
    )

    for side, page_anchor in page_boxes_anchor:
        observations: list[tuple[int, BBox, float, float, float]] = []
        for image_index, transform in transforms_to_anchor.items():
            page_image = transform_bbox(
                page_anchor,
                invert_transform(transform),
                image_shapes[image_index],
            )
            observations.append(
                (
                    image_index,
                    page_image,
                    content_score(images_gray[image_index], page_image),
                    absolute_detail_score(images_gray[image_index], page_image),
                    page_frontness_score(images_gray[image_index], page_image),
                )
            )
        if not observations:
            continue

        max_detail = max(item[3] for item in observations)
        ranked: list[tuple[float, int, BBox, float, float, float]] = []
        for image_index, page_image, content, detail, frontness in observations:
            relative_detail = detail / max(max_detail, 1e-8)
            quality = frontness_weight * frontness + detail_weight * relative_detail
            height, width = image_shapes[image_index]
            if (
                page_image[0] <= 1
                or page_image[1] <= 1
                or page_image[2] >= width - 1
                or page_image[3] >= height - 1
            ):
                quality -= 0.025
            ranked.append(
                (quality, image_index, page_image, content, detail, frontness)
            )

        _, image_index, page_image, content, detail, frontness = max(ranked)
        canonical_image_by_side[side] = image_index
        if frontness < min_frontness:
            continue

        possible_occlusion_area = 0.0
        for candidate_anchor in candidate_boxes_by_side.get(side, []):
            width_ratio, height_ratio, coverage = _candidate_geometry(
                candidate_anchor,
                page_anchor,
            )
            state_details: list[tuple[int, float]] = []
            for state_index, transform in transforms_to_anchor.items():
                candidate_image = transform_bbox(
                    candidate_anchor,
                    invert_transform(transform),
                    image_shapes[state_index],
                )
                state_details.append(
                    (
                        state_index,
                        absolute_detail_score(
                            images_gray[state_index],
                            candidate_image,
                        ),
                    )
                )
            best_detail = max((value for _, value in state_details), default=0.0)
            worst_detail = min((value for _, value in state_details), default=0.0)
            canonical_detail = next(
                (
                    value
                    for state_index, value in state_details
                    if state_index == image_index
                ),
                0.0,
            )
            area = bbox_area(bbox_intersection(page_anchor, candidate_anchor))
            if canonical_detail < visibility_ratio * max(best_detail, 1e-8):
                possible_occlusion_area += area
                continue

            shallow_persistent = (
                width_ratio >= max(shallow_width, 0.82)
                and height_ratio <= max(shallow_height, 0.42)
                and worst_detail >= 0.55 * max(best_detail, 1e-8)
            )
            broad_persistent = (
                coverage >= min(min_region_coverage, 0.34)
                and width_ratio >= 0.70
                and worst_detail >= 0.90 * max(best_detail, 1e-8)
            )
            if shallow_persistent:
                possible_occlusion_area += shallow_visibility_weight * area
            elif broad_persistent:
                possible_occlusion_area += area

        occlusion_fraction = possible_occlusion_area / max(
            bbox_area(page_anchor),
            1,
        )
        if occlusion_fraction <= complete_threshold:
            completeness = "complete"
        elif (
            bool(config.get("selection.submit_partial_best_available", True))
            and occlusion_fraction <= partial_threshold
        ):
            completeness = "partial_best_available"
        else:
            continue

        submissions.append(
            Submission(
                submission_id="",
                group_id=group_id,
                image_index=image_index,
                source_path=Path(""),
                kind="base_page",
                side=side,
                bbox=page_image,
                completeness=completeness,
                confidence=float(
                    np.clip(
                        0.45
                        + 0.45 * frontness
                        + 0.20 * detail
                        - occlusion_fraction,
                        0.0,
                        1.0,
                    )
                ),
                content_score=content,
                occlusion_fraction=occlusion_fraction,
                reason="Best front-facing observation of the persistent page",
            )
        )
    return submissions, canonical_image_by_side


def choose_unique_region_states(
    group_id: str,
    candidates: list[tuple[str, BBox]],
    page_boxes_anchor: list[tuple[str, BBox]],
    canonical_image_by_side: dict[str, int],
    transforms_to_anchor: dict[int, np.ndarray],
    image_shapes: dict[int, tuple[int, int]],
    images_gray: dict[int, np.ndarray],
    config: Config,
) -> list[Submission]:
    """Submit each noncanonical, front-facing data-bearing state once."""
    submissions: list[Submission] = []
    threshold = float(
        config.get("selection.duplicate_correlation_threshold", 0.985)
    )
    min_content = float(config.get("selection.min_content_score", 0.055))
    min_boundary = float(config.get("selection.min_boundary_score", 1.35))
    min_frontness = float(config.get("selection.min_region_frontness", 0.72))
    page_state_coverage = float(
        config.get("selection.page_state_min_page_coverage", 0.68)
    )
    blank_ratio = float(
        config.get("selection.blank_relative_detail_ratio", 0.67)
    )
    blank_content_ratio = float(
        config.get("selection.blank_relative_content_ratio", 0.65)
    )
    blank_frontness_ratio = float(
        config.get("selection.blank_relative_frontness_ratio", 0.72)
    )
    blank_boundary_override = float(
        config.get("selection.blank_boundary_override", 1.50)
    )
    min_region_coverage = float(
        config.get("selection.min_unique_region_page_coverage", 0.25)
    )
    min_state_change = float(
        config.get("selection.min_unique_state_change_fraction", 0.05)
    )
    page_lookup = dict(page_boxes_anchor)

    for candidate_number, (side, bbox_anchor) in enumerate(candidates, start=1):
        page_anchor = page_lookup[side]
        coverage = bbox_area(bbox_intersection(bbox_anchor, page_anchor)) / max(
            bbox_area(page_anchor),
            1,
        )
        if coverage < min_region_coverage:
            continue
        states: list[State] = []
        for image_index, transform in transforms_to_anchor.items():
            bbox_image = transform_bbox(
                bbox_anchor,
                invert_transform(transform),
                image_shapes[image_index],
            )
            crop = images_gray[image_index][
                bbox_image[1] : bbox_image[3],
                bbox_image[0] : bbox_image[2],
            ]
            if crop.size == 0:
                continue
            states.append(
                (
                    content_score(images_gray[image_index], bbox_image),
                    image_index,
                    bbox_image,
                    fingerprint(crop, 64),
                    absolute_detail_score(images_gray[image_index], bbox_image),
                    page_frontness_score(images_gray[image_index], bbox_image),
                )
            )
        unique_states = _dedupe_state_crops(states, threshold)
        canonical_index = canonical_image_by_side.get(side)
        max_detail = max((state[4] for state in states), default=0.0)
        max_content = max((state[0] for state in states), default=0.0)
        max_frontness = max((state[5] for state in states), default=0.0)
        canonical_state = next(
            (state for state in states if state[1] == canonical_index),
            None,
        )

        for state_number, state in enumerate(unique_states, start=1):
            score, image_index, bbox_image, _, detail, frontness = state
            if image_index == canonical_index:
                continue
            if score < min_content or frontness < min_frontness:
                continue
            edge_score = boundary_score(images_gray[image_index], bbox_image)
            if canonical_state is not None:
                canonical_score = canonical_state[0]
                canonical_detail = canonical_state[4]
                canonical_frontness = canonical_state[5]
                relative_change = max(
                    abs(detail - canonical_detail) / max(max_detail, 1e-8),
                    abs(score - canonical_score) / max(max_content, 1e-8),
                    abs(frontness - canonical_frontness)
                    / max(max_frontness, 1e-8),
                )
                if relative_change < min_state_change:
                    continue
            if (
                detail < blank_ratio * max(max_detail, 1e-8)
                and score < blank_content_ratio * max(max_content, 1e-8)
                and frontness
                < blank_frontness_ratio * max(max_frontness, 1e-8)
                and edge_score < blank_boundary_override
            ):
                continue

            kind = "data_bearing_overlay"
            if coverage >= page_state_coverage and edge_score < min_boundary:
                kind = "page_state"
            submissions.append(
                Submission(
                    submission_id="",
                    group_id=group_id,
                    image_index=image_index,
                    source_path=Path(""),
                    kind=kind,
                    side=(
                        f"{side}_change_{candidate_number}_state_{state_number}"
                    ),
                    bbox=bbox_image,
                    completeness="complete",
                    confidence=float(
                        np.clip(
                            0.30
                            + 0.55 * frontness
                            + 0.25 * min(detail, 1.0),
                            0.0,
                            1.0,
                        )
                    ),
                    content_score=score,
                    occlusion_fraction=0.0,
                    reason=(
                        "Unique front-facing content state not present in the "
                        "canonical page"
                    ),
                )
            )
    return submissions
