"""Coverage safeguards for crop-empty clusters and broad foreground sheets.

This module installs a small, explicit integration layer around the existing
sequence cropper. It keeps the core selection code focused while enforcing two
important product contracts:

1. every accepted content-bearing cluster yields at least one recognizer input,
   including literal near duplicates with no residual change candidate;
2. broad, tall foreground sheets are cropped to their physical extent rather
   than only to the active text band.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from .config import Config
from .documents import (
    absolute_detail_score,
    content_score,
    detect_gutter,
    page_boxes,
    page_frontness_score,
)
from .documents import (
    expand_to_sheet_boundary as _base_expand_to_sheet_boundary,
)
from .image_ops import invert_transform, transform_bbox
from .models import BBox, ImageRecord, Submission, WorkImage


def pad_tall_sheet(
    bbox: BBox,
    page_bbox: BBox,
    config: Config,
) -> BBox:
    """Add vertical context to a broad/tall likely foreground sheet."""
    px0, py0, px1, py1 = page_bbox
    page_width = max(px1 - px0, 1)
    page_height = max(py1 - py0, 1)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    min_width = float(config.get("sheet_expansion.tall_sheet_min_width_fraction", 0.68))
    min_height = float(
        config.get("sheet_expansion.tall_sheet_min_height_fraction", 0.45)
    )
    if width < min_width * page_width or height < min_height * page_height:
        return bbox
    top_padding = round(
        float(config.get("sheet_expansion.tall_sheet_top_padding_fraction", 0.12))
        * page_height
    )
    bottom_padding = round(
        float(config.get("sheet_expansion.tall_sheet_bottom_padding_fraction", 0.08))
        * page_height
    )
    return (
        bbox[0],
        max(py0, bbox[1] - top_padding),
        bbox[2],
        min(py1, bbox[3] + bottom_padding),
    )


def expand_to_sheet_boundary(
    image_gray: np.ndarray,
    support_bbox: BBox,
    page_bbox: BBox,
    config: Config,
) -> BBox:
    """Expand residual support and then preserve full tall-sheet context."""
    expanded = _base_expand_to_sheet_boundary(
        image_gray=image_gray,
        support_bbox=support_bbox,
        page_bbox=page_bbox,
        config=config,
    )
    return pad_tall_sheet(expanded, page_bbox, config)


def _working_page_candidates(
    page_boxes_anchor: list[tuple[str, BBox]],
    transforms: dict[int, np.ndarray],
    work: dict[int, WorkImage],
) -> list[tuple[float, int, str, BBox, float, float, float]]:
    candidates: list[tuple[float, int, str, BBox, float, float, float]] = []
    for side, page_anchor in page_boxes_anchor:
        for image_index, transform in transforms.items():
            page_image = transform_bbox(
                page_anchor,
                invert_transform(transform),
                work[image_index].gray.shape,
            )
            image = work[image_index].gray
            content = content_score(image, page_image)
            detail = absolute_detail_score(image, page_image)
            frontness = page_frontness_score(image, page_image)
            quality = 0.45 * frontness + 0.35 * content + 0.20 * min(detail, 1.0)
            candidates.append(
                (
                    quality,
                    image_index,
                    side,
                    page_image,
                    content,
                    detail,
                    frontness,
                )
            )
    return candidates


def fallback_page_submissions(
    *,
    group_id: str,
    anchor: int,
    transforms: dict[int, np.ndarray],
    work: dict[int, WorkImage],
    original_shapes: dict[int, tuple[int, int]],
    records_by_index: dict[int, ImageRecord],
    config: Config,
) -> list[Submission]:
    """Return recognizer pages when normal selection produced nothing.

    Clearly content-bearing page sides are emitted once. A nonblank but
    low-frontness fallback is sent to the review queue rather than disappearing.
    """
    anchor_gray = work[anchor].gray
    pages = page_boxes(anchor_gray.shape, detect_gutter(anchor_gray, config), config)
    candidates = _working_page_candidates(pages, transforms, work)
    if not candidates:
        return []

    min_content = float(config.get("selection.fallback_min_content_score", 0.045))
    min_detail = float(config.get("selection.fallback_min_detail_score", 0.020))
    min_frontness = float(config.get("selection.fallback_min_page_frontness", 0.30))
    normal_frontness = float(config.get("selection.min_page_frontness", 0.58))

    best_by_side: dict[str, tuple[float, int, str, BBox, float, float, float]] = {}
    for candidate in candidates:
        side = candidate[2]
        if side not in best_by_side or candidate[0] > best_by_side[side][0]:
            best_by_side[side] = candidate

    chosen = [
        candidate
        for candidate in best_by_side.values()
        if candidate[4] >= min_content
        and candidate[5] >= min_detail
        and candidate[6] >= min_frontness
    ]
    if not chosen:
        best = max(candidates)
        if best[4] < 0.5 * min_content and best[5] < 0.5 * min_detail:
            return []
        chosen = [best]

    submissions: list[Submission] = []
    for _, image_index, side, bbox_work, content, _detail, frontness in chosen:
        scale = work[image_index].scale
        height, width = original_shapes[image_index]
        x0, y0, x1, y1 = bbox_work
        bbox = (
            max(0, round(x0 / scale)),
            max(0, round(y0 / scale)),
            min(width, round(x1 / scale)),
            min(height, round(y1 / scale)),
        )
        complete = frontness >= normal_frontness
        submissions.append(
            Submission(
                submission_id="",
                group_id=group_id,
                image_index=image_index,
                source_path=records_by_index[image_index].path,
                kind="base_page",
                side=side,
                bbox=bbox,
                completeness="complete" if complete else "review_required",
                confidence=float(
                    np.clip(0.30 + 0.35 * frontness + 0.20 * content, 0.0, 1.0)
                ),
                content_score=content,
                occlusion_fraction=0.0,
                reason=(
                    "Fallback page coverage for a crop-empty accepted cluster; "
                    "near duplicates still require recognizer input"
                ),
            )
        )
    return submissions


def _with_overlay_coverage_config(config: Config) -> Config:
    threshold = float(config.get("selection.page_state_min_page_coverage", 0.68))
    if threshold >= 0.85:
        return config
    data = deepcopy(config.data)
    data.setdefault("selection", {})["page_state_min_page_coverage"] = 0.85
    return Config(data)


def install_coverage_guards() -> None:
    """Install the safeguards exactly once into the cropper orchestration."""
    from . import pipeline

    if getattr(pipeline.SequencePipeline, "_coverage_guards_installed", False):
        return

    original_select = pipeline.SequencePipeline._select_submissions
    original_unique = pipeline.choose_unique_region_states

    def select_with_fallback(
        self: Any,
        group_id: str,
        group: list[int],
        anchor: int,
        transforms: dict[int, np.ndarray],
        candidates: list[Any],
        work: dict[int, WorkImage],
        original_shapes: dict[int, tuple[int, int]],
        records_by_index: dict[int, ImageRecord],
    ) -> list[Submission]:
        submissions = original_select(
            self,
            group_id,
            group,
            anchor,
            transforms,
            candidates,
            work,
            original_shapes,
            records_by_index,
        )
        if submissions:
            return submissions
        return fallback_page_submissions(
            group_id=group_id,
            anchor=anchor,
            transforms=transforms,
            work=work,
            original_shapes=original_shapes,
            records_by_index=records_by_index,
            config=self.config,
        )

    def unique_with_overlay_coverage(*args: Any, **kwargs: Any) -> list[Submission]:
        if "config" in kwargs:
            kwargs["config"] = _with_overlay_coverage_config(kwargs["config"])
        elif args:
            values = list(args)
            values[-1] = _with_overlay_coverage_config(values[-1])
            args = tuple(values)
        return original_unique(*args, **kwargs)

    pipeline.expand_to_sheet_boundary = expand_to_sheet_boundary
    pipeline.choose_unique_region_states = unique_with_overlay_coverage
    pipeline.SequencePipeline._select_submissions = select_with_fallback
    pipeline.SequencePipeline._coverage_guards_installed = True
