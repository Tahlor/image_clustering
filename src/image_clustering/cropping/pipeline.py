from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import Config
from .diagnostics import annotate, residual_heatmap
from .documents import detect_gutter, expand_to_sheet_boundary, page_boxes
from .grouping import connected_groups, select_anchor, transforms_to_anchor
from .image_ops import (
    bbox_area,
    bbox_intersection,
    correlation,
    fingerprint,
    gray,
)
from .io_utils import crop, read_color, resize_max, write_image
from .models import ChangeCandidate, ImageRecord, PairEdge, Submission, WorkImage
from .registration import detect_features, register_pair, warp_to_reference
from .residuals import component_rectangles, compute_residual_grid
from .selection import choose_canonical_pages, choose_unique_region_states

LOGGER = logging.getLogger(__name__)


class SequencePipeline:
    """Recover unique page states from one independent image sequence."""

    def __init__(self, config: Config, output_dir: Path) -> None:
        self.config = config
        self.output_dir = output_dir
        self.quality = int(config.get("io.jpeg_quality", 95))

    def _load_work_images(
        self,
        records: list[ImageRecord],
    ) -> tuple[dict[int, WorkImage], dict[int, tuple[int, int]]]:
        work: dict[int, WorkImage] = {}
        original_shapes: dict[int, tuple[int, int]] = {}
        max_dimension = int(
            self.config.get("scale.registration_max_dimension", 1500)
        )
        fingerprint_size = int(self.config.get("scale.fingerprint_size", 96))
        for record in records:
            original = read_color(record.path)
            resized, scale = resize_max(original, max_dimension)
            gray_image = gray(resized)
            work[record.index] = WorkImage(
                record=record,
                gray=gray_image,
                scale=scale,
                fingerprint=fingerprint(gray_image, fingerprint_size),
            )
            original_shapes[record.index] = original.shape[:2]
        return work, original_shapes

    def _build_edges(
        self,
        records: list[ImageRecord],
        work: dict[int, WorkImage],
    ) -> list[PairEdge]:
        features = {
            index: detect_features(item.gray, self.config)
            for index, item in work.items()
        }
        edges: list[PairEdge] = []
        window = int(self.config.get("pair_search.window", 3))
        minimum_correlation = float(
            self.config.get("pair_search.min_fingerprint_correlation", 0.20)
        )
        for position, reference_record in enumerate(records):
            reference = work[reference_record.index]
            for offset in range(1, window + 1):
                if position + offset >= len(records):
                    break
                moving_record = records[position + offset]
                moving = work[moving_record.index]
                fingerprint_correlation = correlation(
                    reference.fingerprint,
                    moving.fingerprint,
                )
                if fingerprint_correlation < minimum_correlation:
                    continue
                registration = register_pair(
                    reference.gray,
                    moving.gray,
                    features[reference.record.index],
                    features[moving.record.index],
                    fingerprint_correlation,
                    self.config,
                )
                if not registration.accepted:
                    continue
                edges.append(
                    PairEdge(
                        a=reference.record.index,
                        b=moving.record.index,
                        registration_b_to_a=registration,
                    )
                )
                LOGGER.info(
                    "crop edge %s -> %s model=%s residual=%.3f",
                    reference.record.name,
                    moving.record.name,
                    registration.model,
                    registration.stable_residual,
                )
        return edges

    def _group_candidates(
        self,
        group: list[int],
        anchor: int,
        transforms: dict[int, np.ndarray],
        work: dict[int, WorkImage],
        edges: list[PairEdge],
    ) -> tuple[
        list[ChangeCandidate],
        list[tuple[int, int, np.ndarray, np.ndarray]],
    ]:
        candidates: list[ChangeCandidate] = []
        diagnostics: list[tuple[int, int, np.ndarray, np.ndarray]] = []
        anchor_gray = work[anchor].gray
        gutter = detect_gutter(anchor_gray, self.config)
        pages = page_boxes(anchor_gray.shape, gutter, self.config)
        image_area = int(np.prod(anchor_gray.shape))

        for edge in edges:
            if edge.a not in group or edge.b not in group:
                continue
            if edge.a not in transforms or edge.b not in transforms:
                continue
            reference_aligned, reference_valid = warp_to_reference(
                work[edge.a].gray,
                transforms[edge.a],
                anchor_gray.shape,
            )
            moving_aligned, moving_valid = warp_to_reference(
                work[edge.b].gray,
                transforms[edge.b],
                anchor_gray.shape,
            )
            valid = cv2.bitwise_and(reference_valid, moving_valid)
            grid = compute_residual_grid(
                reference_aligned,
                moving_aligned,
                valid,
                self.config,
            )
            diagnostics.append((edge.a, edge.b, grid.scores, grid.valid))
            for page_name, page_bbox in pages:
                rectangles = component_rectangles(
                    grid,
                    self.config,
                    region_bbox=page_bbox,
                )
                for support_bbox, support in rectangles:
                    clipped = bbox_intersection(support_bbox, page_bbox)
                    if bbox_area(clipped) <= 0:
                        continue
                    expanded_reference = expand_to_sheet_boundary(
                        reference_aligned,
                        clipped,
                        page_bbox,
                        self.config,
                    )
                    expanded_moving = expand_to_sheet_boundary(
                        moving_aligned,
                        clipped,
                        page_bbox,
                        self.config,
                    )
                    expanded = (
                        min(expanded_reference[0], expanded_moving[0]),
                        min(expanded_reference[1], expanded_moving[1]),
                        max(expanded_reference[2], expanded_moving[2]),
                        max(expanded_reference[3], expanded_moving[3]),
                    )
                    area_fraction = bbox_area(expanded) / max(image_area, 1)
                    minimum_area = float(
                        self.config.get(
                            "selection.min_changed_area_fraction",
                            0.025,
                        )
                    )
                    maximum_area = float(
                        self.config.get(
                            "selection.max_changed_area_fraction",
                            0.62,
                        )
                    )
                    if not minimum_area <= area_fraction <= maximum_area:
                        continue
                    inside = grid.scores[support & grid.valid]
                    outside = grid.scores[grid.valid & ~support]
                    inside_score = (
                        float(np.percentile(inside, 65))
                        if inside.size
                        else 0.0
                    )
                    outside_score = (
                        float(np.percentile(outside, 70))
                        if outside.size
                        else 0.0
                    )
                    contrast = inside_score - outside_score
                    if contrast <= 0.025:
                        continue
                    support_fraction = float(
                        support.sum() / max(grid.valid.sum(), 1)
                    )
                    confidence = float(
                        np.clip(
                            0.35 + 2.2 * contrast + 0.7 * support_fraction,
                            0.0,
                            1.0,
                        )
                    )
                    candidates.append(
                        ChangeCandidate(
                            pair=(edge.a, edge.b),
                            side=page_name,
                            bbox_anchor=expanded,
                            area_fraction=area_fraction,
                            inside_score=inside_score,
                            outside_score=outside_score,
                            contrast=contrast,
                            support_fraction=support_fraction,
                            confidence=confidence,
                        )
                    )
        return self._merge_candidates(candidates), diagnostics

    @staticmethod
    def _merge_candidates(
        candidates: list[ChangeCandidate],
    ) -> list[ChangeCandidate]:
        merged: list[ChangeCandidate] = []
        for candidate in sorted(
            candidates,
            key=lambda item: item.confidence,
            reverse=True,
        ):
            match: int | None = None
            for index, existing in enumerate(merged):
                intersection = bbox_area(
                    bbox_intersection(
                        candidate.bbox_anchor,
                        existing.bbox_anchor,
                    )
                )
                minimum = min(
                    bbox_area(candidate.bbox_anchor),
                    bbox_area(existing.bbox_anchor),
                )
                if (
                    candidate.side == existing.side
                    and intersection / max(minimum, 1) >= 0.55
                ):
                    match = index
                    break
            if match is None:
                merged.append(candidate)
            elif candidate.confidence > merged[match].confidence:
                merged[match] = candidate
        return merged

    def _select_submissions(
        self,
        group_id: str,
        group: list[int],
        anchor: int,
        transforms: dict[int, np.ndarray],
        candidates: list[ChangeCandidate],
        work: dict[int, WorkImage],
        original_shapes: dict[int, tuple[int, int]],
        records_by_index: dict[int, ImageRecord],
    ) -> list[Submission]:
        anchor_gray = work[anchor].gray
        gutter = detect_gutter(anchor_gray, self.config)
        page_boxes_anchor = page_boxes(anchor_gray.shape, gutter, self.config)
        image_shapes_work = {
            index: work[index].gray.shape
            for index in group
            if index in transforms
        }
        images_gray_work = {
            index: work[index].gray
            for index in group
            if index in transforms
        }
        candidates_by_side: dict[str, list[tuple[int, int, int, int]]] = (
            defaultdict(list)
        )
        candidate_pairs: list[tuple[str, tuple[int, int, int, int]]] = []
        for candidate in candidates:
            candidates_by_side[candidate.side].append(candidate.bbox_anchor)
            candidate_pairs.append((candidate.side, candidate.bbox_anchor))

        page_submissions, canonical_by_side = choose_canonical_pages(
            group_id,
            page_boxes_anchor,
            candidates_by_side,
            transforms,
            image_shapes_work,
            images_gray_work,
            self.config,
        )
        unique_submissions = choose_unique_region_states(
            group_id,
            candidate_pairs,
            page_boxes_anchor,
            canonical_by_side,
            transforms,
            image_shapes_work,
            images_gray_work,
            self.config,
        )
        resolved_sides = {
            submission.side.split("_change_", 1)[0]
            for submission in unique_submissions
            if submission.kind in {"data_bearing_overlay", "page_state"}
        }
        for submission in page_submissions:
            if submission.side in resolved_sides:
                submission.completeness = "complete"
                submission.occlusion_fraction = 0.0
                submission.reason += (
                    "; alternate state was recovered as a separate foreground sheet"
                )

        submissions = page_submissions + unique_submissions
        for submission in submissions:
            record = records_by_index[submission.image_index]
            scale = work[submission.image_index].scale
            x0, y0, x1, y1 = submission.bbox
            original_shape = original_shapes[submission.image_index]
            submission.bbox = (
                max(0, round(x0 / scale)),
                max(0, round(y0 / scale)),
                min(original_shape[1], round(x1 / scale)),
                min(original_shape[0], round(y1 / scale)),
            )
            submission.source_path = record.path
        return submissions

    def process(
        self,
        folder: Path,
        records: list[ImageRecord],
        sequence_id: str | None = None,
        force_single_group: bool = False,
    ) -> dict[str, Any]:
        """Process one filename-sorted folder or one upstream cluster."""
        resolved_sequence_id = sequence_id or folder.name or "root"
        sequence_path = Path(resolved_sequence_id)
        safe_sequence_id = resolved_sequence_id.replace("/", "__").replace(
            "\\",
            "__",
        )
        if not records:
            return {
                "folder": resolved_sequence_id,
                "source_folder": str(folder),
                "groups": [],
                "submissions": [],
            }
        work, original_shapes = self._load_work_images(records)
        records_by_index = {record.index: record for record in records}
        edges = self._build_edges(records, work)
        indices = [record.index for record in records]
        groups = (
            [indices]
            if force_single_group
            else connected_groups(indices, edges)
        )
        all_submissions: list[Submission] = []
        group_payloads: list[dict[str, Any]] = []
        diagnostics_root = self.output_dir / "diagnostics" / sequence_path

        for group_number, group in enumerate(groups, start=1):
            group_id = f"{safe_sequence_id}__group_{group_number:05d}"
            group_edges = [
                edge for edge in edges if edge.a in group and edge.b in group
            ]
            anchor = select_anchor(group, group_edges) if group_edges else group[0]
            transforms, _ = transforms_to_anchor(group, anchor, group_edges)
            if len(group) == 1:
                transforms = {anchor: np.eye(3, dtype=np.float64)}
            candidates, pair_diagnostics = self._group_candidates(
                group,
                anchor,
                transforms,
                work,
                group_edges,
            )
            submissions = self._select_submissions(
                group_id,
                group,
                anchor,
                transforms,
                candidates,
                work,
                original_shapes,
                records_by_index,
            )
            all_submissions.extend(submissions)
            group_payloads.append(
                {
                    "group_id": group_id,
                    "images": [records_by_index[index].path.name for index in group],
                    "anchor": records_by_index[anchor].path.name,
                    "edge_count": len(group_edges),
                    "candidate_count": len(candidates),
                    "submissions": [],
                }
            )
            if bool(self.config.get("io.save_diagnostics", False)):
                for first, second, scores, valid in pair_diagnostics:
                    heatmap = residual_heatmap(
                        scores,
                        valid,
                        work[anchor].gray.shape,
                    )
                    diagnostic_name = (
                        f"{records_by_index[first].path.stem}__to__"
                        f"{records_by_index[second].path.stem}__heatmap.jpg"
                    )
                    write_image(
                        diagnostics_root / diagnostic_name,
                        heatmap,
                        self.quality,
                    )

        for number, submission in enumerate(all_submissions, start=1):
            submission.submission_id = (
                f"{safe_sequence_id}__submission_{number:06d}"
            )
            stem = records_by_index[submission.image_index].path.stem
            if bool(self.config.get("io.save_crops", True)):
                crop_name = (
                    f"{stem}__{submission.submission_id}__{submission.kind}__"
                    f"{submission.completeness}.jpg"
                )
                output_bucket = (
                    "review_queue"
                    if submission.completeness == "review_required"
                    else "submissions"
                )
                crop_path = (
                    self.output_dir
                    / output_bucket
                    / sequence_path
                    / crop_name
                )
                original = read_color(
                    records_by_index[submission.image_index].path
                )
                write_image(
                    crop_path,
                    crop(original, submission.bbox),
                    self.quality,
                )
                submission.crop_path = crop_path.relative_to(self.output_dir)

        submissions_by_image: dict[int, list[Submission]] = defaultdict(list)
        for submission in all_submissions:
            submissions_by_image[submission.image_index].append(submission)
        if bool(self.config.get("io.save_annotated", True)):
            for record in records:
                original = read_color(record.path)
                annotated = annotate(
                    original,
                    submissions_by_image.get(record.index, []),
                )
                write_image(
                    self.output_dir
                    / "annotated"
                    / sequence_path
                    / f"{record.path.stem}.jpg",
                    annotated,
                    self.quality,
                )

        payloads: list[dict[str, Any]] = []
        for submission in all_submissions:
            payload = {
                "submission_id": submission.submission_id,
                "group_id": submission.group_id,
                "source_path": submission.source_path.relative_to(folder).as_posix(),
                "kind": submission.kind,
                "side": submission.side,
                "bbox": list(submission.bbox),
                "completeness": submission.completeness,
                "confidence": round(submission.confidence, 6),
                "content_score": round(submission.content_score, 6),
                "occlusion_fraction": round(
                    submission.occlusion_fraction,
                    6,
                ),
                "reason": submission.reason,
                "crop_path": (
                    submission.crop_path.as_posix()
                    if submission.crop_path
                    else None
                ),
            }
            payloads.append(payload)
            for group_payload in group_payloads:
                if group_payload["group_id"] == submission.group_id:
                    group_payload["submissions"].append(
                        submission.submission_id
                    )
                    break

        return {
            "folder": resolved_sequence_id,
            "source_folder": folder.as_posix(),
            "image_count": len(records),
            "edge_count": len(edges),
            "group_count": len(groups),
            "groups": group_payloads,
            "submissions": payloads,
        }
