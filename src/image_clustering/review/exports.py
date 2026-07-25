"""Export reviewer-corrected clusters and boxes as final manifests.

Exports are derived state: the decision document remains the source of truth, and
canonical clustering/cropping output is never rewritten. Excluded members and
dissolved clusters become standalone single-capture records so no source image is
dropped from the corrected manifest. Clusters marked irregular are intentionally
omitted from corrected cluster and bbox manifests and listed in an exclusion audit.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from image_clustering.review.dataset import ReviewDataset
from image_clustering.review.decisions import (
    STATUS_DISSOLVED,
    STATUS_IRREGULAR,
    STATUS_UNREVIEWED,
    cluster_state,
    image_state,
    progress,
    utc_now,
)


def _final_boxes(
    state: dict[str, Any],
    dataset: ReviewDataset,
    cluster_id: str,
    image_id: str,
) -> tuple[list[dict[str, Any]], str, bool]:
    """Return final boxes, bbox status, and whether the reviewer edited them."""
    image = dataset.image(cluster_id, image_id)
    record = image_state(state, cluster_id, image_id)
    reviewer_boxes = record.get("boxes")
    status = str(record.get("bbox_status") or "unreviewed")
    if reviewer_boxes is None:
        return [box.to_dict() for box in image.boxes], status, False
    return list(reviewer_boxes), status, True


def review_rows(
    state: dict[str, Any],
    dataset: ReviewDataset,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Return corrected manifests plus an exclusion audit for irregular clusters."""
    cluster_records: list[dict[str, Any]] = []
    box_records: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    standalone_index = 0
    for cluster in dataset.clusters:
        record = cluster_state(state, cluster.cluster_id)
        status = str(record.get("status") or STATUS_UNREVIEWED)
        dissolved = bool(record.get("dissolved"))
        excluded = set(record.get("excluded_image_ids") or ())
        kept = [image for image in cluster.images if image.image_id not in excluded]
        removed = [image for image in cluster.images if image.image_id in excluded]
        is_irregular = status == STATUS_IRREGULAR
        if not is_irregular and not dissolved and kept:
            cluster_records.append(
                {
                    "cluster_id": cluster.cluster_id,
                    "origin_cluster_id": cluster.cluster_id,
                    "source_folder": cluster.source_folder,
                    "review_status": status,
                    "dissolved": False,
                    "image_count": len(kept),
                    "image_ids": [image.image_id for image in kept],
                    "image_paths": [str(image.source_path) for image in kept],
                    "removed_image_ids": sorted(excluded),
                    "minimum_confidence": cluster.minimum_confidence,
                    "reviewed": status != STATUS_UNREVIEWED,
                }
            )
        if not is_irregular:
            for image in removed if not dissolved else cluster.images:
                standalone_index += 1
                separated_id = (
                    f"{cluster.cluster_id}__separated_{standalone_index:05d}"
                )
                cluster_records.append(
                    {
                        "cluster_id": separated_id,
                        "origin_cluster_id": cluster.cluster_id,
                        "source_folder": cluster.source_folder,
                        "review_status": STATUS_DISSOLVED if dissolved else status,
                        "dissolved": dissolved,
                        "image_count": 1,
                        "image_ids": [image.image_id],
                        "image_paths": [str(image.source_path)],
                        "removed_image_ids": [],
                        "minimum_confidence": None,
                        "reviewed": True,
                    }
                )
        if not is_irregular:
            for image in cluster.images:
                boxes, bbox_status, edited = _final_boxes(
                    state, dataset, cluster.cluster_id, image.image_id
                )
                in_reviewed_cluster = not dissolved and image.image_id not in excluded
                box_records.append(
                    {
                        "origin_cluster_id": cluster.cluster_id,
                        "image_id": image.image_id,
                        "source_image_path": str(image.source_path),
                        "source_width": image.width,
                        "source_height": image.height,
                        "still_in_cluster": in_reviewed_cluster,
                        "bbox_status": bbox_status,
                        "bboxes_edited_by_reviewer": edited,
                        "box_count": len(boxes),
                        "boxes": boxes,
                    }
                )
        summary_rows.append(
            {
                "cluster_id": cluster.cluster_id,
                "source_folder": cluster.source_folder,
                "original_image_count": cluster.image_count,
                "kept_image_count": (
                    0 if dissolved or is_irregular else len(kept)
                ),
                "removed_image_count": (
                    cluster.image_count if dissolved else len(removed)
                ),
                "excluded_from_training_validation": is_irregular,
                "excluded_image_count": cluster.image_count if is_irregular else 0,
                "excluded_image_ids": (
                    ";".join(image.image_id for image in cluster.images)
                    if is_irregular
                    else ""
                ),
                "review_status": status,
                "dissolved": dissolved,
                "reviewed": status != STATUS_UNREVIEWED,
                "minimum_confidence": cluster.minimum_confidence,
                "bbox_statuses": ";".join(
                    sorted(
                        {
                            str(
                                image_state(
                                    state, cluster.cluster_id, image.image_id
                                ).get("bbox_status")
                            )
                            for image in cluster.images
                        }
                    )
                ),
                "review_reasons": "; ".join(cluster.review_reasons),
            }
        )
    return cluster_records, box_records, summary_rows


def _irregular_exclusion_rows(
    state: dict[str, Any],
    dataset: ReviewDataset,
) -> list[dict[str, Any]]:
    """Return an auditable list of images omitted for irregular clusters."""
    rows: list[dict[str, Any]] = []
    for cluster in dataset.clusters:
        record = cluster_state(state, cluster.cluster_id)
        if str(record.get("status") or STATUS_UNREVIEWED) != STATUS_IRREGULAR:
            continue
        rows.append(
            {
                "cluster_id": cluster.cluster_id,
                "source_folder": cluster.source_folder,
                "review_status": STATUS_IRREGULAR,
                "reason": "irregular",
                "image_count": cluster.image_count,
                "image_ids": [image.image_id for image in cluster.images],
                "image_paths": [str(image.source_path) for image in cluster.images],
            }
        )
    return rows


def write_review_exports(
    state: dict[str, Any],
    dataset: ReviewDataset,
    export_root: Path | None = None,
) -> dict[str, Any]:
    """Write corrected manifests and return the artifact paths and counts."""
    root = Path(export_root) if export_root else dataset.output_root / "review_labels"
    root.mkdir(parents=True, exist_ok=True)
    cluster_records, box_records, summary_rows = review_rows(state, dataset)
    exclusion_rows = _irregular_exclusion_rows(state, dataset)

    clusters_path = root / "clusters_reviewed.jsonl"
    clusters_path.write_text(
        "".join(
            json.dumps(record, separators=(",", ":")) + "\n"
            for record in cluster_records
        ),
        encoding="utf-8",
    )
    boxes_path = root / "crops_reviewed.jsonl"
    boxes_path.write_text(
        "".join(
            json.dumps(record, separators=(",", ":")) + "\n" for record in box_records
        ),
        encoding="utf-8",
    )
    exclusions_path = root / "excluded_from_training_validation.jsonl"
    exclusions_path.write_text(
        "".join(
            json.dumps(record, separators=(",", ":")) + "\n"
            for record in exclusion_rows
        ),
        encoding="utf-8",
    )
    summary_path = root / "review_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        if summary_rows:
            writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
            writer.writeheader()
            writer.writerows(summary_rows)
        else:
            handle.write("")

    counts = progress(state, dataset).to_dict()
    manifest = {
        "schema_version": 1,
        "exported_at": utc_now(),
        "provenance": dataset.provenance,
        "progress": counts,
        "final_cluster_count": len(cluster_records),
        "final_image_count": sum(record["image_count"] for record in cluster_records),
        "excluded_from_training_validation": {
            "cluster_count": len(exclusion_rows),
            "image_count": sum(record["image_count"] for record in exclusion_rows),
        },
        "artifacts": {
            "clusters_reviewed": str(clusters_path),
            "crops_reviewed": str(boxes_path),
            "review_summary": str(summary_path),
            "excluded_from_training_validation": str(exclusions_path),
        },
    }
    manifest_path = root / "review_export.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    manifest["artifacts"]["review_export"] = str(manifest_path)
    return manifest
