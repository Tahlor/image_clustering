"""Reproducible full-dataset evaluation around the package's public APIs.

This is deliberately an evaluation/reporting layer: clustering and crop recovery
remain implemented by ``image_clustering.clustering`` and
``image_clustering.cropping``. Every manifest and HTML page is generated from
those saved results; no source-specific curation is performed here.
"""

# The generated HTML/report strings intentionally remain readable as prose.
# Ruff's formatter is not required for those embedded strings.
# ruff: noqa: E501

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import logging
import os
import platform
import shutil
import sys
import threading
import time
from collections import Counter, defaultdict
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2

from image_clustering import (
    ClusterConfig,
    crop_clustering_result,
    load_crop_config,
    load_result,
    write_result,
)
from image_clustering.clustering.api import cluster_directory
from image_clustering.clustering.discovery import discover_triplet_sequences
from image_clustering.clustering.models import ClusteringResult, ImageCluster

LOGGER = logging.getLogger(__name__)
SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".j2k", ".jp2"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_records(records: Iterable[dict[str, Any]]) -> str:
    encoded = json.dumps(list(records), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def csv_write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def inventory(
    input_root: Path,
    output_root: Path,
    triplet_manifest: Path | None = None,
) -> dict[str, Any]:
    """Inventory exactly the files that the package discovery will process."""
    input_root = input_root.resolve()
    allowed_paths: set[Path] | None = None
    if triplet_manifest is not None:
        allowed_paths = {
            image.path.resolve()
            for sequence in discover_triplet_sequences(input_root, triplet_manifest)
            for image in sequence
        }
    rows: list[dict[str, Any]] = []
    decode_failures: list[dict[str, str]] = []
    for path in sorted(
        (
            candidate
            for candidate in input_root.rglob("*")
            if candidate.is_file()
            and candidate.suffix.lower() in SUPPORTED_SUFFIXES
            and (allowed_paths is None or candidate.resolve() in allowed_paths)
        ),
        key=lambda candidate: candidate.relative_to(input_root).as_posix(),
    ):
        relative = path.relative_to(input_root).as_posix()
        parent = path.parent.relative_to(input_root).as_posix() or "."
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        failure: str | None = None
        if image is None:
            failure = "cv2.imread returned None"
            decode_failures.append({"relative_path": relative, "error": failure})
        if image is None:
            dimensions: list[int] | None = None
            channels = None
        else:
            dimensions = [int(image.shape[1]), int(image.shape[0])]
            channels = int(image.shape[2]) if image.ndim == 3 else 1
        rows.append(
            {
                "filename": path.name,
                "absolute_source_path": str(path),
                "relative_path": relative,
                "immediate_parent_folder": parent,
                "width": dimensions[0] if dimensions else None,
                "height": dimensions[1] if dimensions else None,
                "channels": channels,
                "format": path.suffix.lower().lstrip("."),
                "file_size_bytes": path.stat().st_size,
                "decoding_error": failure,
            }
        )
    folders = sorted({row["immediate_parent_folder"] for row in rows})
    counts = Counter(row["immediate_parent_folder"] for row in rows)
    payload = {
        "schema_version": 1,
        "created_at": utc_now(),
        "input_root": str(input_root),
        "triplet_manifest": str(triplet_manifest.resolve()) if triplet_manifest is not None else None,
        "total_image_count": len(rows),
        "immediate_parent_folder_count": len(folders),
        "image_count_by_folder": dict(sorted(counts.items())),
        "supported_suffixes": sorted(SUPPORTED_SUFFIXES),
        "decoding_failures": decode_failures,
        "images": rows,
    }
    write_json(output_root / "inventory" / "dataset_inventory.json", payload)
    csv_write(output_root / "inventory" / "dataset_inventory.csv", rows)
    return payload


def load_or_create_inventory(
    input_root: Path,
    output_root: Path,
    triplet_manifest: Path | None = None,
) -> dict[str, Any]:
    path = output_root / "inventory" / "dataset_inventory.json"
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        current = inventory(input_root, output_root, triplet_manifest=triplet_manifest)
        if digest_records(payload.get("images", [])) == digest_records(current["images"]):
            return payload
    return inventory(input_root, output_root, triplet_manifest=triplet_manifest)


def source_folder_for(image_path: Path, input_root: Path) -> str:
    try:
        return image_path.parent.resolve().relative_to(input_root.resolve()).as_posix() or "."
    except ValueError:
        return image_path.parent.name


def cluster_pair_payload(result: ClusteringResult, cluster: ImageCluster) -> list[dict[str, Any]]:
    members = {image.image_id for image in result.images_for(cluster.cluster_id)}
    return [
        {
            "first_image_id": comparison.first_image_id,
            "second_image_id": comparison.second_image_id,
            "index_gap": comparison.index_gap,
            "confidence": round(comparison.confidence, 6),
            "same_document": comparison.same_document,
            "reason": comparison.reason,
            "registration_model": comparison.registration_model,
            "good_match_count": comparison.good_match_count,
            "inlier_count": comparison.inlier_count,
            "inlier_ratio": round(comparison.inlier_ratio, 6),
            "feature_overlap": round(comparison.feature_overlap, 6),
            "median_reprojection_error": round(comparison.median_reprojection_error, 6),
            "valid_fraction": round(comparison.valid_fraction, 6),
            "changed_fraction": round(comparison.changed_fraction, 6),
            "stable_fraction": round(comparison.stable_fraction, 6),
            "tiles_changed_fraction": round(comparison.tiles_changed_fraction, 6),
            "largest_change_share": round(comparison.largest_change_share, 6),
            "branch": comparison.branch,
            "unmatched_ink_fraction": round(comparison.unmatched_ink_fraction, 6),
            "unmatched_ink_union_fraction": round(comparison.unmatched_ink_union_fraction, 6),
            "ink_mismatch_tiles_fraction": round(comparison.ink_mismatch_tiles_fraction, 6),
            "coherent_ink_component_count": comparison.coherent_ink_component_count,
            "largest_ink_component_fraction": round(comparison.largest_ink_component_fraction, 6),
            "residual_tiles_changed_fraction": round(comparison.residual_tiles_changed_fraction, 6),
            "occlusion_candidate_count": comparison.occlusion_candidate_count,
            "occlusion_area_fraction": round(comparison.occlusion_area_fraction, 6),
            "occlusion_residual_capture": round(comparison.occlusion_residual_capture, 6),
            "occlusion_rectangularity": round(comparison.occlusion_rectangularity, 6),
            "occlusion_boundary_score": round(comparison.occlusion_boundary_score, 6),
            "occlusion_material_fraction": round(comparison.occlusion_material_fraction, 6),
            "occlusion_material_median": round(comparison.occlusion_material_median, 6),
            "outside_unmatched_ink_fraction": round(comparison.outside_unmatched_ink_fraction, 6),
            "outside_unmatched_ink_union_fraction": round(comparison.outside_unmatched_ink_union_fraction, 6),
            "outside_ink_mismatch_tiles_fraction": round(comparison.outside_ink_mismatch_tiles_fraction, 6),
            "full_page_occlusion_count": comparison.full_page_occlusion_count,
            "shallow_occlusion_count": comparison.shallow_occlusion_count,
            "page_count": comparison.page_count,
            "hard_contradiction": comparison.hard_contradiction,
            "transform": comparison.transform,
        }
        for comparison in result.comparisons
        if comparison.first_image_id in members and comparison.second_image_id in members
    ]


def cluster_review_reasons(
    result: ClusteringResult,
    cluster: ImageCluster,
    pairs: list[dict[str, Any]],
) -> list[str]:
    accepted = [pair for pair in pairs if pair["same_document"]]
    reasons: list[str] = []
    if len(cluster.image_ids) == 1:
        reasons.append("singleton has no registration evidence")
    if accepted and min(pair["confidence"] for pair in accepted) < 0.55:
        reasons.append("low-confidence accepted registration")
    if any(pair["index_gap"] > 1 for pair in accepted):
        reasons.append("nonadjacent relationship")
    if any(not pair["same_document"] for pair in pairs):
        reasons.append("a candidate direct pair failed but the component is connected transitively")
    return reasons


def cluster_artifacts(
    result: ClusteringResult,
    input_root: Path,
    output_root: Path,
) -> list[dict[str, Any]]:
    """Generate model payloads, CSV summary, and linked cluster review pages."""
    output_root = output_root.resolve()
    summary_rows: list[dict[str, Any]] = []
    model_lines: list[dict[str, Any]] = []
    cluster_dir = output_root / "review" / "clusters"
    cluster_dir.mkdir(parents=True, exist_ok=True)
    for cluster in result.clusters:
        images = list(result.images_for(cluster.cluster_id))
        if [image.sequence_index for image in images] != sorted(image.sequence_index for image in images):
            raise ValueError(f"Cluster members are not filename ordered: {cluster.cluster_id}")
        pairs = cluster_pair_payload(result, cluster)
        accepted = [pair for pair in pairs if pair["same_document"]]
        confidence_values = [pair["confidence"] for pair in accepted]
        minimum_confidence = min(confidence_values) if confidence_values else None
        mean_confidence = sum(confidence_values) / len(confidence_values) if confidence_values else None
        maximum_confidence = max(confidence_values) if confidence_values else None
        review_reasons = cluster_review_reasons(result, cluster, pairs)
        review_required = bool(review_reasons)
        relative_paths = [
            image.path.resolve().relative_to(input_root.resolve()).as_posix()
            for image in images
        ]
        absolute_paths = [str(image.path.resolve()) for image in images]
        representative = result.image(cluster.representative_image_id)
        models = sorted({pair["registration_model"] for pair in accepted if pair["registration_model"]})
        model_lines.append(
            {
                "cluster_id": cluster.cluster_id,
                "sequence_id": cluster.sequence_id,
                "image_count": len(images),
                "image_paths": absolute_paths,
                "relative_image_paths": relative_paths,
                "representative_image_path": str(representative.path.resolve()),
                "minimum_accepted_pair_confidence": minimum_confidence,
                "mean_accepted_pair_confidence": mean_confidence,
                "maximum_accepted_pair_confidence": maximum_confidence,
                "pairwise_confidences": pairs,
                "review_required": review_required,
                "review_reasons": review_reasons,
            }
        )
        summary_rows.append(
            {
                "cluster_id": cluster.cluster_id,
                "source_subfolder": cluster.sequence_id,
                "image_count": len(images),
                "ordered_member_filenames": json.dumps([image.path.name for image in images]),
                "ordered_member_absolute_paths": json.dumps(absolute_paths),
                "representative_image": str(representative.path.resolve()),
                "minimum_accepted_pair_confidence": minimum_confidence,
                "mean_accepted_pair_confidence": mean_confidence,
                "maximum_accepted_pair_confidence": maximum_confidence,
                "largest_sequence_gap_bridged": max((pair["index_gap"] for pair in accepted), default=0),
                "registration_models_used": json.dumps(models),
                "direct_pair_failed_but_connected_transitively": any(not pair["same_document"] for pair in pairs),
                "manual_review_required": review_required,
                "review_reasons": "; ".join(review_reasons),
            }
        )
        write_cluster_page(
            output_root=output_root,
            input_root=input_root,
            cluster=cluster,
            images=images,
            pairs=pairs,
            review_reasons=review_reasons,
        )
    csv_write(output_root / "reports" / "cluster_summary.csv", summary_rows)
    model_path = output_root / "reports" / "clusters_for_model.jsonl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text(
        "".join(json.dumps(line, separators=(",", ":")) + "\n" for line in model_lines),
        encoding="utf-8",
    )
    write_cluster_index(output_root, summary_rows)
    return summary_rows


def file_link(path: Path) -> str:
    try:
        return path.resolve().as_uri()
    except ValueError:
        return "file:///" + str(path.resolve()).replace("\\", "/")


def relative_asset_link(page_path: Path, asset_path: Path) -> str:
    """Return a local relative link so generated review pages are portable."""
    try:
        return Path(os.path.relpath(asset_path.resolve(), page_path.parent.resolve())).as_posix()
    except ValueError:
        return file_link(asset_path)


def review_full_resolution_source(image_path: Path) -> Path | None:
    """Return the browser-safe source image, preferring a sibling JPEG for J2K."""
    if image_path.suffix.lower() in {".j2k", ".jp2"}:
        for suffix in (".jpg", ".jpeg"):
            sibling_jpeg = image_path.with_suffix(suffix)
            if sibling_jpeg.is_file() and sibling_jpeg.stat().st_size > 0:
                return sibling_jpeg
        return None
    return image_path if image_path.is_file() and image_path.stat().st_size > 0 else None


def review_full_resolution_path(output_root: Path, image_path: Path) -> Path:
    token = hashlib.sha1(str(image_path.resolve()).encode("utf-8")).hexdigest()[:12]
    return output_root / "review" / "full_resolution" / f"{token}_{image_path.stem}.jpg"


def ensure_review_full_resolution(output_root: Path, image_path: Path) -> Path | None:
    """Copy the full-resolution sibling JPEG into the static review bundle."""
    source = review_full_resolution_source(image_path)
    if source is None:
        return None
    target = review_full_resolution_path(output_root, image_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.is_file() or target.stat().st_size != source.stat().st_size:
        shutil.copy2(source, target)
    return target if target.is_file() and target.stat().st_size > 0 else None


def review_thumbnail_path(output_root: Path, cluster_id: str, image: Any) -> Path:
    token = hashlib.sha1(str(image.path.resolve()).encode("utf-8")).hexdigest()[:12]
    return output_root / "review" / "thumbnails" / cluster_id / f"{image.sequence_index:06d}_{token}.jpg"


def ensure_review_thumbnail(output_root: Path, cluster_id: str, image: Any) -> Path | None:
    """Prefer cropper JPEGs; decode a source only when no reusable preview exists."""
    annotated = output_root / "annotated" / image.path.parent.name / cluster_id / f"{image.path.stem}.jpg"
    if annotated.is_file() and annotated.stat().st_size > 0:
        return annotated
    thumbnail = review_thumbnail_path(output_root, cluster_id, image)
    if thumbnail.is_file() and thumbnail.stat().st_size > 0:
        return thumbnail
    source_for_preview = image.path
    if image.path.suffix.lower() in {".j2k", ".jp2"}:
        sibling_jpeg = image.path.with_suffix(".jpg")
        if sibling_jpeg.is_file() and sibling_jpeg.stat().st_size > 0:
            source_for_preview = sibling_jpeg
    decoded = cv2.imread(str(source_for_preview), cv2.IMREAD_COLOR)
    if decoded is None:
        return None
    height, width = decoded.shape[:2]
    scale = min(1.0, 420 / max(height, width, 1))
    if scale < 1.0:
        decoded = cv2.resize(
            decoded,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    thumbnail.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(thumbnail), decoded, [cv2.IMWRITE_JPEG_QUALITY, 88]):
        return None
    return thumbnail if thumbnail.is_file() and thumbnail.stat().st_size > 0 else None


def write_cluster_page(
    output_root: Path,
    input_root: Path,
    cluster: ImageCluster,
    images: list[Any],
    pairs: list[dict[str, Any]],
    review_reasons: list[str],
) -> None:
    page_dir = output_root / "review" / "clusters" / cluster.cluster_id
    page_path = page_dir / "index.html"
    page_dir.mkdir(parents=True, exist_ok=True)
    pair_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        pair_by_id[pair["first_image_id"]].append(pair)
        pair_by_id[pair["second_image_id"]].append(pair)
    cards = []
    for image in images:
        diagnostics = pair_by_id.get(image.image_id, [])
        diagnostics_html = "<br>".join(
            html.escape(
                f"{pair['first_image_id']} ↔ {pair['second_image_id']}: "
                f"{'accepted' if pair['same_document'] else 'rejected'}, "
                f"confidence={pair['confidence']}, gap={pair['index_gap']}, "
                f"model={pair['registration_model']}"
            )
            for pair in diagnostics
        ) or "No direct candidate pair"
        thumbnail = ensure_review_thumbnail(output_root, cluster.cluster_id, image)
        full_resolution = ensure_review_full_resolution(output_root, image.path)
        source_href = (
            relative_asset_link(page_path, full_resolution)
            if full_resolution
            else file_link(image.path)
        )
        image_html = (
            f"<img loading='lazy' src='{html.escape(relative_asset_link(page_path, thumbnail))}' "
            f"alt='Preview of {html.escape(image.path.name)}'>"
            if thumbnail
            else "<div class='missing-image'>Preview unavailable; use the source link.</div>"
        )
        cards.append(
            "<article class='image-card'>"
            f"<h3>#{image.sequence_index}: {html.escape(image.path.name)}</h3>"
            f"<p><a href='{html.escape(source_href)}'>Open full-resolution source</a></p>"
            f"{image_html}"
            f"<p>{diagnostics_html}</p>"
            "</article>"
        )
    payload = {
        "cluster_id": cluster.cluster_id,
        "sequence_id": cluster.sequence_id,
        "image_paths": [str(image.path.resolve()) for image in images],
        "pairwise_confidences": pairs,
        "review_required": bool(review_reasons),
        "review_reasons": review_reasons,
    }
    body = (
        f"<h1>{html.escape(cluster.cluster_id)}</h1>"
        f"<p>Source folder: <code>{html.escape(cluster.sequence_id)}</code>; "
        f"members: {len(images)}; representative: "
        f"<code>{html.escape(cluster.representative_image_id)}</code></p>"
        f"<p class='warning'>{html.escape('; '.join(review_reasons)) if review_reasons else 'No automatic review warning.'}</p>"
        "<section class='cluster-images'>" + "".join(cards) + "</section>"
        "<details><summary>Machine-readable cluster metadata</summary>"
        f"<pre>{html.escape(json.dumps(payload, indent=2))}</pre></details>"
    )
    write_html(page_path, f"Cluster {cluster.cluster_id}", body)


def write_cluster_index(output_root: Path, rows: list[dict[str, Any]]) -> None:
    table_rows = []
    for row in rows:
        minimum_confidence = row["minimum_accepted_pair_confidence"]
        mean_confidence = row["mean_accepted_pair_confidence"]
        maximum_confidence = row["maximum_accepted_pair_confidence"]
        table_rows.append(
            "<tr>"
            f"<td><a href='{html.escape(row['cluster_id'])}/index.html'>{html.escape(row['cluster_id'])}</a></td>"
            f"<td>{html.escape(row['source_subfolder'])}</td>"
            f"<td data-sort='{row['image_count']}'>{row['image_count']}</td>"
            f"<td data-sort='{minimum_confidence if minimum_confidence is not None else -1}'>"
            f"{minimum_confidence if minimum_confidence is not None else 'no pair'}</td>"
            f"<td data-sort='{mean_confidence if mean_confidence is not None else -1}'>"
            f"{mean_confidence if mean_confidence is not None else 'no pair'}</td>"
            f"<td data-sort='{maximum_confidence if maximum_confidence is not None else -1}'>"
            f"{maximum_confidence if maximum_confidence is not None else 'no pair'}</td>"
            f"<td>{'yes' if row['manual_review_required'] else 'no'}</td>"
            f"<td>{html.escape(row['review_reasons'])}</td></tr>"
        )
    body = (
        "<h1>Cluster review</h1><p>Click a header to sort. Confidence columns use accepted pair confidence; singleton clusters show no pair.</p>"
        "<table id='items'><thead><tr>"
        "<th>Cluster ID</th><th>Source folder</th><th>Size</th><th>Weakest-link confidence</th>"
        "<th>Mean confidence</th><th>Strongest-link confidence</th><th>Review</th><th>Reasons</th>"
        "</tr></thead><tbody>" + "".join(table_rows) + "</tbody></table>"
    )
    write_html(output_root / "review" / "clusters" / "index.html", "Cluster review", body, sortable=True)


def crop_artifacts(
    cropping: dict[str, Any],
    input_root: Path,
    output_root: Path,
    inventory_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    by_cluster: dict[str, list[dict[str, Any]]] = defaultdict(list)
    dimensions = {
        row["absolute_source_path"]: (row["width"], row["height"])
        for row in inventory_payload["images"]
    }
    submission_number = 0
    for cluster_result in cropping.get("clusters", []):
        cluster_id = cluster_result.get("upstream_cluster_id", cluster_result.get("folder"))
        source_folder = Path(cluster_result["source_folder"]).resolve()
        by_cluster.setdefault(cluster_id, [])
        source_records = [
            (source_folder / filename, {})
            for group in cluster_result.get("groups", [])
            for filename in group.get("images", [])
        ]
        for submission in cluster_result.get("submissions", []):
            submission_number += 1
            source_path = (source_folder / submission["source_path"]).resolve()
            crop_path_value = submission.get("crop_path")
            crop_path = (output_root / crop_path_value).resolve() if crop_path_value else None
            width, height = dimensions.get(str(source_path), (None, None))
            bbox = [int(value) for value in submission["bbox"]]
            confidence = float(submission.get("confidence", 0.0))
            review_reasons: list[str] = []
            if submission.get("completeness") == "review_required":
                review_reasons.append("cropper marked confidence insufficient")
            if confidence < 0.55:
                review_reasons.append("low submission confidence")
            if crop_path is None or not crop_path.is_file() or crop_path.stat().st_size == 0:
                review_reasons.append("crop file missing or empty")
            if width is None or height is None:
                review_reasons.append("source dimensions unavailable")
            elif not (0 <= bbox[0] < bbox[2] <= width and 0 <= bbox[1] < bbox[3] <= height):
                review_reasons.append("bounding box outside source dimensions")
            review_required = bool(review_reasons)
            disposition = (
                "REVIEW REQUIRED"
                if review_required
                else "SUBMIT PARTIAL"
                if submission["completeness"] == "partial_best_available"
                else "SUBMIT COMPLETE"
            )
            row = {
                "submission_id": f"submission_{submission_number:05d}",
                "cluster_id": cluster_id,
                "source_folder": source_folder.as_posix(),
                "source_filename": source_path.name,
                "source_image_path": str(source_path),
                "source_width": width,
                "source_height": height,
                "crop_path": str(crop_path) if crop_path else None,
                "bbox": bbox,
                "kind": "data_bearing_overlay" if submission["kind"] in {"data_bearing_overlay", "page_state"} else "base_page",
                "completeness": submission["completeness"],
                "confidence": confidence,
                "content_score": float(submission.get("content_score", 0.0)),
                "estimated_occlusion_fraction": float(submission.get("occlusion_fraction", 0.0)),
                "reason": submission.get("reason", ""),
                "review_required": review_required,
                "review_reasons": review_reasons,
                "disposition": disposition,
                "side": submission.get("side"),
                "group_id": submission.get("group_id"),
            }
            rows.append(row)
            by_cluster[cluster_id].append(row)
            source_records.append((source_path, row))
        write_crop_cluster_page(output_root, cluster_result, source_records, by_cluster[cluster_id])
    csv_rows = []
    for row in rows:
        csv_rows.append(
            {
                "submission_id": row["submission_id"],
                "cluster_id": row["cluster_id"],
                "source_folder": row["source_folder"],
                "source_filename": row["source_filename"],
                "crop_path": row["crop_path"],
                "bbox_original_source_coordinates": json.dumps(row["bbox"]),
                "type": row["kind"],
                "completeness": row["completeness"],
                "confidence": row["confidence"],
                "content_score": row["content_score"],
                "estimated_occlusion_fraction": row["estimated_occlusion_fraction"],
                "reason": row["reason"],
                "manual_review_required": row["review_required"],
            }
        )
    csv_write(output_root / "reports" / "crop_summary.csv", csv_rows)
    manifest_path = output_root / "reports" / "crops_for_recognizer.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        "".join(
            json.dumps(
                {
                    "submission_id": row["submission_id"],
                    "cluster_id": row["cluster_id"],
                    "source_image_path": row["source_image_path"],
                    "crop_path": row["crop_path"],
                    "bbox": row["bbox"],
                    "kind": row["kind"],
                    "completeness": row["completeness"],
                    "confidence": row["confidence"],
                    "review_required": row["review_required"],
                },
                separators=(",", ":"),
            ) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    write_crop_index(output_root, by_cluster, rows)
    return rows


def write_crop_cluster_page(
    output_root: Path,
    cluster_result: dict[str, Any],
    source_records: list[tuple[Path, dict[str, Any]]],
    rows: list[dict[str, Any]],
) -> None:
    path = output_root / "review" / "crops" / "by_cluster" / f"{cluster_result.get('upstream_cluster_id', cluster_result.get('folder'))}.html"
    source_cards = []
    seen_sources: set[Path] = set()
    for source_path, _row in source_records:
        if source_path in seen_sources:
            continue
        seen_sources.add(source_path)
        stem = source_path.stem
        folder = cluster_result.get("folder", "")
        annotated = (output_root / "annotated" / folder / f"{stem}.jpg").resolve()
        full_resolution = ensure_review_full_resolution(output_root, source_path)
        source_href = (
            relative_asset_link(path, full_resolution)
            if full_resolution
            else file_link(source_path)
        )
        annotated_src = relative_asset_link(path, annotated)
        source_cards.append(
            "<article class='card'>"
            f"<h3>{html.escape(source_path.name)}</h3>"
            f"<p><a href='{html.escape(source_href)}'>Original full resolution</a> · "
            f"<a href='{html.escape(file_link(annotated))}'>Annotated source</a></p>"
            f"<img src='{html.escape(annotated_src)}' alt='Annotated {html.escape(source_path.name)}'>"
            "</article>"
        )
    crop_cards = []
    for row in rows:
        crop_path = Path(row["crop_path"]) if row["crop_path"] else None
        crop_src = relative_asset_link(path, crop_path) if crop_path else None
        crop_cards.append(
            "<article class='card'>"
            f"<h3>{html.escape(row['disposition'])}: {html.escape(row['submission_id'])}</h3>"
            f"<p>Source: <a href='{html.escape(file_link(Path(row['source_image_path'])))}'>{html.escape(row['source_filename'])}</a>; "
            f"bbox={html.escape(json.dumps(row['bbox']))}; confidence={row['confidence']:.4f}</p>"
            f"<p>{html.escape(row['reason'])}</p>"
            f"<p>{html.escape('; '.join(row['review_reasons']))}</p>"
            + (f"<img src='{html.escape(crop_src)}' alt='Crop {html.escape(row['submission_id'])}'>" if crop_src else "<p>Missing crop</p>")
            + "</article>"
        )
    cluster_payload = {
        "cluster_id": cluster_result.get("upstream_cluster_id", cluster_result.get("folder")),
        "source_folder": cluster_result.get("source_folder"),
        "groups": cluster_result.get("groups", []),
        "submissions": rows,
    }
    body = (
        f"<h1>Crop review: {html.escape(str(cluster_payload['cluster_id']))}</h1>"
        f"<p>Source folder: <code>{html.escape(str(cluster_result.get('source_folder')))}</code>; "
        f"automatic submissions: {len(rows)}</p>"
        "<h2>Original captures and selected boxes</h2><section class='grid'>"
        + "".join(source_cards)
        + "</section><h2>Exact recognizer crops and dispositions</h2><section class='grid'>"
        + "".join(crop_cards)
        + "</section><h2>Machine-readable cluster metadata</h2>"
        + f"<pre>{html.escape(json.dumps(cluster_payload, indent=2))}</pre>"
    )
    write_html(path, f"Crop review {cluster_payload['cluster_id']}", body)


def write_crop_index(output_root: Path, by_cluster: dict[str, list[dict[str, Any]]], rows: list[dict[str, Any]]) -> None:
    counts = Counter(row["cluster_id"] for row in rows)
    review_counts = Counter(row["cluster_id"] for row in rows if row["review_required"])
    table_rows = []
    for cluster_id in sorted(by_cluster):
        table_rows.append(
            "<tr>"
            f"<td><a href='by_cluster/{html.escape(cluster_id)}.html'>{html.escape(cluster_id)}</a></td>"
            f"<td>{counts[cluster_id]}</td><td>{review_counts[cluster_id]}</td></tr>"
        )
    body = (
        "<h1>Recognizer crop review</h1>"
        "<p>Each page links original captures, annotated selections, exact crops, and automatic dispositions.</p>"
        "<table><thead><tr><th>Cluster</th><th>Submissions</th><th>Review-required</th></tr></thead><tbody>"
        + "".join(table_rows)
        + "</tbody></table>"
    )
    write_html(output_root / "review" / "crops" / "index.html", "Recognizer crop review", body)


def write_html(path: Path, title: str, body: str, sortable: bool = False) -> None:
    script = """
<script>
for (const header of document.querySelectorAll('th')) {
  header.addEventListener('click', () => {
    const table = header.closest('table');
    const index = Array.from(header.parentNode.children).indexOf(header);
    const rows = Array.from(table.tBodies[0].rows);
    rows.sort((a,b) => {
      const left = a.cells[index].dataset.sort ?? a.cells[index].innerText;
      const right = b.cells[index].dataset.sort ?? b.cells[index].innerText;
      return left.localeCompare(right, undefined, {numeric:true});
    });
    for (const row of rows) table.tBodies[0].appendChild(row);
  });
}
</script>
""" if sortable else ""
    document = f"""<!doctype html>
<html><head><meta charset='utf-8'><title>{html.escape(title)}</title>
<style>
body{{font-family:system-ui,sans-serif;margin:1.5rem;background:#f7f7f7;color:#222}}
.grid,.cluster-images{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:1rem}}
.cluster-card{{background:white;border:1px solid #bbc5d1;border-radius:.35rem;padding:1rem;margin:1rem 0;overflow-wrap:anywhere}}
.cluster-card[hidden]{{display:none}}
.cluster-header{{display:flex;justify-content:space-between;gap:1rem;align-items:flex-start;border-bottom:1px solid #ddd;margin-bottom:1rem}}
.cluster-header h2{{margin:.1rem 0 .4rem}}
.cluster-header nav{{white-space:nowrap;padding-top:.4rem}}
.cluster-image{{background:#f5f7fa;border:1px solid #d4dbe4;border-radius:.25rem;padding:.55rem;min-width:0}}
.image-label{{font-size:.85rem;font-weight:600;overflow-wrap:anywhere;margin-bottom:.4rem}}
.cluster-image img{{display:block;width:100%;height:220px;object-fit:contain;background:#e7ebef}}
.image-card{{background:white;border:1px solid #ccd4df;padding:1rem;overflow-wrap:anywhere}}
.image-card img{{display:block;max-width:100%;max-height:520px;width:100%;object-fit:contain;background:#e7ebef}}
.card{{background:white;border:1px solid #ccc;padding:1rem;overflow-wrap:anywhere}}
.card img{{max-width:100%;max-height:520px;object-fit:contain;background:#eee}}
.toolbar{{position:sticky;top:0;z-index:2;display:flex;flex-wrap:wrap;gap:.75rem;align-items:end;background:#e9eef5;border:1px solid #bbc5d1;padding:.8rem;margin:1rem 0}}
.toolbar label{{display:flex;flex-direction:column;gap:.25rem;font-size:.9rem;font-weight:600}}
.toolbar input,.toolbar select{{font:inherit;padding:.4rem;min-width:10rem}}
.result-count{{font-weight:600;color:#46566a}}
.badge{{display:inline-block;background:#e3edf8;border:1px solid #b6cce2;border-radius:1rem;padding:.15rem .5rem;font-size:.78rem;vertical-align:middle}}
.missing-image{{display:grid;place-items:center;height:220px;background:#ececec;color:#666;text-align:center;padding:.5rem}}
table{{border-collapse:collapse;background:white;width:100%}} th,td{{border:1px solid #ccc;padding:.45rem;text-align:left}} th{{cursor:pointer;background:#e9eef5}}
.warning{{padding:.7rem;background:#fff3cd;border:1px solid #e0b400;white-space:pre-wrap}}
pre{{white-space:pre-wrap;max-height:32rem;overflow:auto;background:#111;color:#eee;padding:1rem}}
</style></head><body>{body}{script}</body></html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")


class PeakMemory:
    def __init__(self) -> None:
        self.peak_bytes = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()

    def _sample(self) -> None:
        try:
            import psutil
        except ImportError:
            return
        process = psutil.Process(os.getpid())
        while not self._stop.wait(0.5):
            self.peak_bytes = max(self.peak_bytes, process.memory_info().rss)

    def stop(self) -> int | None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        return self.peak_bytes or None


def _label_pair_rows(
    clustering: ClusteringResult,
    labels_path: Path,
) -> list[dict[str, Any]]:
    """Resolve reviewed filename pairs against clustering diagnostics."""
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    image_by_name = {image.path.name: image for image in clustering.images}
    categories = (("hard_negatives", False), ("near_duplicates", True))
    for category, expected in categories:
        for pair in labels.get(category, []):
            first_name, second_name = pair
            first = image_by_name.get(first_name)
            second = image_by_name.get(second_name)
            comparison = None
            if first is not None and second is not None:
                try:
                    comparison = clustering.comparison(first.image_id, second.image_id)
                except KeyError:
                    comparison = None
            rows.append(
                {
                    "category": category,
                    "first_filename": first_name,
                    "second_filename": second_name,
                    "expected_same_document": expected,
                    "found": comparison is not None,
                    "actual_same_document": comparison.same_document if comparison else None,
                    "passed": (
                        first is not None
                        and second is not None
                        and (
                            (comparison is None and not expected)
                            or (comparison is not None and comparison.same_document == expected)
                        )
                    ),
                    "branch": comparison.branch if comparison else None,
                    "reason": comparison.reason if comparison else "pair not present in candidate comparisons",
                    "confidence": round(comparison.confidence, 6) if comparison else None,
                    "feature_overlap": round(comparison.feature_overlap, 6) if comparison else None,
                    "unmatched_ink_union_fraction": round(comparison.unmatched_ink_union_fraction, 6) if comparison else None,
                    "ink_mismatch_tiles_fraction": round(comparison.ink_mismatch_tiles_fraction, 6) if comparison else None,
                    "residual_tiles_changed_fraction": round(comparison.residual_tiles_changed_fraction, 6) if comparison else None,
                    "occlusion_area_fraction": round(comparison.occlusion_area_fraction, 6) if comparison else None,
                    "outside_unmatched_ink_union_fraction": round(comparison.outside_unmatched_ink_union_fraction, 6) if comparison else None,
                    "hard_contradiction": comparison.hard_contradiction if comparison else None,
                }
            )
    return rows


def _review_card(
    row: dict[str, Any],
    crop_count: int,
    images: list[Any],
    crop_rows: list[dict[str, Any]],
    output_root: Path,
    page_path: Path,
) -> str:
    cluster_id = str(row["cluster_id"])
    size = int(row["image_count"])
    review = "yes" if row["manual_review_required"] else "no"
    reasons = html.escape(row["review_reasons"] or "none")
    minimum_confidence = row["minimum_accepted_pair_confidence"]
    mean_confidence = row["mean_accepted_pair_confidence"]
    maximum_confidence = row["maximum_accepted_pair_confidence"]
    confidence_text = (
        f"weakest-link confidence: {minimum_confidence:.4f}; "
        f"mean: {mean_confidence:.4f}; strongest-link: {maximum_confidence:.4f}"
        if minimum_confidence is not None
        else "no accepted pair confidence (singleton)"
    )
    image_cards = []
    for image in images:
        thumbnail = ensure_review_thumbnail(output_root, cluster_id, image)
        full_resolution = ensure_review_full_resolution(output_root, image.path)
        image_label = f"{cluster_id}, image {image.sequence_index}: {image.path.name}"
        source_href = (
            relative_asset_link(page_path, full_resolution)
            if full_resolution
            else file_link(image.path)
        )
        if thumbnail:
            image_src = relative_asset_link(page_path, thumbnail)
            viewer_src = source_href if full_resolution else image_src
            image_html = (
                f"<img loading='lazy' src='{html.escape(image_src)}' "
                f"alt='Preview of {html.escape(image.path.name)}'>"
            )
            image_control = (
                f"<button type='button' class='image-open' "
                f"data-image-src='{html.escape(viewer_src)}' "
                f"data-image-label='{html.escape(image_label)}' "
                f"aria-label='Open {html.escape(image_label)} in the image viewer' "
                f"title='Open {html.escape(image_label)} in the full-screen viewer'>"
                f"{image_html}</button>"
            )
        else:
            image_control = "<div class='missing-image'>Preview unavailable</div>"
        image_crops = [
            crop for crop in crop_rows
            if Path(crop["source_image_path"]).resolve() == image.path.resolve()
        ]
        crop_controls = []
        if image_crops:
            for crop in image_crops:
                bbox = [int(value) for value in crop["bbox"]]
                width = crop.get("source_width")
                height = crop.get("source_height")
                crop_id = str(crop["submission_id"])
                crop_controls.append(
                    "<fieldset class='crop-review'>"
                    f"<legend>Crop {html.escape(crop_id)} ({html.escape(crop['kind'])})</legend>"
                    "<p class='help-text'>Set a crop decision and edit x-min, y-min, x-max, and y-max in original source pixels.</p>"
                    f"<label>Status <select data-review-crop-status='true' data-cluster-id='{html.escape(cluster_id)}' data-image-id='{html.escape(image.image_id)}' data-submission-id='{html.escape(crop_id)}'>"
                    "<option value='unreviewed'>unreviewed</option><option value='correct'>correct</option>"
                    "<option value='incorrect'>incorrect</option><option value='needs_recrop'>needs recrop</option></select></label>"
                    "<div class='bbox-fields'>"
                    + "".join(
                        f"<label>{axis} <input type='number' min='0' step='1' value='{bbox[index]}' data-review-bbox='{axis}' data-cluster-id='{html.escape(cluster_id)}' data-image-id='{html.escape(image.image_id)}' data-submission-id='{html.escape(crop_id)}' data-width='{html.escape(str(width or ''))}' data-height='{html.escape(str(height or ''))}'></label>"
                        for index, axis in enumerate(("xmin", "ymin", "xmax", "ymax"))
                    )
                    + "</div>"
                    f"<p class='crop-validation' data-crop-validation='{html.escape(crop_id)}' aria-live='polite'>Canonical bbox: {html.escape(json.dumps(bbox))}; source: {html.escape(str(width or '?'))} × {html.escape(str(height or '?'))} px.</p>"
                    "</fieldset>"
                )
        else:
            crop_controls.append("<p class='help-text'>No automatic crop is associated with this image.</p>")
        image_cards.append(
            "<article class='cluster-image' "
            f"data-image-id='{html.escape(image.image_id)}'>"
            f"<div class='image-label'>#{image.sequence_index} · {html.escape(image.path.name)}</div>"
            f"{image_control}"
            f"<label class='membership-control'><input type='checkbox' data-review-image-membership='true' data-cluster-id='{html.escape(cluster_id)}' data-image-id='{html.escape(image.image_id)}'> Include image in cluster</label>"
            f"<p><a href='{html.escape(source_href)}' title='Open the original full-resolution JPG source image'>Open full-resolution source</a></p>"
            + "".join(crop_controls)
            + "</article>"
        )
    search_text = " ".join(
        (
            cluster_id,
            str(row["source_subfolder"]),
            str(row["review_reasons"] or ""),
        )
    )
    confidence_sort = minimum_confidence if minimum_confidence is not None else -1
    return (
        f"<article class='cluster-card' data-search='{html.escape(search_text)}' "
        f"data-id='{html.escape(cluster_id)}' data-folder='{html.escape(row['source_subfolder'])}' "
        f"data-size='{size}' data-review='{review}' data-crops='{int(crop_count)}' "
        f"data-gap='{int(row['largest_sequence_gap_bridged'])}' data-confidence='{confidence_sort}'>"
        f"<header class='cluster-header'><div><h2>{html.escape(cluster_id)} "
        f"<span class='badge'>{size} capture{'s' if size != 1 else ''}</span></h2>"
        f"<p>Folder: <code>{html.escape(row['source_subfolder'])}</code>; crops: {crop_count}; "
        f"review: {review}; {html.escape(confidence_text)}; reasons: {reasons}</p></div>"
        f"<div class='review-controls'><label>Cluster decision <select data-review-cluster-status='true' data-cluster-id='{html.escape(cluster_id)}' aria-label='Cluster decision for {html.escape(cluster_id)}'>"
        "<option value='unreviewed'>unreviewed</option><option value='confirmed'>confirmed</option>"
        "<option value='not_a_cluster'>not a cluster</option></select></label>"
        "<p class='help-text'>Use “not a cluster” when the automatic grouping is wrong. This records a tuning label; it does not mutate canonical clustering.</p></div>"
        f"<nav><a href='clusters/{html.escape(cluster_id)}/index.html'>cluster detail</a> · "
        f"<a href='crops/by_cluster/{html.escape(cluster_id)}.html'>crop detail</a></nav></header>"
        f"<section class='cluster-images'>{''.join(image_cards)}</section>"
        "<details><summary>Cluster diagnostics</summary>"
        f"<pre>{html.escape(json.dumps(row, indent=2, default=str))}</pre></details></article>"
    )


def write_review_package(
    output_root: Path,
    clustering: ClusteringResult,
    cluster_rows: list[dict[str, Any]],
    crop_rows: list[dict[str, Any]],
    labels_path: Path,
) -> list[dict[str, Any]]:
    """Write an image-first, filterable cluster browser and focused review pages."""
    crop_counts = Counter(row["cluster_id"] for row in crop_rows)
    clusters_by_id = {cluster.cluster_id: cluster for cluster in clustering.clusters}
    images_by_cluster = {
        cluster_id: list(clustering.images_for(cluster_id))
        for cluster_id in clusters_by_id
    }
    index_path = output_root / "review" / "index.html"
    review_provenance = {
        "dataset_root": str(clustering.input_root) if clustering.input_root else None,
        "grouping_mode": clustering.grouping_mode,
        "group_manifest": str(clustering.group_manifest) if clustering.group_manifest else None,
        "clustering_config_fingerprint": clustering.config_fingerprint,
        "image_count": len(clustering.images),
        "cluster_count": len(clustering.clusters),
        "crop_count": len(crop_rows),
        "review_package_revision": "review-labels-v1",
    }
    review_template_path = output_root / "reports" / "review_decisions.json"
    if not review_template_path.is_file():
        write_json(
            review_template_path,
            {
                "schema_version": 1,
                "export_type": "review_decisions",
                "provenance": review_provenance,
                "decisions": {},
            },
        )
    cards = [
        _review_card(
            row,
            crop_counts[row["cluster_id"]],
            images_by_cluster[row["cluster_id"]],
            crop_rows,
            output_root,
            index_path,
        )
        for row in cluster_rows
    ]
    toolbar = (
        "<style>"
        ".viewer-open{overflow:hidden}"
        ".image-open{display:block;width:100%;padding:0;border:0;background:transparent;cursor:zoom-in}"
        ".image-open:focus-visible{outline:3px solid #1769aa;outline-offset:3px}"
        ".image-open img{display:block;width:100%;height:220px;object-fit:contain;background:#e7ebef}"
        ".image-hint{font-size:.82rem;color:#46566a;margin:.35rem 0 0}"
        ".viewer-status{font-weight:600;color:#46566a}"
        ".image-viewer{position:fixed;inset:0;z-index:20;display:grid;place-items:center;padding:1rem;background:rgba(8,15,25,.82)}"
        ".image-viewer[hidden]{display:none}"
        ".viewer-dialog{position:relative;display:flex;flex-direction:column;width:min(96vw,1400px);height:min(96vh,1000px);padding:1rem;background:#101820;color:#f7f9fb;border:1px solid #667789;border-radius:.5rem;box-shadow:0 1rem 4rem #000;overflow:hidden}"
        ".viewer-header,.viewer-footer{display:flex;align-items:center;justify-content:space-between;gap:.75rem;flex-wrap:wrap}"
        ".viewer-header{border-bottom:1px solid #445260;padding-bottom:.65rem}"
        ".viewer-header h2{font-size:1rem;margin:0;overflow-wrap:anywhere}"
        ".viewer-stage{display:grid;grid-template-columns:auto minmax(0,1fr) auto;align-items:center;gap:.75rem;min-height:0;flex:1;padding:.75rem 0}"
        ".viewer-stage figure{display:flex;flex-direction:column;align-items:center;justify-content:center;min-width:0;min-height:0;height:100%;margin:0}"
        ".viewer-image{display:block;max-width:100%;max-height:100%;width:auto;height:auto;object-fit:contain;background:#25313d}"
        ".viewer-error{padding:1rem;background:#542b2b;color:#ffdede}"
        ".viewer-nav{min-width:2.75rem;min-height:2.75rem;font-size:1.4rem}"
        ".viewer-footer{border-top:1px solid #445260;padding-top:.75rem}"
        ".viewer-footer label{display:flex;align-items:center;gap:.6rem;flex:1;min-width:16rem}"
        ".viewer-footer input[type=range]{flex:1;min-width:10rem}"
        ".viewer-dialog button{font:inherit;padding:.4rem .65rem;border:1px solid #9eb3c7;border-radius:.3rem;background:#eaf2f9;color:#142433;cursor:pointer}"
        ".viewer-dialog button:disabled{cursor:not-allowed;opacity:.45}"
        ".viewer-dialog button:focus-visible{outline:3px solid #ffd166;outline-offset:2px}"
        ".viewer-dialog .viewer-nav{background:#d8e9f7}"
        ".review-controls{min-width:16rem;max-width:24rem}"
        ".review-controls label,.membership-control,.crop-review label{display:flex;align-items:center;gap:.4rem;font-weight:600}"
        ".review-controls select,.crop-review select{font:inherit;padding:.3rem}"
        ".help-text{font-size:.82rem;color:#46566a;margin:.35rem 0}"
        ".membership-control{margin:.6rem 0;font-size:.88rem}"
        ".crop-review{margin-top:.7rem;border:1px solid #c8d2de;padding:.55rem;background:#fff}"
        ".crop-review legend{font-weight:700}"
        ".bbox-fields{display:grid;grid-template-columns:repeat(4,minmax(4rem,1fr));gap:.35rem;margin-top:.45rem}"
        ".bbox-fields input{width:100%;min-width:0;box-sizing:border-box}"
        ".crop-validation{font-size:.78rem;margin:.45rem 0 0;color:#46566a}"
        ".crop-validation.invalid{color:#a51d2d;font-weight:700}"
        ".review-save-status{font-size:.85rem;color:#46566a;align-self:center}"
        ".review-import{display:none}"
        "</style>"
        "<div class='toolbar' role='region' aria-label='Cluster review filters'>"
        "<label>Search <input id='search' type='search' placeholder='cluster, folder, or review reason'></label>"
        "<label>Cluster size <select id='min-size'>"
        "<option value='0'>all clusters</option><option value='2'>nontrivial (2+ images)</option>"
        "<option value='3'>3+ images</option><option value='4'>4+ images</option>"
        "<option value='10'>10+ images</option></select></label>"
        "<label>Review <select id='review'>"
        "<option value=''>all</option><option value='yes'>review required</option>"
        "<option value='no'>no review warning</option></select></label>"
        "<label>Reviewer decision <select id='decision-filter'>"
        "<option value=''>all decisions</option><option value='unreviewed'>unreviewed</option>"
        "<option value='confirmed'>confirmed</option><option value='not_a_cluster'>not a cluster</option></select></label>"
        "<label>Crop output <select id='crop-filter'>"
        "<option value=''>any crop count</option><option value='yes'>has crops</option>"
        "<option value='no'>no crops</option></select></label>"
        "<label>Sort <select id='sort'>"
        "<option value='size-desc'>largest first</option><option value='size-asc'>smallest first</option>"
        "<option value='confidence-asc'>weakest-link confidence first</option>"
        "<option value='confidence-desc'>strongest-link confidence first</option>"
        "<option value='folder'>source folder</option><option value='id'>cluster ID</option>"
        "<option value='review'>review required first</option></select></label>"
        "<button type='button' id='open-first' title='Open the first image in the filtered review queue'>Open filtered queue</button>"
        "<button type='button' id='export-decisions' title='Download all current cluster, membership, and crop decisions as JSON'>Export review decisions</button>"
        "<button type='button' id='export-tuning' title='Download one machine-readable tuning record per reviewed cluster image and crop'>Export tuning JSONL</button>"
        "<button type='button' id='import-decisions' title='Load a previously exported review decisions JSON file'>Import decisions</button>"
        "<input id='import-decisions-file' class='review-import' type='file' accept='application/json,.json'>"
        "<span id='review-save-status' class='review-save-status' role='status' aria-live='polite'>No reviewer changes saved yet.</span>"
        "</div><p class='help-text'>Review labels stay separate from canonical clustering and cropping. They are saved in this browser and must be explicitly exported for tuning.</p><p id='result-count' class='result-count' aria-live='polite'></p>"
    )
    viewer_markup = (
        "<section id='image-viewer' class='image-viewer' hidden aria-hidden='true' "
        "aria-label='Cluster image viewer'>"
        "<div class='viewer-backdrop' data-close-viewer='true'></div>"
        "<div class='viewer-dialog' role='dialog' aria-modal='true' aria-labelledby='viewer-title'>"
        "<header class='viewer-header'><h2 id='viewer-title'>Image viewer</h2>"
        "<div><button type='button' id='viewer-fullscreen' title='Use the browser full-screen mode'>Full screen</button> "
        "<button type='button' id='viewer-close' title='Close the image viewer'>Close</button></div></header>"
        "<div class='viewer-stage'>"
        "<button type='button' id='viewer-prev-image' class='viewer-nav' title='Previous image in this cluster' aria-label='Previous image in this cluster'>‹</button>"
        "<figure><img id='viewer-image' class='viewer-image' alt=''><figcaption id='viewer-caption'></figcaption>"
        "<p id='viewer-error' class='viewer-error' hidden>Image preview unavailable.</p></figure>"
        "<button type='button' id='viewer-next-image' class='viewer-nav' title='Next image in this cluster' aria-label='Next image in this cluster'>›</button>"
        "</div>"
        "<footer class='viewer-footer'>"
        "<button type='button' id='viewer-prev-cluster' title='Open the previous filtered cluster'>Previous cluster</button>"
        "<label for='viewer-slider'>Skip through filtered images"
        "<input id='viewer-slider' type='range' min='0' max='0' value='0' step='1' aria-label='Skip through filtered images'>"
        "<span id='viewer-slider-status' class='viewer-status'>0 / 0</span></label>"
        "<button type='button' id='viewer-next-cluster' title='Open the next filtered cluster'>Next cluster</button>"
        "</footer></div></section>"
    )
    script = """
<script>
const REVIEW_PROVENANCE = __REVIEW_PROVENANCE__;
const REVIEW_STORAGE_KEY = `image-clustering-review:${REVIEW_PROVENANCE.clustering_config_fingerprint}:${REVIEW_PROVENANCE.review_package_revision}`;
const container = document.querySelector('#clusters');
const cards = [...container.querySelectorAll('.cluster-card')];
const search = document.querySelector('#search');
const minSize = document.querySelector('#min-size');
const review = document.querySelector('#review');
const decisionFilter = document.querySelector('#decision-filter');
const cropFilter = document.querySelector('#crop-filter');
const sort = document.querySelector('#sort');
const resultCount = document.querySelector('#result-count');
const openFirst = document.querySelector('#open-first');
const imageViewer = document.querySelector('#image-viewer');
const viewerDialog = document.querySelector('.viewer-dialog');
const viewerImage = document.querySelector('#viewer-image');
const viewerCaption = document.querySelector('#viewer-caption');
const viewerTitle = document.querySelector('#viewer-title');
const viewerError = document.querySelector('#viewer-error');
const viewerClose = document.querySelector('#viewer-close');
const viewerFullscreen = document.querySelector('#viewer-fullscreen');
const viewerPrevImage = document.querySelector('#viewer-prev-image');
const viewerNextImage = document.querySelector('#viewer-next-image');
const viewerPrevCluster = document.querySelector('#viewer-prev-cluster');
const viewerNextCluster = document.querySelector('#viewer-next-cluster');
const viewerSlider = document.querySelector('#viewer-slider');
const viewerSliderStatus = document.querySelector('#viewer-slider-status');
let filteredCards = [];
let reviewQueue = [];
let activeQueueIndex = -1;
let activeImageIndex = 0;
let pendingQueueIndex = 0;
let lastFocusedElement = null;
const exportDecisions = document.querySelector('#export-decisions');
const exportTuning = document.querySelector('#export-tuning');
const importDecisions = document.querySelector('#import-decisions');
const importDecisionsFile = document.querySelector('#import-decisions-file');
const reviewSaveStatus = document.querySelector('#review-save-status');
let reviewState = loadReviewState();

function emptyReviewState() {
  return {schema_version: 1, provenance: REVIEW_PROVENANCE, decisions: {}};
}

function loadReviewState() {
  try {
    const saved = JSON.parse(localStorage.getItem(REVIEW_STORAGE_KEY) || 'null');
    if (saved && saved.decisions && typeof saved.decisions === 'object') return saved;
  } catch (error) {
    console.warn('Could not load saved review labels', error);
  }
  return emptyReviewState();
}

function clusterDecision(clusterId) {
  if (!reviewState.decisions[clusterId]) reviewState.decisions[clusterId] = {status: 'unreviewed', images: {}};
  const decision = reviewState.decisions[clusterId];
  if (!decision.images || typeof decision.images !== 'object') decision.images = {};
  return decision;
}

function imageDecision(clusterId, imageId) {
  const cluster = clusterDecision(clusterId);
  if (!cluster.images[imageId]) cluster.images[imageId] = {included: null, crops: {}};
  if (!cluster.images[imageId].crops || typeof cluster.images[imageId].crops !== 'object') cluster.images[imageId].crops = {};
  return cluster.images[imageId];
}

function cropDecision(clusterId, imageId, submissionId, canonicalBox) {
  const image = imageDecision(clusterId, imageId);
  if (!image.crops[submissionId]) image.crops[submissionId] = {status: 'unreviewed', corrected_bbox: null};
  const crop = image.crops[submissionId];
  if (!Array.isArray(crop.corrected_bbox) && canonicalBox) crop.corrected_bbox = null;
  return crop;
}

function persistReviewState(message) {
  reviewState.provenance = REVIEW_PROVENANCE;
  try {
    localStorage.setItem(REVIEW_STORAGE_KEY, JSON.stringify(reviewState));
    reviewSaveStatus.textContent = message || `Saved ${new Date().toLocaleTimeString()}; export to create a tuning file.`;
  } catch (error) {
    reviewSaveStatus.textContent = 'Browser storage is unavailable; export immediately to avoid losing labels.';
    console.warn('Could not save review labels', error);
  }
}

function canonicalBoxFor(cropId) {
  const input = document.querySelector(`[data-review-bbox][data-submission-id="${CSS.escape(cropId)}"]`);
  if (!input) return null;
  const fields = ['xmin', 'ymin', 'xmax', 'ymax'].map((axis) => document.querySelector(`[data-review-bbox="${axis}"][data-submission-id="${CSS.escape(cropId)}"]`));
  return fields.map((field) => Number(field.value));
}

function validateBox(fields) {
  const values = fields.map((field) => Number(field.value));
  const width = Number(fields[0].dataset.width);
  const height = Number(fields[0].dataset.height);
  const valid = fields.every((field) => field.value.trim() !== '' && Number.isInteger(Number(field.value)))
    && values[0] >= 0 && values[1] >= 0
    && values[2] > values[0] && values[3] > values[1]
    && (!width || values[2] <= width) && (!height || values[3] <= height);
  const message = valid
    ? `Valid corrected bbox [${values.join(', ')}]`
    : `Invalid bbox. Require integer 0 ≤ xmin < xmax ≤ ${width || '?'} and 0 ≤ ymin < ymax ≤ ${height || '?'}.`;
  return {values, valid, message};
}

function applyReviewStateToControls() {
  for (const control of document.querySelectorAll('[data-review-cluster-status]')) {
    control.value = clusterDecision(control.dataset.clusterId).status || 'unreviewed';
  }
  for (const control of document.querySelectorAll('[data-review-image-membership]')) {
    const included = imageDecision(control.dataset.clusterId, control.dataset.imageId).included;
    control.checked = included !== false;
  }
  for (const control of document.querySelectorAll('[data-review-crop-status]')) {
    const crop = cropDecision(control.dataset.clusterId, control.dataset.imageId, control.dataset.submissionId, canonicalBoxFor(control.dataset.submissionId));
    control.value = crop.status || 'unreviewed';
  }
  for (const field of document.querySelectorAll('[data-review-bbox]')) {
    const fields = ['xmin', 'ymin', 'xmax', 'ymax'].map((axis) => document.querySelector(`[data-review-bbox="${axis}"][data-submission-id="${CSS.escape(field.dataset.submissionId)}"]`));
    const crop = cropDecision(field.dataset.clusterId, field.dataset.imageId, field.dataset.submissionId, fields.map((item) => Number(item.value)));
    if (Array.isArray(crop.corrected_bbox)) fields.forEach((item, index) => { item.value = crop.corrected_bbox[index]; });
  }
  for (const field of document.querySelectorAll('[data-review-bbox]')) updateBoxValidation(field.dataset.submissionId);
}

function updateBoxValidation(submissionId) {
  const fields = ['xmin', 'ymin', 'xmax', 'ymax'].map((axis) => document.querySelector(`[data-review-bbox="${axis}"][data-submission-id="${CSS.escape(submissionId)}"]`));
  if (fields.some((field) => !field)) return {valid: false, values: []};
  const result = validateBox(fields);
  const message = document.querySelector(`[data-crop-validation="${CSS.escape(submissionId)}"]`);
  if (message) {
    message.textContent = result.message;
    message.classList.toggle('invalid', !result.valid);
  }
  fields.forEach((field) => field.setAttribute('aria-invalid', String(!result.valid)));
  return result;
}

function downloadText(filename, content, type) {
  const link = document.createElement('a');
  link.href = URL.createObjectURL(new Blob([content], {type}));
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(link.href), 0);
}

function exportPayload() {
  return {schema_version: 1, export_type: 'review_decisions', exported_at: new Date().toISOString(), provenance: REVIEW_PROVENANCE, decisions: reviewState.decisions};
}

function tuningRows() {
  const rows = [];
  for (const card of cards) {
    const clusterId = card.dataset.id;
    const cluster = clusterDecision(clusterId);
    for (const image of card.querySelectorAll('[data-review-image-membership]')) {
      const decision = imageDecision(clusterId, image.dataset.imageId);
      const imageCard = image.closest('.cluster-image');
      for (const status of imageCard.querySelectorAll('[data-review-crop-status]')) {
        const crop = cropDecision(clusterId, image.dataset.imageId, status.dataset.submissionId, canonicalBoxFor(status.dataset.submissionId));
        const boxFields = ['xmin', 'ymin', 'xmax', 'ymax'].map((axis) => imageCard.querySelector(`[data-review-bbox="${axis}"][data-submission-id="${CSS.escape(status.dataset.submissionId)}"]`));
        const box = updateBoxValidation(status.dataset.submissionId);
        rows.push({schema_version: 1, cluster_id: clusterId, image_id: image.dataset.imageId, included: decision.included, cluster_status: cluster.status, submission_id: status.dataset.submissionId, crop_status: crop.status, corrected_bbox: box.valid ? box.values : null, canonical_bbox: boxFields.map((field) => Number(field.defaultValue)), provenance: REVIEW_PROVENANCE});
      }
      if (!imageCard.querySelector('[data-review-crop-status]')) rows.push({schema_version: 1, cluster_id: clusterId, image_id: image.dataset.imageId, included: decision.included, cluster_status: cluster.status, submission_id: null, crop_status: null, corrected_bbox: null, canonical_bbox: null, provenance: REVIEW_PROVENANCE});
    }
  }
  return rows;
}

function compareCards(a, b) {
  const key = sort.value;
  if (key === 'size-desc') return Number(b.dataset.size) - Number(a.dataset.size) || a.dataset.id.localeCompare(b.dataset.id);
  if (key === 'size-asc') return Number(a.dataset.size) - Number(b.dataset.size) || a.dataset.id.localeCompare(b.dataset.id);
  if (key === 'confidence-asc') return Number(a.dataset.confidence) - Number(b.dataset.confidence) || a.dataset.id.localeCompare(b.dataset.id);
  if (key === 'confidence-desc') return Number(b.dataset.confidence) - Number(a.dataset.confidence) || a.dataset.id.localeCompare(b.dataset.id);
  if (key === 'folder') return a.dataset.folder.localeCompare(b.dataset.folder) || a.dataset.id.localeCompare(b.dataset.id);
  if (key === 'review') return (b.dataset.review === 'yes') - (a.dataset.review === 'yes') || a.dataset.id.localeCompare(b.dataset.id);
  return a.dataset.id.localeCompare(b.dataset.id);
}

function imageButtons(card) {
  return [...card.querySelectorAll('.image-open[data-image-src]')];
}

function updateQueue() {
  const currentButton = reviewQueue[activeQueueIndex]?.button;
  filteredCards = cards.filter((card) => !card.hidden);
  reviewQueue = [];
  for (const card of filteredCards) {
    imageButtons(card).forEach((button, imageIndex) => reviewQueue.push({ card, button, imageIndex }));
  }
  viewerSlider.max = String(Math.max(0, reviewQueue.length - 1));
  viewerSlider.disabled = reviewQueue.length === 0;
  openFirst.disabled = reviewQueue.length === 0;
  const replacementIndex = currentButton ? reviewQueue.findIndex((item) => item.button === currentButton) : -1;
  activeQueueIndex = replacementIndex >= 0 ? replacementIndex : Math.min(Math.max(activeQueueIndex, 0), Math.max(reviewQueue.length - 1, 0));
  pendingQueueIndex = activeQueueIndex >= 0 ? activeQueueIndex : 0;
  viewerSlider.value = String(pendingQueueIndex);
  viewerSliderStatus.textContent = reviewQueue.length ? `${pendingQueueIndex + 1} / ${reviewQueue.length}` : '0 / 0';
}

function setViewerItem(queueIndex) {
  if (!reviewQueue.length) return;
  activeQueueIndex = Math.min(Math.max(Number(queueIndex), 0), reviewQueue.length - 1);
  pendingQueueIndex = activeQueueIndex;
  const item = reviewQueue[activeQueueIndex];
  const clusterButtons = imageButtons(item.card);
  activeImageIndex = item.imageIndex;
  viewerImage.src = item.button.dataset.imageSrc;
  viewerImage.alt = item.button.dataset.imageLabel || 'Cluster image preview';
  viewerImage.hidden = false;
  viewerError.hidden = true;
  viewerTitle.textContent = `${item.card.dataset.id} · image ${activeImageIndex + 1} of ${clusterButtons.length}`;
  viewerCaption.textContent = item.button.dataset.imageLabel || '';
  viewerSlider.value = String(activeQueueIndex);
  viewerSliderStatus.textContent = `${activeQueueIndex + 1} / ${reviewQueue.length}`;
  viewerSlider.setAttribute('aria-valuetext', `${activeQueueIndex + 1} of ${reviewQueue.length}`);
  viewerPrevImage.disabled = activeImageIndex <= 0;
  viewerNextImage.disabled = activeImageIndex >= clusterButtons.length - 1;
  const clusterIndex = filteredCards.indexOf(item.card);
  viewerPrevCluster.disabled = clusterIndex <= 0;
  viewerNextCluster.disabled = clusterIndex < 0 || clusterIndex >= filteredCards.length - 1;
}

function openViewerAt(queueIndex) {
  updateQueue();
  if (!reviewQueue.length) return;
  lastFocusedElement = document.activeElement;
  imageViewer.hidden = false;
  imageViewer.setAttribute('aria-hidden', 'false');
  document.body.classList.add('viewer-open');
  setViewerItem(queueIndex);
  viewerClose.focus();
}

function closeViewer() {
  if (imageViewer.hidden) return;
  imageViewer.hidden = true;
  imageViewer.setAttribute('aria-hidden', 'true');
  document.body.classList.remove('viewer-open');
  viewerImage.removeAttribute('src');
  if (lastFocusedElement && typeof lastFocusedElement.focus === 'function') lastFocusedElement.focus();
}

function navigateWithinCluster(step) {
  if (activeQueueIndex < 0) return;
  const current = reviewQueue[activeQueueIndex];
  const targetImageIndex = current.imageIndex + step;
  const target = reviewQueue.find((item) => item.card === current.card && item.imageIndex === targetImageIndex);
  if (target) setViewerItem(reviewQueue.indexOf(target));
}

function navigateCluster(step) {
  if (activeQueueIndex < 0) return;
  const current = reviewQueue[activeQueueIndex];
  const clusterIndex = filteredCards.indexOf(current.card) + step;
  const targetCard = filteredCards[clusterIndex];
  if (!targetCard) return;
  const targetButtons = imageButtons(targetCard);
  if (!targetButtons.length) return;
  const target = reviewQueue.find((item) => item.card === targetCard && item.imageIndex === (step > 0 ? 0 : targetButtons.length - 1));
  if (target) setViewerItem(reviewQueue.indexOf(target));
}

function refresh() {
  const query = search.value.trim().toLowerCase();
  const minimum = Number(minSize.value);
  const reviewValue = review.value;
  const decisionValue = decisionFilter.value;
  const cropValue = cropFilter.value;
  if (!imageViewer.hidden) closeViewer();
  cards.sort(compareCards);
  for (const card of cards) container.appendChild(card);
  let visible = 0;
  for (const card of cards) {
    const matches = (!query || card.dataset.search.toLowerCase().includes(query))
      && Number(card.dataset.size) >= minimum
      && (!reviewValue || card.dataset.review === reviewValue)
      && (!decisionValue || clusterDecision(card.dataset.id).status === decisionValue)
      && (!cropValue || (cropValue === 'yes' ? Number(card.dataset.crops) > 0 : Number(card.dataset.crops) === 0));
    card.hidden = !matches;
    if (matches) visible += 1;
  }
  updateQueue();
  resultCount.textContent = `Showing ${visible.toLocaleString()} of ${cards.length.toLocaleString()} clusters; ${reviewQueue.length.toLocaleString()} review images. Select a thumbnail for the full-screen viewer.`;
}

for (const control of [search, minSize, review, decisionFilter, cropFilter, sort]) {
  control.addEventListener('input', refresh);
  control.addEventListener('change', refresh);
}
for (const button of document.querySelectorAll('.image-open')) {
  button.addEventListener('click', () => {
    const queueIndex = reviewQueue.findIndex((item) => item.button === button);
    openViewerAt(queueIndex >= 0 ? queueIndex : 0);
  });
}
openFirst.addEventListener('click', () => openViewerAt(0));
for (const control of document.querySelectorAll('[data-review-cluster-status]')) {
  control.addEventListener('change', () => {
    clusterDecision(control.dataset.clusterId).status = control.value;
    persistReviewState(`Saved cluster decision for ${control.dataset.clusterId}.`);
  });
}
for (const control of document.querySelectorAll('[data-review-image-membership]')) {
  control.addEventListener('change', () => {
    imageDecision(control.dataset.clusterId, control.dataset.imageId).included = control.checked;
    persistReviewState(`Saved membership decision for ${control.dataset.imageId}.`);
  });
}
for (const control of document.querySelectorAll('[data-review-crop-status]')) {
  control.addEventListener('change', () => {
    cropDecision(control.dataset.clusterId, control.dataset.imageId, control.dataset.submissionId, canonicalBoxFor(control.dataset.submissionId)).status = control.value;
    persistReviewState(`Saved crop decision for ${control.dataset.submissionId}.`);
  });
}
for (const field of document.querySelectorAll('[data-review-bbox]')) {
  field.addEventListener('input', () => {
    const result = updateBoxValidation(field.dataset.submissionId);
    const status = field.closest('.crop-review').querySelector('[data-review-crop-status]');
    const crop = cropDecision(field.dataset.clusterId, field.dataset.imageId, field.dataset.submissionId, null);
    crop.corrected_bbox = result.valid ? result.values : null;
    persistReviewState(result.valid ? `Saved corrected bbox for ${field.dataset.submissionId}.` : `Bbox for ${field.dataset.submissionId} is invalid and was not exported.`);
    if (status && result.valid && status.value === 'unreviewed') status.value = 'correct';
  });
}
exportDecisions.addEventListener('click', () => {
  persistReviewState('Review labels saved before export.');
  downloadText('review_decisions.json', JSON.stringify(exportPayload(), null, 2), 'application/json');
});
exportTuning.addEventListener('click', () => {
  persistReviewState('Review labels saved before export.');
  downloadText('review_tuning.jsonl', tuningRows().map((row) => JSON.stringify(row)).join('\\n') + '\\n', 'application/jsonl');
});
importDecisions.addEventListener('click', () => importDecisionsFile.click());
importDecisionsFile.addEventListener('change', () => {
  const file = importDecisionsFile.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    try {
      const imported = JSON.parse(reader.result);
      if (!imported || typeof imported.decisions !== 'object') throw new Error('File has no decisions object.');
      reviewState = {schema_version: 1, provenance: REVIEW_PROVENANCE, decisions: imported.decisions};
      applyReviewStateToControls();
      persistReviewState(`Imported review labels from ${file.name}.`);
    } catch (error) {
      reviewSaveStatus.textContent = `Import failed: ${error.message}`;
    }
    importDecisionsFile.value = '';
  };
  reader.readAsText(file);
});
viewerClose.addEventListener('click', closeViewer);
viewerPrevImage.addEventListener('click', () => navigateWithinCluster(-1));
viewerNextImage.addEventListener('click', () => navigateWithinCluster(1));
viewerPrevCluster.addEventListener('click', () => navigateCluster(-1));
viewerNextCluster.addEventListener('click', () => navigateCluster(1));
viewerImage.addEventListener('error', () => {
  viewerImage.hidden = true;
  viewerError.hidden = false;
});
viewerSlider.addEventListener('input', () => {
  pendingQueueIndex = Number(viewerSlider.value);
  viewerSliderStatus.textContent = `Pending ${pendingQueueIndex + 1} / ${reviewQueue.length} — release to open`;
});
viewerSlider.addEventListener('change', () => setViewerItem(pendingQueueIndex));
imageViewer.addEventListener('click', (event) => {
  if (event.target === imageViewer || event.target.matches('[data-close-viewer]')) closeViewer();
});
viewerFullscreen.addEventListener('click', async () => {
  if (document.fullscreenElement) {
    await document.exitFullscreen();
  } else if (viewerDialog.requestFullscreen) {
    await viewerDialog.requestFullscreen();
  }
});
document.addEventListener('keydown', (event) => {
  if (imageViewer.hidden) return;
  if (event.key === 'Escape') { event.preventDefault(); closeViewer(); }
  else if (event.target !== viewerSlider && event.key === 'ArrowLeft') { event.preventDefault(); navigateWithinCluster(-1); }
  else if (event.target !== viewerSlider && event.key === 'ArrowRight') { event.preventDefault(); navigateWithinCluster(1); }
  else if (event.target !== viewerSlider && event.key === 'PageUp') { event.preventDefault(); navigateCluster(-1); }
  else if (event.target !== viewerSlider && event.key === 'PageDown') { event.preventDefault(); navigateCluster(1); }
});
applyReviewStateToControls();
refresh();
</script>
"""
    script = script.replace("__REVIEW_PROVENANCE__", json.dumps(review_provenance, separators=(",", ":")))
    body = (
        "<h1>Cluster image review</h1>"
        f"<p><strong>{len(clustering.images)} source images</strong> in <strong>{len(clustering.clusters)} clusters</strong>. "
        "Use the filters to narrow the queue. Click any thumbnail to open the full-screen viewer; "
        "use its image arrows, cluster arrows, and deferred-load skip slider to move through the filtered review set.</p>"
        + toolbar + "<main id='clusters'>" + "".join(cards) + "</main>" + viewer_markup + script
    )
    write_html(index_path, "Cluster image review", body)

    crop_cluster_counts = Counter(row["cluster_id"] for row in crop_rows)
    suspects = [
        row for row in cluster_rows
        if int(row["image_count"]) >= 4
        or int(row["largest_sequence_gap_bridged"]) > 1
        or row["manual_review_required"]
        or crop_cluster_counts[row["cluster_id"]] == 0
        or crop_cluster_counts[row["cluster_id"]] > int(row["image_count"])
    ]
    suspect_path = output_root / "review" / "suspect_clusters.html"
    suspect_body = (
        "<h1>Suspect clusters</h1>"
        "<p>Automatically ranked review populations. Every listed cluster includes all of its images.</p><ol>"
        + "".join(
            _review_card(
                row,
                crop_cluster_counts[row["cluster_id"]],
                images_by_cluster[row["cluster_id"]],
                crop_rows,
                output_root,
                suspect_path,
            )
            for row in suspects
        )
        + "</ol>"
    )
    write_html(suspect_path, "Suspect clusters", suspect_body)

    crop_cards = []
    crop_index_path = output_root / "review" / "crops.html"
    for row in crop_rows:
        crop = Path(row["crop_path"]) if row.get("crop_path") else None
        source = Path(row["source_image_path"])
        full_resolution = ensure_review_full_resolution(output_root, source)
        source_href = (
            relative_asset_link(crop_index_path, full_resolution)
            if full_resolution
            else file_link(source)
        )
        crop_src = relative_asset_link(crop_index_path, crop) if crop else None
        image = f"<img loading='lazy' src='{html.escape(crop_src)}' alt='Crop {html.escape(row['submission_id'])}'>" if crop_src and crop.is_file() else "<p>Missing crop</p>"
        crop_cards.append(
            f"<article class='card'><h2>{html.escape(row['disposition'])}: {html.escape(row['submission_id'])}</h2>"
            f"<p>Cluster <code>{html.escape(row['cluster_id'])}</code>; source {html.escape(row['source_filename'])}; "
            f"kind={html.escape(row['kind'])}; completeness={html.escape(row['completeness'])}; bbox={html.escape(json.dumps(row['bbox']))}</p>"
            f"<p><a href='{html.escape(source_href)}'>source</a> · "
            f"<a href='crops/by_cluster/{html.escape(row['cluster_id'])}.html'>cluster crop page</a></p>{image}</article>"
        )
    write_html(output_root / "review" / "crops.html", "All recognizer crops", "<h1>Recognizer crops</h1>" + "<section class='grid'>" + "".join(crop_cards) + "</section>")

    label_rows = _label_pair_rows(clustering, labels_path)
    label_table = []
    for row in label_rows:
        status = "PASS" if row["passed"] else "FAIL"
        label_table.append(
            f"<tr class='{status.lower()}'><td><strong>{status}</strong></td>"
            f"<td>{html.escape(row['category'])}</td><td>{html.escape(row['first_filename'])}</td>"
            f"<td>{html.escape(row['second_filename'])}</td><td>{row['expected_same_document']}</td>"
            f"<td>{row['actual_same_document']}</td><td>{html.escape(str(row['branch']))}</td>"
            f"<td>{html.escape(str(row['reason']))}</td><td><details><summary>metrics</summary>"
            f"<pre>{html.escape(json.dumps(row, indent=2))}</pre></details></td></tr>"
        )
    label_body = (
        "<h1>Reviewed content pairs</h1><p>Expected versus actual decisions from the saved clustering result.</p>"
        "<table><thead><tr><th>Status</th><th>Label</th><th>First</th><th>Second</th>"
        "<th>Expected same</th><th>Actual same</th><th>Branch</th><th>Reason</th><th>Diagnostics</th></tr></thead><tbody>"
        + "".join(label_table) + "</tbody></table>"
    )
    write_html(output_root / "review" / "labeled_pairs.html", "Reviewed content pairs", label_body)
    return label_rows


def validate(
    inventory_payload: dict[str, Any],
    clustering: ClusteringResult,
    crop_rows: list[dict[str, Any]],
    output_root: Path,
    label_rows: list[dict[str, Any]] | None = None,
    max_cluster_size: int = 3,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    source_paths = [Path(row["absolute_source_path"]).resolve() for row in inventory_payload["images"]]
    clustered_paths = [image.path.resolve() for image in clustering.images]
    counts = Counter(clustered_paths)
    if set(counts) != set(source_paths):
        missing = sorted(str(path) for path in set(source_paths) - set(counts))
        extra = sorted(str(path) for path in set(counts) - set(source_paths))
        errors.append(f"cluster source set mismatch; missing={missing[:5]}, extra={extra[:5]}")
    if any(count != 1 for count in counts.values()):
        errors.append("some source images occur zero or more than once in clustering images")
    image_to_folder = {Path(row["absolute_source_path"]).resolve(): row["immediate_parent_folder"] for row in inventory_payload["images"]}
    for cluster in clustering.clusters:
        if len(cluster.image_ids) > max_cluster_size:
            errors.append(
                f"cluster exceeds configured maximum of {max_cluster_size}: "
                f"{cluster.cluster_id} has {len(cluster.image_ids)} images"
            )
        folders = {image_to_folder.get(image.path.resolve()) for image in clustering.images_for(cluster.cluster_id)}
        if len(folders) != 1:
            errors.append(f"cluster crosses immediate parent folders: {cluster.cluster_id}")
        indices = [image.sequence_index for image in clustering.images_for(cluster.cluster_id)]
        if indices != sorted(indices):
            errors.append(f"cluster members are not filename ordered: {cluster.cluster_id}")
    submission_ids = [row["submission_id"] for row in crop_rows]
    if len(submission_ids) != len(set(submission_ids)):
        errors.append("submission IDs are not unique")
    cluster_ids = [cluster.cluster_id for cluster in clustering.clusters]
    if len(cluster_ids) != len(set(cluster_ids)):
        errors.append("cluster IDs are not unique")
    inventory_by_path = {Path(row["absolute_source_path"]).resolve(): row for row in inventory_payload["images"]}
    for row in crop_rows:
        source = Path(row["source_image_path"]).resolve()
        crop = Path(row["crop_path"]).resolve() if row["crop_path"] else None
        dimensions = inventory_by_path.get(source)
        if dimensions is None:
            errors.append(f"crop references unknown source: {source}")
        elif not (
            0 <= row["bbox"][0] < row["bbox"][2] <= dimensions["width"]
            and 0 <= row["bbox"][1] < row["bbox"][3] <= dimensions["height"]
        ):
            errors.append(f"crop bbox outside source: {row['submission_id']}")
        if crop is None or not crop.is_file() or crop.stat().st_size == 0:
            errors.append(f"crop file missing or empty: {row['submission_id']}")
        if not source.is_absolute() or (crop is not None and not crop.is_absolute()):
            errors.append(f"manifest path is not absolute: {row['submission_id']}")
    if inventory_payload["decoding_failures"]:
        errors.append(f"{len(inventory_payload['decoding_failures'])} source images failed decoding")
    if not crop_rows:
        warnings.append("no recognizer submissions were produced")
    if label_rows is not None:
        failed_labels = [row for row in label_rows if not row["passed"]]
        if failed_labels:
            errors.append(f"{len(failed_labels)} reviewed content pairs failed")
    validation = {
        "schema_version": 1,
        "validated_at": utc_now(),
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
        "source_image_count": len(source_paths),
        "cluster_image_count": len(clustered_paths),
        "cluster_count": len(clustering.clusters),
        "max_cluster_size": max_cluster_size,
        "grouping_mode": clustering.grouping_mode,
        "group_manifest": str(clustering.group_manifest) if clustering.group_manifest else None,
        "submission_count": len(crop_rows),
        "labeled_pair_count": len(label_rows or []),
        "labeled_pair_failures": sum(not row["passed"] for row in (label_rows or [])),
    }
    write_json(output_root / "reports" / "validation.json", validation)
    markdown = "# Validation\n\n"
    markdown += f"- Status: **{'PASS' if validation['passed'] else 'FAIL'}**\n"
    markdown += f"- Source images: {len(source_paths)}; clustered images: {len(clustered_paths)}\n"
    markdown += f"- Clusters: {len(clustering.clusters)}; crop submissions: {len(crop_rows)}\n\n"
    markdown += "## Errors\n" + ("\n".join(f"- {error}" for error in errors) or "- None\n")
    markdown += "\n## Warnings\n" + ("\n".join(f"- {warning}" for warning in warnings) or "- None\n")
    (output_root / "reports" / "validation.md").write_text(markdown, encoding="utf-8")
    return validation


def load_reusable_clustering(
    clustering_json: Path,
    run_json: Path,
    inventory_payload: dict[str, Any],
    cluster_config: ClusterConfig,
    triplet_manifest: Path | None = None,
) -> ClusteringResult | None:
    """Load a completed clustering stage only when its provenance is exact."""
    if not clustering_json.is_file() or not run_json.is_file():
        return None
    try:
        metadata = json.loads(run_json.read_text(encoding="utf-8"))
        if metadata.get("config") != cluster_config.__dict__:
            return None
        expected_grouping_mode = "triplet_manifest" if triplet_manifest is not None else "folder_sequence"
        if metadata.get("grouping_mode", "folder_sequence") != expected_grouping_mode:
            return None
        if triplet_manifest is not None and metadata.get("group_manifest") != str(triplet_manifest.resolve()):
            return None
        if metadata.get("image_count") != inventory_payload["total_image_count"]:
            return None
        clustering = load_result(clustering_json)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    expected_paths = {Path(row["absolute_source_path"]).resolve() for row in inventory_payload["images"]}
    actual_paths = {image.path.resolve() for image in clustering.images}
    if actual_paths != expected_paths or len(clustering.images) != len(expected_paths):
        return None
    return clustering


def run(args: argparse.Namespace) -> int:
    started = time.monotonic()
    start_at = utc_now()
    input_root = args.input_dir.resolve()
    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    triplet_manifest = args.triplet_manifest.resolve() if args.triplet_manifest else None
    inventory_payload = load_or_create_inventory(
        input_root,
        output_root,
        triplet_manifest=triplet_manifest,
    )
    config_path = args.cluster_config.resolve()
    crop_config_path = args.crop_config.resolve()
    cluster_config = ClusterConfig.from_json(config_path)
    crop_config = load_crop_config(crop_config_path)
    state = {
        "schema_version": 1,
        "input_root": str(input_root),
        "output_root": str(output_root),
        "source_fingerprint": digest_records(inventory_payload["images"]),
        "git_commit": args.git_commit,
        "cluster_config_path": str(config_path),
        "cluster_config_sha256": sha256_file(config_path),
        "cluster_config": cluster_config.__dict__,
        "triplet_manifest": str(triplet_manifest) if triplet_manifest else None,
        "triplet_manifest_sha256": sha256_file(triplet_manifest) if triplet_manifest else None,
        "grouping_mode": "triplet_manifest" if triplet_manifest else "folder_sequence",
        "crop_config_path": str(crop_config_path),
        "crop_config_sha256": sha256_file(crop_config_path),
        "crop_config": crop_config.data,
        "python": sys.version,
        "platform": platform.platform(),
        "started_at": start_at,
        "commands": args.commands,
    }
    state_path = output_root / "logs" / "evaluation_state.json"
    previous = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else None
    state_match = previous is not None and all(previous.get(key) == value for key, value in state.items() if key not in {"started_at", "commands"})
    memory = PeakMemory()
    memory.start()
    cluster_output = output_root / "clustering"
    clustering_json = cluster_output / "clustering.json"
    clustering = load_reusable_clustering(
        clustering_json,
        cluster_output / "run.json",
        inventory_payload,
        cluster_config,
        triplet_manifest=triplet_manifest,
    )
    if clustering is not None:
        LOGGER.info("Reusing completed clustering output with matching source/config provenance")
    else:
        clustering = cluster_directory(
            input_dir=input_root,
            config=cluster_config,
            cache_dir=cluster_output / ".feature_cache",
            triplet_manifest=triplet_manifest,
            show_progress=args.show_progress,
        )
        write_result(clustering, cluster_output, config=cluster_config)
    cluster_rows = cluster_artifacts(clustering, input_root, output_root)
    cropping_json = output_root / "cropping.json"
    if state_match and cropping_json.is_file():
        LOGGER.info("Reusing crop output with matching source/config/revision state")
        cropping = json.loads(cropping_json.read_text(encoding="utf-8"))
    else:
        cropping = crop_clustering_result(
            clustering,
            output_dir=output_root,
            config=crop_config,
            show_progress=args.show_progress,
        )
    crop_rows = crop_artifacts(cropping, input_root, output_root, inventory_payload)
    label_rows = write_review_package(
        output_root=output_root,
        clustering=clustering,
        cluster_rows=cluster_rows,
        crop_rows=crop_rows,
        labels_path=args.pair_labels.resolve(),
    )
    validation = validate(
        inventory_payload,
        clustering,
        crop_rows,
        output_root,
        label_rows=label_rows,
        max_cluster_size=cluster_config.max_cluster_size,
    )
    elapsed = time.monotonic() - started
    peak = memory.stop()
    state.update(
        {
            "completed_at": utc_now(),
            "runtime_seconds": round(elapsed, 3),
            "peak_rss_bytes": peak,
            "inventory_count": inventory_payload["total_image_count"],
            "cluster_count": len(clustering.clusters),
            "submission_count": len(crop_rows),
            "validation_passed": validation["passed"],
        }
    )
    write_json(state_path, state)
    write_final_report(state, inventory_payload, clustering, crop_rows, validation, output_root)
    return 0 if validation["passed"] else 2


def write_final_report(
    state: dict[str, Any],
    inventory_payload: dict[str, Any],
    clustering: ClusteringResult,
    crop_rows: list[dict[str, Any]],
    validation: dict[str, Any],
    output_root: Path,
) -> None:
    cluster_sizes = Counter(len(cluster.image_ids) for cluster in clustering.clusters)
    accepted = [comparison for comparison in clustering.comparisons if comparison.same_document]
    nonadjacent = [comparison for comparison in accepted if comparison.index_gap > 1]
    complete = [row for row in crop_rows if row["kind"] == "base_page" and row["completeness"] == "complete" and not row["review_required"]]
    overlays = [row for row in crop_rows if row["kind"] == "data_bearing_overlay" and not row["review_required"]]
    partial = [row for row in crop_rows if row["completeness"] == "partial_best_available" and not row["review_required"]]
    review = [row for row in crop_rows if row["review_required"]]
    zero_crop_clusters = [cluster.cluster_id for cluster in clustering.clusters if not any(row["cluster_id"] == cluster.cluster_id for row in crop_rows)]
    fractions = []
    for row in crop_rows:
        source = next((item for item in inventory_payload["images"] if item["absolute_source_path"] == row["source_image_path"]), None)
        if source and source["width"] and source["height"]:
            fractions.append((row["bbox"][2] - row["bbox"][0]) * (row["bbox"][3] - row["bbox"][1]) / (source["width"] * source["height"]))
    registration_failures = sum(not comparison.same_document for comparison in clustering.comparisons)
    total_comparisons = len(clustering.comparisons)
    output_bytes = sum(path.stat().st_size for path in output_root.rglob("*") if path.is_file())
    review_clusters = [cluster.cluster_id for cluster in clustering.clusters if len(cluster.image_ids) >= 4]
    report = f"""# Full automatic clustering and unique-crop evaluation

## Run provenance

- Input root: `{inventory_payload['input_root']}`
- Output root: `{output_root}`
- Images: **{inventory_payload['total_image_count']}**
- Grouping mode: **{clustering.grouping_mode}**
- Triplet manifest: `{clustering.group_manifest or 'none'}`
- Immediate parent folders: **{inventory_payload['immediate_parent_folder_count']}**
- Git commit: `{state['git_commit']}`
- Clustering config: `{state['cluster_config_path']}` (sha256 `{state['cluster_config_sha256']}`)
- Cropping config: `{state['crop_config_path']}` (sha256 `{state['crop_config_sha256']}`)
- Start: `{state['started_at']}`; completion: `{state['completed_at']}`
- Runtime: **{state['runtime_seconds']} seconds**
- Approximate peak RSS: **{state.get('peak_rss_bytes') or 'unavailable'} bytes**
- Output disk usage: **{output_bytes} bytes**

## Cluster-context strategy

- Clusters: **{len(clustering.clusters)}**
- Singleton / pair / triplet / larger: **{cluster_sizes[1]} / {cluster_sizes[2]} / {cluster_sizes[3]} / {sum(value for size, value in cluster_sizes.items() if size >= 4)}**
- Maximum cluster size: **{max(cluster_sizes, default=0)}** (configured maximum: **{state['cluster_config']['max_cluster_size']}**)
- Accepted pair registrations: **{len(accepted)}**
- Nonadjacent accepted relationships: **{len(nonadjacent)}**
- Registration failures/rejections: **{registration_failures}/{total_comparisons}**
- Clusters with four or more images inspected by the automatic review report: **{len(review_clusters)}**

The cluster-context deliverables are ready to submit to a stronger multimodal model as a machine-readable grouping proposal: `reports/clusters_for_model.jsonl`. Every source image is represented once, members stay in filename order, and the validation report checks that no cluster crosses an immediate parent folder. Human review is still recommended for singleton clusters, nonadjacent links, and transitive components flagged in `reports/cluster_summary.csv`.

## Unique-crop strategy

- Complete base-page submissions: **{len(complete)}**
- Data-bearing overlay submissions: **{len(overlays)}**
- Partial-best-available submissions: **{len(partial)}**
- Review-required submissions: **{len(review)}**
- Clusters producing zero crops: **{len(zero_crop_clusters)}**
- Crop area fraction min / mean / max: **{min(fractions, default=0):.4f} / {(sum(fractions) / len(fractions) if fractions else 0):.4f} / {max(fractions, default=0):.4f}**
- Crop records: **{len(crop_rows)}**

The exact crop-ready deliverable is `reports/crops_for_recognizer.jsonl`; it references generated nonempty crop files and original-coordinate bounding boxes. The crop output is sufficiently structured for recognizer testing, but reliability is not established for unattended 80,000-image production until the review-required, partial, zero-crop, and lowest-confidence populations are audited.

## Validation

- Status: **{'PASS' if validation['passed'] else 'FAIL'}**
- Errors: {len(validation['errors'])}
- Warnings: {len(validation['warnings'])}
- Details: `reports/validation.md`

## Systematic problems and remaining risks

- This run uses the checked-in pilot-calibrated thresholds without filename-specific rules or manual manifest edits.
- Registration rejection can leave singleton or split clusters; those cases are retained in cluster-context output rather than silently dropped.
- The package's cropper emits per-cluster manifests and annotations; this evaluation layer derives absolute-path JSONL, CSV, HTML review, and validation artifacts from those saved outputs.
- Review the lowest-confidence 50 clusters, every partial/review submission, every zero-crop cluster, every cluster larger than four images, all nonadjacent links, and a random sample of at least 50 accepted clusters before production.
- Recommendation for 80,000 images: **do not run unattended yet**; run a staged pilot and measure false merges, false splits, false-complete pages, missed overlays, duplicates, and decode/registration failure rates first.

## Commands

"""
    report += "\n".join(f"- `{command}`" for command in state["commands"])
    (output_root / "reports" / "final_report.md").write_text(report, encoding="utf-8")


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--input-dir", type=Path, required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    command.add_argument("--cluster-config", type=Path, required=True)
    command.add_argument("--triplet-manifest", type=Path)
    command.add_argument("--crop-config", type=Path, required=True)
    command.add_argument(
        "--pair-labels",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "examples" / "evaluation" / "content_pair_labels.json",
    )
    command.add_argument("--git-commit", required=True)
    command.add_argument("--show-progress", action="store_true")
    command.add_argument("--commands", nargs="+", required=True)
    return command


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return run(parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
