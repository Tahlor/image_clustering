"""Strict integrity validation and canonical output helpers."""

from __future__ import annotations

import csv
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from image_clustering.evaluation.reviewed_models import (
    CONTRACT,
    DatasetContract,
    ManifestRow,
    expand_pairs,
    group_rows,
    load_csv_manifest,
    load_jsonl_manifest,
    sha256,
    size_distribution,
)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def cluster_summary(group: Sequence[ManifestRow]) -> dict[str, Any]:
    return {
        "original_cluster_id": group[0].original_cluster_id,
        "review_decision": group[0].review_decision,
        "cluster_status": group[0].cluster_status,
        "cluster_size": len(group),
        "sequence_id": group[0].sequence_id,
        "assignment_ids": [row.assignment_id for row in group],
        "image_ids": [row.image_id for row in group],
        "sequence_indices": [row.sequence_index for row in group],
    }


def _package_image(package_root: Path, relative_text: str) -> Path | None:
    relative = Path(relative_text)
    candidates = [package_root / relative]
    if relative.parts and relative.parts[0] == "images":
        candidates.append(package_root / Path(*relative.parts[1:]))
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def validate_manifest(
    csv_path: Path,
    *,
    jsonl_path: Path | None = None,
    package_root: Path | None = None,
    contract: DatasetContract = CONTRACT,
) -> dict[str, Any]:
    rows = load_csv_manifest(csv_path)
    errors: list[str] = []
    warnings: list[str] = []
    if len(rows) != contract.rows:
        errors.append(f"expected {contract.rows} rows, found {len(rows)}")
    if [row.index for row in rows] != list(range(len(rows))):
        errors.append("index values must be contiguous and ordered from zero")

    grouped = group_rows(rows)
    if len(grouped) != contract.original_clusters:
        errors.append(
            f"expected {contract.original_clusters} original clusters, "
            f"found {len(grouped)}"
        )
    duplicates = [
        image_id
        for image_id, count in Counter(row.image_id for row in rows).items()
        if count > 1
    ]
    if duplicates:
        errors.append(f"images occur in multiple rows: {duplicates[:10]}")

    accepted: list[list[ManifestRow]] = []
    rejected: list[list[ManifestRow]] = []
    assignment_clusters: dict[str, set[str]] = {}
    for row in rows:
        assignment_clusters.setdefault(row.assignment_id, set()).add(
            row.original_cluster_id
        )
    reused_assignments = {
        assignment_id: cluster_ids
        for assignment_id, cluster_ids in assignment_clusters.items()
        if len(cluster_ids) > 1
    }
    if reused_assignments:
        errors.append(
            "assignment IDs span original clusters: "
            f"{dict(list(reused_assignments.items())[:10])}"
        )

    for cluster_id, group in grouped.items():
        decisions = {row.review_decision for row in group}
        if len(decisions) != 1:
            errors.append(f"{cluster_id} mixes review decisions")
            continue
        if group[0].review_decision == "accepted":
            accepted.append(group)
            if any(
                row.cluster_status != "approved"
                or row.assignment_type != "accepted_cluster"
                or not row.included_in_original_cluster
                for row in group
            ):
                errors.append(f"{cluster_id} violates accepted-row semantics")
            if len({row.assignment_id for row in group}) != 1:
                errors.append(f"{cluster_id} does not retain one shared assignment")
        elif group[0].review_decision == "rejected":
            rejected.append(group)
            if any(
                row.cluster_status != "dissolved"
                or row.assignment_type != "rejected_singleton"
                or row.included_in_original_cluster
                for row in group
            ):
                errors.append(f"{cluster_id} violates rejected-row semantics")
            if len({row.assignment_id for row in group}) != len(group):
                errors.append(f"{cluster_id} lacks unique singleton assignments")
        else:
            errors.append(f"{cluster_id} has unsupported decision")

    positive, negative = expand_pairs(rows)
    counts = {
        "rows": len(rows),
        "original_clusters": len(grouped),
        "accepted_clusters": len(accepted),
        "rejected_clusters": len(rejected),
        "accepted_images": sum(map(len, accepted)),
        "rejected_images": sum(map(len, rejected)),
        "positive_pairs": len(positive),
        "negative_pairs": len(negative),
    }
    expected_counts = {
        "rows": contract.rows,
        "original_clusters": contract.original_clusters,
        "accepted_clusters": contract.accepted_clusters,
        "rejected_clusters": contract.rejected_clusters,
        "accepted_images": contract.accepted_images,
        "rejected_images": contract.rejected_images,
        "positive_pairs": contract.positive_pairs,
        "negative_pairs": contract.negative_pairs,
    }
    for name, expected in expected_counts.items():
        if counts[name] != expected:
            errors.append(f"expected {expected} {name}, found {counts[name]}")

    accepted_sizes = size_distribution(accepted)
    rejected_sizes = size_distribution(rejected)
    if accepted_sizes != dict(contract.accepted_size_distribution):
        errors.append(f"accepted size distribution differs: {accepted_sizes}")
    if rejected_sizes != dict(contract.rejected_size_distribution):
        errors.append(f"rejected size distribution differs: {rejected_sizes}")

    if jsonl_path is not None:
        jsonl_rows = load_jsonl_manifest(jsonl_path)
        if [row.normalized() for row in rows] != [
            row.normalized() for row in jsonl_rows
        ]:
            errors.append("assignments.jsonl does not exactly match assignments.csv")

    missing_images: list[str] = []
    if package_root is None:
        warnings.append(
            "image existence was not checked because package_root was omitted"
        )
    else:
        for row in rows:
            if _package_image(package_root, row.package_relative_path) is None:
                missing_images.append(row.package_relative_path)
        if missing_images:
            errors.append(f"manifest images are missing: {missing_images[:10]}")

    return {
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "warnings": warnings,
        "csv_sha256": sha256(csv_path),
        "jsonl_sha256": sha256(jsonl_path) if jsonl_path else None,
        "counts": counts,
        "accepted_size_distribution": accepted_sizes,
        "rejected_size_distribution": rejected_sizes,
        "missing_image_count": len(missing_images),
    }
